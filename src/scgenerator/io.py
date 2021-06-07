import itertools
import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Generator, List, Sequence, Tuple

import numpy as np
import pkg_resources as pkg
import toml

from . import env, utils
from .const import (
    __version__,
    ENVIRON_KEY_BASE,
    PARAM_FN,
    PARAM_SEPARATOR,
    PBAR_POLICY,
    SPEC1_FN,
    SPECN_FN,
    TMP_FOLDER_KEY_BASE,
    Z_FN,
)
from .errors import IncompleteDataFolderError
from .logger import get_logger

PathTree = List[Tuple[Path, ...]]


class Paths:
    _data_files = [
        "silica.toml",
        "gas.toml",
        "hr_t.npz",
        "submit_job_template.txt",
        "start_worker.sh",
        "start_head.sh",
    ]

    paths = {
        f.split(".")[0]: os.path.abspath(
            pkg.resource_filename("scgenerator", os.path.join("data", f))
        )
        for f in _data_files
    }

    @classmethod
    def get(cls, key):
        if key not in cls.paths:
            if os.path.exists("paths.toml"):
                with open("paths.toml") as file:
                    paths_dico = toml.load(file)
                for k, v in paths_dico.items():
                    cls.paths[k] = v
        if key not in cls.paths:
            get_logger(__name__).info(
                f"{key} was not found in path index, returning current working directory."
            )
            cls.paths[key] = os.getcwd()

        return cls.paths[key]

    @classmethod
    def gets(cls, key):
        """returned the specified file as a string"""
        with open(cls.get(key)) as file:
            return file.read()

    @classmethod
    def plot(cls, name):
        """returns the paths to the specified plot. Used to save new plot
        example
        ---------
        fig.savefig(Paths.plot("figure5.pdf"))
        """
        return os.path.join(cls.get("plots"), name)


def conform_toml_path(path: os.PathLike) -> Path:
    path = Path(path)
    if not path.name.lower().endswith(".toml"):
        path = path.parent / (path.name + ".toml")
    return path


def load_toml(path: os.PathLike):
    """returns a dictionary parsed from the specified toml file"""
    path = conform_toml_path(path)
    with open(path, mode="r") as file:
        dico = toml.load(file)
    return dico


def save_toml(path: os.PathLike, dico):
    """saves a dictionary into a toml file"""
    path = conform_toml_path(path)
    with open(path, mode="w") as file:
        toml.dump(dico, file)
    return dico


def serializable(val):
    """returns True if val is serializable into a Json file"""
    types = (np.ndarray, float, int, str, list, tuple)

    out = isinstance(val, types)
    if isinstance(val, np.ndarray):
        out &= val.dtype != "complex"
    return out


def prepare_for_serialization(dico: Dict[str, Any]) -> Dict[str, Any]:
    """prepares a dictionary for serialization. Some keys may not be preserved
    (dropped due to no conversion available)

    Parameters
    ----------
    dico : dict
        dictionary
    """
    forbiden_keys = ["w_c", "w_power_fact", "field_0", "spec_0", "w", "t", "z_targets"]
    types = (np.ndarray, float, int, str, list, tuple, dict)
    out = {}
    for key, value in dico.items():
        if key in forbiden_keys:
            continue
        if not isinstance(value, types):
            continue
        if isinstance(value, dict):
            out[key] = prepare_for_serialization(value)
        elif isinstance(value, np.ndarray) and value.dtype == complex:
            continue
        else:
            out[key] = value

    return out


def save_parameters(param_dict: Dict[str, Any], destination_dir: Path) -> Path:
    """saves a parameter dictionary. Note that is does remove some entries, particularly
    those that take a lot of space ("t", "w", ...)

    Parameters
    ----------
    param_dict : Dict[str, Any]
        dictionary to save
    data_dir : Path
        destination directory

    Returns
    -------
    Path
        path to newly created the paramter file
    """
    param = param_dict.copy()
    file_path = destination_dir / "params.toml"

    param = prepare_for_serialization(param)
    param["datetime"] = datetime.now()
    param["version"] = __version__

    file_path.parent.mkdir(exist_ok=True)

    # save toml of the simulation
    with open(file_path, "w") as file:
        toml.dump(param, file, encoder=toml.TomlNumpyEncoder())

    return file_path


def load_previous_parameters(path: os.PathLike):
    """loads a parameters toml files and converts data to appropriate type
    It is advised to run initialize.build_sim_grid to recover some parameters that are not saved.

    Parameters
    ----------
    path : PathLike
        path to the toml

    Returns
    ----------
    dict
        flattened parameters dictionary
    """
    params = load_toml(path)

    for k, v in params.items():
        if isinstance(v, list) and isinstance(v[0], (float, int)):
            params[k] = np.array(v)
    return params


def load_material_dico(name):
    """loads a material dictionary
    Parameters
    ----------
        name : str
            name of the material
    Returns
    ----------
        material_dico : dict
    """
    if name == "silica":
        return toml.loads(Paths.gets("silica"))
    else:
        return toml.loads(Paths.gets("gas"))[name]


def get_data_dirs(sim_dir: Path) -> List[Path]:
    """returns a list of absolute paths corresponding to a particular run

    Parameters
    ----------
    sim_dir : Path
        path to directory containing the initial config file and the spectra sub folders

    Returns
    -------
    List[Path]
        paths to sub folders
    """

    return [p.resolve() for p in sim_dir.glob("*") if p.is_dir()]


def check_data_integrity(sub_folders: List[Path], init_z_num: int):
    """checks the integrity and completeness of a simulation data folder

    Parameters
    ----------
    path : str
        path to the data folder
    init_z_num : int
        z_num as specified by the initial configuration file

    Raises
    ------
    IncompleteDataFolderError
        raised if not all spectra are present in any folder
    """

    for sub_folder in utils.PBars(sub_folders, "Checking integrity"):
        if num_left_to_propagate(sub_folder, init_z_num) != 0:
            raise IncompleteDataFolderError(
                f"not enough spectra of the specified {init_z_num} found in {sub_folder}"
            )


def num_left_to_propagate(sub_folder: Path, init_z_num: int) -> int:
    """checks if a propagation has completed

    Parameters
    ----------
    sub_folder : Path
        path to the sub folder containing the spectra
    init_z_num : int
        number of z position to store as specified in the master config file

    Returns
    -------
    bool
        True if the propagation has completed

    Raises
    ------
    IncompleteDataFolderError
        raised if init_z_num doesn't match that specified in the individual parameter file
    """
    z_num = load_toml(sub_folder / "params.toml")["z_num"]
    num_spectra = find_last_spectrum_num(sub_folder) + 1  # because of zero-indexing

    if z_num != init_z_num:
        raise IncompleteDataFolderError(
            f"initial config specifies {init_z_num} spectra per"
            + f" but the parameter file in {sub_folder} specifies {z_num}"
        )

    return z_num - num_spectra


def find_last_spectrum_num(data_dir: Path):
    for num in itertools.count(1):
        p_to_test = data_dir / SPEC1_FN.format(num)
        if not p_to_test.is_file() or len(p_to_test.read_bytes()) == 0:
            return num - 1


def load_last_spectrum(data_dir: Path) -> Tuple[int, np.ndarray]:
    """return the last spectrum stored in path as well as its id"""
    num = find_last_spectrum_num(data_dir)
    return num, np.load(data_dir / SPEC1_FN.format(num))


def update_appended_params(source: Path, destination: Path, z: Sequence):
    z_num = len(z)
    params = load_toml(source)
    if "simulation" in params:
        params["simulation"]["z_num"] = z_num
        params["fiber"]["length"] = float(z[-1] - z[0])
    else:
        params["z_num"] = z_num
        params["length"] = float(z[-1] - z[0])
    save_toml(destination, params)


def build_path_trees(sim_dir: Path) -> List[PathTree]:
    sim_dir = sim_dir.resolve()
    path_branches: List[Tuple[Path, ...]] = []
    to_check = list(sim_dir.glob("id*num*"))
    with utils.PBars(len(to_check), desc="Building path trees") as pbar:
        for branch in map(build_path_branch, to_check):
            if branch is not None:
                path_branches.append(branch)
                pbar.update()
    path_trees = group_path_branches(path_branches)
    return path_trees


def build_path_branch(data_dir: Path) -> Tuple[Path, ...]:
    if not data_dir.is_dir():
        return None
    path_branch = [data_dir]
    while (prev_sim_path := load_toml(path_branch[-1] / PARAM_FN).get("prev_data_dir")) is not None:
        p = Path(prev_sim_path).resolve()
        if not p.exists():
            p = Path(*p.parts[-2:]).resolve()
        path_branch.append(p)
    return tuple(reversed(path_branch))


def group_path_branches(path_branches: List[Tuple[Path, ...]]) -> List[PathTree]:
    """groups path lists

    [
        ("a/id 0 wavelength 100 num 0"," b/id 0 wavelength 100 num 0"),
        ("a/id 2 wavelength 100 num 1"," b/id 2 wavelength 100 num 1"),
        ("a/id 1 wavelength 200 num 0"," b/id 1 wavelength 200 num 0"),
        ("a/id 3 wavelength 200 num 1"," b/id 3 wavelength 200 num 1")
    ]
    ->
    [
        (
            ("a/id 0 wavelength 100 num 0", "a/id 2 wavelength 100 num 1"),
            ("b/id 0 wavelength 100 num 0", "b/id 2 wavelength 100 num 1"),
        )
        (
            ("a/id 1 wavelength 200 num 0", "a/id 3 wavelength 200 num 1"),
            ("b/id 1 wavelength 200 num 0", "b/id 3 wavelength 200 num 1"),
        )
    ]


    Parameters
    ----------
    path_branches : List[Tuple[Path, ...]]
        each element of the list is a path to a folder containing data of one simulation

    Returns
    -------
    List[PathTree]
        List of PathTrees to be used in merge
    """
    sort_key = lambda el: el[0]

    size = len(path_branches[0])
    out_trees_map: Dict[str, Dict[int, Dict[int, Path]]] = {}
    for branch in path_branches:
        b_id = utils.branch_id(branch)
        out_trees_map.setdefault(b_id, {i: {} for i in range(size)})
        for sim_part, data_dir in enumerate(branch):
            *_, num = data_dir.name.split()
            out_trees_map[b_id][sim_part][int(num)] = data_dir

    return [
        tuple(
            tuple(w for _, w in sorted(v.items(), key=sort_key))
            for __, v in sorted(d.items(), key=sort_key)
        )
        for d in out_trees_map.values()
    ]


def merge_path_tree(path_tree: PathTree, destination: Path):
    """given a path tree, copies the file into the right location

    Parameters
    ----------
    path_tree : PathTree
        elements of the list returned by group_path_branches
    destination : Path
        dir where to save the data
    """
    z_arr: List[float] = []

    destination.mkdir(exist_ok=True)

    for i, (z, merged_spectra) in enumerate(merge_spectra(path_tree)):
        z_arr.append(z)
        spec_out_name = SPECN_FN.format(i)
        np.save(destination / spec_out_name, merged_spectra)
    d = np.diff(z_arr)
    d[d < 0] = 0
    z_arr = np.concatenate(([z_arr[0]], np.cumsum(d)))
    np.save(destination / Z_FN, z_arr)
    update_appended_params(path_tree[-1][0] / PARAM_FN, destination / PARAM_FN, z_arr)


def merge_spectra(
    path_tree: PathTree,
) -> Generator[Tuple[float, np.ndarray], None, None]:
    for same_sim_paths in path_tree:
        z_arr = np.load(same_sim_paths[0] / Z_FN)
        for i, z in enumerate(z_arr):
            spectra: List[np.ndarray] = []
            for data_dir in same_sim_paths:
                spec = np.load(data_dir / SPEC1_FN.format(i))
                spectra.append(spec)
            yield z, np.atleast_2d(spectra)


def merge(destination: os.PathLike, path_trees: List[PathTree] = None):

    destination = ensure_folder(Path(destination))

    for i, sim_dir in enumerate(sim_dirs(path_trees)):
        shutil.copy(
            sim_dir / "initial_config.toml",
            destination / f"initial_config_{i}.toml",
        )

    for path_tree in utils.PBars(path_trees, desc="Merging"):
        iden = PARAM_SEPARATOR.join(path_tree[-1][0].name.split()[2:-2])
        merge_path_tree(path_tree, destination / iden)


def sim_dirs(path_trees: List[PathTree]) -> Generator[Path, None, None]:
    for p in path_trees[0]:
        yield p[0].parent


def get_sim_dir(task_id: int, name_if_new: str = "data") -> Path:
    if name_if_new == "":
        name_if_new = "data"
    tmp = env.data_folder(task_id)
    if tmp is None:
        tmp = ensure_folder(Path("scgenerator" + PARAM_SEPARATOR + name_if_new))
        os.environ[TMP_FOLDER_KEY_BASE + str(task_id)] = str(tmp)
    tmp = Path(tmp).resolve()
    if not tmp.exists():
        tmp.mkdir()
    return tmp


def set_data_folder(task_id: int, path: os.PathLike):
    """stores the path to an existing data folder in the environment

    Parameters
    ----------
    task_id : int
        id uniquely identifying the session
    path : str
        path to the root of the data folder
    """
    idstr = str(int(task_id))
    os.environ[TMP_FOLDER_KEY_BASE + idstr] = str(path)


def save_data(data: np.ndarray, data_dir: Path, file_name: str):
    """saves numpy array to disk

    Parameters
    ----------
    data : np.ndarray
        data to save
    file_name : str
        file name
    task_id : int
        id that uniquely identifies the process
    identifier : str, optional
        identifier in the main data folder of the task, by default ""
    """
    path = data_dir / file_name
    np.save(path, data)
    get_logger(__name__).debug(f"saved data in {path}")
    return


def ensure_folder(path: Path, prevent_overwrite: bool = True) -> Path:
    """ensure a folder exists and doesn't overwrite anything if required

    Parameters
    ----------
    path : Path
        desired path
    prevent_overwrite : bool, optional
        whether to create a new directory when one already exists, by default True

    Returns
    -------
    Path
        final path
    """

    path = path.resolve()

    # is path root ?
    if len(path.parts) < 2:
        return path

    # is a part of path an existing *file* ?
    parts = path.parts
    path = Path(path.root)
    for part in parts:
        if path.is_file():
            path = ensure_folder(path, prevent_overwrite=False)
        path /= part

    folder_name = path.name

    for i in itertools.count():
        if not path.is_file() and (not prevent_overwrite or not path.is_dir()):
            path.mkdir(exist_ok=True)
            return path
        path = path.parent / (folder_name + f"_{i}")
