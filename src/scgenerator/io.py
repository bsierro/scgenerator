import os
import shutil
from datetime import datetime
from glob import glob
from typing import Any, Dict, Iterable, List, Tuple, Union

import numpy as np
import pkg_resources as pkg
import toml
from send2trash import TrashPermissionError, send2trash
from tqdm import tqdm
from pathlib import Path
import itertools

from . import utils
from .const import ENVIRON_KEY_BASE, PARAM_SEPARATOR, PREFIX_KEY_BASE, TMP_FOLDER_KEY_BASE
from .errors import IncompleteDataFolderError
from .logger import get_logger

using_ray = False
try:
    import ray
    from ray.util.queue import Queue

    using_ray = True
except ModuleNotFoundError:
    pass


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

    @staticmethod
    def tmp(task_id=0):
        suffix = "" if task_id == 0 else str(task_id)
        return ".scgenerator_tmp" + suffix

    @classmethod
    def plot(cls, name):
        """returns the paths to the specified plot. Used to save new plot
        example
        ---------
        fig.savefig(Paths.plot("figure5.pdf"))
        """
        return os.path.join(cls.get("plots"), name)


class DataBuffer:
    def __init__(self, task_id):
        self.logger = get_logger(__name__)
        self.id = task_id
        self.queue = Queue()

    def empty(self):
        num = self.queue.size()
        if num == 0:
            return 0
        self.logger.info(f"buffer length at time of emptying : {num}")
        while not self.queue.empty():
            name, identifier, data = self.queue.get()
            save_data(data, name, self.id, identifier)

        return num

    def append(self, file_name: str, identifier: str, data: np.ndarray):
        self.queue.put((file_name, identifier, data))


# def abspath(rel_path: str):
#     """returns the complete path with the correct root. In other words, allows to modify absolute paths
#     in case the process accessing this function is a sub-process started from another device.

#     Parameters
#     ----------
#     rel_path : str
#         relative path

#     Returns
#     -------
#     str
#         absolute path
#     """
#     key = utils.formatted_hostname()
#     prefix = os.getenv(key)
#     if prefix is None:
#         p = os.path.abspath(rel_path)
#     else:
#         p = os.path.join(prefix, rel_path)

#     return os.path.normpath(p)


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


def _prepare_for_serialization(dico: Dict[str, Any]):
    """prepares a dictionary for serialization. Some keys may not be preserved
    (dropped due to no conversion available)

    Parameters
    ----------
    dico : dict
        dictionary
    """
    forbiden_keys = ["w_c", "w_power_fact", "field_0", "spec_0", "w"]
    types = (np.ndarray, float, int, str, list, tuple, dict)
    out = {}
    for key, value in dico.items():
        if key in forbiden_keys:
            continue
        if not isinstance(value, types):
            continue
        if isinstance(value, dict):
            out[key] = _prepare_for_serialization(value)
        elif isinstance(value, np.ndarray) and value.dtype == complex:
            continue
        else:
            out[key] = value

    return out


def save_parameters(param_dict: Dict[str, Any], task_id: int, data_dir_name: str):
    param = param_dict.copy()
    file_path = generate_file_path("params.toml", task_id, data_dir_name)

    param = _prepare_for_serialization(param)
    param["datetime"] = datetime.now()

    file_path.parent.mkdir(exist_ok=True)

    # save toml of the simulation
    with open(file_path, "w") as file:
        toml.dump(param, file, encoder=toml.TomlNumpyEncoder())

    return file_path


# def save_parameters_old(param_dict, file_name="param"):
#     """Writes the flattened parameters dictionary specific to a single simulation into a toml file

#     Parameters
#     ----------
#         param_dict : dictionary of parameters. Only floats, int and arrays of
#                      non complex values are stored in the json
#         folder_name : folder where to save the files (relative to cwd)
#         file_name : name of the readable file.
#     """
#     param = param_dict.copy()

#     folder_name, file_name = os.path.split(file_name)
#     folder_name = "tmp" if folder_name == "" else folder_name
#     file_name = os.path.splitext(file_name)[0]

#     if not os.path.exists(folder_name):
#         os.makedirs(folder_name)

#     param = _prepare_for_serialization(param)
#     param["datetime"] = datetime.now()

#     # save toml of the simulation
#     with open(os.path.join(folder_name, file_name + ".toml"), "w") as file:
#         toml.dump(param, file, encoder=toml.TomlNumpyEncoder())

#     return os.path.join(folder_name, file_name)


def load_previous_parameters(path: os.PathLike):
    """loads a parameters toml files and converts data to appropriate type
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


def get_all_environ() -> Dict[str, str]:
    """returns a dictionary of all environment variables set by any instance of scgenerator"""
    d = dict(filter(lambda el: el[0].startswith(ENVIRON_KEY_BASE), os.environ.items()))
    return d


def load_single_spectrum(folder: Path, index) -> np.ndarray:
    return np.load(folder / f"spectra_{index}.npy")


def get_data_subfolders(task_id: int) -> List[Path]:
    """returns a list of relative path/subfolders in the specified directory

    Parameters
    ----------
    path : str
        path to directory containing the initial config file and the spectra sub folders

    Returns
    -------
    List[str]
        paths to sub folders
    """

    return [p.resolve() for p in get_data_folder(task_id).glob("*") if p.is_dir()]


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
    for sub_folder in sub_folders:
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
    params = load_toml(sub_folder / "params.toml")
    z_num = params["z_num"]
    num_spectra = find_last_spectrum_num(sub_folder) + 1  # because of zero-indexing

    if z_num != init_z_num:
        raise IncompleteDataFolderError(
            f"initial config specifies {init_z_num} spectra per"
            + f" but the parameter file in {sub_folder} specifies {z_num}"
        )

    return z_num - num_spectra


def find_last_spectrum_num(data_dir: Path):
    for num in itertools.count(1):
        p_to_test = data_dir / f"spectrum_{num}.npy"
        if not p_to_test.is_file() or len(p_to_test.read_bytes()) == 0:
            return num - 1


def load_last_spectrum(data_dir: Path) -> Tuple[int, np.ndarray]:
    """return the last spectrum stored in path as well as its id"""
    num = find_last_spectrum_num(data_dir)
    return num, np.load(data_dir / f"spectrum_{num}.npy")


def append_and_merge(final_sim_path: os.PathLike, new_name=None):
    final_sim_path = Path(final_sim_path).resolve()
    if new_name is None:
        new_name = final_sim_path.name + " appended"

    destination_path = final_sim_path.parent / new_name
    destination_path.mkdir(exist_ok=True)

    sim_paths = list(final_sim_path.glob("id*num*"))
    pbars = utils.PBars.auto(
        len(sim_paths),
        0,
        head_kwargs=dict(desc="Appending"),
        worker_kwargs=dict(desc=""),
    )

    for sim_path in sim_paths:
        path_tree = [sim_path]
        sim_name = sim_path.name
        appended_sim_path = destination_path / sim_name
        appended_sim_path.mkdir(exist_ok=True)

        while (
            prev_sim_path := load_toml(path_tree[-1] / "params.toml").get("prev_data_dir")
        ) is not None:
            path_tree.append(Path(prev_sim_path).resolve())

        z: List[np.ndarray] = []
        z_num = 0
        last_z = 0
        paths_r = list(reversed(path_tree))

        for path in paths_r:
            curr_z_num = load_toml(path / "params.toml")["z_num"]
            for i in range(curr_z_num):
                shutil.copy(
                    path / f"spectrum_{i}.npy",
                    appended_sim_path / f"spectrum_{i + z_num}.npy",
                )
            z_arr = np.load(path / "z.npy")
            z.append(z_arr + last_z)
            last_z += z_arr[-1]
            z_num += curr_z_num
        z_arr = np.concatenate(z)
        update_appended_params(sim_path / "params.toml", appended_sim_path / "params.toml", z_arr)
        np.save(appended_sim_path / "z.npy", z_arr)
        pbars.update(0)

    update_appended_params(
        final_sim_path / "initial_config.toml", destination_path / "initial_config.toml", z_arr
    )

    merge(destination_path, delete=True)


def update_appended_params(param_path: Path, new_path: Path, z):
    z_num = len(z)
    params = load_toml(param_path)
    if "simulation" in params:
        params["simulation"]["z_num"] = z_num
        params["simulation"]["z_targets"] = z
    else:
        params["z_num"] = z_num
        params["z_targets"] = z
    save_toml(new_path, params)


def merge(paths: Union[Path, List[Path]], delete=False):
    if isinstance(paths, Path):
        paths = [paths]
    for path in paths:
        merge_same_simulations(path, delete=delete)


def merge_same_simulations(path: Path, delete=True):
    logger = get_logger(__name__)
    num_separator = PARAM_SEPARATOR + "num" + PARAM_SEPARATOR
    sub_folders = [p for p in path.glob("*") if p.is_dir()]
    config = load_toml(path / "initial_config.toml")
    repeat = config["simulation"].get("repeat", 1)
    max_repeat_id = repeat - 1
    z_num = config["simulation"]["z_num"]

    check_data_integrity(sub_folders, z_num)

    sim_num, param_num = utils.count_variations(config)
    pbar = utils.PBars.auto(sim_num * z_num, head_kwargs=dict(desc="Merging data"))

    spectra = []
    for z_id in range(z_num):
        for variable_and_ind, _ in utils.required_simulations(config):
            repeat_id = variable_and_ind[-1][1]

            # reset the buffer once we move to a new parameter set
            if repeat_id == 0:
                spectra = []

            in_path = path / utils.format_variable_list(variable_and_ind)
            spectra.append(np.load(in_path / f"spectrum_{z_id}.npy"))
            pbar.update()

            # write new files only once all those from one parameter set are collected
            if repeat_id == max_repeat_id:
                out_path = path / (
                    utils.format_variable_list(variable_and_ind[1:-1]) + PARAM_SEPARATOR + "merged"
                )

                out_path = ensure_folder(out_path, prevent_overwrite=False)
                spectra = np.array(spectra).reshape(repeat, len(spectra[0]))
                np.save(out_path / f"spectra_{z_id}.npy", spectra.squeeze())

                # copy other files only once
                if z_id == 0:
                    for file_name in ["z.npy", "params.toml"]:
                        shutil.copy(in_path / file_name, out_path)
    pbar.close()

    if delete:
        for sub_folder in sub_folders:
            try:
                send2trash(str(sub_folder))
            except TrashPermissionError:
                logger.warning(f"could not send send {sub_folder} to trash")


def get_data_folder(task_id: int, name_if_new: str = "data") -> Path:
    if name_if_new == "":
        name_if_new = "data"
    idstr = str(int(task_id))
    tmp = os.getenv(TMP_FOLDER_KEY_BASE + idstr)
    if tmp is None:
        tmp = ensure_folder(Path("scgenerator" + PARAM_SEPARATOR + name_if_new))
        os.environ[TMP_FOLDER_KEY_BASE + idstr] = str(tmp)
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


def generate_file_path(file_name: str, task_id: int, identifier: str = "") -> Path:
    """generates a path for the desired file name

    Parameters
    ----------
    file_name : str
        desired file name. May be altered if it already exists
    task_id : int
        unique id of the process
    identifier : str
        subfolder in which to store the file. default : ""

    Returns
    -------
    str
        the full path
    """
    path = get_data_folder(task_id) / identifier / file_name
    path.parent.mkdir(exist_ok=True)

    return path


def save_data(data: np.ndarray, file_name: str, task_id: int, identifier: str = ""):
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
    path = generate_file_path(file_name, task_id, identifier)
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