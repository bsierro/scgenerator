import os
import shutil
from datetime import datetime
from glob import glob
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import pkg_resources as pkg
from ray import util
import toml
from send2trash import TrashPermissionError, send2trash

from . import utils
from .const import PARAM_SEPARATOR, PREFIX_KEY_BASE, TMP_FOLDER_KEY_BASE, ENVIRON_KEY_BASE
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
    home = os.path.expanduser("~")
    _data_files = ["silica.toml", "gas.toml", "hr_t.npz"]

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


def load_toml(path: str):
    """returns a dictionary parsed from the specified toml file"""
    if not path.lower().endswith(".toml"):
        path += ".toml"
    with open(path, mode="r") as file:
        dico = toml.load(file)
    return dico


def save_toml(path, dico):
    """saves a dictionary into a toml file"""
    if not path.lower().endswith(".toml"):
        path += ".toml"
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


def _prepare_for_serialization(dico):
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


def save_parameters(param_dict, file_name="param"):
    """Writes the flattened parameters dictionary specific to a single simulation into a toml file

    Parameters
    ----------
        param_dict : dictionary of parameters. Only floats, int and arrays of
                     non complex values are stored in the json
        folder_name : folder where to save the files (relative to cwd)
        file_name : name of the readable file.
    """
    param = param_dict.copy()

    folder_name, file_name = os.path.split(file_name)
    folder_name = "tmp" if folder_name == "" else folder_name
    file_name = os.path.splitext(file_name)[0]

    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    param = _prepare_for_serialization(param)
    param["datetime"] = datetime.now()

    # save toml of the simulation
    with open(os.path.join(folder_name, file_name + ".toml"), "w") as file:
        toml.dump(param, file, encoder=toml.TomlNumpyEncoder())

    return os.path.join(folder_name, file_name)


def load_previous_parameters(path: str):
    """loads a parameters toml files and converts data to appropriate type
    Parameters
    ----------
    path : str
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


# def set_environ(config: dict):
#     """sets environment variables specified in the config

#     Parameters
#     ----------
#     config : dict
#         whole simulation config file
#     """
#     environ = config.get("environment", {})
#     for k, v in environ.get("path_prefixes", {}).items():
#         os.environ[(PREFIX_KEY_BASE + k).upper()] = v


def get_all_environ() -> Dict[str, str]:
    """returns a dictionary of all environment variables set by any instance of scgenerator"""
    d = dict(filter(lambda el: el[0].startswith(ENVIRON_KEY_BASE), os.environ.items()))
    print(d)
    return d


def load_single_spectrum(folder, index) -> np.ndarray:
    return np.load(os.path.join(folder, f"spectra_{index}.npy"))


def get_data_subfolders(path: str) -> List[str]:
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
    sub_folders = glob(os.path.join(path, "*"))
    sub_folders = list(filter(os.path.isdir, sub_folders))
    return sub_folders


def check_data_integrity(sub_folders: List[str], init_z_num: int):
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
        if not propagation_completed(sub_folder, init_z_num):
            raise IncompleteDataFolderError(
                f"not enough spectra of the specified {init_z_num} found in {sub_folder}"
            )


def propagation_initiated(sub_folder) -> bool:
    if os.path.isdir(sub_folder):
        return find_last_spectrum_file(sub_folder) > 0
    return False


def propagation_completed(sub_folder: str, init_z_num: int):
    """checks if a propagation has completed

    Parameters
    ----------
    sub_folder : str
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
    params = load_toml(os.path.join(sub_folder, "params.toml"))
    z_num = params["z_num"]
    num_spectra = find_last_spectrum_file(sub_folder) + 1  # because of zero-indexing

    if z_num != init_z_num:
        raise IncompleteDataFolderError(
            f"initial config specifies {init_z_num} spectra per"
            + f" but the parameter file in {sub_folder} specifies {z_num}"
        )

    return num_spectra == z_num


def find_last_spectrum_file(path: str):
    num = 0
    while True:
        if os.path.isfile(os.path.join(path, f"spectrum_{num}.npy")):
            num += 1
            pass
        else:
            return num - 1


def load_last_spectrum(path: str):
    num = find_last_spectrum_file(path)
    return num, np.load(os.path.join(path, f"spectrum_{num}.npy"))


def merge_same_simulations(path: str):
    logger = get_logger(__name__)
    num_separator = PARAM_SEPARATOR + "num" + PARAM_SEPARATOR
    sub_folders = get_data_subfolders(path)
    config = load_toml(os.path.join(path, "initial_config.toml"))
    repeat = config["simulation"].get("repeat", 1)
    max_repeat_id = repeat - 1
    z_num = config["simulation"]["z_num"]

    check_data_integrity(sub_folders, z_num)

    base_folders = set()
    for sub_folder in sub_folders:
        splitted_base_path = sub_folder.split(num_separator)[:-1]
        base_folder = num_separator.join(splitted_base_path)
        if len(base_folder) > 0:
            base_folders.add(base_folder)

    num_operations = z_num * len(base_folders) + len(base_folders)
    pt = utils.ProgressTracker(num_operations, logger=logger, prefix="merging data : ")

    spectra = []
    for z_id in range(z_num):
        for variable_and_ind, _ in utils.required_simulations(config):
            repeat_id = variable_and_ind[-1][1]

            # reset the buffer once we move to a new parameter set
            if repeat_id == 0:
                spectra = []

            in_path = os.path.join(path, utils.format_variable_list(variable_and_ind))
            spectra.append(np.load(os.path.join(in_path, f"spectrum_{z_id}.npy")))

            # write new files only once all those from one parameter set are collected
            if repeat_id == max_repeat_id:
                out_path = os.path.join(path, utils.format_variable_list(variable_and_ind[:-1]))
                out_path = ensure_folder(out_path, prevent_overwrite=False)
                spectra = np.array(spectra).reshape(repeat, len(spectra[0]))
                np.save(os.path.join(out_path, f"spectra_{z_id}.npy"), spectra.squeeze())
                pt.update()

                # copy other files only once
                if z_id == 0:
                    for file_name in ["z.npy", "params.toml"]:
                        shutil.copy(
                            os.path.join(in_path, file_name),
                            os.path.join(out_path, ""),
                        )
                    pt.update()

    try:
        for sub_folder in sub_folders:
            send2trash(sub_folder)
    except TrashPermissionError:
        logger.warning(f"could not send send {len(base_folders)} folder(s) to trash")


def get_data_folder(task_id: int, name_if_new: str = ""):
    idstr = str(int(task_id))
    tmp = os.getenv(TMP_FOLDER_KEY_BASE + idstr)
    if tmp is None:
        tmp = ensure_folder("scgenerator_" + name_if_new + idstr)
        os.environ[TMP_FOLDER_KEY_BASE + idstr] = tmp
    return tmp


def set_data_folder(task_id: int, path: str):
    """stores the path to an existing data folder in the environment

    Parameters
    ----------
    task_id : int
        id uniquely identifying the session
    path : str
        path to the root of the data folder
    """
    idstr = str(int(task_id))
    os.environ[TMP_FOLDER_KEY_BASE + idstr] = path


def generate_file_path(file_name: str, task_id: int, identifier: str = "") -> str:
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
    # base_name, ext = os.path.splitext(file_name)
    # folder = get_data_folder(task_id)
    # folder = os.path.join(folder, identifier)
    # folder = ensure_folder(folder, prevent_overwrite=False)
    # i = 0
    # base_name = os.path.join(folder, base_name)
    # new_name = base_name + ext
    # while os.path.exists(new_name):
    #     new_name = f"{base_name}_{i}{ext}"
    #     i += 1

    path = os.path.join(get_data_folder(task_id), identifier)
    os.makedirs(path, exist_ok=True)
    path = os.path.join(path, file_name)

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


def ensure_folder(name, i=0, suffix="", prevent_overwrite=True):
    """creates a folder for simulation data named name and prevents overwrite
    by adding a suffix if necessary and returning the name"""
    prefix, last_dir = os.path.split(name)
    exploded = [prefix]
    sub_prefix = prefix
    while not _end_of_path_tree(sub_prefix):
        sub_prefix, _ = os.path.split(sub_prefix)
        exploded.append(sub_prefix)
    if any(os.path.isfile(el) for el in exploded):
        prefix = ensure_folder(prefix)
        name = os.path.join(prefix, last_dir)
    folder_name = name
    if i > 0:
        folder_name += f"_{i}"
    folder_name += suffix
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    else:
        if prevent_overwrite:
            return ensure_folder(name, i + 1)
        else:
            return folder_name
    return folder_name


def _end_of_path_tree(path):
    out = path == os.path.abspath(os.sep)
    out |= path == ""
    return out