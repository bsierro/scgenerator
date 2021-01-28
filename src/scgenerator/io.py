import itertools
import json
import logging
import os
from datetime import datetime
from glob import glob
from typing import Any, Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pkg_resources as pkg
import toml
from matplotlib.gridspec import GridSpec

from scgenerator import utilities
from scgenerator.const import TMP_FOLDER_KEY_BASE, num
from scgenerator.errors import IncompleteDataFolderError

from . import state


def load_toml(path):
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


def get_logger(name=None):
    """returns a logging.Logger instance. This function is there because if scgenerator
    is used with ray, workers are not aware of any configuration done with the logging
    and so it must be reconfigured.

    Parameters
    ----------
    name : str, optional
        name of the logger, by default None

    Returns
    -------
    logging.Logger obj
        logger
    """
    name = __name__ if name is None else name
    logger = logging.getLogger(name)
    return configure_logger(logger)


def configure_logger(logger, logfile="scgenerator.log"):
    """configures a logging.Logger obj

    Parameters
    ----------
    logger : logging.Logger
        logger to configure
    logfile : str or None, optional
        path to log file

    Returns
    -------
    logging.Logger obj
        updated logger
    """
    if not hasattr(logger, "already_configured"):
        if logfile is not None:
            file_handler = logging.FileHandler("scgenerator.log", "a+")
            file_handler.setFormatter(logging.Formatter("{name}: {message}", style="{"))
            logger.addHandler(file_handler)

        stream_handler = logging.StreamHandler()
        logger.addHandler(stream_handler)
        logger.setLevel(logging.INFO)

        logger.already_configured = True
    return logger


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
            if os.path.exists("paths.json"):
                with open("paths.json") as file:
                    paths_dico = json.load(file)
                for k, v in paths_dico.items():
                    cls.paths[k] = os.path.abspath(os.path.expanduser(v))
        if key not in cls.paths:
            print(f"{key} was not found in path index, returning current working directory.")
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
    forbiden_keys = ["w_c", "w_power_fact", "field_0", "w"]
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


def load_previous_parameters(path):
    """loads a parameters json files and converts data to appropriate type
    Parameters
    ----------
        path : path-like
            path to the json
    Returns
    ----------
        params : dict
    """
    params = load_toml(path)
    for k, v in params.items():
        if isinstance(v, list):
            if isinstance(v[0], (float, int)):
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


def load_sim_data(folder_name, ind=None, load_param=True):
    """
    loads the data already simulated.
    defauft shape is (z_targets, n, nt)

    Parameters
    ----------
        folder_name : (string) folder where the simulation data is stored
        ind : list of indices if only certain spectra are desired.
                - If left to None, returns every spectrum
                - If only 1 int, will cast the (1, n, nt) array into a (n, nt) array
        load_param : (bool) return the parameter dictionary as well. returns None
                if not available
        dico_name : name of the params dict stored in state.Params
    Returns
    ----------
        spectra : array
            squeezed array of complex spectra (n simulation on a nt size grid at each ind)
    Raises
    ----------
        FileNotFoundError : folder does not exist or does not contain sufficient
                            data
    """

    print(f"opening {folder_name}")

    # Check if file exists and assert how many z positions there are
    if not os.path.exists(folder_name):
        raise FileNotFoundError(f"Folder {folder_name} does not exist")
    nmax = len(glob(os.path.join(folder_name, "spectra_*.npy")))
    if nmax <= 0:
        raise FileNotFoundError(f"No appropriate file in specified folder {folder_name}")

    if ind is None:
        ind = range(nmax)
    elif isinstance(ind, int):
        ind = [ind]

    # Load the spectra
    spectra = []
    for i in ind:
        spectra.append(load_single_spectrum(folder_name, i))
    spectra = np.array(spectra)

    # Load the parameters dictionary
    try:
        params = load_previous_parameters(os.path.join(folder_name, "params.toml"))
    except FileNotFoundError:
        print(f"parameters corresponding to {folder_name} not found")
        params = None

    print("data successfully loaded")
    if load_param:
        return spectra.squeeze(), params
    else:
        return spectra.squeeze()


def get_all_environ() -> Dict[str, str]:
    """returns a dictionary of all environment variables set by any instance of scgenerator"""
    return dict(filter(lambda el: el[0].startswith(TMP_FOLDER_KEY_BASE), os.environ.items()))


def load_single_spectrum(folder, index) -> np.ndarray:
    return np.load(os.path.join(folder, f"spectra_{index}.npy"))


def iter_load_sim_data(folder_name, with_params=False) -> Iterable[np.ndarray]:
    """
    similar to load_sim_data but works as an iterator
    """

    if not os.path.exists(folder_name):
        raise FileNotFoundError(f"Folder {folder_name} does not exist")
    nmax = len(glob(os.path.join(folder_name, "spectra_*.npy")))
    if nmax <= 0:
        raise FileNotFoundError(f"No appropriate file in specified folder {folder_name}")

    params = {}
    if with_params:
        try:
            params = load_previous_parameters(os.path.join(folder_name, "params.toml"))
        except FileNotFoundError:
            print(f"parameters corresponding to {folder_name} not found")
            params = None

    print(f"iterating through {folder_name}")
    for i in range(nmax):
        if with_params:
            yield load_single_spectrum(folder_name, i), params
        else:
            yield load_single_spectrum(folder_name, i)


def _get_data_subfolders(path: str) -> List[str]:
    """returns a list of subfolders in the specified directory

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


# def _sort_subfolder_list(
#     sub_folders: List[str], varying_lists: List[Tuple[str, Any]]
# ) -> Tuple[list]:
#     """sorts the two lists in parallel according to parameter values

#     Parameters
#     ----------
#     sub_folders : List[str]
#         paths to where spectra are loaded
#     varying_lists : List[Tuple]
#         (param_name, value) tuples corresponding to the sub_folders

#     Returns
#     -------
#     Tuple[list]
#         the input, sorted
#     """
#     both_lists = list(zip(sub_folders, varying_lists))
#     for i in range(len(varying_lists[0])):
#         both_lists.sort(key=lambda el: el[1][i][1])
#     return tuple(zip(*both_lists))


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
        params = load_toml(os.path.join(sub_folder, "params.toml"))
        z_num = params["z_num"]
        num_spectra = len(glob(os.path.join(sub_folder, "spectrum*.npy")))

        if z_num != init_z_num:
            raise IncompleteDataFolderError(
                f"initial config specifies {init_z_num} spectra per"
                + f" but the parameter file in {sub_folder} specifies {z_num}"
            )

        if num_spectra != z_num:
            raise IncompleteDataFolderError(
                f"only {num_spectra} spectra found in {sub_folder} instead of the specified {z_num}"
            )


# def preprocess_data_folder(path: str) -> bool:
#     config = load_toml(os.path.join(path, "initial_config.toml"))
#     num_sims, _ = utilities.count_variations(config)
#     sub_folders = _get_data_subfolders(path)
#     init_z_num = config["simulation"]["z_num"]

#     if len(sub_folders) != num_sims:
#         raise IncompleteDataFolderError(
#             f"only {len(sub_folders)} simulations out of {num_sims} have been made"
#         )

#     varying_lists = [utilities.varying_list_from_path(os.path.split(s)[1]) for s in sub_folders]
#     varying_params = [el[0] for el in varying_lists[0]]
#     sub_folders, varying_lists = _update_varying_lists(
#         sub_folders, varying_lists, varying_params, init_z_num
#     )

#     possible_values = []
#     for i, p in enumerate(varying_params):
#         tmp = set()
#         for v_list in varying_lists:
#             tmp.add(v_list[i][1])
#         tmp = list(tmp)
#         possible_values.append(tmp)

#     return sub_folders, varying_lists, varying_params, possible_values, init_z_num


# def merge_data(path: str):
#     sub_folders, varying_lists, varying_params, possible_values, z_num = preprocess_data_folder(
#         path
#     )
#     z_values = list(range(z_num))
#     pt = utilities.ProgressTracker(z_num, auto_print=True)
#     shape = tuple((len(l) for l in possible_values))
#     for z_num in z_values:
#         to_save = []
#         for i in range(np.product(shape)):
#             to_save.append(np.load(os.path.join(sub_folders[i], f"spectrum_{z_num}.npy")))
#         out = np.array(to_save).reshape((*shape, len(to_save[0])))
#         np.save(os.path.join(path, f"spectra_{z_num}.npy"), out)
#         pt.update()
#     _create_reference_file(varying_params, possible_values)
#     return


def merge_same_simulations(path: str):
    num_separator = "_num_"
    sub_folders = _get_data_subfolders(path)
    config = load_toml(os.path.join(path, "initial_config.toml"))
    repeat = config["simulation"].get("repeat", 1)
    z_num = config["simulation"]["z_num"]

    check_data_integrity(sub_folders, z_num)

    base_folders = set()
    for sub_folder in sub_folders:
        splitted_base_path = sub_folder.split(num_separator)[:-1]
        base_folder = num_separator.join(splitted_base_path)
        if len(base_folder) > 0:
            base_folders.add(base_folder)

    print(base_folders)
    for base_folder in base_folders:
        for j in range(z_num):
            spectra = []
            for i in range(repeat):
                spectra.append(
                    np.load(os.path.join(f"{base_folder}{num_separator}{i}/spectrum_{j}.npy"))
                )
            dest_folder = ensure_folder(base_folder, prevent_overwrite=False)
            spectra = np.array(spectra).reshape(repeat, len(spectra[0]))
            np.save(os.path.join(dest_folder, f"spectra_{j}.npy"), spectra)


# class tmp_index_manager:
#     """Manages a temporary index of files while the simulation is running
#     and merge them at the end automatically"""

#     def __init__(self, config_name="untitled", task_id=0, varying_keys=None):

#         self.path = os.path.join(Paths.tmp(task_id), "index.json")
#         self.config_name = config_name
#         self.varying_keys = varying_keys

#         # set up the directories
#         if not os.path.exists(Paths.tmp(task_id)):
#             os.makedirs(Paths.tmp(task_id))

#         file_num = 0
#         while os.path.exists(self.path):
#             self.path = os.path.join(Paths.tmp(task_id), f"index_{file_num}.json")
#             file_num += 1

#         self.index = dict(spectra={}, z={}, params={})
#         self.ids = set()

#         with open(self.path, "w") as file:
#             json.dump(self.index, file)

#     def get_path(self):
#         return self.path

#     def append_to_index(self, param_id, spectra_file_name="", params_file_name=""):
#         """add one or two files to the index
#         Parameters
#         ----------
#             param_id : id of the parameter set
#             spectra_file_name : name of the spectra file
#             params_file_name : name of the parameters file
#         Returns
#         ----------
#             None
#         """

#         # names of the recorded values in order
#         # here : {"spectra":spectra_file_name, "params":params_file_name}
#         file_names = [spectra_file_name, params_file_name]
#         file_names_dict = dict(zip(state.recorded_types, file_names))

#         param_id = str(param_id)
#         self.ids.add(param_id)

#         with open(self.path, "r") as file:
#             self.index = json.loads(file.read())

#         for type_name, file_name in file_names_dict.items():
#             if file_name != "":
#                 if param_id not in self.index[type_name]:
#                     self.index[type_name][param_id] = []
#                 self.index[type_name][param_id].append(file_name)

#         with open(self.path, "w") as file:
#             json.dump(self.index, file)

#     def convert_sim_data(self):
#         return convert_sim_data(self.path, name=self.config_name, varying_keys=self.varying_keys)


# def convert_sim_data(path, name="untitled", ids=None, varying_keys=[], delete_temps=True):
#     """Converts simulation data that are stored as 1 file/simulation to 1 file
#     per parameters set
#     Parameters
#     ----------
#         path : path to the index containing infos about how to group files together
#         name : name of the final folder
#         ids : list of ids, 1 per set of parameters

#     Returns
#     ----------
#         path to the converted data

#     """
#     with open(path, "r") as file:
#         index = json.loads(file.read())

#     folder_0 = os.path.join(Paths.get("data"), name)
#     folder_0 = ensure_folder(folder_0)  # related to the set of simulation / job

#     # find the ids if not stored already
#     if ids is None:
#         ids = set()
#         for key in state.recorded_types:
#             for k in index[key]:
#                 ids.add(k)

#     not_found = []

#     for param_id in ids:

#         print("ids", ids)

#         # Load the spectra
#         spectra = []
#         for f in index["spectra"][param_id]:
#             try:
#                 spectra.append(np.load(f))
#             except FileNotFoundError:
#                 not_found.append(f)
#                 index["spectra"][param_id].remove(f)

#         spectra = np.array(spectra)

#         # Load the params
#         main_param_name = index["params"][param_id][0] + ".json"
#         try:
#             with open(main_param_name, "r") as file:
#                 params = json.load(file)
#         except FileNotFoundError:
#             print(f"no parameters for id {param_id} found. Skipping this one")
#             not_found += index["params"][param_id]
#             continue

#         if len(not_found) > 0:
#             print(f"{len(not_found)} files not found:")
#             for file_not_found in not_found:
#                 print("\t" + file_not_found)

#         # create sub folder
#         if len(ids) > 1:
#             complement = [param_id]
#             for key in varying_keys:
#                 if key in ["T0_FWHM", "P0"]:
#                     key = "init_" + key
#                 complement.append(key)
#                 complement.append(format(params.get(key, 0), ".2e").split("e")[0])

#             folder_1 = "_".join(complement)  # related to specific parameter
#             folder_name = os.path.join(folder_0, folder_1)
#         else:
#             folder_name = folder_0

#         if not os.path.exists(folder_name):
#             os.makedirs(folder_name)
#         os.rename(main_param_name, os.path.join(folder_name, "param.json"))

#         # Save the data in a more easily manageable format (one file per z position)
#         for k in range(len(spectra[0])):
#             np.save(os.path.join(folder_name, f"spectra_{k}"), spectra[:, k])
#         print(f"{len(spectra)} simulations converted. Data saved in {folder_name}")

#         deleted = 0
#         if delete_temps:
#             # once everything is saved, delete the temporary files to free up space
#             param_file_names = [f + ".json" for f in index["params"][param_id]]
#             try:
#                 param_file_names.remove(main_param_name)
#             except ValueError:
#                 pass

#             fail_list = []
#             for f in index["spectra"][param_id] + param_file_names:
#                 try:
#                     os.remove(f)
#                     deleted += 1
#                 except FileNotFoundError:
#                     fail_list.append(f)

#             if len(fail_list) > 0:
#                 print(f"could not remove {len(fail_list)} temporary files :")
#                 for failed in fail_list:
#                     print("\t" + failed)

#         print(f"Merge finished, deleted {deleted} temporary files.")

#     if delete_temps:
#         os.remove(path)
#         delete_tmp_folder()
#     return folder_0


# def delete_tmp_folder():
#     """deletes temporary folders if they are empty"""
#     for folder in glob(Paths.tmp()):
#         try:
#             os.rmdir(folder)
#         except OSError as err:
#             print(err)


def get_data_folder(task_id: int, name_if_new: str = ""):
    idstr = str(int(task_id))
    tmp = os.getenv(TMP_FOLDER_KEY_BASE + idstr)
    if tmp is None:
        tmp = ensure_folder("scgenerator_" + name_if_new + idstr)
        tmp = os.path.abspath(tmp)
        os.environ[TMP_FOLDER_KEY_BASE + idstr] = tmp
    return tmp


def generate_file_path(file_name: str, task_id: int, sub_folder: str = "") -> str:
    """generates a path for the desired file name

    Parameters
    ----------
    file_name : str
        desired file name. May be altered if it already exists
    task_id : int
        unique id of the process
    sub_folder : str
        subfolder in which to store the file. default : ""

    Returns
    -------
    str
        the full path
    """
    base_name, ext = os.path.splitext(file_name)
    folder = get_data_folder(task_id)
    folder = os.path.join(folder, sub_folder)
    folder = ensure_folder(folder, prevent_overwrite=False)
    i = 0
    base_name = os.path.join(folder, base_name)
    new_name = base_name + ext
    while os.path.exists(new_name):
        print(f"{i=}")
        new_name = f"{base_name}_{i}{ext}"
        i += 1

    return new_name


def save_data(data: np.ndarray, file_name: str, task_id: int, subfolder: str = ""):
    """saves numpy array to disk

    Parameters
    ----------
    data : np.ndarray
        data to save
    file_name : str
        file name
    task_id : int
        id that uniquely identifies the process
    subfolder : str, optional
        subfolder in the main data folder of the task, by default ""
    """
    path = generate_file_path(file_name, task_id, subfolder)
    np.save(path, data)


def generate_tmp_file_name_old(file_name, job_id=0, param_id=0, task_id=0, ext=""):
    """returns a guaranteed available file name"""
    main_suffix = f"_JOBID{job_id}_PARAMID{param_id}"
    suffix = main_suffix + "_" + str(0)

    no_dup = 1
    while os.path.exists(os.path.join(Paths.tmp(task_id), file_name + suffix + ext)):
        suffix = main_suffix + "_" + str(no_dup)
        no_dup += 1

    return os.path.join(Paths.tmp(task_id), file_name + suffix + ext)


def ensure_folder(name, i=0, suffix="", prevent_overwrite=True):
    """creates a folder for simulation data named name and prevents overwrite
    by adding a suffix if necessary and returning the name"""
    prefix, last_dir = os.path.split(os.path.abspath(name))
    exploded = [prefix]
    sub_prefix = prefix
    while sub_prefix != os.path.abspath("/"):
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


# class Logger:
#     def __init__(self, print_level=10000):
#         """
#         Parameters
#         ----------
#             print_level : messages above this priority will be printed as well as recorded
#         """
#         log_file_name = (
#             "scgenerator_log_"
#             + format(datetime.today())[:-7].replace(" ", "_").replace(":", "-")
#             + ".txt"
#         )
#         self.log_file = os.path.join(Paths.get("logs"), log_file_name)
#         self.print_level = print_level
#         self.prefix_length = 0
#         self.default_prefix = "Main Thread"

#         if not os.path.exists(self.log_file):
#             with open(self.log_file, "w"):
#                 pass

#         with open(self.log_file, "a") as file:
#             file.write(
#                 f"\n---------------------------\nNew Log {str(datetime.today()):19.19}\n---------------------------\n"
#             )

#     def log(self, s, priority=0, prefix=None):
#         """logs a message
#         Parameters
#         ----------
#             s : the string to log
#             priority : will be compared to the logger's print_level to decide whether to print the string
#             prefix : string identifying which thread or part of the program is giving the message
#         Returns
#         ----------
#             nothing
#         """
#         if prefix is None:
#             prefix = self.default_prefix
#         if priority >= self.print_level:
#             print(s)
#         with open(self.log_file, "a") as file:
#             if len(prefix) > self.prefix_length:
#                 self.prefix_length = len(prefix)
#             prefix = format(prefix[: self.prefix_length], str(self.prefix_length))
#             file.write(prefix + " : " + str(s) + "\n")


def plot_setup(
    folder_name=None,
    file_name=None,
    file_type="png",
    figsize=state.plot_default_figsize,
    params=None,
    mode="default",
):
    """It should return :
    - a folder_name
    - a file name
    - a fig
    - an axis
    """
    file_name = state.plot_default_name if file_name is None else file_name

    if params is not None:
        folder_name = params.get("plot.folder_name", folder_name)
        file_name = params.get("plot.file_name", file_name)
        file_type = params.get("plot.file_type", file_type)
        figsize = params.get("plot.figsize", figsize)

    # ensure output folder_name exists
    folder_name, file_name = (
        os.path.split(file_name)
        if folder_name is None
        else (folder_name, os.path.split(file_name)[1])
    )
    folder_name = os.path.join(Paths.get("plots"), folder_name)
    if not os.path.exists(os.path.abspath(folder_name)):
        os.makedirs(os.path.abspath(folder_name))

    # ensure no overwrite
    ind = 0
    while os.path.exists(os.path.join(folder_name, file_name + "_" + str(ind) + "." + file_type)):
        ind += 1
    file_name = file_name + "_" + str(ind) + "." + file_type

    if mode == "default":
        fig, ax = plt.subplots(figsize=figsize)
    elif mode == "coherence":
        n = state.plot_avg_default_main_to_coherence_ratio
        gs1 = GridSpec(n + 1, 1, hspace=0.4)
        fig = plt.figure(figsize=state.plot_default_figsize)
        top = fig.add_subplot(gs1[:n])
        top.tick_params(labelbottom=False)
        bot = fig.add_subplot(gs1[n], sharex=top)

        bot.set_ylim(-0.1, 1.1)
        bot.set_ylabel(r"|$g_{12}$|")
        ax = (top, bot)
    elif mode == "coherence_T":
        n = state.plot_avg_default_main_to_coherence_ratio
        gs1 = GridSpec(1, n + 1, wspace=0.4)
        fig = plt.figure(figsize=state.plot_default_figsize)
        top = fig.add_subplot(gs1[:n])
        top.tick_params(labelleft=False, left=False, right=True)
        bot = fig.add_subplot(gs1[n], sharey=top)

        bot.set_xlim(1.1, -0.1)
        bot.set_xlabel(r"|$g_{12}$|")
        ax = (top, bot)
    else:
        raise ValueError(f"mode {mode} not understood")

    return folder_name, file_name, fig, ax
