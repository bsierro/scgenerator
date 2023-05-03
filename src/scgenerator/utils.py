"""
This files includes utility functions designed more or less to be used specifically with the
scgenerator module but some function may be used in any python program

"""
from __future__ import annotations

import datetime
import inspect
import itertools
import os
import re
from collections import defaultdict
from dataclasses import dataclass
from functools import cache, lru_cache
from pathlib import Path
from string import printable as str_printable
from typing import Any, Callable, MutableMapping, Sequence, TypeVar, Union

import numpy as np
from numpy.ma.extras import isin
import pkg_resources as pkg
import tomli
import tomli_w

from scgenerator.const import (PARAM_FN, PARAM_SEPARATOR, ROOT_PARAMETERS,
                               SPEC1_FN, Z_FN)
from scgenerator.errors import DuplicateParameterError
from scgenerator.logger import get_logger

T_ = TypeVar("T_")



class TimedMessage:
    def __init__(self, interval: float = 10.0):
        self.interval = datetime.timedelta(seconds=interval)
        self.next = datetime.datetime.now()

    def ready(self) -> bool:
        t = datetime.datetime.now()
        if self.next <= t:
            self.next = t + self.interval
            return True
        return False


class Paths:
    _data_files = [
        "materials.toml",
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
                with open("paths.toml", "rb") as file:
                    paths_dico = tomli.load(file)
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


@dataclass(init=False)
class SubConfig:
    fixed: dict[str, Any]
    variable: list[dict[str, list]]
    fixed_keys: set[str]
    variable_keys: set[str]

    def __init__(self, dico: dict[str, Any]):
        dico = dico.copy()
        self.variable = conform_variable_entry(dico.pop("variable", []))
        self.fixed = dico
        self.__update

    def __update(self):
        self.variable_keys = set()
        self.fixed_keys = set()
        for dico in self.variable:
            for key in dico:
                if key in self.variable_keys:
                    raise DuplicateParameterError(f"{key} is specified twice")
                self.variable_keys.add(key)
        for key in self.fixed:
            if key in self.variable_keys:
                raise DuplicateParameterError(f"{key} is specified twice")
            self.fixed_keys.add(key)

    def weak_update(self, other: SubConfig = None, **kwargs):
        """similar to a dict update method put prioritizes existing values

        Parameters
        ----------
        other : SubConfig
            other obj
        """
        if other is None:
            other = SubConfig(kwargs)
        self.fixed = other.fixed | self.fixed
        self.variable = other.variable + self.variable
        self.__update()


def conform_variable_entry(d) -> list[dict[str, list]]:
    if isinstance(d, MutableMapping):
        d = [{k: v} for k, v in d.items()]
    return d


def load_previous_spectrum(prev_data_dir: str) -> np.ndarray:
    prev_data_dir = Path(prev_data_dir)
    num = find_last_spectrum_num(prev_data_dir)
    return load_spectrum(prev_data_dir / SPEC1_FN.format(num))


@lru_cache(20000)
def load_spectrum(file: os.PathLike) -> np.ndarray:
    return np.load(file)


def conform_toml_path(path: os.PathLike) -> Path:
    path: str = str(path)
    if not path.lower().endswith(".toml"):
        path = path + ".toml"
    return Path(path)


def open_single_config(path: os.PathLike) -> dict[str, Any]:
    d = _open_config(path)
    f = d.pop("Fiber", [{}])[0]
    return d | f


def _open_config(path: os.PathLike):
    """returns a dictionary parsed from the specified toml file
    This also handle having a 'INCLUDE' argument that will fill
    otherwise unspecified keys with what's in the INCLUDE file(s)"""

    path = conform_toml_path(path)
    dico = resolve_loadfile_arg(load_toml(path))

    if "Fiber" not in dico:
        dico = dict(name=path.name, Fiber=[dico])

    resolve_relative_paths(dico, path.parent)

    return dico

def resolve_relative_paths(d:dict[str, Any], root:os.PathLike | None=None):
    root = Path(root) if root is not None else Path.cwd()
    for k, v in d.items():
        if isinstance(v, MutableMapping):
            resolve_relative_paths(v, root)
        elif not isinstance(v, str) and isinstance(v, Sequence):
            for el in v:
                if isinstance(el, MutableMapping):
                    resolve_relative_paths(el, root)
        elif "file" in k:
            d[k] = str(root / v)



def resolve_loadfile_arg(dico: dict[str, Any]) -> dict[str, Any]:
    if (f_list := dico.pop("INCLUDE", None)) is not None:
        if isinstance(f_list, str):
            f_list = [f_list]
        for to_load in f_list:
            loaded = load_toml(to_load)
            for k, v in loaded.items():
                if k not in dico and k not in dico.get("variable", {}):
                    dico[k] = v
    for k, v in dico.items():
        if isinstance(v, MutableMapping):
            dico[k] = resolve_loadfile_arg(v)
        elif isinstance(v, Sequence):
            for i, vv in enumerate(v):
                if isinstance(vv, MutableMapping):
                    dico[k][i] = resolve_loadfile_arg(vv)
    return dico


def load_toml(descr: os.PathLike) -> dict[str, Any]:
    descr = str(descr)
    if ":" in descr:
        path, entry = descr.split(":", 1)
        with open(path, "rb") as file:
            return tomli.load(file)[entry]
    else:
        with open(descr, "rb") as file:
            return tomli.load(file)


def load_flat(descr: os.PathLike) -> dict[str, Any]:
    with open(descr, "rb") as file:
        d = tomli.load(file)
    if "Fiber" in d:
        for fib in d["Fiber"]:
            for k, v in fib.items():
                d[k] = v
            break
    return d


def save_toml(path: os.PathLike, dico):
    """saves a dictionary into a toml file"""
    path = conform_toml_path(path)
    with open(path, mode="wb") as file:
        tomli_w.dump(dico, file)
    return dico


def load_config_sequence(path: os.PathLike) -> tuple[Path, list[SubConfig]]:
    """loads a configuration file

    Parameters
    ----------
    path : os.PathLike
        path to the config toml file or a directory containing config files

    Returns
    -------
    final_path : Path
        output name of the simulation
    list[dict[str, Any]]
        one config per fiber

    """
    path = Path(path)
    if path.name.lower().endswith(".toml"):
        master_config_dict = _open_config(path)
        fiber_list = [SubConfig(d) for d in master_config_dict.pop("Fiber")]
        master_config = SubConfig(master_config_dict)
    else:
        master_config = SubConfig(dict(name=path.name))
        fiber_list = [SubConfig(_open_config(p)) for p in sorted(path.glob("initial_config*.toml"))]

    if len(fiber_list) == 0:
        raise ValueError(f"No fiber in config {path}")
    for fiber in fiber_list:
        fiber.weak_update(master_config)
    if "num" not in fiber_list[0].variable_keys:
        repeat_arg = list(range(fiber_list[0].fixed.get("repeat", 1)))
        fiber_list[0].weak_update(variable=dict(num=repeat_arg))
    for p_name in ROOT_PARAMETERS:
        if any(p_name in conf.variable_keys for conf in fiber_list[1:]):
            raise ValueError(f"{p_name} should only be specified in the root or first fiber")
    configs = fiber_list
    return Path(master_config.fixed["name"]), configs


@cache
def load_material_dico(name: str) -> dict[str, Any]:
    """loads a material dictionary
    Parameters
    ----------
        name : str
            name of the material
    Returns
    ----------
        material_dico : dict
    """
    return tomli.loads(Paths.gets("materials"))[name]


def save_data(data: Union[np.ndarray, MutableMapping], data_dir: Path, file_name: str):
    """saves numpy array to disk

    Parameters
    ----------
    data : Union[np.ndarray, MutableMapping]
        data to save
    file_name : str
        file name
    task_id : int
        id that uniquely identifies the process
    identifier : str, optional
        identifier in the main data folder of the task, by default ""
    """
    path = data_dir / file_name
    if isinstance(data, np.ndarray):
        np.save(path, data)
    elif isinstance(data, MutableMapping):
        np.savez(path, **data)
    get_logger(__name__).debug(f"saved data in {path}")
    return


def ensure_folder(path: Path, prevent_overwrite: bool = True, mkdir=True) -> Path:
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
            path = ensure_folder(path, mkdir=mkdir, prevent_overwrite=False)
        path /= part

    folder_name = path.name

    for i in itertools.count():
        if not path.is_file() and (not prevent_overwrite or not path.is_dir()):
            if mkdir:
                path.mkdir(exist_ok=True)
            return path
        path = path.parent / (folder_name + f"_{i}")


def branch_id(branch: Path) -> tuple[int, int]:
    sim_match = branch.resolve().parent.name.split()[0]
    if sim_match.isdigit():
        s_int = int(sim_match)
    else:
        s_int = 0
    branch_match = re.search(r"(?<=b_)[0-9]+", branch.name)
    if branch_match is None:
        b_int = 0
    else:
        b_int = int(branch_match[0])
    return s_int, b_int


def find_last_spectrum_num(data_dir: Path):
    for num in itertools.count(1):
        p_to_test = data_dir / SPEC1_FN.format(num)
        if not p_to_test.is_file() or os.path.getsize(p_to_test) == 0:
            return num - 1


def auto_crop(x: np.ndarray, y: np.ndarray, rel_thr: float = 0.01) -> np.ndarray:
    threshold = y.min() + rel_thr * (y.max() - y.min())
    above_threshold = y > threshold
    ind = np.argsort(x)
    valid_ind = [
        np.array(list(g)) for k, g in itertools.groupby(ind, key=lambda i: above_threshold[i]) if k
    ]
    ind_above = sorted(valid_ind, key=lambda el: len(el), reverse=True)[0]
    width = len(ind_above)
    return np.concatenate(
        (
            np.arange(max(ind_above[0] - width, 0), ind_above[0]),
            ind_above,
            np.arange(ind_above[-1] + 1, min(len(y), ind_above[-1] + width)),
        )
    )


def to_62(i: int) -> str:
    arr = []
    if i == 0:
        return "0"
    i = abs(i)
    while i:
        i, value = divmod(i, 62)
        arr.append(str_printable[value])
    return "".join(reversed(arr))


def get_arg_names(func: Callable) -> list[str]:
    """returns the positional argument names of func.

    Parameters
    ----------
    func : Callable
        if a function, returns the names of the positional arguments


    Returns
    -------
    list[str]
        [description]
    """
    return [k for k, v in inspect.signature(func).parameters.items() if v.default is inspect._empty]


def validate_arg_names(names: list[str]):
    for n in names:
        if re.match(r"^[^\s\-'\(\)\"\d][^\(\)\-\s'\"]*$", n) is None:
            raise ValueError(f"{n} is an invalid parameter name")


def func_rewrite(func: Callable, kwarg_names: list[str], arg_names: list[str] = None) -> Callable:
    if arg_names is None:
        arg_names = get_arg_names(func)
    else:
        validate_arg_names(arg_names)
    validate_arg_names(kwarg_names)
    sign_arg_str = ", ".join(arg_names + kwarg_names)
    call_arg_str = ", ".join(arg_names + [f"{s}={s}" for s in kwarg_names])
    tmp_name = f"{func.__name__}_0"
    func_str = f"def {tmp_name}({sign_arg_str}):\n    return __func__({call_arg_str})"
    scope = dict(__func__=func)
    exec(func_str, scope)
    out_func = scope[tmp_name]
    out_func.__module__ = "evaluator"
    return out_func


@cache
def _mock_function(num_args: int, num_returns: int) -> Callable:
    arg_str = ", ".join("a" * (n + 1) for n in range(num_args))
    return_str = ", ".join("True" for _ in range(num_returns))
    func_name = f"__mock_{num_args}_{num_returns}"
    func_str = f"def {func_name}({arg_str}):\n    return {return_str}"
    scope = {}
    exec(func_str, scope)
    out_func = scope[func_name]
    out_func.__module__ = "evaluator"
    return out_func


def fft_functions(
    full_field: bool,
) -> tuple[Callable[[np.ndarray], np.ndarray], Callable[[np.ndarray], np.ndarray]]:
    if full_field:
        return np.fft.rfft, np.fft.irfft
    else:
        return np.fft.fft, np.fft.ifft


def combine_simulations(path: Path, dest: Path = None):
    """combines raw simulations into one folder per branch

    Parameters
    ----------
    path : Path
        source of the simulations (must contain u_xx directories)
    dest : Path, optional
        if given, moves the simulations to dest, by default None
    """
    paths: dict[str, list[Path]] = defaultdict(list)
    if dest is None:
        dest = path

    for p in path.glob("u_*b_*"):
        if p.is_dir():
            paths[p.name.split()[1]].append(p)
    for l in paths.values():
        try:
            l.sort(key=lambda el: re.search(r"(?<=num )[0-9]+", el.name)[0])
        except TypeError:
            pass
    for pulses in paths.values():
        new_path = dest / update_path_name(pulses[0].name)
        os.makedirs(new_path, exist_ok=True)
        for num, pulse in enumerate(pulses):
            params_ok = False
            for file in pulse.glob("*"):
                if file.name == PARAM_FN:
                    if not params_ok:
                        update_params(new_path, file)
                        params_ok = True
                    else:
                        file.unlink()
                elif file.name == Z_FN:
                    file.rename(new_path / file.name)
                elif file.name.startswith("spectr") and num == 0:
                    file.rename(new_path / file.name)
                else:
                    file.rename(new_path / (file.stem + f"_{num}" + file.suffix))
            pulse.rmdir()


def update_params(new_path: Path, file: Path):
    params = load_toml(file)
    if (p := params.get("prev_data_dir")) is not None:
        p = Path(p)
        params["prev_data_dir"] = str(Path("../..") / p.parent.name / update_path_name(p.name))
    params["output_path"] = str(new_path)
    save_toml(new_path / PARAM_FN, params)
    file.unlink()


def save_parameters(
    params: dict[str, Any], destination_dir: Path, file_name: str = PARAM_FN
) -> Path:
    """saves a parameter dictionary. Note that is does remove some entries, particularly
    those that take a lot of space ("t", "w", ...)

    Parameters
    ----------
    params : dict[str, Any]
        dictionary to save
    destination_dir : Path
        destination directory

    Returns
    -------
    Path
        path to newly created the paramter file
    """
    file_path = destination_dir / file_name
    os.makedirs(file_path.parent, exist_ok=True)

    # save toml of the simulation
    with open(file_path, "wb") as file:
        tomli_w.dump(params, file)

    return file_path


def update_path_name(p: str) -> str:
    return re.sub(r"( ?num [0-9]+)|(u_[0-9]+ )", "", p)


def fiber_folder(i: int, sim_name: str, fiber_name: str) -> str:
    return PARAM_SEPARATOR.join([format(i), sim_name, fiber_name])


def simulations_list(path: os.PathLike) -> list[Path]:
    """finds simulations folders contained in a parent directory

    Parameters
    ----------
    path : os.PathLike
        parent path

    Returns
    -------
    list[Path]
        Absolute Path to the simulation folder
    """
    paths: list[Path] = []
    for pwd, _, files in os.walk(path):
        if PARAM_FN in files and SPEC1_FN.format(0) in files:
            paths.append(Path(pwd))
    paths.sort(key=branch_id)
    return [p for p in paths if p.parent.name == paths[-1].parent.name]
