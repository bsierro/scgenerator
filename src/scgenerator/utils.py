"""
This files includes utility functions designed more or less to be used specifically with the
scgenerator module but some function may be used in any python program

"""
from __future__ import annotations
from dataclasses import dataclass
import inspect
import itertools
import os
import re
from collections import defaultdict
from functools import cache
from pathlib import Path
from string import printable as str_printable
from typing import Any, Callable, MutableMapping, Sequence, TypeVar

import numpy as np
import pkg_resources as pkg
import toml

from .const import PARAM_FN, PARAM_SEPARATOR, SPEC1_FN, Z_FN
from .logger import get_logger

T_ = TypeVar("T_")


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


class ConfigFileParser:
    path: Path
    repeat: int
    master: ConfigFileParser.SubConfig
    configs: list[ConfigFileParser.SubConfig]

    @dataclass
    class SubConfig:
        fixed: dict[str, Any]
        variable: dict[str, list]

    def __init__(self, path: os.PathLike):
        self.path = Path(path)
        fiber_list: list[dict[str, Any]]
        if self.path.name.lower().endswith(".toml"):
            loaded_config = _open_config(self.path)
            fiber_list = loaded_config.pop("Fiber")
        else:
            loaded_config = dict(name=self.path.name)
            fiber_list = [_open_config(p) for p in sorted(self.path.glob("initial_config*.toml"))]

        if len(fiber_list) == 0:
            raise ValueError(f"No fiber in config {self.path}")
        final_path = loaded_config.get("name")
        configs = []
        for i, params in enumerate(fiber_list):
            configs.append(loaded_config | params)
        for root_vary, first_vary in itertools.product(
            loaded_config["variable"], configs[0]["variable"]
        ):
            if len(common := root_vary.keys() & first_vary.keys()) != 0:
                raise ValueError(f"These variable keys are specified twice : {common!r}")
        configs[0] |= {k: v for k, v in loaded_config.items() if k != "variable"}
        configs[0]["variable"].append(dict(num=list(range(configs[0].get("repeat", 1)))))


def load_previous_spectrum(prev_data_dir: str) -> np.ndarray:
    prev_data_dir = Path(prev_data_dir)
    num = find_last_spectrum_num(prev_data_dir)
    return load_spectrum(prev_data_dir / SPEC1_FN.format(num))


@cache
def load_spectrum(file: os.PathLike) -> np.ndarray:
    return np.load(file)


def conform_toml_path(path: os.PathLike) -> Path:
    path: str = str(path)
    if not path.lower().endswith(".toml"):
        path = path + ".toml"
    return Path(path)


def open_single_config(path: os.PathLike) -> dict[str, Any]:
    d = _open_config(path)
    f = d.pop("Fiber")[0]
    return d | f


def _open_config(path: os.PathLike):
    """returns a dictionary parsed from the specified toml file
    This also handle having a 'INCLUDE' argument that will fill
    otherwise unspecified keys with what's in the INCLUDE file(s)"""

    path = conform_toml_path(path)
    dico = resolve_loadfile_arg(load_toml(path))

    dico = standardize_variable_dicts(dico)
    if "Fiber" not in dico:
        dico = dict(name=path.name, Fiber=[dico])
    return dico


def standardize_variable_dicts(dico: dict[str, Any]):
    if "Fiber" in dico:
        dico["Fiber"] = [standardize_variable_dicts(fiber) for fiber in dico["Fiber"]]
    if (var := dico.get("variable")) is not None:
        if isinstance(var, MutableMapping):
            dico["variable"] = [var]
    else:
        dico["variable"] = [{}]
    return dico


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
        with open(path) as file:
            return toml.load(file)[entry]
    else:
        with open(descr) as file:
            return toml.load(file)


def save_toml(path: os.PathLike, dico):
    """saves a dictionary into a toml file"""
    path = conform_toml_path(path)
    with open(path, mode="w") as file:
        toml.dump(dico, file)
    return dico


def load_config_sequence(path: os.PathLike) -> tuple[Path, list[dict[str, Any]]]:
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
    fiber_list: list[dict[str, Any]]
    if path.name.lower().endswith(".toml"):
        loaded_config = _open_config(path)
        fiber_list = loaded_config.pop("Fiber")
    else:
        loaded_config = dict(name=path.name)
        fiber_list = [_open_config(p) for p in sorted(path.glob("initial_config*.toml"))]

    if len(fiber_list) == 0:
        raise ValueError(f"No fiber in config {path}")
    final_path = loaded_config.get("name")
    configs = []
    for i, params in enumerate(fiber_list):
        configs.append(loaded_config | params)
    for root_vary, first_vary in itertools.product(
        loaded_config["variable"], configs[0]["variable"]
    ):
        if len(common := root_vary.keys() & first_vary.keys()) != 0:
            raise ValueError(f"These variable keys are specified twice : {common!r}")
    configs[0] |= {k: v for k, v in loaded_config.items() if k != "variable"}
    configs[0]["variable"].append(dict(num=list(range(configs[0].get("repeat", 1)))))
    return Path(final_path), configs


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
    return toml.loads(Paths.gets("materials"))[name]


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


def branch_id(branch: tuple[Path, ...]) -> str:
    return branch[-1].name.split()[1]


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


def translate_parameters(d: dict[str, Any]) -> dict[str, Any]:
    old_names = dict(
        interp_degree="interpolation_degree",
        beta="beta2_coefficients",
        interp_range="interpolation_range",
    )
    deleted_names = {"lower_wavelength_interp_limit", "upper_wavelength_interp_limit"}
    defaults_to_add = dict(repeat=1)
    new = {}
    for k, v in d.items():
        if k == "error_ok":
            new["tolerated_error" if d.get("adapt_step_size", True) else "step_size"] = v
        elif k in deleted_names:
            continue
        elif isinstance(v, MutableMapping):
            new[k] = translate_parameters(v)
        else:
            new[old_names.get(k, k)] = v
    return defaults_to_add | new


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
    # spec = inspect.getfullargspec(func)
    # args = spec.args
    # if spec.defaults is not None and len(spec.defaults) > 0:
    #     args = args[: -len(spec.defaults)]
    # return args
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
        l.sort(key=lambda el: re.search(r"(?<=num )[0-9]+", el.name)[0])
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
        params["prev_data_dir"] = str(p.parent / update_path_name(p.name))
    params["output_path"] = new_path
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
    with open(file_path, "w") as file:
        toml.dump(params, file, encoder=toml.TomlNumpyEncoder())

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
        if PARAM_FN in files:
            paths.append(Path(pwd))
    paths.sort(key=lambda el: el.parent.name)
    return [p for p in paths if p.parent.name == paths[-1].parent.name]
