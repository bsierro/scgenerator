"""
This files includes utility functions designed more or less to be used specifically with the
scgenerator module but some function may be used in any python program

"""

from __future__ import annotations

import itertools
import multiprocessing
import os
import random
import re
import shutil
import threading
from collections import abc
from io import StringIO
from pathlib import Path
from string import printable as str_printable
from functools import cache
from typing import Any, Callable, Generator, Iterable, MutableMapping, Sequence, TypeVar, Union


import numpy as np
import pkg_resources as pkg
import toml
from tqdm import tqdm

from .pbar import PBars
from ..const import PARAM_FN, PARAM_SEPARATOR, SPEC1_FN, SPECN_FN1, Z_FN, __version__
from ..env import pbar_policy
from ..logger import get_logger

T_ = TypeVar("T_")

PathTree = list[tuple[Path, ...]]


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


def load_previous_spectrum(prev_data_dir: str) -> np.ndarray:
    prev_data_dir = Path(prev_data_dir)
    num = find_last_spectrum_num(prev_data_dir)
    return load_spectrum(prev_data_dir / SPEC1_FN.format(num))


@cache
def load_spectrum(file: os.PathLike) -> np.ndarray:
    return np.load(file)


def conform_toml_path(path: os.PathLike) -> str:
    path: str = str(path)
    if not path.lower().endswith(".toml"):
        path = path + ".toml"
    return path


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

    dico.setdefault("variable", {})
    for key in {"simulation", "fiber", "gas", "pulse"} & dico.keys():
        section = dico.pop(key)
        dico["variable"].update(section.pop("variable", {}))
        dico.update(section)
    if len(dico["variable"]) == 0:
        dico.pop("variable")
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
        params.setdefault("variable", {})
        configs.append(loaded_config | params)
    configs[0]["variable"] = loaded_config.get("variable", {}) | configs[0]["variable"]
    configs[0]["variable"]["num"] = list(range(configs[0].get("repeat", 1)))

    return Path(final_path), configs


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
