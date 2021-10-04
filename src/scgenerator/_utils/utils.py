import inspect
import os
import re
from collections import defaultdict
from functools import cache
from pathlib import Path
from string import printable as str_printable
from typing import Any, Callable

import numpy as np
import toml
from pydantic import BaseModel

from .._utils import load_toml, save_toml
from ..const import PARAM_FN, PARAM_SEPARATOR, Z_FN
from ..physics.units import get_unit


class HashableBaseModel(BaseModel):
    """Pydantic BaseModel that's immutable and can be hashed"""

    def __hash__(self) -> int:
        return hash(type(self)) + sum(hash(v) for v in self.__dict__.values())

    class Config:
        allow_mutation = False


def to_62(i: int) -> str:
    arr = []
    if i == 0:
        return "0"
    i = abs(i)
    while i:
        i, value = divmod(i, 62)
        arr.append(str_printable[value])
    return "".join(reversed(arr))


class PlotRange(HashableBaseModel):
    left: float
    right: float
    unit: Callable[[float], float]
    conserved_quantity: bool = True

    def __init__(self, left, right, unit, **kwargs):
        super().__init__(left=left, right=right, unit=get_unit(unit), **kwargs)

    def __str__(self):
        return f"{self.left:.1f}-{self.right:.1f} {self.unit.__name__}"

    def sort_axis(self, axis: np.ndarray) -> tuple[np.ndarray, np.ndarray, tuple[float, float]]:
        return sort_axis(axis, self)

    def __iter__(self):
        yield self.left
        yield self.right
        yield self.unit.__name__


def sort_axis(
    axis: np.ndarray, plt_range: PlotRange
) -> tuple[np.ndarray, np.ndarray, tuple[float, float]]:
    """
    given an axis, returns this axis cropped according to the given range, converted and sorted

    Parameters
    ----------
    axis : 1D array containing the original axis (usual the w or t array)
    plt_range : tupple (min, max, conversion_function) used to crop the axis

    Returns
    -------
    cropped : the axis cropped, converted and sorted
    indices : indices to use to slice and sort other array in the same fashion
    extent : tupple with min and max of cropped

    Example
    -------
    w = np.append(np.linspace(0, -10, 20), np.linspace(0, 10, 20))
    t = np.linspace(-10, 10, 400)
    W, T = np.meshgrid(w, t)
    y = np.exp(-W**2 - T**2)

    # Define ranges
    rw = (-4, 4, s)
    rt = (-2, 6, s)

    w, cw = sort_axis(w, rw)
    t, ct = sort_axis(t, rt)

    # slice y according to the given ranges
    y = y[ct][:, cw]
    """
    if isinstance(plt_range, tuple):
        plt_range = PlotRange(*plt_range)
    r = np.array((plt_range.left, plt_range.right), dtype="float")

    indices = np.arange(len(axis))[
        (axis <= np.max(plt_range.unit(r))) & (axis >= np.min(plt_range.unit(r)))
    ]
    cropped = axis[indices]
    order = np.argsort(plt_range.unit.inv(cropped))
    indices = indices[order]
    cropped = cropped[order]
    out_ax = plt_range.unit.inv(cropped)

    return out_ax, indices, (out_ax[0], out_ax[-1])


def get_arg_names(func: Callable) -> list[str]:
    spec = inspect.getfullargspec(func)
    args = spec.args
    if spec.defaults is not None and len(spec.defaults) > 0:
        args = args[: -len(spec.defaults)]
    return args


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
        new_path = dest / update_path(pulses[0].name)
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
        params["prev_data_dir"] = str(p.parent / update_path(p.name))
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
    with open(file_path, "w") as file:
        toml.dump(params, file, encoder=toml.TomlNumpyEncoder())

    return file_path


def update_path(p: str) -> str:
    return re.sub(r"( ?num [0-9]+)|(u_[0-9]+ )", "", p)


def fiber_folder(i: int, sim_name: str, fiber_name: str) -> str:
    return PARAM_SEPARATOR.join([format(i), sim_name, fiber_name])
