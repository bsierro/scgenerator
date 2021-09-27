import inspect
import re
from functools import cache
from string import printable as str_printable
from typing import Callable

import numpy as np
from pydantic import BaseModel

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
