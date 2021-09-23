from pydantic import BaseModel, validator
from typing import Union, Iterable, Generator, Any
from collections.abc import Sequence, MutableMapping
import itertools
from ..const import PARAM_SEPARATOR
from . import utils
import numpy as np
from pathlib import Path


def format_value(name: str, value) -> str:
    if value is True or value is False:
        return str(value)
    elif isinstance(value, (float, int)):
        try:
            return getattr(Parameters, name).display(value)
        except AttributeError:
            return format(value, ".9g")
    elif isinstance(value, (list, tuple, np.ndarray)):
        return "-".join([str(v) for v in value])
    elif isinstance(value, str):
        p = Path(value)
        if p.exists():
            return p.stem
    return str(value)


def pretty_format_value(name: str, value) -> str:
    try:
        return getattr(Parameters, name).display(value)
    except AttributeError:
        return name + PARAM_SEPARATOR + str(value)


class HashableBaseModel(BaseModel):
    """Pydantic BaseModel that's immutable and can be hashed"""

    def __hash__(self) -> int:
        return hash(type(self)) + sum(hash(v) for v in self.__dict__.values())

    class Config:
        allow_mutation = False


class VariationSpecsError(ValueError):
    pass


class Variationer:
    """
    manages possible combinations of values given dicts of lists

    Example
    -------
    `var = Variationer([dict(a=[1, 2]), [dict(b=["000", "111"], c=["a", "-1"])]])`

    """

    all_indices: list[list[int]]
    all_dicts: list[list[dict[str, list]]]

    def __init__(self, variables: Iterable[Union[list[MutableMapping], MutableMapping]]):
        self.all_indices = []
        self.all_dicts = []
        for i, el in enumerate(variables):
            if not isinstance(el, Sequence):
                el = [{k: v} for k, v in el.items()]
            else:
                el = list(el)
            self.append(el)

    def append(self, var_list: list[dict[str, list]]):
        num_vars = []
        for d in var_list:
            values = list(d.values())
            len_to_test = len(values[0])
            if not all(len(v) == len_to_test for v in values[1:]):
                raise VariationSpecsError(
                    f"variable items should all have the same number of parameters"
                )
            num_vars.append(len_to_test)
        if len(num_vars) == 0:
            num_vars = [1]
        self.all_indices.append(num_vars)
        self.all_dicts.append(var_list)

    def iterate(self, index: int = -1) -> Generator["SimulationDescriptor", None, None]:
        if index < 0:
            index = len(self.all_indices) + index + 1
        flattened_indices = sum(self.all_indices[:index], [])
        index_positions = np.cumsum([0] + [len(i) for i in self.all_indices[:index]])
        ranges = [range(i) for i in flattened_indices]
        for r in itertools.product(*ranges):
            out: list[list[tuple[str, Any]]] = []
            for i, (start, end) in enumerate(zip(index_positions[:-1], index_positions[1:])):
                out.append([])
                for value_index, var_d in zip(r[start:end], self.all_dicts[i]):
                    for k, v in var_d.items():
                        out[-1].append((k, v[value_index]))
            yield SimulationDescriptor(raw_descr=out)


class SimulationDescriptor(HashableBaseModel):
    raw_descr: tuple[tuple[tuple[str, Any], ...], ...]
    separator: str = "fiber"

    def __str__(self) -> str:
        return self.descriptor(add_identifier=False)

    def descriptor(self, add_identifier=False) -> str:
        """formats a variable list into a str such that each simulation has a unique
        directory name. A u_XXX unique identifier and b_XXX (ignoring repeat simulations)
        branch identifier can added at the beginning.

        Parameters
        ----------
        add_identifier : bool
            add unique simulation and parameter-set identifiers

        Returns
        -------
        str
            simulation descriptor
        """
        str_list = []

        for p_name, p_value in self.flat:
            ps = p_name.replace("/", "").replace("\\", "").replace(PARAM_SEPARATOR, "")
            vs = format_value(p_name, p_value).replace("/", "").replace(PARAM_SEPARATOR, "")
            str_list.append(ps + PARAM_SEPARATOR + vs)
        tmp_name = PARAM_SEPARATOR.join(str_list)
        if not add_identifier:
            return tmp_name
        return (
            self.identifier + PARAM_SEPARATOR + self.branch.identifier + PARAM_SEPARATOR + tmp_name
        )

    @property
    def flat(self) -> list[tuple[str, Any]]:
        out = []
        for n, variable_list in enumerate(self.raw_descr):
            out += [
                (self.separator, "ABCDEFGHIJKLMNOPQRSTUVWXYZ"[n % 26] * (n // 26 + 1)),
                *variable_list,
            ]
        return out

    @property
    def branch(self) -> "BranchDescriptor":
        return SimulationDescriptor(raw_descr=self.raw_descr, separator=self.separator)

    @property
    def identifier(self) -> str:
        return "u_" + utils.to_62(hash(str(self.flat)))


class BranchDescriptor(SimulationDescriptor):
    @property
    def identifier(self) -> str:
        return "b_" + utils.to_62(hash(str(self.flat)))

    @validator("raw_descr")
    def validate_raw_descr(cls, v):
        return tuple(tuple(el for el in variable if el[0] != "num") for variable in v)
