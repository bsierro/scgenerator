from math import prod
import itertools
from collections.abc import MutableMapping, Sequence
from pathlib import Path
from typing import Any, Callable, Generator, Iterable, Union

import numpy as np
from pydantic import validator

from ..const import PARAM_SEPARATOR
from . import utils


class VariationSpecsError(ValueError):
    pass


class Variationer:
    """
    manages possible combinations of values given dicts of lists

    Example
    -------
    `>> var = Variationer([dict(a=[1, 2]), [dict(b=["000", "111"], c=["a", "-1"])]])
    list(v.raw_descr for v in var.iterate())

    [
        ((("a", 1),), (("b", "000"), ("c", "a"))),
        ((("a", 1),), (("b", "111"), ("c", "-1"))),
        ((("a", 2),), (("b", "000"), ("c", "a"))),
        ((("a", 2),), (("b", "111"), ("c", "-1"))),
    ]`

    """

    all_indices: list[list[int]]
    all_dicts: list[list[dict[str, list]]]

    def __init__(self, variables: Iterable[Union[list[MutableMapping], MutableMapping]] = None):
        self.all_indices = []
        self.all_dicts = []
        if variables is not None:
            for i, el in enumerate(variables):
                self.append(el)

    def append(self, var_list: Union[list[MutableMapping], MutableMapping]):
        """append a list of variable parameter sets
        each call to append creates a new group of parameters

        Parameters
        ----------
        var_list : Union[list[MutableMapping], MutableMapping]
            each dict in the list is treated as an independent parameter
            this means that if for one dict, len > 1, the lists of possible values
            must be the same length

        Example
        -------
        `>> append([dict(wavelength=[800e-9, 900e-9], power=[1e3, 2e3]), dict(length=[3e-2, 3.5e-2, 4e-2])])`

        means that for every parameter variations, wavelength=800e-9 will always occur when power=1e3 and
        vice versa, while length is free to vary independently

        Raises
        ------
        VariationSpecsError
            raised when possible values lists in a same dict are not the same length
        """
        if not isinstance(var_list, Sequence):
            var_list = [{k: v} for k, v in var_list.items()]
        else:
            var_list = list(var_list)
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

    def iterate(self, index: int = -1) -> Generator["VariationDescriptor", None, None]:
        index = self.__index(index)
        flattened_indices = sum(self.all_indices[: index + 1], [])
        index_positions = np.cumsum([0] + [len(i) for i in self.all_indices[: index + 1]])
        ranges = [range(i) for i in flattened_indices]
        for r in itertools.product(*ranges):
            out: list[list[tuple[str, Any]]] = []
            indicies: list[list[int]] = []
            for i, (start, end) in enumerate(zip(index_positions[:-1], index_positions[1:])):
                out.append([])
                indicies.append([])
                for value_index, var_d in zip(r[start:end], self.all_dicts[i]):
                    for k, v in var_d.items():
                        out[-1].append((k, v[value_index]))
                        indicies[-1].append(value_index)
            yield VariationDescriptor(raw_descr=out, index=indicies)

    def __index(self, index: int) -> int:
        if index < 0:
            index = len(self.all_indices) + index
        return index

    def var_num(self, index: int = -1) -> int:
        index = self.__index(index)
        return max(1, prod(prod(el) for el in self.all_indices[: index + 1]))


class VariationDescriptor(utils.HashableBaseModel):
    raw_descr: tuple[tuple[tuple[str, Any], ...], ...]
    index: tuple[tuple[int, ...], ...]
    separator: str = "fiber"
    _format_registry: dict[str, Callable[..., str]] = {}
    __ids: dict[int, int] = {}

    def __str__(self) -> str:
        return self.formatted_descriptor(add_identifier=False)

    def formatted_descriptor(self, add_identifier=False) -> str:
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
            vs = self.format_value(p_name, p_value).replace("/", "").replace(PARAM_SEPARATOR, "")
            str_list.append(ps + PARAM_SEPARATOR + vs)
        tmp_name = PARAM_SEPARATOR.join(str_list)
        if not add_identifier:
            return tmp_name
        return (
            self.identifier + PARAM_SEPARATOR + self.branch.identifier + PARAM_SEPARATOR + tmp_name
        )

    @classmethod
    def register_formatter(cls, p_name: str, func: Callable[..., str]):
        cls._format_registry[p_name] = func

    def format_value(self, name: str, value) -> str:
        if value is True or value is False:
            return str(value)
        elif isinstance(value, (float, int)):
            try:
                return self._format_registry[name](value)
            except KeyError:
                return format(value, ".9g")
        elif isinstance(value, (list, tuple, np.ndarray)):
            return "-".join([str(v) for v in value])
        elif isinstance(value, str):
            p = Path(value)
            if p.exists():
                return p.stem
        return str(value)

    def __getitem__(self, key) -> "VariationDescriptor":
        return VariationDescriptor(
            raw_descr=self.raw_descr[key], index=self.index[key], separator=self.separator
        )

    def update_config(self, cfg: dict[str, Any]) -> dict[str, Any]:
        """updates a dictionary with the value of the descriptor

        Parameters
        ----------
        cfg : dict[str, Any]
            dict to be updated

        Returns
        -------
        dict[str, Any]
            same as cfg but with key from the descriptor added/updated.
        """
        return cfg | {k: v for k, v in self.raw_descr[-1]}

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
        descr = []
        ind = []
        for i, l in enumerate(self.raw_descr):
            descr.append([])
            ind.append([])
            for j, (k, v) in enumerate(l):
                if k != "num":
                    descr[-1].append((k, v))
                    ind[-1].append(self.index[i][j])
        return BranchDescriptor(raw_descr=descr, index=ind, separator=self.separator)

    @property
    def identifier(self) -> str:
        unique_id = hash(str(self.flat))
        self.__ids.setdefault(unique_id, len(self.__ids))
        return "u_" + str(self.__ids[unique_id])


class BranchDescriptor(VariationDescriptor):
    __ids: dict[int, int] = {}

    @property
    def identifier(self) -> str:
        branch_id = hash(str(self.flat))
        self.__ids.setdefault(branch_id, len(self.__ids))
        return "b_" + str(self.__ids[branch_id])

    @validator("raw_descr")
    def validate_raw_descr(cls, v):
        return tuple(tuple(el for el in variable if el[0] != "num") for variable in v)
