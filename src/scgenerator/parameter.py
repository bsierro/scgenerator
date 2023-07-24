from __future__ import annotations

import datetime as datetime_module
import os
from copy import copy
from dataclasses import dataclass, field, fields
from functools import lru_cache, wraps
from math import isnan
from pathlib import Path
from typing import (Any, Callable, ClassVar, Iterable, Iterator, Set, Type,
                    TypeVar)

import numpy as np

from scgenerator import utils
from scgenerator.const import MANDATORY_PARAMETERS, __version__
from scgenerator.errors import EvaluatorError
from scgenerator.evaluator import Evaluator
from scgenerator.operators import Qualifier, SpecOperator
from scgenerator.utils import update_path_name

T = TypeVar("T")
DISPLAY_INFO = {}


def format_value(name: str, value) -> str:
    if value is True or value is False:
        return str(value)
    elif isinstance(value, (float, int)):
        try:
            return DISPLAY_INFO[name](value)
        except KeyError:
            return format(value, ".9g")
    elif isinstance(value, np.ndarray):
        return np.array2string(value)
    elif isinstance(value, (list, tuple)):
        return "-".join([str(v) for v in value])
    elif isinstance(value, str):
        p = Path(value)
        if p.exists():
            return p.stem
    elif callable(value):
        return getattr(value, "__name__", repr(value))
    return str(value)


# Validator
@lru_cache
def type_checker(*types):
    def _type_checker_wrapper(validator, n=None):
        if isinstance(validator, str) and n is not None:
            _name = validator

            def validator(*args):
                pass

        @wraps(validator)
        def _type_checker_wrapped(name, n):
            if not isinstance(n, types):
                raise TypeError(
                    f"{name!r} value must be of type {' or '.join(format(t) for t in types)} "
                    f"instead of {type(n)}"
                )
            validator(name, n)

        if n is None:
            return _type_checker_wrapped
        else:
            _type_checker_wrapped(_name, n)

    return _type_checker_wrapper


@type_checker(str)
def string(name, n):
    if len(n) == 0:
        raise ValueError(f"{name!r} must not be empty")


def in_range_excl(_min, _max):
    @type_checker(float, int)
    def _in_range(name, n):
        if n <= _min or n >= _max:
            raise ValueError(f"{name!r} must be between {_min} and {_max} (exclusive)")

    return _in_range


def in_range_incl(_min, _max):
    @type_checker(float, int)
    def _in_range(name, n):
        if n < _min or n > _max:
            raise ValueError(f"{name!r} must be between {_min} and {_max} (inclusive)")

    return _in_range


def boolean(name, n):
    if n is not True and n is not False:
        raise ValueError(f"{name!r} must be True or False")


def is_function(name, n):
    if not callable(n):
        raise TypeError(f"{name!r} must be callable")


@lru_cache
def non_negative(*types):
    @type_checker(*types)
    def _non_negative(name, n):
        if n < 0:
            raise ValueError(f"{name!r} must be non negative")

    return _non_negative


@lru_cache
def positive(*types):
    @type_checker(*types)
    def _positive(name, n):
        if n <= 0:
            raise ValueError(f"{name!r} must be positive")

    return _positive


@type_checker(tuple, list)
def int_pair(name, t):
    invalid = len(t) != 2
    for m in t:
        if invalid or not isinstance(m, int):
            raise ValueError(f"{name!r} must be a list or a tuple of 2 int. got {t!r} instead")


@type_checker(tuple, list)
def float_pair(name, t):
    invalid = len(t) != 2
    for m in t:
        if invalid or not isinstance(m, (int, float)):
            raise ValueError(f"{name!r} must be a list or a tuple of 2 numbers. got {t!r} instead")


def literal(*l):
    l = set(l)

    @type_checker(str)
    def _string(name, s):
        if s not in l:
            raise ValueError(f"{name!r} must be a str in {l}")

    return _string


def validator_list(validator):
    """returns a new validator that applies validator to each el of an iterable"""

    @type_checker(list, tuple, np.ndarray)
    def _list_validator(name, l):
        for i, el in enumerate(l):
            validator(name + f"[{i}]", el)

    return _list_validator


def validator_or(*validators):
    """combines many validators and raises an exception only if all of them raise an exception"""

    n = len(validators)

    def _or_validator(name, value):
        errors = []
        for validator in validators:
            try:
                validator(name, value)
            except (ValueError, TypeError) as e:
                errors.append(e)
        errors.sort(key=lambda el: isinstance(el, ValueError))
        if len(errors) == n:
            raise errors[-1]

    return _or_validator


def validator_and(*validators):
    def _and_validator(name, n):
        for v in validators:
            v(name, n)

    return _and_validator


@type_checker(list, tuple, np.ndarray)
def num_list(name, l):
    for i, el in enumerate(l):
        type_checker(int, float)(name + f"[{i}]", el)


def func_validator(name, n):
    if not callable(n):
        raise TypeError(f"{name!r} must be callable")


# classes


class Parameter:
    def __init__(
        self,
        validator: Callable[[str, Any], None],
        converter: Callable = None,
        default=None,
        display_info: tuple[float, str] = None,
    ):
        """Single parameter

        Parameters
        ----------
        validator : Callable[[str, Any], None]
            signature : validator(name, value)
            must raise a ValueError when value doesn't fit the criteria checked by
            validator. name is passed to validator to be included in the error message
        converter : Callable, optional
            converts a valid value (for example, str.lower), by default None
        default : callable, optional
            factory function for a default value (for example, list), by default None
        display_info : tuple[float, str], optional
            a factor by which to multiply the value and a string to be appended as a suffix
            when displaying the value
            example : (1e-6, "MW") will mean the value 1.12e6 is displayed as '1.12MW'
        """
        self.__validator = validator
        self.converter = converter
        self.default = default
        self.display_info = display_info

    def __set_name__(self, owner: Type[Parameters], name):
        self.name = name
        try:
            owner._p_names.add(self.name)
        except AttributeError:
            pass
        if self.default is not None:
            Evaluator.register_default_param(self.name, self.default)
        DISPLAY_INFO[self.name] = self.display

    def __get__(self, instance: Parameters, owner):
        if instance is None:
            return self
        if self.name not in instance._param_dico:
            try:
                instance._evaluator.compute(self.name)
            except EvaluatorError:
                pass
        return instance._param_dico.get(self.name)
        # return instance.__dict__[self.name]

    def __delete__(self, instance):
        raise AttributeError("Cannot delete parameter")

    def __set__(self, instance: Parameters, value):
        if isinstance(value, Parameter):
            if self.default is not None:
                instance._param_dico[self.name] = copy(self.default)
        else:
            is_value, value = self.validate(value)
            if is_value:
                instance._param_dico[self.name] = value
            else:
                if self.name in instance._param_dico:
                    del instance._param_dico[self.name]

    def display(self, num: float) -> str:
        if self.display_info is None:
            return str(num)
        else:
            fac, unit = self.display_info
            num_str = format(num * fac, ".2f")
            if num_str.endswith(".00"):
                num_str = num_str[:-3]
            return f"{num_str} {unit}"

    def validate(self: Parameter, v) -> tuple[bool, Any]:
        if v is None:
            is_value = False
        else:
            try:
                is_value = not isnan(v)
            except TypeError:
                is_value = True
        if is_value:
            if self.converter is not None:
                v = self.converter(v)
            self.__validator(self.name, v)
        return is_value, v

    def validator(self, name, value):
        self.validate(value)


@dataclass(repr=False)
class Parameters:
    """
    This class defines each valid parameter's name, type and valid value.
    """

    # internal machinery
    _param_dico: dict[str, Any] = field(init=False, default_factory=dict, repr=False)
    _evaluator: Evaluator = field(init=False, repr=False)
    _p_names: ClassVar[Set[str]] = set()

    # root
    name: str = Parameter(string, default="no name")
    prev_data_dir: str = Parameter(string)
    recovery_data_dir: str = Parameter(string)
    output_path: Path = Parameter(type_checker(Path), default=Path("sc_data"), converter=Path)

    # fiber
    input_transmission: float = Parameter(in_range_incl(0, 1), default=1.0)
    gamma: float = Parameter(non_negative(float, int))
    n2: float = Parameter(non_negative(float, int))
    chi3: float = Parameter(non_negative(float, int))
    loss: str = Parameter(literal("capillary"))
    loss_file: str = Parameter(string)
    effective_mode_diameter: float = Parameter(positive(float, int))
    A_eff: float = Parameter(non_negative(float, int))
    A_eff_file: str = Parameter(string)
    numerical_aperture: float = Parameter(in_range_excl(0, 1))
    pitch: float = Parameter(in_range_excl(0, 1e-3), display_info=(1e6, "μm"))
    pitch_ratio: float = Parameter(in_range_excl(0, 1))
    core_radius: float = Parameter(in_range_excl(0, 1e-3), display_info=(1e6, "μm"))
    he_mode: tuple[int, int] = Parameter(int_pair, default=(1, 1))
    fit_parameters: tuple[int, int] = Parameter(float_pair, default=(0.08, 200e-9))
    beta2_coefficients: Iterable[float] = Parameter(num_list)
    dispersion_file: str = Parameter(string)
    model: str = Parameter(
        literal("pcf", "marcatili", "marcatili_adjusted", "hasan", "custom"),
    )
    zero_dispersion_wavelength: float = Parameter(
        in_range_incl(100e-9, 5000e-9), display_info=(1e9, "nm")
    )
    length: float = Parameter(non_negative(float, int), display_info=(1e2, "cm"))
    capillary_num: int = Parameter(positive(int))
    capillary_radius: float = Parameter(in_range_excl(0, 1e-3), display_info=(1e6, "μm"))
    capillary_thickness: float = Parameter(in_range_excl(0, 1e-3), display_info=(1e6, "μm"))
    capillary_spacing: float = Parameter(in_range_excl(0, 1e-3), display_info=(1e6, "μm"))
    capillary_resonance_strengths: Iterable[float] = Parameter(
        validator_list(type_checker(int, float, np.ndarray))
    )
    capillary_resonance_max_order: int = Parameter(non_negative(int), default=0)
    capillary_nested: int = Parameter(non_negative(int), default=0)

    # gas
    gas_name: str = Parameter(string, converter=str.lower, default="vacuum")
    pressure: float = Parameter(non_negative(float, int), display_info=(1e-5, "bar"))
    pressure_in: float = Parameter(non_negative(float, int), display_info=(1e-5, "bar"))
    pressure_out: float = Parameter(non_negative(float, int), display_info=(1e-5, "bar"))
    temperature: float = Parameter(positive(float, int), display_info=(1, "K"), default=300)
    plasma_density: float = Parameter(non_negative(float, int), default=0)

    # pulse
    field_file: str = Parameter(string)
    input_time: np.ndarray = Parameter(type_checker(np.ndarray))
    input_field: np.ndarray = Parameter(type_checker(np.ndarray))
    repetition_rate: float = Parameter(
        non_negative(float, int), display_info=(1e-3, "kHz"), default=40e6
    )
    peak_power: float = Parameter(positive(float, int), display_info=(1e-3, "kW"))
    mean_power: float = Parameter(positive(float, int), display_info=(1e3, "mW"))
    energy: float = Parameter(positive(float, int), display_info=(1e6, "μJ"))
    soliton_num: float = Parameter(non_negative(float, int))
    additional_noise_factor: float = Parameter(positive(float, int), default=1)
    shape: str = Parameter(literal("gaussian", "sech"), default="gaussian")
    wavelength: float = Parameter(in_range_incl(100e-9, 10000e-9), display_info=(1e9, "nm"))
    intensity_noise: float = Parameter(in_range_incl(0, 1), display_info=(1e2, "%"), default=0)
    noise_correlation: float = Parameter(in_range_incl(-10, 10), default=0)
    width: float = Parameter(in_range_excl(0, 1e-9), display_info=(1e15, "fs"))
    t0: float = Parameter(in_range_excl(0, 1e-9), display_info=(1e15, "fs"))

    # Behaviors to include
    quantum_noise: bool = Parameter(boolean, default=False)
    self_steepening: bool = Parameter(boolean, default=True)
    ideal_gas: bool = Parameter(boolean, default=False)
    photoionization: bool = Parameter(boolean, default=False)

    # simulation
    full_field: bool = Parameter(boolean, default=False)
    integration_scheme: str = Parameter(
        literal("erk43", "erk54", "cqe", "sd", "constant"),
        converter=str.lower,
        default="erk43",
    )
    raman_type: str = Parameter(literal("measured", "agrawal", "stolen"), converter=str.lower)
    raman_fraction: float = Parameter(non_negative(float, int))
    spm: bool = Parameter(boolean, default=True)
    repeat: int = Parameter(positive(int), default=1)
    t_num: int = Parameter(positive(int), default=8192)
    z_num: int = Parameter(positive(int), default=128)
    time_window: float = Parameter(positive(float, int))
    dt: float = Parameter(in_range_excl(0, 10e-15))
    tolerated_error: float = Parameter(in_range_excl(1e-15, 1e-3), default=1e-11)
    step_size: float = Parameter(non_negative(float, int), default=0)
    wavelength_window: tuple[float, float] = Parameter(
        validator_and(float_pair, validator_list(in_range_incl(100e-9, 10000e-9)))
    )
    interpolation_degree: int = Parameter(validator_and(type_checker(int), in_range_incl(2, 18)))
    prev_sim_dir: str = Parameter(string)
    recovery_last_stored: int = Parameter(non_negative(int), default=0)
    parallel: bool = Parameter(boolean, default=True)
    worker_num: int = Parameter(positive(int))

    # computed
    linear_operator: SpecOperator = Parameter(is_function)
    nonlinear_operator: SpecOperator = Parameter(is_function)
    conserved_quantity: Qualifier = Parameter(is_function)
    fft: Callable[[np.ndarray], np.ndarray] = Parameter(is_function)
    ifft: Callable[[np.ndarray], np.ndarray] = Parameter(is_function)
    field_0: np.ndarray = Parameter(type_checker(np.ndarray))
    spec_0: np.ndarray = Parameter(type_checker(np.ndarray))
    beta2: float = Parameter(type_checker(int, float))
    alpha_arr: np.ndarray = Parameter(type_checker(np.ndarray))
    alpha: float = Parameter(non_negative(float, int))
    gamma_arr: np.ndarray = Parameter(type_checker(np.ndarray))
    A_eff_arr: np.ndarray = Parameter(type_checker(np.ndarray))
    spectrum_factor: float = Parameter(type_checker(float))
    c_to_a_factor: np.ndarray = Parameter(type_checker(float, int, np.ndarray))
    w: np.ndarray = Parameter(type_checker(np.ndarray))
    l: np.ndarray = Parameter(type_checker(np.ndarray))
    w_c: np.ndarray = Parameter(type_checker(np.ndarray))
    w0: float = Parameter(positive(float))
    t: np.ndarray = Parameter(type_checker(np.ndarray))
    L_D: float = Parameter(non_negative(float, int))
    L_NL: float = Parameter(non_negative(float, int))
    L_sol: float = Parameter(non_negative(float, int))
    adapt_step_size: bool = Parameter(boolean)
    hr_w: np.ndarray = Parameter(type_checker(np.ndarray))
    z_targets: np.ndarray = Parameter(type_checker(np.ndarray))
    const_qty: np.ndarray = Parameter(type_checker(np.ndarray))

    num: int = Parameter(non_negative(int))
    datetime: datetime_module.datetime = Parameter(type_checker(datetime_module.datetime))
    version: str = Parameter(string)

    def __post_init__(self):
        self._evaluator = Evaluator.default(self.full_field)
        self._evaluator.set(self._param_dico)

    def __repr__(self) -> str:
        return "Parameter(" + ", ".join(self.__repr_list__()) + ")"

    def __pformat__(self) -> str:
        return "\n".join(["Parameter(", *list(self.__repr_list__()), ")"])

    def __repr_list__(self) -> Iterator[str]:
        yield from (f"{k}={v}" for k, v in self.dump_dict().items())

    def __getstate__(self) -> dict[str, Any]:
        return self.dump_dict(add_metadata=False)

    def __setstate__(self, dumped_dict: dict[str, Any]):
        self._param_dico = dict()
        for k, v in dumped_dict.items():
            setattr(self, k, v)
        self.__post_init__()

    def dump_dict(self, compute=True, add_metadata=True) -> dict[str, Any]:
        if compute:
            self.compute_in_place()
        param = Parameters.strip_params_dict(self._param_dico)
        if add_metadata:
            param["datetime"] = datetime_module.datetime.now()
            param["version"] = __version__
        return param

    def compute_in_place(self, *to_compute: str):
        if len(to_compute) == 0:
            to_compute = MANDATORY_PARAMETERS
        for k in to_compute:
            getattr(self, k)

    def compute(self, key: str) -> Any:
        return self._evaluator.compute(key)

    def pretty_str(self, params: Iterable[str] = None, exclude=None) -> str:
        """return a pretty formatted string describing the parameters"""
        params = params or self.dump_dict().keys()
        exclude = exclude or []
        if isinstance(exclude, str):
            exclude = [exclude]
        p_pairs = [(k, format_value(k, getattr(self, k))) for k in params if k not in exclude]
        max_left = max(len(el[0]) for el in p_pairs)
        max_right = max(len(el[1]) for el in p_pairs)
        return "\n".join("{:>{l}} = {:{r}}".format(*p, l=max_left, r=max_right) for p in p_pairs)

    @classmethod
    def all_parameters(cls) -> list[str]:
        return [f.name for f in fields(cls)]

    @classmethod
    def load(cls, path: os.PathLike) -> "Parameters":
        return cls(**utils.load_toml(path))

    @classmethod
    def strip_params_dict(cls, dico: dict[str, Any]) -> dict[str, Any]:
        """prepares a dictionary for serialization. Some keys may not be preserved
        (dropped because they take a lot of space and can be exactly reconstructed)

        Parameters
        ----------
        dico : dict
            dictionary
        """
        forbiden_keys = {
            "_param_dico",
            "_evaluator",
            "w_c",
            "w_power_fact",
            "field_0",
            "spec_0",
            "w",
            "t",
            "z_targets",
            "l",
            "wl_for_disp",
            "alpha",
            "gamma_arr",
            "A_eff_arr",
            "nonlinear_op",
            "linear_op",
        }
        types = (np.ndarray, float, int, str, list, tuple, dict, Path)
        out = {}
        for key, value in dico.items():
            if key in forbiden_keys or key not in cls._p_names:
                continue
            if not isinstance(value, types):
                continue
            if isinstance(value, dict):
                out[key] = Parameters.strip_params_dict(value)
            elif isinstance(value, Path):
                out[key] = str(value)
            elif isinstance(value, np.ndarray) and value.dtype == complex:
                continue
            elif isinstance(value, np.ndarray):
                out[key] = value.tolist()
            else:
                out[key] = value

        if "variable" in out and len(out["variable"]) == 0:
            del out["variable"]

        return out

    @property
    def final_path(self) -> Path:
        if self.output_path is not None:
            return self.output_path.parent / update_path_name(self.output_path.name)
        return None


if __name__ == "__main__":
    numero = type_checker(int)

    @numero
    def natural_number(name, n):
        if n < 0:
            raise ValueError(f"{name!r} must be positive")

    try:
        numero("a", np.arange(45))
    except Exception as e:
        print(e)
    try:
        natural_number("b", -1)
    except Exception as e:
        print(e)
    try:
        natural_number("c", 1.0)
    except Exception as e:
        print(e)
    try:
        natural_number("d", 1)
        print("success !")
    except Exception as e:
        print(e)
