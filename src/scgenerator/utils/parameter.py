import datetime
from copy import copy
from dataclasses import asdict, dataclass
from functools import lru_cache
from typing import Any, Callable, Dict, Iterable, List, Tuple, Union

import numpy as np

from ..const import __version__


@lru_cache
def type_checker(*types):
    def _type_checker_wrapper(validator, n=None):
        if isinstance(validator, str) and n is not None:
            _name = validator
            validator = lambda *args: None

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
    if not n is True and not n is False:
        raise ValueError(f"{name!r} must be True or False")


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
            raise ValueError(f"{name!r} must be a list or a tuple of 2 int")


def literal(*l):
    l = set(l)

    @type_checker(str)
    def _string(name, s):
        if not s in l:
            raise ValueError(f"{name!r} must be a str in {l}")

    return _string


def validator_list(validator):
    """returns a new validator that applies validator to each el of an iterable"""

    @type_checker(list, tuple)
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


class Parameter:
    def __init__(self, validator, converter=None, default=None):
        """Single parameter

        Parameters
        ----------
        tpe : type
            type of the paramter
        validators : Callable[[str, Any], None]
            signature : validator(name, value)
            must raise a ValueError when value doesn't fit the criteria checked by
            validator. name is passed to validator to be included in the error message
        converter : Callable, optional
            converts a valid value (for example, str.lower), by default None
        default : callable, optional
            factory function for a default value (for example, list), by default None
        """

        self.validator = validator
        self.converter = converter
        self.default = default

    def __set_name__(self, owner, name):
        self.name = name

    def __get__(self, instance, owner):
        if not instance:
            return self
        return instance.__dict__[self.name]

    def __delete__(self, instance):
        del instance.__dict__[self.name]

    def __set__(self, instance, value):
        if isinstance(value, Parameter):
            defaut = None if self.default is None else copy(self.default)
            instance.__dict__[self.name] = defaut
        else:
            if value is not None:
                self.validator(self.name, value)
                if self.converter is not None:
                    value = self.converter(value)
            instance.__dict__[self.name] = value


class VariableParameter:
    def __init__(self, parameterBase):
        self.pbase = parameterBase

    def __set_name__(self, owner, name):
        self.name = name

    def __get__(self, instance, owner):
        if not instance:
            return self
        return instance.__dict__[self.name]

    def __delete__(self, instance):
        del instance.__dict__[self.name]

    def __set__(self, instance, value: dict):
        if isinstance(value, VariableParameter):
            value = {}
        else:
            for k, v in value.items():
                if k not in valid_variable:
                    raise TypeError(f"{k!r} is not a valide variable parameter")
                if len(v) == 0:
                    raise ValueError(f"variable parameter {k!r} must not be empty")

                p = getattr(self.pbase, k)

                for el in v:
                    p.validator(k, el)
        instance.__dict__[self.name] = value


valid_variable = {
    "beta",
    "gamma",
    "pitch",
    "pitch_ratio",
    "core_radius",
    "capillary_num",
    "capillary_outer_d",
    "capillary_thickness",
    "capillary_spacing",
    "capillary_resonance_strengths",
    "capillary_nested",
    "he_mode",
    "fit_parameters",
    "input_transmission",
    "n2",
    "pressure",
    "temperature",
    "gas_name",
    "plasma_density" "peak_power",
    "mean_power",
    "peak_power",
    "energy",
    "quantum_noise",
    "shape",
    "wavelength",
    "intensity_noise",
    "width",
    "soliton_num",
    "behaviors",
    "raman_type",
    "tolerated_error",
    "step_size",
    "ideal_gas",
    "readjust_wavelength",
}

hc_model_specific_parameters = dict(
    marcatili=["core_radius", "he_mode"],
    marcatili_adjusted=["core_radius", "he_mode", "fit_parameters"],
    hasan=[
        "core_radius",
        "capillary_num",
        "capillary_thickness",
        "capillary_resonance_strengths",
        "capillary_nested",
        "capillary_spacing",
        "capillary_outer_d",
    ],
)
"""dependecy map only includes actual fiber parameters and exclude gas parameters"""


@dataclass
class BareParams:
    """
    This class defines each valid parameter's name, type and valid value but doesn't provide
    any method to act on those. For that, use initialize.Params
    """

    # root
    name: str = Parameter(string)
    prev_data_dir: str = Parameter(string)

    # # fiber
    input_transmission: float = Parameter(in_range_incl(0, 1))
    gamma: float = Parameter(non_negative(float, int))
    n2: float = Parameter(non_negative(float, int))
    effective_mode_diameter: float = Parameter(positive(float, int))
    A_eff: float = Parameter(non_negative(float, int))
    pitch: float = Parameter(in_range_excl(0, 1e-3))
    pitch_ratio: float = Parameter(in_range_excl(0, 1))
    core_radius: float = Parameter(in_range_excl(0, 1e-3))
    he_mode: Tuple[int, int] = Parameter(int_pair)
    fit_parameters: Tuple[int, int] = Parameter(int_pair)
    beta: Iterable[float] = Parameter(num_list)
    dispersion_file: str = Parameter(string)
    model: str = Parameter(literal("pcf", "marcatili", "marcatili_adjusted", "hasan", "custom"))
    length: float = Parameter(non_negative(float, int))
    capillary_num: int = Parameter(positive(int))
    capillary_outer_d: float = Parameter(in_range_excl(0, 1e-3))
    capillary_thickness: float = Parameter(in_range_excl(0, 1e-3))
    capillary_spacing: float = Parameter(in_range_excl(0, 1e-3))
    capillary_resonance_strengths: Iterable[float] = Parameter(num_list)
    capillary_nested: int = Parameter(non_negative(int))

    # gas
    gas_name: str = Parameter(literal("vacuum", "helium", "air"), converter=str.lower)
    pressure: Union[float, Iterable[float]] = Parameter(
        validator_or(non_negative(float, int), num_list)
    )
    temperature: float = Parameter(positive(float, int))
    plasma_density: float = Parameter(non_negative(float, int))

    # pulse
    field_file: str = Parameter(string)
    repetition_rate: float = Parameter(non_negative(float, int))
    peak_power: float = Parameter(positive(float, int))
    mean_power: float = Parameter(positive(float, int))
    energy: float = Parameter(positive(float, int))
    soliton_num: float = Parameter(positive(float, int))
    quantum_noise: bool = Parameter(boolean)
    shape: str = Parameter(literal("gaussian", "sech"))
    wavelength: float = Parameter(in_range_incl(100e-9, 3000e-9))
    intensity_noise: float = Parameter(in_range_incl(0, 1))
    width: float = Parameter(in_range_excl(0, 1e-9))
    t0: float = Parameter(in_range_excl(0, 1e-9))

    # simulation
    behaviors: str = Parameter(validator_list(literal("spm", "raman", "ss")))
    parallel: bool = Parameter(boolean)
    raman_type: str = Parameter(literal("measured", "agrawal", "stolen"), converter=str.lower)
    ideal_gas: bool = Parameter(boolean)
    repeat: int = Parameter(positive(int))
    t_num: int = Parameter(positive(int))
    z_num: int = Parameter(positive(int))
    time_window: float = Parameter(positive(float, int))
    dt: float = Parameter(in_range_excl(0, 5e-15))
    tolerated_error: float = Parameter(in_range_excl(1e-15, 1e-5))
    step_size: float = Parameter(positive(float, int))
    lower_wavelength_interp_limit: float = Parameter(in_range_incl(100e-9, 3000e-9))
    upper_wavelength_interp_limit: float = Parameter(in_range_incl(200e-9, 5000e-9))
    frep: float = Parameter(positive(float, int))
    prev_sim_dir: str = Parameter(string)
    readjust_wavelength: bool = Parameter(boolean)
    recovery_last_stored: int = Parameter(non_negative(int))

    # computed
    field_0: np.ndarray = Parameter(type_checker(np.ndarray))
    spec_0: np.ndarray = Parameter(type_checker(np.ndarray))
    w: np.ndarray = Parameter(type_checker(np.ndarray))
    w_c: np.ndarray = Parameter(type_checker(np.ndarray))
    t: np.ndarray = Parameter(type_checker(np.ndarray))
    L_D: float = Parameter(non_negative(float, int))
    L_NL: float = Parameter(non_negative(float, int))
    L_sol: float = Parameter(non_negative(float, int))
    dynamic_dispersion: bool = Parameter(boolean)
    adapt_step_size: bool = Parameter(boolean)
    error_ok: float = Parameter(positive(float))
    hr_w: np.ndarray = Parameter(type_checker(np.ndarray))
    z_targets: np.ndarray = Parameter(type_checker(np.ndarray))
    const_qty: np.ndarray = Parameter(type_checker(np.ndarray))
    beta_func: Callable[[float], List[float]] = Parameter(func_validator)
    gamma_func: Callable[[float], float] = Parameter(func_validator)

    def prepare_for_dump(self) -> Dict[str, Any]:
        param = asdict(self)
        param = BareParams.strip_params_dict(param)
        param["datetime"] = datetime.datetime.now()
        param["version"] = __version__
        return param

    @staticmethod
    def strip_params_dict(dico: Dict[str, Any]) -> Dict[str, Any]:
        """prepares a dictionary for serialization. Some keys may not be preserved
        (dropped because they take a lot of space and can be exactly reconstructed)

        Parameters
        ----------
        dico : dict
            dictionary
        """
        forbiden_keys = ["w_c", "w_power_fact", "field_0", "spec_0", "w", "t", "z_targets"]
        types = (np.ndarray, float, int, str, list, tuple, dict)
        out = {}
        for key, value in dico.items():
            if key in forbiden_keys:
                continue
            if not isinstance(value, types):
                continue
            if isinstance(value, dict):
                out[key] = BareParams.strip_params_dict(value)
            elif isinstance(value, np.ndarray) and value.dtype == complex:
                continue
            else:
                out[key] = value

        if "variable" in out and len(out["variable"]) == 0:
            del out["variable"]

        return out


@dataclass
class BareConfig(BareParams):
    variable: dict = VariableParameter(BareParams)


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