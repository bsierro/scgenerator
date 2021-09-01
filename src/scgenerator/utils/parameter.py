import datetime as datetime_module
import enum
import inspect
import itertools
import os
import re
import time
from collections import defaultdict
from copy import deepcopy
from dataclasses import asdict, dataclass, fields
from functools import cache, lru_cache
from pathlib import Path
from typing import Any, Callable, Generator, Iterable, Literal, Optional, TypeVar, Union

import numpy as np

from .. import math, utils
from ..const import PARAM_SEPARATOR, __version__
from ..errors import EvaluatorError, NoDefaultError
from ..logger import get_logger
from ..physics import fiber, materials, pulse, units

T = TypeVar("T")

# Validator


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
        if not s in l:
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


# other


def translate(p_name: str, p_value: T) -> tuple[str, T]:
    """translates old parameters

    Parameters
    ----------
    p_name : str
        parameter name
    p_value : T
        parameter value

    Returns
    -------
    tuple[str, T]
        translated pair
    """
    old_names = dict(interp_degree="interpolation_degree")
    return old_names.get(p_name, p_name), p_value


# classes


class Parameter:
    def __init__(self, validator, converter=None, default=None, display_info=None, rules=None):
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
        self.display_info = display_info
        if rules is None:
            self.rules = []
        else:
            self.rules = rules

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
            # defaut = None if self.default is None else copy(self.default)
            # instance.__dict__[self.name] = defaut
            instance.__dict__[self.name] = None
        else:
            if value is not None:
                self.validator(self.name, value)
                if self.converter is not None:
                    value = self.converter(value)
            instance.__dict__[self.name] = value

    def display(self, num: float):
        if self.display_info is None:
            return str(num)
        else:
            fac, unit = self.display_info
            num_str = format(num * fac, ".2f")
            if num_str.endswith(".00"):
                num_str = num_str[:-3]
            return f"{num_str} {unit}"


valid_variable = {
    "dispersion_file",
    "prev_data_dir",
    "field_file",
    "loss_file",
    "A_eff_file",
    "beta2_coefficients",
    "gamma",
    "pitch",
    "pitch_ratio",
    "effective_mode_diameter",
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
    "plasma_density",
    "peak_power",
    "mean_power",
    "peak_power",
    "energy",
    "quantum_noise",
    "shape",
    "wavelength",
    "intensity_noise",
    "width",
    "t0",
    "soliton_num",
    "behaviors",
    "raman_type",
    "tolerated_error",
    "step_size",
    "interpolation_degree",
    "ideal_gas",
}

mandatory_parameters = [
    "name",
    "w_c",
    "w",
    "w0",
    "w_power_fact",
    "alpha",
    "spec_0",
    "field_0",
    "input_transmission",
    "z_targets",
    "length",
    "beta2_coefficients",
    "gamma_arr",
    "behaviors",
    "raman_type",
    "hr_w",
    "adapt_step_size",
    "tolerated_error",
    "dynamic_dispersion",
    "recovery_last_stored",
    "output_path",
]


@dataclass
class Parameters:
    """
    This class defines each valid parameter's name, type and valid value. Initializing
    such an obj will automatically compute all possible parameters
    """

    # root
    name: str = Parameter(string, default="no name")
    prev_data_dir: str = Parameter(string)
    recovery_data_dir: str = Parameter(string)
    previous_config_file: str = Parameter(string)
    output_path: str = Parameter(string, default="sc_data")

    # # fiber
    input_transmission: float = Parameter(in_range_incl(0, 1), default=1.0)
    gamma: float = Parameter(non_negative(float, int))
    n2: float = Parameter(non_negative(float, int), default=2.2e-20)
    loss: str = Parameter(literal("capillary"))
    loss_file: str = Parameter(string)
    effective_mode_diameter: float = Parameter(positive(float, int))
    A_eff: float = Parameter(non_negative(float, int))
    A_eff_file: str = Parameter(string)
    pitch: float = Parameter(in_range_excl(0, 1e-3))
    pitch_ratio: float = Parameter(in_range_excl(0, 1))
    core_radius: float = Parameter(in_range_excl(0, 1e-3))
    he_mode: tuple[int, int] = Parameter(int_pair, default=(1, 1))
    fit_parameters: tuple[int, int] = Parameter(int_pair, default=(0.08, 200e-9))
    beta2_coefficients: Iterable[float] = Parameter(num_list)
    dispersion_file: str = Parameter(string)
    model: str = Parameter(
        literal("pcf", "marcatili", "marcatili_adjusted", "hasan", "custom"), default="custom"
    )
    length: float = Parameter(non_negative(float, int))
    capillary_num: int = Parameter(positive(int))
    capillary_outer_d: float = Parameter(in_range_excl(0, 1e-3))
    capillary_thickness: float = Parameter(in_range_excl(0, 1e-3))
    capillary_spacing: float = Parameter(in_range_excl(0, 1e-3))
    capillary_resonance_strengths: Iterable[float] = Parameter(num_list, default=[])
    capillary_nested: int = Parameter(non_negative(int), default=0)

    # gas
    gas_name: str = Parameter(string, converter=str.lower, default="vacuum")
    pressure: Union[float, Iterable[float]] = Parameter(
        validator_or(non_negative(float, int), num_list), display_info=(1e-5, "bar"), default=1e5
    )
    temperature: float = Parameter(positive(float, int), display_info=(1, "K"), default=300)
    plasma_density: float = Parameter(non_negative(float, int), default=0)

    # pulse
    field_file: str = Parameter(string)
    repetition_rate: float = Parameter(
        non_negative(float, int), display_info=(1e-6, "MHz"), default=40e6
    )
    peak_power: float = Parameter(positive(float, int), display_info=(1e-3, "kW"))
    mean_power: float = Parameter(positive(float, int), display_info=(1e3, "mW"))
    energy: float = Parameter(positive(float, int), display_info=(1e6, "μJ"))
    soliton_num: float = Parameter(non_negative(float, int))
    quantum_noise: bool = Parameter(boolean, default=False)
    shape: str = Parameter(literal("gaussian", "sech"), default="gaussian")
    wavelength: float = Parameter(in_range_incl(100e-9, 3000e-9), display_info=(1e9, "nm"))
    intensity_noise: float = Parameter(in_range_incl(0, 1), display_info=(1e2, "%"), default=0)
    noise_correlation: float = Parameter(in_range_incl(-10, 10), default=0)
    width: float = Parameter(in_range_excl(0, 1e-9), display_info=(1e15, "fs"))
    t0: float = Parameter(in_range_excl(0, 1e-9), display_info=(1e15, "fs"))

    # simulation
    behaviors: str = Parameter(validator_list(literal("spm", "raman", "ss")), default=["spm", "ss"])
    parallel: bool = Parameter(boolean, default=True)
    raman_type: str = Parameter(
        literal("measured", "agrawal", "stolen"), converter=str.lower, default="agrawal"
    )
    ideal_gas: bool = Parameter(boolean, default=False)
    repeat: int = Parameter(positive(int), default=1)
    t_num: int = Parameter(positive(int))
    z_num: int = Parameter(positive(int))
    time_window: float = Parameter(positive(float, int))
    dt: float = Parameter(in_range_excl(0, 5e-15))
    tolerated_error: float = Parameter(in_range_excl(1e-15, 1e-3), default=1e-11)
    step_size: float = Parameter(non_negative(float, int), default=0)
    interpolation_range: tuple[float, float] = Parameter(float_pair)
    interpolation_degree: int = Parameter(positive(int), default=8)
    prev_sim_dir: str = Parameter(string)
    recovery_last_stored: int = Parameter(non_negative(int), default=0)
    worker_num: int = Parameter(positive(int))

    # computed
    field_0: np.ndarray = Parameter(type_checker(np.ndarray))
    spec_0: np.ndarray = Parameter(type_checker(np.ndarray))
    beta2: float = Parameter(type_checker(int, float))
    alpha_arr: np.ndarray = Parameter(type_checker(np.ndarray))
    alpha: float = Parameter(non_negative(float, int), default=0)
    gamma_arr: np.ndarray = Parameter(type_checker(np.ndarray))
    A_eff_arr: np.ndarray = Parameter(type_checker(np.ndarray))
    w: np.ndarray = Parameter(type_checker(np.ndarray))
    l: np.ndarray = Parameter(type_checker(np.ndarray))
    w_c: np.ndarray = Parameter(type_checker(np.ndarray))
    w0: float = Parameter(positive(float))
    w_power_fact: np.ndarray = Parameter(validator_list(type_checker(np.ndarray)))
    t: np.ndarray = Parameter(type_checker(np.ndarray))
    L_D: float = Parameter(non_negative(float, int))
    L_NL: float = Parameter(non_negative(float, int))
    L_sol: float = Parameter(non_negative(float, int))
    dynamic_dispersion: bool = Parameter(boolean)
    adapt_step_size: bool = Parameter(boolean)
    hr_w: np.ndarray = Parameter(type_checker(np.ndarray))
    z_targets: np.ndarray = Parameter(type_checker(np.ndarray))
    const_qty: np.ndarray = Parameter(type_checker(np.ndarray))
    beta_func: Callable[[float], list[float]] = Parameter(func_validator)
    gamma_func: Callable[[float], float] = Parameter(func_validator)
    datetime: datetime_module.datetime = Parameter(type_checker(datetime_module.datetime))
    version: str = Parameter(string)

    def prepare_for_dump(self) -> dict[str, Any]:
        param = asdict(self)
        param = Parameters.strip_params_dict(param)
        param["datetime"] = datetime_module.datetime.now()
        param["version"] = __version__
        return param

    def __post_init__(self):
        param_dict = {k: v for k, v in asdict(self).items() if v is not None}
        evaluator = Evaluator.default()
        evaluator.set(**param_dict)
        for p_name in mandatory_parameters:
            evaluator.compute(p_name)
        valid_fields = self.all_parameters()
        for k, v in evaluator.params.items():
            if k in valid_fields:
                setattr(self, k, v)

    @classmethod
    def all_parameters(cls) -> list[str]:
        return [f.name for f in fields(cls)]

    @classmethod
    def load(cls, path: os.PathLike) -> "Parameters":
        return cls(**utils.load_toml(path))

    @staticmethod
    def strip_params_dict(dico: dict[str, Any]) -> dict[str, Any]:
        """prepares a dictionary for serialization. Some keys may not be preserved
        (dropped because they take a lot of space and can be exactly reconstructed)

        Parameters
        ----------
        dico : dict
            dictionary
        """
        forbiden_keys = [
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
        ]
        types = (np.ndarray, float, int, str, list, tuple, dict)
        out = {}
        for key, value in dico.items():
            if key in forbiden_keys:
                continue
            if not isinstance(value, types):
                continue
            if isinstance(value, dict):
                out[key] = Parameters.strip_params_dict(value)
            elif isinstance(value, np.ndarray) and value.dtype == complex:
                continue
            else:
                out[key] = value

        if "variable" in out and len(out["variable"]) == 0:
            del out["variable"]

        return out


class Rule:
    def __init__(
        self,
        target: Union[str, list[Optional[str]]],
        func: Callable,
        args: list[str] = None,
        priorities: Union[int, list[int]] = None,
        conditions: dict[str, str] = None,
    ):
        targets = list(target) if isinstance(target, (list, tuple)) else [target]
        self.func = func
        if priorities is None:
            priorities = [1] * len(targets)
        elif isinstance(priorities, (int, float, np.integer, np.floating)):
            priorities = [priorities]
        self.targets = dict(zip(targets, priorities))
        if args is None:
            args = get_arg_names(func)
        self.args = args
        self.mock_func = _mock_function(len(self.args), len(self.targets))
        self.conditions = conditions or {}

    def __repr__(self) -> str:
        return f"Rule(targets={self.targets!r}, func={self.func!r}, args={self.args!r})"

    @classmethod
    def deduce(
        cls,
        target: Union[str, list[Optional[str]]],
        func: Callable,
        kwarg_names: list[str],
        n_var: int,
        args_const: list[str] = None,
        priorities: Union[int, list[int]] = None,
    ) -> list["Rule"]:
        """given a function that doesn't need all its keyword arguemtn specified, will
        return a list of Rule obj, one for each combination of n_var specified kwargs

        Parameters
        ----------
        target : str | list[str | None]
            name of the variable(s) that func returns
        func : Callable
            function to work with
        kwarg_names : list[str]
            list of all kwargs of the function to be used
        n_var : int
            how many shoulf be used per rule
        arg_const : list[str], optional
            override the name of the positional arguments

        Returns
        -------
        list[Rule]
            list of all possible rules

        Example
        -------
        >> def lol(a, b=None, c=None):
            pass
        >> print(Rule.deduce(["d"], lol, ["b", "c"], 1))
        [
            Rule(targets={'d': 1}, func=<function lol_0 at 0x7f9bce31d0d0>, args=['a', 'b']),
            Rule(targets={'d': 1}, func=<function lol_0 at 0x7f9bce31d160>, args=['a', 'c'])
        ]
        """
        rules: list[cls] = []
        for var_possibility in itertools.combinations(kwarg_names, n_var):

            new_func = func_rewrite(func, list(var_possibility), args_const)

            rules.append(cls(target, new_func, priorities=priorities))
        return rules


@dataclass
class EvalStat:
    priority: float = np.inf


class Evaluator:
    @classmethod
    def default(cls) -> "Evaluator":
        evaluator = cls()
        evaluator.append(*default_rules)
        return evaluator

    @classmethod
    def evaluate_default(cls, params: dict[str, Any], check_only=False) -> dict[str, Any]:
        evaluator = cls.default()
        evaluator.set(**params)
        for target in mandatory_parameters:
            evaluator.compute(target, check_only=check_only)
        return evaluator.params

    def __init__(self):
        self.rules: dict[str, list[Rule]] = defaultdict(list)
        self.params = {}
        self.__curent_lookup = set()
        self.eval_stats: dict[str, EvalStat] = defaultdict(EvalStat)
        self.logger = get_logger(__name__)

    def append(self, *rule: Rule):
        for r in rule:
            for t in r.targets:
                if t is not None:
                    self.rules[t].append(r)
                    self.rules[t].sort(key=lambda el: el.targets[t], reverse=True)

    def set(self, **params: Any):
        self.params.update(params)
        for k in params:
            self.eval_stats[k].priority = np.inf

    def reset(self):
        self.params = {}
        self.eval_stats = defaultdict(EvalStat)

    def get_default(self, key: str) -> Any:
        try:
            return getattr(Parameters, key).default
        except AttributeError:
            return None

    def compute(self, target: str, check_only=False) -> Any:
        """computes a target

        Parameters
        ----------
        target : str
            name of the target

        Returns
        -------
        Any
            return type of the target function

        Raises
        ------
        EvaluatorError
            a cyclic dependence exists
        KeyError
            there is no saved rule for the target
        """
        value = self.params.get(target)
        if value is None:
            prefix = "\t" * len(self.__curent_lookup)
            # Avoid cycles
            if target in self.__curent_lookup:
                raise EvaluatorError(
                    "cyclic dependency detected : "
                    f"{target!r} seems to depend on itself, "
                    f"please provide a value for at least one variable in {self.__curent_lookup}"
                )
            else:
                self.__curent_lookup.add(target)

            if len(self.rules[target]) == 0:
                error = EvaluatorError(f"no rule for {target}")
            else:
                error = None

            # try every rule until one succeeds
            for ii, rule in enumerate(
                filter(lambda r: self.validate_condition(r), self.rules[target])
            ):
                self.logger.debug(
                    prefix + f"attempt {ii+1} to compute {target}, this time using {rule!r}"
                )
                try:
                    args = [self.compute(k, check_only=check_only) for k in rule.args]
                    if check_only:
                        returned_values = rule.mock_func(*args)
                    else:
                        returned_values = rule.func(*args)
                    if len(rule.targets) == 1:
                        returned_values = [returned_values]
                    for ((param_name, param_priority), returned_value) in zip(
                        rule.targets.items(), returned_values
                    ):
                        if (
                            param_name == target
                            or param_name not in self.params
                            or self.eval_stats[param_name].priority < param_priority
                        ):
                            self.logger.info(
                                prefix
                                + f"computed {param_name}={returned_value} using {rule.func.__name__} from {rule.func.__module__}"
                            )
                            self.set_value(param_name, returned_value, param_priority)
                        if param_name == target:
                            value = returned_value
                    break
                except (EvaluatorError, KeyError, NoDefaultError) as e:
                    error = e
                    self.logger.debug(
                        prefix + f"error using {rule.func.__name__} : {str(error).strip()}"
                    )
                    continue
            else:
                default = self.get_default(target)
                if default is None:
                    error = NoDefaultError(prefix + f"No default provided for {target}")
                else:
                    value = default
                    self.set_value(target, value, 0)

            if value is None and error is not None:
                raise error

            self.__curent_lookup.remove(target)
        return value

    def __getitem__(self, key: str) -> Any:
        return self.params[key]

    def set_value(self, key: str, value: Any, priority: int):
        self.params[key] = value
        self.eval_stats[key].priority = priority

    def validate_condition(self, rule: Rule) -> bool:
        return all(self.compute(k) == v for k, v in rule.conditions.items())

    def __call__(self, target: str, args: list[str] = None):
        """creates a wrapper that adds decorated functions to the set of rules

        Parameters
        ----------
        target : str
            name of the target
        args : list[str], optional
            list of name of arguments. Automatically deduced from function signature if
            not provided, by default None
        """

        def wrapper(func):
            self.append(Rule(target, func, args))
            return func

        return wrapper


class Configuration:
    """
    Primary role is to load the final config file of the simulation and deduce every
    simulatin that has to happen. Iterating through the Configuration obj yields a list of
    parameter names and values that change throughout the simulation as well as parameter
    obj with the output path of the simulation saved in its output_path attribute.
    """

    configs: list[dict[str, Any]]
    data_dirs: list[list[Path]]
    sim_dirs: list[Path]
    num_sim: int
    repeat: int
    z_num: int
    total_length: float
    total_num_steps: int
    worker_num: int
    parallel: bool
    overwrite: bool
    name: str
    all_required: list[list[tuple[list[tuple[str, Any]], dict[str, Any]]]]
    #             |    |    |     |    |
    #             |    |    |     |    param name and value
    #             |    |    |     all variable parameters
    #             |    |    list of all variable parameters associated with the full config dict
    #             |    list of all configs for 1 fiber
    #             list of all fibers

    class State(enum.Enum):
        COMPLETE = enum.auto()
        PARTIAL = enum.auto()
        ABSENT = enum.auto()

    class Action(enum.Enum):
        RUN = enum.auto()
        WAIT = enum.auto()
        SKIP = enum.auto()

    @classmethod
    def load(cls, path: os.PathLike) -> "Configuration":
        return cls(utils.load_toml(path))

    def __init__(
        self,
        final_config: dict[str, Any],
        overwrite: bool = True,
        skip_callback: Callable[[int], None] = None,
    ):
        self.logger = get_logger(__name__)

        self.configs = [final_config]
        self.z_num = 0
        self.total_length = 0.0
        self.total_num_steps = 0
        self.sim_dirs = []
        self.overwrite = overwrite
        self.skip_callback = skip_callback
        self.worker_num = self.configs[0].get("worker_num", max(1, os.cpu_count() // 2))

        while "previous_config_file" in self.configs[0]:
            self.configs.insert(0, utils.load_toml(self.configs[0]["previous_config_file"]))
        self.override_configs()
        self.name = self.configs[-1].get("name", Parameters.name.default)
        self.repeat = self.configs[0].get("repeat", 1)

        for i, config in enumerate(self.configs):
            self.z_num += config["z_num"]
            self.total_length += config["length"]
            config.setdefault("name", f"{Parameters.name.default} {i}")
            self.sim_dirs.append(
                utils.ensure_folder(
                    Path(config["name"] + PARAM_SEPARATOR + "sc_tmp"),
                    mkdir=False,
                    prevent_overwrite=not self.overwrite,
                )
            )

            self.__validate_variable(config)
            Evaluator.evaluate_default(
                {
                    **{k: v for k, v in config.items() if k != "variable"},
                    **{k: v[0] for k, v in config["variable"].items()},
                },
                check_only=True,
            )
        self.__compute_sim_dirs()
        self.num_sim = len(self.data_dirs[-1])
        self.total_num_steps = sum(
            config["z_num"] * len(self.data_dirs[i]) for i, config in enumerate(self.configs)
        )
        self.final_sim_dir = utils.ensure_folder(
            Path(self.configs[-1]["name"]), mkdir=False, prevent_overwrite=not self.overwrite
        )
        self.parallel = self.configs[0].get("parallel", Parameters.parallel.default)

    def override_configs(self):
        self.configs[0].setdefault("variable", {})
        for pre, nex in zip(self.configs[:-1], self.configs[1:]):
            variable = nex.pop("variable", {})
            nex.update({k: v for k, v in pre.items() if k not in nex})
            nex["variable"] = variable

    def __validate_variable(self, config: dict[str, Any]):
        for k, v in config.get("variable", {}).items():
            p = getattr(Parameters, k)
            validator_list(p.validator)("variable " + k, v)
            if k not in valid_variable:
                raise TypeError(f"{k!r} is not a valid variable parameter")
            if len(v) == 0:
                raise ValueError(f"variable parameter {k!r} must not be empty")

    def __compute_sim_dirs(self):
        self.data_dirs: list[list[Path]] = [None] * len(self.configs)
        self.all_required: list[list[tuple[list[tuple[str, Any]], dict[str, Any]]]] = [None] * len(
            self.configs
        )
        self.all_required[0], self.data_dirs[0] = self.__prepare_1_fiber(
            0, self.configs[0], first=True
        )

        for i, config in enumerate(self.configs[1:]):
            config["variable"]["prev_data_dir"] = [str(p.resolve()) for p in self.data_dirs[i]]
            self.all_required[i + 1], self.data_dirs[i + 1] = self.__prepare_1_fiber(i + 1, config)

    def __prepare_1_fiber(
        self, index: int, config: dict[str, Any], first=False
    ) -> tuple[list[tuple[list[tuple[str, Any]], dict[str, Any]]], list[Path]]:
        required: list[tuple[list[tuple[str, Any]], dict[str, Any]]] = list(
            variable_iterator(config, self.repeat if first else 1)
        )
        for vary_list, _ in required:
            vary_list.insert(
                1, ("Fiber", "ABCDEFGHIJKLMNOPQRSTUVWXYZ"[index % 26] * (index // 26 + 1))
            )
        return required, [
            utils.ensure_folder(
                self.sim_dirs[index] / format_variable_list(vary_list),
                mkdir=False,
                prevent_overwrite=not self.overwrite,
            )
            for vary_list, c in required
        ]

    def __iter__(self) -> Generator[tuple[list[tuple[str, Any]], Parameters], None, None]:
        for sim_paths, fiber in zip(self.data_dirs, self.all_required):
            for variable_list, data_dir, params in self.__iter_1_sim(sim_paths, fiber):
                params.output_path = str(data_dir)
                yield variable_list, params

    def __iter_1_sim(
        self, sim_paths: list[Path], fiber: list[tuple[list[tuple[str, Any]], dict[str, Any]]]
    ) -> Generator[tuple[list[tuple[str, Any]], Path, Parameters], None, None]:
        sim_dict: dict[Path, tuple[list[tuple[str, Any]], dict[str, Any]]] = dict(
            zip(sim_paths, fiber)
        )
        while len(sim_dict) > 0:
            for data_dir, (variable_list, config_dict) in sim_dict.items():
                task, config_dict = self.__decide(data_dir, config_dict)
                if task == self.Action.RUN:
                    sim_dict.pop(data_dir)
                    yield variable_list, data_dir, Parameters(**config_dict)
                    if "recovery_last_stored" in config_dict and self.skip_callback is not None:
                        self.skip_callback(config_dict["recovery_last_stored"])
                    break
                elif task == self.Action.SKIP:
                    sim_dict.pop(data_dir)
                    self.logger.debug(f"skipping {data_dir} as it is already complete")
                    if self.skip_callback is not None:
                        self.skip_callback(config_dict["z_num"])
                    break
            else:
                self.logger.debug("sleeping while waiting for other simulations to complete")
                time.sleep(1)

    def __decide(
        self, data_dir: Path, config_dict: dict[str, Any]
    ) -> tuple["Configuration.Action", dict[str, Any]]:
        """decide what to to with a particular simulation

        Parameters
        ----------
        data_dir : Path
            path to the output of the simulation
        config_dict : dict[str, Any]
            configuration of the simulation

        Returns
        -------
        str : {'run', 'wait', 'skip'}
            what to do
        config_dict : dict[str, Any]
            config dictionary. The only key possibly modified is 'prev_data_dir', which
            gets set if the simulation is partially completed
        """
        out_status, num = self.sim_status(data_dir, config_dict)
        if out_status == self.State.COMPLETE:
            return self.Action.SKIP, config_dict
        elif out_status == self.State.PARTIAL:
            config_dict["recovery_data_dir"] = str(data_dir)
            config_dict["recovery_last_stored"] = num
            return self.Action.RUN, config_dict

        if "prev_data_dir" in config_dict:
            prev_data_path = Path(config_dict["prev_data_dir"])
            prev_status, _ = self.sim_status(prev_data_path)
            if prev_status in {self.State.PARTIAL, self.State.ABSENT}:
                return self.Action.WAIT, config_dict
        return self.Action.RUN, config_dict

    def sim_status(
        self, data_dir: Path, config_dict: dict[str, Any] = None
    ) -> tuple["Configuration.State", int]:
        """returns the status of a simulation

        Parameters
        ----------
        data_dir : Path
            directory where simulation data is to be saved
        config_dict : dict[str, Any], optional
            configuration of the simulation. If None, will attempt to load
            the params.toml file if present, by default None

        Returns
        -------
        Configuration.State
            status
        """
        num = utils.find_last_spectrum_num(data_dir)
        if config_dict is None:
            try:
                config_dict = utils.load_toml(data_dir / "params.toml")
            except FileNotFoundError:
                self.logger.warning(f"did not find 'params.toml' in {data_dir}")
                return self.State.ABSENT, 0
        if num == config_dict["z_num"] - 1:
            return self.State.COMPLETE, num
        elif config_dict["z_num"] - 1 > num > 0:
            return self.State.PARTIAL, num
        elif num == 0:
            return self.State.ABSENT, 0
        else:
            raise ValueError(f"Too many spectra in {data_dir}")

    def save_parameters(self):
        for config, sim_dir in zip(self.configs, self.sim_dirs):
            os.makedirs(sim_dir, exist_ok=True)
            utils.save_toml(sim_dir / f"initial_config.toml", config)


@dataclass
class PlotRange:
    left: float = Parameter(type_checker(int, float))
    right: float = Parameter(type_checker(int, float))
    unit: Callable[[float], float] = Parameter(units.is_unit, converter=units.get_unit)
    conserved_quantity: bool = Parameter(boolean, default=True)

    def __str__(self):
        return f"{self.left:.1f}-{self.right:.1f} {self.unit.__name__}"

    def sort_axis(self, axis: np.ndarray) -> tuple[np.ndarray, np.ndarray, tuple[float, float]]:
        return sort_axis(axis, self)


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
    if not isinstance(num_args, int) and isinstance(num_returns, int):
        raise TypeError(f"num_args and num_returns must be int")
    arg_str = ", ".join("a" * (n + 1) for n in range(num_args))
    return_str = ", ".join("True" for _ in range(num_returns))
    func_name = f"__mock_{num_args}_{num_returns}"
    func_str = f"def {func_name}({arg_str}):\n    return {return_str}"
    scope = {}
    exec(func_str, scope)
    out_func = scope[func_name]
    out_func.__module__ = "evaluator"
    return out_func


def format_variable_list(l: list[tuple[str, Any]]) -> str:
    str_list = []
    previous_fibers = []
    for p_name, p_value in l:
        if p_name == "prev_data_dir":
            previous_fibers += Path(p_value).name.split()[2:]
        else:
            ps = p_name.replace("/", "").replace(PARAM_SEPARATOR, "")
            vs = format_value(p_name, p_value).replace("/", "").replace(PARAM_SEPARATOR, "")
            str_list.append(ps + PARAM_SEPARATOR + vs)
    return PARAM_SEPARATOR.join(str_list[:1] + previous_fibers + str_list[1:])


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


def pretty_format_from_sim_name(name: str) -> str:
    """formats a pretty version of a simulation directory

    Parameters
    ----------
    name : str
        name of the simulation (directory name)

    Returns
    -------
    str
        prettier name
    """
    s = name.split(PARAM_SEPARATOR)
    out = []
    for key, value in zip(s[::2], s[1::2]):
        try:
            out += [key.replace("_", " "), getattr(Parameters, key).display(float(value))]
        except (AttributeError, ValueError):
            out.append(key + PARAM_SEPARATOR + value)
    return PARAM_SEPARATOR.join(out)


def variable_iterator(
    config: dict[str, Any], repeat: int = 1
) -> Generator[tuple[list[tuple[str, Any]], dict[str, Any]], None, None]:
    """given a config with "variable" parameters, iterates through every possible combination,
    yielding a a list of (parameter_name, value) tuples and a full config dictionary.

    Parameters
    ----------
    config : BareConfig
        initial config obj

    Yields
    -------
    Iterator[tuple[list[tuple[str, Any]], dict[str, Any]]]
        variable_list : a list of (name, value) tuple of parameter name and value that are variable.

        params : a dict[str, Any] to be fed to Parameters
    """
    possible_keys = []
    possible_ranges = []

    for key, values in config.get("variable", {}).items():
        possible_keys.append(key)
        possible_ranges.append(range(len(values)))

    combinations = itertools.product(*possible_ranges)

    master_index = 0

    for combination in combinations:
        indiv_config = {}
        variable_list = []
        for i, key in enumerate(possible_keys):
            parameter_value = config["variable"][key][combination[i]]
            indiv_config[key] = parameter_value
            variable_list.append((key, parameter_value))
        param_dict = deepcopy(config)
        param_dict.pop("variable")
        param_dict.update(indiv_config)
        for repeat_index in range(repeat):
            variable_ind = [("id", master_index)] + variable_list
            variable_ind += [("num", repeat_index)]
            yield variable_ind, param_dict
            master_index += 1


def reduce_all_variable(all_variable: list[list[tuple[str, Any]]]) -> list[tuple[str, Any]]:
    out = []
    for n, variable_list in enumerate(all_variable):
        out += [("fiber", "ABCDEFGHIJKLMNOPQRSTUVWXYZ"[n % 26] * (n // 26 + 1)), *variable_list]
    return out


default_rules: list[Rule] = [
    # Grid
    *Rule.deduce(
        ["z_targets", "t", "time_window", "t_num", "dt", "w_c", "w0", "w", "w_power_fact", "l"],
        math.build_sim_grid,
        ["time_window", "t_num", "dt"],
        2,
    ),
    Rule("adapt_step_size", lambda step_size: step_size == 0),
    Rule("dynamic_dispersion", lambda pressure: isinstance(pressure, (list, tuple, np.ndarray))),
    # Pulse
    Rule("spec_0", np.fft.fft, ["field_0"]),
    Rule("field_0", np.fft.ifft, ["spec_0"]),
    Rule("spec_0", utils.load_previous_spectrum, ["recovery_data_dir"], priorities=4),
    Rule("spec_0", utils.load_previous_spectrum, priorities=3),
    *Rule.deduce(
        ["pre_field_0", "peak_power", "energy", "width"],
        pulse.load_and_adjust_field_file,
        ["energy", "peak_power"],
        1,
        priorities=[2, 1, 1, 1],
    ),
    Rule("pre_field_0", pulse.initial_field, priorities=1),
    Rule(
        "field_0",
        pulse.add_shot_noise,
        ["pre_field_0", "quantum_noise", "w_c", "w0", "time_window", "dt"],
    ),
    Rule("peak_power", pulse.E0_to_P0, ["energy", "t0", "shape"]),
    Rule("peak_power", pulse.soliton_num_to_peak_power),
    Rule("energy", pulse.P0_to_E0, ["peak_power", "t0", "shape"]),
    Rule("energy", pulse.mean_power_to_energy),
    Rule("t0", pulse.width_to_t0),
    Rule("t0", pulse.soliton_num_to_t0),
    Rule("width", pulse.t0_to_width),
    Rule("soliton_num", pulse.soliton_num),
    Rule("L_D", pulse.L_D),
    Rule("L_NL", pulse.L_NL),
    Rule("L_sol", pulse.L_sol),
    # Fiber Dispersion
    Rule("wl_for_disp", fiber.lambda_for_dispersion),
    Rule("w_for_disp", units.m, ["wl_for_disp"]),
    Rule(
        "beta2_coefficients",
        fiber.dispersion_coefficients,
        ["wl_for_disp", "beta2_arr", "w0", "interpolation_range", "interpolation_degree"],
    ),
    Rule("beta2_arr", fiber.beta2),
    Rule("beta2_arr", fiber.dispersion_from_coefficients),
    Rule("beta2", lambda beta2_coefficients: beta2_coefficients[0]),
    Rule(
        ["wl_for_disp", "beta2_arr", "interpolation_range"],
        fiber.load_custom_dispersion,
        priorities=[2, 2, 2],
    ),
    Rule("hr_w", fiber.delayed_raman_w),
    Rule("n_eff", fiber.n_eff_hasan, conditions=dict(model="hasan")),
    Rule("n_eff", fiber.n_eff_marcatili, conditions=dict(model="marcatili")),
    Rule("n_eff", fiber.n_eff_marcatili_adjusted, conditions=dict(model="marcatili_adjusted")),
    Rule(
        "n_eff",
        fiber.n_eff_pcf,
        ["wl_for_disp", "pitch", "pitch_ratio"],
        conditions=dict(model="pcf"),
    ),
    Rule("capillary_spacing", fiber.HCARF_gap),
    # Fiber nonlinearity
    Rule("A_eff", fiber.A_eff_from_V),
    Rule("A_eff", fiber.A_eff_from_diam),
    Rule("A_eff", fiber.A_eff_hasan, conditions=dict(model="hasan")),
    Rule("A_eff", fiber.A_eff_from_gamma, priorities=-1),
    Rule("A_eff_arr", fiber.A_eff_from_V, ["core_radius", "V_eff_arr"]),
    Rule("A_eff_arr", fiber.load_custom_A_eff),
    Rule("A_eff_arr", fiber.constant_A_eff_arr, priorities=-1),
    Rule(
        "V_eff",
        fiber.V_parameter_koshiba,
        ["wavelength", "pitch", "pitch_ratio"],
        conditions=dict(model="pcf"),
    ),
    Rule("V_eff", fiber.V_eff_marcuse, ["wavelength", "core_radius", "numerical_aperture"]),
    Rule("V_eff_arr", fiber.V_parameter_koshiba, conditions=dict(model="pcf")),
    Rule("V_eff_arr", fiber.V_eff_marcuse),
    Rule("gamma", lambda gamma_arr: gamma_arr[0]),
    Rule("gamma_arr", fiber.gamma_parameter, ["n2", "w0", "A_eff_arr"]),
    # Fiber loss
    Rule("alpha_arr", fiber.compute_capillary_loss),
    Rule("alpha_arr", fiber.load_custom_loss),
    Rule("alpha_arr", lambda alpha, t: np.ones_like(t) * alpha),
    # gas
    Rule("n_gas_2", materials.n_gas_2),
]


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
