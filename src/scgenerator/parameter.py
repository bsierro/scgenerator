from __future__ import annotations

import datetime as datetime_module
import enum
import os
import time
from copy import copy
from dataclasses import asdict, dataclass, fields
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, TypeVar, Union

import numpy as np

from . import env, utils
from .const import PARAM_FN, __version__, VALID_VARIABLE, MANDATORY_PARAMETERS
from .logger import get_logger
from .utils import fiber_folder, update_path_name
from .variationer import VariationDescriptor, Variationer
from .evaluator import Evaluator
from .operators import NonLinearOperator, LinearOperator

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
        """

        self.validator = validator
        self.converter = converter
        self.default = default
        self.display_info = display_info

    def __set_name__(self, owner, name):
        self.name = name
        if self.default is not None:
            Evaluator.register_default_param(self.name, self.default)
        VariationDescriptor.register_formatter(self.name, self.display)

    def __get__(self, instance, owner):
        if instance is None:
            return self
        return instance.__dict__[self.name]

    def __delete__(self, instance):
        raise AttributeError("Cannot delete parameter")

    def __set__(self, instance, value):
        if isinstance(value, Parameter):
            defaut = None if self.default is None else copy(self.default)
            instance.__dict__[self.name] = defaut
        else:
            if value is not None:
                if self.converter is not None:
                    value = self.converter(value)
                self.validator(self.name, value)
            instance.__dict__[self.name] = value

    def display(self, num: float) -> str:
        if self.display_info is None:
            return str(num)
        else:
            fac, unit = self.display_info
            num_str = format(num * fac, ".2f")
            if num_str.endswith(".00"):
                num_str = num_str[:-3]
            return f"{num_str} {unit}"


@dataclass
class Parameters:
    """
    This class defines each valid parameter's name, type and valid value.
    """

    # root
    name: str = Parameter(string, default="no name")
    prev_data_dir: str = Parameter(string)
    recovery_data_dir: str = Parameter(string)
    previous_config_file: str = Parameter(string)
    output_path: Path = Parameter(type_checker(Path), default=Path("sc_data"), converter=Path)

    # # fiber
    input_transmission: float = Parameter(in_range_incl(0, 1), default=1.0)
    gamma: float = Parameter(non_negative(float, int))
    n2: float = Parameter(non_negative(float, int))
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
        literal("pcf", "marcatili", "marcatili_adjusted", "hasan", "custom"), default="custom"
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
    pressure: Union[float, Iterable[float]] = Parameter(
        validator_or(non_negative(float, int), num_list), display_info=(1e-5, "bar"), default=1e5
    )
    temperature: float = Parameter(positive(float, int), display_info=(1, "K"), default=300)
    plasma_density: float = Parameter(non_negative(float, int), default=0)

    # pulse
    field_file: str = Parameter(string)
    repetition_rate: float = Parameter(
        non_negative(float, int), display_info=(1e-3, "kHz"), default=40e6
    )
    peak_power: float = Parameter(positive(float, int), display_info=(1e-3, "kW"))
    mean_power: float = Parameter(positive(float, int), display_info=(1e3, "mW"))
    energy: float = Parameter(positive(float, int), display_info=(1e6, "μJ"))
    soliton_num: float = Parameter(non_negative(float, int))
    quantum_noise: bool = Parameter(boolean, default=False)
    additional_noise_factor: float = Parameter(positive(float, int), default=1)
    shape: str = Parameter(literal("gaussian", "sech"), default="gaussian")
    wavelength: float = Parameter(in_range_incl(100e-9, 3000e-9), display_info=(1e9, "nm"))
    intensity_noise: float = Parameter(in_range_incl(0, 1), display_info=(1e2, "%"), default=0)
    noise_correlation: float = Parameter(in_range_incl(-10, 10), default=0)
    width: float = Parameter(in_range_excl(0, 1e-9), display_info=(1e15, "fs"))
    t0: float = Parameter(in_range_excl(0, 1e-9), display_info=(1e15, "fs"))

    # simulation
    behaviors: tuple[str] = Parameter(
        validator_list(literal("spm", "raman", "ss")), converter=tuple, default=("spm", "ss")
    )
    parallel: bool = Parameter(boolean, default=True)
    raman_type: str = Parameter(literal("measured", "agrawal", "stolen"), converter=str.lower)
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
    linear_operator: LinearOperator = Parameter(type_checker(LinearOperator))
    nonlinear_operator: NonLinearOperator = Parameter(type_checker(NonLinearOperator))
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

    num: int = Parameter(non_negative(int))
    datetime: datetime_module.datetime = Parameter(type_checker(datetime_module.datetime))
    version: str = Parameter(string)

    def prepare_for_dump(self) -> dict[str, Any]:
        param = asdict(self)
        param = Parameters.strip_params_dict(param)
        param["datetime"] = datetime_module.datetime.now()
        param["version"] = __version__
        return param

    def compute(self, to_compute: list[str] = MANDATORY_PARAMETERS):
        param_dict = {k: v for k, v in asdict(self).items() if v is not None}
        evaluator = Evaluator.default()
        evaluator.set(**param_dict)
        results = [evaluator.compute(p_name) for p_name in to_compute]
        valid_fields = self.all_parameters()
        for k, v in evaluator.params.items():
            if k in valid_fields:
                setattr(self, k, v)
        return results

    def pformat(self) -> str:
        return "\n".join(
            f"{k} = {VariationDescriptor.format_value(k, v)}"
            for k, v in self.prepare_for_dump().items()
        )

    @classmethod
    def all_parameters(cls) -> list[str]:
        return [f.name for f in fields(cls)]

    @classmethod
    def load(cls, path: os.PathLike) -> "Parameters":
        return cls(**utils.load_toml(path))

    @classmethod
    def load_and_compute(cls, path: os.PathLike) -> "Parameters":
        p = cls.load(path)
        p.compute()
        return p

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

    @property
    def final_path(self) -> Path:
        if self.output_path is not None:
            return self.output_path.parent / update_path_name(self.output_path.name)
        return None


class Configuration:
    """
    Primary role is to load the final config file of the simulation and deduce every
    simulatin that has to happen. Iterating through the Configuration obj yields a list of
    parameter names and values that change throughout the simulation as well as parameter
    obj with the output path of the simulation saved in its output_path attribute.
    """

    fiber_configs: list[dict[str, Any]]
    vary_dicts: list[dict[str, list]]
    master_config: dict[str, Any]
    fiber_paths: list[Path]
    num_sim: int
    num_fibers: int
    repeat: int
    z_num: int
    total_num_steps: int
    worker_num: int
    parallel: bool
    overwrite: bool
    final_path: str
    all_configs: dict[tuple[tuple[int, ...], ...], "Configuration.__SimConfig"]

    @dataclass(frozen=True)
    class __SimConfig:
        descriptor: VariationDescriptor
        config: dict[str, Any]
        output_path: Path

        @property
        def sim_num(self) -> int:
            return len(self.descriptor.index)

    class State(enum.Enum):
        COMPLETE = enum.auto()
        PARTIAL = enum.auto()
        ABSENT = enum.auto()

    class Action(enum.Enum):
        RUN = enum.auto()
        WAIT = enum.auto()
        SKIP = enum.auto()

    def __init__(
        self,
        config_path: os.PathLike,
        overwrite: bool = True,
        wait: bool = False,
        skip_callback: Callable[[int], None] = None,
        final_output_path: os.PathLike = None,
    ):
        self.logger = get_logger(__name__)
        self.wait = wait

        self.overwrite = overwrite
        self.final_path, self.fiber_configs = utils.load_config_sequence(config_path)
        self.final_path = env.get(env.OUTPUT_PATH, self.final_path)
        if final_output_path is not None:
            self.final_path = final_output_path
        self.final_path = utils.ensure_folder(
            Path(self.final_path),
            mkdir=False,
            prevent_overwrite=not self.overwrite,
        )
        self.master_config = self.fiber_configs[0].copy()
        self.name = self.final_path.name
        self.z_num = 0
        self.total_num_steps = 0
        self.fiber_paths = []
        self.all_configs = {}
        self.skip_callback = skip_callback
        self.worker_num = self.master_config.get("worker_num", max(1, os.cpu_count() // 2))
        self.repeat = self.master_config.get("repeat", 1)
        self.variationer = Variationer()

        fiber_names = set()
        self.num_fibers = 0
        for i, config in enumerate(self.fiber_configs):
            config.setdefault("name", Parameters.name.default)
            self.z_num += config["z_num"]
            fiber_names.add(config["name"])
            vary_dict_list: list[dict[str, list]] = config.pop("variable")
            self.variationer.append(vary_dict_list)
            self.fiber_paths.append(
                utils.ensure_folder(
                    self.final_path / fiber_folder(i, self.name, config["name"]),
                    mkdir=False,
                    prevent_overwrite=not self.overwrite,
                )
            )
            self.__validate_variable(vary_dict_list)
            self.num_fibers += 1
            Evaluator.evaluate_default(
                self.__build_base_config()
                | config
                | {k: v[0] for vary_dict in vary_dict_list for k, v in vary_dict.items()},
                True,
            )
        self.num_sim = self.variationer.var_num()
        self.total_num_steps = sum(
            config["z_num"] * self.variationer.var_num(i)
            for i, config in enumerate(self.fiber_configs)
        )
        self.parallel = self.master_config.get("parallel", Parameters.parallel.default)

    def __build_base_config(self):
        cfg = self.master_config.copy()
        vary: list[dict[str, list]] = cfg.pop("variable")
        return cfg | {k: v[0] for vary_dict in vary for k, v in vary_dict.items()}

    def __validate_variable(self, vary_dict_list: list[dict[str, list]]):
        for vary_dict in vary_dict_list:
            for k, v in vary_dict.items():
                p = getattr(Parameters, k)
                validator_list(p.validator)("variable " + k, v)
                if k not in VALID_VARIABLE:
                    raise TypeError(f"{k!r} is not a valid variable parameter")
                if len(v) == 0:
                    raise ValueError(f"variable parameter {k!r} must not be empty")

    def __iter__(self) -> Iterator[tuple[VariationDescriptor, Parameters]]:
        for i in range(self.num_fibers):
            yield from self.iterate_single_fiber(i)

    def iterate_single_fiber(self, index: int) -> Iterator[tuple[VariationDescriptor, Parameters]]:
        """iterates through the parameters of only one fiber. It takes care of recovering partially
        completed simulations, skipping complete ones and waiting for the previous fiber to finish

        Parameters
        ----------
        index : int
            which fiber to iterate over

        Yields
        -------
        __SimConfig
            configuration obj
        """
        if index < 0:
            index = self.num_fibers + index
        sim_dict: dict[Path, Configuration.__SimConfig] = {}
        for descriptor in self.variationer.iterate(index):
            cfg = descriptor.update_config(self.fiber_configs[index])
            if index > 0:
                cfg["prev_data_dir"] = str(
                    self.fiber_paths[index - 1] / descriptor[:index].formatted_descriptor(True)
                )
            p = utils.ensure_folder(
                self.fiber_paths[index] / descriptor.formatted_descriptor(True),
                not self.overwrite,
                False,
            )
            cfg["output_path"] = p
            sim_config = self.__SimConfig(descriptor, cfg, p)
            sim_dict[p] = self.all_configs[sim_config.descriptor.index] = sim_config
        while len(sim_dict) > 0:
            for data_dir, sim_config in sim_dict.items():
                task, config_dict = self.__decide(sim_config)
                if task == self.Action.RUN:
                    sim_dict.pop(data_dir)
                    yield sim_config.descriptor, Parameters(**sim_config.config)
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
        self, sim_config: "Configuration.__SimConfig"
    ) -> tuple["Configuration.Action", dict[str, Any]]:
        """decide what to to with a particular simulation

        Parameters
        ----------
        sim_config : __SimConfig

        Returns
        -------
        str : Configuration.Action
            what to do
        config_dict : dict[str, Any]
            config dictionary. The only key possibly modified is 'prev_data_dir', which
            gets set if the simulation is partially completed
        """
        if not self.wait:
            return self.Action.RUN, sim_config.config
        out_status, num = self.sim_status(sim_config.output_path, sim_config.config)
        if out_status == self.State.COMPLETE:
            return self.Action.SKIP, sim_config.config
        elif out_status == self.State.PARTIAL:
            sim_config.config["recovery_data_dir"] = str(sim_config.output_path)
            sim_config.config["recovery_last_stored"] = num
            return self.Action.RUN, sim_config.config

        if "prev_data_dir" in sim_config.config:
            prev_data_path = Path(sim_config.config["prev_data_dir"])
            prev_status, _ = self.sim_status(prev_data_path)
            if prev_status in {self.State.PARTIAL, self.State.ABSENT}:
                return self.Action.WAIT, sim_config.config
        return self.Action.RUN, sim_config.config

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
                config_dict = utils._open_config(data_dir / PARAM_FN)
            except FileNotFoundError:
                self.logger.warning(f"did not find {PARAM_FN!r} in {data_dir}")
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
        os.makedirs(self.final_path, exist_ok=True)
        cfgs = [
            cfg | dict(variable=self.variationer.all_dicts[i])
            for i, cfg in enumerate(self.fiber_configs)
        ]
        utils.save_toml(self.final_path / f"initial_config.toml", dict(name=self.name, Fiber=cfgs))

    @property
    def first(self) -> Parameters:
        for _, param in self:
            return param


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
