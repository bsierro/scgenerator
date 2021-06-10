import os
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Tuple, Union

import numpy as np
from numpy import pi

from . import io, utils
from .defaults import default_parameters
from .errors import *
from .logger import get_logger
from .math import abs2, power_fact
from .physics import fiber, pulse, units
from .utils import count_variations, override_config, required_simulations
from .utils.parameter import BareConfig, BareParams, hc_model_specific_parameters


@dataclass
class Params(BareParams):
    @classmethod
    def from_bare(cls, bare: BareParams):
        return cls(**asdict(bare))

    def __post_init__(self):
        self.compute()

    def compute(self):
        logger = get_logger(__name__)

        (
            self.z_targets,
            self.t,
            self.time_window,
            self.t_num,
            self.dt,
            self.w_c,
            self.w0,
            self.w,
            self.w_power_fact,
        ) = build_sim_grid(
            self.length, self.z_num, self.wavelength, self.time_window, self.t_num, self.dt
        )

        # Initial field may influence the grid
        if self.mean_power is not None:
            self.energy = self.mean_power / self.repetition_rate
        (
            custom_field,
            self.width,
            self.peak_power,
            self.energy,
            self.field_0,
        ) = pulse.setup_custom_field(self)
        if self.readjust_wavelength:
            delta_w = self.w_c[np.argmax(abs2(np.fft.fft(self.field_0)))]
            logger.debug(f"adjusted w by {delta_w}")
            self.wavelength = units.m.inv(units.m(self.wavelength) - delta_w)
            self.w_c, self.w0, self.w, self.w_power_fact = update_frequency_domain(
                self.t, self.wavelength
            )

        if self.step_size is not None:
            self.error_ok = self.step_size
            self.adapt_step_size = False
        else:
            self.error_ok = self.tolerated_error
            self.adapt_step_size = True

        # FIBER
        self.interp_range = [
            max(self.lower_wavelength_interp_limit, units.m.inv(np.max(self.w[self.w > 0]))),
            min(self.upper_wavelength_interp_limit, units.m.inv(np.min(self.w[self.w > 0]))),
        ]

        temp_gamma = None
        if self.effective_mode_diameter is not None:
            self.A_eff = (self.effective_mode_diameter / 2) ** 2 * pi
        if self.beta is not None:
            self.beta = np.array(self.beta)
            self.dynamic_dispersion = False
        else:
            self.dynamic_dispersion = fiber.is_dynamic_dispersion(self.pressure)
            self.beta, temp_gamma = fiber.compute_dispersion(self)
            if self.dynamic_dispersion:
                self.gamma_func = temp_gamma
                self.beta_func = self.beta
                self.beta = self.beta_func(0)
                temp_gamma = temp_gamma(0)

        if self.gamma is None:
            self.gamma = temp_gamma
            logger.info(f"using computed \u0263 = {self.gamma:.2e} W/m^2")

        # Raman response
        if "raman" in self.behaviors:
            self.hr_w = fiber.delayed_raman_w(self.t, self.dt, self.raman_type)

        # GENERIC PULSE
        if not custom_field:
            custom_field = False
            (
                self.width,
                self.t0,
                self.peak_power,
                self.energy,
                self.soliton_num,
            ) = pulse.conform_pulse_params(
                self.shape,
                self.width,
                self.t0,
                self.peak_power,
                self.energy,
                self.soliton_num,
                self.gamma,
                self.beta,
            )
            logger.info(f"computed initial N = {self['soliton_num']:.3g}")

            self.L_D = self.t0 ** 2 / abs(self.beta[0])
            self.L_NL = 1 / (self.gamma * self.peak_power) if self.gamma else np.inf
            self.L_sol = pi / 2 * self.L_D

            # Technical noise
            if self.intensity_noise is not None and self.intensity_noise > 0:
                delta_int, delta_T0 = pulse.technical_noise(self.intensity_noise)
                self["peak_power"] *= delta_int
                self["t0"] *= delta_T0
                self["width"] *= delta_T0

            self.field_0 = pulse.initial_field(self.t, self.shape, self.t0, self.peak_power)

        if self.quantum_noise:
            self.field_0 = self.field_0 + pulse.shot_noise(
                self.w_c, self.w0, self.time_window, self.dt
            )

        self.spec_0 = np.fft.fft(self.field_0)

    def build_sim_grid(self):
        (
            self.z_targets,
            self.t,
            self.time_window,
            self.t_num,
            self.dt,
            self.w_c,
            self.w0,
            self.w,
            self.w_power_fact,
        ) = build_sim_grid(
            self.length, self.z_num, self.wavelength, self.time_window, self.t_num, self.dt
        )


@dataclass
class Config(BareConfig):
    @classmethod
    def from_bare(cls, bare: BareConfig):
        return cls(**asdict(bare))

    def __post_init__(self):
        for p_name, value in self.__dict__.items():
            if value is not None and p_name in self.variable:
                raise DuplicateParameterError(f"got multiple values for parameter {p_name!r}")
        self.setdefault("name", "no name")
        self.fiber_consistency()
        if self.model in hc_model_specific_parameters:
            self.gas_consistency()
        self.pulse_consistency()
        self.simulation_consistency()

    def fiber_consistency(self):
        if self.contains("beta"):
            if not (self.contains("A_eff") or self.contains("effective_mode_diameter")):
                self.gamma = self.get("gamma", specified_parameters=["beta"])
            self.setdefault("model", "custom")

        elif self.contains("dispersion_file"):
            if not (self.contains("A_eff") or self.contains("effective_mode_diameter")):
                fiber = self.get("gamma", specified_parameters=["dispersion_file"])
            self.setdefault("model", "custom")

        else:
            fiber = self.get("model")

            if self.model == "pcf":
                fiber = self.get_fiber("pitch")
                fiber = self.get_fiber("pitch_ratio")

            elif self.model == "hasan":
                fiber = self.get_multiple(
                    fiber, ["capillary_spacing", "capillary_outer_d"], 1, fiber_model="hasan"
                )
                for param in [
                    "core_radius",
                    "capillary_num",
                    "capillary_thickness",
                    "capillary_resonance_strengths",
                    "capillary_nested",
                ]:
                    fiber = self.get_fiber(param)
            else:
                for param in hc_model_specific_parameters[self.model]:
                    fiber = self.get_fiber(param)
        for param in ["length", "input_transmission"]:
            fiber = self.get(param)

    def gas_consistency(self):
        for param in ["gas_name", "temperature", "pressure", "plasma_density"]:
            self.get(param, specified_params=["gas"])

    def pulse_consistency(self):
        for param in ["wavelength", "quantum_noise", "intensity_noise"]:
            self.get(param)

        if not self.contains("field_file"):
            self.get("shape")

            if self.contains("soliton_num"):
                self.get_multiple(
                    ["peak_power", "mean_power", "energy", "width", "t0"],
                    1,
                    specified_parameters=["soliton_num"],
                )

            else:
                self.get_multiple(["t0", "width"], 1)
                self.get_multiple(["peak_power", "energy", "mean_power"], 1)
        if self.contains("mean_power"):
            self.get("repetition_rate", specified_parameters=["mean_power"])

    def simulation_consistency(self):
        self.get_multiple(["dt", "t_num", "time_window"], 2)

        for param in [
            "behaviors",
            "z_num",
            "frep",
            "tolerated_error",
            "parallel",
            "repeat",
            "lower_wavelength_interp_limit",
            "upper_wavelength_interp_limit",
            "ideal_gas",
            "readjust_wavelength",
            "recovery_last_stored",
        ]:
            self.get(param)

        if (
            any(["raman" in l for l in self.variable.get("behaviors", [])])
            or "raman" in self.behaviors
        ):
            self.get("raman_type", specified_parameters=["raman"])

    def contains(self, key):
        return self.variable.get(key) is not None or getattr(self, key) is not None

    def get(self, param, **kwargs) -> Any:
        """checks if param is in the parameter section dict and attempts to fill in a default value

        Parameters
        ----------
        param : str
            the name of the parameter (dict key)
        kwargs : any
            key word arguments passed to the MissingParameterError constructor

        Raises
        ------
        MissingFiberParameterError
            raised when a parameter is missing and no default exists
        """

        # whether the parameter is in the right place and valid is checked elsewhere,
        # here, we just make sure it is present.
        if not self.contains(param):
            try:
                setattr(self, param, default_parameters[param])
            except KeyError:
                raise MissingParameterError(param, **kwargs)

    def get_fiber(self, param, **kwargs):
        """wrapper for fiber parameters that depend on fiber model"""
        self.get(param, fiber_model=self.model, **kwargs)

    def get_multiple(self, params, num, **kwargs):
        """similar to the get method but works with several parameters

        Parameters
        ----------
        params : list of str
            names of the required parameters
        num : int
            how many of the parameters in params are required

        Raises
        ------
        MissingParameterError
            raised when not enough parameters are provided and no defaults exist
        """
        gotten = 0
        for param in params:
            try:
                self.get(param, **kwargs)
                gotten += 1
            except MissingParameterError:
                pass
            if gotten >= num:
                return
        raise MissingParameterError(params, num_required=num, **kwargs)

    def setdefault(self, param, value):
        if getattr(self, param) is None:
            setattr(self, param, value)


class ParamSequence:
    def __init__(self, config_dict: Union[Dict[str, Any], os.PathLike, BareConfig]):
        """creates a param sequence from a base config

        Parameters
        ----------
        config_dict : Union[Dict[str, Any], os.PathLike, BareConfig]
            Can be either a dictionary, a path to a config toml file or BareConfig obj
        """
        if isinstance(config_dict, BareConfig):
            self.config = config_dict
        else:
            if not isinstance(config_dict, Mapping):
                config_dict = io.load_toml(config_dict)
            self.config = Config(**config_dict)
        self.name = self.config.name
        self.logger = get_logger(__name__)

        self.num_sim, self.num_variable = count_variations(self.config)
        self.num_steps = self.num_sim * self.config.z_num
        self.single_sim = self.num_sim == 1

    def __iter__(self) -> Iterator[Tuple[List[Tuple[str, Any]], Params]]:
        """iterates through all possible parameters, yielding a config as well as a flattened
        computed parameters set each time"""
        for variable_list, bare_params in required_simulations(self.config):
            yield variable_list, Params.from_bare(bare_params)

    def __len__(self):
        return self.num_sim

    def __repr__(self) -> str:
        return f"dispatcher generated from config {self.name}"


class ContinuationParamSequence(ParamSequence):
    def __init__(self, prev_sim_dir: os.PathLike, new_config_dict: Dict[str, Any]):
        """Parameter sequence that builds on a previous simulation but with a new configuration
        It is recommended that only the fiber and the number of points stored may be changed and
        changing other parameters could results in unexpected behaviors. The new config doesn't have to
        be a full configuration (i.e. you can specify only the parameters that change).

        Parameters
        ----------
        prev_sim_dir : PathLike
            path to the folder of the previous simulation containing 'initial_config.toml'
        new_config : Dict[str, Any]
            new config
        """
        self.prev_sim_dir = Path(prev_sim_dir)
        init_config = io.load_config(self.prev_sim_dir / "initial_config.toml")

        self.prev_variable_lists = [
            (set(variable_list[1:]), self.prev_sim_dir / utils.format_variable_list(variable_list))
            for variable_list, _ in required_simulations(init_config)
        ]

        new_config = utils.override_config(new_config_dict, init_config)
        super().__init__(new_config)

    def __iter__(self) -> Iterator[Tuple[List[Tuple[str, Any]], Params]]:
        """iterates through all possible parameters, yielding a config as well as a flattened
        computed parameters set each time"""
        for variable_list, bare_params in required_simulations(self.config):
            prev_data_dir = self.find_prev_data_dir(variable_list).resolve()
            bare_params.prev_data_dir = str(prev_data_dir)
            yield variable_list, Params.from_bare(bare_params)

    def find_prev_data_dir(self, new_variable_list: List[Tuple[str, Any]]) -> Path:
        """finds the previous simulation data that this new config should start from

        Parameters
        ----------
        new_variable_list : List[Tuple[str, Any]]
            as yielded by required_simulations

        Returns
        -------
        Path
            path to the data folder

        Raises
        ------
        ValueError
            no data folder found
        """
        to_test = set(new_variable_list[1:])
        for old_v_list, path in self.prev_variable_lists:
            if to_test.issuperset(old_v_list):
                return path

        raise ValueError(
            f"cannot find a previous data folder for {new_variable_list} in {self.prev_sim_dir}"
        )


class RecoveryParamSequence(ParamSequence):
    def __init__(self, config_dict, task_id):
        super().__init__(config_dict)
        self.id = task_id
        self.num_steps = 0

        not_started = self.num_sim
        sub_folders = io.get_data_dirs(io.get_sim_dir(self.id))

        for sub_folder in utils.PBars(
            sub_folders, "Initial recovery", head_kwargs=dict(unit="sim")
        ):
            num_left = io.num_left_to_propagate(sub_folder, self.config.z_num)
            if num_left == 0:
                self.num_sim -= 1
            self.num_steps += num_left
            not_started -= 1

        self.num_steps += not_started * self.config.z_num
        self.single_sim = self.num_sim == 1

        self.prev_sim_dir = None
        if self.config.prev_sim_dir is not None:
            self.prev_sim_dir = Path(self.config.prev_sim_dir)
            init_config = io.load_config(self.prev_sim_dir / "initial_config.toml")
            self.prev_variable_lists = [
                (
                    set(variable_list[1:]),
                    self.prev_sim_dir / utils.format_variable_list(variable_list),
                )
                for variable_list, _ in required_simulations(init_config)
            ]

    def __iter__(self) -> Iterator[Tuple[List[Tuple[str, Any]], Params]]:
        for variable_list, bare_params in required_simulations(self.config):

            data_dir = io.get_sim_dir(self.id) / utils.format_variable_list(variable_list)

            if not data_dir.is_dir() or io.find_last_spectrum_num(data_dir) == 0:
                if (prev_data_dir := self.find_prev_data_dir(variable_list)) is not None:
                    bare_params.prev_data_dir = str(prev_data_dir)
                yield variable_list, Params.from_bare(bare_params)
            elif io.num_left_to_propagate(data_dir, self.config.z_num) != 0:
                yield variable_list, recover_params(bare_params, data_dir)
            else:
                continue

    def find_prev_data_dir(self, new_variable_list: List[Tuple[str, Any]]) -> Path:
        """finds the previous simulation data that this new config should start from

        Parameters
        ----------
        new_variable_list : List[Tuple[str, Any]]
            as yielded by required_simulations

        Returns
        -------
        Path
            path to the data folder

        Raises
        ------
        ValueError
            no data folder found
        """
        if self.prev_sim_dir is None:
            return None
        to_test = set(new_variable_list[1:])
        for old_v_list, path in self.prev_variable_lists:
            if to_test.issuperset(old_v_list):
                return path

        raise ValueError(
            f"cannot find a previous data folder for {new_variable_list} in {self.prev_sim_dir}"
        )


def validate_config_sequence(*configs: os.PathLike) -> Config:
    """validates a sequence of configs where all but the first one may have
    parameters missing

    Parameters
    ----------
    configs : os.PathLike
        sequence of paths to toml config files. The first element may be a folder containing data intead

    Returns
    -------
    Dict[str, Any]
        the final config as would be simulated, but of course missing input fields in the middle
    """
    previous = None
    for config in configs:
        if (p := Path(config)).is_dir():
            config = p / "initial_config.toml"
        dico = io.load_toml(config)
        previous = Config.from_bare(override_config(dico, previous))
    return previous


def wspace(t, t_num=0):
    """frequency array such that x(t) <-> np.fft(x)(w)
    Parameters
    ----------
        t : float or array
            float : total width of the time window
            array : time array
        t_num : int-
            if t is a float, specifies the number of points
    Returns
    ----------
        w : array
            linspace of frencies corresponding to t
    """
    if isinstance(t, (np.ndarray, list, tuple)):
        dt = t[1] - t[0]
        t_num = len(t)
        t = t[-1] - t[0] + dt
    else:
        dt = t / t_num
    w = 2 * pi * np.arange(t_num) / t
    w = np.where(w >= pi / dt, w - 2 * pi / dt, w)
    return w


def tspace(time_window=None, t_num=None, dt=None):
    """returns a time array centered on 0
    Parameters
    ----------
        time_window : float
            total time spanned
        t_num : int
            number of points
        dt : float
            time resolution

        at least 2 arguments must be given. They are prioritize as such
        t_num > time_window > dt

    Returns
    -------
        t : array
            a linearily spaced time array
    Raises
    ------
        TypeError
            missing at least 1 argument
    """
    if t_num is not None:
        if isinstance(time_window, (float, int)):
            return np.linspace(-time_window / 2, time_window / 2, int(t_num))
        elif isinstance(dt, (float, int)):
            time_window = (t_num - 1) * dt
            return np.linspace(-time_window / 2, time_window / 2, t_num)
    elif isinstance(time_window, (float, int)) and isinstance(dt, (float, int)):
        t_num = int(time_window / dt) + 1
        return np.linspace(-time_window / 2, time_window / 2, t_num)
    else:
        raise TypeError("not enough parameter to determine time vector")


def recover_params(params: BareParams, data_folder: Path) -> Params:
    params = Params.from_bare(params)
    try:
        prev = io.load_params(data_folder / "params.toml")
        (
            prev.z_targets,
            prev.t,
            prev.time_window,
            prev.t_num,
            prev.dt,
            prev.w_c,
            prev.w0,
            prev.w,
            prev.w_power_fact,
        ) = build_sim_grid(
            prev.length, prev.z_num, prev.wavelength, prev.time_window, prev.t_num, prev.dt
        )
    except FileNotFoundError:
        prev = BareParams()
    for k, v in filter(lambda el: el[1] is not None, vars(prev).items()):
        if getattr(params, k) is None:
            setattr(params, k, v)
    num, last_spectrum = io.load_last_spectrum(data_folder)
    params.spec_0 = last_spectrum
    params.field_0 = np.fft.ifft(last_spectrum)
    params.recovery_last_stored = num
    params.cons_qty = np.load(data_folder / "cons_qty.npy")
    return params


def build_sim_grid(
    length: float,
    z_num: int,
    wavelength: float,
    time_window: float = None,
    t_num: int = None,
    dt: float = None,
):
    """computes a bunch of values that relate to the simulation grid

    Parameters
    ----------
    params : dict
        flattened parameter dictionary

    Returns
    -------
    dict
        updated parameter dictionary
    """
    t = tspace(time_window, t_num, dt)

    time_window = t.max() - t.min()
    dt = t[1] - t[0]
    t_num = len(t)
    z_targets = np.linspace(0, length, z_num)
    w_c, w0, w, w_power_fact = update_frequency_domain(t, wavelength)
    return z_targets, t, time_window, t_num, dt, w_c, w0, w, w_power_fact


def update_frequency_domain(t, wavelength):
    w_c = wspace(t)
    w0 = units.m(wavelength)
    w = w_c + w0
    w_power_fact = np.array([power_fact(w_c, k) for k in range(2, 11)])
    return w_c, w0, w, w_power_fact
