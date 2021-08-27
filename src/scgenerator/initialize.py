import os
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Tuple, Union
from collections import defaultdict

import numpy as np

from . import io, utils
from .defaults import default_parameters
from .errors import *
from .logger import get_logger
from .utils import override_config, required_simulations
from .utils.evaluator import Evaluator
from .utils.parameter import (
    BareConfig,
    Parameters,
    hc_model_specific_parameters,
    mandatory_parameters,
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

        if self.contains("dispersion_file") or self.contains("beta2_coefficients"):
            if not (
                self.contains("A_eff")
                or self.contains("A_eff_file")
                or self.contains("effective_mode_diameter")
            ):
                self.get("gamma", specified_parameters=["custom fiber model"])
                self.get("n2", specified_parameters=["custom fiber model"])
            self.setdefault("model", "custom")

        else:
            self.get("model")

            if self.model == "pcf":
                self.get_fiber("pitch")
                self.get_fiber("pitch_ratio")

            elif self.model == "hasan":
                self.get_multiple(
                    ["capillary_spacing", "capillary_outer_d"], 1, fiber_model="hasan"
                )
                for param in [
                    "core_radius",
                    "capillary_num",
                    "capillary_thickness",
                    "capillary_resonance_strengths",
                    "capillary_nested",
                ]:
                    self.get_fiber(param)
            else:
                for param in hc_model_specific_parameters[self.model]:
                    self.get_fiber(param)
        if self.contains("loss"):
            if self.loss == "capillary":
                for param in ["core_radius", "he_mode"]:
                    self.get_fiber(param)
        for param in ["length", "input_transmission"]:
            self.get(param)

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
            "tolerated_error",
            "parallel",
            "repeat",
            "interpolation_range",
            "interpolation_degree",
            "ideal_gas",
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
        if isinstance(config_dict, Config):
            self.config = config_dict
        elif isinstance(config_dict, BareConfig):
            self.config = Config.from_bare(config_dict)
        else:
            if not isinstance(config_dict, Mapping):
                config_dict = io.load_toml(config_dict)
            self.config = Config(**config_dict)
        self.name = self.config.name
        self.logger = get_logger(__name__)

        self.update_num_sim()

    def __iter__(self) -> Iterator[Tuple[List[Tuple[str, Any]], Parameters]]:
        """iterates through all possible parameters, yielding a config as well as a flattened
        computed parameters set each time"""
        for variable_list, params in required_simulations(self.config):
            yield variable_list, params

    def __len__(self):
        return self.num_sim

    def __repr__(self) -> str:
        return f"dispatcher generated from config {self.name}"

    def update_num_sim(self):
        num_sim = self.count_variations()
        self.num_sim = num_sim
        self.num_steps = self.num_sim * self.config.z_num
        self.single_sim = self.num_sim == 1

    def count_variations(self) -> int:
        return count_variations(self.config)


class ContinuationParamSequence(ParamSequence):
    def __init__(self, prev_sim_dir: os.PathLike, new_config: BareConfig):
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
        self.bare_configs = BareConfig.load_sequence(new_config.previous_config_file)
        self.bare_configs.append(new_config)
        self.bare_configs[0] = Config.from_bare(self.bare_configs[0])
        final_config = utils.final_config_from_sequence(*self.bare_configs)
        super().__init__(final_config)

    def __iter__(self) -> Iterator[Tuple[List[Tuple[str, Any]], Parameters]]:
        """iterates through all possible parameters, yielding a config as well as a flattened
        computed parameters set each time"""
        for variable_list, params in required_simulations(*self.bare_configs):
            prev_data_dir = self.find_prev_data_dirs(variable_list)[0]
            params.prev_data_dir = str(prev_data_dir.resolve())
            yield variable_list, params

    def find_prev_data_dirs(self, new_variable_list: List[Tuple[str, Any]]) -> List[Path]:
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
        new_target = set(utils.format_variable_list(new_variable_list).split()[2:])
        path_dic = defaultdict(list)
        max_in_common = 0
        for data_dir in self.prev_sim_dir.glob("id*"):
            candidate = set(data_dir.name.split()[2:])
            in_common = candidate & new_target
            num_in_common = len(in_common)
            max_in_common = max(num_in_common, max_in_common)
            path_dic[num_in_common].append(data_dir)

        return path_dic[max_in_common]

    def count_variations(self) -> int:
        return count_variations(*self.bare_configs)


def count_variations(*bare_configs: BareConfig) -> int:
    sim_num = 1
    for conf in bare_configs:
        for l in conf.variable.values():
            sim_num *= len(l)
    return sim_num * (bare_configs[0].repeat or 1)


class RecoveryParamSequence(ParamSequence):
    def __init__(self, config_dict, task_id):
        super().__init__(config_dict)
        self.id = task_id
        self.num_steps = 0

        self.prev_sim_dir = None
        if self.config.prev_sim_dir is not None:
            self.prev_sim_dir = Path(self.config.prev_sim_dir)
            init_config = BareConfig.load(self.prev_sim_dir / "initial_config.toml")
            self.prev_variable_lists = [
                (
                    set(variable_list[1:]),
                    self.prev_sim_dir / utils.format_variable_list(variable_list),
                )
                for variable_list, _ in required_simulations(init_config)
            ]
            additional_sims_factor = int(
                np.prod(
                    [
                        len(init_config.variable[k])
                        for k in (self.config.variable.keys() & init_config.variable.keys())
                        if init_config.variable[k] != self.config.variable[k]
                    ]
                )
            )
            self.update_num_sim(self.num_sim * additional_sims_factor)
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

    def __iter__(self) -> Iterator[Tuple[List[Tuple[str, Any]], Parameters]]:
        for variable_list, params in required_simulations(self.config):

            data_dir = io.get_sim_dir(self.id) / utils.format_variable_list(variable_list)

            if not data_dir.is_dir() or io.find_last_spectrum_num(data_dir) == 0:
                if (prev_data_dir := self.find_prev_data_dir(variable_list)) is not None:
                    params.prev_data_dir = str(prev_data_dir)
                yield variable_list, params
            elif io.num_left_to_propagate(data_dir, self.config.z_num) != 0:
                yield variable_list, params + "Needs to rethink recovery procedure"
            else:
                continue

    def find_prev_data_dirs(self, new_variable_list: List[Tuple[str, Any]]) -> List[Path]:
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
        new_set = set(new_variable_list[1:])
        path_dic = defaultdict(list)
        max_in_common = 0
        for stored_set, path in self.prev_variable_lists:
            in_common = stored_set & new_set
            num_in_common = len(in_common)
            max_in_common = max(num_in_common, max_in_common)
            path_dic[num_in_common].append(path)

        return path_dic[max_in_common]


def validate_config_sequence(*configs: os.PathLike) -> tuple[str, int]:
    """validates a sequence of configs where all but the first one may have
    parameters missing

    Parameters
    ----------
    configs : os.PathLike
        sequence of paths to toml config files. The first element may be a folder containing data intead

    Returns
    -------
    int
        total number of simulations
    """

    previous = None
    configs = BareConfig.load_sequence(*configs)
    for config in configs:
        # if (p := Path(config)).is_dir():
        #     config = p / "initial_config.toml"
        new_conf = config
        previous = Config.from_bare(override_config(new_conf, previous))
    return previous.name, count_variations(*configs)


# def wspace(t, t_num=0):
#     """frequency array such that x(t) <-> np.fft(x)(w)
#     Parameters
#     ----------
#         t : float or array
#             float : total width of the time window
#             array : time array
#         t_num : int-
#             if t is a float, specifies the number of points
#     Returns
#     ----------
#         w : array
#             linspace of frencies corresponding to t
#     """
#     if isinstance(t, (np.ndarray, list, tuple)):
#         dt = t[1] - t[0]
#         t_num = len(t)
#         t = t[-1] - t[0] + dt
#     else:
#         dt = t / t_num
#     w = 2 * pi * np.arange(t_num) / t
#     w = np.where(w >= pi / dt, w - 2 * pi / dt, w)
#     return w


# def tspace(time_window=None, t_num=None, dt=None):
#     """returns a time array centered on 0
#     Parameters
#     ----------
#         time_window : float
#             total time spanned
#         t_num : int
#             number of points
#         dt : float
#             time resolution

#         at least 2 arguments must be given. They are prioritize as such
#         t_num > time_window > dt

#     Returns
#     -------
#         t : array
#             a linearily spaced time array
#     Raises
#     ------
#         TypeError
#             missing at least 1 argument
#     """
#     if t_num is not None:
#         if isinstance(time_window, (float, int)):
#             return np.linspace(-time_window / 2, time_window / 2, int(t_num))
#         elif isinstance(dt, (float, int)):
#             time_window = (t_num - 1) * dt
#             return np.linspace(-time_window / 2, time_window / 2, t_num)
#     elif isinstance(time_window, (float, int)) and isinstance(dt, (float, int)):
#         t_num = int(time_window / dt) + 1
#         return np.linspace(-time_window / 2, time_window / 2, t_num)
#     else:
#         raise TypeError("not enough parameter to determine time vector")


# def recover_params(params: Parameters, data_folder: Path) -> Parameters:
#     try:
#         prev = Parameters.load(data_folder / "params.toml")
#     except FileNotFoundError:
#         prev = Parameters()
#     for k, v in filter(lambda el: el[1] is not None, vars(prev).items()):
#         if getattr(params, k) is None:
#             setattr(params, k, v)
#     num, last_spectrum = io.load_last_spectrum(data_folder)
#     params.spec_0 = last_spectrum
#     params.field_0 = np.fft.ifft(last_spectrum)
#     params.recovery_last_stored = num
#     params.cons_qty = np.load(data_folder / "cons_qty.npy")
#     return params


# def build_sim_grid(
#     length: float,
#     z_num: int,
#     wavelength: float,
#     deg: int,
#     time_window: float = None,
#     t_num: int = None,
#     dt: float = None,
# ) -> tuple[
#     np.ndarray, np.ndarray, float, int, float, np.ndarray, float, np.ndarray, np.ndarray, np.ndarray
# ]:
#     """computes a bunch of values that relate to the simulation grid

#     Parameters
#     ----------
#     length : float
#         length of the fiber in m
#     z_num : int
#         number of spatial points
#     wavelength : float
#         pump wavelength in m
#     deg : int
#         dispersion interpolation degree
#     time_window : float, optional
#         total width of the temporal grid in s, by default None
#     t_num : int, optional
#         number of temporal grid points, by default None
#     dt : float, optional
#         spacing of the temporal grid in s, by default None

#     Returns
#     -------
#     z_targets : np.ndarray, shape (z_num, )
#         spatial points in m
#     t : np.ndarray, shape (t_num, )
#         temporal points in s
#     time_window : float
#         total width of the temporal grid in s, by default None
#     t_num : int
#         number of temporal grid points, by default None
#     dt : float
#         spacing of the temporal grid in s, by default None
#     w_c : np.ndarray, shape (t_num, )
#         centered angular frequencies in rad/s where 0 is the pump frequency
#     w0 : float
#         pump angular frequency
#     w : np.ndarray, shape (t_num, )
#         actual angualr frequency grid in rad/s
#     w_power_fact : np.ndarray, shape (deg, t_num)
#         set of all the necessaray powers of w_c
#     l : np.ndarray, shape (t_num)
#         wavelengths in m
#     """
#     t = tspace(time_window, t_num, dt)

#     time_window = t.max() - t.min()
#     dt = t[1] - t[0]
#     t_num = len(t)
#     z_targets = np.linspace(0, length, z_num)
#     w_c, w0, w, w_power_fact = update_frequency_domain(t, wavelength, deg)
#     l = units.To.m(w)
#     return z_targets, t, time_window, t_num, dt, w_c, w0, w, w_power_fact, l


# def build_sim_grid_in_place(params: BareParams):
#     """similar to calling build_sim_grid, but sets the attributes in place"""
#     (
#         params.z_targets,
#         params.t,
#         params.time_window,
#         params.t_num,
#         params.dt,
#         params.w_c,
#         params.w0,
#         params.w,
#         params.w_power_fact,
#         params.l,
#     ) = build_sim_grid(
#         params.length,
#         params.z_num,
#         params.wavelength,
#         params.interpolation_degree,
#         params.time_window,
#         params.t_num,
#         params.dt,
#     )


# def update_frequency_domain(
#     t: np.ndarray, wavelength: float, deg: int
# ) -> Tuple[np.ndarray, float, np.ndarray, np.ndarray]:
#     """updates the frequency grid

#     Parameters
#     ----------
#     t : np.ndarray
#         time array
#     wavelength : float
#         wavelength
#     deg : int
#         interpolation degree of the dispersion

#     Returns
#     -------
#     Tuple[np.ndarray, float, np.ndarray, np.ndarray]
#         w_c, w0, w, w_power_fact
#     """
#     w_c = wspace(t)
#     w0 = units.m(wavelength)
#     w = w_c + w0
#     w_power_fact = np.array([power_fact(w_c, k) for k in range(2, deg + 3)])
#     return w_c, w0, w, w_power_fact
