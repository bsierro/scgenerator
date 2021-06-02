import os
from collections.abc import Mapping
from typing import Any, Dict, Iterator, List, Set, Tuple, Union

import numpy as np
from numpy import pi
from scipy.interpolate.interpolate import interp1d
from tqdm import tqdm
from pathlib import Path

from . import defaults, io, utils
from .const import hc_model_specific_parameters, valid_param_types, valid_variable
from .errors import *
from .logger import get_logger
from .math import abs2, length, power_fact
from .physics import fiber, pulse, units
from .utils import count_variations, override_config, required_simulations


class ParamSequence(Mapping):
    def __init__(self, config: Union[Dict[str, Any], os.PathLike]):
        if not isinstance(config, Mapping):
            config = io.load_toml(config)
        self.config = validate(config)
        self.name = self.config["name"]
        self.logger = get_logger(__name__)

        self.num_sim, self.num_variable = count_variations(self.config)
        self.num_steps = self.num_sim * self.config["simulation"]["z_num"]
        self.single_sim = self.num_sim == 1

    def __iter__(self) -> Iterator[Tuple[List[Tuple[str, Any]], Dict[str, Any]]]:
        """iterates through all possible parameters, yielding a config as well as a flattened
        computed parameters set each time"""
        for variable_list, full_config in required_simulations(self.config):
            yield variable_list, compute_init_parameters(full_config)

    def __len__(self):
        return self.num_sim

    def __getitem__(self, key):
        return self.config[key[0]][key[1]]

    def __str__(self) -> str:
        return f"dispatcher generated from config {self.name}"


class ContinuationParamSequence(ParamSequence):
    def __init__(self, prev_sim_dir: str, new_config: Dict[str, Any]):
        """Parameter sequence that builds on a previous simulation but with a new configuration
        It is recommended that only the fiber and the number of points stored may be changed and
        changing other parameters could results in unexpected behaviors. The new config doesn't have to
        be a full configuration (i.e. you can specify only the parameters that change).

        Parameters
        ----------
        prev_sim_dir : str
            path to the folder of the previous simulation containing 'initial_config.toml'
        new_config : Dict[str, Any]
            new config
        """
        self.prev_sim_dir = Path(prev_sim_dir)
        init_config = io.load_previous_parameters(
            os.path.join(self.prev_sim_dir, "initial_config.toml")
        )

        self.prev_variable_lists = [
            (set(variable_list[1:]), self.prev_sim_dir / utils.format_variable_list(variable_list))
            for variable_list, _ in required_simulations(init_config)
        ]

        new_config = utils.override_config(new_config, init_config)
        super().__init__(new_config)

    def __iter__(self) -> Iterator[Tuple[List[Tuple[str, Any]], Dict[str, Any]]]:
        """iterates through all possible parameters, yielding a config as well as a flattened
        computed parameters set each time"""
        for variable_list, full_config in required_simulations(self.config):
            prev_data_dir = self.find_prev_data_dir(variable_list).resolve()
            full_config["prev_data_dir"] = str(prev_data_dir)
            yield variable_list, compute_init_parameters(full_config)

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
    def __init__(self, config, task_id):
        super().__init__(config)
        self.id = task_id
        self.num_steps = 0

        z_num = config["simulation"]["z_num"]
        started = self.num_sim
        sub_folders = io.get_data_dirs(io.get_sim_dir(self.id))

        pbar_store = utils.PBars(
            tqdm(
                total=len(sub_folders),
                desc="Initial recovery process",
                unit="sim",
                ncols=100,
            )
        )

        for sub_folder in sub_folders:
            num_left = io.num_left_to_propagate(sub_folder, z_num)
            if num_left == 0:
                self.num_sim -= 1
            self.num_steps += num_left
            started -= 1
            pbar_store.update()

        pbar_store.close()

        self.num_steps += started * z_num
        self.single_sim = self.num_sim == 1

        self.prev_sim_dir = None
        if "prev_sim_dir" in self.config.get("simulation", {}):
            self.prev_sim_dir = Path(self.config["simulation"]["prev_sim_dir"])
            init_config = io.load_previous_parameters(
                os.path.join(self.prev_sim_dir, "initial_config.toml")
            )
            self.prev_variable_lists = [
                (
                    set(variable_list[1:]),
                    self.prev_sim_dir / utils.format_variable_list(variable_list),
                )
                for variable_list, _ in required_simulations(init_config)
            ]

    def __iter__(self) -> Iterator[Tuple[List[Tuple[str, Any]], dict]]:
        for variable_list, params in required_simulations(self.config):

            data_dir = io.get_sim_dir(self.id) / utils.format_variable_list(variable_list)

            if not data_dir.is_dir() or io.find_last_spectrum_num(data_dir) == 0:
                if (prev_data_dir := self.find_prev_data_dir(variable_list)) is not None:
                    params["prev_data_dir"] = str(prev_data_dir)
                yield variable_list, compute_init_parameters(params)
            elif io.num_left_to_propagate(data_dir, self.config["simulation"]["z_num"]) != 0:
                yield variable_list, recover_params(params, data_dir)
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


def validate(config: dict) -> dict:
    """validates a configuration dictionary and attempts to fill in defaults

    Parameters
    ----------
    config : dict
        loaded configuration

    Returns
    -------
    dict
        updated configuration
    """
    _validate_types(config)
    return _ensure_consistency(config)


def validate_config_sequence(*configs: os.PathLike) -> Dict[str, Any]:
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
        previous = override_config(dico, previous)
        validate(previous)
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


def validate_single_parameter(section: str, key: str, value: Any):
    try:
        func = valid_param_types[section][key]
    except KeyError:
        s = f"The parameter '{key}' does not belong "
        if section == "root":
            s += "at the root of the config file"
        else:
            s += f"in the category '{section}'"
        s += ". Make sure it is a valid parameter in the first place"
        raise TypeError(s)
    if not func(value):
        raise TypeError(
            f"value '{value}' of type {type(value).__name__} for key '{key}' is not valid, {func.__doc__}"
        )
    return


def _validate_types(config):
    """validates the data types in the initial config dictionary

    Parameters
    ----------
    config : dict
        the initial config dictionary

    Raises
    ------
    TypeError
        raised when a parameter has the wrong type
    """

    for domain, parameters in config.items():
        if isinstance(parameters, dict):
            for param_name, param_value in parameters.items():
                if param_name == "variable":
                    for k_vary, v_vary in param_value.items():
                        if not isinstance(v_vary, list):
                            raise TypeError(f"Variable parameters should be specified in a list")

                        if len(v_vary) < 1:
                            raise ValueError(
                                f"Variable parameters lists should contain at least 1 element"
                            )

                        if k_vary not in valid_variable[domain]:
                            raise TypeError(f"'{k_vary}' is not a valid variable parameter")

                        [
                            validate_single_parameter(domain, k_vary, v_vary_indiv)
                            for v_vary_indiv in v_vary
                        ]
                else:
                    validate_single_parameter(domain, param_name, param_value)
        else:
            validate_single_parameter("root", domain, parameters)


def _contains(sub_conf, param):
    return param in sub_conf or param in sub_conf.get("variable", {})


def _ensure_consistency_fiber(fiber: Dict[str, Any]):
    """ensure the fiber sub-dictionary of the parameter set is consistent

    Parameters
    ----------
    fiber : dict
        dictionary containing the fiber parameters

    Returns
    -------
    dict
        the updated dictionary

    Raises
    ------
    MissingParameterError
        When at least one required parameter with no default is missing
    """

    if _contains(fiber, "beta"):
        if not (_contains(fiber, "A_eff") or _contains(fiber, "effective_mode_diameter")):
            fiber = defaults.get(fiber, "gamma", specified_parameters=["beta"])
        fiber.setdefault("model", "custom")

    elif _contains(fiber, "dispersion_file"):
        if not (_contains(fiber, "A_eff") or _contains(fiber, "effective_mode_diameter")):
            fiber = defaults.get(fiber, "gamma", specified_parameters=["dispersion_file"])
        fiber.setdefault("model", "custom")

    else:
        fiber = defaults.get(fiber, "model")

        if fiber["model"] == "pcf":
            fiber = defaults.get_fiber(fiber, "pitch")
            fiber = defaults.get_fiber(fiber, "pitch_ratio")

        elif fiber["model"] == "hasan":
            fiber = defaults.get_multiple(
                fiber, ["capillary_spacing", "capillary_outer_d"], 1, fiber_model="hasan"
            )
            for param in [
                "core_radius",
                "capillary_num",
                "capillary_thickness",
                "capillary_resonance_strengths",
                "capillary_nested",
            ]:
                fiber = defaults.get_fiber(fiber, param)
        else:
            for param in hc_model_specific_parameters[fiber["model"]]:
                fiber = defaults.get_fiber(fiber, param)
    for param in ["length", "input_transmission"]:
        fiber = defaults.get(fiber, param)
    return fiber


def _ensure_consistency_gas(gas):
    """ensure the gas sub-dictionary of the parameter set is consistent

    Parameters
    ----------
    gas : dict
        dictionary containing the gas parameters

    Returns
    -------
    dict
        the updated dictionary

    Raises
    ------
    MissingParameterError
        When at least one required parameter with no default is missing
    """
    for param in ["gas_name", "temperature", "pressure", "plasma_density"]:
        gas = defaults.get(gas, param, specified_params=["gas"])
    return gas


def _ensure_consistency_pulse(pulse):
    """ensure the pulse sub-dictionary of the parameter set is consistent

    Parameters
    ----------
    pulse : dict
        dictionary of the pulse section of parameters

    Returns
    -------
    dict
        the updated pulse dictionary

    Raises
    ------
    MissingParameterError
        When at least one required parameter with no default is missing
    """
    for param in ["wavelength", "quantum_noise", "intensity_noise"]:
        pulse = defaults.get(pulse, param)

    if not _contains(pulse, "field_file"):
        pulse = defaults.get(pulse, "shape")

        if _contains(pulse, "soliton_num"):
            pulse = defaults.get_multiple(
                pulse,
                ["peak_power", "mean_power", "energy", "width", "t0"],
                1,
                specified_parameters=["soliton_num"],
            )

        else:
            pulse = defaults.get_multiple(pulse, ["t0", "width"], 1)
            pulse = defaults.get_multiple(pulse, ["peak_power", "energy", "mean_power"], 1)
    if _contains(pulse, "mean_power"):
        pulse = defaults.get(pulse, "repetition_rate", specified_parameters=["mean_power"])
    return pulse


def _ensure_consistency_simulation(simulation):
    """ensure the simulation sub-dictionary of the parameter set is consistent

    Parameters
    ----------
    pulse : dict
        dictionary of the pulse section of parameters

    Returns
    -------
    dict
        the updated pulse dictionary

    Raises
    ------
    MissingParameterError
        When at least one required parameter with no default is missing
    """
    simulation = defaults.get_multiple(simulation, ["dt", "t_num", "time_window"], 2)

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
    ]:
        simulation = defaults.get(simulation, param)

    if "raman" in simulation.get("behaviors", {}) or any(
        ["raman" in l for l in simulation.get("variable", {}).get("behaviors", [])]
    ):
        simulation = defaults.get(simulation, "raman_type", specified_parameters=["raman"])
    return simulation


def _ensure_consistency(config):
    """ensure the config dictionary is consistent and that certain parameters are set,
    either by filling in defaults or by raising an error. This is not where new values are calculated.

    Parameters
    ----------
    config : dict
        original config dict loaded from the toml file

    Returns
    -------
    dict
        the consistent config dict
    """

    _validate_types(config)

    # ensure parameters are not specified multiple times
    for sub_dict in valid_param_types.values():
        for param_name in sub_dict:
            for set_param in config.values():
                if isinstance(set_param, dict):
                    if param_name in set_param and param_name in set_param.get("variable", {}):
                        raise DuplicateParameterError(
                            f"got multiple values for parameter '{param_name}'"
                        )

    # ensure every required parameter has a value
    config["name"] = config.get("name", "no name")

    config["fiber"] = _ensure_consistency_fiber(config.get("fiber", {}))

    if config["fiber"]["model"] in hc_model_specific_parameters:
        config["gas"] = _ensure_consistency_gas(config.get("gas", {}))

    config["pulse"] = _ensure_consistency_pulse(config.get("pulse", {}))
    config["simulation"] = _ensure_consistency_simulation(config.get("simulation", {}))

    return config


def recover_params(config: Dict[str, Any], data_folder: Path) -> Dict[str, Any]:
    params = compute_init_parameters(config)
    try:
        prev_params = io.load_previous_parameters(data_folder / "params.toml")
        prev_params = build_sim_grid(prev_params)
    except FileNotFoundError:
        prev_params = {}
    for k, v in prev_params.items():
        params.setdefault(k, v)
    num, last_spectrum = io.load_last_spectrum(data_folder)
    params["spec_0"] = last_spectrum
    params["field_0"] = np.fft.ifft(last_spectrum)
    params["recovery_last_stored"] = num
    params["cons_qty"] = np.load(data_folder / "cons_qty.npy")
    return params


def compute_init_parameters(config: Dict[str, Any]) -> Dict[str, Any]:
    """computes all derived values from a config dictionary

    Parameters
    ----------
    config : dict
        a configuration dictionary containing the pulse, fiber and simulation sections with no variable parameter.
        a flattened parameters dictionary may be provided instead
        Note : checking the validity of the configuration shall be done before calling this function.

    Returns
    -------
    dict
        a flattened dictionary (no fiber, pulse, simulation subsections) with all the necessary values to run RK4IP
    """

    logger = get_logger(__name__)

    # copy and flatten the config
    params = {k: v for k, v in config.items() if isinstance(v, (str, int, float))}
    for section in ["pulse", "fiber", "simulation", "gas"]:
        for key, value in config.get(section, {}).items():
            params[key] = value

    params = build_sim_grid(params)

    # Initial field may influence the grid
    if "mean_power" in params:
        params["energy"] = params["mean_power"] / params["repetition_rate"]
    custom_field = setup_custom_field(params)

    if "step_size" in params:
        params["error_ok"] = params["step_size"]
        params["adapt_step_size"] = False
    else:
        params["error_ok"] = params["tolerated_error"]
        params["adapt_step_size"] = True

    # FIBER
    params["interp_range"] = _interp_range(
        params["w"],
        params["upper_wavelength_interp_limit"],
        params["lower_wavelength_interp_limit"],
    )

    temp_gamma = None
    if "effective_mode_diameter" in params:
        params["A_eff"] = (params["effective_mode_diameter"] / 2) ** 2 * pi
    if "beta" in params:
        params["beta"] = np.array(params["beta"])
        params["dynamic_dispersion"] = False
    else:
        params["dynamic_dispersion"] = fiber.is_dynamic_dispersion(params)
        params["beta"], temp_gamma = fiber.dispersion_central(params["model"], params)
        if params["dynamic_dispersion"]:
            params["gamma_func"] = temp_gamma
            params["beta_func"] = params["beta"]
            params["beta"] = params["beta_func"](0)
            temp_gamma = temp_gamma(0)

    if "gamma" not in params:
        params["gamma"] = temp_gamma
        logger.info(f"using computed \u0263 = {params['gamma']:.2e} W/m^2")

    # Raman response
    if "raman" in params["behaviors"]:
        params["hr_w"] = fiber.delayed_raman_w(params["t"], params["dt"], params["raman_type"])

    # GENERIC PULSE
    if not custom_field:
        custom_field = False
        params = _update_pulse_parameters(params)
        logger.info(f"computed initial N = {params['soliton_num']:.3g}")

        params["L_D"] = params["t0"] ** 2 / abs(params["beta"][0])
        params["L_NL"] = 1 / (params["gamma"] * params["peak_power"]) if params["gamma"] else np.inf
        params["L_sol"] = pi / 2 * params["L_D"]

        # Technical noise
        if "intensity_noise" in params:
            params = _technical_noise(params)

        params["field_0"] = pulse.initial_field(
            params["t"], params["shape"], params["t0"], params["peak_power"]
        )

    if params["quantum_noise"]:
        params["field_0"] = params["field_0"] + pulse.shot_noise(
            params["w_c"], params["w0"], params["time_window"], params["dt"]
        )

    params["spec_0"] = np.fft.fft(params["field_0"])

    return params


def setup_custom_field(params: Dict[str, Any]) -> bool:
    """sets up a custom field function if necessary and returns
    True if it did so, False otherwise

    Parameters
    ----------
    params : Dict[str, Any]
        params dictionary

    Returns
    -------
    bool
        True if the field has been modified
    """
    logger = get_logger(__name__)
    if "prev_data_dir" in params:
        spec = io.load_last_spectrum(Path(params["prev_data_dir"]))[1]
        params["field_0"] = np.fft.ifft(spec) * np.sqrt(params["input_transmission"])
    else:
        if "field_file" in params:
            field_data = np.load(params["field_file"])
            field_interp = interp1d(
                field_data["time"], field_data["field"], bounds_error=False, fill_value=(0, 0)
            )
            params["field_0"] = field_interp(params["t"])
        elif "field_0" in params:
            params = _evalutate_custom_field_equation(params)
        else:
            return False

        params["field_0"] = params["field_0"] * pulse.modify_field_ratio(
            params["t"],
            params["field_0"],
            params.get("peak_power"),
            params.get("energy"),
            params.get("intensity_noise"),
        )
        params["width"], params["peak_power"], params["energy"] = pulse.measure_field(
            params["t"], params["field_0"]
        )
    delta_w = params["w_c"][np.argmax(abs2(np.fft.fft(params["field_0"])))]
    logger.debug(f"adjusted w by {delta_w}")
    params["wavelength"] = units.m.inv(units.m(params["wavelength"]) - delta_w)
    _update_frequency_domain(params)
    return True


def _update_pulse_parameters(params):
    (
        params["width"],
        params["t0"],
        params["peak_power"],
        params["energy"],
        params["soliton_num"],
    ) = pulse.conform_pulse_params(
        shape=params["shape"],
        width=params.get("width", None),
        t0=params.get("t0", None),
        peak_power=params.get("peak_power", None),
        energy=params.get("energy", None),
        gamma=params["gamma"],
        beta2=params["beta"][0],
    )
    return params


def _evalutate_custom_field_equation(params):
    field_info = params["field_0"]
    if isinstance(field_info, str):
        field_0 = eval(
            field_info,
            dict(
                sin=np.sin,
                cos=np.cos,
                tan=np.tan,
                exp=np.exp,
                pi=np.pi,
                sqrt=np.sqrt,
                **params,
            ),
        )

        params["field_0"] = field_0
    elif len(field_info) != params["t_num"]:
        raise ValueError(
            "initial field is given but doesn't match size and type with the time array"
        )
    return params


def _technical_noise(params):
    logger = get_logger(__name__)

    if params["intensity_noise"] > 0:
        logger.info(f"intensity noise of {params['intensity_noise']}")
        delta_int, delta_T0 = pulse.technical_noise(params["intensity_noise"])
        params["peak_power"] *= delta_int
        params["t0"] *= delta_T0
        params["width"] *= delta_T0
        params = _update_pulse_parameters(params)
    return params


def _interp_range(w, upper, lower):
    # by default, the interpolation range of the dispersion polynomial stops exactly
    # at the boundary of the frequency window we consider

    interp_range = [
        max(lower, units.m.inv(np.max(w[w > 0]))),
        min(upper, units.m.inv(np.min(w[w > 0]))),
    ]

    return interp_range


def build_sim_grid(params):
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
    t = params.get(
        "t",
        tspace(
            time_window=params.get("time_window", None),
            t_num=params.get("t_num", None),
            dt=params.get("dt", None),
        ),
    )
    params["t"] = t
    params["time_window"] = length(t)
    params["dt"] = t[1] - t[0]
    params["t_num"] = len(t)
    params["z_targets"] = np.linspace(0, params["length"], params["z_num"])
    params = _update_frequency_domain(params)
    return params


def _update_frequency_domain(params):
    w_c = wspace(params["t"])
    w0 = units.m(params["wavelength"])
    params["w0"] = w0
    params["w_c"] = w_c
    params["w"] = w_c + w0
    params["w_power_fact"] = np.array([power_fact(w_c, k) for k in range(2, 11)])
    return params


def sanitize_z_targets(z_targets):
    """
    processes the 'z_targets' arguments and guarantees that:
        - it is sorted
        - it doesn't contain the same value twice
        - it starts with 0
    Parameters
    ----------
        z_targets : float, int or array-like
            float or int : end point of the fiber starting from 0
            array-like of len(.) == 3 : `numpy.linspace` arguments
            array-like of other length : target distances at which to store the spectra
    Returns
    ----------
        z_targets : list (mutability is important)
    """
    if isinstance(z_targets, (float, int)):
        z_targets = np.linspace(0, z_targets, defaults.default_parameters["length"])
    else:
        z_targets = np.array(z_targets).flatten()

    if len(z_targets) == 3:
        z_targets = np.linspace(*z_targets[:2], int(z_targets[2]))

    z_targets = list(set(value for value in z_targets if value >= 0))
    z_targets.sort()

    if 0 not in z_targets:
        z_targets = [0] + z_targets

    return z_targets
