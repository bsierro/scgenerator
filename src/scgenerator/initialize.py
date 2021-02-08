import os
from collections.abc import Mapping
from typing import Any, Iterator, List, Tuple

import numpy as np
from numpy import pi

from . import defaults, io, utils
from .const import hc_model_specific_parameters, valid_param_types, valid_variable
from .errors import *
from .logger import get_logger
from .math import length, power_fact
from .physics import fiber, pulse, units
from .utils import count_variations, required_simulations


class ParamSequence(Mapping):
    def __init__(self, config):
        self.config = validate(config)
        self.name = self.config["name"]

        self.num_sim, self.num_variable = count_variations(self.config)
        self.single_sim = self.num_sim == 1

    def __iter__(self) -> Iterator[Tuple[List[Tuple[str, Any]], dict]]:
        """iterates through all possible parameters, yielding a config as welle as a flattened
        computed parameters set each time"""
        for variable_list, full_config in required_simulations(self.config):
            yield variable_list, compute_init_parameters(full_config)

    def __len__(self):
        return self.num_sim

    def __getitem__(self, key):
        return self.config[key[0]][key[1]]

    def __str__(self) -> str:
        return f"dispatcher generated from config {self.name}"


class RecoveryParamSequence(ParamSequence):
    def __init__(self, config, task_id):
        super().__init__(config)
        self.id = task_id
        for sub_folder in io.get_data_subfolders(io.get_data_folder(self.id)):
            if io.propagation_completed(sub_folder, config["simulation"]["z_num"]):
                self.num_sim -= 1
        self.single_sim = self.num_sim == 1

    def __iter__(self) -> Iterator[Tuple[List[Tuple[str, Any]], dict]]:
        for variable_list, full_config in required_simulations(self.config):

            sub_folder = os.path.join(
                io.get_data_folder(self.id), utils.format_variable_list(variable_list)
            )

            if not io.propagation_initiated(sub_folder):
                yield variable_list, compute_init_parameters(full_config)
            elif not io.propagation_completed(sub_folder, self.config["simulation"]["z_num"]):
                yield variable_list, recover_params(full_config, variable_list, self.id)
            else:
                continue


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


def validate_single_parameter(section, key, value):
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
            f"value '{value}' of type {type(value)} for key '{key}' is not valid, {func.__doc__}"
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


def _ensure_consistency_fiber(fiber):
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
        fiber = defaults.get(fiber, "gamma", specified_parameters=["beta"])
        fiber["model"] = fiber.get("model", "custom")

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

    fiber = defaults.get(fiber, "length")
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
    for param in ["wavelength", "shape", "quantum_noise", "intensity_noise"]:
        pulse = defaults.get(pulse, param)

    if _contains(pulse, "soliton_num"):
        pulse = defaults.get_multiple(
            pulse, ["power", "energy", "width", "t0"], 1, specified_parameters=["soliton_num"]
        )

    else:
        pulse = defaults.get_multiple(pulse, ["t0", "width"], 1)
        pulse = defaults.get_multiple(pulse, ["power", "energy"], 1)
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


def recover_params(params: dict, variable_only: List[Tuple[str, Any]], task_id: int):
    params = compute_init_parameters(params)
    vary_str = utils.format_variable_list(variable_only)
    path = os.path.join(io.get_data_folder(task_id), vary_str)
    num, last_spectrum = io.load_last_spectrum(path)
    params["spec_0"] = last_spectrum
    params["field_0"] = np.fft.ifft(last_spectrum)
    params["recovery_last_stored"] = num
    params["cons_qty"] = np.load(os.path.join(path, "cons_qty.npy"))
    return params


def compute_init_parameters(config):
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
    params = dict(name=config["name"])
    for section in ["pulse", "fiber", "simulation", "gas"]:
        for key, value in config.get(section, {}).items():
            params[key] = value

    params = _generate_sim_grid(params)
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

    if "beta" in params:
        params["beta"] = np.array(params["beta"])
        temp_gamma = 0
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

    # PULSE
    params = _update_pulse_parameters(params)
    logger.info(f"computed initial N = {params['soliton_num']:.3g}")

    params["L_D"] = params["t0"] ** 2 / abs(params["beta"][0])
    params["L_NL"] = 1 / (params["gamma"] * params["power"]) if params["gamma"] else np.inf
    params["L_sol"] = pi / 2 * params["L_D"]

    # Technical noise
    if "intensity_noise" in params:
        params = _technical_noise(params)

    # Initial field
    if "field_0" in params:
        params = _validate_custom_init_field(params)
    else:
        params["field_0"] = pulse.initial_field(
            params["t"], params["shape"], params["t0"], params["power"]
        )

    if params["quantum_noise"]:
        params["field_0"] = params["field_0"] + pulse.shot_noise(
            params["w_c"], params["w0"], params["time_window"], params["dt"]
        )

    params["spec_0"] = np.fft.fft(params["field_0"])

    return params


def _update_pulse_parameters(params):
    (
        params["width"],
        params["t0"],
        params["power"],
        params["energy"],
        params["soliton_num"],
    ) = pulse.conform_pulse_params(
        shape=params["shape"],
        width=params.get("width", None),
        t0=params.get("t0", None),
        power=params.get("power", None),
        energy=params.get("energy", None),
        gamma=params["gamma"],
        beta2=params["beta"][0],
    )
    return params


def _validate_custom_init_field(params):
    if isinstance(params["field_0"], str):
        field_0 = evaluate_field_equation(params["field_0"], **params)
        params["field_0"] = field_0
    elif len(params["field_0"]) != params["t_num"]:
        raise ValueError(
            "initial field is given but doesn't match size and type with the time array"
        )
    return params


def _technical_noise(params):
    logger = get_logger(__name__)

    if params["intensity_noise"] > 0:
        logger.info(f"intensity noise of {params['intensity_noise']}")
        delta_int, delta_T0 = pulse.technical_noise(params["intensity_noise"])
        params["power"] *= delta_int
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


def _generate_sim_grid(params):
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

    w_c = wspace(t)
    w0 = units.m(params["wavelength"])
    params["w0"] = w0
    params["w_c"] = w_c
    params["w"] = w_c + w0
    params["w_power_fact"] = [power_fact(w_c, k) for k in range(2, 11)]

    params["z_targets"] = np.linspace(0, params["length"], params["z_num"])

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


def evaluate_field_equation(eq, **kwargs):
    return eval(
        eq,
        dict(
            sin=np.sin,
            cos=np.cos,
            tan=np.tan,
            exp=np.exp,
            pi=np.pi,
            sqrt=np.sqrt,
            **kwargs,
        ),
    )
