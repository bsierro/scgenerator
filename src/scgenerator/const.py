import numpy as np


def in_range_excl(func, r):
    def _in_range(n):
        if not func(n):
            return False
        return n > r[0] and n < r[1]

    _in_range.__doc__ = func.__doc__ + f" between {r[0]} and {r[1]}"
    return _in_range


def in_range_incl(func, r):
    def _in_range(n):
        if not func(n):
            return False
        return n >= r[0] and n <= r[1]

    _in_range.__doc__ = func.__doc__ + f" between {r[0]} and {r[1]}"
    return _in_range


def num(n):
    """must be a single, real, non-negative number"""
    return isinstance(n, (float, int)) and n >= 0


def integer(n):
    """must be a strictly positive integer"""
    return isinstance(n, int) and n > 0


def boolean(b):
    """must be a boolean"""
    return type(b) == bool


def behaviors(l):
    """must be a valid list of behaviors"""
    for s in l:
        if s.lower() not in ["spm", "raman", "ss"]:
            return False
    return True


def beta(l):
    """must be a valid beta array"""
    for n in l:
        if not isinstance(n, (float, int)):
            return False
    return True


def field_0(f):
    return isinstance(f, (str, tuple, list, np.ndarray))


def he_mode(mode):
    """must be a valide HE mode"""
    if not isinstance(mode, (list, tuple)):
        return False
    if not len(mode) == 2:
        return False
    for m in mode:
        if not integer(m):
            return False
    return True


def fit_parameters(param):
    """must be a valide fitting parameter tuple of the mercatili_adjusted model"""
    if not isinstance(param, (list, tuple)):
        return False
    if not len(param) == 2:
        return False
    for n in param:
        if not integer(n):
            return False
    return True


def string(l):
    def _string(s):
        return isinstance(s, str) and s.lower() in l

    _string.__doc__ = f"must be a str matching one of {l}"
    return _string


def capillary_resonance_strengths(l):
    """must be a list of non-zero, real number"""
    if not isinstance(l, (list, tuple)):
        return False
    for m in l:
        if not num(m):
            return False
    return True


def capillary_nested(n):
    """must be a non negative integer"""
    return isinstance(n, int) and n >= 0


valid_param_types = dict(
    root=dict(
        name=lambda s: isinstance(s, str),
    ),
    fiber=dict(
        gamma=num,
        pitch=in_range_excl(num, (0, 1e-3)),
        pitch_ratio=in_range_excl(num, (0, 1)),
        core_radius=in_range_excl(num, (0, 1e-3)),
        he_mode=he_mode,
        fit_parameters=fit_parameters,
        beta=beta,
        model=string(["pcf", "marcatili", "marcatili_adjusted", "hasan", "custom"]),
        length=num,
        capillary_num=integer,
        capillary_outer_d=in_range_excl(num, (0, 1e-3)),
        capillary_thickness=in_range_excl(num, (0, 1e-3)),
        capillary_spacing=in_range_excl(num, (0, 1e-3)),
        capillary_resonance_strengths=capillary_resonance_strengths,
        capillary_nested=capillary_nested,
    ),
    gas=dict(
        gas_name=string(["vacuum", "helium", "air"]),
        pressure=num,
        temperature=num,
        plasma_density=num,
    ),
    pulse=dict(
        field_0=field_0,
        power=num,
        energy=num,
        soliton_num=num,
        quantum_noise=boolean,
        shape=string(["gaussian", "sech"]),
        wavelength=in_range_excl(num, (100e-9, 3000e-9)),
        intensity_noise=in_range_incl(num, (0, 1)),
        width=in_range_excl(num, (0, 1e-9)),
        t0=in_range_excl(num, (0, 1e-9)),
    ),
    simulation=dict(
        behaviors=behaviors,
        parallel=boolean,
        raman_type=string(["measured", "agrawal", "stolen"]),
        ideal_gas=boolean,
        repeat=integer,
        t_num=integer,
        z_num=integer,
        time_window=num,
        dt=in_range_excl(num, (0, 5e-15)),
        tolerated_error=in_range_excl(num, (1e-15, 1e-5)),
        step_size=num,
        lower_wavelength_interp_limit=in_range_excl(num, (100e-9, 3000e-9)),
        upper_wavelength_interp_limit=in_range_excl(num, (100e-9, 5000e-9)),
        frep=num,
    ),
)

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

valid_variable = dict(
    fiber=[
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
    ],
    gas=["pressure", "temperature", "gas_name", "plasma_density"],
    pulse=[
        "power",
        "quantum_noise",
        "shape",
        "wavelength",
        "intensity_noise",
        "width",
        "soliton_num",
    ],
    simulation=["behaviors", "raman_type", "tolerated_error", "step_size", "ideal_gas"],
)

ENVIRON_KEY_BASE = "SCGENERATOR_"
TMP_FOLDER_KEY_BASE = ENVIRON_KEY_BASE + "TMP_"
PREFIX_KEY_BASE = ENVIRON_KEY_BASE + "PREFIX_"
PARAM_SEPARATOR = " "
