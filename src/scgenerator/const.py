import numpy as np


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


# def find_parent(param):
#     """find the parent dictionary name of param"""
#     for k, v in valid_param_types.items():
#         if param in v:
#             return k
#     raise ValueError(f"'{param}' is an invalid parameter name")


valid_param_types = dict(
    root=dict(
        name=lambda s: isinstance(s, str),
    ),
    fiber=dict(
        gamma=num,
        pitch=num,
        pitch_ratio=num,
        core_radius=num,
        he_mode=he_mode,
        fit_parameters=fit_parameters,
        beta=beta,
        model=string(["pcf", "marcatili", "marcatili_adjusted", "hasan", "custom"]),
        length=num,
        capillary_num=integer,
        capillary_outer_d=num,
        capillary_thickness=num,
        capillary_spacing=num,
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
        wavelength=num,
        intensity_noise=num,
        width=num,
        t0=num,
    ),
    simulation=dict(
        behaviors=behaviors,
        parallel=integer,
        raman_type=string(["measured", "agrawal", "stolen"]),
        ideal_gas=boolean,
        repeat=integer,
        t_num=integer,
        z_num=integer,
        time_window=num,
        dt=num,
        tolerated_error=num,
        step_size=num,
        lower_wavelength_interp_limit=num,
        upper_wavelength_interp_limit=num,
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

valid_varying = dict(
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


TMP_FOLDER_KEY_BASE = "SCGENERATOR_TMP"
PARAM_SEPARATOR = " "
