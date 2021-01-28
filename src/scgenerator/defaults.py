import numpy as np

from .errors import MissingParameterError

default_parameters = dict(
    name="no name",
    he_mode=(1, 1),
    fit_parameters=(0.08, 200e-9),
    model="pcf",
    length=1,
    capillary_resonance_strengths=[],
    capillary_nested=0,
    gas_name="vacuum",
    plasma_density=0,
    pressure=1e5,
    temperature=300,
    quantum_noise=False,
    intensity_noise=0,
    shape="gaussian",
    frep=80e6,
    behaviors=["spm", "ss"],
    raman_type="agrawal",
    parallel=1,
    repeat=1,
    tolerated_error=1e-11,
    lower_wavelength_interp_limit=0,
    upper_wavelength_interp_limit=1900e-9,
    ideal_gas=False,
)


def get(section_dict, param, **kwargs):
    """checks if param is in the parameter section dict and attempts to fill in a default value

    Parameters
    ----------
    section_dict : dict
        the parameters section {fiber, pulse, simulation, root} sub-dictionary
    param : str
        the name of the parameter (dict key)
    kwargs : any
        key word arguments passed to the MissingParameterError constructor

    Returns
    -------
    dict
        the updated section_dict dictionary

    Raises
    ------
    MissingFiberParameterError
        raised when a parameter is missing and no default exists
    """

    # whether the parameter is in the right place and valid is checked elsewhere,
    # here, we just make sure it is present.
    if param not in section_dict and param not in section_dict.get("varying", {}):
        try:
            section_dict[param] = default_parameters[param]
            # LOG
        except KeyError:
            raise MissingParameterError(param, **kwargs)
    return section_dict


def get_fiber(section_dict, param, **kwargs):
    """wrapper for fiber parameters that depend on fiber model"""
    return get(section_dict, param, fiber_model=section_dict["model"], **kwargs)


def get_multiple(section_dict, params, num, **kwargs):
    """similar to th get method but works with several parameters

    Parameters
    ----------
    section_dict : dict
        the parameters section {fiber, pulse, simulation, root}, sub-dictionary
    params : list of str
        names of the required parameters
    num : int
        how many of the parameters in params are required

    Returns
    -------
    dict
        the updated section_dict

    Raises
    ------
    MissingParameterError
        raised when not enough parameters are provided and no defaults exist
    """
    gotten = 0
    for param in params:
        try:
            section_dict = get(section_dict, param, **kwargs)
            gotten += 1
        except MissingParameterError:
            pass
        if gotten >= num:
            return section_dict
    raise MissingParameterError(params, num_required=num, **kwargs)
