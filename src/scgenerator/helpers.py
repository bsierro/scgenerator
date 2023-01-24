"""
series of helper functions
"""

import numpy as np

from scgenerator.math import all_zeros
from scgenerator.physics.fiber import beta2, n_eff_marcatili, n_eff_hasan
from scgenerator.physics.materials import n_gas_2
from scgenerator.physics.units import c

__all__ = ["capillary_dispersion", "capillary_zdw", "revolver_dispersion"]


def capillary_dispersion(
    wl: np.ndarray, radius: float, gas_name: str, pressure=None, temperature=None
) -> np.ndarray:
    """computes the dispersion (beta2) of a capillary tube

    Parameters
    ----------
    wl : np.ndarray
        wavelength in m
    radius : float
        core radius in m
    gas_name : str
        gas name (case insensitive)
    pressure : float, optional
        pressure in Pa (multiply mbar by 100 to get Pa), by default atm pressure
    temperature : float, optional
        temperature in K, by default 20°C

    Returns
    -------
    np.ndarray
        D parameter
    """
    wl = extend_axis(wl)
    if pressure is None:
        pressure = 101325
    if temperature is None:
        temperature = 293.15
    n = n_eff_marcatili(wl, n_gas_2(wl, gas_name.lower(), pressure, temperature, False), radius)
    w = 2 * np.pi * c / wl
    return beta2(w, n)[2:-2]


def capillary_zdw(
    radius: float,
    gas_name: str,
    pressure=None,
    temperature=None,
    search_range: tuple[float, float] = (200e-9, 3000e-9),
) -> np.ndarray:
    """find the zero dispersion wavelength of a capilally

    Parameters
    ----------
    radius : float
        in mnm
    gas_name : str
        gas name (case insensitive)
    pressure : float, optional
        pressure in Pa (multiply mbar by 100 to get Pa), by default atm pressure
    temperature : float, optional
        temperature in K, by default 20°C
    search_range : (float, float), optional
        range of wavelength (in m) in which to search for the ZDW, by default (200e-9, 3000e-9)

    Returns
    -------
    np.ndarray
        array of zero dispersion wavelength(s)
    """
    wl = np.linspace(*search_range[:2], 1024)
    disp = capillary_dispersion(wl, radius, gas_name, pressure, temperature)
    return all_zeros(wl, disp)


def revolver_dispersion(
    wl: np.ndarray,
    core_radius: float,
    gas_name: str,
    capillary_num: int,
    capillary_thickness: float,
    capillary_spacing: float,
    capillary_nested: int = 0,
    capillary_resonance_strengths: list[float] = None,
    pressure=None,
    temperature=None,
) -> np.ndarray:
    """computes the dispersion (beta2) of a capillary tube

    Parameters
    ----------
    wl : np.ndarray
        wavelength in m
    radius : float
        core radius in m
    gas_name : str
        gas name (case insensitive)
    pressure : float, optional
        pressure in Pa (multiply mbar by 100 to get Pa), by default atm pressure
    temperature : float, optional
        temperature in K, by default 20°C

    Returns
    -------
    np.ndarray
        D parameter
    """
    capillary_resonance_strengths = capillary_resonance_strengths or []
    wl = extend_axis(wl)
    if pressure is None:
        pressure = 101325
    if temperature is None:
        temperature = 293.15
    n = n_eff_hasan(
        wl,
        n_gas_2(wl, gas_name.lower(), pressure, temperature, False),
        core_radius,
        capillary_num,
        capillary_nested,
        capillary_thickness,
        capillary_spacing,
        capillary_resonance_strengths,
    )
    w = 2 * np.pi * c / wl
    return beta2(w, n)[2:-2]


def extend_axis(axis: np.ndarray) -> np.ndarray:
    """add 4 values to an array, 2 on each 'side'"""
    dwl_left = axis[1] - axis[0]
    dwl_right = axis[-1] - axis[-2]
    axis = np.concatenate(
        (
            [axis[0] - 2 * dwl_left, axis[0] - dwl_left],
            axis,
            [axis[-1] + dwl_right, axis[-1] + 2 * dwl_right],
        )
    )

    return axis
