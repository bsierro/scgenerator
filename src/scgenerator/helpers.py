"""
series of helper functions
"""

from scgenerator.physics.materials import n_gas_2
from scgenerator.physics.fiber import n_eff_marcatili, beta2, beta2_to_D
from scgenerator.physics.units import c
import numpy as np

__all__ = ["capillary_dispersion"]


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


def extend_axis(wl):
    dwl_left = wl[1] - wl[0]
    dwl_right = wl[-1] - wl[-2]
    wl = np.concatenate(
        ([wl[0] - 2 * dwl_left, wl[0] - dwl_left], wl, [wl[-1] + dwl_right, wl[-1] + 2 * dwl_right])
    )

    return wl
