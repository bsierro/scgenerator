"""
This file contains functions that don't properly fit into the 'fiber' or 'pulse' category.
They're also not necessarily 'pure' function as they do some io and stuff.
"""

from typing import TypeVar

import numpy as np
from scipy.optimize import minimize_scalar

from .. import math
from . import fiber, materials, units, pulse
from .._utils import cache, load_material_dico

T = TypeVar("T")


def group_delay_to_gdd(wavelength: np.ndarray, group_delay: np.ndarray) -> np.ndarray:
    w = units.m.inv(wavelength)
    gdd = np.gradient(group_delay, w)
    return gdd


@cache.np_cache
def material_dispersion(
    wavelengths: np.ndarray,
    material: str,
    pressure=None,
    temperature=None,
    ideal=True,
    safe=True,
):
    """returns the dispersion profile (beta_2) of a bulk material.

    Parameters
    ----------
    wavelengths : ndarray, shape (n, )
        wavelengths over which to calculate the dispersion
    material : str
        material name in lower case
    temperature : float, optional
        Temperature of the material
    pressure : float, optional
        constant pressure
    ideal : bool, optional
        whether to use the ideal gas law instead of the van der Waals equation, by default True

    Returns
    -------
    out : ndarray, shape (n, )
        beta2 as function of wavelength
    """

    w = units.m(wavelengths)

    if safe:
        disp = np.zeros(len(w))
        ind = w > 0
        disp[ind] = material_dispersion(
            units.To.m(w[ind]), material, pressure, temperature, ideal, False
        )
        return disp
    else:
        material_dico = load_material_dico(material)
        if ideal:
            n_gas_2 = materials.sellmeier(wavelengths, material_dico, pressure, temperature) + 1
        else:
            N_1 = materials.number_density_van_der_waals(
                pressure=pressure, temperature=temperature, material_dico=material_dico
            )
            N_0 = materials.number_density_van_der_waals(material_dico=material_dico)
            n_gas_2 = materials.sellmeier(wavelengths, material_dico) * N_1 / N_0 + 1
        order = np.argsort(w)
        unorder = np.argsort(order)
        return fiber.beta2(w[order], np.sqrt(n_gas_2[order]))[unorder]


def find_optimal_depth(
    spectrum: T, w_c: np.ndarray, w0: float, material: str, max_z: float = 1.0
) -> tuple[T, pulse.OptimizeResult]:
    """finds the optimal silica depth to compress a pulse

    Parameters
    ----------
    spectrum : np.ndarray or Spectrum, shape (n, )
        spectrum from which to remove 2nd order dispersion
    w_c : np.ndarray, shape (n, )
        corresponding centered angular frequencies (w-w0)
    w0 : float
        pump central angular frequency
    material : str
        material name, by default 'silica'
    max_z : float
        maximum propagation distance in m

    Returns
    -------
    float
        distance in m
    """
    w = w_c + w0
    disp = np.zeros(len(w))
    ind = w > (w0 / 10)
    disp[ind] = material_dispersion(units.To.m(w[ind]), material)

    propagate = lambda z: spectrum * np.exp(-0.5j * disp * w_c ** 2 * z)
    integrate = lambda z: math.abs2(np.fft.ifft(propagate(z)))

    def score(z):
        return -np.nansum(integrate(z) ** 6)

    opti = minimize_scalar(score, method="bounded", bounds=(0, max_z))
    return propagate(opti.x), opti


def propagate_field(
    t: np.ndarray, field: np.ndarray, z: float, material: str, center_wl_nm: float = 1540.0
) -> np.ndarray:
    """propagates a field through bulk material

    Parameters
    ----------
    t : np.ndarray, shape (n,)
        time grid
    field : np.ndarray, shape (n,)
        corresponding complex field
    z : float
        distance to propagate in m
    material : str
        material name
    center_wl_nm : float, optional
        center wavelength of the grid in nm, by default 1540

    Returns
    -------
    np.ndarray, shape (n,)
        propagated field
    """
    w_c = math.wspace(t)
    l = units.m(w_c + units.nm(center_wl_nm))
    disp = material_dispersion(l, material)
    return np.fft.ifft(np.fft.fft(field) * np.exp(0.5j * disp * w_c ** 2 * z))
