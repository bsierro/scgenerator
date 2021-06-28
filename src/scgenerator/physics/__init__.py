"""
This file contains functions that don't properly fit into the 'fiber' or 'pulse' category.
They're also not necessarily 'pure' function as they do some io and stuff.
"""

from typing import TypeVar

import numpy as np
from scipy.optimize import minimize_scalar

from ..io import load_material_dico
from .. import math
from . import fiber, materials, units, pulse

T = TypeVar("T")


def group_delay_to_gdd(wavelength: np.ndarray, group_delay: np.ndarray) -> np.ndarray:
    w = units.m.inv(wavelength)
    gdd = np.gradient(group_delay, w)
    return gdd


def material_dispersion(
    wavelengths,
    material: str,
    pressure=None,
    temperature=None,
    ideal=False,
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
        whether to use the ideal gas law instead of the van der Waals equation, by default False

    Returns
    -------
    out : ndarray, shape (n, )
        beta2 as function of wavelength
    """

    w = units.m(wavelengths)
    material_dico = load_material_dico(material)
    if ideal:
        n_gas_2 = materials.sellmeier(wavelengths, material_dico, pressure, temperature) + 1
    else:
        N_1 = materials.number_density_van_der_waals(
            pressure=pressure, temperature=temperature, material_dico=material_dico
        )
        N_0 = materials.number_density_van_der_waals(material_dico=material_dico)
        n_gas_2 = materials.sellmeier(wavelengths, material_dico) * N_1 / N_0 + 1

    return fiber.beta2(w, np.sqrt(n_gas_2))


def find_optimal_depth(
    spectrum: T, w_c: np.ndarray, w0: float, material: str = "silica", max_z: float = 1.0
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
    silica_disp = material_dispersion(units.To.m(w_c + w0), material)

    propagate = lambda z: spectrum * np.exp(-0.5j * silica_disp * w_c ** 2 * z)

    def score(z):
        return 1 / np.max(math.abs2(np.fft.ifft(propagate(z))))

    opti = minimize_scalar(score, bracket=(0, max_z))
    return opti.x
