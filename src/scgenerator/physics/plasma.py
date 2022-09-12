from typing import Any, Callable, NamedTuple, TypeVar

import numpy as np
import scipy.special
from scipy.interpolate import interp1d
from numpy.core.numeric import zeros_like

from ..math import cumulative_simpson, expm1_int
from .units import e, hbar, me


class PlasmaInfo(NamedTuple):
    polarization: np.ndarray
    electron_density: np.ndarray
    rate: np.ndarray
    debug: Any = None


def ion_rate_adk(
    field_abs: np.ndarray, ion_energy: float, Z: float
) -> Callable[[np.ndarray], np.ndarray]:

    nstar = Z * np.sqrt(2.1787e-18 / ion_energy)
    omega_p = ion_energy / hbar
    Cnstar = 2 ** (2 * nstar) / (scipy.special.gamma(nstar + 1) ** 2)
    omega_pC = omega_p * Cnstar
    omega_t = e * field_abs / np.sqrt(2 * me * ion_energy)
    opt4 = 4 * omega_p / omega_t
    return omega_pC * opt4 ** (2 * nstar - 1) * np.exp(-opt4 / 3)


def cache_ion_rate(
    ion_energy, rate_func: Callable[[np.ndarray, float, float], np.ndarray]
) -> Callable[[np.ndarray], np.ndarray]:
    Z = 1
    E_max = barrier_suppression(ion_energy, Z) * 2
    E_min = E_max / 5000
    field = np.linspace(E_min, E_max, 4096)
    interp = interp1d(
        field,
        rate_func(field, ion_energy, Z),
        "cubic",
        assume_sorted=True,
        fill_value=0,
        bounds_error=False,
    )

    def compute(field_abs: np.ndarray) -> np.ndarray:
        if field_abs.max() > E_max or field_abs.min() < -E_max:
            raise ValueError("E field is out of bounds")
        return interp(field_abs)

    return compute


def create_ion_rate_func(
    ionization_energy: float, model="ADK"
) -> Callable[[np.ndarray], np.ndarray]:
    if model == "ADK":
        func = ion_rate_adk
    else:
        raise ValueError(f"Ionization model {model!r} unrecognized")

    return cache_ion_rate(ionization_energy, func)


class Plasma:
    dt: float
    ionization_energy: float
    rate: Callable[[np.ndarray], np.ndarray]

    def __init__(self, dt: float, ionization_energy: float):
        self.dt = dt
        self.ionization_energy = ionization_energy
        self.rate = create_ion_rate_func(self.ionization_energy)

    def __call__(self, field: np.ndarray, N0: float) -> PlasmaInfo:
        """returns the number density of free electrons as function of time

        Parameters
        ----------
        field : np.ndarray
            electric field in V/m
        N0 : float
            total number density of matter

        Returns
        -------
        np.ndarray
            number density of free electrons as function of time
        """
        field_abs: np.ndarray = np.abs(field)
        nzi = field != 0
        rate = zeros_like(field_abs)
        rate[nzi] = self.rate(field_abs[nzi])
        electron_density = free_electron_density(rate, self.dt, N0)
        dn_dt: np.ndarray = (N0 - electron_density) * rate

        loss_term = np.zeros_like(field)
        loss_term[nzi] = dn_dt[nzi] * self.ionization_energy / field[nzi]

        phase_term = self.dt * e ** 2 / me * cumulative_simpson(electron_density * field)

        dp_dt = loss_term + phase_term
        polarization = self.dt * cumulative_simpson(dp_dt)
        return PlasmaInfo(polarization, electron_density, rate)


def adiabadicity(w: np.ndarray, I: float, field: np.ndarray) -> np.ndarray:
    return w * np.sqrt(2 * me * I) / (e * np.abs(field))


def free_electron_density(rate: np.ndarray, dt: float, N0: float) -> np.ndarray:
    return N0 * expm1_int(rate, dt)


def barrier_suppression(ionpot, Z):
    Ip_au = ionpot / 4.359744650021498e-18
    ns = Z / np.sqrt(2 * Ip_au)
    return Z ** 3 / (16 * ns ** 4) * 5.14220670712125e11
