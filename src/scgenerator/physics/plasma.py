from dataclasses import dataclass
from typing import TypeVar

import numpy as np
import scipy.special
from numpy.core.numeric import zeros_like

from ..math import cumulative_simpson, expm1_int
from .units import e, hbar, me

T_a = TypeVar("T_a", np.floating, np.ndarray, float)


@dataclass
class PlasmaInfo:
    electron_density: np.ndarray
    dp_dt: np.ndarray
    rate: np.ndarray
    dn_dt: np.ndarray
    debug: np.ndarray


class IonizationRate:
    ionization_energy: float

    def __init__(self, ionization_energy: float):
        self.Z = 1
        self.ionization_energy = ionization_energy
        self.nstar = self.Z * np.sqrt(2.1787e-18 / ionization_energy)

    def __call__(self, field_abs: T_a) -> T_a:
        ...


class IonizationRateADK(IonizationRate):
    def __init__(self, ionization_energy: float):
        super().__init__(ionization_energy)
        self.omega_p = ionization_energy / hbar

        Cnstar = 2 ** (2 * self.nstar) / (scipy.special.gamma(self.nstar + 1) ** 2)
        self.omega_pC = self.omega_p * Cnstar

    def omega_t(self, field_abs: T_a) -> T_a:
        return e * field_abs / np.sqrt(2 * me * self.ionization_energy)

    def __call__(self, field_abs: T_a) -> T_a:
        ot = self.omega_t(field_abs)
        opt4 = 4 * self.omega_p / ot
        return self.omega_pC * opt4 ** (2 * self.nstar - 1) * np.exp(-opt4 / 3)


class IonizationRatePPT(IonizationRate):
    def __init__(self, ionization_energy: float) -> None:
        self.ionization_energy = ionization_energy
        self.numerator = 2 * (2 * self.ionization_energy) ** 1.5

    def __call__(self, field_abs: T_a) -> T_a:

        return (
            self.factor
            * (self.numerator / field_abs) ** (2 * self.nstart - 1)
            * np.exp(-self.numerator / (3 * field_abs))
        )


class Plasma:
    dt: float
    ionization_energy: float
    rate: IonizationRate

    def __init__(self, dt: float, ionization_energy: float, rate: IonizationRate):
        self.dt = dt
        self.ionization_energy = ionization_energy
        self.rate = rate

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

        # polarization = -loss_integrated + phase_integrated
        dp_dt = loss_term
        return PlasmaInfo(electron_density, dp_dt, rate, dn_dt, phase_term)


def adiabadicity(w: np.ndarray, I: float, field: np.ndarray) -> np.ndarray:
    return w * np.sqrt(2 * me * I) / (e * np.abs(field))


def free_electron_density(rate: np.ndarray, dt: float, N0: float) -> np.ndarray:
    return N0 * expm1_int(rate, dt)
