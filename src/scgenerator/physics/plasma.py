from dataclasses import dataclass
from typing import TypeVar
import numpy as np
import scipy.special

from .units import e, hbar, me
from ..math import expm1_int, cumulative_simpson

T_a = TypeVar("T_a", np.floating, np.ndarray)


@dataclass
class PlasmaInfo:
    electron_density: np.ndarray
    polarization: np.ndarray
    rate: np.ndarray
    dn_dt: np.ndarray
    debug: np.ndarray


class IonizationRate:
    ionization_energy: float

    def __call__(self, field: np.ndarray) -> np.ndarray:
        ...


class IonizationRateADK(IonizationRate):
    def __init__(self, ionization_energy: float):
        self.Z = 1
        self.ionization_energy = ionization_energy
        self.omega_p = ionization_energy / hbar

        self.nstar = self.Z * np.sqrt(2.1787e-18 / ionization_energy)
        Cnstar = 2 ** (2 * self.nstar) / (scipy.special.gamma(self.nstar + 1) ** 2)
        self.omega_pC = self.omega_p * Cnstar

    def omega_t(self, field: T_a) -> T_a:
        return e * np.abs(field) / np.sqrt(2 * me * self.ionization_energy)

    def __call__(self, field: T_a) -> T_a:
        ot = self.omega_t(field)
        opt4 = 4 * self.omega_p / (ot + 1e-14 * ot.max())
        return self.omega_pC * opt4 ** (2 * self.nstar - 1) * np.exp(-opt4 / 3)


class IonizationRatePPT(IonizationRate):
    def __init__(self, ionization_energy: float) -> None:
        self.ionization_energy = ionization_energy


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
        rate = self.rate(field_abs)
        electron_density = free_electron_density(rate, self.dt, N0)
        dn_dt: np.ndarray = (N0 - electron_density) * rate
        integrand = np.zeros_like(field)
        integrand[nzi] = dn_dt[nzi] * self.ionization_energy / field[nzi]

        energy_loss = self.dt * cumulative_simpson(integrand)
        added_phase = (
            self.dt ** 2
            * e ** 2
            / me
            * cumulative_simpson(cumulative_simpson(electron_density * field))
        )
        polarization = energy_loss + added_phase
        return PlasmaInfo(electron_density, polarization, rate, dn_dt, (energy_loss, added_phase))


def adiabadicity(w: np.ndarray, I: float, field: np.ndarray) -> np.ndarray:
    return w * np.sqrt(2 * me * I) / (e * np.abs(field))


def free_electron_density(rate: np.ndarray, dt: float, N0: float) -> np.ndarray:
    return N0 * expm1_int(rate, dt)
