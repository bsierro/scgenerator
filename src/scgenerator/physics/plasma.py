import numpy as np
import scipy.special
from scipy.integrate import cumulative_trapezoid

from .units import e, hbar, me


class IonizationRate:
    def __call__(self, field: np.ndarray) -> np.ndarray:
        ...


class IonizationRateADK(IonizationRate):
    def __init__(self, ionization_energy: float, atomic_number: int):
        self.Z = -(atomic_number - 1) * e
        self.ionization_energy = ionization_energy
        self.omega_p = ionization_energy / hbar

        self.nstar = self.Z * np.sqrt(2.1787e-18 / ionization_energy)
        Cnstar = 2 ** (2 * self.nstar) / (scipy.special.gamma(self.nstar + 1) ** 2)
        self.omega_pC = self.omega_p * Cnstar

    def omega_t(self, field):
        return e * np.abs(field) / np.sqrt(2 * me * self.ionization_energy)

    def __call__(self, field: np.ndarray) -> np.ndarray:
        opt4 = 4 * self.omega_p / self.omega_t(field)
        return self.omega_pC * opt4 ** (2 * self.nstar - 1) * np.exp(-opt4 / 3)


class Plasma:
    def __init__(self, t: np.ndarray, ionization_energy: float, atomic_number: int):
        self.t = t
        self.Ip = ionization_energy
        self.atomic_number = atomic_number
        self.rate = IonizationRateADK(self.Ip, self.atomic_number)

    def __call__(self, field: np.ndarray, N0: float) -> np.ndarray:
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
        Ne = free_electron_density(self.t, field, N0, self.rate)
        return cumulative_trapezoid(
            np.gradient(Ne, self.t) * self.Ip / field
            + e ** 2 / me * cumulative_trapezoid(Ne * field, self.t, initial=0),
            self.t,
            initial=0,
        )


def adiabadicity(w: np.ndarray, I: float, field: np.ndarray) -> np.ndarray:
    return w * np.sqrt(2 * me * I) / (e * np.abs(field))


def free_electron_density(
    t: np.ndarray, field: np.ndarray, N0: float, rate: IonizationRate
) -> np.ndarray:
    return N0 * (1 - np.exp(-cumulative_trapezoid(rate(field), t, initial=0)))
