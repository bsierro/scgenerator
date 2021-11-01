from dataclasses import dataclass
import numpy as np
import scipy.special
from scipy.integrate import cumulative_trapezoid

from .units import e, hbar, me
from ..math import inverse_integral_exponential


@dataclass
class PlasmaInfo:
    electron_density: np.ndarray
    dn_dt: np.ndarray
    polarization: np.ndarray


class IonizationRate:
    def __call__(self, field: np.ndarray) -> np.ndarray:
        ...


class IonizationRateADK(IonizationRate):
    def __init__(self, ionization_energy: float, atomic_number: int):
        self.Z = (atomic_number - 1) * e
        self.ionization_energy = ionization_energy
        self.omega_p = ionization_energy / hbar

        self.nstar = self.Z * np.sqrt(2.1787e-18 / ionization_energy)
        Cnstar = 2 ** (2 * self.nstar) / (scipy.special.gamma(self.nstar + 1) ** 2)
        self.omega_pC = self.omega_p * Cnstar

    def omega_t(self, field):
        return e * np.abs(field) / np.sqrt(2 * me * self.ionization_energy)

    def __call__(self, field: np.ndarray) -> np.ndarray:
        ot = self.omega_t(field)
        opt4 = 4 * self.omega_p / (ot + 1e-14 * ot.max())
        return self.omega_pC * opt4 ** (2 * self.nstar - 1) * np.exp(-opt4 / 3)


class Plasma:
    def __init__(self, t: np.ndarray, ionization_energy: float, atomic_number: int):
        self.t = t
        self.Ip = ionization_energy
        self.atomic_number = atomic_number
        self.rate = IonizationRateADK(self.Ip, self.atomic_number)

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
        # Ne = free_electron_density(self.t, field, N0, self.rate)
        field_abs = np.abs(field)
        delta = 1e-14 * field_abs.max()
        exp_int = inverse_integral_exponential(self.t, self.rate(field_abs))
        electron_density = N0 * (1 - exp_int)
        dn_dt = N0 * self.rate(field_abs) * exp_int
        out = cumulative_trapezoid(
            # np.gradient(self.density, self.t) * self.Ip / field
            dn_dt * self.Ip / (field + delta)
            + e ** 2 / me * cumulative_trapezoid(electron_density * field, self.t, initial=0),
            self.t,
            initial=0,
        )
        return PlasmaInfo(electron_density, dn_dt, out)


def adiabadicity(w: np.ndarray, I: float, field: np.ndarray) -> np.ndarray:
    return w * np.sqrt(2 * me * I) / (e * np.abs(field))


def free_electron_density(
    t: np.ndarray, field: np.ndarray, N0: float, rate: IonizationRate
) -> np.ndarray:
    return N0 * (1 - np.exp(-cumulative_trapezoid(rate(field), t, initial=0)))
