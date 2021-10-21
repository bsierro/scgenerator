from typing import Callable

import numpy as np
import scipy.special
from scipy.integrate import cumulative_trapezoid

from .. import utils
from ..cache import np_cache
from ..logger import get_logger
from . import units
from .units import NA, c, e, hbar, kB, me


@np_cache
def n_gas_2(
    wl_for_disp: np.ndarray, gas_name: str, pressure: float, temperature: float, ideal_gas: bool
):
    """Returns the sqare of the index of refraction of the specified gas"""
    material_dico = utils.load_material_dico(gas_name)

    if ideal_gas:
        n_gas_2 = sellmeier(wl_for_disp, material_dico, pressure, temperature) + 1
    else:
        N_1 = number_density_van_der_waals(
            pressure=pressure, temperature=temperature, material_dico=material_dico
        )
        N_0 = number_density_van_der_waals(material_dico=material_dico)
        n_gas_2 = sellmeier(wl_for_disp, material_dico) * N_1 / N_0 + 1
    return n_gas_2


def pressure_from_gradient(ratio, p0, p1):
    """returns the pressure as function of distance with eq. 20 in Markos et al. (2017)
    Parameters
    ----------
        ratio : relative position in the fiber (0 = start, 1 = end)
        p0 : pressure at the start
        p1 : pressure at the end
    Returns
    ----------
        the pressure (float)
    """
    return np.sqrt(p0 ** 2 - ratio * (p0 ** 2 - p1 ** 2))


def number_density_van_der_waals(
    a=None, b=None, pressure=None, temperature=None, material_dico=None
):
    """returns the number density of a gas
    Parameters
    ----------
        P : pressure
        T : temperature
            for pressure and temperature, the default
        a : Van der Waals a coefficient
        b : Van der Waals b coefficient
        material_dico : optional. If passed, will compute the number density at given reference values found in material_dico
    Returns
    ----------
        the numbers density (/m^3)
    Raises
    ----------
        ValueError : Since the Van der Waals equation is a cubic one, there could be more than one real, positive solution
    """

    logger = get_logger(__name__)

    if pressure == 0:
        return 0
    if material_dico is not None:
        a = material_dico.get("a", 0) if a is None else a
        b = material_dico.get("b", 0) if b is None else b
        pressure = material_dico["sellmeier"].get("P0", 101325) if pressure is None else pressure
        temperature = (
            material_dico["sellmeier"].get("t0", 273.15) if temperature is None else temperature
        )
    else:
        a = 0 if a is None else a
        b = 0 if b is None else b
        pressure = 101325 if pressure is None else pressure
        temperature = 273.15 if temperature is None else temperature

    ap = a / NA ** 2
    bp = b / NA

    # setup van der Waals equation for the number density
    p3 = -ap * bp
    p2 = ap
    p1 = -(pressure * bp + kB * temperature)
    p0 = pressure

    # filter out unwanted matches
    roots = np.roots([p3, p2, p1, p0])
    roots = roots[np.isreal(roots)].real
    roots = roots[roots > 0]
    if len(roots) != 1:
        s = f"Van der Waals eq with parameters P={pressure}, T={temperature}, a={a}, b={b}"
        s += f", There is more than one possible number density : {roots}."
        s += f", {np.min(roots)} was returned"
        logger.warning(s)
    return np.min(roots)


def sellmeier(lambda_, material_dico, pressure=None, temperature=None):
    """reads a file containing the Sellmeier values corresponding to the choses material and returns the real susceptibility
    pressure and temperature adjustments are made according to ideal gas law.
    Parameters
    ----------
        lambda_ : wl vector over which to compute the refractive index
        material_dico : material dictionary as explained in scgenerator.utils.load_material_dico
        pressure : pressure in mbar if material is a gas. Can be a constant or a tupple if a presure gradient is considered
        temperature : temperature of the gas in Kelvin
    Returns
    ----------
        an array n(lambda_)^2 - 1
    """
    logger = get_logger(__name__)

    WL_THRESHOLD = 8.285e-6
    ind = lambda_ < WL_THRESHOLD
    temp_l = lambda_[ind]

    B = material_dico["sellmeier"]["B"]
    C = material_dico["sellmeier"]["C"]
    const = material_dico["sellmeier"].get("const", 0)
    P0 = material_dico["sellmeier"].get("P0", 1e5)
    t0 = material_dico["sellmeier"].get("t0", 273.15)
    kind = material_dico["sellmeier"].get("kind", 1)

    chi = np.zeros_like(lambda_)  # = n^2 - 1
    if kind == 1:
        logger.debug("materials : using Sellmeier 1st kind equation")
        for b, c_ in zip(B, C):
            chi[ind] += temp_l ** 2 * b / (temp_l ** 2 - c_)
    elif kind == 2:  # gives n-1
        logger.debug("materials : using Sellmeier 2nd kind equation")
        for b, c_ in zip(B, C):
            chi[ind] += b / (c_ - 1 / temp_l ** 2)
        chi += const
        chi = (chi + 1) ** 2 - 1
    elif kind == 3:  # Schott formula
        logger.debug("materials : using Schott equation")
        for i, b in reversed(list(enumerate(B))):
            chi[ind] += b * temp_l ** (-2 * (i - 1))
        chi[ind] = chi[ind] - 1
    else:
        raise ValueError(f"kind {kind} is not recognized.")

    if temperature is not None:
        chi *= t0 / temperature

    if pressure is not None:
        chi *= pressure / P0

    logger.debug(f"computed chi between {np.min(chi):.2e} and {np.max(chi):.2e}")
    return chi


def delta_gas(w, material_dico):
    """returns the value delta_t (eq. 24 in Markos(2017))
    Parameters
    ----------
        w : angular frequency array
        material_dico : material dictionary as explained in scgenerator.utils.load_material_dico
    Returns
    ----------
        delta_t
        since 2 gradients are computed, it is recommended to exclude the 2 extremum values
    """
    chi = sellmeier(units.m.inv(w), material_dico)
    N0 = number_density_van_der_waals(material_dico=material_dico)

    dchi_dw = np.gradient(chi, w)
    return 1 / (N0 * c) * (dchi_dw + w / 2 * np.gradient(dchi_dw, w))


def non_linear_refractive_index(material_dico, pressure=None, temperature=None):
    """returns the non linear refractive index n2 adjusted for pressure and temperature
    NOTE : so far, there is no adjustment made for wavelength
    Parameters
    ----------
        lambda_ : wavelength array
        material_dico :
        pressure : pressure in Pa
        temperature : temperature in Kelvin
    Returns
    ----------
        n2
    """

    n2_ref = material_dico["kerr"]["n2"]

    # if pressure and/or temperature are specified, adjustment is made according to number density ratio
    if pressure is not None or temperature is not None:
        N0 = number_density_van_der_waals(material_dico=material_dico)
        N = number_density_van_der_waals(
            pressure=pressure, temperature=temperature, material_dico=material_dico
        )
        ratio = N / N0
    else:
        ratio = 1
    return ratio * n2_ref


def gas_n2(gas_name: str, pressure: float, temperature: float) -> float:
    """returns the nonlinear refractive index

    Parameters
    ----------
    gas_name : str
        gas name
    pressure : float
        pressure in Pa
    temperature : float
        temperature in K

    Returns
    -------
    float
        n2 in m2/W
    """
    return non_linear_refractive_index(utils.load_material_dico(gas_name), pressure, temperature)


def adiabadicity(w: np.ndarray, I: float, field: np.ndarray) -> np.ndarray:
    return w * np.sqrt(2 * me * I) / (e * np.abs(field))


def free_electron_density(
    t: np.ndarray, field: np.ndarray, N0: float, rate_func: Callable[[np.ndarray], np.ndarray]
) -> np.ndarray:
    return N0 * (1 - np.exp(-cumulative_trapezoid(rate_func(field), t, initial=0)))


def ionization_rate_ADK(
    ionization_energy: float, atomic_number
) -> Callable[[np.ndarray], np.ndarray]:
    Z = -(atomic_number - 1) * e

    omega_p = ionization_energy / hbar
    nstar = Z * np.sqrt(2.1787e-18 / ionization_energy)

    def omega_t(field):
        return e * np.abs(field) / np.sqrt(2 * me * ionization_energy)

    Cnstar = 2 ** (2 * nstar) / (scipy.special.gamma(nstar + 1) ** 2)
    omega_pC = omega_p * Cnstar

    def rate(field: np.ndarray) -> np.ndarray:
        opt4 = 4 * omega_p / omega_t(field)
        return omega_pC * opt4 ** (2 * nstar - 1) * np.exp(-opt4 / 3)

    return rate


class Plasma:
    def __init__(self, t: np.ndarray, ionization_energy: float, atomic_number: int):
        self.t = t
        self.Ip = ionization_energy
        self.atomic_number = atomic_number
        self.rate = ionization_rate_ADK(self.Ip, self.atomic_number)

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
