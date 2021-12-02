import functools
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .. import utils
from ..cache import np_cache
from ..logger import get_logger
from . import math, units
from .units import NA, c, epsilon0, kB


@dataclass
class Sellmeier:
    B: list[float] = field(default_factory=list)
    C: list[float] = field(default_factory=list)
    pressure_ref: float = 101325
    temperature_ref: float = 273.15
    kind: int = 2
    constant: float = 0

    def chi(self, wl: np.ndarray) -> np.ndarray:
        """n^2 - 1"""
        chi = np.zeros_like(wl)  # = n^2 - 1
        if self.kind == 1:
            for b, c_ in zip(self.B, self.C):
                chi += wl ** 2 * b / (wl ** 2 - c_)
        elif self.kind == 2:  # gives n-1
            for b, c_ in zip(self.B, self.C):
                chi += b / (c_ - 1 / wl ** 2)
            chi += self.constant
            chi = (chi + 1) ** 2 - 1
        elif self.kind == 3:  # Schott formula
            for i, b in reversed(list(enumerate(self.B))):
                chi += b * wl ** (-2 * (i - 1))
            chi = chi - 1
        else:
            raise ValueError(f"kind {self.kind} is not recognized.")

        # if temperature is not None:
        #     chi *= self.temperature_ref / temperature

        # if pressure is not None:
        #     chi *= pressure / self.pressure_ref
        return chi

    def n_gas_2(self, wl: np.ndarray) -> np.ndarray:
        return self.chi(wl) + 1


class GasInfo:
    name: str
    sellmeier: Sellmeier
    n2: float
    atomic_number: int
    atomic_mass: float
    ionization_energy: float

    def __init__(self, gas_name: str):
        self.name = gas_name
        self.mat_dico = utils.load_material_dico(gas_name)
        self.n2 = self.mat_dico["kerr"]["n2"]
        self.atomic_mass = self.mat_dico["atomic_mass"]
        self.atomic_number = self.mat_dico["atomic_number"]
        self.ionization_energy = self.mat_dico.get("ionization_energy")

        s = self.mat_dico.get("sellmeier", {})
        self.sellmeier = Sellmeier(
            **{
                newk: s.get(k, None)
                for newk, k in zip(
                    ["B", "C", "pressure_ref", "temperature_ref", "kind", "constant"],
                    ["B", "C", "P0", "T0", "kind", "const"],
                )
                if k in s
            }
        )

    def pressure_from_relative_density(self, density: float, temperature: float = None) -> float:
        temperature = temperature or self.sellmeier.temperature_ref
        return self.sellmeier.temperature_ref / temperature * density * self.sellmeier.pressure_ref

    def density_factor(self, temperature: float, pressure: float, ideal_gas: bool) -> float:
        """returns the number density relative to reference values

        Parameters
        ----------
        temperature : float
            target temperature in K
        pressure : float
            target pressure in Pa
        ideal_gas : bool
            whether to use ideal gas law

        Returns
        -------
        float
            N / N_0
        """
        if ideal_gas:
            return (
                pressure
                / self.sellmeier.pressure_ref
                * self.sellmeier.temperature_ref
                / temperature
            )
        else:
            return number_density_van_der_waals(
                self.get("a"), self.get("b"), pressure, temperature
            ) / number_density_van_der_waals(
                self.get("a"),
                self.get("b"),
                self.sellmeier.pressure_ref,
                self.sellmeier.temperature_ref,
            )

    @property
    def ionic_charge(self):
        return self.atomic_number - 1

    def get(self, key, default=None):
        return self.mat_dico.get(key, default)

    def __getitem__(self, key):
        return self.mat_dico[key]

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.name!r})"


@np_cache
def n_gas_2(
    wl_for_disp: np.ndarray, gas_name: str, pressure: float, temperature: float, ideal_gas: bool
):
    """Returns the sqare of the index of refraction of the specified gas"""
    material_dico = utils.load_material_dico(gas_name)

    n_gas_2 = fast_n_gas_2(wl_for_disp, pressure, temperature, ideal_gas, material_dico)
    return n_gas_2


def fast_n_gas_2(
    wl_for_disp: np.ndarray,
    pressure: float,
    temperature: float,
    ideal_gas: bool,
    material_dico: dict[str, Any],
):
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
            material_dico["sellmeier"].get("T0", 273.15) if temperature is None else temperature
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


@functools.singledispatch
def sellmeier(
    wl_for_disp: np.ndarray,
    material_dico: dict[str, Any],
    pressure: float = None,
    temperature: float = None,
) -> np.ndarray:
    """reads a file containing the Sellmeier values corresponding to the choses material and
    returns the real susceptibility pressure and temperature adjustments are made according to
    ideal gas law.

    Parameters
    ----------
    wl_for_disp : array, shape (n,)
        wavelength vector over which to compute the refractive index
    material_dico : dict[str, Any]
        material dictionary as explained in scgenerator.utils.load_material_dico
    pressure : float, optional
        pressure in Pa, by default None
    temperature : float, optional
        temperature of the gas in Kelvin

    Returns
    -------
    array, shape (n,)
        chi = n^2 - 1
    """
    logger = get_logger(__name__)

    WL_THRESHOLD = 8.285e-6
    ind = wl_for_disp < WL_THRESHOLD
    temp_l = wl_for_disp[ind]

    B = material_dico["sellmeier"]["B"]
    C = material_dico["sellmeier"]["C"]
    const = material_dico["sellmeier"].get("const", 0)
    P0 = material_dico["sellmeier"].get("P0", 1e5)
    t0 = material_dico["sellmeier"].get("t0", 273.15)
    kind = material_dico["sellmeier"].get("kind", 1)

    chi = np.zeros_like(wl_for_disp)  # = n^2 - 1
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


@sellmeier.register
def sellmeier_scalar(
    wavelength: float,
    material_dico: dict[str, Any],
    pressure: float = None,
    temperature: float = None,
) -> float:
    """n^2 - 1"""
    return float(sellmeier(np.array([wavelength]), material_dico, pressure, temperature)[0])


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


def gas_chi3(gas_name: str, wavelength: float, pressure: float, temperature: float) -> float:
    """returns the chi3 of a particular material

    Parameters
    ----------
    gas_name : str
        [description]
    pressure : float
        [description]
    temperature : float
        [description]

    Returns
    -------
    float
        [description]
    """
    mat_dico = utils.load_material_dico(gas_name)
    chi = sellmeier_scalar(wavelength, mat_dico, pressure=pressure, temperature=temperature)
    return n2_to_chi3(
        non_linear_refractive_index(mat_dico, pressure, temperature), np.sqrt(chi + 1)
    )


def n2_to_chi3(n2: float, n0: float) -> float:
    return n2 * 4 * epsilon0 * n0 ** 2 * c / 3


def chi3_to_n2(chi3: float, n0: float) -> float:
    return 3.0 * chi3 / (4.0 * epsilon0 * c * n0 ** 2)
