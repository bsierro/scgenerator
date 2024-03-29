from __future__ import annotations

from dataclasses import dataclass, field
from functools import cache
from typing import Any, TypeVar

import numpy as np

import scgenerator.math as math
from scgenerator import utils
from scgenerator.cache import np_cache
from scgenerator.logger import get_logger
from scgenerator.physics import units
from scgenerator.physics.units import NA, c, epsilon0, kB

T = TypeVar("T", np.floating, np.ndarray)


@dataclass
class Sellmeier:
    B: list[float] = field(default_factory=list)
    C: list[float] = field(default_factory=list)
    pressure_ref: float = 101325
    temperature_ref: float = 273.15
    kind: int = 2
    constant: float = 0

    @classmethod
    @cache
    def load(cls, name: str) -> Sellmeier:
        mat_dico = utils.load_material_dico(name)
        s = mat_dico.get("sellmeier", {})
        return cls(
            **{
                newk: s.get(k, None)
                for newk, k in zip(
                    ["B", "C", "pressure_ref", "temperature_ref", "kind", "constant"],
                    ["B", "C", "P0", "T0", "kind", "const"],
                )
                if k in s
            }
        )

    def chi(self, wl: T, temperature: float | None = None, pressure: float | None = None) -> T:
        """n^2 - 1"""
        if isinstance(wl, np.ndarray):
            chi = np.zeros_like(wl)  # = n^2 - 1
        else:
            chi = 0
        if self.kind == 1:
            for b, c_ in zip(self.B, self.C):
                chi += wl**2 * b / (wl**2 - c_)
        elif self.kind == 2:  # gives n-1
            for b, c_ in zip(self.B, self.C):
                chi += b / (c_ - 1 / wl**2)
            chi += self.constant
            chi = (chi + 1) ** 2 - 1
        elif self.kind == 3:  # Schott formula
            for i, b in reversed(list(enumerate(self.B))):
                chi += b * wl ** (-2 * (i - 1))
            chi = chi - 1
        else:
            raise ValueError(f"kind {self.kind} is not recognized.")

        if temperature is not None:
            chi *= self.temperature_ref / temperature

        if pressure is not None:
            chi *= pressure / self.pressure_ref
        return chi

    def n_gas_2(self, wl: T, temperature: float | None = None, pressure: float | None = None) -> T:
        return self.chi(wl, temperature, pressure) + 1

    def n(self, wl: T, temperature: float | None = None, pressure: float | None = None) -> T:
        return np.sqrt(self.n_gas_2(wl, temperature, pressure))

    def delta(self, wl: np.ndarray, wl_zero_disp: float) -> np.ndarray:
        """
        'delta' quantity that describes the gas dispersion according to eq. S7 in [1]

        Parameters
        ----------
        wl : ndarray
            wavelength in m
        wl_zero_disp : float
            zero dispersion wavelength in m

        Reference
        ---------
        [1] TRAVERS, John C., GRIGOROVA, Teodora F., BRAHMS, Christian, et al. High-energy
            pulse self-compression and ultraviolet generation through soliton dynamics in hollow
            capillary fibres. Nature Photonics, 2019, vol. 13, no 8, p. 547-554.
        """
        factor = (math.u_nm(1, 1) / c) ** 2 * (0.5 * wl / np.pi) ** 3
        f = np.gradient(np.gradient(self.chi(wl), wl), wl)
        ind = math.argclosest(wl, wl_zero_disp)
        return factor * (f / f[ind] - 1)


class Gas:
    name: str
    sellmeier: Sellmeier
    atomic_number: int | float
    atomic_mass: float
    chi3_0: float
    ionization_energy: float | None

    _raw_sellmeier: dict[str, Any]

    def __init__(self, gas_name: str):
        self.name = gas_name
        self._raw_sellmeier = utils.load_material_dico(gas_name)
        self.atomic_mass = self._raw_sellmeier["atomic_mass"]
        self.atomic_number = self._raw_sellmeier["atomic_number"]
        self.ionization_energy = self._raw_sellmeier.get("ionization_energy")

        s = self._raw_sellmeier.get("sellmeier", {})
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
        kerr = self._raw_sellmeier["kerr"]
        n2_0 = kerr["n2"]
        self._kerr_wl = kerr.get("wavelength", 800e-9)
        self.chi3_0 = (
            4
            / 3
            * units.epsilon0
            * units.c
            * self.sellmeier.n_gas_2(self._kerr_wl, kerr["T0"], kerr["P0"])
            * n2_0
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
            return self.number_density_van_der_waals(
                pressure, temperature
            ) / self.number_density_van_der_waals(
                self.sellmeier.pressure_ref,
                self.sellmeier.temperature_ref,
            )

    def number_density(
        self, temperature: float = None, pressure: float = None, ideal_gas: bool = False
    ) -> float:
        """returns the number density in 1/m^3 using van der Waals equation

        Parameters
        ----------
        temperature : float, optional
            temperature in K, by default None
        pressure : float, optional
            pressure in Pa, by default None

        Returns
        -------
        float
            number density in 1/m^3
        """
        pressure = pressure or self.sellmeier.pressure_ref
        temperature = temperature or self.sellmeier.temperature_ref
        if ideal_gas:
            return pressure / temperature / kB
        else:
            return self.number_density_van_der_waals(pressure, temperature)

    def number_density_van_der_waals(self, pressure: float = None, temperature: float = None):
        """returns the number density of a gas

        Parameters
        ----------
        pressure : float, optional
            pressure in Pa, by default the reference pressure
        temperature : float, optional
            temperature in K, by default the reference temperature

        Returns
        ----------
        the numbers density (/m^3)

        Raises
        ----------
        ValueError : Since the Van der Waals equation is a cubic one, there could be more than one
        real, positive solution
        """

        logger = get_logger(__name__)

        if pressure == 0:
            return 0
        a = self._raw_sellmeier.get("a", 0)
        b = self._raw_sellmeier.get("b", 0)
        pressure = (
            self._raw_sellmeier["sellmeier"].get("P0", 101325) if pressure is None else pressure
        )
        temperature = (
            self._raw_sellmeier["sellmeier"].get("T0", 273.15)
            if temperature is None
            else temperature
        )
        ap = a / NA**2
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

    def chi3(self, temperature: float | None = None, pressure: float | None = None) -> float:
        """nonlinear susceptibility"""

        # if pressure and/or temperature are specified, adjustment is made according to number density ratio
        if pressure is not None or temperature is not None:
            N0 = self.number_density_van_der_waals()
            N = self.number_density_van_der_waals(pressure, temperature)
            ratio = N / N0
        else:
            ratio = 1
        return ratio * self.chi3_0

    def n2(self, temperature: float | None = None, pressure: float | None = None) -> float:
        return (
            0.75
            * self.chi3(temperature, pressure)
            / (
                units.epsilon0
                * units.c
                * self.sellmeier.n_gas_2(self._kerr_wl, temperature, pressure)
            )
        )

    @property
    def ionic_charge(self):
        return 1

    @property
    def barrier_suppression(self) -> float:
        """
        Compute the barrier suppression threshold (W/m^2)
        """
        field_max = self.barrier_suppression_field
        return 0.5 * units.c * units.epsilon0 * field_max**2

    @property
    def barrier_suppression_field(self) -> float:
        """
        Barrier suppression threshold electric field

        Reference
        ---------
        [1] KRAINOV, V. P. Theory of barrier-suppression ionization of atoms. Journal of Nonlinear
            Optical Physics & Materials, 1995, vol. 4, no 04, p. 775-798.
        [2] BETHE, H. A. et SALPETER, E. E. Quantum mechanics of one-and two-electron atoms. 1977.
        """
        # from [1-2] :
        # field_max = 0.0625 / self.ionic_charge * (self.ionization_energy / 2.1787e-18) ** 2
        # return field_max = 5.14220670712125e11

        # from [3] :
        Z = self.ionic_charge
        Ip_au = self.ionization_energy / 4.359744650021498e-18
        ns = Z / np.sqrt(2 * Ip_au)
        return Z**3 / (16 * ns**4) * 5.14220670712125e11

    def get(self, key, default=None):
        return self._raw_sellmeier.get(key, default)

    def __getitem__(self, key):
        return self._raw_sellmeier[key]

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.name!r})"


@np_cache
def n_gas_2(wl_for_disp: np.ndarray, gas_name: str, pressure: float, temperature: float):
    """Returns the sqare of the index of refraction of the specified gas"""
    return Sellmeier.load(gas_name).n_gas_2(wl_for_disp, temperature, pressure)


def pressure_from_gradient(ratio: float, p0: float, p1: float) -> float:
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
    return np.sqrt(p0**2 - ratio * (p0**2 - p1**2))


def delta_gas(w: np.ndarray, gas: Gas) -> np.ndarray:
    """returns the value delta_t (eq. 24 in Markos(2017))
    Parameters
    ----------
        w : np.ndarray
            angular frequency array
        gas : Gas
    Returns
    ----------
        delta_t
        since 2 gradients are computed, it is recommended to exclude the 2 extremum values
    """
    chi = gas.sellmeier.chi(units.m.inv(w))
    N0 = gas.number_density_van_der_waals()

    dchi_dw = np.gradient(chi, w)
    return 1 / (N0 * c) * (dchi_dw + w / 2 * np.gradient(dchi_dw, w))


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
    return Gas(gas_name).n2(temperature, pressure)


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
    gas = Gas(gas_name)
    n = gas.sellmeier.n(wavelength, temperature, pressure)
    n2 = gas.n2(temperature, pressure)
    return n2_to_chi3(n2, n)


def n2_to_chi3(n2: float, n0: float) -> float:
    return n2 * 4 * epsilon0 * n0**2 * c / 3


def chi3_to_n2(chi3: float, n0: float) -> float:
    return 3.0 * chi3 / (4.0 * epsilon0 * c * n0**2)
