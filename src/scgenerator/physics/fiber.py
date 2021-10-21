from typing import Any, Iterable, Literal, TypeVar

import numpy as np
from numpy import e
from numpy.fft import fft
from numpy.polynomial.chebyshev import Chebyshev, cheb2poly
from scipy.interpolate import interp1d
from sympy import re

from .. import utils
from ..cache import np_cache
from ..math import argclosest, u_nm
from . import materials as mat
from . import units
from .units import c, pi

pipi = 2 * pi
T = TypeVar("T")


def lambda_for_dispersion(
    l: np.ndarray, interpolation_range: tuple[float, float]
) -> tuple[np.ndarray, np.ndarray]:
    """Returns a wl vector for dispersion calculation

    Returns
    -------
    np.ndarray
        subset of l in the interpolation range with two extra values on each side
        to accomodate for taking gradients
    np.ndarray
        indices of the original l where the values are valid (i.e. without the two extra on each side)
    """
    su = np.where((l >= interpolation_range[0]) & (l <= interpolation_range[1]))[0]
    ind_above_cond = su >= len(l) // 2
    ind_above = su[ind_above_cond]
    ind_below = su[~ind_above_cond]
    fu = np.concatenate((ind_below, (ind_below + 2)[-2:], (ind_above - 2)[:2], ind_above))
    fs = fu[np.argsort(l[fu])[::-1]]
    l_out = l[fs]
    ind_out = fs[2:-2]
    return l_out, ind_out


def is_dynamic_dispersion(pressure=None):
    """tests if the parameter dictionary implies that the dispersion profile of the fiber changes with z

    Parameters
    ----------
    params : dict
        flattened parameters dict

    Returns
    -------
    bool : True if dispersion is supposed to change with z
    """
    out = False
    if pressure is not None:
        out |= isinstance(pressure, (tuple, list)) and len(pressure) == 2

    return out


def gvd_from_n_eff(n_eff: np.ndarray, wl_for_disp: np.ndarray):
    """computes the dispersion parameter D from an effective index of refraction n_eff
    Since computing gradients/derivatives of discrete arrays is not well defined on the boundary, it is
    advised to chop off the two values on either end of the returned array

    Parameters
    ----------
    n_eff : 1D array
        a wl-dependent index of refraction
    wl_for_disp : 1D array
        the wavelength array (len must match n_eff)

    Returns
    -------
    D : 1D array
        wl-dependent dispersion parameter as function of wl_for_disp
    """

    return -wl_for_disp / c * (np.gradient(np.gradient(n_eff, wl_for_disp), wl_for_disp))


def beta2_to_D(beta2, wl_for_disp):
    """returns the D parameter corresponding to beta2(wl_for_disp)"""
    return -(pipi * c) / (wl_for_disp ** 2) * beta2


def D_to_beta2(D, wl_for_disp):
    """returns the beta2 parameters corresponding to D(wl_for_disp)"""
    return -(wl_for_disp ** 2) / (pipi * c) * D


def A_to_C(A: np.ndarray, A_eff_arr: np.ndarray) -> np.ndarray:
    return (A_eff_arr / A_eff_arr[0]) ** (-1 / 4) * A


def C_to_A(C: np.ndarray, A_eff_arr: np.ndarray) -> np.ndarray:
    return (A_eff_arr / A_eff_arr[0]) ** (1 / 4) * C


def plasma_dispersion(wl_for_disp, number_density, simple=False):
    """computes dispersion (beta2) for constant plasma

    Parameters
    ----------
    wl_for_disp : array-like
        wavelengths over which to calculate the dispersion
    number_density : number of ionized atoms /m^3

    Returns
    -------
    beta2 : ndarray
        WL-dependent dispersion parameter
    """

    e2_me_e0 = 3182.60735  # e^2 /(m_e * epsilon_0)
    w = units.m(wl_for_disp)
    if simple:
        w_pl = number_density * e2_me_e0
        return -(w_pl ** 2) / (c * w ** 2)

    beta2_arr = beta2(w, np.sqrt(1 - number_density * e2_me_e0 / w ** 2))
    return beta2_arr


def n_eff_marcatili(wl_for_disp, n_gas_2, core_radius, he_mode=(1, 1)):
    """computes the effective refractive index according to the Marcatili model of a capillary

    Parameters
    ----------
    wl_for_disp : ndarray, shape (n, )
        wavelengths array (m)
    n_gas_2 : ndarray, shape (n, )
        square of the refractive index of the gas as function of wl_for_disp
    core_radius : float
        inner radius of the capillary (m)
    he_mode : tuple, shape (2, ), optional
        n and m value of the HE_nm mode. 1 and 1 corresponds to the fundamental mode

    Returns
    -------
    n_eff : ndarray, shape (n, )

    Reference
    ---------
    Marcatili, E., and core_radius. Schmeltzer, 1964, Bell Syst. Tech. J. 43, 1783.
    """
    u = u_nm(*he_mode)

    return np.sqrt(n_gas_2 - (wl_for_disp * u / (pipi * core_radius)) ** 2)


def n_eff_marcatili_adjusted(wl_for_disp, n_gas_2, core_radius, he_mode=(1, 1), fit_parameters=()):
    """computes the effective refractive index according to the Marcatili model of a capillary but adjusted at longer wavelengths

    Parameters
    ----------
    wl_for_disp : ndarray, shape (n, )
        wavelengths array (m)
    n_gas_2 : ndarray, shape (n, )
        refractive index of the gas as function of wl_for_disp
    core_radius : float
        inner radius of the capillary (m)
    he_mode : tuple, shape (2, ), optional
        n and m value of the HE_nm mode. 1 and 1 corresponds to the fundamental mode
    fit_parameters : tuple, shape (2, ), optional
        fitting parameters (s, h). See reference for more info

    Returns
    -------
    n_eff : ndarray, shape (n, )

    Reference
    ----------
    Köttig, F., et al. "Novel mid-infrared dispersive wave generation in gas-filled PCF by transient ionization-driven changes in dispersion." arXiv preprint arXiv:1701.04843 (2017).
    """
    u = u_nm(*he_mode)

    corrected_radius = effective_core_radius(wl_for_disp, core_radius, *fit_parameters)

    return np.sqrt(n_gas_2 - (wl_for_disp * u / (pipi * corrected_radius)) ** 2)


def A_eff_marcatili(core_radius: float) -> float:
    """Effective mode-field area for fundamental mode hollow capillaries

    Parameters
    ----------
    core_radius : float
        radius of the core

    Returns
    -------
    float
        effective mode field area
    """
    return 1.5 * core_radius ** 2


def capillary_spacing_hasan(
    capillary_num: int, capillary_radius: float, core_radius: float
) -> float:
    return (
        2 * (capillary_radius + core_radius) * np.sin(np.pi / capillary_num) - 2 * capillary_radius
    )


def resonance_thickness(
    wl_for_disp: np.ndarray, order: int, n_gas_2: np.ndarray, core_radius: float
) -> float:
    """computes at which wall thickness the specified wl is resonant

    Parameters
    ----------
    wl_for_disp : np.ndarray
        in m
    order : int
        0, 1, ...
    n_gas_2 : np.ndarray
        as returned by materials.n_gas_2
    core_radius : float
        in m

    Returns
    -------
    float
        thickness in m
    """
    n_si_2 = mat.n_gas_2(wl_for_disp, "silica", None, None, True)
    return (
        wl_for_disp
        / (4 * np.sqrt(n_si_2))
        * (2 * order + 1)
        * (1 - n_gas_2 / n_si_2 + wl_for_disp ** 2 / (4 * n_si_2 * core_radius ** 2)) ** -0.5
    )


def resonance_strength(
    wl_for_disp: np.ndarray, core_radius: float, capillary_thickness: float, order: int
) -> float:
    a = 1.83 + (2.3 * capillary_thickness / core_radius)
    n_si = np.sqrt(mat.n_gas_2(wl_for_disp, "silica", None, None, True))
    return (
        capillary_thickness
        / (n_si * core_radius) ** (2.303 * a / n_si)
        * ((order + 2) / (3 * order)) ** (3.57 * a)
    )


def capillary_resonance_strengths(
    wl_for_disp: np.ndarray,
    core_radius: float,
    capillary_thickness: float,
    capillary_resonance_max_order: int,
) -> list[float]:
    return [
        resonance_strength(wl_for_disp, core_radius, capillary_thickness, o)
        for o in range(1, capillary_resonance_max_order + 1)
    ]


@np_cache
def n_eff_hasan(
    wl_for_disp: np.ndarray,
    n_gas_2: np.ndarray,
    core_radius: float,
    capillary_num: int,
    capillary_nested: int,
    capillary_thickness: float,
    capillary_spacing: float,
    capillary_resonance_strengths: list[float],
) -> np.ndarray:
    """computes the effective refractive index of the fundamental mode according to the Hasan model for a anti-resonance fiber

    Parameters
    ----------
    wl_for_disp
        wavelenghs array (m)
    n_gas_2 : ndarray, shape (n, )
        squared refractive index of the gas as a function of wl_for_disp
    core_radius : float
        radius of the core (m) (from cented to edge of a capillary)
    capillary_num : int
        number of capillaries
    capillary_nested : int, optional
        number of levels of nested capillaries. default : 0
    capillary_thickness : float
        thickness of the capillaries (m)
    capillary_spacing : float
        spacing between capillaries (m)
    capillary_resonance_strengths : list or tuple
        strengths of the resonance lines. may be empty

    Returns
    -------
    n_eff : ndarray, shape (n, )
        the effective refractive index as function of wavelength

    Reference
    ----------
    Hasan, Md Imran, Nail Akhmediev, and Wonkeun Chang. "Empirical formulae for dispersion and effective mode area in hollow-core antiresonant fibers." Journal of Lightwave Technology 36.18 (2018): 4060-4065.
    """
    u = u_nm(1, 1)
    alpha = 5e-12

    Rg = core_radius / capillary_spacing

    f1 = 1.095 * np.exp(0.097041 / Rg)
    f2 = 0.007584 * capillary_num * np.exp(0.76246 / Rg) - capillary_num * 0.002 + 0.012
    if capillary_nested > 0:
        f2 += 0.0045 * np.exp(-4.1589 / (capillary_nested * Rg))

    R_eff = f1 * core_radius * (1 - f2 * wl_for_disp ** 2 / (core_radius * capillary_thickness))

    n_eff_2 = n_gas_2 - (u * wl_for_disp / (pipi * R_eff)) ** 2

    chi_sil = mat.sellmeier(wl_for_disp, utils.load_material_dico("silica"))

    with np.errstate(divide="ignore", invalid="ignore"):
        for m, strength in enumerate(capillary_resonance_strengths):
            n_eff_2 += (
                strength
                * wl_for_disp ** 2
                / (alpha + wl_for_disp ** 2 - chi_sil * (2 * capillary_thickness / (m + 1)) ** 2)
            )

    return np.sqrt(n_eff_2)


def A_eff_hasan(core_radius, capillary_num, capillary_spacing):
    """computed the effective mode area

    Parameters
    ----------
    core_radius : float
        radius of the core (m) (from cented to edge of a capillary)
    capillary_num : int
        number of capillaries
    capillary_spacing : float
        spacing between capillaries (m)

    Returns
    -------
    A_eff : float
    """
    M_f = 1.5 / (1 - 0.5 * np.exp(-0.245 * capillary_num))
    return M_f * core_radius ** 2 * np.exp((capillary_spacing / 22e-6) ** 2.5)


def V_eff_step_index(
    l: T,
    core_radius: float,
    numerical_aperture: float,
    interpolation_range: tuple[float, float] = None,
) -> T:
    """computes the V parameter of a step-index fiber

    Parameters
    ----------
    l : T
        wavelength
    core_radius : float
        radius of the core
    numerical_aperture : float
        as a decimal number
    interpolation_range : tuple[float, float], optional
        when provided, only computes V over this range, wavelengths outside this range will
        yield V=inf, by default None

    Returns
    -------
    T
        V parameter
    """
    pi2cn = 2 * pi * core_radius * numerical_aperture
    if interpolation_range is not None and isinstance(l, np.ndarray):
        low, high = interpolation_range
        l = np.where((l >= low) & (l <= high), l, np.inf)
    return pi2cn / l


def V_parameter_koshiba(l: np.ndarray, pitch: float, pitch_ratio: float) -> float:
    """returns the V parameter according to Koshiba2004


    Parameters
    ----------
    l : np.ndarray, shape (n,)
        wavelength
    pitch : float
        distance between air holes in m
    pitch_ratio : float
        ratio diameter of holes / distance
    w0 : float
        pump angular frequency

    Returns
    -------
    float
        effective mode field area
    """
    ratio_l = l / pitch
    n_co = 1.45
    a_eff = pitch / np.sqrt(3)
    pi2a = pipi * a_eff
    A, B = saitoh_paramters(pitch_ratio)

    V = A[0] + A[1] / (1 + A[2] * np.exp(A[3] * ratio_l))

    n_FSM2 = 1.45 ** 2 - (l * V / (pi2a)) ** 2
    V_eff = pi2a / l * np.sqrt(n_co ** 2 - n_FSM2)

    return V_eff


def A_eff_from_V(core_radius: float, V_eff: T) -> T:
    """According to Marcuse1978

    Parameters
    ----------
    core_radius : float
        in m
    V_eff : T
        effective V parameter.

    Returns
    -------
    T
        A_eff
    """
    out = np.ones_like(V_eff)
    out[V_eff > 0] = core_radius * (
        0.65 + 1.619 / V_eff[V_eff > 0] ** 1.5 + 2.879 / V_eff[V_eff > 0] ** 6
    )
    return out


def HCPCF_find_with_given_ZDW(
    variable: Literal["pressure", "temperature"],
    target: float,
    search_range: tuple[float, float],
    material_dico: dict[str, Any],
    model="marcatili",
    model_params={},
    pressure=None,
    temperature=None,
    ideal=False,
):
    """finds the parameters (pressure or temperature) to yield the target ZDW. assign the string value 'vary' to the parameter

    Parameters
    ----------
    variable : str {"pressure", "temperature"}
        which parameter to vary
    target : float
        the ZDW target, in m
    search_range : array, shape (2,)
        (min, max) of the search range
    other parameters : see HCPCF_dispersion. Pressure or temperature is used as initial value if it is variable

    Returns
    -------
    the parameter that satisfies the ZDW
    """
    from scipy import optimize

    l_search = [120e-9, 6000e-9]
    #
    fixed = [material_dico, model, model_params, ideal]

    if variable == "pressure":
        fixed.append(temperature)
        x0 = 1e5 if pressure is None else pressure

        def zdw(x, *args):
            current_ZDW = HCPF_ZDW(
                l_search,
                args[0],
                model=args[1],
                model_params=args[2],
                pressure=x,
                temperature=args[4],
                ideal=args[3],
            )
            out = current_ZDW - target
            return out

    elif variable == "temperature":
        fixed.append(pressure)
        x0 = 273.15 if temperature is None else temperature

        def zdw(x, *args):
            current_ZDW = HCPF_ZDW(
                l_search,
                args[0],
                model=args[1],
                model_params=args[2],
                pressure=args[4],
                temperature=x,
                ideal=args[3],
            )
            out = current_ZDW - target
            return out

    else:
        raise AttributeError(f"'variable' arg must be 'pressure' or 'temperature', not {variable}")

    optimized = optimize.root_scalar(
        zdw, x0=x0, args=tuple(fixed), method="brentq", bracket=search_range
    )

    return optimized.root


def HCPF_ZDW(
    search_range,
    material_dico,
    model="marcatili",
    model_params={},
    pressure=None,
    temperature=None,
    ideal=False,
    max_iter=10,
    threshold=1e-36,
):
    """finds one Zero Dispersion Wavelength (ZDW) of a given HC-PCF fiber

    Parameters
    ----------
    see HCPCF_dispersion for description of most arguments
    max_iter : float
        How many iterations are allowed at most to reach the threashold
    threshold : float
        upper bound of what counts as beta2 == 0 (in si units)

    Returns
    -------
    float:
        the ZDW in m
    """
    prev_find = np.inf
    l = np.linspace(*search_range, 50)

    core_radius = model_params["core_radius"]

    zdw_ind = 0
    for i in range(max_iter):
        beta2 = HCPCF_dispersion(
            l,
            material_dico,
            model=model,
            model_params=model_params,
            pressure=pressure,
            temperature=temperature,
            ideal=ideal,
        )
        zdw_ind = argclosest(beta2, 0)
        if beta2[zdw_ind] < threshold:
            break
        elif beta2[zdw_ind] < prev_find:
            l = np.linspace(
                l[zdw_ind] - (100 / (i + 1)) * 1e-9, l[zdw_ind] + (100 / (i + 1)) * 1e-9, 50
            )
            prev_find = beta2[zdw_ind]
        else:
            raise RuntimeError(
                f"Could not find a ZDW with parameters {1e6*core_radius} um, {1e-5 * pressure} bar, {temperature} K."
            )
    else:
        print(f"Could not get to threshold in {max_iter} iterations")

    return l[zdw_ind]


def beta(w_for_disp: np.ndarray, n_eff: np.ndarray) -> np.ndarray:
    return n_eff * w_for_disp / c


def beta1(w_for_disp: np.ndarray, n_eff: np.ndarray) -> np.ndarray:
    return np.gradient(beta(w_for_disp, n_eff), w_for_disp)


def beta2(w_for_disp: np.ndarray, n_eff: np.ndarray) -> np.ndarray:
    """computes the dispersion parameter beta2 according to the effective refractive index of the fiber and the frequency range

    Parameters
    ----------
    w : ndarray, shape (n, )
        angular frequencies over which to calculate the dispersion
    n_eff : ndarray_ shape (n, )
        effective refractive index of the fiber computed with one of the n_eff_* methods

    Returns
    -------
    beta2 : ndarray, shape (n, )
    """
    return np.gradient(np.gradient(beta(w_for_disp, n_eff), w_for_disp), w_for_disp)


def HCPCF_dispersion(
    wl_for_disp,
    material_dico=None,
    model="marcatili",
    model_params={},
    pressure=None,
    temperature=None,
    ideal=False,
):
    """returns the dispersion profile (beta_2) of a hollow-core photonic crystal fiber.

    Parameters
    ----------
    wl_for_disp : ndarray, shape (n, )
        wavelengths over which to calculate the dispersion
    material_dico : dict
        material dictionary respecting standard format explained in FIXME
    model : string {"marcatili", "marcatili_adjusted", "hasan"}
        which model of effective refractive index to use
    model_params : tuple
        to be cast to the function in charge of computing the effective index of the fiber. Every n_eff_* function has a signature
        n_eff_(wl_for_disp, n_gas_2, radius, *args) and model_params corresponds to args
    temperature : float
        Temperature of the material
    pressure : float
        constant pressure

    Returns
    -------
    out : 1D array
        beta2 as function of wavelength
    """

    w = units.m(wl_for_disp)
    if material_dico is None:
        n_gas_2 = np.ones_like(wl_for_disp)
    else:
        if ideal:
            n_gas_2 = mat.sellmeier(wl_for_disp, material_dico, pressure, temperature) + 1
        else:
            N_1 = mat.number_density_van_der_waals(
                pressure=pressure, temperature=temperature, material_dico=material_dico
            )
            N_0 = mat.number_density_van_der_waals(material_dico=material_dico)
            n_gas_2 = mat.sellmeier(wl_for_disp, material_dico) * N_1 / N_0 + 1

    n_eff_func = dict(
        marcatili=n_eff_marcatili, marcatili_adjusted=n_eff_marcatili_adjusted, hasan=n_eff_hasan
    )[model]
    n_eff = n_eff_func(wl_for_disp, n_gas_2, **model_params)

    return beta2(w, n_eff)


def dynamic_HCPCF_dispersion(
    wl_for_disp: np.ndarray,
    pressure_values: list[float],
    core_radius: float,
    fiber_model: str,
    model_params: dict[str, Any],
    temperature: float,
    ideal_gas: bool,
    w0: float,
    interp_range: tuple[float, float],
    material_dico: dict[str, Any],
    deg: int,
):
    """returns functions for beta2 coefficients and gamma instead of static values

    Parameters
    ----------
    wl_for_disp : wavelength array
    params : dict
        flattened parameter dictionary
    material_dico : dict
        material dictionary (see README for details)

    Returns
    -------
    beta2_coef : func(r), r is the relative position in the fiber
        a function that returns an array of coefficients as function of the relative position in the fiber
        to be used in disp_op
    gamma : func(r), r is the relative position in the fiber
        a function that returns a float corresponding to the nonlinear parameter at the relative position
        in the fiber
    """

    A_eff = 1.5 * core_radius ** 2

    # defining function instead of storing every possilble value
    def pressure(r):
        return mat.pressure_from_gradient(r, *pressure_values)

    def beta2(r):
        return HCPCF_dispersion(
            wl_for_disp,
            core_radius,
            material_dico,
            fiber_model,
            model_params,
            pressure(r),
            temperature,
            ideal_gas,
        )

    def n2(r):
        return mat.non_linear_refractive_index(material_dico, pressure(r), temperature)

    ratio_range = np.linspace(0, 1, 256)

    gamma_grid = np.array([gamma_parameter(n2(r), w0, A_eff) for r in ratio_range])
    gamma_interp = interp1d(ratio_range, gamma_grid)

    beta2_grid = np.array(
        [dispersion_coefficients(wl_for_disp, beta2(r), w0, interp_range, deg) for r in ratio_range]
    )
    beta2_interp = [
        interp1d(ratio_range, beta2_grid[:, i], assume_sorted=True) for i in range(deg + 1)
    ]

    def beta2_func(r):
        return [beta2_interp[i](r)[()] for i in range(deg + 1)]

    def gamma_func(r):
        return gamma_interp(r)[()]

    return beta2_func, gamma_func


def gamma_parameter(n2: float, w0: float, A_eff: T) -> T:
    if isinstance(A_eff, np.ndarray):
        A_eff_term = np.sqrt(A_eff * A_eff[0])
    else:
        A_eff_term = A_eff
    return n2 * w0 / (A_eff_term * c)


def constant_A_eff_arr(l: np.ndarray, A_eff: float) -> np.ndarray:
    return np.ones_like(l) * A_eff


@np_cache
def n_eff_pcf(wl_for_disp: np.ndarray, pitch: float, pitch_ratio: float) -> np.ndarray:
    """
    semi-analytical computation of the dispersion profile of a triangular Index-guiding PCF

    Parameters
    ----------
    wl_for_disp : np.ndarray, shape (n,)
        wavelengths over which to calculate the dispersion
    pitch : float
        distance between air holes in m
    pitch_ratio : float
        ratio diameter of hole / pitch

    Returns
    -------
    n_eff : np.ndarray, shape (n,)
        effective index of refraction

    Reference
    ---------
        Formulas and values are from Saitoh K and Koshiba M, "Empirical relations for simple design of photonic crystal fibers" (2005)

    """
    # Check validity
    if pitch_ratio < 0.2 or pitch_ratio > 0.8:
        print("WARNING : Fitted formula valid only for pitch ratio between 0.2 and 0.8")

    a_eff = pitch / np.sqrt(3)
    pi2a = pipi * a_eff

    ratio_l = wl_for_disp / pitch

    A, B = saitoh_paramters(pitch_ratio)

    V = A[0] + A[1] / (1 + A[2] * np.exp(A[3] * ratio_l))
    W = B[0] + B[1] / (1 + B[2] * np.exp(B[3] * ratio_l))

    n_FSM2 = 1.45 ** 2 - (wl_for_disp * V / (pi2a)) ** 2
    n_eff2 = (wl_for_disp * W / (pi2a)) ** 2 + n_FSM2
    n_eff = np.sqrt(n_eff2)

    material_dico = utils.load_material_dico("silica")
    chi_mat = mat.sellmeier(wl_for_disp, material_dico)
    return n_eff + np.sqrt(chi_mat + 1)


def A_eff_from_diam(effective_mode_diameter: float) -> float:
    return pi * (effective_mode_diameter / 2) ** 2


def A_eff_from_gamma(gamma: float, n2: float, w0: float):
    return n2 * w0 / (c * gamma)


def saitoh_paramters(pitch_ratio: float) -> tuple[float, float]:
    # Table 1 and 2 in Saitoh2005
    ai0 = np.array([0.54808, 0.71041, 0.16904, -1.52736])
    ai1 = np.array([5.00401, 9.73491, 1.85765, 1.06745])
    ai2 = np.array([-10.43248, 47.41496, 18.96849, 1.93229])
    ai3 = np.array([8.22992, -437.50962, -42.4318, 3.89])
    bi1 = np.array([5, 1.8, 1.7, -0.84])
    bi2 = np.array([7, 7.32, 10, 1.02])
    bi3 = np.array([9, 22.8, 14, 13.4])
    ci0 = np.array([-0.0973, 0.53193, 0.24876, 5.29801])
    ci1 = np.array([-16.70566, 6.70858, 2.72423, 0.05142])
    ci2 = np.array([67.13845, 52.04855, 13.28649, -5.18302])
    ci3 = np.array([-50.25518, -540.66947, -36.80372, 2.7641])
    di1 = np.array([7, 1.49, 3.85, -2])
    di2 = np.array([9, 6.58, 10, 0.41])
    di3 = np.array([10, 24.8, 15, 6])

    A = ai0 + ai1 * pitch_ratio ** bi1 + ai2 * pitch_ratio ** bi2 + ai3 * pitch_ratio ** bi3
    B = ci0 + ci1 * pitch_ratio ** di1 + ci2 * pitch_ratio ** di2 + ci3 * pitch_ratio ** di3
    return A, B


def load_custom_A_eff(A_eff_file: str, l: np.ndarray) -> np.ndarray:
    """loads custom effective area file

    Parameters
    ----------
    A_eff_file : str
        relative or absolute path to the file
    l : np.ndarray, shape (n,)
        wavelength array of the simulation

    Returns
    -------
    np.ndarray, shape (n,)
        wl-dependent effective mode field area
    """
    data = np.load(A_eff_file)
    A_eff = data["A_eff"]
    wl = data["wavelength"]
    return interp1d(wl, A_eff, fill_value=1, bounds_error=False)(l)


def load_custom_dispersion(dispersion_file: str) -> tuple[np.ndarray, np.ndarray]:
    disp_file = np.load(dispersion_file)
    wl_for_disp = disp_file["wavelength"]
    interp_range = (np.min(wl_for_disp), np.max(wl_for_disp))
    D = disp_file["dispersion"]
    beta2 = D_to_beta2(D, wl_for_disp)
    return wl_for_disp, beta2, interp_range


def load_custom_loss(l: np.ndarray, loss_file: str) -> np.ndarray:
    """loads a npz loss file that contains a wavelength and a loss entry

    Parameters
    ----------
    l : np.ndarray, shape (n,)
        wavelength array of the simulation
    loss_file : str
        relative or absolute path to the loss file

    Returns
    -------
    np.ndarray, shape (n,)
        loss in 1/m units
    """
    loss_data = np.load(loss_file)
    wl = loss_data["wavelength"]
    loss = loss_data["loss"]
    return interp1d(wl, loss, fill_value=0, bounds_error=False)(l)


@np_cache
def dispersion_coefficients(
    w_for_disp: np.ndarray,
    beta2_arr: np.ndarray,
    w0: float,
    interpolation_degree: int,
):
    """Computes the taylor expansion of beta2 to be used in dispersion_op

    Parameters
    ----------
    wl_for_disp : 1D array
        wavelength
    beta2 : 1D array
        beta2 as function of wl_for_disp
    w0 : float
        pump angular frequency
    interpolation_degree : int
        degree of polynomial fit. Will return deg+1 coefficients

    Returns
    -------
        beta2_coef : 1D array
            Taylor coefficients in decreasing order
    """

    # we get the beta2 Taylor coeffiecients by making a fit around w0
    w_c = w_for_disp - w0

    w_c = w_c[2:-2]
    beta2_arr = beta2_arr[2:-2]
    fit = Chebyshev.fit(w_c, beta2_arr, interpolation_degree)
    poly_coef = cheb2poly(fit.convert().coef)
    beta2_coef = poly_coef * np.cumprod([1] + list(range(1, interpolation_degree + 1)))

    return beta2_coef


def dispersion_from_coefficients(
    w_c: np.ndarray, beta2_coefficients: Iterable[float]
) -> np.ndarray:
    """computes the dispersion profile (beta2) from the beta coefficients

    Parameters
    ----------
    w_c : np.ndarray, shape (n, )
        centered angular frequency (0 <=> pump frequency)
    beta2_coefficients : Iterable[float]
        beta coefficients

    Returns
    -------
    np.ndarray, shape (n, )
        beta2 as function of w_c
    """

    coef = np.array(beta2_coefficients) / np.cumprod([1] + list(range(1, len(beta2_coefficients))))
    beta_arr = np.zeros_like(w_c)
    for k, b in reversed(list(enumerate(coef))):
        beta_arr = beta_arr + b * w_c ** k
    return beta_arr


def delayed_raman_t(t: np.ndarray, raman_type: str) -> np.ndarray:
    """
    computes the unnormalized temporal Raman response function applied to the array t

    Parameters
    ----------
    t : 1D array
        time in the co-moving frame of reference
    raman_type : str {"stolen", "agrawal", "measured"}
        indicates what type of Raman effect modelization to use

    Returns
    -------
    hr_arr : 1D array
        temporal response function
    """
    tau1 = 12.2e-15
    tau2 = 32e-15
    t_ = t - t[0]
    t = t_
    if raman_type == "stolen":
        hr_arr = (tau1 / tau2 ** 2 + 1 / tau1) * np.exp(-t_ / tau2) * np.sin(t_ / tau1)

    elif raman_type == "agrawal":
        taub = 96e-15
        h_a = (tau1 / tau2 ** 2 + 1 / tau1) * np.exp(-t_ / tau2) * np.sin(t_ / tau1)
        h_b = (2 * taub - t_) / taub ** 2 * np.exp(-t_ / taub)
        hr_arr = 0.79 * h_a + 0.21 * h_b

    elif raman_type == "measured":
        try:
            path = utils.Paths.get("hr_t")
            loaded = np.load(path)
        except FileNotFoundError:
            print("Not able to find the measured Raman response function. Going with agrawal model")
            return delayed_raman_t(t, raman_type="agrawal")

        t_stored, hr_arr_stored = loaded["t"], loaded["hr_arr"]
        hr_arr = interp1d(t_stored, hr_arr_stored, bounds_error=False, fill_value=0)(t)
    else:
        print("invalid raman response function, aborting")
        quit()

    return hr_arr


def delayed_raman_w(t: np.ndarray, raman_type: str) -> np.ndarray:
    """returns the delayed raman response function as function of w
    see delayed_raman_t for detailes"""
    return fft(delayed_raman_t(t, raman_type)) * (t[1] - t[0])


def fast_poly_dispersion_op(w_c, beta_arr, power_fact_arr, where=slice(None)):
    """
    dispersive operator

    Parameters
    ----------
    w_c : 1d array
        angular frequencies centered around 0
    beta_arr : 1d array
        beta coefficients returned by scgenerator.physics.fiber.dispersion_coefficients
    power_fact : list of arrays of len == len(w_c)
        precomputed values for w_c^k / k!
    where : slice-like
        indices over which to apply the operator, otherwise 0

    Returns
    -------
    array of len == len(w_c)
        dispersive component
    """

    dispersion = _fast_disp_loop(np.zeros_like(w_c), beta_arr, power_fact_arr)

    out = np.zeros_like(dispersion)
    out[where] = dispersion[where]
    return -1j * out


def _fast_disp_loop(dispersion: np.ndarray, beta_arr, power_fact_arr):
    for k in range(len(beta_arr) - 1, -1, -1):
        dispersion = dispersion + beta_arr[k] * power_fact_arr[k]
    return dispersion


def direct_dispersion(w: np.ndarray, w0: float, n_eff: np.ndarray) -> np.ndarray:
    """returns the dispersive operator in direct form (without polynomial interpolation)
    i.e. -1j * (beta(w) - beta1 * (w - w0) - beta0)

    Parameters
    ----------
    w : np.ndarray
        angular frequency array
    w0 : float
        center frequency
    n_eff : np.ndarray
        effectiv refractive index

    Returns
    -------
    np.ndarray
        dispersive operator
    """
    w0_ind = argclosest(w, w0)
    return fast_direct_dispersion(w, w0, n_eff, w0_ind)


def fast_direct_dispersion(w: np.ndarray, w0: float, n_eff: np.ndarray, w0_ind: int) -> np.ndarray:
    beta_arr = beta(w, n_eff)
    beta1_arr = np.gradient(beta_arr, w)
    return -1j * (beta_arr - beta1_arr[w0_ind] * (w - w0) - beta_arr[w0_ind])


def effective_core_radius(wl_for_disp, core_radius, s=0.08, h=200e-9):
    """return the variable core radius according to Eq. S2.2 from Köttig2017

    Parameters
    ----------
    wl_for_disp : ndarray, shape (n, )
        array of wl over which to calculate the effective core radius
    core_radius : float
        physical core radius in m
    s : float
        s parameter from the equation S2.2
    h : float
        wall thickness in m

    Returns
    -------
        effective_core_radius : ndarray, shape (n, )
    """
    return core_radius / (1 + s * wl_for_disp ** 2 / (core_radius * h))


def effective_radius_HCARF(core_radius, t, f1, f2, wl_for_disp):
    """eq. 3 in Hasan 2018"""
    return f1 * core_radius * (1 - f2 * wl_for_disp ** 2 / (core_radius * t))


def capillary_loss(wl: np.ndarray, he_mode: tuple[int, int], core_radius: float) -> np.ndarray:
    """computes the wavelenth dependent capillary loss according to Marcatili

    Parameters
    ----------
    wl : np.ndarray, shape (n, )
        wavelength array
    he_mode : tuple[int, int]
        tuple of mode (n, m)
    core_radius : float
        in m

    Returns
    -------
    np.ndarray
        loss in 1/m
    """
    chi_silica = mat.sellmeier(wl, utils.load_material_dico("silica"))
    nu_n = 0.5 * (chi_silica + 2) / np.sqrt(chi_silica)
    return nu_n * (u_nm(*he_mode) * wl / pipi) ** 2 * core_radius ** -3


def extinction_distance(loss: T, ratio=1 / e) -> T:
    return np.log(ratio) / -loss
