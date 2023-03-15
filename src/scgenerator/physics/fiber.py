from typing import Iterable, TypeVar

import numpy as np
from numpy import e
from numpy.fft import fft
from numpy.polynomial.chebyshev import Chebyshev, cheb2poly
from scipy.interpolate import interp1d

from scgenerator import utils
from scgenerator.cache import np_cache
from scgenerator.math import argclosest, u_nm
from scgenerator.physics import materials as mat
from scgenerator.physics import units
from scgenerator.physics.units import c, pi

pipi = 2 * pi
T = TypeVar("T")


def lambda_for_envelope_dispersion(
    l: np.ndarray, interpolation_range: tuple[float, float]
) -> tuple[np.ndarray, np.ndarray]:
    """Returns a wl vector for dispersion calculation in envelope mode

    Returns
    -------
    np.ndarray
        subset of l in the interpolation range with two extra values on each side
        to accomodate for taking gradients
    np.ndarray
        indices of the original l where the values are valid (i.e. without the two extra on each side)
    """
    su = np.where((l >= interpolation_range[0]) & (l <= interpolation_range[1]))[0]
    if l[su].min() > 1.01 * interpolation_range[0]:
        raise ValueError(
            f"lower range of {1e9*interpolation_range[0]:.1f}nm is not reached by the grid. "
            "try a finer grid"
        )

    ind_above_cond = su >= len(l) // 2
    ind_above = su[ind_above_cond]
    ind_below = su[~ind_above_cond]
    fu = np.concatenate((ind_below, (ind_below + 2)[-2:], (ind_above - 2)[:2], ind_above))
    fs = fu[np.argsort(l[fu])[::-1]]
    l_out = l[fs]
    ind_out = fs[2:-2]
    return l_out, ind_out


def lambda_for_full_field_dispersion(
    l: np.ndarray, interpolation_range: tuple[float, float]
) -> tuple[np.ndarray, np.ndarray]:
    """Returns a wl vector for dispersion calculation in full field mode

    Returns
    -------
    np.ndarray
        subset of l in the interpolation range with two extra values on each side
        to accomodate for taking gradients
    np.ndarray
        indices of the original l where the values are valid (i.e. without the two extra on each side)
    """
    su = np.where((l >= interpolation_range[0]) & (l <= interpolation_range[1]))[0]
    if l[su].min() > 1.01 * interpolation_range[0]:
        raise ValueError(
            f"lower range of {1e9*interpolation_range[0]:.1f}nm is not reached by the grid. "
            "try a finer grid"
        )
    fu = np.concatenate((su[:2] - 2, su, su[-2:] + 2))
    fu = np.where(fu < 0, 0, fu)
    fu = np.where(fu >= len(l), len(l) - 1, fu)
    return l[fu], su


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
    return -(pipi * c) / (wl_for_disp**2) * beta2


def D_to_beta2(D, wl_for_disp):
    """returns the beta2 parameters corresponding to D(wl_for_disp)"""
    return -(wl_for_disp**2) / (pipi * c) * D


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
        return -(w_pl**2) / (c * w**2)

    beta2_arr = beta2(w, np.sqrt(1 - number_density * e2_me_e0 / w**2))
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
    return 1.5 * core_radius**2


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
    n_si_2 = mat.n_gas_2(wl_for_disp, "silica", None, None)
    return (
        wl_for_disp
        / (4 * np.sqrt(n_si_2))
        * (2 * order + 1)
        * (1 - n_gas_2 / n_si_2 + wl_for_disp**2 / (4 * n_si_2 * core_radius**2)) ** -0.5
    )


def resonance_strength(
    wl_for_disp: np.ndarray, core_radius: float, capillary_thickness: float, order: int
) -> float:
    a = 1.83 + (2.3 * capillary_thickness / core_radius)
    n_si = np.sqrt(mat.n_gas_2(wl_for_disp, "silica", None, None))
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
    Hasan, Md Imran, Nail Akhmediev, and Wonkeun Chang. "Empirical formulae for dispersion and
    effective mode area in hollow-core antiresonant fibers." Journal of Lightwave Technology 36.18
    (2018): 4060-4065.
    """
    u = u_nm(1, 1)
    alpha = 5e-12

    Rg = core_radius / capillary_spacing

    f1 = 1.095 * np.exp(0.097041 / Rg)
    f2 = 0.007584 * capillary_num * np.exp(0.76246 / Rg) - capillary_num * 0.002 + 0.012
    if capillary_nested > 0:
        f2 += 0.0045 * np.exp(-4.1589 / (capillary_nested * Rg))

    R_eff = f1 * core_radius * (1 - f2 * wl_for_disp**2 / (core_radius * capillary_thickness))

    n_eff_2 = n_gas_2 - (u * wl_for_disp / (pipi * R_eff)) ** 2

    chi_sil = mat.Sellmeier.load("silica").chi(wl_for_disp)

    with np.errstate(divide="ignore", invalid="ignore"):
        for m, strength in enumerate(capillary_resonance_strengths):
            n_eff_2 += (
                strength
                * wl_for_disp**2
                / (alpha + wl_for_disp**2 - chi_sil * (2 * capillary_thickness / (m + 1)) ** 2)
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
    return M_f * core_radius**2 * np.exp((capillary_spacing / 22e-6) ** 2.5)


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

    n_FSM2 = 1.45**2 - (l * V / (pi2a)) ** 2
    V_eff = pi2a / l * np.sqrt(n_co**2 - n_FSM2)

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


def frame_velocity(beta1_arr: np.ndarray, w0_ind: int) -> float:
    return 1.0 / beta1_arr[w0_ind]


def HCPCF_dispersion(
    wl_for_disp,
    gas_name: str,
    model="marcatili",
    pressure=None,
    temperature=None,
    **model_params,
):
    """returns the dispersion profile (beta_2) of a hollow-core photonic crystal fiber.

    Parameters
    ----------
    wl_for_disp : ndarray, shape (n, )
        wavelengths over which to calculate the dispersion
    gas_name : str
        name of the filling gas in lower case
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
    n_gas_2 = mat.Sellmeier.load(gas_name).n_gas_2(wl_for_disp, temperature, pressure)

    n_eff_func = dict(
        marcatili=n_eff_marcatili, marcatili_adjusted=n_eff_marcatili_adjusted, hasan=n_eff_hasan
    )[model]
    n_eff = n_eff_func(wl_for_disp, n_gas_2, **model_params)

    return beta2(w, n_eff)


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

    n_FSM2 = 1.45**2 - (wl_for_disp * V / (pi2a)) ** 2
    n_eff2 = (wl_for_disp * W / (pi2a)) ** 2 + n_FSM2
    n_eff = np.sqrt(n_eff2)

    chi_mat = mat.Sellmeier.load("silica").chi(wl_for_disp)
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

    A = ai0 + ai1 * pitch_ratio**bi1 + ai2 * pitch_ratio**bi2 + ai3 * pitch_ratio**bi3
    B = ci0 + ci1 * pitch_ratio**di1 + ci2 * pitch_ratio**di2 + ci3 * pitch_ratio**di3
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
    if interpolation_degree < 2:
        raise ValueError(f"interpolation_degree must be at least 2, got {interpolation_degree}")
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
        beta_arr = beta_arr + b * w_c**k
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
        hr_arr = (tau1 / tau2**2 + 1 / tau1) * np.exp(-t_ / tau2) * np.sin(t_ / tau1)

    elif raman_type == "agrawal":
        taub = 96e-15
        h_a = (tau1 / tau2**2 + 1 / tau1) * np.exp(-t_ / tau2) * np.sin(t_ / tau1)
        h_b = (2 * taub - t_) / taub**2 * np.exp(-t_ / taub)
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
    return core_radius / (1 + s * wl_for_disp**2 / (core_radius * h))


def effective_radius_HCARF(core_radius, t, f1, f2, wl_for_disp):
    """eq. 3 in Hasan 2018"""
    return f1 * core_radius * (1 - f2 * wl_for_disp**2 / (core_radius * t))


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
    chi_silica = abs(mat.Sellmeier.load("silica").chi(wl))
    # the real loss alpha is 2*Im(n_eff), which differs from the notation of the paper
    nu_n = (chi_silica + 2) / np.sqrt(chi_silica)
    return nu_n * (u_nm(*he_mode) * wl / pipi) ** 2 * core_radius**-3


def extinction_distance(loss: T, ratio=1 / e) -> T:
    return np.log(ratio) / -loss


def L_eff(loss: T, length: float) -> T:
    return -np.expm1(-loss * length) / loss


def core_radius_from_capillaries(tube_radius: float, gap: float, n_tubes: int) -> float:
    k = 1 + 0.5 * gap / tube_radius
    return tube_radius * (k / np.sin(np.pi / n_tubes) - 1)


def gap_from_capillaries(core_radius: float, tube_radius: float, n_tubes: int) -> float:
    s = np.sin(np.pi / n_tubes)
    return 2 * (s * (tube_radius + core_radius) - tube_radius)


def tube_radius_from_gap(core_radius: float, gap: float, n_tubes: int) -> float:
    s = np.sin(np.pi / n_tubes)
    return (core_radius * s - 0.5 * gap) / (1 - s)


def normalized_frequency_vincetti(
    wl: np.ndarray, thickness: float, n_clad_2: np.ndarray, n_gas_2: np.ndarray
) -> np.ndarray:
    """
    eq. 3 of [1] in n_eff_vincetti

    Parameters
    ----------
    wl : ndarray
        wavelength array
    thickness : float
        thickness of the capillary tube
    n_clad_2 : ndarray
        real refractive index of the cladding squared corresponding to wavelengths in wl
    n_gas_2 : ndarray
        real refractive index of the filling gas squared
    """
    return 2 * thickness / wl * np.sqrt(n_clad_2 - n_gas_2)


def effective_core_radius_vincetti(
    wl: np.ndarray, f: np.ndarray, r: float, g: float, n: int
) -> np.ndarray:
    """
    Parameters
    ----------
    wl : ndarray
        wavelength array
    f : ndarray
        corresponding normalized frequency
    r : float
        capillary external radius
    g : float
        gap size bewteen capillaries
    n : int
        number of tubes
    """
    r_co = core_radius_from_capillaries(r, g, n)
    factor = 1.027 + 0.001 * (f + 2 / f**4)
    #                                          | Missing in paper
    #                                          V
    inner = r_co**2 + n / np.pi * 0.046875 * r**2 * (1 + (3 + 20 * wl / r_co) * g / r)
    return factor * np.sqrt(inner)


def li_vincetti(f_2: T, f0_2: float) -> T:
    k = f0_2 - f_2
    return k / (k**2 + 9e-4 * f_2)


def cutoff_frequency_he_vincetti(mu: int, nu: int, t_ratio: float, n_clad_0: float) -> np.ndarray:
    """
    eq. (4) in [2] of n_eff_vincetti
    Parameters
    ----------
    mu : int
        azimuthal mode number
    nu : int
        radial mode number
    t_ratio : float
        t/r_ext
    n_clad_0 : float
        refractive index of the cladding material, generally at the pump wavelength
    """
    if nu == 1:
        base = np.abs(0.21 + 0.175 * mu - 0.1 * (mu - 0.35) ** -2)
        corr = 0.04 * np.sqrt(mu) * t_ratio
        return base * t_ratio ** (0.55 + 5e-3 * np.sqrt(n_clad_0**4 - 1)) + corr
    elif nu >= 2:
        return 0.3 / n_clad_0**0.3 * (0.5 * nu) ** -1.2 * np.abs(mu - 0.8) * t_ratio + nu - 1
    else:
        raise ValueError(f"nu must be a strictly positive integer, got {nu}")


def cutoff_frequency_eh_vincetti(mu: int, nu: int, t_ratio: float, n_clad_0: float) -> np.ndarray:
    """
    eq. (5) in [2] of n_eff_vincetti
    Parameters
    ----------
    mu : int
        azimuthal mode number
    nu : int
        radial mode number
    t_ratio : float
        t/r_ext
    n_clad_0 : float
        refractive index of the cladding material, generally at the pump wavelength
    """
    if nu == 1:
        base = 0.73 + 0.1425 * (mu**0.8 + 1.5) - 0.04 / (mu - 0.35)
        expo = 0.5 - 0.1 * (n_clad_0 - 1) * (mu + 0.5) ** -0.1
        corr = 0
    elif nu >= 2:
        base = (
            (11.5 * nu**-1.2 / (7.75 - nu))
            * (0.34 + 0.25 * mu * (n_clad_0 / 1.2) ** 1.15)
            * (mu + 0.2 / n_clad_0) ** -0.15
        )
        expo = 0.75 + 0.06 * n_clad_0**-1.15 + 0.1 * np.sqrt(1.44 / n_clad_0) * (nu - 2)
        corr = nu - 1
    else:
        raise ValueError(f"nu must be a positive integer, got {nu}")
    return base * t_ratio**expo + corr


def v_sum_vincetti(f: np.ndarray, t_ratio: float, n_clad_0: np.ndarray, n_terms: int) -> np.ndarray:
    f_2 = f**2
    out = np.zeros_like(f)
    for nu in range(1, n_terms + 1):
        out[:] += li_vincetti(f_2, cutoff_frequency_he_vincetti(1, nu, t_ratio, n_clad_0) ** 2)
        out[:] += li_vincetti(f_2, cutoff_frequency_eh_vincetti(1, nu, t_ratio, n_clad_0) ** 2)
    out *= 2e3
    return out


def n_eff_correction_vincetti(
    wl: np.ndarray, f: np.ndarray, t_ratio: float, r_co: float, n_clad_0: float, n_terms: int
) -> np.ndarray:
    """
    eq. 6 from [1] in n_eff_vincetti

    Parameters
    ----------
    wl : np.ndarray
        wavelength array
    f : np.ndarray
        corresponding normalized frequency
    t_ratio : float
        t (tube thickness) / r (external tube radius)
    r_co : float
        core radius
    n_clad_0 : float
        refractive index of the cladding (usu. at pump wavelength)
    n_terms : int
        how many resonances to calulcate
    """
    factor = 4.5e-7 / (1 - t_ratio) ** 4 * (wl / r_co) ** 2
    return factor * v_sum_vincetti(f, t_ratio, n_clad_0, n_terms)


def n_eff_vincetti(
    wl_for_disp: np.ndarray,
    wavelength: float,
    n_gas_2: np.ndarray,
    thickness: float,
    tube_radius: float,
    gap: float,
    n_tubes: int,
    n_terms: int = 8,
    n_clad_2: np.ndarray | None = None,
):
    """
    Parameters
    ----------
    wl_for_disp : ndarray
        wavelength (m) array over which to compute the refractive index
    wavelength : float
        center wavelength / pump wavelength
    n_gas_2 : ndarray
        n^2 of the filling gas
    thickness : float
        thickness of the structural capillary tubes
    tube_radius : float
        external radius of the strucural capillaries
    gap : float
        gap between the structural capillary tubes
    n_tubes : int
        number of capillary tubes
    n_terms : int
        number of resonances to calculate, by default 8
    n_clad_2: ndarray | None
        n^2 of the cladding, by default Silica

    Returns
    -------
    effective refractive index according to the Vincetti model



    Internal symbols help
    ---------------------
    f: normalized frequency
    r_co_eff: effective core radius


    References
    ----------
    [1] ROSA, Lorenzo, MELLI, Federico, et VINCETTI, Luca. Analytical Formulas for Dispersion and
        Effective Area in Hollow-Core Tube Lattice Fibers. Fibers, 2021, vol. 9, no 10, p. 58.
    [2] VINCETTI, Luca et ROSA, Lorenzo. A simple analytical model for confinement loss estimation
        kin hollow-core Tube Lattice Fibers. Optics Express, 2019, vol. 27, no 4, p. 5230-5237.

    """

    if n_clad_2 is None:
        n_clad_2 = mat.Sellmeier.load("silica").n_gas_2(wl_for_disp)

    n_clad_0 = np.sqrt(n_clad_2[argclosest(wl_for_disp, wavelength)])

    f = normalized_frequency_vincetti(wl_for_disp, thickness, n_clad_2, n_gas_2)
    r_co_eff = effective_core_radius_vincetti(wl_for_disp, f, tube_radius, gap, n_tubes)
    r_co = core_radius_from_capillaries(tube_radius, gap, n_tubes)
    d_n_eff = n_eff_correction_vincetti(
        wl_for_disp, f, thickness / tube_radius, r_co, n_clad_0, n_terms
    )

    n_gas = np.sqrt(n_gas_2)

    # eq. (21) in [1]
    return n_gas - 0.125 / n_gas * (u_nm(1, 1) * wl_for_disp / (np.pi * r_co_eff)) ** 2 + d_n_eff
