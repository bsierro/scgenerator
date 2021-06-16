from typing import Any, Dict, List, Tuple

import numpy as np
import toml
from numpy.fft import fft, ifft
from numpy.polynomial.chebyshev import Chebyshev, cheb2poly
from scipy.interpolate import interp1d

from .. import io
from ..math import abs2, argclosest, power_fact, u_nm
from ..utils.parameter import BareParams, hc_model_specific_parameters
from ..utils.cache import np_cache
from . import materials as mat
from . import units
from .units import c, pi


def lambda_for_dispersion():
    """Returns a wl vector for dispersion calculation

    Returns
    -------
    array of wl values
    """
    return np.linspace(190e-9, 3000e-9, 4000)


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


def HCARF_gap(core_radius: float, capillary_num: int, capillary_outer_d: float):
    """computes the gap length between capillaries of a hollow core anti-resonance fiber

    Parameters
    ----------
    core_radius : float
        radius of the core (m) (from cented to edge of a capillary)
    capillary_num : int
        number of capillaries
    capillary_outer_d : float
        diameter of the capillaries including the wall thickness(m). The core together with the microstructure has a diameter of 2R + 2d

    Returns
    -------
    gap : float
    """
    return (core_radius + capillary_outer_d / 2) * 2 * np.sin(
        pi / capillary_num
    ) - capillary_outer_d


def dispersion_parameter(n_eff: np.ndarray, lambda_: np.ndarray):
    """computes the dispersion parameter D from an effective index of refraction n_eff
    Since computing gradients/derivatives of discrete arrays is not well defined on the boundary, it is
    advised to chop off the two values on either end of the returned array

    Parameters
    ----------
    n_eff : 1D array
        a wl-dependent index of refraction
    lambda_ : 1D array
        the wavelength array (len must match n_eff)

    Returns
    -------
    D : 1D array
        wl-dependent dispersion parameter as function of lambda_
    """

    return -lambda_ / c * (np.gradient(np.gradient(n_eff, lambda_), lambda_))


def beta2_to_D(beta2, λ):
    """returns the D parameter corresponding to beta2(λ)"""
    return -(2 * pi * c) / (λ ** 2) * beta2


def D_to_beta2(D, λ):
    """returns the beta2 parameters corresponding to D(λ)"""
    return -(λ ** 2) / (2 * pi * c) * D


def plasma_dispersion(lambda_, number_density, simple=False):
    """computes dispersion (beta2) for constant plasma

    Parameters
    ----------
    lambda_ : array-like
        wavelengths over which to calculate the dispersion
    number_density : number of ionized atoms /m^3

    Returns
    -------
    beta2 : ndarray
        WL-dependent dispersion parameter
    """

    e2_me_e0 = 3182.60735  # e^2 /(m_e * epsilon_0)
    w = units.m(lambda_)
    if simple:
        w_pl = number_density * e2_me_e0
        return -(w_pl ** 2) / (c * w ** 2)

    beta = w / c * np.sqrt(1 - number_density * e2_me_e0 / w ** 2)
    beta2 = np.gradient(np.gradient(beta, w), w)
    return beta2


def n_eff_marcatili(lambda_, n_gas_2, core_radius, he_mode=(1, 1)):
    """computes the effective refractive index according to the Marcatili model of a capillary

    Parameters
    ----------
    lambda_ : ndarray, shape (n, )
        wavelengths array (m)
    n_gas_2 : ndarray, shape (n, )
        refractive index of the gas as function of lambda_
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

    return np.sqrt(n_gas_2 - (lambda_ * u / (2 * pi * core_radius)) ** 2)


def n_eff_marcatili_adjusted(lambda_, n_gas_2, core_radius, he_mode=(1, 1), fit_parameters=()):
    """computes the effective refractive index according to the Marcatili model of a capillary but adjusted at longer wavelengths

    Parameters
    ----------
    lambda_ : ndarray, shape (n, )
        wavelengths array (m)
    n_gas_2 : ndarray, shape (n, )
        refractive index of the gas as function of lambda_
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

    corrected_radius = effective_core_radius(lambda_, core_radius, *fit_parameters)

    return np.sqrt(n_gas_2 - (lambda_ * u / (2 * pi * corrected_radius)) ** 2)


@np_cache
def n_eff_hasan(
    lambda_: np.ndarray,
    n_gas_2: np.ndarray,
    core_radius: float,
    capillary_num: int,
    capillary_thickness: float,
    capillary_outer_d: float = None,
    capillary_spacing: float = None,
    capillary_resonance_strengths: list[float] = [],
    capillary_nested: int = 0,
) -> np.ndarray:
    """computes the effective refractive index of the fundamental mode according to the Hasan model for a anti-resonance fiber

    Parameters
    ----------
    lambda_
        wavelenghs array (m)
    n_gas_2 : ndarray, shape (n, )
        squared refractive index of the gas as a function of lambda_
    core_radius : float
        radius of the core (m) (from cented to edge of a capillary)
    capillary_num : int
        number of capillaries
    capillary_thickness : float
        thickness of the capillaries (m)
    capillary_outer_d : float, optional if capillary_spacing is given
        diameter of the capillaries including the wall thickness(m). The core together with the microstructure has a diameter of 2R + 2d
    capillary_spacing : float, optional if capillary_outer_d is given
        spacing between capillaries (m)
    capillary_resonance_strengths : list or tuple, optional
        strengths of the resonance lines. default : []
    capillary_nested : int, optional
        number of levels of nested capillaries. default : 0

    Returns
    -------
    n_eff : ndarray, shape (n, )
        the effective refractive index as function of wavelength

    Reference
    ----------
    Hasan, Md Imran, Nail Akhmediev, and Wonkeun Chang. "Empirical formulae for dispersion and effective mode area in hollow-core antiresonant fibers." Journal of Lightwave Technology 36.18 (2018): 4060-4065.
    """
    u = u_nm(1, 1)
    if capillary_spacing is None:
        capillary_spacing = HCARF_gap(core_radius, capillary_num, capillary_outer_d)
    elif capillary_outer_d is None:
        capillary_outer_d = (2 * core_radius * np.sin(pi / capillary_num) - capillary_spacing) / (
            1 - np.sin(pi / capillary_num)
        )
    Rg = core_radius / capillary_spacing

    f1 = 1.095 * np.exp(0.097041 / Rg)
    f2 = 0.007584 * capillary_num * np.exp(0.76246 / Rg) - capillary_num * 0.002 + 0.012
    if capillary_nested > 0:
        f2 += 0.0045 * np.exp(-4.1589 / (capillary_nested * Rg))

    R_eff = f1 * core_radius * (1 - f2 * lambda_ ** 2 / (core_radius * capillary_thickness))

    n_eff_2 = n_gas_2 - (u * lambda_ / (2 * pi * R_eff)) ** 2

    chi_sil = mat.sellmeier(lambda_, io.load_material_dico("silica"))

    with np.errstate(divide="ignore", invalid="ignore"):
        for m, strength in enumerate(capillary_resonance_strengths):
            n_eff_2 += (
                strength
                * lambda_ ** 2
                / (lambda_ ** 2 - chi_sil * (2 * capillary_thickness / (m + 1)) ** 2)
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


def HCPCF_find_with_given_ZDW(
    variable,
    target,
    search_range,
    material_dico,
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


def beta2(w: np.ndarray, n_eff: np.ndarray) -> np.ndarray:
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
    return np.gradient(np.gradient(n_eff * w / c, w), w)


def HCPCF_dispersion(
    lambda_,
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
    lambda_ : ndarray, shape (n, )
        wavelengths over which to calculate the dispersion
    material_dico : dict
        material dictionary respecting standard format explained in FIXME
    model : string {"marcatili", "marcatili_adjusted", "hasan"}
        which model of effective refractive index to use
    model_params : tuple
        to be cast to the function in charge of computing the effective index of the fiber. Every n_eff_* function has a signature
        n_eff_(lambda_, n_gas_2, radius, *args) and model_params corresponds to args
    temperature : float
        Temperature of the material
    pressure : float
        constant pressure
        FIXME tupple : a pressure gradient from pressure[0] to pressure[1] is computed

    Returns
    -------
    out : 1D array
        beta2 as function of wavelength
    """

    w = units.m(lambda_)
    if material_dico is None:
        n_gas_2 = np.ones_like(lambda_)
    else:
        if ideal:
            n_gas_2 = mat.sellmeier(lambda_, material_dico, pressure, temperature) + 1
        else:
            N_1 = mat.number_density_van_der_waals(
                pressure=pressure, temperature=temperature, material_dico=material_dico
            )
            N_0 = mat.number_density_van_der_waals(material_dico=material_dico)
            n_gas_2 = mat.sellmeier(lambda_, material_dico) * N_1 / N_0 + 1

    n_eff_func = dict(
        marcatili=n_eff_marcatili, marcatili_adjusted=n_eff_marcatili_adjusted, hasan=n_eff_hasan
    )[model]
    n_eff = n_eff_func(lambda_, n_gas_2, **model_params)

    return beta2(w, n_eff)


def dynamic_HCPCF_dispersion(
    lambda_: np.ndarray,
    pressure_values: List[float],
    core_radius: float,
    fiber_model: str,
    model_params: Dict[str, Any],
    temperature: float,
    ideal_gas: bool,
    w0: float,
    interp_range: Tuple[float, float],
    material_dico: Dict[str, Any],
    deg: int,
):
    """returns functions for beta2 coefficients and gamma instead of static values

    Parameters
    ----------
    lambda_ : wavelength array
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
    pressure = lambda r: mat.pressure_from_gradient(r, *pressure_values)
    beta2 = lambda r: HCPCF_dispersion(
        lambda_,
        core_radius,
        material_dico,
        fiber_model,
        model_params,
        pressure(r),
        temperature,
        ideal_gas,
    )

    n2 = lambda r: mat.non_linear_refractive_index(material_dico, pressure(r), temperature)
    ratio_range = np.linspace(0, 1, 256)

    gamma_grid = np.array([gamma_parameter(n2(r), w0, A_eff) for r in ratio_range])
    gamma_interp = interp1d(ratio_range, gamma_grid)

    beta2_grid = np.array(
        [dispersion_coefficients(lambda_, beta2(r), w0, interp_range, deg) for r in ratio_range]
    )
    beta2_interp = [
        interp1d(ratio_range, beta2_grid[:, i], assume_sorted=True) for i in range(deg + 1)
    ]

    def beta2_func(r):
        return [beta2_interp[i](r)[()] for i in range(deg + 1)]

    def gamma_func(r):
        return gamma_interp(r)[()]

    return beta2_func, gamma_func


def gamma_parameter(n2, w0, A_eff):
    return n2 * w0 / (A_eff * c)


@np_cache
def PCF_dispersion(lambda_, pitch, ratio_d, w0=None, n2=None, A_eff=None):
    """
    semi-analytical computation of the dispersion profile of a triangular Index-guiding PCF

    Parameters
    ----------
    lambda_ : 1D array-like
        wavelengths over which to calculate the dispersion
    pitch : float
        distance between air holes in m
    ratio_d : float
        ratio diameter of hole / pitch
    w0 : float, optional
        pump angular frequency. If given, the gamma value is also returned in adition to the GVD. default : None

    Returns
    -------
    beta2 : 1D array
        Dispersion parameter as function of wavelength
    gamma : float
        non-linear coefficient

    Reference
    ---------
        Formulas and values are from Saitoh K and Koshiba M, "Empirical relations for simple design of photonic crystal fibers" (2005)

    """
    # Check validity
    if ratio_d < 0.2 or ratio_d > 0.8:
        print("WARNING : Fitted formula valid only for pitch ratio between 0.2 and 0.8")

    n_co = 1.45
    a_eff = pitch / np.sqrt(3)
    pi2a = 2 * pi * a_eff

    ratio_l = lambda_ / pitch

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

    A = ai0 + ai1 * ratio_d ** bi1 + ai2 * ratio_d ** bi2 + ai3 * ratio_d ** bi3
    B = ci0 + ci1 * ratio_d ** di1 + ci2 * ratio_d ** di2 + ci3 * ratio_d ** di3

    V = A[0] + A[1] / (1 + A[2] * np.exp(A[3] * ratio_l))
    W = B[0] + B[1] / (1 + B[2] * np.exp(B[3] * ratio_l))

    n_FSM2 = 1.45 ** 2 - (lambda_ * V / (pi2a)) ** 2
    n_eff2 = (lambda_ * W / (pi2a)) ** 2 + n_FSM2
    n_eff = np.sqrt(n_eff2)

    D_wave_guide = dispersion_parameter(n_eff, lambda_)

    material_dico = io.load_material_dico("silica")
    chi_mat = mat.sellmeier(lambda_, material_dico)
    D_mat = dispersion_parameter(np.sqrt(chi_mat + 1), lambda_)

    # material index of refraction (Sellmeier formula)

    D = D_wave_guide + D_mat

    beta2 = D_to_beta2(D, lambda_)

    if w0 is None:
        return beta2

    else:
        # effective mode field area (koshiba2004)
        if A_eff is None:
            V_eff = pi2a / lambda_ * np.sqrt(n_co ** 2 - n_FSM2)
            w_eff = a_eff * (0.65 + 1.619 / V_eff ** 1.5 + 2.879 / V_eff ** 6)
            A_eff = interp1d(lambda_, w_eff, kind="linear")(units.m.inv(w0)) ** 2 * pi

        if n2 is None:
            n2 = 2.6e-20
        gamma = gamma_parameter(n2, w0, A_eff)

        return beta2, gamma


def compute_dispersion(params: BareParams):
    """dispatch function depending on what type of fiber is used

    Parameters
    ----------
    fiber_model : str {"PCF", "HCPCF"}
        describes the type of fiber
        - PCF : triangular Index-guiding photonic crystal fiber
        - HCPCF : hollow core fiber (filled with gas, or not)
    params : dict
        parameter dictionary as in `parameters.toml`

    Returns
    -------
    beta2_coef : 1D array of size deg
        beta coefficients to be used in disp_op
    gamma : float
        nonlinear parameter
    """

    if params.dispersion_file is not None:
        disp_file = np.load(params.dispersion_file)
        lambda_ = disp_file["wavelength"]
        D = disp_file["dispersion"]
        beta2 = D_to_beta2(D, lambda_)
        gamma = None
    else:
        lambda_ = lambda_for_dispersion()
        beta2 = np.zeros_like(lambda_)

        if params.model == "pcf":
            beta2, gamma = PCF_dispersion(
                lambda_,
                params.pitch,
                params.pitch_ratio,
                w0=params.w0,
                n2=params.n2,
                A_eff=params.A_eff,
            )

        else:
            # Load material info
            gas_name = params.gas_name

            if gas_name == "vacuum":
                material_dico = None
            else:
                material_dico = toml.loads(io.Paths.gets("gas"))[gas_name]

            # compute dispersion
            if params.dynamic_dispersion:
                return dynamic_HCPCF_dispersion(
                    lambda_,
                    params.pressure,
                    params.core_radius,
                    params.model,
                    {k: getattr(params, k) for k in hc_model_specific_parameters[params.model]},
                    params.temperature,
                    params.ideal_gas,
                    params.w0,
                    params.interp_range,
                    material_dico,
                    params.interp_degree,
                )
            else:

                # actually compute the dispersion

                beta2 = HCPCF_dispersion(
                    lambda_,
                    material_dico,
                    params.model,
                    {k: getattr(params, k) for k in hc_model_specific_parameters[params.model]},
                    params.pressure,
                    params.temperature,
                    params.ideal_gas,
                )

                if material_dico is not None:
                    A_eff = 1.5 * params.core_radius ** 2
                    n2 = mat.non_linear_refractive_index(
                        material_dico, params.pressure, params.temperature
                    )
                    gamma = gamma_parameter(n2, params.w0, A_eff)
                else:
                    gamma = None

            # add plasma if wanted
            if params.plasma_density > 0:
                beta2 += plasma_dispersion(lambda_, params.plasma_density)

    beta2_coef = dispersion_coefficients(
        lambda_, beta2, params.w0, params.interp_range, params.interp_degree
    )

    if gamma is None:
        if params.A_eff is not None:
            gamma = gamma_parameter(params.n2, params.w0, params.A_eff)
        else:
            gamma = 0

    return beta2_coef, gamma


@np_cache
def dispersion_coefficients(
    lambda_: np.ndarray, beta2: np.ndarray, w0: float, interp_range=None, deg=8
):
    """Computes the taylor expansion of beta2 to be used in dispersion_op

    Parameters
    ----------
    lambda_ : 1D array
        wavelength
    beta2 : 1D array
        beta2 as function of lambda_
    w0 : float
        pump angular frequency
    interp_range : slice-like
        index-style specifying wl range over which to fit to get beta2 coefficients
    deg : int
        degree of polynomial fit. Will return deg+1 coefficients

    Returns
    -------
        beta2_coef : 1D array
            Taylor coefficients in decreasing order
    """

    if interp_range is None:
        r = slice(2, -2)
    else:
        # 2 discrete gradients are computed before getting to
        # beta2, so we need to make sure coefficients are not affected
        # by edge effects
        r = (lambda_ > max(lambda_[2], interp_range[0])) & (
            lambda_ < min(lambda_[-2], interp_range[1])
        )

    # we get the beta2 Taylor coeffiecients by making a fit around w0
    w_c = units.m(lambda_) - w0
    fit = Chebyshev.fit(w_c[r], beta2[r], deg)
    beta2_coef = cheb2poly(fit.convert().coef) * np.cumprod([1] + list(range(1, deg + 1)))

    return beta2_coef


def delayed_raman_t(t, raman_type="stolen"):
    """
    computes the unnormalized temporal Raman response function applied to the array t

    Parameters
    ----------
    t : 1D array
        time in the co-moving frame of reference
    raman_type : str {"stolen", "agrawal", "measured"}
        indicates what type of Raman effect modelization to use
        default : "stolen"

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
            path = io.Paths.get("hr_t")
            loaded = np.load(path)
        except FileNotFoundError:
            print(
                f"Not able to find the measured Raman response function. Going with agrawal model"
            )
            return delayed_raman_t(t, raman_type="agrawal")

        t_stored, hr_arr_stored = loaded["t"], loaded["hr_arr"]
        hr_arr = interp1d(t_stored, hr_arr_stored, bounds_error=False, fill_value=0)(t)
    else:
        print("invalid raman response function, aborting")
        quit()

    return hr_arr


def delayed_raman_w(t, dt, raman_type="stolen"):
    """returns the delayed raman response function as function of w
    see delayed_raman_t for detailes"""
    return fft(delayed_raman_t(t, raman_type)) * dt


def create_non_linear_op(behaviors, w_c, w0, gamma, raman_type="stolen", f_r=0, hr_w=None):
    """
    Creates a non-linear operator with the desired features

    Parameters
    ----------
    behaviors : list of str
        behaviors wanted
    w_c : 1d array
        symetric frequency array generated by scgenerator.initialize.wspace
    w0 : float
        pump angular frenquency
    gamma : float
        nonlinear parameter
    raman_type : str, optional
        name of the raman response function model. default : "stolen"
    f_r : float, optional
        fractional contribution of the delayed raman effect. default : 0
    hr_w : 1d array, optional unless "raman" in behaviors
        pre-calculated frequency-dependent delayed raman response function

    returns
    -------
    func
        a function to be passed to RK4IP which takes a spectrum as input and returns
        a new spectrum modified with the non-linear interactions.
    """

    # Compute raman response function if necessary
    if "raman" in behaviors:
        if hr_w is None:
            raise ValueError("freq-dependent Raman response must be give")
        if f_r == 0:
            f_r = 0.18
            if raman_type == "agrawal":
                f_r = 0.245

    if "spm" in behaviors:
        spm_part = lambda fi: (1 - f_r) * abs2(fi)
    else:
        spm_part = lambda fi: 0

    if "raman" in behaviors:
        raman_part = lambda fi: f_r * ifft(hr_w * fft(abs2(fi)))
    else:
        raman_part = lambda fi: 0

    ss_part = w_c / w0 if "ss" in behaviors else 0

    if isinstance(gamma, (float, int)):

        def N_func(spectrum: np.ndarray, r=0):
            field = ifft(spectrum)
            return -1j * gamma * (1 + ss_part) * fft(field * (spm_part(field) + raman_part(field)))

    else:

        def N_func(spectrum: np.ndarray, r=0):
            field = ifft(spectrum)
            return (
                -1j * gamma(r) * (1 + ss_part) * fft(field * (spm_part(field) + raman_part(field)))
            )

    return N_func


def fast_dispersion_op(w_c, beta_arr, power_fact_arr, where=slice(None)):
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


def dispersion_op(w_c, beta_arr, where=None):
    """
    dispersive operator

    Parameters
    ----------
    w_c : 1d array
        angular frequencies centered around 0
    beta_arr : 1d array
        beta coefficients returned by scgenerator.physics.fiber.dispersion_coefficients
    where : indices over which to apply the operatory, otherwise 0

    Returns
    -------
    disp_arr : dispersive component as an array of len = len(w_c)
    """

    dispersion = np.zeros_like(w_c)

    for k, beta in reversed(list(enumerate(beta_arr))):
        dispersion = dispersion + beta * power_fact(w_c, k + 2)

    out = np.zeros_like(dispersion)
    out[where] = dispersion[where]

    return -1j * out


def _get_radius(radius_param, lambda_=None):
    if isinstance(radius_param, tuple) and lambda_ is not None:
        return effective_core_radius(lambda_, *radius_param)
    else:
        return radius_param


def effective_core_radius(lambda_, core_radius, s=0.08, h=200e-9):
    """return the variable core radius according to Eq. S2.2 from Köttig2017

    Parameters
    ----------
    lambda_ : ndarray, shape (n, )
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
    return core_radius / (1 + s * lambda_ ** 2 / (core_radius * h))


def effective_radius_HCARF(core_radius, t, f1, f2, lambda_):
    """eq. 3 in Hasan 2018"""
    return f1 * core_radius * (1 - f2 * lambda_ ** 2 / (core_radius * t))
