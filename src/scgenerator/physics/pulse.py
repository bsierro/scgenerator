"""
This files incluedes funcitons used by the scgenerator module to compute properties of pulses.
This include computing initial pulse shape and pulse noise as well as transforming the pulse
or measuring its properties.

NOTE
the term `sc-ordering` is used throughout this module. An array that follows sc-ordering is
of shape `([what, ever,] n, nt)` (could be just `(n, nt)` for 2D sc-ordered array) such that
n is the number of spectra at the same z position and nt is the size of the time/frequency grid
"""

import itertools
import os
from dataclasses import astuple, dataclass, fields
from pathlib import Path
from typing import Literal, Tuple, TypeVar

import matplotlib.pyplot as plt
import numpy as np
from numpy import pi
from numpy.fft import fft, fftshift, ifft
from scipy.interpolate import UnivariateSpline
from scipy.optimize import minimize_scalar
from scipy.optimize.optimize import OptimizeResult

from scgenerator import utils

from ..defaults import default_plotting
from ..logger import get_logger
from ..math import *
from . import units

c = 299792458.0
hbar = 1.05457148e-34
T = TypeVar("T")

#
fwhm_to_T0_fac = dict(
    sech=1 / (2 * np.log(1 + np.sqrt(2))),
    gaussian=1 / (np.sqrt(2 * np.log(2))),
)
"""relates the fwhm of the intensity profile (amplitue^2) to the t0 parameter of the amplitude"""

P0T0_to_E0_fac = dict(
    sech=2,  # int(a * sech(x / b)^2 * dx) from -inf to inf = 2 * a * b
    gaussian=np.sqrt(pi / 2),  # int(a * exp(-(x/b)^2)^2 * dx) from -inf to inf = sqrt(pi/2) * a * b
)
"""relates the total energy (amplitue^2) to the t0 parameter of the amplitude and the peak intensity (peak_amplitude^2)"""


@dataclass
class PulseProperties:
    quality: float
    mean_coherence: float
    fwhm_noise: float
    mean_fwhm: float
    peak_rin: float
    energy_rin: float
    timing_jitter: float

    @classmethod
    def header(cls, delimiter: str = ",", quotechar: str = "") -> str:
        return delimiter.join(
            quotechar + f + quotechar
            for f in [
                "quality",
                "mean_coherence",
                "fwhm_noise",
                "mean_fwhm",
                "peak_rin",
                "energy_rin",
                "timing_jitter",
            ]
        )

    @classmethod
    def save_all(
        cls,
        destination: os.PathLike,
        *props: "PulseProperties",
        delimiter: str = ",",
        quotechar: str = "",
    ):
        out = np.zeros((len(props), 7))
        for i, p in enumerate(props):
            out[i] = astuple(p)
        np.savetxt(
            destination,
            out,
            header=cls.header(delimiter=delimiter, quotechar=quotechar),
            delimiter=delimiter,
        )

    @classmethod
    def load(cls, path: os.PathLike, delimiter: str = ",") -> list["PulseProperties"]:
        arr = np.loadtxt(path, delimiter=delimiter)
        return [cls(*a) for a in arr]


def initial_field(t: np.ndarray, shape: str, t0: float, peak_power: float) -> np.ndarray:
    """returns the initial field

    Parameters
    ----------
    t : 1d array
        time array
    shape : str {"gaussian", "sech"}
        shape of the pulse
    t0 : float
        time parameters. Can be obtained by dividing the FWHM by
        `scgenerator.physics.pulse.fwhm_to_T0_fac[shape]`
    peak_power : float
        peak power

    Returns
    -------
    1d array
        field array

    Raises
    ------
    ValueError
        raised when shape is not recognized
    """
    if shape == "gaussian":
        return gaussian_pulse(t, t0, peak_power)
    elif shape == "sech":
        return sech_pulse(t, t0, peak_power)
    else:
        raise ValueError(f"shape '{shape}' not understood")


def modify_field_ratio(
    t: np.ndarray,
    field: np.ndarray,
    target_power: float = None,
    target_energy: float = None,
    intensity_noise: float = None,
    noise_correlation: float = 0,
) -> float:
    """multiply a field by this number to get the desired effects

    Parameters
    ----------
    t : np.ndarray
        time (only used when target_energy is not None)
    field : np.ndarray
        initial field
    target_power : float, optional
        abs2(field).max() == target_power, by default None
    intensity_noise : float, optional
        intensity noise, by default None

    Returns
    -------
    float
        ratio (multiply field by this number)
    """
    ratio = 1
    if target_energy is not None:
        ratio *= np.sqrt(target_energy / np.trapz(abs2(field), t))
    elif target_power is not None:
        ratio *= np.sqrt(target_power / abs2(field).max())

    if intensity_noise is not None:
        d_int, _ = technical_noise(intensity_noise, noise_correlation)
        ratio *= np.sqrt(d_int)
    return ratio


def convert_field_units(envelope: np.ndarray, n: np.ndarray, A_eff: float) -> np.ndarray:
    """[summary]

    Parameters
    ----------
    envelope : np.ndarray, shape (n,)
        complex envelope in units such that |envelope|^2 is in W
    n : np.ndarray, shape (n,)
        refractive index
    A_eff : float
        effective mode field area in m^2

    Returns
    -------
    np.ndarray, shape (n,)
        real field in V/m
    """
    return 2 * envelope.real / np.sqrt(2 * units.epsilon0 * units.c * n * A_eff)


def conform_pulse_params(
    shape: Literal["gaussian", "sech"],
    width: float = None,
    t0: float = None,
    peak_power: float = None,
    energy: float = None,
    soliton_num: float = None,
    gamma: float = None,
    beta2: float = None,
):
    """makes sure all parameters of the pulse are set and consistent

    Parameters
    ----------
    shape : str {"gaussian", "sech"}
        shape of the pulse
    width : float, optional
        fwhm of the intensity pulse, by default None
    t0 : float, optional
        time parameter of the amplitude pulse, by default None
    peak_power : float, optional
        peak power, by default None
    energy : float, optional
        total energy of the pulse, by default None
    soliton_num : float, optional
        soliton number, by default None
    gamma : float, optional
        nonlinear parameter, by default None
    beta2 : float, optional
        second order dispersion coefficient, by default None

    if more parameters than required are specified, the order of precedence
    indicated by the order in which the parameters are enumerated below holds,
    meaning the superflous parameters will be overwritten.
    choose one of the possible combinations :
        1 of (width, t0), 1 of (peak_power, energy), gamma and beta2 together optional (not one without the other)
        soliton_num, gamma, 1 of (width, peak_power, energy, t0)
    examples :
        specify width, peak_power and energy -> t0 and energy will be computed
        specify soliton_num, gamma, peak_power, t0 -> width, t0 and energy will be computed

    Returns
    -------
    width, t0, peak_power, energy
        when no gamma is specified
    width, t0, peak_power, energy, soliton_num
        when gamma is specified

    Raises
    ------
    TypeError
        [description]
    """

    if gamma is not None and beta2 is None or beta2 is not None and gamma is None:
        raise TypeError("when soliton number is desired, both gamma and beta2 must be specified")

    if soliton_num is not None:
        if gamma is None:
            raise TypeError("gamma must be specified when soliton_num is")

        if width is not None:
            peak_power = soliton_num ** 2 * abs(beta2) / (gamma * t0 ** 2)
        elif peak_power is not None:
            t0 = np.sqrt(soliton_num ** 2 * abs(beta2) / (peak_power * gamma))
        elif energy is not None:
            t0 = P0T0_to_E0_fac[shape] * soliton_num ** 2 * abs(beta2) / (energy * gamma)
        elif t0 is not None:
            width = t0 / fwhm_to_T0_fac[shape]
            peak_power = soliton_num ** 2 * abs(beta2) / (gamma * t0 ** 2)
        else:
            raise TypeError("not enough parameters to determine pulse")

    if width is not None:
        t0 = width * fwhm_to_T0_fac[shape]
    else:
        width = t0 / fwhm_to_T0_fac[shape]

    if peak_power is not None:
        energy = P0_to_E0(peak_power, t0, shape)
    else:
        peak_power = E0_to_P0(energy, t0, shape)

    if gamma is None:
        return width, t0, peak_power, energy
    else:
        if soliton_num is None:
            soliton_num = np.sqrt(peak_power * gamma * t0 ** 2 / abs(beta2))
        return width, t0, peak_power, energy, soliton_num


def t0_to_width(t0: float, shape: str):
    return t0 / fwhm_to_T0_fac[shape]


def width_to_t0(width: float, shape: str):
    return width * fwhm_to_T0_fac[shape]


def mean_power_to_energy(mean_power: float, repetition_rate: float) -> float:
    return mean_power / repetition_rate


def soliton_num_to_peak_power(soliton_num, beta2, gamma, t0):
    return soliton_num ** 2 * abs(beta2) / (gamma * t0 ** 2)


def soliton_num_to_t0(soliton_num, beta2, gamma, peak_power):
    return np.sqrt(soliton_num ** 2 * abs(beta2) / (peak_power * gamma))


def soliton_num(L_D, L_NL):
    return np.sqrt(L_D / L_NL)


def L_D(t0, beta2):
    return t0 ** 2 / abs(beta2)


def L_NL(peak_power, gamma):
    return 1 / (gamma * peak_power)


def L_sol(L_D):
    return pi / 2 * L_D


def load_and_adjust_field_file(
    field_file: str,
    t: np.ndarray,
    intensity_noise: float,
    noise_correlation: float,
    energy: float = None,
    peak_power: float = None,
) -> np.ndarray:
    field_0 = load_field_file(field_file, t)
    if energy is not None:
        curr_energy = np.trapz(abs2(field_0), t)
        field_0 = field_0 * np.sqrt(energy / curr_energy)
    elif peak_power is not None:
        ratio = np.sqrt(peak_power / abs2(field_0).max())
        field_0 = field_0 * ratio
    else:
        raise ValueError(f"Not enough parameters specified to load {field_file} correctly")

    field_0 = field_0 * modify_field_ratio(
        t, field_0, peak_power, energy, intensity_noise, noise_correlation
    )
    width, peak_power, energy = measure_field(t, field_0)
    return field_0, peak_power, energy, width


def load_field_file(field_file: str, t: np.ndarray) -> np.ndarray:
    field_data = np.load(field_file)
    field_interp = interp1d(
        field_data["time"], field_data["field"], bounds_error=False, fill_value=(0, 0)
    )
    field_0 = field_interp(t)
    return field_0


def correct_wavelength(init_wavelength: float, w_c: np.ndarray, field_0: np.ndarray) -> float:
    """
    finds a new wavelength parameter such that the maximum of the spectrum corresponding
    to field_0 is located at init_wavelength
    """
    delta_w = w_c[np.argmax(abs2(np.fft.fft(field_0)))]
    return units.m.inv(units.m(init_wavelength) - delta_w)


def E0_to_P0(E0, t0, shape):
    """convert an initial total pulse energy to a pulse peak peak_power"""
    return E0 / (t0 * P0T0_to_E0_fac[shape])


def P0_to_E0(P0, t0, shape):
    """converts initial peak peak_power to pulse energy"""
    return P0 * t0 * P0T0_to_E0_fac[shape]


def sech_pulse(t, t0, P0, offset=0):
    return np.sqrt(P0) / np.cosh((t - offset) / t0)


def gaussian_pulse(t, t0, P0, offset=0):
    return np.sqrt(P0) * np.exp(-(((t - offset) / t0) ** 2))


def photon_number(spectrum, w, dw, gamma) -> float:
    return np.sum(1 / gamma * abs2(spectrum) / w * dw)


def photon_number_with_loss(spectrum, w, dw, gamma, alpha, h) -> float:
    spec2 = abs2(spectrum)
    return np.sum(1 / gamma * spec2 / w * dw) - h * np.sum(alpha / gamma * spec2 / w * dw)


def pulse_energy(spectrum, dw) -> float:
    return np.sum(abs2(spectrum) * dw)


def pulse_energy_with_loss(spectrum, dw, alpha, h) -> float:
    spec2 = abs2(spectrum)
    return np.sum(spec2 * dw) - h * np.sum(alpha * spec2 * dw)


def technical_noise(rms_noise, noise_correlation=-0.4):
    """
    To implement technical noise as described in Grenier2019, we need to know the
    noise properties of the laser, summarized into the RMS amplitude noise

    Parameters
    ----------
        rms_noise : float
            RMS amplitude noise of the laser
        relative factor : float
            magnitude of the anticorrelation between peak_power and pulse width noise
    Returns
    ----------
        delta_int : float
        delta_T0 : float
    """
    psy = np.random.normal(1, rms_noise)
    return psy, 1 + noise_correlation * (psy - 1)


def shot_noise(w_c, w0, T, dt):
    """

    Parameters
    ----------
        w_c : 1D array
            angular frequencies centered around 0
        w0 : float
            pump angular frequency
        T : float
            length of the time windows
        dt : float
            resolution of time grid

    Returns
    ----------
        out : 1D array of size len(w_c)
            noise field to be added on top of initial field in time domain
    """
    rand_phase = np.random.rand(len(w_c)) * 2 * pi
    A_oppm = np.sqrt(hbar * (np.abs(w_c + w0)) * T) * np.exp(-1j * rand_phase)
    out = ifft(A_oppm / dt * np.sqrt(2 * pi))
    return out


def add_shot_noise(
    field_0: np.ndarray, quantum_noise: bool, w_c: bool, w0: float, time_window: float, dt: float
) -> np.ndarray:
    if quantum_noise:
        field_0 = field_0 + shot_noise(w_c, w0, time_window, dt)
    return field_0


def mean_phase(spectra):
    """computes the mean phase of spectra
    Parameter
    ----------
        spectra : 2D array
            The mean is taken on the 0th axis. This means the array has to be of shape (n, nt)
    Returns
    ----------
        mean_phase : 1D array of shape (len(spectra[0]))
            array of complex numbers of unit length representing the mean phase
    Example
    ----------
        >>> x = np.array([[1 + 1j, 0 + 2j, -3 - 1j],
                          [1 + 0j, 2 + 3j, -3 + 1j]])
        >>> mean_phase(x)
        array([ 0.92387953+0.38268343j,  0.28978415+0.95709203j,  -1.        +0.j        ])

    """

    total_phase = np.sum(
        spectra / np.abs(spectra),
        axis=0,
        where=spectra != 0,
        out=np.zeros(len(spectra[0]), dtype="complex"),
    )
    return (total_phase) / np.abs(total_phase)


def flatten_phase(spectra):
    """
    takes the mean phase out of an array of complex numbers

    Parameters
    ----------
        spectra : 2D array of shape (n, nt)
            spectra arranged in the same fashion as in `scgenerator.physics.pulse.mean_phase`

    Returns
    ----------
        output : array of same dimensions and amplitude, but with a flattened phase
    """
    mean_theta = mean_phase(spectra)
    tiled = np.tile(mean_theta, (len(spectra), 1))
    output = spectra * np.conj(tiled)
    return output


def compress_pulse(spectra):
    """given some complex spectrum, returns the compressed pulse in the time domain
    Parameters
    ----------
        spectra : ND array
            spectra to compress. The shape must be at least 2D. Compression occurs along the -2th axis.
            This means spectra have to be of shape ([what, ever,] n, nt) where n is the number of spectra
            brought together for one compression operation and nt the resolution of the grid.

    Returns
    ----------
        out : array of shape ([what, ever,] nt)
            compressed inverse Fourier-transformed pulse
    """
    if spectra.ndim > 2:
        return np.array([compress_pulse(spec) for spec in spectra])
    else:
        return fftshift(ifft(flatten_phase(spectra)), axes=1)


def ideal_compressed_pulse(spectra):
    """returns the ideal compressed pulse assuming flat phase
    Parameters
    ----------
        spectra : 2D array, sc-ordering
    Returns
    ----------
        compressed : 1D array
            time envelope of the compressed field
    """
    return abs2(fftshift(ifft(np.sqrt(np.mean(abs2(spectra), axis=0)))))


def spectrogram(time, values, t_res=256, t_win=24e-12, gate_width=200e-15, shift=False):
    """
    returns the spectorgram of the field given in values

    Parameters
    ----------
        time : 1D array-like
            time in the co-moving frame of reference
        values : 1D array-like
            field array that matches the time array
        t_res : int, optional
            how many "bins" the time array is subdivided into. Default : 256
        t_win : float, optional
            total time window (=length of time) over which the spectrogram is computed. Default : 24e-12
        gate_width : float, optional
            width of the gaussian gate function (=sqrt(2 log(2)) * FWHM). Default : 200e-15

    Returns
    ----------
        spec : 2D array
            real 2D spectrogram
        delays : 1D array of size t_res
            new time axis
    """
    t_lim = t_win / 2
    delays = np.linspace(-t_lim, t_lim, t_res)
    spec = np.zeros((t_res, len(time)))
    for i, delay in enumerate(delays):
        masked = values * np.exp(-(((time - delay) / gate_width) ** 2))
        spec[i] = abs2(fft(masked))
        if shift:
            spec[i] = fftshift(spec[i])
    return spec, delays


def g12(values):
    """
    computes the first order coherence function of a ensemble of values

    Parameters
    ----------
        values : 2D array
            complex values following sc-ordering
    return:
        g12_arr : coherence function as a n-D array
    """

    # Create all the possible pairs of values
    n = len(values)
    field_pairs = itertools.combinations(values, 2)
    mean_spec = np.mean(abs2(values), axis=0)
    mask = mean_spec > 1e-15 * mean_spec.max()
    corr = np.zeros_like(values[0])
    for pair in field_pairs:
        corr[mask] += pair[0][mask].conj() * pair[1][mask]
    corr[mask] = corr[mask] / (n * (n - 1) / 2 * mean_spec[mask])

    return np.abs(corr)


def avg_g12(values):
    """
    comptutes the average of the coherence function weighted by amplitude of spectrum

    Parameters
    ----------
        values : (m, n)-D array containing m complex values

    Returns
    ----------
        (float) average g12
    """

    if len(values.shape) > 2:
        pass

    avg_values = np.mean(abs2(values), axis=0)
    coherence = g12(values)
    return np.sum(coherence * avg_values) / np.sum(avg_values)


def fwhm_ind(values, mam=None):
    """returns the indices where values is bigger than half its maximum
    Parameters
    ----------
        values : array
            real values with ideally only one smooth peak
        mam : tupple (float, int)
            (maximum value, index of the maximum value)
    Returns
    ----------
        left_ind, right_ind : int
            indices of the the left and right spots where values drops below 1/2 the maximum
    """

    if mam is None:
        m = np.max(values)
        am = np.argmax(values)
    else:
        m, am = mam

    left_ind = am - np.where(values[am::-1] < m / 2)[0][0]
    right_ind = am + np.where(values[am:] < m / 2)[0][0]
    return left_ind - 1, right_ind + 1


def peak_ind(values, mam=None):
    """returns the indices that encapsulate the entire peak
    Parameters
    ----------
        values : array
            real values with ideally only one smooth peak
        mam : tupple (float, int)
            (maximum value, index of the maximum value)
    Returns
    ----------
        left_ind, right_ind : int
            indices of the the left and right spots where values starts rising again, with a margin of 3
    """

    if mam is None:
        m = np.max(values)
        am = np.argmax(values)
    else:
        m, am = mam

    try:
        left_ind = (
            am
            - np.where((values[am:0:-1] - values[am - 1 :: -1] <= 0) & (values[am:0:-1] < m / 2))[
                0
            ][0]
        ) - 3
    except IndexError:
        left_ind = 0
    try:
        right_ind = (
            am + np.where((values[am:-1] - values[am + 1 :] <= 0) & (values[am:-1] < m / 2))[0][0]
        ) + 3
    except IndexError:
        right_ind = len(values) - 1
    return max(0, left_ind), min(len(values) - 1, right_ind)


def setup_splines(x_axis, values, mam=None):
    """sets up spline interpolation to better measure a peak. Different splines with different orders are
    necessary because derivatives and second derivatives are computed to find extremea and inflection points
    Parameters
    ----------
        x_axis : 1D array
            domain of values
        values : 1D array
            real values that ideally contain only one smooth peak to measure
        mam : tupple (float, int)
            (maximum value, index of the maximum value)
    Returns
    ----------
        small_spline : scipy.interpolate.UnivariateSpline
            order 3 spline that interpolates `values - m/2` around the peak
        spline_4 : scipy.interpolate.UnivariateSpline
            order 4 spline that interpolate values around the peak
        spline 5 : scipy.interpolate.UnivariateSpline
            order 5 spline that interpolates values around the peak
        d_spline : scipy.interpolate.UnivariateSpline
            order 3 spline that interpolates the derivative of values around the peak
        d_roots : list
            roots of d_spline
        dd_roots : list
            inflection points of spline_5
        l_ind, r_ind : int
            return values of peak_ind
    """

    # Isolate part thats roughly above max/2
    l_ind_h, r_ind_h = fwhm_ind(values, mam)
    l_ind, r_ind = peak_ind(values, mam)

    if mam is None:
        mm = np.max(values)
    else:
        mm, _ = mam

    # Only roots of deg=3 splines can be computed, so we need 3 splines to find
    # zeros, local extrema and inflection points
    small_spline = UnivariateSpline(
        x_axis[l_ind_h : r_ind_h + 1], values[l_ind_h : r_ind_h + 1] - mm / 2, k=3, s=0
    )
    spline_4 = UnivariateSpline(x_axis[l_ind : r_ind + 1], values[l_ind : r_ind + 1], k=4, s=0)
    spline_5 = UnivariateSpline(x_axis[l_ind : r_ind + 1], values[l_ind : r_ind + 1], k=5, s=0)
    d_spline = spline_4.derivative()
    d_roots = spline_4.derivative().roots()
    dd_roots = spline_5.derivative(2).roots()

    return small_spline, spline_4, spline_5, d_spline, d_roots, dd_roots, l_ind, r_ind


def find_lobe_limits(x_axis, values, debug="", already_sorted=True):
    """find the limits of the centra lobe given 2 derivatives of the values and
    the position of the FWHM

    Parameters
    ----------
        x_axis : 1D array
            domain of values
        values : 1D array
            real values that present a peak whose properties we want to meausure
        debug : str
            if the peak is not distinct, a plot is made to assess the measurement
            providing a debug label can help identify which plot correspond to which function call
        sorted : bool
            faster computation if arrays are already sorted

    Returns
    ----------
        peak_lim : 1D array (left_lim, right_lim, peak_pos)
            values that delimit the left, right and maximum of the peak in units of x_axis
        fwhm_pos : 1D array (left_pos, right_pos)
            values corresponding to fwhm positions in units of x_axis
        good_roots : 1D array
            all candidate values that could delimit the peak position
        spline_4 : scipy.interpolate.UnivariateSpline
            order 4 spline that interpolate values around the peak
    """
    logger = get_logger(__name__)

    if not already_sorted:
        x_axis, values = x_axis.copy(), values.copy()
        values = values[np.argsort(x_axis)]
        x_axis.sort()

    debug_str = f"debug : {debug}" if debug != "" else ""

    small_spline, spline_4, spline_5, d_spline, d_roots, dd_roots, l_ind, r_ind = setup_splines(
        x_axis, values
    )

    # get premliminary values for fwhm limits and peak limits
    # if the peak is distinct, it should be sufficient
    fwhm_pos = np.array(small_spline.roots())
    peak_pos = d_roots[np.argmax(spline_4(d_roots))]

    # if there are more than 2 fwhm position, a detailed analysis can help
    # determining the true ones. If that fails, there is no meaningful peak to measure
    detailed_measurement = len(fwhm_pos) > 2
    if detailed_measurement:

        print("trouble measuring the peak.{}".format(debug_str))
        (
            spline_4,
            d_spline,
            d_roots,
            dd_roots,
            fwhm_pos,
            peak_pos,
            out_path,
            fig,
            ax,
            color,
        ) = _detailed_find_lobe_limits(
            x_axis,
            values,
            debug,
            debug_str,
            spline_4,
            spline_5,
            fwhm_pos,
            peak_pos,
            d_spline,
            d_roots,
            dd_roots,
            l_ind,
            r_ind,
        )

        good_roots, left_lim, right_lim = _select_roots(d_spline, d_roots, dd_roots, fwhm_pos)
        if debug != "":
            ax.scatter(
                [left_lim, right_lim],
                spline_4([left_lim, right_lim]),
                marker="|",
                label="lobe pos",
                c=color[5],
            )
            ax.legend()
            fig.savefig(out_path, bbox_inches="tight")
        plt.close()

    else:
        good_roots, left_lim, right_lim = _select_roots(d_spline, d_roots, dd_roots, fwhm_pos)

    return np.array([left_lim, right_lim, peak_pos]), fwhm_pos, np.array(good_roots), spline_4


def _select_roots(d_spline, d_roots, dd_roots, fwhm_pos):
    """selects the limits of a lobe

    Parameters
    ----------
    d_spline : scipy.interpolate.UnivariateSpline
        spline of the first derivative of the lobe
    d_roots : list
        roots of the first derivarive (extrema of the original function)
    dd_roots : list
        roots of the second derivative (inflection points of the original function)
    fwhm_pos : list
        locations where the lobe is half of its maximum

    Returns
    -------
    good_roots : list
        valid roots
    left_lim : list
        location of the left limit
    right_lim : list
        location of the right limit
    """
    # includes inflection points when slope is low (avoids considering the inflection points around fwhm limits)

    all_roots = np.append(d_roots, dd_roots)
    good_roots = all_roots[np.abs(d_spline(all_roots)) < np.max(d_spline(all_roots)) / 10]

    try:
        left_lim = np.max(good_roots[good_roots < np.min(fwhm_pos)])
    except ValueError:
        left_lim = np.min(good_roots)

    try:
        right_lim = np.min(good_roots[good_roots > np.max(fwhm_pos)])
    except ValueError:
        right_lim = np.max(good_roots)

    return good_roots, left_lim, right_lim


def _detailed_find_lobe_limits(
    x_axis,
    values,
    debug,
    debug_str,
    spline_4,
    spline_5,
    fwhm_pos,
    peak_pos,
    d_spline,
    d_roots,
    dd_roots,
    l_ind,
    r_ind,
):

    left_pos = fwhm_pos[fwhm_pos < peak_pos]
    right_pos = fwhm_pos[fwhm_pos > peak_pos]

    iterations = 0

    # spline maximum may not be on same peak as the original one. In this
    # case it means that there is no distinct peak, but we try to
    # compute everything again anyway. If spline inaccuracies lead to a cycle,
    # we break it by choosing two values arbitrarily

    while len(left_pos) == 0 or len(right_pos) == 0:
        if iterations > 4:

            left_pos, right_pos = [np.min(peak_pos)], [np.max(peak_pos)]
            print(
                "Cycle had to be broken. Peak measurement is probably wrong : {}".format(debug_str)
            )
            break
        else:
            iterations += 1

        mam = (spline_4(peak_pos), argclosest(x_axis, peak_pos))

        small_spline, spline_4, spline_5, d_spline, d_roots, dd_roots, l_ind, r_ind = setup_splines(
            x_axis, values, mam
        )

        fwhm_pos = np.array(small_spline.roots())
        peak_pos = d_roots[np.argmax(spline_4(d_roots))]

        left_pos = fwhm_pos[fwhm_pos < peak_pos]
        right_pos = fwhm_pos[fwhm_pos > peak_pos]

    # if measurement of the peak is not straightforward, we plot the situation to see
    # if the final measurement is good or not

    out_path, fig, ax = (
        (Path(f"measurement_errors_plots/it_{iterations}_{debug}"), *plt.subplots())
        if debug != ""
        else (None, None, None)
    )

    new_fwhm_pos = np.array([np.max(left_pos), np.min(right_pos)])

    # PLOT

    color = default_plotting["color_cycle"]
    if debug != "":
        newx = np.linspace(*span(x_axis[l_ind : r_ind + 1]), 1000)
        ax.plot(x_axis[l_ind - 5 : r_ind + 6], values[l_ind - 5 : r_ind + 6], c=color[0])
        ax.plot(newx, spline_5(newx), c=color[1])
        ax.scatter(fwhm_pos, spline_4(fwhm_pos), marker="+", label="all fwhm", c=color[2])
        ax.scatter(peak_pos, spline_4(peak_pos), marker=".", label="peak pos", c=color[3])
        ax.scatter(new_fwhm_pos, spline_4(new_fwhm_pos), marker="_", label="2 chosen", c=color[4])

    fwhm_pos = new_fwhm_pos
    return (
        spline_4,
        d_spline,
        d_roots,
        dd_roots,
        fwhm_pos,
        peak_pos,
        out_path,
        fig,
        ax,
        color,
    )


def measure_properties(spectra, t, compress=True, return_limits=False, debug="") -> PulseProperties:
    """measure the quality factor, the fwhm variation, the peak power variation,

    Parameters
    ----------
    spectra : np.ndarray, shape (n, nt)
        set of n spectra on a grid of nt angular frequency points
    t : np.ndarray, shape (nt, )
        time axis of the simulation
    compress : bool, optional
        whether to perform pulse compression. Default value is True, but this
        should be set to False to measure the initial pulse as output by gaussian_pulse
        or sech_pulse because compressing it would result in glitches and wrong measurements
    return_limits : bool, optional
        return the time values of the limits

    Returns
    ----------
    PulseProperties object:
        quality : float
            quality factor of the pulse ensemble
        mean_gmean_coherence12 : float
            mean coherence of the spectra ensemble
        fwhm_noise : float
            relative noise in temporal width of the (compressed) pulse
        mean_fwhm : float
            width of the mean (compressed) pulse
        peak_rin : float
            relative noise in the (compressed) pulse peak intensity
        energy_rin : float
            relative noise in the (compressed) pulse total_energy
        timing_jitter : float
            standard deviantion in absolute temporal peak position
    all_limits : list[tuple[np.ndarray, np.ndarray]], only if return_limits = True
        list of tuples of the form ([left_lobe_lim, right_lobe_lim, lobe_pos], [left_hm, right_hm])
    """
    if compress:
        fields = abs2(compress_pulse(spectra))
    else:
        fields = abs2(ifft(spectra))

    field = np.mean(fields, axis=0)
    ideal_field = abs2(fftshift(ifft(np.sqrt(np.mean(abs2(spectra), axis=0)))))

    # Isolate whole central lobe of bof mean and ideal field
    lobe_lim, fwhm_lim, _, big_spline = find_lobe_limits(t, field, debug)
    lobe_lim_i, _, _, big_spline_i = find_lobe_limits(t, ideal_field, debug)

    # Compute quality factor
    energy_fraction = (big_spline.integral(*span(lobe_lim[:2]))) / np.trapz(field, x=t)
    energy_fraction_i = (big_spline_i.integral(*span(lobe_lim_i[:2]))) / np.trapz(ideal_field, x=t)
    qf = energy_fraction / energy_fraction_i

    # Compute mean coherence
    mean_g12 = avg_g12(spectra)
    fwhm_abs = length(fwhm_lim)

    # To compute amplitude and fwhm fluctuations, we need to measure every single peak
    P0 = []
    fwhm = []
    t_offset = []
    energies = []
    all_lims: list[tuple[np.ndarray, np.ndarray]] = []
    for f in fields:
        lobe_lim, fwhm_lim, _, big_spline = find_lobe_limits(t, f, debug)
        all_lims.append((lobe_lim, fwhm_lim))
        P0.append(big_spline(lobe_lim[2]))
        fwhm.append(length(fwhm_lim))
        t_offset.append(lobe_lim[2])
        energies.append(np.trapz(fields, t))
    fwhm_var = np.std(fwhm) / np.mean(fwhm)
    int_var = np.std(P0) / np.mean(P0)
    en_var = np.std(energies) / np.mean(energies)
    t_jitter = np.std(t_offset)

    if isinstance(mean_g12, np.ndarray) and mean_g12.ndim == 0:
        mean_g12 = mean_g12[()]
    pp = PulseProperties(qf, mean_g12, fwhm_var, fwhm_abs, int_var, en_var, t_jitter)

    if return_limits:
        return pp, all_lims
    else:
        return pp


def rin_curve(spectra: np.ndarray) -> np.ndarray:
    """computes the rin curve, i.e. the rin at every single point

    Parameters
    ----------
    spectra : np.ndarray, shape (n, nt)
        a collection of n spectra from which to compute the RIN

    Returns
    -------
    rin_curve : np.ndarray
        RIN curve
    """
    A2 = abs2(spectra)
    return np.std(A2, axis=0) / np.mean(A2, axis=0)


def measure_field(t: np.ndarray, field: np.ndarray) -> Tuple[float, float, float]:
    """returns fwhm, peak_power, energy"""
    intensity = abs2(field)
    _, fwhm_lim, _, _ = find_lobe_limits(t, intensity)
    fwhm = length(fwhm_lim)
    peak_power = intensity.max()
    energy = np.trapz(intensity, t)
    return fwhm, peak_power, energy


def remove_2nd_order_dispersion(
    spectrum: T, w_c: np.ndarray, beta2: float, max_z: float = -100.0
) -> tuple[T, OptimizeResult]:
    """attempts to remove 2nd order dispersion from a complex spectrum

    Parameters
    ----------
    spectrum : np.ndarray or Spectrum, shape (n, )
        spectrum from which to remove 2nd order dispersion
    w_c : np.ndarray, shape (n, )
        corresponding centered angular frequencies (w-w0)
    beta2 : float
        2nd order dispersion coefficient

    Returns
    -------
    np.ndarray, shape (n, )
        spectrum with 2nd order dispersion removed
    """
    propagate = lambda z: spectrum * np.exp(-0.5j * beta2 * w_c ** 2 * z)

    def score(z):
        return -np.max(abs2(np.fft.ifft(propagate(z))))

    opti = minimize_scalar(score, bracket=(max_z, 0))
    return propagate(opti.x), opti


def remove_2nd_order_dispersion2(
    spectrum: T, w_c: np.ndarray, max_gdd: float = 1000e-30
) -> tuple[T, OptimizeResult]:
    """attempts to remove 2nd order dispersion from a complex spectrum

    Parameters
    ----------
    spectrum : np.ndarray or Spectrum, shape (n, )
        spectrum from which to remove 2nd order dispersion
    w_c : np.ndarray, shape (n, )
        corresponding centered angular frequencies (w-w0)

    Returns
    -------
    np.ndarray, shape (n, )
        spectrum with 2nd order dispersion removed
    """
    propagate = lambda gdd: spectrum * np.exp(-0.5j * w_c ** 2 * 1e-30 * gdd)
    integrate = lambda gdd: abs2(np.fft.ifft(propagate(gdd)))

    def score(gdd):
        return -np.sum(integrate(gdd) ** 6)

    # def score(gdd):
    #     return -np.max(integrate(gdd))

    # to_test = np.linspace(-max_gdd, max_gdd, 200)
    # scores = [score(g) for g in to_test]
    # fig, ax = plt.subplots()
    # ax.plot(to_test, scores / np.min(scores))
    # plt.show()
    # plt.close(fig)
    # ama = np.argmin(scores)

    opti = minimize_scalar(score, bounds=(-max_gdd * 1e30, max_gdd * 1e30))
    opti["x"] *= 1e-30
    return propagate(opti.x * 1e30), opti
