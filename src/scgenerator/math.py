"""
collection of purely mathematical function
"""

import math
from dataclasses import dataclass
from functools import cache
from typing import Sequence

import numba
import numpy as np
from scipy.interpolate import interp1d, lagrange
from scipy.special import jn_zeros

pi = np.pi
c = 299792458.0


def expm1_int(y: np.ndarray, dx: float) -> np.ndarray:
    """evaluates 1 - exp( -∫func(y(x))dx ) from x=-inf to x"""
    return -np.expm1(-cumulative_simpson(y) * dx)


def span(*vec: np.ndarray) -> tuple[float, float]:
    """returns the min and max of whatever array-like is given. can accept many args"""
    out = (np.inf, -np.inf)
    if len(vec) == 0 or len(vec[0]) == 0:
        raise ValueError("did not provide any value to span")
    for x in vec:
        x = np.atleast_1d(x)
        out = (min(np.min(x), out[0]), max(np.max(x), out[1]))
    return out


def total_extent(*vec: np.ndarray) -> float:
    """measure the distance between the min and max value of all given arrays"""
    left, right = span(*vec)
    return right - left


def argclosest(array: np.ndarray, target: float | int | Sequence[float | int]) -> int | np.ndarray:
    """
    returns the index/indices corresponding to the closest matches of target in array

    Parameters
    ----------
    array : np.ndarray, shape (n,)
        array of values
    target : number | np.ndarray
        find the closest value to target in `array`. The index of the closest match is returned.

    Returns
    -------
    int | np.ndarray
        index / indices of the closest match

    """
    min_dist = np.inf
    index = None
    if isinstance(target, (list, tuple, np.ndarray)):
        return np.array([argclosest(array, t) for t in target])
    for k, val in enumerate(array):
        dist = abs(val - target)
        if dist < min_dist:
            min_dist = dist
            index = k

    return index


def extent(x):
    return np.max(x) - np.min(x)


def power_fact(x, n):
    """
    returns x ^ n / n!
    """
    if isinstance(x, (int, float)):
        return _power_fact_single(x, n)

    elif isinstance(x, np.ndarray):
        return _power_fact_array(x, n)
    else:
        raise TypeError(f"type {type(x)} of x not supported.")


def _power_fact_single(x, n):
    result = 1.0
    for k in range(n):
        result = result * x / (n - k)
    return result


def _power_fact_array(x, n):
    """returns x^2/n!"""
    result = np.ones(len(x), dtype=np.float64)
    for k in range(n):
        result = result * x / (n - k)
    return result


def abs2(z: np.ndarray) -> np.ndarray:
    return z.real**2 + z.imag**2


def normalized(z: np.ndarray) -> np.ndarray:
    ab = abs2(z)
    return ab / ab.max()


def sigmoid(x):
    return 1 / (np.exp(-x) + 1)


def u_nm(n, m):
    """returns the mth zero of the Bessel function of order n-1
    Parameters
    ----------
        n-1 : order of the Bessel function
        m : order of the zero
    Returns
    ----------
        float
    """
    return jn_zeros(n - 1, m)[-1]


def all_zeros(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """find all the x values such that y(x)=0 with linear interpolation"""
    pos = np.argwhere(y[1:] * y[:-1] < 0)[:, 0]
    m = (y[pos + 1] - y[pos]) / (x[pos + 1] - x[pos])
    return -y[pos] / m + x[pos]


def wspace(t, t_num=0):
    """
    frequency array such that x(t) <-> np.fft(x)(w)

    Parameters
    ----------
        t : float or array
            float : total width of the time window
            array : time array
        t_num : int-
            if t is a float, specifies the number of points

    Returns
    -------
        w : array
            linspace of frencies corresponding to t
    """
    if isinstance(t, (np.ndarray, list, tuple)):
        dt = t[1] - t[0]
        t_num = len(t)
        t = t[-1] - t[0] + dt
    else:
        dt = t / t_num
    w = 2 * pi * np.arange(t_num) / t
    w = np.where(w >= pi / dt, w - 2 * pi / dt, w)
    return w


def tspace(time_window=None, t_num=None, dt=None):
    """
    returns a time array centered on 0

    Parameters
    ----------
        time_window : float
            total time spanned
        t_num : int
            number of points
        dt : float
            time resolution

        at least 2 arguments must be given. They are prioritize as such
        t_num > time_window > dt

    Returns
    -------
        t : array
            a linearily spaced time array
    Raises
    ------
        TypeError
            missing at least 1 argument
    """
    if t_num is not None:
        if isinstance(time_window, (float, int)):
            return np.linspace(-time_window / 2, time_window / 2, int(t_num))
        elif isinstance(dt, (float, int)):
            time_window = (t_num - 1) * dt
            return np.linspace(-time_window / 2, time_window / 2, t_num)
    elif isinstance(time_window, (float, int)) and isinstance(dt, (float, int)):
        t_num = int(time_window / dt) + 1
        return np.linspace(-time_window / 2, time_window / 2, t_num)
    else:
        raise TypeError("not enough parameter to determine time vector")


def build_envelope_w_grid(t: np.ndarray, w0: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    convenience function to

    Parameters
    ----------
    t : np.ndarray, shape (t_num,)
        time array
    w0 : float
        center frequency

    Returns
    -------
    w_c : np.ndarray, shape (n, )
        centered angular frequencies in rad/s where 0 is the pump frequency
    w : np.ndarray, shape (n, )
        angular frequency grid
    w_order : np.ndarray, shape (n,)
        arrays of indices such that w[w_order] is sorted
    """
    w_c = wspace(t)
    w = w_c + w0
    w_order = np.argsort(w)
    return w_c, w, w_order


def build_full_field_w_grid(t_num: int, dt: float) -> tuple[np.ndarray, float, float, int]:
    """
    Paramters
    ---------
    t_num : int
        number of temporal sampling points
    dt : float
        uniform spacing between sample points

    Returns
    -------
    w : np.ndarray, shape (n, )
        angular frequency grid
    w_order : np.ndarray, shape (n,)
        arrays of indices such that w[w_order] is sorted
    """
    w = 2 * pi * np.fft.rfftfreq(t_num, dt)
    w_order = np.argsort(w)
    ind = w != 0
    l = np.zeros(len(w))
    l[ind] = 2 * pi * c / w[ind]
    if any(ind):
        l[~ind] = 2 * pi * c / (np.abs(w[ind]).min())
    return w, w_order, l


def build_z_grid(z_num: int, length: float) -> np.ndarray:
    return np.linspace(0, length, z_num)


def build_t_grid(
    time_window: float = None, t_num: int = None, dt: float = None
) -> tuple[np.ndarray, float, float, int]:
    """[summary]

    Returns
    -------
    t : np.ndarray, shape (t_num, )
        temporal points in s
    time_window : float
        total width of the temporal grid in s, by default None
    dt : float
        spacing of the temporal grid in s, by default None
    t_num : int
        number of temporal grid points, by default None
    """
    t = tspace(time_window, t_num, dt)
    time_window = t.max() - t.min()
    dt = t[1] - t[0]
    t_num = len(t)
    return t, time_window, dt, t_num


def polynom_extrapolation(x: np.ndarray, y: np.ndarray, degree: int) -> np.ndarray:
    """
    performs a polynomial extrapolation on both ends of the support

    Parameters
    ----------
    x : np.ndarray (n,)
        x values
    y : np.ndarray (n,)
        y values. The shape must correspond to that of x.
    degree : int
        degree of the polynom.

    Returns
    -------
    np.ndarray, shape (n,)
        y array with zero values on either side replaces with polynomial extrapolation

    Example



    """
    out = y.copy()
    order = np.argsort(x)
    left_ind, *_, right_ind = np.nonzero(out[order])[0]
    return _polynom_extrapolation_in_place(out[order], left_ind, right_ind, degree)[order.argsort]


def _polynom_extrapolation_in_place(y: np.ndarray, left_ind: int, right_ind: int, degree: float):
    """extrapolates IN PLACE linearily on both side of the support

    Parameters
    ----------
    y : np.ndarray
        array
    left_ind : int
        first value we want to keep (extrapolate to the left of that)
    right_ind : int
        last value we want to keep (extrapolate to the right of that)
    """
    r_left = (1 + np.arange(left_ind))[::-1] ** degree
    r_right = np.arange(len(y) - right_ind) ** degree
    y[:left_ind] = r_left * (y[left_ind] - y[left_ind + 1]) + y[left_ind]
    y[right_ind:] = r_right * (y[right_ind] - y[right_ind - 1]) + y[right_ind]
    return y


def interp_2d(
    old_x: np.ndarray,
    old_y: np.ndarray,
    z: np.ndarray,
    new_x: np.ndarray | tuple,
    new_y: np.ndarray | tuple,
    kind="linear",
) -> np.ndarray:
    if isinstance(new_x, tuple):
        new_x = np.linspace(*new_x)
    if isinstance(new_y, tuple):
        new_y = np.linspace(*new_y)
    z = interp1d(old_y, z, axis=0, kind=kind, bounds_error=False, fill_value=0)(new_y)
    return interp1d(old_x, z, kind=kind, bounds_error=False, fill_value=0)(new_x)


@numba.jit(nopython=True)
def linear_interp_2d(old_x: np.ndarray, old_y: np.ndarray, new_x: np.ndarray):
    new_vals = np.zeros((len(old_y), len(new_x)))
    interpolable = (new_x > old_x[0]) & (new_x <= old_x[-1])
    equal = new_x == old_x[0]
    inds = np.searchsorted(old_x, new_x[interpolable])
    for i, val in enumerate(old_y):
        new_vals[i][interpolable] = val[inds - 1] + (new_x[interpolable] - old_x[inds - 1]) * (
            val[inds] - val[inds - 1]
        ) / (old_x[inds] - old_x[inds - 1])
        new_vals[i][equal] = val[0]
    return new_vals


@numba.jit(nopython=True)
def linear_interp_1d(old_x: np.ndarray, old_y: np.ndarray, new_x: np.ndarray):
    new_vals = np.zeros(len(new_x))
    interpolable = (new_x > old_x[0]) & (new_x <= old_x[-1])
    inds = np.searchsorted(old_x, new_x[interpolable])
    new_vals[interpolable] = old_y[inds - 1] + (new_x[interpolable] - old_x[inds - 1]) * (
        old_y[inds] - old_y[inds - 1]
    ) / (old_x[inds] - old_x[inds - 1])
    new_vals[new_x == old_x[0]] = old_y[0]
    return new_vals


def envelope_ind(
    signal: np.ndarray, dmin: int = 1, dmax: int = 1, split: bool = False
) -> tuple[np.ndarray, np.ndarray]:
    """
    Attempts to separate the envolope from a periodic signal and return the indices of the
    top/bottom envelope of a signal

    Parameters
    ----------
    signal : np.ndarray, shape (n,)
        signal array (must be sorted)
    dmin, dmax : int, optional
        size of chunks for lower/upper envelope
    split: bool, optional
        split the signal in half along its mean, might help to generate the envelope in some cases
        this has the effect of forcing the envlopes to be on either side of the dc signal
        by default False

    Returns
    -------
    np.ndarray, shape (m,), m <= n
        lower envelope indices
    np.ndarray, shape (l,), l <= n
        upper envelope indices
    """

    local_min = (np.diff(np.sign(np.diff(signal))) > 0).nonzero()[0] + 1
    local_max = (np.diff(np.sign(np.diff(signal))) < 0).nonzero()[0] + 1

    if split:
        dc_value = np.mean(signal)
        local_min = local_min[signal[local_min] < dc_value]
        local_max = local_max[signal[local_max] > dc_value]

    if dmin > 1:
        local_min = local_min[
            [i + np.argmin(signal[local_min[i : i + dmin]]) for i in range(0, len(local_min), dmin)]
        ]
    if dmax > 1:
        local_max = local_max[
            [i + np.argmax(signal[local_max[i : i + dmax]]) for i in range(0, len(local_max), dmax)]
        ]

    return local_min, local_max


def envelope_2d(x: np.ndarray, values: np.ndarray) -> np.ndarray:
    """returns the envelope of a 2d propagation-like array

    Parameters
    ----------
    x : np.ndarray, shape (nt,)
        x axis
    values : np.ndarray, shape (nz, nt)
        values of which to find the envelope

    Returns
    -------
    np.ndarray, shape (nz, nt)
        interpolated values
    """
    return np.array([envelope_1d(x, y) for y in values])


def envelope_1d(y: np.ndarray, x: np.ndarray = None) -> np.ndarray:
    if x is None:
        x = np.arange(len(y))
    _, hi = envelope_ind(y)
    return interp1d(x[hi], y[hi], kind="cubic", fill_value=0, bounds_error=False)(x)


@dataclass(frozen=True)
class LobeProps:
    left_pos: float
    left_fwhm: float
    center: float
    right_fwhm: float
    right_pos: float
    interp: interp1d

    @property
    @cache
    def x(self) -> np.ndarray:
        return np.array(
            [self.left_pos, self.left_fwhm, self.center, self.right_fwhm, self.right_pos]
        )

    @property
    @cache
    def y(self) -> np.ndarray:
        return self.interp(self.x)

    @property
    @cache
    def fwhm(self) -> float:
        return abs(self.right_fwhm - self.left_fwhm)

    @property
    @cache
    def width(self) -> float:
        return abs(self.right_pos - self.left_pos)


def measure_lobe(x_in, y_in, /, lobe_pos: int = None, thr_rel: float = 1 / 50) -> LobeProps:
    """given a fairly smooth signal, finds the highest lobe and returns its position as well
    as its fwhm points

    Parameters
    ----------
    x_in : np.ndarray, shape (n,)
        x values
    y_in : np.ndarray, shape (n,)
        y values
    lobe_pos : int, optional
        index of the desired lobe, by default None (take highest peak)
    thr_rel : float, optional


    Returns
    -------
    np.ndarray
        (left limit, left half maximum, maximum position, right half maximum, right limit)
    """
    interp = interp1d(x_in, y_in)
    lobe_pos = lobe_pos or np.argmax(y_in)
    maxi = y_in[lobe_pos]
    maxi2 = maxi / 2
    thr_abs = maxi * thr_rel
    half_max_left = all_zeros(x_in[:lobe_pos], y_in[:lobe_pos] - maxi2)[-1]
    half_max_right = all_zeros(x_in[lobe_pos:], y_in[lobe_pos:] - maxi2)[0]

    poly = lagrange((half_max_left, x_in[lobe_pos], half_max_right), (maxi2, maxi2 * 2, maxi2))
    parabola_left, parabola_right = sorted(poly.roots)

    r_cand = x_in > half_max_right
    x_right = x_in[r_cand]
    y_right = y_in[r_cand]

    l_cand = x_in < half_max_left
    x_left = x_in[l_cand][::-1]
    y_left = y_in[l_cand][::-1]

    d = {}
    for x, y, central_parabola_root, sign in [
        (x_left, y_left, parabola_left, 1),
        (x_right, y_right, parabola_right, -1),
    ]:
        candidates = []
        slope = sign * np.gradient(y, x)

        for y_test, num_to_take in [
            (sign * np.gradient(slope, x), 2),
            (y - thr_abs, 1),
            (slope, 3),
        ]:
            candidates.extend(all_zeros(x, y_test)[:num_to_take])
        candidates = np.array(sorted(candidates))

        side_parabola_root = x[0] - 2 * y[0] / (sign * slope[0])
        weights = (
            np.abs(candidates - side_parabola_root)
            + np.abs(candidates - central_parabola_root)
            + interp(candidates) / thr_abs
        )
        d[sign] = candidates[np.argmin(weights)]

    return LobeProps(d[1], half_max_left, x_in[lobe_pos], half_max_right, d[-1], interp)


@numba.jit(nopython=True)
def cumulative_simpson(x: np.ndarray) -> np.ndarray:
    out = np.zeros_like(x)
    c1 = 1 / 3
    c2 = 4 / 3
    out[1] = (x[1] + x[0]) * 0.5
    for i in range(2, len(x)):
        out[i] = out[i - 2] + x[i - 2] * c1 + x[i - 1] * c2 + c1 * x[i]
    return out


@numba.jit(nopython=True)
def cumulative_boole(x: np.ndarray) -> np.ndarray:
    out = np.zeros_like(x)
    c1 = 14 / 45
    c2 = 64 / 45
    c3 = 24 / 45
    c = np.array([c1, c2, c3, c2, c1])
    ind = np.arange(5)
    out[ind] = cumulative_simpson(x[ind])
    for i in range(4, len(x)):
        out[i] = out[i - 4] + np.sum(c * x[i - 4 : i + 1])
    return out


def stencil_coefficients(stencil_points: Sequence, order: int) -> np.ndarray:
    """
    Reference
    ---------
    https://en.wikipedia.org/wiki/Finite_difference_coefficient#Arbitrary_stencil_points
    """
    mat = np.power.outer(stencil_points, np.arange(len(stencil_points))).T
    d = np.zeros(len(stencil_points))
    d[order] = math.factorial(order)
    return np.linalg.solve(mat, d)


@cache
def central_stencil_coefficients(n: int, order: int) -> np.ndarray:
    """
    returns the coefficients of a centered finite difference scheme

    Parameters
    ----------
    n : int
        number of points
    order : int
        order of differentiation
    """
    return stencil_coefficients(np.arange(n * 2 + 1) - n, order)


@cache
def stencil_coefficients_set(n: int, order: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    coefficients for a forward, centered and backward finite difference differentation scheme of
    order `order` that extends `n` points away from the evaluation point (`x_0`)
    """
    central = central_stencil_coefficients(n, order)
    left = stencil_coefficients(np.arange(n + 1), order)
    right = stencil_coefficients(-np.arange(n + 1)[::-1], order)
    return left, central, right


def differentiate_arr(
    values: np.ndarray,
    diff_order: int,
    extent: int | None = None,
    h: float = 1.0,
    correct_edges=True,
) -> np.ndarray:
    """
    takes a derivative of order `diff_order` using equally spaced values

    Parameters
    ---------
    values : ndarray
        equally spaced values
    diff_order : int
        order of differentiation
    extent : int, optional
        how many points away from the center the scheme uses. This determines accuracy.
        example: extent=6 means that 13 (6 on either side + center) points are used to evaluate the
        derivative at each point. by default diff_order + 2
    h : float, optional
        step size, by default 1.0
    correct_edges : bool, optional
        avoid artifacts by using forward/backward schemes on the edges, by default True

    Reference
    ---------
    https://en.wikipedia.org/wiki/Finite_difference_coefficient

    """
    if extent is None:
        extent = diff_order + 2
    n_points = (diff_order + extent) // 2

    if not correct_edges:
        central_coefs = central_stencil_coefficients(n_points, diff_order)
        result = np.convolve(values, central_coefs[::-1], mode="same")
    else:
        left_coefs, central_coefs, right_coefs = stencil_coefficients_set(n_points, diff_order)
        result = np.concatenate(
            (
                np.convolve(values[: 2 * n_points], left_coefs[::-1], mode="valid"),
                np.convolve(values, central_coefs[::-1], mode="valid"),
                np.convolve(values[-2 * n_points :], right_coefs[::-1], mode="valid"),
            )
        )
    return result / h**diff_order
