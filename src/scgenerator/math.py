from typing import Union

import numpy as np
from scipy.integrate import cumulative_trapezoid
from scipy.special import jn_zeros

from .cache import np_cache

pi = np.pi
c = 299792458.0


def inverse_integral_exponential(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """evaluates exp(-func(y)) from x=-inf to x"""
    return np.exp(-cumulative_trapezoid(y, x, initial=0))


def span(*vec):
    """returns the min and max of whatever array-like is given. can accept many args"""
    out = (np.inf, -np.inf)
    for x in vec:
        x = np.atleast_1d(x)
        if len(x.shape) > 1:
            x = x.ravel()
        minx = np.min(x)
        maxx = np.max(x)
        out = (np.min([minx, out[0]]), np.max([maxx, out[1]]))

    if out[0] == np.inf or out[1] == -np.inf:
        out = (0, 1)
        print(f"failed to evaluate the span of {vec}")
    return out


def argclosest(array: np.ndarray, target: Union[float, int]):
    """returns the index/indices corresponding to the closest matches of target in array"""
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


def length(x):
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
    return z.real ** 2 + z.imag ** 2


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


@np_cache
def ndft_matrix(t: np.ndarray, f: np.ndarray) -> np.ndarray:
    """creates the nfft matrix

    Parameters
    ----------
    t : np.ndarray, shape = (n,)
        time array
    f : np.ndarray, shape = (m,)
        frequency array

    Returns
    -------
    np.ndarray, shape = (m, n)
        multiply x(t) by this matrix to get ~X(f)
    """
    P, F = np.meshgrid(t, f)
    return np.exp(-2j * np.pi * P * F)


@np_cache
def indft_matrix(t: np.ndarray, f: np.ndarray) -> np.ndarray:
    """creates the nfft matrix

    Parameters
    ----------
    t : np.ndarray, shape = (n,)
        time array
    f : np.ndarray, shape = (m,)
        frequency array

    Returns
    -------
    np.ndarray, shape = (m, n)
        multiply ~X(f) by this matrix to get x(t)
    """
    return np.linalg.pinv(ndft_matrix(t, f))


def ndft(t: np.ndarray, s: np.ndarray, f: np.ndarray) -> np.ndarray:
    """computes the Fourier transform of an uneven signal

    Parameters
    ----------
    t : np.ndarray, shape = (n,)
        time array
    s : np.ndarray, shape = (n, )
        amplitute at each point of t
    f : np.ndarray, shape = (m, )
        desired frequencies

    Returns
    -------
    np.ndarray, shape = (m, )
        amplitude at each frequency
    """
    return ndft_matrix(t, f) @ s


def indft(f: np.ndarray, a: np.ndarray, t: np.ndarray) -> np.ndarray:
    """computes the inverse Fourier transform of an uneven spectrum

    Parameters
    ----------
    f : np.ndarray, shape = (n,)
        frequency array
    a : np.ndarray, shape = (n, )
        amplitude at each point of f
    t : np.ndarray, shape = (m, )
        time array

    Returns
    -------
    np.ndarray, shape = (m, )
        amplitude at each point of t
    """
    return indft_matrix(t, f) @ a


def all_zeros(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """find all the x values such that y(x)=0 with linear interpolation"""
    pos = np.argwhere(y[1:] * y[:-1] < 0)[:, 0]
    m = (y[pos + 1] - y[pos]) / (x[pos + 1] - x[pos])
    return -y[pos] / m + x[pos]


def wspace(t, t_num=0):
    """frequency array such that x(t) <-> np.fft(x)(w)
    Parameters
    ----------
        t : float or array
            float : total width of the time window
            array : time array
        t_num : int-
            if t is a float, specifies the number of points
    Returns
    ----------
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
    """returns a time array centered on 0
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


def build_sim_grid(
    length: float,
    z_num: int,
    wavelength: float,
    time_window: float = None,
    t_num: int = None,
    dt: float = None,
) -> tuple[np.ndarray, np.ndarray, float, int, float, np.ndarray, float, np.ndarray, np.ndarray]:
    """computes a bunch of values that relate to the simulation grid

    Parameters
    ----------
    length : float
        length of the fiber in m
    z_num : int
        number of spatial points
    wavelength : float
        pump wavelength in m
    deg : int
        dispersion interpolation degree
    time_window : float, optional
        total width of the temporal grid in s, by default None
    t_num : int, optional
        number of temporal grid points, by default None
    dt : float, optional
        spacing of the temporal grid in s, by default None

    Returns
    -------
    z_targets : np.ndarray, shape (z_num, )
        spatial points in m
    t : np.ndarray, shape (t_num, )
        temporal points in s
    time_window : float
        total width of the temporal grid in s, by default None
    t_num : int
        number of temporal grid points, by default None
    dt : float
        spacing of the temporal grid in s, by default None
    w_c : np.ndarray, shape (t_num, )
        centered angular frequencies in rad/s where 0 is the pump frequency
    w0 : float
        pump angular frequency
    w : np.ndarray, shape (t_num, )
        actual angualr frequency grid in rad/s
    w_order : np.ndarray, shape (t_num, )
        result of w.argsort()
    l : np.ndarray, shape (t_num)
        wavelengths in m
    """
    t = tspace(time_window, t_num, dt)

    time_window = t.max() - t.min()
    dt = t[1] - t[0]
    t_num = len(t)
    z_targets = np.linspace(0, length, z_num)
    w_c = wspace(t)
    w0 = 2 * pi * c / wavelength
    w = w_c + w0
    w_order = np.argsort(w)
    l = 2 * pi * c / w
    return z_targets, t, time_window, t_num, dt, w_c, w0, w, w_order, l


def polynom_extrapolation(x: np.ndarray, y: np.ndarray, degree: float) -> np.ndarray:
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
    out = y.copy()
    order = np.argsort(x)
    left_ind, *_, right_ind = np.nonzero(out[order])[0]
    return _polynom_extrapolation_in_place(out[order], left_ind, right_ind, degree)[order.argsort]


def _polynom_extrapolation_in_place(y: np.ndarray, left_ind: int, right_ind: int, degree: float):
    r_left = (1 + np.arange(left_ind))[::-1] ** degree
    r_right = np.arange(len(y) - right_ind) ** degree
    y[:left_ind] = r_left * (y[left_ind] - y[left_ind + 1]) + y[left_ind]
    y[right_ind:] = r_right * (y[right_ind] - y[right_ind - 1]) + y[right_ind]
    return y
