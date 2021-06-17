from typing import Union

import numpy as np
from scipy.interpolate import griddata, interp1d
from scipy.special import jn_zeros
from .utils.cache import np_cache


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
    result = np.ones(len(x), dtype=np.float64)
    for k in range(n):
        result = result * x / (n - k)
    return result


def abs2(z: np.ndarray) -> np.ndarray:
    return z.real ** 2 + z.imag ** 2


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


def make_uniform_2D(values, x_axis, y_axis, n=1024, method="linear"):
    """Interpolates a 2D array with the help of griddata
    Parameters
    ----------
        values : 2D array of real values
        x_axis : x-coordinates of values
        y_axis : y-coordinates of values
        method : method of interpolation to be passed to griddata
    Returns
    ----------
        array of shape n
    """
    xx, yy = np.meshgrid(x_axis, y_axis)
    xx = xx.flatten()
    yy = yy.flatten()

    if not isinstance(n, tuple):
        n = (n, n)

    # old_points = np.array([gridx.ravel(), gridy.ravel()])

    newx, newy = np.meshgrid(np.linspace(*span(x_axis), n[0]), np.linspace(*span(y_axis), n[1]))

    print("interpolating")
    out = griddata((xx, yy), values.flatten(), (newx, newy), method=method, fill_value=0)
    print("interpolating done!")
    return out.reshape(n[1], n[0])


def make_uniform_1D(values, x_axis, n=1024, method="linear"):
    """Interpolates a 2D array with the help of interp1d
    Parameters
    ----------
        values : 1D array of real values
        x_axis : x-coordinates of values
        method : method of interpolation to be passed to interp1d
    Returns
    ----------
        array of length n
    """
    xx = np.linspace(*span(x_axis), len(x_axis))
    return interp1d(x_axis, values, kind=method)(xx)
