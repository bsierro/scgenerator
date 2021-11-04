# series of functions to convert different values to angular frequencies
# For example, nm(X) means "I give the number X in nm, figure out the ang. freq."
# to be used especially when giving plotting ranges : (400, 1400, nm), (-4, 8, ps), ...

from __future__ import annotations

from operator import itemgetter
from typing import Callable, TypeVar, Union

import numpy as np
from numpy import pi

c = 299792458.0
hbar = 1.05457148e-34
NA = 6.02214076e23
R = 8.31446261815324
kB = 1.380649e-23
epsilon0 = 8.85418781e-12
me = 9.1093837015e-31
e = 1.602176634e-19


prefix = dict(P=1e12, G=1e9, M=1e6, k=1e3, d=1e-1, c=1e-2, m=1e-3, u=1e-6, n=1e-9, p=1e-12, f=1e-15)

"""
Below are common units. You can define your own unit function
provided you decorate it with @unit and provide at least a type and a label
types are "WL", "FREQ", "AFREQ", "TIME", "PRESSURE", "TEMPERATURE", "OTHER"
"""
_T = TypeVar("_T")
_UT = Callable[[_T], _T]
PRIMARIES = dict(WL="Rad/s", FREQ="Rad/s", AFREQ="Rad/s", TIME="s", PRESSURE="Pa", TEMPERATURE="K")


class UnitMap(dict):
    def __setitem__(self, new_name, new_func):
        super().__setitem__(new_name, new_func)
        already_here = [name for name in self if isinstance(name, str)]
        for old_name in already_here:
            super().__setitem__((old_name, new_name), self._chain(old_name, new_name))
            super().__setitem__((new_name, old_name), self._chain(new_name, old_name))

    def _chain(self, name_1: str, name_2: str) -> _UT:
        c1 = self[name_1]
        c2 = self[name_2]

        def chained_function(x: _T) -> _T:
            return c2.inv(c1(x))

        chained_function.__name__ = f"{name_1}_to_{name_2}"
        chained_function.__doc__ = f"converts x from {name_1} to {name_2}"

        return chained_function


units_map: dict[Union[str, tuple[str, str]], _UT] = UnitMap()


class To:
    def __init__(self, name: str):
        self.name = name

    def __getstate__(self):
        """make pickle happy"""
        return self.name

    def __setstate__(self, name):
        """make pickle happy"""
        self.name = name

    def __getattr__(self, key: str):
        try:
            return units_map[self.name, key]
        except KeyError:
            raise KeyError(f"no registered unit named {key!r}") from None


def W_to_Vm(n0: float, A_eff: float) -> float:
    """returns the factor to convert from [|E|²] = W to [|E|] = V/m

    Parameters
    ----------
    n : float
        refractive index

    Returns
    -------
    float
        p such that E_sqrt(W) * p = W_Vm
    """
    return 1.0 / np.sqrt(A_eff * 0.5 * epsilon0 * c * n0)


class unit:
    inv = None

    def __init__(self, tpe: str, label: str):
        self.type = tpe
        self.label = label

    def __call__(self, func):
        self.name = func.__name__
        self.func = func
        self.inv = func

        units_map[self.name] = self.func

        func.to = To(self.name)
        func.inv = func
        func.inverse = self.inverse
        func.type = self.type
        func.label = self.label
        func.name = self.name
        func.__doc__ = f"Transform x from {self.name!r} to {PRIMARIES.get(self.type)!r}"
        return func

    def inverse(self, func: _UT):
        if func.__name__ == self.name:
            raise ValueError(
                f"inverse function of {self.name} must be named something else than {self.name}"
            )
        self.inv = func
        self.func.inv = func
        return func


@unit("WL", r"Wavelength $\lambda$ (m)")
def m(l: _T) -> _T:
    return 2 * pi * c / l


@unit("WL", r"Wavelength $\lambda$ (nm)")
def nm(l: _T) -> _T:
    return 2 * pi * c / (l * 1e-9)


@unit("WL", r"Wavelength $\lambda$ (μm)")
def um(l: _T) -> _T:
    return 2 * pi * c / (l * 1e-6)


@unit("FREQ", r"Frequency $f$ (Hz)")
def Hz(f: _T) -> _T:
    return 2 * pi * f


@Hz.inverse
def Hz_inv(w):
    return w / (2 * pi)


@unit("FREQ", r"Frequency $f$ (THz)")
def THz(f: _T) -> _T:
    return 1e12 * 2 * pi * f


@THz.inverse
def THz_inv(w):
    return w / (1e12 * 2 * pi)


@unit("FREQ", r"Frequency $f$ (PHz)")
def PHz(f: _T) -> _T:
    return 1e15 * 2 * pi * f


@PHz.inverse
def PHz_inv(w):
    return w / (1e15 * 2 * pi)


@unit("AFREQ", r"Angular frequency $\omega$ ($\frac{\mathrm{rad}}{\mathrm{s}}$)")
def rad_s(w: _T) -> _T:
    return w


@unit("AFREQ", r"Angular frequency $\omega$ ($\frac{\mathrm{Prad}}{\mathrm{s}}$)")
def Prad_s(w: _T) -> _T:
    return w * 1e15


@Prad_s.inverse
def Prad_s_inv(w):
    return w * 1e-15


@unit("TIME", r"relative time ${\tau}/{\tau_\mathrm{0, FWHM}}$")
def rel_time(t: _T) -> _T:
    return t


@unit("FREQ", r"relative angular freq. $(\omega - \omega_0)/\Delta\omega_0$")
def rel_freq(f: _T) -> _T:
    return f


@unit("TIME", r"Time $t$ (s)")
def s(t: _T) -> _T:
    return t


@unit("TIME", r"Time $t$ (us)")
def us(t: _T) -> _T:
    return t * 1e-6


@us.inverse
def us_inv(t):
    return t * 1e6


@unit("TIME", r"Time $t$ (ns)")
def ns(t: _T) -> _T:
    return t * 1e-9


@ns.inverse
def ns_inv(t):
    return t * 1e9


@unit("TIME", r"Time $t$ (ps)")
def ps(t: _T) -> _T:
    return t * 1e-12


@ps.inverse
def ps_inv(t):
    return t * 1e12


@unit("TIME", r"Time $t$ (fs)")
def fs(t: _T) -> _T:
    return t * 1e-15


@fs.inverse
def fs_inv(t):
    return t * 1e15


@unit("WL", "inverse")
def inv(x: _T) -> _T:
    return 1 / x


@unit("PRESSURE", "Pressure (bar)")
def bar(p: _T) -> _T:
    return 1e5 * p


@bar.inverse
def bar_inv(p):
    return p * 1e-5


@unit("OTHER", r"$\beta_2$ (fs$^2$/cm)")
def beta2_fs_cm(b2: _T) -> _T:
    return 1e-28 * b2


@beta2_fs_cm.inverse
def beta2_fs_cm_inv(b2):
    return 1e28 * b2


@unit("OTHER", r"$\beta_2$ (ps$^2$/km)")
def beta2_ps_km(b2: _T) -> _T:
    return 1e-27 * b2


@beta2_ps_km.inverse
def beta2_ps_km_inv(b2):
    return 1e27 * b2


@unit("OTHER", r"$D$ (ps/(nm km))")
def D_ps_nm_km(D: _T) -> _T:
    return 1e-6 * D


@D_ps_nm_km.inverse
def D_ps_nm_km_inv(D):
    return 1e6 * D


@unit("OTHER", r"a.u.")
def unity(x: _T) -> _T:
    return x


@unit("TEMPERATURE", r"Temperature (K)")
def K(t: _T) -> _T:
    return t


@unit("TEMPERATURE", r"Temperature (°C)")
def C(t_C: _T) -> _T:
    return t_C + 272.15


@C.inverse
def C_inv(t_K):
    return t_K - 272.15


def get_unit(unit: Union[str, Callable]) -> Callable[[float], float]:
    if isinstance(unit, str):
        return units_map[unit]
    return unit


def is_unit(name, value):
    if not hasattr(get_unit(value), "inv"):
        raise TypeError("invalid unit specified")


def beta2_coef(beta2_coefficients):
    fac = 1e27
    out = np.zeros_like(beta2_coefficients)
    for i, b in enumerate(beta2_coefficients):
        out[i] = fac * b
        fac *= 1e12
    return out


def standardize_dictionary(dico):
    """convert lists of number and units into a float with SI units inside a dictionary
    Parameters
    ----------
        dico : a dictionary
    Returns
    ----------
        same dictionary with units converted
    Example
    ----------
    standardize_dictionary({"peak_power": [23, "kW"], "points": [1, 2, 3]})
    {"peak_power": 23000, "points": [1, 2, 3]})
    """
    for key, item in dico.items():
        if (
            isinstance(item, list)
            and len(item) == 2
            and isinstance(item[0], (int, float))
            and isinstance(item[1], str)
        ):
            num, unit = item
            fac = 1
            if len(unit) == 2:
                fac = prefix[unit[0]]
            elif unit == "bar":
                fac = 1e5
            dico[key] = num * fac
    return dico


def to_WL(spectrum: np.ndarray, lambda_: np.ndarray) -> np.ndarray:
    """rescales the spectrum because of uneven binning when going from freq to wl

    Parameters
    ----------
    spectrum : np.ndarray, shape (n, )
        intensity spectrum
    lambda_ : np.ndarray, shape (n, )
        wavelength in m

    Returns
    -------
    np.ndarray, shape (n, )
        intensity spectrum correctly scaled
    """
    m = 2 * pi * c / (lambda_ ** 2) * spectrum
    return m


def to_log(arr, ref=None):
    """takes the log of each 1D array relative to the max of said array. Useful
    to plot spectrum evolution, but use to_log2D for spectrograms
    Parameters
    ----------
        arr : 1D array of real numbers. >1D array : operation is applied on axis=0
        ref : reference value corresponding to 0dB (default : max(arr))
    Returns
    ----------
        arr array in dB
    """
    if arr.ndim > 1:
        return np.apply_along_axis(to_log, -1, arr, ref)
    else:
        if ref is None:
            ref = np.max(arr)
        m = arr / ref
        m = 10 * np.log10(m, out=np.zeros_like(m) - 100, where=m > 0)
        return m


def to_log2D(arr, ref=None):
    """computes the log of a 2D array
    Parameters
    ----------
        arr : 2D array of real numbers
        ref : reference value corresponding to 0dB
    Returns
    ----------
        arr array in dB
    """
    if ref is None:
        ref = np.max(arr)
    m = arr / ref
    m = 10 * np.log10(m, out=np.zeros_like(m) - 100, where=m > 0)
    return m


class PlotRange(tuple):
    left: float = property(itemgetter(0))
    right: float = property(itemgetter(1))
    unit: Callable[[float], float] = property(itemgetter(2))
    conserved_quantity: bool = property(itemgetter(3))
    __slots__ = []

    def __new__(cls, left, right, unit, conserved_quantity=True):
        return tuple.__new__(cls, (left, right, get_unit(unit), conserved_quantity))

    def __iter__(self):
        yield self.left
        yield self.right
        yield self.unit.name

    def __repr__(self):
        return "PlotRange" + super().__repr__()

    def sort_axis(self, axis: np.ndarray) -> tuple[np.ndarray, np.ndarray, tuple[float, float]]:
        return sort_axis(axis, self)


def sort_axis(
    axis: np.ndarray, plt_range: PlotRange
) -> tuple[np.ndarray, np.ndarray, tuple[float, float]]:
    """
    given an axis, returns this axis cropped according to the given range, converted and sorted

    Parameters
    ----------
    axis : 1D array containing the original axis (usual the w or t array)
    plt_range : tupple (min, max, conversion_function) used to crop the axis

    Returns
    -------
    cropped : the axis cropped, converted and sorted
    indices : indices to use to slice and sort other array in the same fashion
    extent : tupple with min and max of cropped

    Example
    -------
    w = np.append(np.linspace(0, -10, 20), np.linspace(0, 10, 20))
    t = np.linspace(-10, 10, 400)
    W, T = np.meshgrid(w, t)
    y = np.exp(-W**2 - T**2)

    # Define ranges
    rw = (-4, 4, s)
    rt = (-2, 6, s)

    w, cw = sort_axis(w, rw)
    t, ct = sort_axis(t, rt)

    # slice y according to the given ranges
    y = y[ct][:, cw]
    """
    if not isinstance(plt_range, PlotRange):
        plt_range = PlotRange(*plt_range)
    r = np.array((plt_range.left, plt_range.right), dtype="float")

    indices = np.arange(len(axis))[
        (axis <= np.max(plt_range.unit(r))) & (axis >= np.min(plt_range.unit(r)))
    ]
    cropped = axis[indices]
    order = np.argsort(plt_range.unit.inv(cropped))
    indices = indices[order]
    cropped = cropped[order]
    out_ax = plt_range.unit.inv(cropped)

    return out_ax, indices, (out_ax[0], out_ax[-1])
