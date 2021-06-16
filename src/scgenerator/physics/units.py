# series of functions to convert different values to angular frequencies
# For example, nm(X) means "I give the number X in nm, figure out the ang. freq."
# to be used especially when giving plotting ranges : (400, 1400, nm), (-4, 8, ps), ...

from typing import Callable, Union
from dataclasses import dataclass
from ..utils.parameter import Parameter, type_checker
import numpy as np
from numpy import pi

c = 299792458.0
hbar = 1.05457148e-34
NA = 6.02214076e23
R = 8.31446261815324
kB = 1.380649e-23
epsilon0 = 8.85418781e-12

prefix = dict(P=1e12, G=1e9, M=1e6, k=1e3, d=1e-1, c=1e-2, m=1e-3, u=1e-6, n=1e-9, p=1e-12, f=1e-15)

"""
Below are common units. You can define your own unit function
this function must have a few porperties:
inv : function
    inverse of the function. example :
        um(1) -> 883651567308853.2
        um.inv(883651567308853.2) -> 1.0
label : str
    label to be displayed on plot
type : ("WL", "FREQ", "AFREQ", "TIME", "OTHER")
"""


def m(l):
    return 2 * pi * c / l


m.inv = m
m.label = r"Wavelength $\lambda$ (m)"
m.type = "WL"


def nm(l):
    return 2 * pi * c / (l * 1e-9)


nm.inv = nm
nm.label = r"Wavelength $\lambda$ (nm)"
nm.type = "WL"


def um(l):
    return 2 * pi * c / (l * 1e-6)


um.inv = um
um.label = r"Wavelength $\lambda$ ($\mathrm{\mu}$m)"
um.type = "WL"


def THz(f):
    return 1e12 * 2 * pi * f


THz.inv = lambda w: w / (1e12 * 2 * pi)
THz.label = r"Frequency $f$ (THz)"
THz.type = "FREQ"


def PHz(f):
    return 1e15 * 2 * pi * f


PHz.inv = lambda w: w / (1e15 * 2 * pi)
PHz.label = r"Frequency $f$ (PHz)"
PHz.type = "FREQ"


def rad_s(w):
    return w


rad_s.inv = rad_s
rad_s.label = r"Angular frequency $\omega$ ($\frac{\mathrm{rad}}{\mathrm{s}}$)"
rad_s.type = "AFREQ"


def Prad_s(w):
    return w * 1e15


Prad_s.inv = lambda w: 1e-15 * w
Prad_s.label = r"Angular frequency $\omega$ ($\frac{\mathrm{Prad}}{\mathrm{s}}$)"
Prad_s.type = "AFREQ"


def rel_time(t):
    return t


rel_time.inv = rel_time
rel_time.label = r"relative time ${\tau}/{\tau_\mathrm{0, FWHM}}$"
rel_time.type = "TIME"


def rel_freq(f):
    return f


rel_freq.inv = rel_freq
rel_freq.label = r"relative angular freq. $(\omega - \omega_0)/\Delta\omega_0$"
rel_freq.type = "FREQ"


def s(t):
    return t


s.inv = s
s.label = r"Time $t$ (s)"
s.type = "TIME"


def us(t):
    return t * 1e-6


us.inv = lambda t: t * 1e6
us.label = r"Time $t$ (us)"
us.type = "TIME"


def ns(t):
    return t * 1e-9


ns.inv = lambda t: t * 1e9
ns.label = r"Time $t$ (ns)"
ns.type = "TIME"


def ps(t):
    return t * 1e-12


ps.inv = lambda t: t * 1e12
ps.label = r"Time $t$ (ps)"
ps.type = "TIME"


def fs(t):
    return t * 1e-15


fs.inv = lambda t: t * 1e15
fs.label = r"Time $t$ (fs)"
fs.type = "TIME"


def inv(x):
    return 1 / x


inv.inv = inv
inv.label = "inverse"
inv.type = "WL"


def bar(p):
    return 1e5 * p


bar.inv = lambda p: 1e-5 * p
bar.label = "Pressure (bar)"
bar.type = "PRESSURE"


def beta2_fs_cm(b2):
    return 1e-28 * b2


beta2_fs_cm.inv = lambda b2: 1e28 * b2
beta2_fs_cm.label = r"$\beta_2$ (fs$^2$/cm)"
beta2_fs_cm.type = "OTHER"


def beta2_ps_km(b2):
    return 1e-27 * b2


beta2_ps_km.inv = lambda b2: 1e27 * b2
beta2_ps_km.label = r"$\beta_2$ (ps$^2$/km)"
beta2_ps_km.type = "OTHER"


def D_ps_nm_km(D):
    return 1e-6 * D


D_ps_nm_km.inv = lambda D: 1e6 * D
D_ps_nm_km.label = r"$D$ (ps/(nm km))"
D_ps_nm_km.type = "OTHER"


units_map = dict(
    nm=nm,
    um=um,
    m=m,
    THz=THz,
    PHz=PHz,
    rad_s=rad_s,
    Prad_s=Prad_s,
    rel_freq=rel_freq,
    rel_time=rel_time,
    s=s,
    us=us,
    ns=ns,
    ps=ps,
    fs=fs,
)


def get_unit(unit: Union[str, Callable]) -> Callable[[float], float]:
    if isinstance(unit, str):
        return units_map[unit]
    return unit


def is_unit(name, value):
    if not hasattr(get_unit(value), "inv"):
        raise TypeError("invalid unit specified")


@dataclass
class PlotRange:
    left: float = Parameter(type_checker(int, float))
    right: float = Parameter(type_checker(int, float))
    unit: Callable[[float], float] = Parameter(is_unit, converter=get_unit)


def beta2_coef(beta):
    fac = 1e27
    out = np.zeros_like(beta)
    for i, b in enumerate(beta):
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


def sort_axis(axis, plt_range: PlotRange):
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


def to_WL(spectrum, frep, lambda_):
    """
    When a spectrogram is displayed as function of wl instead of frequency, we
    need to adjust the amplitude of each bin for the integral over the whole frequency
    range to match.
    """
    m = 2 * pi * c / (lambda_ ** 2) * frep * spectrum
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
