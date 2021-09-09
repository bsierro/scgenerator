import os
from collections.abc import Sequence
from pathlib import Path
from typing import Callable, Dict, Iterable, Union

import matplotlib.pyplot as plt
import numpy as np

from . import math
from .const import SPECN_FN
from .logger import get_logger
from .physics import pulse, units
from .plotting import (
    mean_values_plot,
    propagation_plot,
    single_position_plot,
    transform_2D_propagation,
)
from .utils.parameter import Parameters, PlotRange


class Spectrum(np.ndarray):
    params: Parameters

    def __new__(cls, input_array, params: Parameters):
        # Input array is an already formed ndarray instance
        # We first cast to be our class type
        obj = np.asarray(input_array).view(cls)
        # add the new attribute to the created instance
        obj.params = params

        # Finally, we must return the newly created object:
        return obj

    def __array_finalize__(self, obj):
        # see InfoArray.__array_finalize__ for comments
        if obj is None:
            return
        self.params = getattr(obj, "params", None)

    def __getitem__(self, key) -> "Spectrum":
        return super().__getitem__(key)

    def energy(self) -> Union[np.ndarray, float]:
        if self.ndim == 1:
            m = np.argwhere(self.params.l > 0)[:, 0]
            m = np.array(sorted(m, key=lambda el: self.params.l[el]))
            return np.trapz(self.wl_int[m], self.params.l[m])
        else:
            return np.array([s.energy() for s in self])

    def crop_wl(self, left: float, right: float) -> np.ndarray:
        cond = (self.params.l >= left) & (self.params.l <= right)
        return cond

    @property
    def wl_int(self):
        return units.to_WL(math.abs2(self), self.params.l)

    @property
    def freq_int(self):
        return math.abs2(self)

    @property
    def afreq_int(self):
        return math.abs2(self)

    @property
    def time_int(self):
        return math.abs2(np.fft.ifft(self))

    def amplitude(self, unit):
        if unit.type in ["WL", "FREQ", "AFREQ"]:
            x_axis = unit.inv(self.params.w)
        else:
            x_axis = unit.inv(self.params.t)

        order = np.argsort(x_axis)
        func = dict(
            WL=self.wl_amp,
            FREQ=self.freq_amp,
            AFREQ=self.afreq_amp,
            TIME=self.time_amp,
        )[unit.type]

        for spec in self:
            yield x_axis[order], func(spec)[:, order]

    @property
    def wl_amp(self):
        return (
            np.sqrt(
                units.to_WL(
                    math.abs2(self),
                    self.params.l,
                )
            )
            * self
            / np.abs(self)
        )

    @property
    def freq_amp(self):
        return self

    @property
    def afreq_amp(self):
        return self

    @property
    def time_amp(self):
        return np.fft.ifft(self)

    @property
    def wl_max(self):
        if self.ndim == 1:
            return self.params.l[np.argmax(self.wl_int, axis=-1)]
        return np.array([s.wl_max for s in self])

    def mask_wl(self, pos: float, width: float) -> "Spectrum":
        return self * np.exp(
            -(((self.params.l - pos) / (pulse.fwhm_to_T0_fac["gaussian"] * width)) ** 2)
        )

    def measure(self) -> tuple[float, float, float]:
        return pulse.measure_field(self.params.t, self.time_amp)


class Pulse(Sequence):
    def __init__(self, path: os.PathLike, default_ind: Union[int, Iterable[int]] = None):
        """load a data folder as a pulse

        Parameters
        ----------
        path : os.PathLike
            path to the data (folder containing .npy files)
        default_ind : int | Iterable[int], optional
            default indices to be loaded, by default None

        Raises
        ------
        FileNotFoundError
            path does not contain proper data
        """
        self.logger = get_logger(__name__)
        self.path = Path(path)
        self.default_ind = default_ind

        if not self.path.is_dir():
            raise FileNotFoundError(f"Folder {self.path} does not exist")

        self.params = Parameters.load(self.path / "params.toml")
        self.params.compute(["name", "t", "l", "w_c", "w0", "z_targets"])
        if self.params.fiber_map is None:
            self.params.fiber_map = {0.0: self.params.name}

        try:
            self.z = np.load(os.path.join(path, "z.npy"))
        except FileNotFoundError:
            if self.params is not None:
                self.z = self.params.z_targets
            else:
                raise
        self.cache: Dict[int, Spectrum] = {}
        self.nmax = len(list(self.path.glob("spectra_*.npy")))
        if self.nmax <= 0:
            raise FileNotFoundError(f"No appropriate file in specified folder {self.path}")

        self.t = self.params.t
        w = math.wspace(self.t) + units.m(self.params.wavelength)
        self.w_order = np.argsort(w)
        self.w = w
        self.wl = units.m.inv(self.w)
        self.params.w = self.w
        self.params.z_targets = self.z

    def __iter__(self):
        """
        similar to all_spectra but works as an iterator
        """

        self.logger.debug(f"iterating through {self.path}")
        for i in range(self.nmax):
            yield self._load1(i)

    def __len__(self):
        return self.nmax

    def __getitem__(self, key) -> Spectrum:
        return self.all_spectra(key)

    def intensity(self, unit):
        if unit.type in ["WL", "FREQ", "AFREQ"]:
            x_axis = unit.inv(self.w)
        else:
            x_axis = unit.inv(self.t)

        order = np.argsort(x_axis)
        func = dict(
            WL=self._to_wl_int,
            FREQ=self._to_freq_int,
            AFREQ=self._to_afreq_int,
            TIME=self._to_time_int,
        )[unit.type]

        for spec in self:
            yield x_axis[order], func(spec)[:, order]

    def _to_wl_int(self, spectrum):
        return units.to_WL(math.abs2(spectrum), spectrum.wl)

    def _to_freq_int(self, spectrum):
        return math.abs2(spectrum)

    def _to_afreq_int(self, spectrum):
        return math.abs2(spectrum)

    def _to_time_int(self, spectrum):
        return math.abs2(np.fft.ifft(spectrum))

    def amplitude(self, unit):
        if unit.type in ["WL", "FREQ", "AFREQ"]:
            x_axis = unit.inv(self.w)
        else:
            x_axis = unit.inv(self.t)

        order = np.argsort(x_axis)
        func = dict(
            WL=self._to_wl_amp,
            FREQ=self._to_freq_amp,
            AFREQ=self._to_afreq_amp,
            TIME=self._to_time_amp,
        )[unit.type]

        for spec in self:
            yield x_axis[order], func(spec)[:, order]

    def _to_wl_amp(self, spectrum):
        return (
            np.sqrt(
                units.to_WL(
                    math.abs2(spectrum),
                    spectrum.wl,
                )
            )
            * spectrum
            / np.abs(spectrum)
        )

    def _to_freq_amp(self, spectrum):
        return spectrum

    def _to_afreq_amp(self, spectrum):
        return spectrum

    def _to_time_amp(self, spectrum):
        return np.fft.ifft(spectrum)

    def all_spectra(self, ind=None) -> Spectrum:
        """
        loads the data already simulated.
        defauft shape is (z_targets, n, nt)

        Parameters
        ----------
        ind : int or list of int
            if only certain spectra are desired
        Returns
        ----------
        spectra : array of shape (nz, m, nt)
            array of complex spectra (pulse at nz positions consisting
            of nm simulation on a nt size grid)
        """

        self.logger.debug(f"opening {self.path}")

        # Check if file exists and assert how many z positions there are

        if ind is None:
            if self.default_ind is None:
                ind = range(self.nmax)
            else:
                ind = self.default_ind
        if isinstance(ind, (int, np.integer)):
            ind = [ind]
        elif isinstance(ind, (float, np.floating)):
            ind = [self.z_ind(ind)]
        elif isinstance(ind[0], (float, np.floating)):
            ind = [self.z_ind(ii) for ii in ind]

        # Load the spectra
        spectra = []
        for i in ind:
            spectra.append(self._load1(i))
        spectra = Spectrum(spectra, self.params)

        self.logger.debug(f"all spectra from {self.path} successfully loaded")
        if len(ind) == 1:
            return spectra[0]
        else:
            return spectra

    def all_fields(self, ind=None):
        return np.fft.ifft(self.all_spectra(ind=ind), axis=-1)

    def _load1(self, i: int):
        if i < 0:
            i = self.nmax + i
        if i in self.cache:
            return self.cache[i]
        spec = np.load(self.path / SPECN_FN.format(i))
        spec = np.atleast_2d(spec)
        spec = Spectrum(spec, self.params)
        self.cache[i] = spec
        return spec

    def plot_2D(
        self,
        left: float,
        right: float,
        unit: Union[Callable[[float], float], str],
        ax: plt.Axes,
        z_pos: Union[int, Iterable[int]] = None,
        sim_ind: int = 0,
        **kwargs,
    ):
        plt_range, vals = self.retrieve_plot_values(left, right, unit, z_pos, sim_ind)
        return propagation_plot(vals, plt_range, self.params, ax, **kwargs)

    def plot_1D(
        self,
        left: float,
        right: float,
        unit: Union[Callable[[float], float], str],
        ax: plt.Axes,
        z_pos: int,
        sim_ind: int = 0,
        **kwargs,
    ):
        plt_range, vals = self.retrieve_plot_values(left, right, unit, z_pos, sim_ind)
        return single_position_plot(vals, plt_range, self.params, ax, **kwargs)

    def plot_mean(
        self,
        left: float,
        right: float,
        unit: Union[Callable[[float], float], str],
        ax: plt.Axes,
        z_pos: int,
        **kwargs,
    ):
        plt_range, vals = self.retrieve_plot_values(left, right, unit, z_pos, slice(None))
        return mean_values_plot(vals, plt_range, self.params, ax, **kwargs)

    def retrieve_plot_values(self, left, right, unit, z_pos, sim_ind):
        plt_range = PlotRange(left, right, unit)
        if plt_range.unit.type == "TIME":
            vals = self.all_fields(ind=z_pos)
        else:
            vals = self.all_spectra(ind=z_pos)
        if vals.ndim == 3:
            vals = vals[:, sim_ind]
        else:
            vals = vals[sim_ind]
        return plt_range, vals

    def rin_propagation(
        self, left: float, right: float, unit: str
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """returns the RIN as function of unit and z

        Parameters
        ----------
        left : float
            left limit in unit
        right : float
            right limit in unit
        unit : str
            unit descriptor

        Returns
        -------
        x : np.ndarray, shape (nt,)
            x axis
        y : np.ndarray, shape (z_num, )
            y axis
        rin_prop : np.ndarray, shape (z_num, nt)
            RIN
        """
        spectra = []
        for spec in np.moveaxis(self.all_spectra(), 1, 0):
            x, z, tmp = transform_2D_propagation(spec, (left, right, unit), self.params, False)
            spectra.append(tmp)
        return x, z, pulse.rin_curve(np.moveaxis(spectra, 0, 1))

    def z_ind(self, z: float) -> int:
        """return the closest z index to the given target

        Parameters
        ----------
        z : float
            target

        Returns
        -------
        int
            index
        """
        return math.argclosest(self.z, z)
