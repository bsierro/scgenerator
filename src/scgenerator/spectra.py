from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Callable, Iterator, Optional, Union

import matplotlib.pyplot as plt
import numpy as np

from . import math
from .const import PARAM_FN, SPEC1_FN, SPEC1_FN_N
from .logger import get_logger
from .parameter import Parameters
from .physics import pulse, units
from .physics.units import PlotRange
from .plotting import (
    mean_values_plot,
    propagation_plot,
    single_position_plot,
    transform_2D_propagation,
)
from .utils import load_spectrum, simulations_list


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

    def mask_wl(self, pos: float, width: float) -> Spectrum:
        return self * np.exp(
            -(((self.params.l - pos) / (pulse.fwhm_to_T0_fac["gaussian"] * width)) ** 2)
        )

    def measure(self) -> tuple[float, float, float]:
        return pulse.measure_field(self.params.t, self.time_amp)


class SimulationSeries:
    path: Path
    params: Parameters
    total_length: float
    total_num_steps: int
    previous: SimulationSeries = None
    fiber_lengths: list[tuple[str, float]]
    fiber_positions: list[tuple[str, float]]
    z_inds: np.ndarray

    def __init__(self, path: os.PathLike):
        self.logger = get_logger()
        for self.path in simulations_list(path):
            break
        else:
            raise FileNotFoundError(f"No simulation in {path}")
        self.params = Parameters.load(self.path / PARAM_FN)
        self.t = self.params.t
        self.w = self.params.w
        if self.params.prev_data_dir is not None:
            self.previous = SimulationSeries(self.params.prev_data_dir)
        self.total_length = self.accumulate_params("length")
        self.total_num_steps = self.accumulate_params("z_num")
        self.z_inds = np.arange(len(self.params.z_targets))
        self.z = self.params.z_targets
        if self.previous is not None:
            self.z += self.previous.params.z_targets[-1]
            self.params.z_targets = np.concatenate((self.previous.z, self.params.z_targets))
            self.z_inds += self.previous.z_inds[-1] + 1
        self.fiber_lengths = self.all_params("length")
        self.fiber_positions = [
            (this[0], following[1])
            for this, following in zip(self.fiber_lengths, [(None, 0.0)] + self.fiber_lengths)
        ]

    def all_params(self, key: str) -> list[tuple[str, Any]]:
        """returns the value of a parameter for each fiber

        Parameters
        ----------
        key : str
            name of the parameter

        Returns
        -------
        list[tuple[str, Any]]
            list of (fiber_name, param_value) tuples
        """
        return list(reversed(self._all_params(key, [])))

    def accumulate_params(self, key: str) -> Any:
        """returns the sum of all the values a parameter takes. Useful to
        get the total length of the fiber, the total number of steps, etc.

        Parameters
        ----------
        key : str
            name of the parameter

        Returns
        -------
        Any
            final sum
        """
        return sum(el[1] for el in self.all_params(key))

    def spectra(
        self, z_descr: Union[float, int, None] = None, sim_ind: Optional[int] = 0
    ) -> Spectrum:
        if z_descr is None:
            out = [self.spectra(i, sim_ind) for i in range(self.total_num_steps)]
        else:
            if isinstance(z_descr, (float, np.floating)):
                return self.spectra(self.z_ind(z_descr), sim_ind)
            else:
                z_ind = z_descr
            if 0 <= z_ind < self.z_inds[0]:
                return self.previous.spectra(z_ind, sim_ind)
            elif z_ind < 0:
                z_ind = self.total_num_steps + z_ind
            if sim_ind is None:
                out = [self._load_1(z_ind, i) for i in range(self.params.repeat)]
            else:
                out = self._load_1(z_ind)
        return Spectrum(out, self.params)

    def z_ind(self, pos: float) -> int:
        if self.z[0] <= pos <= self.z[-1]:
            return self.z_inds[np.argmin(np.abs(self.z - pos))]
        elif 0 <= pos < self.z[0]:
            return self.previous.z_ind(pos)
        else:
            raise ValueError(f"cannot match z={pos} with max length of {self.total_length}")

    def fields(
        self, z_descr: Union[float, int, None] = None, sim_ind: Optional[int] = 0
    ) -> Spectrum:
        return np.fft.ifft(self.spectra(z_descr, sim_ind))

    # Plotting

    def plot_2D(
        self,
        left: float,
        right: float,
        unit: Union[Callable[[float], float], str],
        ax: plt.Axes,
        sim_ind: int = 0,
        **kwargs,
    ):
        plot_range = PlotRange(left, right, unit)
        vals = self.retrieve_plot_values(plot_range, None, sim_ind)
        return propagation_plot(vals, plot_range, self.params, ax, **kwargs)

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
        plot_range = PlotRange(left, right, unit)
        vals = self.retrieve_plot_values(plot_range, z_pos, sim_ind)
        return single_position_plot(vals, plot_range, self.params, ax, **kwargs)

    def plot_mean(
        self,
        left: float,
        right: float,
        unit: Union[Callable[[float], float], str],
        ax: plt.Axes,
        z_pos: int,
        **kwargs,
    ):
        plot_range = PlotRange(left, right, unit)
        vals = self.retrieve_plot_values(plot_range, z_pos, None)
        return mean_values_plot(vals, plot_range, self.params, ax, **kwargs)

    def retrieve_plot_values(
        self, plot_range: PlotRange, z_pos: Optional[Union[int, float]], sim_ind: Optional[int]
    ):

        if plot_range.unit.type == "TIME":
            return self.fields(z_pos, sim_ind)
        else:
            return self.spectra(z_pos, sim_ind)

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
        for spec in np.moveaxis(self.spectra(None, None), 1, 0):
            x, z, tmp = transform_2D_propagation(spec, (left, right, unit), self.params, False)
            spectra.append(tmp)
        return x, z, pulse.rin_curve(np.moveaxis(spectra, 0, 1))

    # Private

    def _load_1(self, z_ind: int, sim_ind=0) -> np.ndarray:
        """loads a spectrum file

        Parameters
        ----------
        z_ind : int
            z_index relative to the entire simulation
        sim_ind : int, optional
            simulation index, used when repeated simulations with same parameters are ran, by default 0

        Returns
        -------
        np.ndarray
            loaded spectrum file
        """
        if sim_ind > 0:
            return load_spectrum(self.path / SPEC1_FN_N.format(z_ind - self.z_inds[0], sim_ind))
        else:
            return load_spectrum(self.path / SPEC1_FN.format(z_ind - self.z_inds[0]))

    def _all_params(self, key: str, l: list) -> list:
        l.append((self.params.name, getattr(self.params, key)))
        if self.previous is not None:
            return self.previous._all_params(key, l)
        return l

    # Magic methods

    def __iter__(self) -> Iterator[Spectrum]:
        for i in range(self.total_num_steps):
            yield self.spectra(i, None)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(path={self.path}, previous={self.previous!r})"

    def __eq__(self, other: SimulationSeries) -> bool:
        return (
            self.path == other.path
            and self.params == other.params
            and self.previous == other.previous
        )

    def __contains__(self, other: SimulationSeries) -> bool:
        if other is self or other == self:
            return True
        if self.previous is not None:
            return other in self.previous

    def __getitem__(self, key) -> Spectrum:
        if isinstance(key, tuple):
            return self.spectra(*key)
        else:
            return self.spectra(key, None)
