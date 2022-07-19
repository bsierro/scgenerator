from __future__ import annotations

import os
from pathlib import Path
from typing import Callable, Iterator, Optional, Union

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
    transform_1D_values,
    transform_2D_propagation,
)
from .utils import load_spectrum, simulations_list, load_toml
from .legacy import translate_parameters


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
        return math.abs2(self.params.ifft(self))

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
    """
    SimulationsSeries are the interface the user should use to load and
    interact with simulation data. The object loads each fiber of the simulation
    into a separate object and exposes convenience methods to make the series behave
    as a single fiber.

    It should be noted that the last spectrum of a fiber and the first one of the next
    fibers are identical. Therefore, SimulationSeries object will return fewer datapoints
    than when manually mergin the corresponding data.

    """

    path: Path
    fibers: list[SimulatedFiber]
    params: Parameters
    z_indices: list[tuple[int, int]]
    fiber_positions: list[tuple[str, float]]

    def __init__(self, path: os.PathLike):
        """Create a SimulationSeries

        Parameters
        ----------
        path : os.PathLike
            path to the last fiber of the series

        Raises
        ------
        FileNotFoundError
            No simulation found in specified directory
        """
        self.logger = get_logger()
        for self.path in simulations_list(path):
            break
        else:
            raise FileNotFoundError(f"No simulation in {path}")
        self.fibers = [SimulatedFiber(self.path)]
        while (p := self.fibers[-1].params.prev_data_dir) is not None:
            self.fibers.append(SimulatedFiber(p))
        self.fibers = self.fibers[::-1]

        self.fiber_positions = [(self.fibers[0].params.name, 0.0)]
        self.params = Parameters(**self.fibers[0].params.dump_dict(False, False))
        z_targets = list(self.params.z_targets)
        self.z_indices = [(0, j) for j in range(self.params.z_num)]
        for i, fiber in enumerate(self.fibers[1:]):
            self.fiber_positions.append((fiber.params.name, z_targets[-1]))
            z_targets += list(fiber.params.z_targets[1:] + z_targets[-1])
            self.z_indices += [(i + 1, j) for j in range(1, fiber.params.z_num)]
        self.params.z_targets = np.array(z_targets)
        self.params.length = self.params.z_targets[-1]
        self.params.z_num = len(self.params.z_targets)

    def spectra(
        self, z_descr: Union[float, int, None] = None, sim_ind: Optional[int] = 0
    ) -> Spectrum:
        ...
        if z_descr is None:
            out = [self.fibers[i].spectra(j, sim_ind) for i, j in self.z_indices]
        else:
            if isinstance(z_descr, (float, np.floating)):
                fib_ind, z_ind = self.z_ind(z_descr)
            else:
                fib_ind, z_ind = self.z_indices[z_descr]
            out = self.fibers[fib_ind].spectra(z_ind, sim_ind)
        return Spectrum(out, self.params)

    def fields(
        self, z_descr: Union[float, int, None] = None, sim_ind: Optional[int] = 0
    ) -> Spectrum:
        return self.params.ifft(self.spectra(z_descr, sim_ind))

    def z_ind(self, pos: float) -> tuple[int, int]:
        if 0 <= pos <= self.params.length:
            ind = np.argmin(np.abs(self.params.z_targets - pos))
            return self.z_indices[ind]
        else:
            raise ValueError(f"cannot match z={pos} with max length of {self.params.length}")

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

    def plot_values_2D(
        self,
        left: float,
        right: float,
        unit: Union[Callable[[float], float], str],
        sim_ind: int = 0,
        **kwargs,
    ):
        plot_range = PlotRange(left, right, unit)
        vals = self.retrieve_plot_values(plot_range, None, sim_ind)
        return transform_2D_propagation(vals, plot_range, self.params, **kwargs)

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

    def plot_values_1D(
        self,
        left: float,
        right: float,
        unit: Union[Callable[[float], float], str],
        z_pos: int,
        sim_ind: int = 0,
        **kwargs,
    ) -> tuple[np.ndarray, np.ndarray]:
        """gives the desired values already tranformes according to the give range

        Parameters
        ----------
        left : float
            leftmost limit in unit
        right : float
            rightmost limit in unit
        unit : Union[Callable[[float], float], str]
            unit
        z_pos : Union[int, float]
            position either as an index (int) or a real position (float)
        sim_ind : Optional[int]
            which simulation to take when more than one are present

        Returns
        -------
        np.ndarray
            x axis
        np.ndarray
            y values
        """
        plot_range = PlotRange(left, right, unit)
        vals = self.retrieve_plot_values(plot_range, z_pos, sim_ind)
        return transform_1D_values(vals, plot_range, self.params, **kwargs)

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

    # Magic methods

    def __iter__(self) -> Iterator[Spectrum]:
        for i, j in self.z_indices:
            yield self.fibers[i].spectra(j, None)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(path={self.path})"

    def __eq__(self, other: SimulationSeries) -> bool:
        return self.path == other.path and self.params == other.params

    def __contains__(self, fiber: SimulatedFiber) -> bool:
        return fiber in self.fibers

    def __getitem__(self, key) -> Spectrum:
        if isinstance(key, tuple):
            return self.spectra(*key)
        else:
            return self.spectra(key, None)


class SimulatedFiber:
    params: Parameters
    t: np.ndarray
    w: np.ndarray

    def __init__(self, path: os.PathLike):
        self.path = Path(path)
        self.params = Parameters(**translate_parameters(load_toml(self.path / PARAM_FN)))
        self.t = self.params.t
        self.w = self.params.w
        self.z = self.params.z_targets

    def spectra(
        self, z_descr: Union[float, int, None] = None, sim_ind: Optional[int] = 0
    ) -> np.ndarray:
        if z_descr is None:
            out = [self.spectra(i, sim_ind) for i in range(self.params.z_num)]
        else:
            if isinstance(z_descr, (float, np.floating)):
                return self.spectra(self.z_ind(z_descr), sim_ind)
            else:
                z_ind = z_descr

            if z_ind < 0:
                z_ind = self.params.z_num + z_ind

            if sim_ind is None:
                out = [self._load_1(z_ind, i) for i in range(self.params.repeat)]
            else:
                out = self._load_1(z_ind)
        return Spectrum(out, self.params)

    def z_ind(self, pos: float) -> int:
        if 0 <= pos <= self.z[-1]:
            return np.argmin(np.abs(self.z - pos))
        else:
            raise ValueError(f"cannot match z={pos} with max length of {self.params.length}")

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
            return load_spectrum(self.path / SPEC1_FN_N.format(z_ind, sim_ind))
        else:
            return load_spectrum(self.path / SPEC1_FN.format(z_ind))

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(path={self.path})"
