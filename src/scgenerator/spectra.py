import os
from collections.abc import Mapping, Sequence
from glob import glob
from typing import Any, Dict, List, Tuple

import numpy as np

from . import io, initialize, math
from .plotting import units
from .logger import get_logger


class Spectrum(np.ndarray):
    def __new__(cls, input_array, wl, frep=1):
        # Input array is an already formed ndarray instance
        # We first cast to be our class type
        obj = np.asarray(input_array).view(cls)
        # add the new attribute to the created instance
        obj.frep = frep
        obj.wl = wl
        # Finally, we must return the newly created object:
        return obj

    def __array_finalize__(self, obj):
        # see InfoArray.__array_finalize__ for comments
        if obj is None:
            return
        self.frep = getattr(obj, "frep", None)
        self.wl = getattr(obj, "wl", None)


class Pulse(Sequence):
    def __init__(self, path: str, ensure_2d=True):
        self.logger = get_logger(__name__)
        self.path = str(path)
        self.__ensure_2d = ensure_2d

        if not os.path.isdir(self.path):
            raise FileNotFoundError(f"Folder {self.path} does not exist")

        self.params = None
        try:
            self.params = io.load_previous_parameters(os.path.join(self.path, "params.toml"))
        except FileNotFoundError:
            self.logger.info(f"parameters corresponding to {self.path} not found")

        try:
            self.z = np.load(os.path.join(path, "z.npy"))
        except FileNotFoundError:
            if self.params is not None:
                self.z = self.params["z_targets"]
            else:
                raise
        self.cache: Dict[int, Spectrum] = {}
        self.nmax = len(glob(os.path.join(self.path, "spectra_*.npy")))
        if self.nmax <= 0:
            raise FileNotFoundError(f"No appropriate file in specified folder {self.path}")

        self.t = self.params["t"]
        w = initialize.wspace(self.t) + units.m(self.params["wavelength"])
        self.w_order = np.argsort(w)
        self.w = w
        self.wl = units.m.inv(self.w)
        self.params["w"] = self.w
        self.params["z_targets"] = self.z

    def __iter__(self):
        """
        similar to all_spectra but works as an iterator
        """

        self.logger.debug(f"iterating through {self.path}")
        for i in range(self.nmax):
            yield self._load1(i)

    def __len__(self):
        return self.nmax

    def __getitem__(self, key):
        return self.all_spectra(ind=range(self.nmax)[key])

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
        return units.to_WL(math.abs2(spectrum), spectrum.frep, spectrum.wl)

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
                    spectrum.frep,
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

    def all_spectra(self, ind=None):
        """
        loads the data already simulated.
        defauft shape is (z_targets, n, nt)

        Parameters
        ----------
        ind : int or list of int
        if only certain spectra are desired.
                - If left to None, returns every spectrum
                - If only 1 int, will cast the (1, n, nt) array into a (n, nt) array
        Returns
        ----------
        spectra : array
            squeezed array of complex spectra (n simulation on a nt size grid at each ind)
        """

        self.logger.debug(f"opening {self.path}")

        # Check if file exists and assert how many z positions there are

        if ind is None:
            ind = range(self.nmax)
        elif isinstance(ind, int):
            ind = [ind]

        # Load the spectra
        spectra = []
        for i in ind:
            spectra.append(self._load1(i))
        spectra = np.array(spectra)

        self.logger.debug(f"all spectra from {self.path} successfully loaded")

        return spectra

    def all_fields(self, ind=None):
        return np.fft.ifft(self.all_spectra(ind=ind), axis=-1)

    def _load1(self, i: int):
        if i in self.cache:
            return self.cache[i]
        spec = io.load_single_spectrum(self.path, i)
        if self.__ensure_2d:
            spec = np.atleast_2d(spec)
        spec = Spectrum(spec, self.wl, self.params["frep"])
        self.cache[i] = spec
        return spec


class SpectraCollection(Mapping, Sequence):
    def __init__(self, path: str):
        self.path = path
        self.collection: List[Spectra] = []
        if not os.path.isdir(self.path):
            raise FileNotFoundError(f"Folder {self.path} does not exist")

        self.variable_list

    def __getitem__(self, key):
        return self.collection[key]

    def __len__(self):
        pass
