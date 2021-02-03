import os
from collections.abc import Mapping, Sequence
from glob import glob
from typing import Any, List, Tuple

import numpy as np

from . import io
from .logger import get_logger


class Spectra(Sequence):
    def __init__(self, path: str):
        self.logger = get_logger(__name__)
        self.path = path

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

        self.nmax = len(glob(os.path.join(self.path, "spectra_*.npy")))
        if self.nmax <= 0:
            raise FileNotFoundError(f"No appropriate file in specified folder {self.path}")

    def __iter__(self):
        """
        similar to all_spectra but works as an iterator
        """

        self.logger.debug(f"iterating through {self.path}")
        for i in range(self.nmax):
            yield io.load_single_spectrum(self.path, i)

    def __len__(self):
        return self.nmax

    def __getitem__(self, key):
        return self.all_spectra(ind=range(self.nmax)[key])

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
            spectra.append(io.load_single_spectrum(self.path, i))
        spectra = np.array(spectra)

        self.logger.debug(f"all spectra from {self.path} successfully loaded")

        return spectra.squeeze()


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
