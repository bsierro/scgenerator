from __future__ import annotations

import json
import os
import warnings
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterator, Sequence

import numba
import numpy as np

from scgenerator.math import abs2
from scgenerator.operators import SpecOperator, VariableQuantity
from scgenerator.utils import TimedMessage


class SimulationResult:
    spectra: np.ndarray
    size: int
    stats: dict[str, list[Any]]
    z: np.ndarray

    def __init__(
        self,
        spectra: Sequence[np.ndarray],
        stats: dict[str, list[Any]],
        z: np.ndarray | None = None,
    ):
        if z is not None:
            self.z = z
        elif "z" in stats:
            self.z = np.array(stats["z"])
        else:
            self.z = np.arange(len(spectra), dtype=float)
        self.size = len(self.z)
        self.spectra = np.array(spectra)
        self.stats = stats

    def stat(self, stat_name: str) -> np.ndarray:
        return np.array(self.stats[stat_name])

    def save(self, path: os.PathLike):
        path = Path(path)
        if not path.name.endswith(".zip"):
            path = path.parent / (path.name + ".zip")
        with zipfile.ZipFile(path, "w") as zf:
            with zf.open("spectra.npy", "w") as file:
                np.save(file, self.spectra)
            with zf.open("z.npy", "w") as file:
                np.save(file, self.z)
            with zf.open("stats.json", "w") as file:
                file.write(json.dumps(self.stats).encode())

    @classmethod
    def load(cls, path: os.PathLike):
        with zipfile.ZipFile(path, "r") as zf:
            with zf.open("spectra.npy", "r") as file:
                spectra = np.load(file)
            with zf.open("z.npy", "r") as file:
                z = np.load(file)
            with zf.open("stats.json", "r") as file:
                stats = json.loads(file.read().decode())
        return cls(spectra, stats, z)


@numba.jit(nopython=True)
def compute_diff(coarse_spec: np.ndarray, fine_spec: np.ndarray) -> float:
    diff = coarse_spec - fine_spec
    diff2 = diff.imag**2 + diff.real**2
    return np.sqrt(diff2.sum() / (fine_spec.real**2 + fine_spec.imag**2).sum())


def weaknorm(fine: np.ndarray, coarse: np.ndarray, rtol: float, atol: float) -> float:
    alpha = max(max(np.sqrt(abs2(fine).sum()), np.sqrt(abs2(coarse).sum())), atol)
    return 1 / (alpha * rtol) * np.sqrt(abs2(coarse - fine).sum())


def norm_hairer(fine: np.ndarray, coarse: np.ndarray, rtol: float, atol: float) -> float:
    alpha = np.maximum(np.abs(fine), np.abs(coarse))
    return np.sqrt(abs2((fine - coarse) / (atol + rtol * alpha)).mean())


def pi_step_factor(error: float, last_error: float, order: int, eps: float = 0.8):
    """
    computes the next step factor based on the current and last error.

    Parameters
    ----------
    error : float
        error on which to base the new step size
    last_error : float
        error of the last accepted step size
    order : int
        order of the integration method
    eps : arbitrary factor

    Returns
    -------
    float
        factor such that `h_new = factor * h_old`. The factor is smoothly limited to avoid
        increasing or decreasing the step size too fast.

    Reference
    ---------
    [1] SÖDERLIND, Gustaf et WANG, Lina. Adaptive time-stepping and computational stability.
        Journal of Computational and Applied Mathematics, 2006, vol. 185, no 2, p. 225-243.
    """
    b1 = 3 / 5 / order
    b2 = -1 / 5 / order
    last_error = last_error or error
    fac = (eps / error) ** b1 * (eps / last_error) ** b2
    return 1 + np.arctan(fac - 1)


def solve43(
    spec: np.ndarray,
    linear: VariableQuantity,
    nonlinear: SpecOperator,
    z_max: float,
    atol: float,
    rtol: float,
    safety: float,
    h_const: float | None = None,
    targets: Sequence[float] | None = None,
) -> Iterator[tuple[np.ndarray, dict[str, Any]]]:
    """
    Solve the GNLSE using an embedded Runge-Kutta of order 4(3) in the interaction picture.

    Parameters
    ----------
    spec : np.ndarray
        initial spectrum
    linear : Operator
        linear operator
    nonlinear : Operator
        nonlinear operator
    z_max : float
        stop propagation when z >= z_max (the last step is not guaranteed to be exactly on z_max)
    atol : float
        absolute tolerance
    rtol : float
        relative tolerance
    safety : float
        safety factor when computing new step size
    h_const : float | None, optional
        constant step size to use, by default None (automatic step size based on atol and rtol)

    Yields
    ------
    np.ndarray
        last computed spectrum
    dict[str, Any]
        stats about the last step, including `z`
    """
    if h_const is not None:
        h = h_const
        const_step_size = True
    else:
        h = 0.000664237859  # from Luna
        const_step_size = False
    k5 = nonlinear(spec, 0)
    z = 0
    stats = {}
    rejected = []
    if targets is not None:
        targets = list(sorted(set(targets)))
        if targets[0] == 0:
            targets.pop(0)
        h = min(h, targets[0] / 2)

    step_ind = 0
    msg = TimedMessage(2)
    running = True
    last_error = 0
    error = 0
    store_next = False

    def stats():
        return dict(z=z, rejected=rejected.copy(), error=error, h=h)

    yield spec, stats() | dict(h=0)

    while running:
        expD = np.exp(h * 0.5 * linear(z))

        A_I = expD * spec
        k1 = expD * k5
        k2 = nonlinear(A_I + 0.5 * h * k1, z + 0.5 * h)
        k3 = nonlinear(A_I + 0.5 * h * k2, z + 0.5 * h)
        k4 = nonlinear(expD * (A_I + h * k3), z + h)

        r = expD * (A_I + h / 6 * (k1 + 2 * k2 + 2 * k3))

        fine = r + h / 6 * k4

        new_k5 = nonlinear(fine, z + h)

        coarse = r + h / 30 * (2 * k4 + 3 * new_k5)

        error = weaknorm(fine, coarse, rtol, atol)
        if error == 0:  # solution is exact if no nonlinerity is included
            next_h_factor = 1.5
        elif 0 < error <= 1:
            next_h_factor = safety * pi_step_factor(error, last_error, 4, 0.8)
        else:
            next_h_factor = max(0.1, safety * error ** (-0.25))

        if const_step_size or error <= 1:
            k5 = new_k5
            spec = fine
            z += h

            step_ind += 1
            last_error = error

            if targets is None or store_next:
                if targets is not None:
                    targets.pop(0)
                yield fine, stats()

            rejected.clear()
            if z >= z_max:
                return
            if const_step_size:
                continue
        else:
            rejected.append((h, error))
            print(f"{z = :.3f} rejected step {step_ind} with {h = :.2g}, {error = :.2g}")

        h = h * next_h_factor

        if targets is not None and z + h > targets[0]:
            h = targets[0] - z
            store_next = True
        else:
            store_next = False

        if msg.ready():
            print(f"step {step_ind}, {z = :.3f}, {error = :g}, {h = :.3g}")


def integrate(
    initial_spectrum: np.ndarray,
    length: float,
    linear: SpecOperator,
    nonlinear: SpecOperator,
    atol: float = 1e-6,
    rtol: float = 1e-6,
    safety: float = 0.9,
    targets: Sequence[float] | None = None,
) -> SimulationResult:
    spec0 = initial_spectrum.copy()
    all_spectra = []
    stats = defaultdict(list)
    msg = TimedMessage(2)
    with warnings.catch_warnings():
        warnings.filterwarnings("error")
        for i, (spec, new_stat) in enumerate(
            solve43(spec0, linear, nonlinear, length, atol, rtol, safety, targets=targets)
        ):
            if msg.ready():
                print(f"step {i}, z = {new_stat['z']*100:.2f}cm")
            all_spectra.append(spec)
            for k, v in new_stat.items():
                stats[k].append(v)

    return SimulationResult(all_spectra, stats)
