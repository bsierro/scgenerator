from __future__ import annotations

import warnings
from typing import Any, Callable, Iterator, Protocol, Type

import numba
import numpy as np

from scgenerator import math
from scgenerator.logger import get_logger
from scgenerator.operators import Operator, Qualifier, SimulationState
from scgenerator.utils import get_arg_names

Integrator = None


class Stepper(Protocol):
    def set_state(self, state: SimulationState):
        """set the initial state of the stepper"""

    def take_step(
        self, state: SimulationState, step_size: float, new_step: bool
    ) -> tuple[SimulationState, float]:
        """
        Paramters
        ---------
        state : SimulationState
            state of the simulation at the start of the step
        step_size : float
            step size
        new_step : bool
            whether it's the first time this particular step is attempted

        Returns
        -------
        SimulationState
            final state at the end of the step
        float
            estimated numerical error
        """


class StepJudge(Protocol):
    def __call__(self, error: float, step_size: float) -> tuple[float, bool]:
        """
        Parameters
        ----------
        error : float
            relative error
        step_size : float
            step size that lead to `error`

        Returns
        -------
        float
            new step size
        bool
            the given `error` was acceptable and the step should be accepted
        """


def no_judge(error: float, step_size: float) -> tuple[float, bool]:
    """always accept the step and keep the same step size"""
    return step_size, True


def adaptive_judge(
    target_error: float, order: int, step_size_change_bounds: tuple[float, float] = (0.5, 2.0)
) -> Callable[[float, float], tuple[float, bool]]:
    """
    smoothly adapt the step size to stay within a range of tolerated error

    Parameters
    ----------
    target_error : float
        desirable relative local error
    order : float
        order of the integration method
    step_size_change_bounds : tuple[float, float], optional
        lower and upper limit determining how fast the step size may change. By default, the new
        step size it at least half the previous one and at most double.
    """
    exponent = 1 / order
    smin, smax = step_size_change_bounds

    def judge_step(error: float, step_size: float) -> tuple[float, bool]:
        if error > 0:
            next_h_factor = max(smin, min(smax, (target_error / error) ** exponent))
        else:
            next_h_factor = 2.0
        h_next_step = step_size * next_h_factor
        accepted = error <= 2 * target_error
        return h_next_step, accepted

    return judge_step


def decide_step_alt(self, h: float) -> tuple[float, bool]:
    """decides if the current step must be accepted and computes the next step
    size regardless

    Parameters
    ----------
    h : float
        size of the step used to set the current self.current_error

    Returns
    -------
    float
        next step size
    bool
        True if the step must be accepted
    """
    error = self.current_error / self.target_error
    if error > 2:
        accepted = False
        next_h_factor = 0.5
    elif 1 < error <= 2:
        accepted = True
        next_h_factor = 2 ** (-1 / self.order)
    elif 0.1 < error <= 1:
        accepted = True
        next_h_factor = 1.0
    else:
        accepted = True
        next_h_factor = 2 ** (1 / self.order)
    h_next_step = h * next_h_factor
    if not accepted:
        self.steps_rejected += 1
        self.logger.info(
            f"step {self.state.step} rejected : {h=}, {self.current_error=}, {h_next_step=}"
        )
    return h_next_step, accepted


class ConservedQuantityIntegrator:
    last_qty: float
    conserved_quantity: Qualifier
    current_error: float = 0.0

    def __init__(
        self,
        init_state: SimulationState,
        linear_operator: Operator,
        nonlinear_operator: Operator,
        tolerated_error: float,
        conserved_quantity: Qualifier,
    ):
        super().__init__(init_state, linear_operator, nonlinear_operator, tolerated_error)
        self.conserved_quantity = conserved_quantity
        self.last_qty = self.conserved_quantity(self.state)

    def __iter__(self) -> Iterator[SimulationState]:
        while True:
            h_next_step = self.state.current_step_size
            lin = self.linear_operator(self.state)
            nonlin = self.nonlinear_operator(self.state)
            self.record_tracked_values()
            while True:
                h = h_next_step
                new_spec = rk4ip_step(
                    self.nonlinear_operator,
                    self.state,
                    self.state.spectrum,
                    h,
                    lin,
                    nonlin,
                )
                new_state = self.state.replace(new_spec)

                new_qty = self.conserved_quantity(new_state)
                self.current_error = np.abs(new_qty - self.last_qty) / self.last_qty
                h_next_step, accepted = self.decide_step(h)
                if accepted:
                    break
            self.last_qty = new_qty
            yield self.accept_step(new_state, h, h_next_step)

    def values(self) -> dict[str, float]:
        return super().values() | dict(cons_qty=self.last_qty, relative_error=self.current_error)


class RK4IPSD:
    """Runge-Kutta 4 in Interaction Picture with step doubling"""

    next_h_factor: float = 1.0
    current_error: float = 0.0

    def __iter__(self) -> Iterator[SimulationState]:
        while True:
            h_next_step = self.state.current_step_size
            lin = self.linear_operator(self.state)
            nonlin = self.nonlinear_operator(self.state)
            self.record_tracked_values()
            while True:
                h = h_next_step
                new_fine_inter = self.take_step(h / 2, self.state.spectrum, lin, nonlin)
                new_fine_inter_state = self.state.replace(new_fine_inter)
                new_fine = self.take_step(
                    h / 2,
                    new_fine_inter,
                    self.linear_operator(new_fine_inter_state),
                    self.nonlinear_operator(new_fine_inter_state),
                )
                new_coarse = self.take_step(h, self.state.spectrum, lin, nonlin)
                self.current_error = compute_diff(new_coarse, new_fine)
                h_next_step, accepted = self.decide_step(h)
                if accepted:
                    break

            self.state.spectrum = new_fine
            yield self.accept_step(self.state, h, h_next_step)

    def take_step(
        self, h: float, spec: np.ndarray, lin: np.ndarray, nonlin: np.ndarray
    ) -> np.ndarray:
        return rk4ip_step(self.nonlinear_operator, self.state, spec, h, lin, nonlin)

    def values(self) -> dict[str, float]:
        return super().values() | dict(
            z=self.state.z,
            local_error=self.current_error,
        )


class ERKIP43Stepper:
    k5: np.ndarray

    fine: SimulationState
    coarse: SimulationState
    tmp: SimulationState

    def __init__(self, linear_operator: Operator, nonlinear_operator: Operator):
        self.linear_operator = linear_operator
        self.nonlinear_operator = nonlinear_operator

    def set_state(self, state: SimulationState):
        self.k5 = self.nonlinear_operator(state)
        self.fine = state.copy()
        self.tmp = state.copy()
        self.coarse = state.copy()

    def take_step(
        self, state: SimulationState, step_size: float, new_step: bool
    ) -> tuple[SimulationState, float]:
        if not new_step:
            self.k5 = self.nonlinear_operator(state)
        lin = self.linear_operator(state)

        t = self.tmp
        t.z = state.z
        expD = np.exp(step_size * 0.5 * lin)
        A_I = expD * state.spectrum
        k1 = expD * self.k5

        t.set_spectrum(A_I + 0.5 * step_size * k1)
        t.z += step_size * 0.5
        k2 = self.nonlinear_operator(t)

        t.set_spectrum(A_I + 0.5 * step_size * k2)
        k3 = self.nonlinear_operator(t)

        t.set_spectrum(expD * A_I + step_size * k3)
        t.z += step_size * 0.5
        k4 = self.nonlinear_operator(t)

        r = expD * (A_I + step_size / 6 * (k1 + 2 * k2 + 2 * k3))

        self.fine.set_spectrum(r + step_size / 6 * k4)

        self.k5 = self.nonlinear_operator(self.fine)

        self.coarse.set_spectrum(r + step_size / 30 * (2 * k4 + 3 * self.k5))

        error = compute_diff(self.coarse.spectrum, self.fine.spectrum)
        return self.fine, error


class ERKIP54Stepper:
    """
    Reference
    ---------
    [1] BALAC, Stéphane. High order embedded Runge-Kutta scheme for adaptive step-size control in
        the Interaction Picture method. Journal of the Korean Society for Industrial and Applied
        Mathematics, 2013, vol. 17, no 4, p. 238-266.
    """

    k7: np.ndarray
    fine: SimulationState
    coarse: SimulationState
    tmp: SimulationState

    def __init__(
        self,
        linear_operator: Operator,
        nonlinear_operator: Operator,
        error: Callable[[np.ndarray, np.ndarray], float] = None,
    ):
        self.error = error or press_error(1e-6, 1e-6)
        self.linear_operator = linear_operator
        self.nonlinear_operator = nonlinear_operator

    def set_state(self, state: SimulationState):
        self.k7 = self.nonlinear_operator(state)
        self.fine = state.copy()
        self.tmp = state.copy()
        self.coarse = state.copy()

    def take_step(
        self, state: SimulationState, step_size: float, new_step: bool
    ) -> tuple[SimulationState, float]:
        if not new_step:
            self.k7 = self.nonlinear_operator(state)
        lin = self.linear_operator(state)

        t = self.tmp
        expD2p = np.exp(step_size * 0.5 * lin)
        expD4p = np.exp(step_size * 0.25 * lin)
        expD4m = np.exp(-step_size * 0.25 * lin)

        A_I = expD2p * state.spectrum
        k1 = expD2p * self.k7

        t.set_spectrum(A_I + 0.5 * step_size * k1)
        t.z += step_size * 0.5
        k2 = self.nonlinear_operator(t)

        t.set_spectrum(expD4m * (A_I + 0.0625 * step_size * (3 * k1 + k2)))
        t.z -= step_size * 0.25
        k3 = expD4p * self.nonlinear_operator(t)

        t.set_spectrum(A_I + 0.25 * step_size * (-k1 - k2 + 4 * k3))
        t.z += step_size * 0.25
        k4 = self.nonlinear_operator(t)

        t.set_spectrum(expD4p * (A_I + 0.1875 * step_size * (k1 + 3 * k4)))
        t.z += step_size * 0.25
        k5 = expD4m * self.nonlinear_operator(t)

        t.set_spectrum(
            expD2p * (A_I + 1 / 7 * step_size * (-2 * k1 + k2 + 12 * k3 - 12 * k4 + 8 * k5))
        )
        t.z += step_size * 0.25
        k6 = self.nonlinear_operator(t)

        self.fine.set_spectrum(
            expD2p * (A_I + step_size / 90 * (7 * k1 + 32 * k3 + 12 * k4 + 32 * k5))
            + step_size / 90 * k6
        )

        self.k7 = self.nonlinear_operator(self.fine)

        self.coarse.set_spectrum(
            expD2p * (A_I + step_size / 42 * (3 * k1 + 16 * k3 + 4 * k4 + 16 * k5))
            + step_size / 14 * self.k7
        )

        error = compute_diff(self.coarse.spectrum, self.fine.spectrum)
        return self.fine, error


def press_error(atol: float, rtol: float):
    def compute(coarse, fine):
        scale = atol + np.maximum(np.abs(coarse), np.abs(fine)) * rtol
        return np.sqrt(np.mean(math.abs2((coarse - fine) / scale)))

    return compute


def press_judge(error: float, step_size: float) -> tuple[float, bool]:
    return 0.99 * step_size * error**-5, error < 1


def rk4ip_step(
    nonlinear_operator: Operator,
    init_state: SimulationState,
    spectrum: np.ndarray,
    h: float,
    init_linear: np.ndarray,
    init_nonlinear: np.ndarray,
) -> np.ndarray:
    """Take a normal RK4IP step

    Parameters
    ----------
    nonlinear_operator : NonLinearOperator
        non linear operator
    init_state : CurrentState
        state at the start of the step
    h : float
        step size
    spectrum : np.ndarray
        spectrum to propagate
    init_linear : np.ndarray
        linear operator already applied on the initial state
    init_nonlinear : np.ndarray
        nonlinear operator already applied on the initial state

    Returns
    -------
    np.ndarray
        resutling spectrum
    """
    expD = np.exp(h * 0.5 * init_linear)
    A_I = expD * spectrum

    k1 = h * expD * init_nonlinear
    k2 = h * nonlinear_operator(init_state.replace(A_I + k1 * 0.5))
    k3 = h * nonlinear_operator(init_state.replace(A_I + k2 * 0.5))
    k4 = h * nonlinear_operator(init_state.replace(expD * (A_I + k3)))

    return expD * (A_I + k1 / 6 + k2 / 3 + k3 / 3) + k4 / 6


@numba.jit(nopython=True)
def compute_diff(coarse_spec: np.ndarray, fine_spec: np.ndarray) -> float:
    diff = coarse_spec - fine_spec
    diff2 = diff.imag**2 + diff.real**2
    return np.sqrt(diff2.sum() / (fine_spec.real**2 + fine_spec.imag**2).sum())


class ConstantEuler:
    """
    Euler method with constant step size. This is for testing purposes, please do not use this
    method to carry out actual simulations
    """

    def __init__(self, rhs: Callable[[SimulationState], np.ndarray]):
        self.rhs = rhs

    def set_state(self, state: SimulationState):
        self.state = state.copy()

    def take_step(
        self, state: SimulationState, step_size: float, new_step: bool
    ) -> tuple[SimulationState, float]:
        self.state.spectrum = state.spectrum * (1 + step_size * self.rhs(self.state))
        return self.state, 0.0


def integrate(
    stepper: Stepper,
    initial_state: SimulationState,
    step_judge: StepJudge = no_judge,
    min_step_size: float = 1e-6,
    max_step_size: float = float("inf"),
) -> Iterator[SimulationState]:
    state = initial_state.copy()
    state.stats |= dict(rejected_steps=0, z=state.z)
    yield state.copy()
    new_step = True
    num_rejected = 0
    z = 0
    step_size = state.current_step_size
    stepper.set_state(state)
    with warnings.catch_warnings():
        warnings.filterwarnings("error")
        while True:
            new_state, error = stepper.take_step(state, step_size, new_step)
            new_h, step_is_valid = step_judge(error, step_size)
            if step_is_valid:
                z += step_size
                new_state.z = z
                new_state.stats |= dict(rejected_steps=num_rejected, z=z)

                num_rejected = 0

                yield new_state.copy()

                state = new_state
                state.clear()
                new_step = True
            else:
                if num_rejected > 1 and step_size == min_step_size:
                    raise RuntimeError("Solution got rejected even with smallest allowed step size")
                print(f"rejected with h = {step_size:g}")
                num_rejected += 1
                new_step = False

            step_size = min(max_step_size, max(min_step_size, new_h))
            state.current_step_size = step_size
            state.z = z
