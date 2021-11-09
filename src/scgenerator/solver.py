from abc import abstractmethod

import numpy as np

from . import math
from .logger import get_logger
from .operators import (
    AbstractConservedQuantity,
    CurrentState,
    LinearOperator,
    NonLinearOperator,
    ValueTracker,
)

##################################################
################### STEP-TAKER ###################
##################################################


class StepTaker(ValueTracker):
    linear_operator: LinearOperator
    nonlinear_operator: NonLinearOperator
    _tracked_values: dict[str, float]

    def __init__(self, linear_operator: LinearOperator, nonlinear_operator: NonLinearOperator):
        self.linear_operator = linear_operator
        self.nonlinear_operator = nonlinear_operator
        self._tracked_values = {}

    @abstractmethod
    def __call__(self, state: CurrentState, step_size: float) -> np.ndarray:
        ...

    def all_values(self) -> dict[str, float]:
        """override ValueTracker.all_values to account for the fact that operators are called
        multiple times per step, sometimes with different state, so we use value recorded
        earlier. Please call self.recorde_tracked_values() one time only just after calling
        the linear and nonlinear operators in your StepTaker.

        Returns
        -------
        dict[str, float]
            tracked values
        """
        return self.values() | self._tracked_values

    def record_tracked_values(self):
        self._tracked_values = super().all_values()


class RK4IPStepTaker(StepTaker):
    c2 = 1 / 2
    c3 = 1 / 3
    c6 = 1 / 6
    _cached_values: tuple[np.ndarray, np.ndarray]
    _cached_key: float
    _cache_hits: int
    _cache_misses: int

    def __init__(self, linear_operator: LinearOperator, nonlinear_operator: NonLinearOperator):
        super().__init__(linear_operator, nonlinear_operator)
        self._cached_key = None
        self._cached_values = None
        self._cache_hits = 0
        self._cache_misses = 0

    def __call__(self, state: CurrentState, step_size: float) -> np.ndarray:
        h = step_size
        l0, nl0 = self.cached_values(state)
        expD = np.exp(h * self.c2 * l0)

        A_I = expD * state.spectrum
        k1 = expD * (h * nl0)
        k2 = h * self.nonlinear_operator(state.replace(A_I + k1 * self.c2))
        k3 = h * self.nonlinear_operator(state.replace(A_I + k2 * self.c2))
        k4 = h * self.nonlinear_operator(state.replace(expD * (A_I + k3)))

        return expD * (A_I + k1 * self.c6 + k2 * self.c3 + k3 * self.c3) + k4 * self.c6

    def cached_values(self, state: CurrentState) -> tuple[np.ndarray, np.ndarray]:
        """the evaluation of the linear and nonlinear operators at the start of the step don't
        depend on the step size, so we cache them in case we need them more than once (which
        can happen depending on the adaptive step size controller)


        Parameters
        ----------
        state : CurrentState
            current state of the simulation. state.z is used as the key for the cache

        Returns
        -------
        np.ndarray
            result of the linear operator
        np.ndarray
            result of the nonlinear operator
        """
        if self._cached_key != state.z:
            self._cache_misses += 1
            self._cached_key = state.z
            self._cached_values = self.linear_operator(state), self.nonlinear_operator(state)
            self.record_tracked_values()
        else:
            self._cache_hits += 1
        return self._cached_values

    def values(self) -> dict[str, float]:
        return dict(RK4IP_cache_hits=self._cache_hits, RK4IP_cache_misses=self._cache_misses)


##################################################
################### INTEGRATOR ###################
##################################################


class Integrator(ValueTracker):
    last_step = 0.0

    @abstractmethod
    def __call__(self, state: CurrentState) -> CurrentState:
        """propagate the state with a step size of state.current_step_size
        and return a new state with updated z and previous_step_size attributes"""
        ...


class ConstantStepIntegrator(Integrator):
    def __call__(self, state: CurrentState) -> CurrentState:
        new_state = state.replace(self.step_taker(state, state.current_step_size))
        new_state.z += new_state.current_step_size
        new_state.previous_step_size = new_state.current_step_size
        return new_state

    def values(self) -> dict[str, float]:
        return dict(h=self.last_step)


class ConservedQuantityIntegrator(Integrator):
    step_taker: StepTaker
    conserved_quantity: AbstractConservedQuantity
    last_quantity_value: float
    tolerated_error: float
    local_error: float = 0.0

    def __init__(
        self,
        step_taker: StepTaker,
        conserved_quantity: AbstractConservedQuantity,
        tolerated_error: float,
    ):
        self.conserved_quantity = conserved_quantity
        self.last_quantity_value = 0
        self.tolerated_error = tolerated_error
        self.logger = get_logger(self.__class__.__name__)
        self.size_fac = 2.0 ** (1.0 / 5.0)
        self.step_taker = step_taker

    def __call__(self, state: CurrentState) -> CurrentState:
        keep = False
        h_next_step = state.current_step_size
        while not keep:
            h = h_next_step

            new_state = state.replace(self.step_taker(state, h))

            new_qty = self.conserved_quantity(new_state)
            delta = np.abs(new_qty - self.last_quantity_value) / self.last_quantity_value

            if delta > 2 * self.tolerated_error:
                progress_str = f"step {state.step} rejected with h = {h:.4e}, doing over"
                self.logger.info(progress_str)
                keep = False
                h_next_step = h * 0.5
            elif self.tolerated_error < delta <= 2.0 * self.tolerated_error:
                keep = True
                h_next_step = h / self.size_fac
            elif delta < 0.1 * self.tolerated_error:
                keep = True
                h_next_step = h * self.size_fac
            else:
                keep = True
                h_next_step = h

        self.local_error = delta
        self.last_quantity_value = new_qty
        new_state.current_step_size = h_next_step
        new_state.previous_step_size = h
        new_state.z += h
        self.last_step = h
        return new_state

    def values(self) -> dict[str, float]:
        return dict(
            cons_qty=self.last_quantity_value, h=self.last_step, relative_error=self.local_error
        )


class LocalErrorIntegrator(Integrator):
    step_taker: StepTaker
    tolerated_error: float
    local_error: float

    def __init__(self, step_taker: StepTaker, tolerated_error: float, w_num: int):
        self.tolerated_error = tolerated_error
        self.local_error = 0.0
        self.logger = get_logger(self.__class__.__name__)
        self.size_fac, self.fine_fac, self.coarse_fac = 2.0 ** (1.0 / 5.0), 16 / 15, -1 / 15
        self.step_taker = step_taker

    def __call__(self, state: CurrentState) -> CurrentState:
        keep = False
        h_next_step = state.current_step_size
        while not keep:
            h = h_next_step
            h_half = h / 2
            coarse_spec = self.step_taker(state, h)

            fine_spec1 = self.step_taker(state, h_half)
            fine_state = state.replace(fine_spec1, z=state.z + h_half)
            fine_spec = self.step_taker(fine_state, h_half)

            delta = self.compute_diff(coarse_spec, fine_spec)

            if delta > 2 * self.tolerated_error:
                keep = False
                h_next_step = h_half
            elif self.tolerated_error <= delta <= 2 * self.tolerated_error:
                keep = True
                h_next_step = h / self.size_fac
            elif 0.5 * self.tolerated_error <= delta < self.tolerated_error:
                keep = True
                h_next_step = h
            else:
                keep = True
                h_next_step = h * self.size_fac

        self.local_error = delta
        fine_state.spectrum = fine_spec * self.fine_fac + coarse_spec * self.coarse_fac
        fine_state.current_step_size = h_next_step
        fine_state.previous_step_size = h
        fine_state.z += h
        self.last_step = h
        return fine_state

    def compute_diff(self, coarse_spec: np.ndarray, fine_spec: np.ndarray) -> float:
        return np.sqrt(math.abs2(coarse_spec - fine_spec).sum() / math.abs2(fine_spec).sum())

    def values(self) -> dict[str, float]:
        return dict(relative_error=self.local_error, h=self.last_step)


class ERK43(Integrator):
    linear_operator: LinearOperator
    nonlinear_operator: NonLinearOperator
    dt: float

    def __init__(
        self, linear_operator: LinearOperator, nonlinear_operator: NonLinearOperator, dt: float
    ):
        self.linear_operator = linear_operator
        self.nonlinear_operator = nonlinear_operator
        self.dt = dt

    def __call__(self, state: CurrentState) -> CurrentState:
        keep = False
        h_next_step = state.current_step_size
        while not keep:
            h = h_next_step
            expD = np.exp(h * 0.5 * self.linear_operator(state))
            A_I = expD * state.spectrum
            k1 = expD * state.prev_spectrum
            k2 = self.nonlinear_operator(state.replace(A_I + 0.5 * h * k1))
            k3 = self.nonlinear_operator(state.replace(A_I + 0.5 * h * k2))
            k4 = self.nonlinear_operator(state.replace(expD * A_I + h * k3))
            r = expD * (A_I + h / 6 * (k1 + 2 * k2 + 2 * k3))

            new_fine = r + h / 6 * k4

            k5 = self.nonlinear_operator(state.replace(new_fine))

            new_coarse = r + h / 30 * (2 * k4 + 3 * k5)
