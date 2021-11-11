from __future__ import annotations

import logging
from abc import abstractmethod
from typing import Iterator, Type

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


class Integrator(ValueTracker):
    linear_operator: LinearOperator
    nonlinear_operator: NonLinearOperator
    state: CurrentState
    tolerated_error: float
    _tracked_values: dict[str, float]
    logger: logging.Logger
    __registry: dict[str, Type[Integrator]] = {}

    def __init__(
        self,
        init_state: CurrentState,
        linear_operator: LinearOperator,
        nonlinear_operator: NonLinearOperator,
        tolerated_error: float,
    ):
        self.state = init_state
        self.linear_operator = linear_operator
        self.nonlinear_operator = nonlinear_operator
        self.tolerated_error = tolerated_error
        self._tracked_values = {}
        self.logger = get_logger(self.__class__.__name__)

    def __init_subclass__(cls):
        cls.__registry[cls.__name__] = cls

    @classmethod
    def get(cls, integr: str) -> Type[Integrator]:
        return cls.__registry[integr]

    @abstractmethod
    def __iter__(self) -> Iterator[CurrentState]:
        """propagate the state with a step size of state.current_step_size
        and yield a new state with updated z and step attributes"""
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
        return self.values() | self._tracked_values | dict(z=self.state.z, step=self.state.step)

    def record_tracked_values(self):
        self._tracked_values = super().all_values()

    def nl(self, spectrum: np.ndarray) -> np.ndarray:
        return self.nonlinear_operator(self.state.replace(spectrum))


class ConstantStepIntegrator(Integrator):
    def __init__(
        self,
        init_state: CurrentState,
        linear_operator: LinearOperator,
        nonlinear_operator: NonLinearOperator,
    ):
        super().__init__(init_state, linear_operator, nonlinear_operator, 0.0)

    def __iter__(self) -> Iterator[CurrentState]:
        while True:
            lin = self.linear_operator(self.state)
            nonlin = self.nonlinear_operator(self.state)
            self.record_tracked_values()
            new_spec = RK4IP_step(
                self.nonlinear_operator,
                self.state,
                self.state.solution.spectrum,
                self.state.current_step_size,
                lin,
                nonlin,
            )

            self.state.z += self.state.current_step_size
            self.state.step += 1
            self.state.solution = new_spec
            yield self.state


class ConservedQuantityIntegrator(Integrator):
    last_qty: float
    conserved_quantity: AbstractConservedQuantity
    current_error: float = 0.0

    def __init__(
        self,
        init_state: CurrentState,
        linear_operator: LinearOperator,
        nonlinear_operator: NonLinearOperator,
        tolerated_error: float,
        conserved_quantity: AbstractConservedQuantity,
    ):
        super().__init__(init_state, linear_operator, nonlinear_operator, tolerated_error)
        self.conserved_quantity = conserved_quantity
        self.last_qty = self.conserved_quantity(self.state)

    def __iter__(self) -> Iterator[CurrentState]:
        h_next_step = self.state.current_step_size
        size_fac = 2.0 ** (1.0 / 5.0)
        while True:
            lin = self.linear_operator(self.state)
            nonlin = self.nonlinear_operator(self.state)
            self.record_tracked_values()
            while True:
                h = h_next_step
                new_state = self.state.replace(
                    RK4IP_step(
                        self.nonlinear_operator,
                        self.state,
                        self.state.solution.spectrum,
                        h,
                        lin,
                        nonlin,
                    )
                )

                new_qty = self.conserved_quantity(new_state)
                self.current_error = np.abs(new_qty - self.last_qty) / self.last_qty

                if self.current_error > 2 * self.tolerated_error:
                    h_next_step = h * 0.5
                elif self.tolerated_error < self.current_error <= 2.0 * self.tolerated_error:
                    h_next_step = h / size_fac
                    break
                elif self.current_error < 0.1 * self.tolerated_error:
                    h_next_step = h * size_fac
                    break
                else:
                    h_next_step = h
                    break
                self.logger.info(
                    f"step {new_state.step} rejected : {h=}, {self.current_error=}, {h_next_step=}"
                )
            self.last_qty = new_qty
            self.state = new_state
            self.state.current_step_size = h_next_step
            self.state.z += h
            self.state.step += 1
            yield self.state

    def values(self) -> dict[str, float]:
        return dict(cons_qty=self.last_qty, relative_error=self.current_error)


class RK4IPSD(Integrator):
    """Runge-Kutta 4 in Interaction Picture with step doubling"""

    next_h_factor: float = 1.0
    current_error: float = 0.0

    def __iter__(self) -> Iterator[CurrentState]:
        h_next_step = self.state.current_step_size
        size_fac = 2.0 ** (1.0 / 5.0)
        while True:
            lin = self.linear_operator(self.state)
            nonlin = self.nonlinear_operator(self.state)
            self.record_tracked_values()
            while True:
                h = h_next_step
                new_fine_inter = self.take_step(h / 2, self.state.solution.spectrum, lin, nonlin)
                new_fine_inter_state = self.state.replace(new_fine_inter)
                new_fine = self.take_step(
                    h / 2,
                    new_fine_inter,
                    self.linear_operator(new_fine_inter_state),
                    self.nonlinear_operator(new_fine_inter_state),
                )
                new_coarse = self.take_step(h, self.state.solution.spectrum, lin, nonlin)
                self.current_error = self.compute_diff(new_coarse, new_fine)

                if self.current_error > 2 * self.tolerated_error:
                    h_next_step = h * 0.5
                elif self.tolerated_error <= self.current_error <= 2 * self.tolerated_error:
                    h_next_step = h / size_fac
                    break
                elif 0.5 * self.tolerated_error <= self.current_error < self.tolerated_error:
                    h_next_step = h
                    break
                else:
                    h_next_step = h * size_fac
                    break

            self.state.current_step_size = h_next_step
            self.state.z += h
            self.state.step += 1
            self.state.solution = new_fine
            yield self.state

    def take_step(
        self, h: float, spec: np.ndarray, lin: np.ndarray, nonlin: np.ndarray
    ) -> np.ndarray:
        return RK4IP_step(self.nonlinear_operator, self.state, spec, h, lin, nonlin)

    def compute_diff(self, coarse_spec: np.ndarray, fine_spec: np.ndarray) -> float:
        return np.sqrt(math.abs2(coarse_spec - fine_spec).sum() / math.abs2(fine_spec).sum())

    def values(self) -> dict[str, float]:
        return dict(
            step=self.state.step,
            z=self.state.z,
            local_error=self.current_error,
            next_h_factor=self.next_h_factor,
        )


class ERK43(Integrator):
    state: CurrentState
    linear_operator: LinearOperator
    nonlinear_operator: NonLinearOperator
    tolerated_error: float
    dt: float
    current_error: float
    next_h_factor = 1.0

    def __init__(
        self,
        init_state: CurrentState,
        linear_operator: LinearOperator,
        nonlinear_operator: NonLinearOperator,
        tolerated_error: float,
        dw: float,
    ):
        self.state = init_state
        self.linear_operator = linear_operator
        self.nonlinear_operator = nonlinear_operator
        self.dw = dw
        self.tolerated_error = tolerated_error
        self.current_error = 0.0

    def __iter__(self) -> Iterator[CurrentState]:
        h_next_step = self.state.current_step_size
        k5 = self.nonlinear_operator(self.state)
        while True:
            lin = self.linear_operator(self.state)
            self.record_tracked_values()
            while True:
                h = h_next_step
                expD = np.exp(h * 0.5 * lin)
                A_I = expD * self.state.solution.spectrum
                k1 = expD * k5
                k2 = self.nl(A_I + 0.5 * h * k1)
                k3 = self.nl(A_I + 0.5 * h * k2)
                k4 = self.nl(expD * A_I + h * k3)
                r = expD * (A_I + h / 6 * (k1 + 2 * k2 + 2 * k3))

                new_fine = r + h / 6 * k4

                tmp_k5 = self.nl(new_fine)

                new_coarse = r + h / 30 * (2 * k4 + 3 * tmp_k5)

                self.current_error = np.sqrt(self.dw * math.abs2(new_fine - new_coarse).sum())
                self.next_h_factor = max(
                    0.5, min(2.0, (self.tolerated_error / self.current_error) ** 0.25)
                )
                h_next_step = self.next_h_factor * h
                if self.current_error <= self.tolerated_error:
                    break
                self.logger.info(
                    f"step {self.state.step} rejected : {h=}, {self.current_error=}, {h_next_step=}"
                )

            self.state.current_step_size = h_next_step
            self.state.z += h
            self.state.step += 1
            self.state.solution = new_fine
            k5 = tmp_k5
            yield self.state

    def values(self) -> dict[str, float]:
        return dict(
            step=self.state.step,
            z=self.state.z,
            local_error=self.current_error,
            next_h_factor=self.next_h_factor,
        )


class ERK54(ERK43):
    def __iter__(self) -> Iterator[CurrentState]:
        self.logger.info("using ERK54")
        h_next_step = self.state.current_step_size
        k7 = self.nonlinear_operator(self.state)
        while True:
            lin = self.linear_operator(self.state)
            self.record_tracked_values()
            while True:
                h = h_next_step
                expD2 = np.exp(h * 0.5 * lin)
                expD4p = np.exp(h * 0.25 * lin)
                expD4m = 1 / expD4p

                A_I = expD2 * self.state.solution.spectrum
                k1 = expD2 * k7
                k2 = self.nl(A_I + 0.5 * h * k1)
                k3 = expD4p * self.nl(expD4m * (A_I + h / 16 * (3 * k1 + k2)))
                k4 = self.nl(A_I + h / 4 * (k1 - k2 + 4 * k3))
                k5 = expD4m * self.nl(expD4p * (A_I + 3 * h / 16 * (k1 + 3 * k4)))
                k6 = self.nl(expD2 * (A_I + h / 7 * (-2 * k1 + k2 + 12 * k3 - 12 * k4 + 8 * k5)))

                new_fine = (
                    expD2 * (A_I + h / 90 * (7 * k1 + 32 * k3 + 12 * k4 + 32 * k5))
                    + 7 * h / 90 * k6
                )
                tmp_k7 = self.nl(new_fine)
                new_coarse = (
                    expD2 * (A_I + h / 42 * (3 * k1 + 16 * k3 + 4 * k4 + 16 * k5)) + h / 14 * k7
                )

                self.current_error = np.sqrt(self.dw * math.abs2(new_fine - new_coarse).sum())
                self.next_h_factor = max(
                    0.5, min(2.0, (self.tolerated_error / self.current_error) ** 0.25)
                )
                h_next_step = self.next_h_factor * h
                if self.current_error <= self.tolerated_error:
                    break
                self.logger.info(
                    f"step {self.state.step} rejected : {h=}, {self.current_error=}, {h_next_step=}"
                )
            self.state.current_step_size = h_next_step
            self.state.z += h
            self.state.step += 1
            self.state.solution = new_fine
            k7 = tmp_k7
            yield self.state


def RK4IP_step(
    nonlinear_operator: NonLinearOperator,
    init_state: CurrentState,
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

    k1 = expD * init_nonlinear
    k2 = nonlinear_operator(init_state.replace(A_I + k1 * 0.5 * h))
    k3 = nonlinear_operator(init_state.replace(A_I + k2 * 0.5 * h))
    k4 = nonlinear_operator(init_state.replace(expD * (A_I + h * k3)))

    return expD * (A_I + h / 6 * (k1 + 2 * k2 + 2 * k3)) + h / 6 * k4
