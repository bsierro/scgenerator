"""
This file includes Dispersion, NonLinear and Loss classes to be used in the solver
Nothing except the solver should depend on this file
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np
from scipy.interpolate import interp1d

from . import fiber
from .. import math


class SpectrumDescriptor:
    name: str
    value: np.ndarray

    def __set__(self, instance, value):
        instance.field = np.fft.ifft(value)
        self.value = value

    def __get__(self, instance, owner):
        return self.value

    def __delete__(self, instance):
        raise AttributeError("Cannot delete Spectrum field")

    def __set_name__(self, owner, name):
        self.name = name


@dataclass
class CurrentState:
    length: float
    z: float
    h: float
    spectrum: np.ndarray = SpectrumDescriptor()
    field: np.ndarray = field(init=False)

    @property
    def z_ratio(self) -> float:
        return self.z / self.length


class NoOp:
    def __init__(self, w: np.ndarray):
        self.zero_arr = np.zeros_like(w)


##################################################
################### DISPERSION ###################
##################################################


class AbstractDispersion(ABC):
    @abstractmethod
    def __call__(self, state: CurrentState) -> np.ndarray:
        """returns the dispersion in the frequency domain

        Parameters
        ----------
        state : CurrentState
            current state of the simulation

        Returns
        -------
        np.ndarray
            dispersive component
        """


class ConstantPolyDispersion(AbstractDispersion):
    """
    dispersion approximated by fitting a polynom on the dispersion and
    evaluating on the envelope
    """

    coefs: np.ndarray
    w_c: np.ndarray

    def __init__(
        self,
        wl_for_disp: np.ndarray,
        beta2_arr: np.ndarray,
        w0: float,
        w_c: np.ndarray,
        interpolation_range: tuple[float, float] = None,
        interpolation_degree: int = 8,
    ):
        self.coefs = fiber.dispersion_coefficients(
            wl_for_disp, beta2_arr, w0, interpolation_range, interpolation_degree
        )
        self.w_c = w_c
        self.w_power_fact = np.array(
            [math.power_fact(w_c, k) for k in range(2, interpolation_degree + 3)]
        )

    def __call__(self, state: CurrentState) -> np.ndarray:
        return fiber.fast_dispersion_op(self.w_c, self.coefs, self.w_power_fact)


##################################################
##################### LINEAR #####################
##################################################


class LinearOperator:
    def __init__(self, disp: AbstractDispersion, loss: AbstractLoss):
        self.disp = disp
        self.loss = loss

    def __call__(self, state: CurrentState) -> np.ndarray:
        """returns the linear operator to be multiplied by the spectrum in the frequency domain

        Parameters
        ----------
        state : CurrentState
            current state of the simulation

        Returns
        -------
        np.ndarray
            linear component
        """
        return self.disp(state) - self.loss(state) / 2


##################################################
################### NON LINEAR ###################
##################################################

# Raman


class AbstractRaman(ABC):
    f_r: float = 0.0

    @abstractmethod
    def __call__(self, state: CurrentState) -> np.ndarray:
        """returns the raman component

        Parameters
        ----------
        state : CurrentState
            current state of the simulation

        Returns
        -------
        np.ndarray
            raman component
        """


class NoRaman(NoOp, AbstractRaman):
    def __call__(self, state: CurrentState) -> np.ndarray:
        return self.zero_arr


class Raman(AbstractRaman):
    def __init__(self, raman_type: str, t: np.ndarray):
        self.hr_w = fiber.delayed_raman_w(t, raman_type)
        self.f_r = 0.245 if raman_type == "agrawal" else 0.18

    def __call__(self, state: CurrentState) -> np.ndarray:
        return self.f_r * np.fft.ifft(self.hr_w * np.fft.fft(math.abs2(state.field)))


# SPM


class AbstractSPM(ABC):
    fraction: float = 1.0

    @abstractmethod
    def __call__(self, state: CurrentState) -> np.ndarray:
        """returns the SPM component

        Parameters
        ----------
        state : CurrentState
            current state of the simulation

        Returns
        -------
        np.ndarray
            SPM component
        """


class NoSPM(NoOp, AbstractSPM):
    def __call__(self, state: CurrentState) -> np.ndarray:
        return self.zero_arr


class SPM(AbstractSPM):
    def __init__(self, raman_op: AbstractRaman):
        self.fraction = 1 - raman_op.f_r

    def __call__(self, state: CurrentState) -> np.ndarray:
        return self.fraction * math.abs2(state.field)


# Selt Steepening


class AbstractSelfSteepening(ABC):
    @abstractmethod
    def __call__(self, state: CurrentState) -> np.ndarray:
        """returns the self-steepening component

        Parameters
        ----------
        state : CurrentState
            current state of the simulation

        Returns
        -------
        np.ndarray
            self-steepening component
        """


class NoSelfSteepening(NoOp, AbstractSelfSteepening):
    def __call__(self, state: CurrentState) -> np.ndarray:
        return self.zero_arr


class SelfSteepening(AbstractSelfSteepening):
    def __init__(self, w_c: np.ndarray, w0: float):
        self.arr = w_c / w0

    def __call__(self, state: CurrentState) -> np.ndarray:
        return self.arr


# Gamma operator


class AbstractGamma(ABC):
    @abstractmethod
    def __call__(self, state: CurrentState) -> np.ndarray:
        """returns the gamma component

        Parameters
        ----------
        state : CurrentState
            current state of the simulation

        Returns
        -------
        np.ndarray
            gamma component
        """


class NoGamma(AbstractSPM):
    def __init__(self, w: np.ndarray) -> None:
        self.ones_arr = np.ones_like(w)

    def __call__(self, state: CurrentState) -> np.ndarray:
        return self.ones_arr


class ConstantGamma(AbstractSelfSteepening):
    def __init__(self, gamma_arr: np.ndarray):
        self.arr = gamma_arr

    def __call__(self, state: CurrentState) -> np.ndarray:
        return self.arr


# Nonlinear combination


class AbstractNonLinearOperator(ABC):
    @abstractmethod
    def __call__(self, state: CurrentState) -> np.ndarray:
        """returns the nonlinear operator applied on the spectrum in the frequency domain

        Parameters
        ----------
        state : CurrentState
            current state of the simulation

        Returns
        -------
        np.ndarray
            nonlinear component
        """


class EnvelopeNonLinearOperator(AbstractNonLinearOperator):
    def __init__(
        self,
        gamma_op: AbstractGamma,
        ss_op: AbstractSelfSteepening,
        spm_op: AbstractSPM,
        raman_op: AbstractRaman,
    ):
        self.gamma_op = gamma_op


##################################################
###################### LOSS ######################
##################################################


class AbstractLoss(ABC):
    @abstractmethod
    def __call__(self, state: CurrentState) -> np.ndarray:
        """returns the loss in the frequency domain

        Parameters
        ----------
        state : CurrentState
            current state of the simulation

        Returns
        -------
        np.ndarray
            loss in 1/m
        """


class ConstantLoss(AbstractLoss):
    alpha_arr: np.ndarray

    def __init__(self, alpha: float, w: np.ndarray):
        self.alpha_arr = alpha * np.ones_like(w)

    def __call__(self, state: CurrentState) -> np.ndarray:
        return self.alpha_arr


class CapillaryLoss(ConstantLoss):
    def __init__(
        self,
        l: np.ndarray,
        core_radius: float,
        interpolation_range: tuple[float, float],
        he_mode: tuple[int, int],
    ):
        mask = (l < interpolation_range[1]) & (l > 0)
        alpha = fiber.capillary_loss(l[mask], he_mode, core_radius)
        self.alpha_arr = np.zeros_like(l)
        self.alpha_arr[mask] = alpha


class CustomConstantLoss(ConstantLoss):
    def __init__(self, l: np.ndarray, loss_file: str):
        loss_data = np.load(loss_file)
        wl = loss_data["wavelength"]
        loss = loss_data["loss"]
        self.alpha_arr = interp1d(wl, loss, fill_value=0, bounds_error=False)(l)
