"""
This file includes Dispersion, NonLinear and Loss classes to be used in the solver
Nothing except the solver should depend on this file
"""
from __future__ import annotations

from abc import ABC, abstractmethod
import dataclasses
import numpy as np
from scipy.interpolate import interp1d

from . import math
from .logger import get_logger

from .physics import fiber, pulse


class SpectrumDescriptor:
    name: str
    value: np.ndarray = None

    def __set__(self, instance, value):
        instance.spec2 = math.abs2(value)
        instance.field = np.fft.ifft(value)
        instance.field2 = math.abs2(instance.field)
        self.value = value

    def __get__(self, instance, owner):
        return self.value

    def __delete__(self, instance):
        raise AttributeError("Cannot delete Spectrum field")

    def __set_name__(self, owner, name):
        self.name = name


@dataclasses.dataclass
class CurrentState:
    length: float
    z: float
    h: float
    C_to_A_factor: np.ndarray
    spectrum: np.ndarray = SpectrumDescriptor()
    spec2: np.ndarray = dataclasses.field(init=False)
    field: np.ndarray = dataclasses.field(init=False)
    field2: np.ndarray = dataclasses.field(init=False)

    @property
    def z_ratio(self) -> float:
        return self.z / self.length

    def replace(self, new_spectrum) -> CurrentState:
        return CurrentState(self.length, self.z, self.h, self.C_to_A_factor, new_spectrum)


class Operator(ABC):
    def __repr__(self) -> str:
        value_pair_list = list(self.__dict__.items())
        if len(value_pair_list) > 1:
            value_pair_str_list = [k + "=" + self.__value_repr(k, v) for k, v in value_pair_list]
        else:
            value_pair_str_list = [self.__value_repr(value_pair_list[0][0], value_pair_list[0][1])]

        return self.__class__.__name__ + "(" + ", ".join(value_pair_str_list) + ")"

    def __value_repr(self, k: str, v) -> str:
        if k.endswith("_const"):
            return repr(v[0])
        return repr(v)

    @abstractmethod
    def __call__(self, state: CurrentState) -> np.ndarray:
        pass


class NoOp:
    def __init__(self, t_num: int):
        self.arr_const = np.zeros(t_num)


##################################################
################### DISPERSION ###################
##################################################


class AbstractDispersion(Operator):
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
    dispersion approximated by fitting a polynome on the dispersion and
    evaluating on the envelope
    """

    coefs: np.ndarray
    w_c: np.ndarray

    def __init__(
        self,
        w_for_disp: np.ndarray,
        beta2_arr: np.ndarray,
        w0: float,
        w_c: np.ndarray,
        interpolation_degree: int,
    ):
        self.coefs = fiber.dispersion_coefficients(w_for_disp, beta2_arr, w0, interpolation_degree)
        self.w_c = w_c
        self.w_power_fact = np.array(
            [math.power_fact(w_c, k) for k in range(2, interpolation_degree + 3)]
        )

    def __call__(self, state: CurrentState = None) -> np.ndarray:
        return fiber.fast_poly_dispersion_op(self.w_c, self.coefs, self.w_power_fact)


class ConstantDirectDispersion(AbstractDispersion):
    """
    Direct dispersion for when the refractive index is known
    """

    disp_arr_const: np.ndarray

    def __init__(
        self,
        w_for_disp: np.ndarray,
        w0: np.ndarray,
        t_num: int,
        n_eff: np.ndarray,
        dispersion_ind: np.ndarray,
    ):
        self.disp_arr_const = np.zeros(t_num)
        w0_ind = math.argclosest(w_for_disp, w0)
        self.disp_arr_const[dispersion_ind] = fiber.fast_direct_dispersion(
            w_for_disp, w0, n_eff, w0_ind
        )[2:-2]

    def __call__(self, state: CurrentState = None):
        return self.disp_arr_const


##################################################
###################### LOSS ######################
##################################################


class AbstractLoss(Operator):
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
    arr_const: np.ndarray

    def __init__(self, alpha: float, t_num: int):
        self.arr_const = alpha * np.ones(t_num)

    def __call__(self, state: CurrentState = None) -> np.ndarray:
        return self.arr_const


class NoLoss(ConstantLoss):
    def __init__(self, w: np.ndarray):
        super().__init__(0, w)


class CapillaryLoss(ConstantLoss):
    def __init__(
        self,
        wl_for_disp: np.ndarray,
        dispersion_ind: np.ndarray,
        t_num: int,
        core_radius: float,
        he_mode: tuple[int, int],
    ):
        alpha = fiber.capillary_loss(wl_for_disp, he_mode, core_radius)
        self.arr = np.zeros(t_num)
        self.arr[dispersion_ind] = alpha[2:-2]

    def __call__(self, state: CurrentState) -> np.ndarray:
        return self.arr


class CustomConstantLoss(ConstantLoss):
    def __init__(self, l: np.ndarray, loss_file: str):
        loss_data = np.load(loss_file)
        wl = loss_data["wavelength"]
        loss = loss_data["loss"]
        self.arr = interp1d(wl, loss, fill_value=0, bounds_error=False)(l)

    def __call__(self, state: CurrentState) -> np.ndarray:
        return self.arr


##################################################
##################### LINEAR #####################
##################################################


class LinearOperator(Operator):
    def __init__(self, dispersion_op: AbstractDispersion, loss_op: AbstractLoss):
        self.dispersion_op = dispersion_op
        self.loss_op = loss_op

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
        return self.dispersion_op(state) - self.loss_op(state) / 2


##################################################
###################### RAMAN #####################
##################################################


class AbstractRaman(Operator):
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
        return self.arr_const


class Raman(AbstractRaman):
    def __init__(self, raman_type: str, t: np.ndarray):
        self.hr_w = fiber.delayed_raman_w(t, raman_type)
        self.f_r = 0.245 if raman_type == "agrawal" else 0.18

    def __call__(self, state: CurrentState) -> np.ndarray:
        return self.f_r * np.fft.ifft(self.hr_w * np.fft.fft(state.field2))


##################################################
####################### SPM ######################
##################################################


class AbstractSPM(Operator):
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
        return self.arr_const


class SPM(AbstractSPM):
    def __init__(self, raman_op: AbstractRaman):
        self.fraction = 1 - raman_op.f_r

    def __call__(self, state: CurrentState) -> np.ndarray:
        return self.fraction * state.field2


##################################################
################# SELF-STEEPENING ################
##################################################


class AbstractSelfSteepening(Operator):
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
        return self.arr_const


class SelfSteepening(AbstractSelfSteepening):
    def __init__(self, w_c: np.ndarray, w0: float):
        self.arr = w_c / w0

    def __call__(self, state: CurrentState) -> np.ndarray:
        return self.arr


##################################################
###################### GAMMA #####################
##################################################


class AbstractGamma(Operator):
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


class ConstantScalarGamma(AbstractGamma):
    def __init__(self, gamma: np.ndarray, t_num: int):
        self.arr_const = gamma * np.ones(t_num)

    def __call__(self, state: CurrentState) -> np.ndarray:
        return self.arr_const


class NoGamma(ConstantScalarGamma):
    def __init__(self, w: np.ndarray) -> None:
        super().__init__(0, w)


class ConstantGamma(AbstractGamma):
    def __init__(self, gamma_arr: np.ndarray):
        self.arr = gamma_arr

    def __call__(self, state: CurrentState) -> np.ndarray:
        return self.arr


##################################################
##################### PLASMA #####################
##################################################

##################################################
#################### NONLINEAR ###################
##################################################


class NonLinearOperator(Operator):
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


class EnvelopeNonLinearOperator(NonLinearOperator):
    def __init__(
        self,
        gamma_op: AbstractGamma,
        ss_op: AbstractSelfSteepening,
        spm_op: AbstractSPM,
        raman_op: AbstractRaman,
    ):
        self.gamma_op = gamma_op
        self.ss_op = ss_op
        self.spm_op = spm_op
        self.raman_op = raman_op

    def __call__(self, state: CurrentState) -> np.ndarray:
        return (
            -1j
            * self.gamma_op(state)
            * (1 + self.ss_op(state))
            * np.fft.fft(state.field * (self.spm_op(state) + self.raman_op(state)))
        )


##################################################
############### CONSERVED QUANTITY ###############
##################################################


class ConservedQuantity(Operator):
    @classmethod
    def create(
        cls, raman_op: AbstractGamma, gamma_op: AbstractGamma, loss_op: AbstractLoss, w: np.ndarray
    ) -> ConservedQuantity:
        logger = get_logger(__name__)
        raman = not isinstance(raman_op, NoRaman)
        loss = not isinstance(raman_op, NoLoss)
        if raman and loss:
            logger.debug("Conserved quantity : photon number with loss")
            return PhotonNumberLoss(w, gamma_op, loss_op)
        elif raman:
            logger.debug("Conserved quantity : photon number without loss")
            return PhotonNumberNoLoss(w, gamma_op)
        elif loss:
            logger.debug("Conserved quantity : energy with loss")
            return EnergyLoss(w, loss_op)
        else:
            logger.debug("Conserved quantity : energy without loss")
            return EnergyNoLoss(w)

    @abstractmethod
    def __call__(self, state: CurrentState) -> float:
        pass


class NoConservedQuantity(ConservedQuantity):
    def __call__(self, state: CurrentState) -> float:
        return 0.0


class PhotonNumberLoss(ConservedQuantity):
    def __init__(self, w: np.ndarray, gamma_op: AbstractGamma, loss_op=AbstractLoss):
        self.w = w
        self.dw = w[1] - w[0]
        self.gamma_op = gamma_op
        self.loss_op = loss_op

    def __call__(self, state: CurrentState) -> float:
        return pulse.photon_number_with_loss(
            state.spec2, self.w, self.dw, self.gamma_op(state), self.loss_op(state), state.h
        )


class PhotonNumberNoLoss(ConservedQuantity):
    def __init__(self, w: np.ndarray, gamma_op: AbstractGamma):
        self.w = w
        self.dw = w[1] - w[0]
        self.gamma_op = gamma_op

    def __call__(self, state: CurrentState) -> float:
        return pulse.photon_number(state.spec2, self.w, self.dw, self.gamma_op(state))


class EnergyLoss(ConservedQuantity):
    def __init__(self, w: np.ndarray, loss_op: AbstractLoss):
        self.dw = w[1] - w[0]
        self.loss_op = loss_op

    def __call__(self, state: CurrentState) -> float:
        return pulse.pulse_energy_with_loss(
            math.abs2(state.C_to_A_factor * state.spectrum), self.dw, self.loss_op(state), state.h
        )


class EnergyNoLoss(ConservedQuantity):
    def __init__(self, w: np.ndarray):
        self.dw = w[1] - w[0]

    def __call__(self, state: CurrentState) -> float:
        return pulse.pulse_energy(math.abs2(state.C_to_A_factor * state.spectrum), self.dw)
