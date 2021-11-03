"""
This file includes Dispersion, NonLinear and Loss classes to be used in the solver
Nothing except the solver should depend on this file
"""
from __future__ import annotations

import dataclasses
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
from scipy.interpolate import interp1d

from . import math
from .errors import OperatorError
from .logger import get_logger
from .physics import fiber, materials, pulse, units
from .utils import load_material_dico


class SpectrumDescriptor:
    name: str
    value: np.ndarray = None
    _counter = 0
    _full_field: bool = False
    _converter: Callable[[np.ndarray], np.ndarray]

    def __set__(self, instance: CurrentState, value: np.ndarray):
        self._counter += 1
        instance.spec2 = math.abs2(value)
        instance.field = instance.converter(value)
        instance.field2 = math.abs2(instance.field)
        self.value = value

    def __get__(self, instance, owner):
        return self.value

    def __delete__(self, instance):
        raise AttributeError("Cannot delete Spectrum field")

    def __set_name__(self, owner, name):
        for field_name in ["converter", "field", "field2", "spec2"]:
            if not hasattr(owner, field_name):
                raise AttributeError(f"{owner!r} doesn't have a {field_name!r} attribute")
        self.name = name


@dataclasses.dataclass
class CurrentState:
    length: float
    z: float
    h: float
    C_to_A_factor: np.ndarray
    converter: Callable[[np.ndarray], np.ndarray] = np.fft.ifft
    spectrum: np.ndarray = SpectrumDescriptor()
    spec2: np.ndarray = dataclasses.field(init=False)
    field: np.ndarray = dataclasses.field(init=False)
    field2: np.ndarray = dataclasses.field(init=False)

    @property
    def z_ratio(self) -> float:
        return self.z / self.length

    def replace(self, new_spectrum) -> CurrentState:
        return CurrentState(
            self.length, self.z, self.h, self.C_to_A_factor, self.converter, new_spectrum
        )


class Operator(ABC):
    def __repr__(self) -> str:
        value_pair_list = list(self.__dict__.items())
        if len(value_pair_list) == 0:
            value_pair_str_list = ""
        elif len(value_pair_list) == 1:
            value_pair_str_list = [self.__value_repr(value_pair_list[0][0], value_pair_list[0][1])]
        else:
            value_pair_str_list = [k + "=" + self.__value_repr(k, v) for k, v in value_pair_list]

        return self.__class__.__name__ + "(" + ", ".join(value_pair_str_list) + ")"

    def __value_repr(self, k: str, v) -> str:
        if k.endswith("_const"):
            return repr(v[0])
        return repr(v)

    @abstractmethod
    def __call__(self, state: CurrentState) -> np.ndarray:
        pass


class NoOpTime(Operator):
    def __init__(self, t_num: int):
        self.arr_const = np.zeros(t_num)

    def __call__(self, state: CurrentState) -> np.ndarray:
        return self.arr_const


class NoOpFreq(Operator):
    def __init__(self, w_num: int):
        self.arr_const = np.zeros(w_num)

    def __call__(self, state: CurrentState) -> np.ndarray:
        return self.arr_const


##################################################
###################### GAS #######################
##################################################


class AbstractGas(ABC):
    @abstractmethod
    def pressure(self, state: CurrentState) -> float:
        """returns the pressure at the current

        Parameters
        ----------
        state : CurrentState

        Returns
        -------
        float
            pressure un bar
        """

    @abstractmethod
    def number_density(self, state: CurrentState) -> float:
        """returns the number density in 1/m^3 of at the current state

        Parameters
        ----------
        state : CurrentState

        Returns
        -------
        float
            number density in 1/m^3
        """

    @abstractmethod
    def square_index(self, state: CurrentState) -> np.ndarray:
        """returns the square of the material refractive index at the current state

        Parameters
        ----------
        state : CurrentState

        Returns
        -------
        np.ndarray
            n^2
        """


class ConstantGas(AbstractGas):
    pressure_const: float
    number_density_const: float
    n_gas_2_const: np.ndarray

    def __init__(
        self,
        gas_name: str,
        pressure: float,
        temperature: float,
        ideal_gas: bool,
        wl_for_disp: np.ndarray,
    ):
        gas_dico = load_material_dico(gas_name)
        self.pressure_const = pressure
        if ideal_gas:
            self.number_density_const = materials.number_density_van_der_waals(
                pressure=pressure, temperature=temperature, material_dico=gas_dico
            )
        else:
            self.number_density_const = self.pressure_const / (units.kB * temperature)
        self.n_gas_2_const = materials.n_gas_2(
            wl_for_disp, gas_name, self.pressure_const, temperature, ideal_gas
        )

    def pressure(self, state: CurrentState = None) -> float:
        return self.pressure_const

    def number_density(self, state: CurrentState = None) -> float:
        return self.number_density_const

    def square_index(self, state: CurrentState = None) -> float:
        return self.n_gas_2_const


class PressureGradientGas(AbstractGas):
    name: str
    p_in: float
    p_out: float
    temperature: float
    gas_dico: dict[str, Any]

    def __init__(
        self,
        gas_name: str,
        pressure_in: float,
        pressure_out: float,
        temperature: float,
        ideal_gas: bool,
        wl_for_disp: np.ndarray,
    ):
        self.name = gas_name
        self.p_in = pressure_in
        self.p_out = pressure_out
        self.gas_dico = load_material_dico(gas_name)
        self.temperature = temperature
        self.ideal_gas = ideal_gas
        self.wl_for_disp = wl_for_disp

    def pressure(self, state: CurrentState) -> float:
        return materials.pressure_from_gradient(state.z_ratio, self.p_in, self.p_out)

    def number_density(self, state: CurrentState) -> float:
        if self.ideal:
            return self.pressure(state) / (units.kB * self.temperature)
        else:
            return materials.number_density_van_der_waals(
                pressure=self.pressure(state),
                temperature=self.temperature,
                material_dico=self.gas_dico,
            )

    def square_index(self, state: CurrentState) -> np.ndarray:
        return materials.fast_n_gas_2(
            self.wl_for_disp, self.pressure(state), self.temperature, self.ideal_gas, self.gas_dico
        )


##################################################
################### DISPERSION ###################
##################################################


class AbstractRefractiveIndex(Operator):
    @abstractmethod
    def __call__(self, state: CurrentState) -> np.ndarray:
        """returns the total/effective refractive index at this state

        Parameters
        ----------
        state : CurrentState

        Returns
        -------
        np.ndarray
            refractive index
        """


class ConstantRefractiveIndex(AbstractRefractiveIndex):
    n_eff_arr: np.ndarray

    def __init__(self, n_eff: np.ndarray):
        self.n_eff_arr = n_eff

    def __call__(self, state: CurrentState = None) -> np.ndarray:
        return self.n_eff_arr


class MarcatiliRefractiveIndex(AbstractRefractiveIndex):
    gas_op: AbstractGas
    core_radius: float
    wl_for_disp: np.ndarray

    def __init__(self, gas_op: ConstantGas, core_radius: float, wl_for_disp: np.ndarray):
        self.gas_op = gas_op
        self.core_radius = core_radius
        self.wl_for_disp = wl_for_disp

    def __call__(self, state: CurrentState) -> np.ndarray:
        return fiber.n_eff_marcatili(
            self.wl_for_disp, self.gas_op.square_index(state), self.core_radius
        )


class MarcatiliAdjustedRefractiveIndex(MarcatiliRefractiveIndex):
    def __call__(self, state: CurrentState) -> np.ndarray:
        return fiber.n_eff_marcatili_adjusted(
            self.wl_for_disp, self.gas_op.square_index(state), self.core_radius
        )


@dataclass(repr=False, eq=False)
class HasanRefractiveIndex(AbstractRefractiveIndex):
    gas_op: ConstantGas
    core_radius: float
    capillary_num: int
    capillary_nested: int
    capillary_thickness: float
    capillary_radius: float
    capillary_resonance_strengths: list[float]
    wl_for_disp: np.ndarray

    def __call__(self, state: CurrentState) -> np.ndarray:
        return fiber.n_eff_hasan(
            self.wl_for_disp,
            self.gas_op.square_index(state),
            self.core_radius,
            self.capillary_num,
            self.capillary_nested,
            self.capillary_thickness,
            fiber.capillary_spacing_hasan(
                self.capillary_num, self.capillary_radius, self.core_radius
            ),
            self.capillary_resonance_strengths,
        )


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

    disp_arr: np.ndarray

    def __init__(
        self,
        w_for_disp: np.ndarray,
        beta2_arr: np.ndarray,
        w0: float,
        w_c: np.ndarray,
        interpolation_degree: int,
    ):
        if interpolation_degree < 1:
            raise OperatorError(
                f"interpolation degree of degree {interpolation_degree} incompatible"
            )
        coefs = fiber.dispersion_coefficients(w_for_disp, beta2_arr, w0, interpolation_degree)
        w_power_fact = np.array(
            [math.power_fact(w_c, k) for k in range(2, interpolation_degree + 3)]
        )
        self.disp_arr = fiber.fast_poly_dispersion_op(w_c, coefs, w_power_fact)

    def __call__(self, state: CurrentState = None) -> np.ndarray:
        return self.disp_arr


class ConstantDirectDispersion(AbstractDispersion):
    """
    Direct dispersion for when the refractive index is known
    """

    disp_arr: np.ndarray

    def __init__(
        self,
        w_for_disp: np.ndarray,
        w0: np.ndarray,
        t_num: int,
        n_op: ConstantRefractiveIndex,
        dispersion_ind: np.ndarray,
        w_order: np.ndarray,
    ):
        self.disp_arr = np.zeros(t_num, dtype=complex)
        w0_ind = math.argclosest(w_for_disp, w0)
        self.disp_arr[dispersion_ind] = fiber.fast_direct_dispersion(
            w_for_disp, w0, n_op(), w0_ind
        )[2:-2]
        left_ind, *_, right_ind = np.nonzero(self.disp_arr[w_order])[0]
        self.disp_arr[w_order] = math._polynom_extrapolation_in_place(
            self.disp_arr[w_order], left_ind, right_ind, 1
        )

    def __call__(self, state: CurrentState = None) -> np.ndarray:
        return self.disp_arr


class DirectDispersion(AbstractDispersion):
    def __new__(
        cls,
        w_for_disp: np.ndarray,
        w0: np.ndarray,
        t_num: int,
        n_op: ConstantRefractiveIndex,
        dispersion_ind: np.ndarray,
        w_order: np.ndarray,
    ):
        if isinstance(n_op, ConstantRefractiveIndex):
            return ConstantDirectDispersion(w_for_disp, w0, t_num, n_op, dispersion_ind, w_order)
        return object.__new__(cls)

    def __init__(
        self,
        w_for_disp: np.ndarray,
        w0: np.ndarray,
        t_num: int,
        n_op: ConstantRefractiveIndex,
        dispersion_ind: np.ndarray,
        w_order: np.ndarray,
    ):
        self.w_for_disp = w_for_disp
        self.disp_ind = dispersion_ind
        self.n_op = n_op
        self.disp_arr = np.zeros(t_num)
        self.w0 = w0
        self.w0_ind = math.argclosest(w_for_disp, w0)

    def __call__(self, state: CurrentState) -> np.ndarray:
        self.disp_arr[self.disp_ind] = fiber.fast_direct_dispersion(
            self.w_for_disp, self.w0, self.n_op(state), self.w0_ind
        )[2:-2]
        return self.disp_arr


##################################################
################## WAVE VECTOR ###################
##################################################


class AbstractWaveVector(Operator):
    @abstractmethod
    def __call__(self, state: CurrentState) -> np.ndarray:
        """returns the wave vector beta in the frequency domain

        Parameters
        ----------
        state : CurrentState

        Returns
        -------
        np.ndarray
            wave vector
        """


class ConstantWaveVector(AbstractWaveVector):
    beta_arr: np.ndarray

    def __init__(
        self,
        n_op: ConstantRefractiveIndex,
        w_for_disp: np.ndarray,
        w_num: int,
        dispersion_ind: np.ndarray,
        w_order: np.ndarray,
    ):

        self.beta_arr = np.zeros(w_num, dtype=float)
        self.beta_arr[dispersion_ind] = fiber.beta(w_for_disp, n_op())[2:-2]
        left_ind, *_, right_ind = np.nonzero(self.beta_arr[w_order])[0]
        self.beta_arr[w_order] = math._polynom_extrapolation_in_place(
            self.beta_arr[w_order], left_ind, right_ind, 1.0
        )

    def __call__(self, state: CurrentState) -> np.ndarray:
        return self.beta_arr


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

        Returns
        -------
        np.ndarray
            loss in 1/m
        """


class ConstantLoss(AbstractLoss):
    arr_const: np.ndarray

    def __init__(self, alpha: float, w_num: int):
        self.arr_const = alpha * np.ones(w_num)

    def __call__(self, state: CurrentState = None) -> np.ndarray:
        return self.arr_const


class NoLoss(ConstantLoss):
    def __init__(self, w_num: int):
        super().__init__(0, w_num)


class CapillaryLoss(ConstantLoss):
    def __init__(
        self,
        wl_for_disp: np.ndarray,
        dispersion_ind: np.ndarray,
        w_num: int,
        core_radius: float,
        he_mode: tuple[int, int],
    ):
        alpha = fiber.capillary_loss(wl_for_disp, he_mode, core_radius)
        self.arr = np.zeros(w_num)
        self.arr[dispersion_ind] = alpha[2:-2]

    def __call__(self, state: CurrentState = None) -> np.ndarray:
        return self.arr


class CustomLoss(ConstantLoss):
    def __init__(self, l: np.ndarray, loss_file: str):
        loss_data = np.load(loss_file)
        wl = loss_data["wavelength"]
        loss = loss_data["loss"]
        self.arr = interp1d(wl, loss, fill_value=0, bounds_error=False)(l)

    def __call__(self, state: CurrentState = None) -> np.ndarray:
        return self.arr


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

        Returns
        -------
        np.ndarray
            raman component
        """


class NoRaman(NoOpTime, AbstractRaman):
    pass


class EnvelopeRaman(AbstractRaman):
    def __init__(self, raman_type: str, t: np.ndarray):
        self.hr_w = fiber.delayed_raman_w(t, raman_type)
        self.f_r = 0.245 if raman_type == "agrawal" else 0.18

    def __call__(self, state: CurrentState) -> np.ndarray:
        return self.f_r * np.fft.ifft(self.hr_w * np.fft.fft(state.field2))


class FullFieldRaman(AbstractRaman):
    def __init__(self, raman_type: str, t: np.ndarray, chi3: float):
        self.hr_w = fiber.delayed_raman_w(t, raman_type)
        self.f_r = 0.245 if raman_type == "agrawal" else 0.18
        self.multiplier = units.epsilon0 * chi3 * self.f_r

    def __call__(self, state: CurrentState) -> np.ndarray:
        return self.multiplier * np.fft.ifft(np.fft.fft(state.field2) * self.hr_w)


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

        Returns
        -------
        np.ndarray
            SPM component
        """


class NoEnvelopeSPM(NoOpFreq, AbstractSPM):
    pass


class NoFullFieldSPM(NoOpTime, AbstractSPM):
    pass


class EnvelopeSPM(AbstractSPM):
    def __init__(self, raman_op: AbstractRaman):
        self.fraction = 1 - raman_op.f_r

    def __call__(self, state: CurrentState) -> np.ndarray:
        return self.fraction * state.field2


class FullFieldSPM(AbstractSPM):
    def __init__(self, raman_op: AbstractRaman, chi3: float):
        self.fraction = 1 - raman_op.f_r
        self.factor = self.fraction * chi3 * units.epsilon0

    def __call__(self, state: CurrentState) -> np.ndarray:
        return self.factor * state.field2 * state.field


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

        Returns
        -------
        np.ndarray
            self-steepening component
        """


class NoSelfSteepening(NoOpFreq, AbstractSelfSteepening):
    pass


class SelfSteepening(AbstractSelfSteepening):
    def __init__(self, w_c: np.ndarray, w0: float):
        self.arr = w_c / w0

    def __call__(self, state: CurrentState = None) -> np.ndarray:
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


class Plasma(Operator):
    pass


class NoPlasma(NoOpTime, Plasma):
    pass


##################################################
############### CONSERVED QUANTITY ###############
##################################################


class AbstractConservedQuantity(Operator):
    @abstractmethod
    def __call__(self, state: CurrentState) -> float:
        pass


class NoConservedQuantity(AbstractConservedQuantity):
    def __call__(self, state: CurrentState) -> float:
        return 0.0


class PhotonNumberLoss(AbstractConservedQuantity):
    def __init__(self, w: np.ndarray, gamma_op: AbstractGamma, loss_op: AbstractLoss):
        self.w = w
        self.dw = w[1] - w[0]
        self.gamma_op = gamma_op
        self.loss_op = loss_op

    def __call__(self, state: CurrentState) -> float:
        return pulse.photon_number_with_loss(
            state.spec2, self.w, self.dw, self.gamma_op(state), self.loss_op(state), state.h
        )


class PhotonNumberNoLoss(AbstractConservedQuantity):
    def __init__(self, w: np.ndarray, gamma_op: AbstractGamma):
        self.w = w
        self.dw = w[1] - w[0]
        self.gamma_op = gamma_op

    def __call__(self, state: CurrentState) -> float:
        return pulse.photon_number(state.spec2, self.w, self.dw, self.gamma_op(state))


class EnergyLoss(AbstractConservedQuantity):
    def __init__(self, w: np.ndarray, loss_op: AbstractLoss):
        self.dw = w[1] - w[0]
        self.loss_op = loss_op

    def __call__(self, state: CurrentState) -> float:
        return pulse.pulse_energy_with_loss(
            math.abs2(state.C_to_A_factor * state.spectrum), self.dw, self.loss_op(state), state.h
        )


class EnergyNoLoss(AbstractConservedQuantity):
    def __init__(self, w: np.ndarray):
        self.dw = w[1] - w[0]

    def __call__(self, state: CurrentState) -> float:
        return pulse.pulse_energy(math.abs2(state.C_to_A_factor * state.spectrum), self.dw)


def conserved_quantity(
    adapt_step_size: bool,
    raman_op: AbstractGamma,
    gamma_op: AbstractGamma,
    loss_op: AbstractLoss,
    w: np.ndarray,
) -> AbstractConservedQuantity:
    if not adapt_step_size:
        return NoConservedQuantity()
    logger = get_logger(__name__)
    raman = not isinstance(raman_op, NoRaman)
    loss = not isinstance(loss_op, NoLoss)
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


##################################################
##################### LINEAR #####################
##################################################


class LinearOperator(Operator):
    def __call__(self, state: CurrentState) -> np.ndarray:
        """returns the linear operator to be multiplied by the spectrum in the frequency domain

        Parameters
        ----------
        state : CurrentState

        Returns
        -------
        np.ndarray
            linear component
        """


class EnvelopeLinearOperator(LinearOperator):
    def __init__(self, dispersion_op: AbstractDispersion, loss_op: AbstractLoss):
        self.dispersion_op = dispersion_op
        self.loss_op = loss_op

    def __call__(self, state: CurrentState) -> np.ndarray:
        return self.dispersion_op(state) - self.loss_op(state) / 2


class FullFieldLinearOperator(LinearOperator):
    def __init__(
        self,
        beta_op: AbstractWaveVector,
        loss_op: AbstractLoss,
        frame_velocity: float,
        w: np.ndarray,
    ):
        self.delay = w / frame_velocity
        self.wave_vector = beta_op
        self.loss_op = loss_op

    def __call__(self, state: CurrentState) -> np.ndarray:
        return 1j * (self.wave_vector(state) - self.delay) - self.loss_op(state) / 2


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


class FullFieldNonLinearOperator(NonLinearOperator):
    def __init__(
        self,
        raman_op: AbstractRaman,
        spm_op: AbstractSPM,
        plasma_op: Plasma,
        w: np.ndarray,
        beta_op: AbstractWaveVector,
    ):
        self.raman_op = raman_op
        self.spm_op = spm_op
        self.plasma_op = plasma_op
        self.factor = 1j * w ** 2 / (2.0 * units.c ** 2 * units.epsilon0)
        self.beta_op = beta_op

    def __call__(self, state: CurrentState) -> np.ndarray:
        total_nonlinear = self.spm_op(state) + self.raman_op(state) + self.plasma_op(state)
        return self.factor / self.beta_op(state) * np.fft.rfft(total_nonlinear)
