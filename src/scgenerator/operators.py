"""
This file includes Dispersion, NonLinear and Loss classes to be used in the solver
Nothing except the solver should depend on this file
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
from scipy.interpolate import interp1d

from scgenerator import math
from scgenerator.logger import get_logger
from scgenerator.physics import fiber, materials, plasma, pulse, units


class CurrentState:
    length: float
    z: float
    current_step_size: float
    conversion_factor: np.ndarray | float
    converter: Callable[[np.ndarray], np.ndarray]
    stats: dict[str, Any]
    spectrum: np.ndarray
    spec2: np.ndarray
    field: np.ndarray
    field2: np.ndarray

    __slots__ = [
        "length",
        "z",
        "current_step_size",
        "conversion_factor",
        "converter",
        "spectrum",
        "spectrum2",
        "field",
        "field2",
        "stats",
    ]

    def __init__(
        self,
        length: float,
        z: float,
        current_step_size: float,
        spectrum: np.ndarray,
        conversion_factor: np.ndarray | float,
        converter: Callable[[np.ndarray], np.ndarray] = np.fft.ifft,
        spectrum2: np.ndarray | None = None,
        field: np.ndarray | None = None,
        field2: np.ndarray | None = None,
        stats: dict[str, Any] | None = None,
    ):
        self.length = length
        self.z = z
        self.current_step_size = current_step_size
        self.conversion_factor = conversion_factor
        self.converter = converter

        if spectrum2 is None and field is None and field2 is None:
            self.set_spectrum(spectrum)
        elif any(el is None for el in (spectrum2, field, field2)):
            raise ValueError(
                "You must provide either all three of (spectrum2, field, field2) or none of them"
            )
        else:
            self.spectrum2 = spectrum2
            self.field = field
            self.field2 = field2
        self.stats = stats or {}

    @property
    def z_ratio(self) -> float:
        return self.z / self.length

    @property
    def actual_spectrum(self) -> np.ndarray:
        return self.conversion_factor * self.spectrum

    def set_spectrum(self, new_spectrum: np.ndarray):
        self.spectrum = new_spectrum
        self.spectrum2 = math.abs2(self.spectrum)
        self.field = self.converter(self.spectrum)
        self.field2 = math.abs2(self.field)

    def copy(self) -> CurrentState:
        return CurrentState(
            self.length,
            self.z,
            self.current_step_size,
            self.spectrum.copy(),
            self.conversion_factor,
            self.converter,
            self.spectrum2.copy(),
            self.field.copy(),
            self.field2.copy(),
            deepcopy(self.stats),
        )


class NoOpTime(Operator):
    def __init__(self, t_num: int):
        self.arr_const = np.zeros(t_num)

    def __call__(self, state: CurrentState) -> np.ndarray:
        """returns 0"""
        return self.arr_const


class NoOpFreq(Operator):
    def __init__(self, w_num: int):
        self.arr_const = np.zeros(w_num)

    def __call__(self, state: CurrentState) -> np.ndarray:
        return self.arr_const


##################################################
###################### GAS #######################
##################################################


class AbstractGas(Operator):
    gas: materials.Gas

    def __call__(self, state: CurrentState) -> np.ndarray:
        return self.square_index(state)

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
        self.gas = materials.Gas(gas_name)
        self.pressure_const = pressure
        self.number_density_const = self.gas.number_density(temperature, pressure, ideal_gas)
        self.n_gas_2_const = self.gas.sellmeier.n_gas_2(wl_for_disp, temperature, pressure)

    def pressure(self, state: CurrentState = None) -> float:
        return self.pressure_const

    def number_density(self, state: CurrentState = None) -> float:
        return self.number_density_const

    def square_index(self, state: CurrentState = None) -> float:
        return self.n_gas_2_const


class PressureGradientGas(AbstractGas):
    p_in: float
    p_out: float
    temperature: float

    def __init__(
        self,
        gas_name: str,
        pressure_in: float,
        pressure_out: float,
        temperature: float,
        ideal_gas: bool,
        wl_for_disp: np.ndarray,
    ):
        self.p_in = pressure_in
        self.p_out = pressure_out
        self.gas = materials.Gas(gas_name)
        self.temperature = temperature
        self.ideal_gas = ideal_gas
        self.wl_for_disp = wl_for_disp

    def pressure(self, state: CurrentState) -> float:
        return materials.pressure_from_gradient(state.z_ratio, self.p_in, self.p_out)

    def number_density(self, state: CurrentState) -> float:
        return self.gas.number_density(self.temperature, self.pressure(state), self.ideal_gas)

    def square_index(self, state: CurrentState) -> np.ndarray:
        return self.gas.sellmeier.n_gas_2(self.wl_for_disp, self.temperature, self.pressure(state))


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
        beta2_coefficients: np.ndarray,
        w_c: np.ndarray,
    ):
        w_power_fact = np.array(
            [math.power_fact(w_c, k) for k in range(2, len(beta2_coefficients) + 2)]
        )
        self.disp_arr = fiber.fast_poly_dispersion_op(w_c, beta2_coefficients, w_power_fact)

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
        self.disp_arr = np.zeros(t_num, dtype=complex)
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
    fraction: float = 0.0

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


class NoEnvelopeRaman(NoOpTime, AbstractRaman):
    pass


class NoFullFieldRaman(NoOpFreq, AbstractRaman):
    pass


class EnvelopeRaman(AbstractRaman):
    def __init__(self, raman_type: str, t: np.ndarray):
        self.hr_w = fiber.delayed_raman_w(t, raman_type)
        self.fraction = 0.245 if raman_type == "agrawal" else 0.18

    def __call__(self, state: CurrentState) -> np.ndarray:
        return self.fraction * np.fft.ifft(self.hr_w * np.fft.fft(state.field2))


class FullFieldRaman(AbstractRaman):
    def __init__(self, raman_type: str, t: np.ndarray, w: np.ndarray, chi3: float):
        self.hr_w = fiber.delayed_raman_w(t, raman_type)
        self.fraction = 0.245 if raman_type == "agrawal" else 0.18
        self.factor_in = units.epsilon0 * chi3 * self.fraction
        self.factor_out = 1j * w**2 / (2.0 * units.c**2 * units.epsilon0)

    def __call__(self, state: CurrentState) -> np.ndarray:
        return self.factor_out * np.fft.rfft(
            self.factor_in * state.field * np.fft.irfft(self.hr_w * np.fft.rfft(state.field2))
        )


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
        self.fraction = 1 - raman_op.fraction

    def __call__(self, state: CurrentState) -> np.ndarray:
        return self.fraction * state.field2


class FullFieldSPM(AbstractSPM):
    def __init__(self, raman_op: AbstractRaman, w: np.ndarray, chi3: float):
        self.fraction = 1 - raman_op.fraction
        self.factor_out = 1j * w**2 / (2.0 * units.c**2 * units.epsilon0)
        self.factor_in = self.fraction * chi3 * units.epsilon0

    def __call__(self, state: CurrentState) -> np.ndarray:
        return self.factor_out * np.fft.rfft(self.factor_in * state.field2 * state.field)


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
    def __init__(self, gamma: float, t_num: int):
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


class VariableScalarGamma(AbstractGamma):
    def __init__(
        self, gas_op: AbstractGas, temperature: float, w0: float, A_eff: float, t_num: int
    ):
        self.gas_op = gas_op
        self.temperature = temperature
        self.w0 = w0
        self.A_eff = A_eff
        self.arr = np.ones(t_num)

    def __call__(self, state: CurrentState) -> np.ndarray:
        n2 = self.gas_op.gas.n2(self.temperature, self.gas_op.pressure(state))
        return self.arr * fiber.gamma_parameter(n2, self.w0, self.A_eff)


##################################################
##################### PLASMA #####################
##################################################


class Plasma(Operator):
    mat_plasma: plasma.Plasma
    gas_op: AbstractGas

    def __init__(self, dt: float, w: np.ndarray, gas_op: AbstractGas):
        self.gas_op = gas_op
        self.mat_plasma = plasma.Plasma(dt, self.gas_op.gas["ionization_energy"])
        self.factor_out = -w / (2.0 * units.c**2 * units.epsilon0)

    def __call__(self, state: CurrentState) -> np.ndarray:
        N0 = self.gas_op.number_density(state)
        plasma_info = self.mat_plasma(state.field, N0)
        state.stats["ionization_fraction"] = plasma_info.electron_density[-1] / N0
        return self.factor_out * np.fft.rfft(plasma_info.polarization)


class NoPlasma(NoOpFreq, Plasma):
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
            state.spectrum2,
            self.w,
            self.dw,
            self.gamma_op(state),
            self.loss_op(state),
            state.current_step_size,
        )


class PhotonNumberNoLoss(AbstractConservedQuantity):
    def __init__(self, w: np.ndarray, gamma_op: AbstractGamma):
        self.w = w
        self.dw = w[1] - w[0]
        self.gamma_op = gamma_op

    def __call__(self, state: CurrentState) -> float:
        return pulse.photon_number(state.spectrum2, self.w, self.dw, self.gamma_op(state))


class EnergyLoss(AbstractConservedQuantity):
    def __init__(self, w: np.ndarray, loss_op: AbstractLoss):
        self.dw = w[1] - w[0]
        self.loss_op = loss_op

    def __call__(self, state: CurrentState) -> float:
        return pulse.pulse_energy_with_loss(
            math.abs2(state.conversion_factor * state.spectrum),
            self.dw,
            self.loss_op(state),
            state.current_step_size,
        )


class EnergyNoLoss(AbstractConservedQuantity):
    def __init__(self, w: np.ndarray):
        self.dw = w[1] - w[0]

    def __call__(self, state: CurrentState) -> float:
        return pulse.pulse_energy(math.abs2(state.conversion_factor * state.spectrum), self.dw)


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
    raman = not isinstance(raman_op, NoEnvelopeRaman)
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
        self.factor = 1j * w**2 / (2.0 * units.c**2 * units.epsilon0)
        self.beta_op = beta_op

    def __call__(self, state: CurrentState) -> np.ndarray:
        total_nonlinear = self.spm_op(state) + self.raman_op(state) + self.plasma_op(state)
        return total_nonlinear / self.beta_op(state)
