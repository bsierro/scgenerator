"""
This file includes Dispersion, NonLinear and Loss classes to be used in the solver
Nothing except the solver should depend on this file
"""
from __future__ import annotations

from abc import abstractmethod
from copy import deepcopy
from dataclasses import dataclass
from functools import cached_property
from typing import Any, Callable, Protocol

import numpy as np
from matplotlib.cbook import operator

from scgenerator import math
from scgenerator.logger import get_logger
from scgenerator.physics import fiber, materials, plasma, pulse, units


class SimulationState:
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

    def __init__(
        self,
        spectrum: np.ndarray,
        length: float = 10.0,
        current_step_size: float = 1.0,
        z: float = 0.0,
        conversion_factor: np.ndarray | float = 1.0,
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
            self.spectrum = spectrum
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

    @cached_property
    def spectrum2(self) -> np.ndarray:
        return math.abs2(self.spectrum)

    @cached_property
    def field(self) -> np.ndarray:
        return self.converter(self.spectrum)

    @cached_property
    def field2(self) -> np.ndarray:
        return math.abs2(self.field)

    def set_spectrum(self, new_spectrum: np.ndarray)->SimulationState:
        """sets the new spectrum and clears cached properties"""
        self.spectrum = new_spectrum
        for el in ["spectrum2", "field", "field2"]:
            if el in self.__dict__:
                delattr(self, el)
        return self

    def clear(self):
        """clears cached properties and stats dict"""
        self.stats = {}
        for el in ["spectrum2", "field", "field2"]:
            if el in self.__dict__:
                delattr(self, el)

    def copy(self) -> SimulationState:
        return SimulationState(
            self.spectrum.copy(),
            self.length,
            self.current_step_size,
            self.z,
            self.conversion_factor,
            self.converter,
            self.spectrum2.copy(),
            self.field.copy(),
            self.field2.copy(),
            deepcopy(self.stats),
        )


Operator = Callable[[SimulationState], np.ndarray]
Qualifier = Callable[[SimulationState], float]


def no_op_time(t_num) -> Operator:
    arr_const = np.zeros(t_num)

    def operate(state: SimulationState) -> np.ndarray:
        return arr_const

    return operate


def no_op_freq(w_num) -> Operator:
    arr_const = np.zeros(w_num)

    def operate(state: SimulationState) -> np.ndarray:
        return arr_const

    return operate


def constant_array_operator(array: np.ndarray) -> Operator:
    def operate(state: SimulationState) -> np.ndarray:
        return array

    return operate


##################################################
###################### GAS #######################
##################################################


class GasOp(Protocol):
    def pressure(self, state: SimulationState) -> float:
        """returns the pressure at the current

        Parameters
        ----------
        state : CurrentState

        Returns
        -------
        float
            pressure un bar
        """

    def number_density(self, state: SimulationState) -> float:
        """returns the number density in 1/m^3 of at the current state

        Parameters
        ----------
        state : CurrentState

        Returns
        -------
        float
            number density in 1/m^3
        """

    def square_index(self, state: SimulationState) -> np.ndarray:
        """returns the square of the material refractive index at the current state

        Parameters
        ----------
        state : CurrentState

        Returns
        -------
        np.ndarray
            n^2
        """


class ConstantGas:
    gas: materials.Gas
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

    def pressure(self, state: SimulationState = None) -> float:
        return self.pressure_const

    def number_density(self, state: SimulationState = None) -> float:
        return self.number_density_const

    def square_index(self, state: SimulationState = None) -> float:
        return self.n_gas_2_const


class PressureGradientGas:
    gas: materials.Gas
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

    def pressure(self, state: SimulationState) -> float:
        return materials.pressure_from_gradient(state.z_ratio, self.p_in, self.p_out)

    def number_density(self, state: SimulationState) -> float:
        return self.gas.number_density(self.temperature, self.pressure(state), self.ideal_gas)

    def square_index(self, state: SimulationState) -> np.ndarray:
        return self.gas.sellmeier.n_gas_2(self.wl_for_disp, self.temperature, self.pressure(state))


##################################################
################### DISPERSION ###################
##################################################


def constant_refractive_index(n_eff: np.ndarray) -> Operator:
    def operate(state: SimulationState) -> np.ndarray:
        return n_eff

    return operate


def marcatili_refractive_index(
    gas_op: GasOp, core_radius: float, wl_for_disp: np.ndarray
) -> Operator:
    def operate(state: SimulationState) -> np.ndarray:
        return fiber.n_eff_marcatili(wl_for_disp, gas_op.square_index(state), core_radius)

    return operate


def marcatili_adjusted_refractive_index(
    gas_op: GasOp, core_radius: float, wl_for_disp: np.ndarray
) -> Operator:
    def operate(state: SimulationState) -> np.ndarray:
        return fiber.n_eff_marcatili_adjusted(wl_for_disp, gas_op.square_index(state), core_radius)

    return operate


def vincetti_refractive_index(
    gas_op: GasOp,
    core_radius: float,
    wl_for_disp: np.ndarray,
    wavelength: float,
    wall_thickness: float,
    tube_radius: float,
    gap: float,
    n_tubes: int,
    n_terms: int,
) -> Operator:
    def operate(state: SimulationState) -> np.ndarray:
        return fiber.n_eff_vincetti(
            wl_for_disp,
            wavelength,
            gas_op.square_index(state),
            wall_thickness,
            tube_radius,
            gap,
            n_tubes,
            n_terms,
        )

    return operate


##################################################
################### DISPERSION ###################
##################################################


def constant_polynomial_dispersion(
    beta2_coefficients: np.ndarray,
    w_c: np.ndarray,
) -> Operator:
    """
    dispersion approximated by fitting a polynome on the dispersion and
    evaluating on the envelope
    """
    w_power_fact = np.array(
        [math.power_fact(w_c, k) for k in range(2, len(beta2_coefficients) + 2)]
    )
    disp_arr = fiber.fast_poly_dispersion_op(w_c, beta2_coefficients, w_power_fact)

    def operate(state: SimulationState) -> np.ndarray:
        return disp_arr

    return operate


def constant_direct_dispersion(
    w_for_disp: np.ndarray,
    w0: np.ndarray,
    t_num: int,
    n_eff: np.ndarray,
    dispersion_ind: np.ndarray,
    w_order: np.ndarray,
) -> Operator:
    """
    Direct dispersion for when the refractive index is known
    """
    disp_arr = np.zeros(t_num, dtype=complex)
    w0_ind = math.argclosest(w_for_disp, w0)
    disp_arr[dispersion_ind] = fiber.fast_direct_dispersion(w_for_disp, w0, n_eff, w0_ind)[2:-2]
    left_ind, *_, right_ind = np.nonzero(disp_arr[w_order])[0]
    disp_arr[w_order] = math._polynom_extrapolation_in_place(
        disp_arr[w_order], left_ind, right_ind, 1
    )

    def operate(state: SimulationState) -> np.ndarray:
        return disp_arr

    return operate


def direct_dispersion(
    w_for_disp: np.ndarray,
    w0: np.ndarray,
    t_num: int,
    n_eff_op: Operator,
    dispersion_ind: np.ndarray,
) -> Operator:
    disp_arr = np.zeros(t_num, dtype=complex)
    w0_ind = math.argclosest(w_for_disp, w0)

    def operate(state: SimulationState) -> np.ndarray:
        disp_arr[dispersion_ind] = fiber.fast_direct_dispersion(
            w_for_disp, w0, n_eff_op(state), w0_ind
        )[2:-2]
        return disp_arr

    return operate


##################################################
################## WAVE VECTOR ###################
##################################################


def constant_wave_vector(
    n_eff: np.ndarray,
    w_for_disp: np.ndarray,
    w_num: int,
    dispersion_ind: np.ndarray,
    w_order: np.ndarray,
):
    beta_arr = np.zeros(w_num, dtype=float)
    beta_arr[dispersion_ind] = fiber.beta(w_for_disp, n_eff)[2:-2]
    left_ind, *_, right_ind = np.nonzero(beta_arr[w_order])[0]
    beta_arr[w_order] = math._polynom_extrapolation_in_place(
        beta_arr[w_order], left_ind, right_ind, 1.0
    )

    def operate(state: SimulationState) -> np.ndarray:
        return beta_arr

    return operate


##################################################
###################### RAMAN #####################
##################################################


def envelope_raman(raman_type: str, raman_fraction: float, t: np.ndarray) -> Operator:
    hr_w = fiber.delayed_raman_w(t, raman_type)

    def operate(state: SimulationState) -> np.ndarray:
        return raman_fraction * np.fft.ifft(hr_w * np.fft.fft(state.field2))

    return operate


def full_field_raman(
    raman_type: str, raman_fraction: float, t: np.ndarray, w: np.ndarray, chi3: float
) -> Operator:
    hr_w = fiber.delayed_raman_w(t, raman_type)
    factor_in = units.epsilon0 * chi3 * raman_fraction
    factor_out = 1j * w**2 / (2.0 * units.c**2 * units.epsilon0)

    def operate(state: SimulationState) -> np.ndarray:
        return factor_out * np.fft.rfft(
            factor_in * state.field * np.fft.irfft(hr_w * np.fft.rfft(state.field2))
        )

    return operate


##################################################
####################### SPM ######################
##################################################


def envelope_spm(raman_fraction: float) -> Operator:
    fraction = 1 - raman_fraction

    def operate(state: SimulationState) -> np.ndarray:
        return fraction * state.field2

    return operate


def full_field_spm(raman_fraction: float, w: np.ndarray, chi3: float) -> Operator:
    fraction = 1 - raman_fraction
    factor_out = 1j * w**2 / (2.0 * units.c**2 * units.epsilon0)
    factor_in = fraction * chi3 * units.epsilon0

    def operate(state: SimulationState) -> np.ndarray:
        return factor_out * np.fft.rfft(factor_in * state.field2 * state.field)

    return operate


##################################################
###################### GAMMA #####################
##################################################


def variable_gamma(gas_op: GasOp, w0: float, A_eff: float, t_num: int) -> Operator:
    arr = np.ones(t_num)

    def operate(state: SimulationState) -> np.ndarray:
        n2 = gas_op.square_index(state)
        return arr * fiber.gamma_parameter(n2, w0, A_eff)

    return operate


##################################################
##################### PLASMA #####################
##################################################


def ionization(w: np.ndarray, gas_op: GasOp, plasma_obj: plasma.Plasma) -> Operator:
    factor_out = -w / (2.0 * units.c**2 * units.epsilon0)

    def operate(state: SimulationState) -> np.ndarray:
        N0 = gas_op.number_density(state)
        plasma_info = plasma_obj(state.field, N0)
        state.stats["ionization_fraction"] = plasma_info.electron_density[-1] / N0
        state.stats["electron_density"] = plasma_info.electron_density[-1]
        return factor_out * np.fft.rfft(plasma_info.polarization)

    return operate


##################################################
############### CONSERVED QUANTITY ###############
##################################################


def photon_number_with_loss(w: np.ndarray, gamma_op: Operator, loss_op: Operator) -> Qualifier:
    w = w
    dw = w[1] - w[0]

    def qualify(state: SimulationState) -> float:
        return pulse.photon_number_with_loss(
            state.spectrum2,
            w,
            dw,
            gamma_op(state),
            loss_op(state),
            state.current_step_size,
        )

    return qualify


def photon_number_without_loss(w: np.ndarray, gamma_op: Operator) -> Qualifier:
    dw = w[1] - w[0]

    def qualify(state: SimulationState) -> float:
        return pulse.photon_number(state.spectrum2, w, dw, gamma_op(state))

    return qualify


def energy_with_loss(w: np.ndarray, loss_op: Operator) -> Qualifier:
    dw = w[1] - w[0]

    def qualify(state: SimulationState) -> float:
        return pulse.pulse_energy_with_loss(
            math.abs2(state.conversion_factor * state.spectrum),
            dw,
            loss_op(state),
            state.current_step_size,
        )

    return qualify


def energy_without_loss(w: np.ndarray) -> Qualifier:
    dw = w[1] - w[0]

    def qualify(state: SimulationState) -> float:
        return pulse.pulse_energy(math.abs2(state.conversion_factor * state.spectrum), dw)

    return qualify


def conserved_quantity(
    adapt_step_size: bool,
    raman: bool,
    loss: bool,
    gamma_op: Operator,
    loss_op: Operator,
    w: np.ndarray,
) -> Qualifier:
    if not adapt_step_size:
        return lambda state: 0.0
    logger = get_logger(__name__)
    if raman and loss:
        logger.debug("Conserved quantity : photon number with loss")
        return photon_number_with_loss(w, gamma_op, loss_op)
    elif raman:
        logger.debug("Conserved quantity : photon number without loss")
        return photon_number_without_loss(w, gamma_op)
    elif loss:
        logger.debug("Conserved quantity : energy with loss")
        return energy_with_loss(w, loss_op)
    else:
        logger.debug("Conserved quantity : energy without loss")
        return energy_without_loss(w)


##################################################
##################### LINEAR #####################
##################################################


def envelope_linear_operator(dispersion_op: Operator, loss_op: Operator) -> Operator:
    def operate(state: SimulationState) -> np.ndarray:
        return dispersion_op(state) - loss_op(state) / 2

    return operate


def full_field_linear_operator(
    beta_op: Operator,
    loss_op: Operator,
    frame_velocity: float,
    w: np.ndarray,
) -> operator:
    delay = w / frame_velocity

    def operate(state: SimulationState) -> np.ndarray:
        return 1j * (beta_op(state) - delay) - loss_op(state) / 2

    return operate


##################################################
#################### NONLINEAR ###################
##################################################


def envelope_nonlinear_operator(
    gamma_op: Operator, ss_op: Operator, spm_op: Operator, raman_op: Operator
) -> Operator:
    def operate(state: SimulationState) -> np.ndarray:
        return (
            -1j
            * gamma_op(state)
            * (1 + ss_op(state))
            * np.fft.fft(state.field * (spm_op(state) + raman_op(state)))
        )

    return operate


def full_field_nonlinear_operator(
    w: np.ndarray,
    raman_op: Operator,
    spm_op: Operator,
    plasma_op: Operator,
    beta_op: Operator,
) -> Operator:
    def operate(state: SimulationState) -> np.ndarray:
        total_nonlinear = spm_op(state) + raman_op(state) + plasma_op(state)
        return total_nonlinear / beta_op(state)

    return operate
