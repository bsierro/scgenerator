"""
This file includes Dispersion, NonLinear and Loss classes to be used in the solver
Nothing except the solver should depend on this file
"""
from __future__ import annotations

from typing import Callable

import numpy as np
from scgenerator import math
from scgenerator.logger import get_logger
from scgenerator.physics import fiber, materials, plasma, pulse, units

SpecOperator = Callable[[np.ndarray, float], np.ndarray]
FieldOperator = Callable[[np.ndarray, float], np.ndarray]
VariableQuantity = Callable[[float], float | np.ndarray]
Qualifier = Callable[[np.ndarray, float, float], float]


def no_op_time(t_num) -> SpecOperator:
    arr_const = np.zeros(t_num)

    def operate(spec: np.ndarray, z: float) -> np.ndarray:
        return arr_const

    return operate


def no_op_freq(w_num) -> SpecOperator:
    arr_const = np.zeros(w_num)

    def operate(spec: np.ndarray, z: float) -> np.ndarray:
        return arr_const

    return operate


def constant_array_operator(array: np.ndarray) -> SpecOperator:
    def operate(spec: np.ndarray, z: float) -> np.ndarray:
        return array

    return operate


def constant_quantity(qty: float | np.ndarray) -> VariableQuantity:
    def quantity(z: float) -> float | np.ndarray:
        return qty

    return quantity


##################################################
###################### GAS #######################
##################################################


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
        self.n2_const = self.gas.n2(temperature, pressure)
        self.chi3_const = self.gas.chi3(temperature, pressure)

    def number_density(self, z: float) -> float:
        return self.number_density_const

    def square_index(self, z: float) -> np.ndarray:
        return self.n_gas_2_const

    def n2(self, z: float) -> np.ndarray:
        return self.n2_const

    def chi3(self, z: float) -> np.ndarray:
        return self.chi3_const


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
        length: float,
    ):
        self.p_in = pressure_in
        self.p_out = pressure_out
        self.gas = materials.Gas(gas_name)
        self.temperature = temperature
        self.ideal_gas = ideal_gas
        self.wl_for_disp = wl_for_disp
        self.z_max = length

    def _pressure(self, z: float) -> float:
        return materials.pressure_from_gradient(z / self.z_max, self.p_in, self.p_out)

    def number_density(self, z: float) -> float:
        return self.gas.number_density(self.temperature, self._pressure(z), self.ideal_gas)

    def square_index(self, z: float) -> np.ndarray:
        return self.gas.sellmeier.n_gas_2(self.wl_for_disp, self.temperature, self._pressure(z))

    def n2(self, z: float) -> np.ndarray:
        return self.gas.n2(self.temperature, self._pressure(z))

    def chi3(self, z: float) -> np.ndarray:
        return self.gas.chi3(self.temperature, self._pressure(z))


##################################################
################### DISPERSION ###################
##################################################


def marcatili_refractive_index(
    square_index: VariableQuantity, core_radius: float, wl_for_disp: np.ndarray
) -> VariableQuantity:
    def operate(z: float) -> np.ndarray:
        return fiber.n_eff_marcatili(wl_for_disp, square_index(z), core_radius)

    return operate


def marcatili_adjusted_refractive_index(
    square_index: VariableQuantity, core_radius: float, wl_for_disp: np.ndarray
) -> VariableQuantity:
    def operate(z: float) -> np.ndarray:
        return fiber.n_eff_marcatili_adjusted(wl_for_disp, square_index(z), core_radius)

    return operate


def vincetti_refractive_index(
    square_index: VariableQuantity,
    core_radius: float,
    wl_for_disp: np.ndarray,
    wavelength: float,
    wall_thickness: float,
    tube_radius: float,
    gap: float,
    n_tubes: int,
    n_terms: int,
) -> VariableQuantity:
    def operate(z: float) -> np.ndarray:
        return fiber.n_eff_vincetti(
            wl_for_disp,
            wavelength,
            square_index(z),
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
) -> VariableQuantity:
    """
    dispersion approximated by fitting a polynome on the dispersion and
    evaluating on the envelope
    """
    w_power_fact = np.array(
        [math.power_fact(w_c, k) for k in range(2, len(beta2_coefficients) + 2)]
    )
    disp_arr = fiber.fast_poly_dispersion_op(w_c, beta2_coefficients, w_power_fact)

    return constant_quantity(disp_arr)


def constant_direct_dispersion(
    w_for_disp: np.ndarray,
    w0: np.ndarray,
    t_num: int,
    n_eff: np.ndarray,
    dispersion_ind: np.ndarray,
    w_order: np.ndarray,
) -> VariableQuantity:
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

    return constant_quantity(disp_arr)


def direct_dispersion(
    w_for_disp: np.ndarray,
    w0: np.ndarray,
    t_num: int,
    n_eff_op: SpecOperator,
    dispersion_ind: np.ndarray,
) -> VariableQuantity:
    disp_arr = np.zeros(t_num, dtype=complex)
    w0_ind = math.argclosest(w_for_disp, w0)

    def operate(z: float) -> np.ndarray:
        disp_arr[dispersion_ind] = fiber.fast_direct_dispersion(
            w_for_disp, w0, n_eff_op(z), w0_ind
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

    return constant_quantity(beta_arr)


##################################################
###################### RAMAN #####################
##################################################


def envelope_raman(hr_w:np.ndarra, raman_fraction: float) -> FieldOperator:

    def operate(field: np.ndarray, z: float) -> np.ndarray:
        return raman_fraction * np.fft.ifft(hr_w * np.fft.fft(math.abs2(field)))

    return operate


def full_field_raman(
    raman_type: str, raman_fraction: float, t: np.ndarray, w: np.ndarray, chi3: float
) -> FieldOperator:
    hr_w = fiber.delayed_raman_w(t, raman_type)
    factor_in = units.epsilon0 * chi3 * raman_fraction

    def operate(field: np.ndarray, z: float) -> np.ndarray:
        return factor_in * field * np.fft.irfft(hr_w * np.fft.rfft(math.abs2(field)))

    return operate


##################################################
####################### SPM ######################
##################################################


def envelope_spm(raman_fraction: float) -> FieldOperator:
    fraction = 1 - raman_fraction

    def operate(field: np.ndarray, z: float) -> np.ndarray:
        return fraction * math.abs2(field)

    return operate


def full_field_spm(raman_fraction: float, w: np.ndarray, chi3: float) -> FieldOperator:
    fraction = 1 - raman_fraction
    factor_in = fraction * chi3 * units.epsilon0

    def operate(field: np.ndarray, z: float) -> np.ndarray:
        return factor_in * field**3

    return operate


##################################################
###################### GAMMA #####################
##################################################


def variable_gamma(n2_op: VariableQuantity, w0: float, A_eff: float, t_num: int) -> SpecOperator:
    arr = np.ones(t_num)

    def operate(z: float) -> np.ndarray:
        return arr * fiber.gamma_parameter(n2_op(z), w0, A_eff)

    return operate


##################################################
##################### PLASMA #####################
##################################################


def ionization(
    w: np.ndarray, number_density: VariableQuantity, plasma_obj: plasma.Plasma
) -> FieldOperator:
    def operate(field: np.ndarray, z: float) -> np.ndarray:
        N0 = number_density(z)
        plasma_info = plasma_obj(field, N0)
        # state.stats["ionization_fraction"] = plasma_info.electron_density[-1] / N0
        # state.stats["electron_density"] = plasma_info.electron_density[-1]
        return plasma_info.polarization

    return operate


##################################################
############### CONSERVED QUANTITY ###############
##################################################


def photon_number_with_loss(
    w: np.ndarray, gamma: VariableQuantity, loss: VariableQuantity
) -> Qualifier:
    w = w
    dw = w[1] - w[0]

    def qualify(spec: np.ndarray, z: float, h: float) -> float:
        return pulse.photon_number_with_loss(math.abs2(spec), w, dw, gamma(z), loss(z), h)

    return qualify


def photon_number_without_loss(w: np.ndarray, gamma: VariableQuantity) -> Qualifier:
    dw = w[1] - w[0]

    def qualify(spec: np.ndarray, z: float, h: float) -> float:
        return pulse.photon_number(math.abs2(spec), w, dw, gamma(z))

    return qualify


def energy_with_loss(w: np.ndarray, loss: SpecOperator, conversion_factor: float) -> Qualifier:
    dw = w[1] - w[0]

    def qualify(spec: np.ndarray, z: float, h: float) -> float:
        return pulse.pulse_energy_with_loss(math.abs2(conversion_factor * spec), dw, loss(z), h)

    return qualify


def energy_without_loss(w: np.ndarray, conversion_factor: float) -> Qualifier:
    dw = w[1] - w[0]

    def qualify(spec: np.ndarray, z: float, h: float) -> float:
        return pulse.pulse_energy(math.abs2(conversion_factor * spec), dw)

    return qualify


def conserved_quantity(
    adapt_step_size: bool,
    raman: bool,
    loss: bool,
    gamma: VariableQuantity,
    loss_op: VariableQuantity,
    w: np.ndarray,
    conversion_factor: float,
) -> Qualifier:
    if not adapt_step_size:
        return lambda state: 0.0
    logger = get_logger(__name__)
    if raman and loss:
        logger.debug("Conserved quantity : photon number with loss")
        return photon_number_with_loss(w, gamma, loss_op)
    elif raman:
        logger.debug("Conserved quantity : photon number without loss")
        return photon_number_without_loss(w, gamma)
    elif loss:
        logger.debug("Conserved quantity : energy with loss")
        return energy_with_loss(w, loss_op, conversion_factor)
    else:
        logger.debug("Conserved quantity : energy without loss")
        return energy_without_loss(w, conversion_factor)


##################################################
##################### LINEAR #####################
##################################################


def envelope_linear_operator(
    dispersion_op: VariableQuantity, loss_op: VariableQuantity
) -> VariableQuantity:
    def operate(z: float) -> np.ndarray:
        return dispersion_op(z) - loss_op(z) / 2

    return operate


def full_field_linear_operator(
    beta_op: VariableQuantity,
    loss_op: VariableQuantity,
    frame_velocity: float,
    w: np.ndarray,
) -> VariableQuantity:
    delay = w / frame_velocity

    def operate(z: float) -> np.ndarray:
        return 1j * (beta_op(z) - delay) - loss_op(z) / 2

    return operate


def fullfield_nl_prefactor(w: np.ndarray, n_eff: VariableQuantity) -> VariableQuantity:
    def operate(z: float) -> np.ndarray:
        return w / (2 * units.c * units.epsilon0 * n_eff(z))

    return operate


##################################################
#################### NONLINEAR ###################
##################################################


def envelope_nonlinear_operator(
    gamma_op: VariableQuantity,
    ss_op: VariableQuantity,
    spm_op: FieldOperator,
    raman_op: FieldOperator,
) -> SpecOperator:
    def operate(spec: np.ndarray, z: float) -> np.ndarray:
        field = np.fft.ifft(spec)
        return (
            -1j
            * gamma_op(z)
            * (1 + ss_op(z))
            * np.fft.fft(field * (spm_op(field, z) + raman_op(field, z)))
        )

    return operate


def full_field_nonlinear_operator(
    w: np.ndarray,
    raman_op: FieldOperator,
    spm_op: FieldOperator,
    plasma_op: FieldOperator,
    fullfield_nl_prefactor: VariableQuantity,
) -> SpecOperator:
    def operate(spec: np.ndarray, z: float) -> np.ndarray:
        field = np.fft.irfft(spec)
        total_nonlinear = spm_op(field) + raman_op(field) + plasma_op(field)
        return 1j * fullfield_nl_prefactor(z) * np.fft.rfft(total_nonlinear)

    return operate
