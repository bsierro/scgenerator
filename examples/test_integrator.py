from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import pytest
from scipy.interpolate import interp1d

import scgenerator as sc
import scgenerator.operators as op
import scgenerator.solver as so


def test_rk43_absorbtion_only():
    n = 129
    end = 1.0
    h = 2**-3
    w_c = np.linspace(-5, 5, n)
    ind = np.argsort(w_c)
    spec0 = np.exp(-(w_c**2))
    init_state = op.SimulationState(spec0, end, h)

    lin = op.envelope_linear_operator(
        op.no_op_freq(n),
        op.constant_array_operator(np.ones(n) * np.log(2)),
    )
    non_lin = op.no_op_freq(n)

    judge = so.adaptive_judge(1e-6, 4)
    stepper = so.ERK43(lin, non_lin)

    for state in so.integrate(stepper, init_state, h, judge, max_step_size=0.125):
        if state.z >= end:
            break
    assert np.max(state.spectrum2) == pytest.approx(0.5)


def test_rk43_soliton(plot=False):
    n = 1024
    b2 = sc.fiber.D_to_beta2(sc.units.D_ps_nm_km(24), 835e-9)
    gamma = 0.08
    t0_fwhm = 50e-15
    p0 = 1.26e3
    t0 = sc.pulse.width_to_t0(t0_fwhm, "sech")
    print(np.sqrt(t0**2 / np.abs(b2) * gamma * p0))

    disp_len = t0**2 / np.abs(b2)
    end = disp_len * 0.5 * np.pi
    targets = np.linspace(0, end, 32)
    print(end)

    h = 2**-6
    t = np.linspace(-200e-15, 200e-15, n)
    w_c = np.pi * 2 * np.fft.fftfreq(n, t[1] - t[0])
    ind = np.argsort(w_c)
    field0 = sc.pulse.sech_pulse(t, t0, p0)
    init_state = op.SimulationState(np.fft.fft(field0), end, h)
    no_op = op.no_op_freq(n)

    lin = op.envelope_linear_operator(
        op.constant_polynomial_dispersion([b2], w_c),
        no_op,
        # op.constant_array_operator(np.ones(n) * np.log(2)),
    )
    non_lin = op.envelope_nonlinear_operator(
        op.constant_array_operator(np.ones(n) * gamma),
        no_op,
        op.envelope_spm(0),
        no_op,
    )

    # new_state = init_state.copy()
    # plt.plot(t, init_state.field2)
    # new_state.set_spectrum(non_lin(init_state))
    # plt.plot(t, new_state.field2)
    # new_state.set_spectrum(lin(init_state))
    # plt.plot(t, new_state.field2)
    # print(new_state.spectrum2.max())
    # plt.show()
    # return

    judge = so.adaptive_judge(1e-6, 4)
    stepper = so.ERKIP43Stepper(lin, non_lin)

    # stepper.set_state(init_state)
    # state, error = stepper.take_step(init_state, 1e-3, True)
    # print(error)
    # plt.plot(t, stepper.fine.field2)
    # plt.plot(t, stepper.coarse.field2)
    # plt.show()
    # return

    target = 0
    stats = defaultdict(list)
    saved = []
    zs = []
    for state in so.integrate(stepper, init_state, h, judge, max_step_size=0.125):
        # print(f"z = {state.z*100:.2f}")
        saved.append(state.spectrum2[ind])
        zs.append(state.z)
        for k, v in state.stats.items():
            stats[k].append(v)
        if state.z > end:
            break
    print(len(saved))
    if plot:
        interp = interp1d(zs, saved, axis=0)
        plt.imshow(sc.units.to_log(interp(targets)), origin="lower", aspect="auto", vmin=-40)
        plt.show()

        plt.plot(stats["z"][1:], np.diff(stats["z"]))
        plt.show()


def test_simple_euler():
    n = 129
    end = 1.0
    h = 2**-3
    w_c = np.linspace(-5, 5, n)
    ind = np.argsort(w_c)
    spec0 = np.exp(-(w_c**2))
    init_state = op.SimulationState(spec0, end, h)

    lin = op.envelope_linear_operator(
        op.no_op_freq(n),
        op.constant_array_operator(np.ones(n) * np.log(2)),
    )
    euler = so.ConstantEuler(lin)

    target = 0
    end = 1.0
    h = 2**-6
    for state in so.integrate(euler, init_state, h):
        if state.z >= target:
            target += 0.125
            plt.plot(w_c[ind], state.spectrum2[ind], label=f"z={state.z:.3f}")
        if target > end:
            print(np.max(state.spectrum2))
            break
    plt.title(f"{h = }")
    plt.legend()
    plt.show()


def benchmark():
    for _ in range(50):
        test_rk43_soliton()


if __name__ == "__main__":
    test_rk43_soliton()
    benchmark()
