import matplotlib.pyplot as plt
import numpy as np
import pytest

import scgenerator as sc
import scgenerator.operators as op


def test_rk43_absorbtion_only():
    n = 129
    w_c = np.linspace(-5, 5, n)
    spec0 = np.exp(-(w_c**2))

    lin = op.envelope_linear_operator(
        op.constant_quantity(np.zeros(n)),
        op.constant_quantity(np.ones(n) * np.log(2)),
    )
    non_lin = op.no_op_freq(n)

    res = sc.integrate(spec0, 1.0, lin, non_lin, targets=[1.0])
    assert np.max(sc.abs2(res.spectra[-1])) == pytest.approx(0.5)


def test_rk43_soliton(plot=False):
    """
    create a N=3 soliton and test that the spectrum at after one oscillation goes back to the same
    maximum value
    """
    n = 1024
    l0 = 835e-9
    w0 = sc.units.m(l0)
    b2 = sc.fiber.D_to_beta2(sc.units.D_ps_nm_km(24), l0)
    gamma = 0.08
    t0_fwhm = 50e-15
    p0 = 1.26e3
    t0 = sc.pulse.width_to_t0(t0_fwhm, "sech")
    soliton_num = 3
    p0 = soliton_num**2 * np.abs(b2) / (gamma * t0**2)

    disp_len = t0**2 / np.abs(b2)
    end = disp_len * 0.5 * np.pi
    targets = np.linspace(0, end, 64)

    t = np.linspace(-200e-15, 200e-15, n)
    w_c = np.pi * 2 * np.fft.fftfreq(n, t[1] - t[0])
    field0 = sc.pulse.sech_pulse(t, t0, p0)
    spec0 = np.fft.fft(field0)
    no_op = op.no_op_freq(n)

    lin = op.envelope_linear_operator(
        op.constant_polynomial_dispersion([b2], w_c),
        op.constant_quantity(np.zeros(n)),
    )
    non_lin = op.envelope_nonlinear_operator(
        op.constant_quantity(np.ones(n) * gamma),
        op.constant_quantity(np.zeros(n)),
        op.envelope_spm(0),
        no_op,
    )

    res = sc.integrate(spec0, end, lin, non_lin, targets=targets, atol=1e-10, rtol=1e-9)
    if plot:
        x, y, z = sc.plotting.transform_2D_propagation(
            res.spectra,
            sc.PlotRange(500, 1300, "nm"),
            w_c + w0,
            targets,
        )
        plt.imshow(z, extent=sc.plotting.get_extent(x, y), origin="lower", aspect="auto", vmin=-40)
        plt.show()

        plt.plot(sc.abs2(spec0))
        plt.plot(sc.abs2(res.spectra[-1]))
        plt.yscale("log")
        plt.show()

    assert sc.abs2(spec0).max() == pytest.approx(sc.abs2(res.spectra[-1]).max(), rel=0.01)


def benchmark():
    for _ in range(50):
        test_rk43_soliton()


if __name__ == "__main__":
    test_rk43_soliton()
    benchmark()
