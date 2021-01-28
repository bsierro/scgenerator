import numpy as np
import scgenerator as sc
import matplotlib.pyplot as plt


def convert(l, beta2):
    return l[2:-2] * 1e9, sc.units.beta2_fs_cm.inv(beta2[2:-2])


def test_empty_marcatili():
    l = np.linspace(250, 1200, 500) * 1e-9
    beta2 = sc.fiber.HCPCF_dispersion(l, 15e-6)
    plt.plot(*convert(l, beta2))
    plt.show()


def test_empty_hasan_no_resonance():
    l = np.linspace(250, 1200, 500) * 1e-9
    beta2 = sc.fiber.HCPCF_dispersion(
        l, 12e-6, model="hasan", model_params=dict(t=0.2e-6, g=1e-6, n=6)
    )
    plt.plot(*convert(l, beta2))
    plt.show()


def test_empty_hasan():
    l = np.linspace(250, 1200, 500) * 1e-9
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(6, 7), gridspec_kw=dict(height_ratios=[3, 1]))
    ax.set_ylim(-40, 20)
    ax2.set_ylim(-100, 0)
    beta2 = sc.fiber.HCPCF_dispersion(
        l,
        12e-6,
        model="hasan",
        model_params=dict(t=0.2e-6, g=1e-6, n=6, resonance_strength=(2e-6,)),
    )
    ax.plot(*convert(l, beta2))
    beta2 = sc.fiber.HCPCF_dispersion(
        l, 12e-6, model="hasan", model_params=dict(t=0.2e-6, g=1e-6, n=6)
    )
    ax.plot(*convert(l, beta2))

    l = np.linspace(500, 1500, 500) * 1e-9
    beta2 = sc.fiber.HCPCF_dispersion(
        l, 12e-6, model="hasan", model_params=dict(t=0.2e-6, g=1e-6, n=10)
    )
    ax2.plot(*convert(l, beta2))
    plt.show()


def test_custom_initial_field():
    param = {
        "name": "test",
        "lambda0": [1030, "nm"],
        "E0": [6, "uJ"],
        "T0_FWHM": [27, "fs"],
        "frep": 151e3,
        "z_targets": [0, 0.07, 128],
        "gas": "argon",
        "pressure": 4e5,
        "temperature": 293,
        "pulse_shape": "sech",
        "behaviors": [],
        "fiber_model": "marcatili",
        "model_params": {"core_radius": 18e-6},
        "field_0": "exp(-(t/t0)**2)*P0 + P0/10 * cos(t/t0)*2*exp(-(0.05*t/t0)**2)",
        "nt": 16384,
        "T": 2e-12,
        "adapt_step_size": True,
        "error_ok": 1e-10,
        "interp_range": [120, 2000],
        "n_percent": 2,
    }

    p = sc.compute_init_parameters(dictionary=param)
    fig, ax = plt.subplots()
    ax.plot(p["t"], abs(p["field_0"]))
    plt.show()


if __name__ == "__main__":
    # test_empty_marcatili()
    # test_empty_hasan()
    test_custom_initial_field()