from math import factorial

import numpy as np
import pytest

import scgenerator.math as m


def test__power_fact_array():
    x = np.random.rand(5)
    for i in range(5):
        assert m._power_fact_array(x, i) == pytest.approx(x**i / factorial(i))


def test__power_fact_single():
    pass


def test_abs2():
    x = np.random.rand(5)
    assert m.abs2(5) == 25
    assert m.abs2(2 - 2j) == 8
    assert all(m.abs2(x) == abs(x) ** 2)


def test_all_zeros():
    x = np.geomspace(0.1, 1, 100)
    y = np.sin(1 / x)
    target = [1 / (3 * np.pi), 1 / (2 * np.pi), 1 / np.pi]
    assert m.all_zeros(x, y) == pytest.approx(target, abs=1e-4)

    x = np.array([0, 1])
    y = np.array([-1, 1])
    assert all(m.all_zeros(x, y) == np.array([0.5]))

    x = np.array([0, 1])
    y = np.array([1, 1])
    assert len(m.all_zeros(x, y)) == 0


def test_argclosest():
    pass


def test_build_sim_grid():
    pass


def test_indft():
    pass


def test_indft_matrix():
    pass


def test_jn_zeros():
    pass


def test_length():
    pass


def test_ndft():
    pass


def test_ndft_matrix():
    pass


def test_np_cache():
    pass


def test_power_fact():
    pass


def test_sigmoid():
    pass


def test_span():
    pass


def test_tspace():
    pass


def test_u_nm():
    pass


def test_update_frequency_domain():
    pass


def test_wspace():
    pass


def test_differentiate():
    x = np.linspace(-10, 10, 256)
    y = np.exp(-((x / 3) ** 2)) * (1 + 0.2 * np.sin(x * 5))

    y[100] = 1e4
    # true = np.exp(-(x/3)**2) * (x*(-0.4/9 * np.sin(5*x) - 2/9) + np.cos(5*x))
    true = np.exp(-((x / 3) ** 2)) * (
        x**2 * (0.00987654321 * np.sin(5 * x) + 0.0493827)
        - 5.044444 * np.sin(5 * x)
        - 0.44444 * x * np.cos(5 * x)
        - 0.2222222
    )

    import matplotlib.pyplot as plt

    h = x[1] - x[0]

    grad = np.gradient(np.gradient(y)) / h**2
    fine = m.differentiate_arr(y, 2, 6) / h**2

    plt.plot(x, y)
    plt.plot(x, grad, label="gradient")
    plt.plot(x, fine, label="fine")
    plt.plot(x, true, label="ture", ls=":")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    test_differentiate()
