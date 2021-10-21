import numpy as np
import pytest
import scgenerator.math as m
from math import factorial


def test__power_fact_array():
    x = np.random.rand(5)
    for i in range(5):
        assert m._power_fact_array(x, i) == pytest.approx(x ** i / factorial(i))


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
