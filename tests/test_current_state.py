import numpy as np
import pytest

from scgenerator.operators import SimulationState


def test_creation():
    x = np.linspace(0, 1, 128, dtype=complex)
    cs = SimulationState(1.0, 0, 0.1, x, 1.0)

    assert cs.converter is np.fft.ifft
    assert cs.stats == {}
    assert np.allclose(cs.field2, np.abs(np.fft.ifft(x)) ** 2)

    with pytest.raises(ValueError):
        cs = SimulationState(1.0, 0, 0.0, x, 1.0, spectrum2=np.abs(x) ** 3)

    cs = SimulationState(1.0, 0, 0.1, x, 1.0, spectrum2=x.copy(), field=x.copy(), field2=x.copy())

    assert np.allclose(cs.spectrum2, cs.spectrum)
    assert np.allclose(cs.spectrum, cs.field)
    assert np.allclose(cs.field, cs.field2)


def test_copy():
    x = np.linspace(0, 1, 128, dtype=complex)
    start = SimulationState(1.0, 0, 0.1, x, 1.0)
    end = start.copy()

    assert start.spectrum is not end.spectrum
    assert np.all(start.field2 == end.field2)
