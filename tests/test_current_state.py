import numpy as np
import pytest

from scgenerator.operators import CurrentState


def test_creation():
    x = (np.linspace(0, 1, 128, dtype=complex),)
    cs = CurrentState(1.0, 0, 0.1, x, 1.0)

    assert cs.converter is np.fft.ifft
    assert cs.stats == {}
    assert np.allclose(cs.spectrum2, np.abs(np.fft.ifft(x)) ** 2)

    with pytest.raises(ValueError):
        cs = CurrentState(1.0, 0, 0.0, x, 1.0, spectrum2=np.abs(x) ** 3)

    cs = CurrentState(1.0, 0, 0.1, x, 1.0, spectrum2=x.copy(), field=x.copy(), field2=x.copy())

    assert np.allclose(cs.spectrum2, cs.spectrum)
    assert np.allclose(cs.spectrum, cs.field)
    assert np.allclose(cs.field, cs.field2)


def test_copy():
    x = (np.linspace(0, 1, 128, dtype=complex),)
    cs = CurrentState(1.0, 0, 0.1, x, 1.0)
    cs2 = cs.copy()

    assert cs.spectrum is not cs2.spectrum
    assert np.all(cs.field2 == cs2.field2)
