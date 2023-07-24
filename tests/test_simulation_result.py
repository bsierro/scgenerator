from pathlib import Path

import numpy as np

from scgenerator.solver import SimulationResult


def test_load_save(tmp_path: Path):
    sim = SimulationResult(
        np.random.randint(0, 20, (5, 5)), dict(a=[], b=[1, 2, 3], z=list(range(32)))
    )
    sim.save(tmp_path / "mysim")
    sim2 = SimulationResult.load(tmp_path / "mysim.zip")
    assert np.all(sim2.spectra == sim.spectra)
    assert np.all(sim2.z == sim.z)
    for k, v in sim.stats.items():
        assert sim2.stats[k] == v
