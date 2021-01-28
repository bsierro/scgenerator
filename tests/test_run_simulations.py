from scgenerator.physics.simulate import new_simulations
from scgenerator import io
import ray

ray.init()

sim = new_simulations("testing/configs/run_simulations/full_anomalous.toml", 123)
sim.run()