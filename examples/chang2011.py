from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
from tqdm import tqdm

import scgenerator as sc
from scgenerator import solver as so

PARAMS = dict(
    wavelength=800e-9,
    width=30e-15,
    energy=2.5e-6,
    core_radius=10e-6,
    length=10e-2,
    gas_name="argon",
    pressure=4e5,
    t_num=4096,
    dt=0.1e-15,
    photoionization=True,
    full_field=True,
    model="marcatili",
)


params = sc.Parameters(**PARAMS)
init_state = so.SimulationState(params.spec_0, params.length, 5e-6, converter=params.ifft)
stepper = so.ERKIP43Stepper(params.linear_operator, params.nonlinear_operator)
solution = []
stats = defaultdict(list)
for state in so.integrate(stepper, init_state, step_judge=so.adaptive_judge(1e-6, 4)):
    solution.append(state.spectrum2)
    for k, v in state.stats.items():
        stats[k].append(v)
    if state.z > params.length:
        break
quit()
interp = interp1d(stats["z"], solution, axis=0)
z = np.linspace(0, params.length, 128)
plt.imshow(
    sc.units.to_log(interp(z)),
    vmin=-50,
    extent=sc.get_extent(sc.units.THz_inv(params.w), z),
    origin="lower",
    aspect="auto",
)
plt.figure()
plt.plot(stats["z"][1:], stats["electron_density"])
plt.show()
