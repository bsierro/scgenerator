import matplotlib.pyplot as plt
import numpy as np

import scgenerator as sc


def field(t: np.ndarray, fwhm=10e-15) -> np.ndarray:
    t0 = sc.pulse.width_to_t0(fwhm, "gaussian")
    return sc.pulse.initial_full_field(t, "gaussian", 1e-4, t0, 2e14, sc.units.nm(800), 1)


rate = sc.plasma.create_ion_rate_func(sc.materials.Gas("argon").ionization_energy)

fig, (top, mid, bot) = plt.subplots(3, 1)
t = np.linspace(-10e-15, 10e-15, 1024)
E = field(t)
top.plot(t * 1e15, field(t))
mid.plot(t * 1e15, rate(np.abs(E)))
plt.show()
