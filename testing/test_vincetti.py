import matplotlib.pyplot as plt
import numpy as np

import scgenerator as sc

wl = np.linspace(200e-9, 2e-6, 2048)
w = sc.units.m(wl)
wl0 = 800e-9
gas = sc.materials.Gas("argon")
ng2 = gas.sellmeier.n_gas_2(wl, pressure=1e5)

n = sc.fiber.n_eff_vincetti(wl, wl0, ng2, 1e-6, 20e-6, 5e-6, 7)
b2 = sc.fiber.beta2(w, n)

bcap = sc.capillary_dispersion(
    wl, sc.fiber.core_radius_from_capillaries(20e-6, 5e-6, 7), "argon", pressure=1e5
)

plt.plot(wl, b2)
plt.plot(wl, bcap)
plt.show()
