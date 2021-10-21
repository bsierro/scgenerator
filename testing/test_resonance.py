import numpy as np
import scgenerator as sc
import matplotlib.pyplot as plt


def main():
    capillary_thickness = 1.4e-6
    wl = np.linspace(200e-9, 2000e-9, 500)
    n_gas_2 = sc.materials.n_gas_2(wl, "air", 3e5, 300, True)
    resonances = []
    for i in range(5):
        t = sc.fiber.resonance_thickness(wl, i, n_gas_2, 40e-6)
        resonances += list(1e9 * sc.math.all_zeros(wl, t - capillary_thickness))
        plt.plot(1e9 * wl, 1e6 * t)
    plt.xlabel("nm")
    plt.ylabel("μm")
    plt.show()


if __name__ == "__main__":
    main()
