from scgenerator.math import all_zeros
import matplotlib.pyplot as plt
import numpy as np


def main():
    x = np.linspace(-10, 10, 30)
    y = np.sin(x)
    z = all_zeros(x, y)
    plt.plot(x, y)
    plt.plot(z, z * 0, ls="", marker="o")
    plt.show()


if __name__ == "__main__":
    main()
