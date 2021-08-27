from scgenerator import Parameters
from scgenerator.physics.simulate import RK4IP
import os
import matplotlib.pyplot as plt


def main():
    cwd = os.getcwd()
    try:
        os.chdir("/Users/benoitsierro/Nextcloud/PhD/Supercontinuum/PCF Simulations")

        pa = Parameters.load(
            "/Users/benoitsierro/Nextcloud/PhD/Supercontinuum/PCF Simulations/PM1550+PM2000D/PM1550_RIN.toml"
        )

        plt.plot(pa.t, pa.field_0.imag)
        plt.plot(pa.t, pa.field_0.real)
        plt.show()
    finally:
        os.chdir(cwd)


if __name__ == "__main__":
    main()
