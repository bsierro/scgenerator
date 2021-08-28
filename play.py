from dataclasses import fields
from scgenerator import Parameters
from scgenerator.physics.simulate import RK4IP
import os
import matplotlib.pyplot as plt

from pprint import pprint


def main():
    cwd = os.getcwd()
    try:
        os.chdir("/Users/benoitsierro/Nextcloud/PhD/Supercontinuum/PCF Simulations")

        pa = Parameters.load(
            "/Users/benoitsierro/Nextcloud/PhD/Supercontinuum/PCF Simulations/PM1550+PM2000D/PM2000D.toml"
        )
        x = 1, 2
        print(pa.input_transmission)
        print(x)
    finally:
        os.chdir(cwd)


if __name__ == "__main__":
    main()
