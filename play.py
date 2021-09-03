import os
import numpy as np
import scgenerator as sc
import matplotlib.pyplot as plt
from pathlib import Path


def main():
    drr = os.getcwd()
    os.chdir("/Users/benoitsierro/Nextcloud/PhD/Supercontinuum/PCF Simulations")
    try:
        sc.run_simulation("PM1550+PM2000D/Pos30000.toml")
    finally:
        os.chdir(drr)


if __name__ == "__main__":
    main()
