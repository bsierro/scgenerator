import os
import numpy as np
import scgenerator as sc
import matplotlib.pyplot as plt
from pathlib import Path
from pprint import pprint


def _main():
    print(os.getcwd())
    for v_list, params in sc.Configuration("PM1550+PM2000D+PM1550/Pos30000.toml"):
        print(params.fiber_map)


def main():
    drr = os.getcwd()
    os.chdir("/Users/benoitsierro/Nextcloud/PhD/Supercontinuum/PCF Simulations")
    try:
        _main()
    finally:
        os.chdir(drr)


if __name__ == "__main__":
    main()
