from scgenerator import io
from scgenerator.io import _get_data_subfolders
import numpy as np
from glob import glob
from scgenerator.math import abs2
from matplotlib import pyplot as plt

path = "scgenerator_full anomalous123/wavelength_8.35e-07"


for i in [0, 63]:
    dat = np.load(f"{path}/spectra_{i}.npy")
    for d in dat:
        plt.plot(abs2(d))
        plt.show()