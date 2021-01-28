from json import encoder
import toml
import numpy as np
from datetime import datetime
from scgenerator.io import save_parameters

x = np.arange(5)
y = np.arange(5, dtype="complex")
z = np.arange(5, dtype="float")

dico = dict(a=x, c=list(x), b=print, aa=y, bb=z, ddd=datetime.now())

with open("tests/numpy.toml", "w") as file:
    toml.dump(dico, file, encoder=toml.TomlNumpyEncoder())

save_parameters(dico, "tests/param")
save_parameters(toml.load("tests/test_config.toml"), "tests/test_save_config")