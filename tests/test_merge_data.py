from scgenerator.io import merge_same_simulations
from prettyprinter import pprint

# a, b = [
#     "scgenerator_full anomalous123_1/wavelength_8.35e-07_num_3",
#     "scgenerator_full anomalous123_1/wavelength_8.35e-07_num_2",
#     "scgenerator_full anomalous123_1/wavelength_8.3e-07_num_1",
#     "scgenerator_full anomalous123_1/wavelength_8.3e-07_num_0",
#     "scgenerator_full anomalous123_1/wavelength_8.35e-07_num_0",
#     "scgenerator_full anomalous123_1/wavelength_8.35e-07_num_1",
#     "scgenerator_full anomalous123_1/wavelength_8.3e-07_num_2",
#     "scgenerator_full anomalous123_1/wavelength_8.3e-07_num_3",
# ], [
#     [("wavelength", 8.3e-07), ("num", 0)],
#     [("wavelength", 8.35e-07), ("num", 0)],
#     [("wavelength", 8.3e-07), ("num", 1)],
#     [("wavelength", 8.35e-07), ("num", 1)],
#     [("wavelength", 8.3e-07), ("num", 2)],
#     [("wavelength", 8.35e-07), ("num", 2)],
#     [("wavelength", 8.3e-07), ("num", 3)],
#     [("wavelength", 8.35e-07), ("num", 3)],
# ]

# pprint(list(zip(a, b)))


all = merge_same_simulations("scgenerator_full anomalous123")

pprint(all)
