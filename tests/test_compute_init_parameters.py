from prettyprinter import pprint
from scgenerator import initialize as init
from scgenerator.io import load_toml

debug = 56

config = load_toml("testing/configs/compute_init_parameters/good.toml")
config = init.ensure_consistency(config)
try:
    params = init.compute_init_parameters(config)
except:
    raise
pprint(params)