import scgenerator as sc
from pathlib import Path

p = Path("/Users/benoitsierro/Nextcloud/PhD/Supercontinuum/PCF Simulations/PPP")

configs = [
    sc.io.load_config(p / c)
    for c in ("PM1550.toml", "PMHNLF_appended.toml", "PM2000_appended.toml")
]

for variable, params in sc.utils.required_simulations(*configs):
    print(variable)

# sc.initialize.ContinuationParamSequence(configs[-1])
