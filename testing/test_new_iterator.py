import scgenerator as sc
from pathlib import Path
import os

os.chdir("/Users/benoitsierro/Nextcloud/PhD/Supercontinuum/PCF Simulations/")

root = Path("PM1550+PMHNLF+PM1550+PM2000")

confs = sc.io.load_config_sequence(root / "4_PM2000.toml")
final = sc.utils.final_config_from_sequence(*confs)

print(final)
