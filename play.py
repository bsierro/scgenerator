from pathlib import Path
from pprint import pprint
import scgenerator as sc
import os

cwd = os.getcwd()
os.chdir("/Users/benoitsierro/Nextcloud/PhD/Supercontinuum/PCF Simulations/")
conf = sc.Configuration(sc.load_toml("PM1550+PM2000D/RIN_PM2000D_appended.toml"))


pprint(conf.data_dirs)
print(conf.total_num_steps)
os.chdir(cwd)
