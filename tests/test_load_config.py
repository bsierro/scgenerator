import toml
from prettyprinter import pprint
import json

config = toml.load("tests/test_config.toml")
pprint(config)
with open("tests/test_config.toml") as file:
    config = toml.load(file)

# with open("tests/test_config.json", "w") as file:
#     json.dump(config, file)