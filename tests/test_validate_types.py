from scgenerator.initialize import validate_types
from scgenerator.io import load_toml

config = load_toml("tests/test_config.toml")

validate_types(config)