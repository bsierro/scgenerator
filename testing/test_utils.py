import unittest

import numpy as np
import toml
from scgenerator import initialize, utils


def load_conf(name):
    with open("testing/configs/" + name + ".toml") as file:
        conf = toml.load(file)
    return conf


def conf_maker(folder, val=True):
    def conf(name):
        if val:
            return initialize.validate(load_conf(folder + "/" + name))
        else:
            return load_conf(folder + "/" + name)

    return conf


class TestUtilsMethods(unittest.TestCase):
    def test_count_variations(self):
        conf = conf_maker("count_variations")

        for sim, vary in [(1, 0), (1, 1), (2, 1), (2, 0), (120, 3)]:
            self.assertEqual((sim, vary), utils.count_variations(conf(f"{sim}sim_{vary}vary")))

    def test_format_value(self):
        values = [
            122e-6,
            True,
            ["raman", "ss"],
            np.arange(5),
            1.123,
            1.1230001,
            0.002e122,
            12.3456e-9,
        ]
        s = [
            "0.000122",
            "True",
            "raman-ss",
            "0-1-2-3-4",
            "1.123",
            "1.1230001",
            "2e+119",
            "1.23456e-08",
        ]

        for value, target in zip(values, s):
            self.assertEqual(target, utils.format_value(value))

    def test_override_config(self):
        conf = conf_maker("override", False)
        old = conf("initial_config")
        new = conf("fiber2")

        over = utils.override_config(old, new)
        self.assertIn("input_transmission", over["fiber"]["variable"])
        self.assertNotIn("input_transmission", over["fiber"])


if __name__ == "__main__":
    unittest.main()
