import unittest
from scgenerator import utils, initialize
import toml


def load_conf(name):
    with open("testing/configs/" + name + ".toml") as file:
        conf = toml.load(file)
    return conf


def conf_maker(folder):
    def conf(name):
        return initialize.validate(load_conf(folder + "/" + name))

    return conf


class TestUtilsMethods(unittest.TestCase):
    def test_count_variations(self):
        conf = conf_maker("count_variations")

        self.assertEqual((1, 0), utils.count_variations(conf("1sim_0vary")))
        self.assertEqual((1, 1), utils.count_variations(conf("1sim_1vary")))
        self.assertEqual((2, 1), utils.count_variations(conf("2sim_1vary")))
        self.assertEqual((2, 0), utils.count_variations(conf("2sim_0vary")))


if __name__ == "__main__":
    unittest.main()