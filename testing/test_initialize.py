import unittest
import toml
import scgenerator.initialize as init
from scgenerator.errors import *
from prettyprinter import pprint


def load_conf(name):
    with open("testing/configs/" + name + ".toml") as file:
        conf = toml.load(file)
    return conf


def conf_maker(folder):
    def conf(name):
        return load_conf(folder + "/" + name)

    return conf


class TestInitializeMethods(unittest.TestCase):
    def test_validate_types(self):
        conf = lambda s: load_conf("validate_types/" + s)
        with self.assertRaisesRegex(TypeError, "belong"):
            init._validate_types(conf("bad1"))

        with self.assertRaisesRegex(TypeError, "valid list of behaviors"):
            init._validate_types(conf("bad2"))

        with self.assertRaisesRegex(TypeError, "single, real, non-negative number"):
            init._validate_types(conf("bad3"))

        with self.assertRaisesRegex(TypeError, "'parallel' is not a valid variable parameter"):
            init._validate_types(conf("bad4"))

        with self.assertRaisesRegex(TypeError, "Varying parameters should be specified in a list"):
            init._validate_types(conf("bad5"))

        with self.assertRaisesRegex(
            TypeError,
            "value '0' of type <class 'int'> for key 'repeat' is not valid, must be a strictly positive integer",
        ):
            init._validate_types(conf("bad6"))

        with self.assertRaisesRegex(
            ValueError,
            r"Varying parameters lists should contain at least 1 element",
        ):
            init._ensure_consistency(conf("bad7"))

        self.assertIsNone(init._validate_types(conf("good")))

    def test_ensure_consistency(self):
        conf = lambda s: load_conf("ensure_consistency/" + s)
        with self.assertRaisesRegex(
            MissingParameterError,
            r"1 of '\['t0', 'width'\]' is required and no defaults have been set",
        ):
            init._ensure_consistency(conf("bad1"))

        with self.assertRaisesRegex(
            MissingParameterError,
            r"1 of '\['power', 'energy', 'width', 't0'\]' is required when 'soliton_num' is specified and no defaults have been set",
        ):
            init._ensure_consistency(conf("bad2"))

        with self.assertRaisesRegex(
            MissingParameterError,
            r"2 of '\['dt', 't_num', 'time_window'\]' are required and no defaults have been set",
        ):
            init._ensure_consistency(conf("bad3"))

        with self.assertRaisesRegex(
            DuplicateParameterError,
            r"got multiple values for parameter 'width'",
        ):
            init._ensure_consistency(conf("bad4"))

        with self.assertRaisesRegex(
            MissingParameterError,
            r"'capillary_thickness' is a required parameter for fiber model 'hasan' and no defaults have been set",
        ):
            init._ensure_consistency(conf("bad5"))

        with self.assertRaisesRegex(
            MissingParameterError,
            r"1 of '\['capillary_spacing', 'capillary_outer_d'\]' is required for fiber model 'hasan' and no defaults have been set",
        ):
            init._ensure_consistency(conf("bad6"))

        self.assertLessEqual(
            {"model": "pcf"}.items(), init._ensure_consistency(conf("good1"))["fiber"].items()
        )

        self.assertNotIn("gas", init._ensure_consistency(conf("good1")))

        self.assertNotIn("gamma", init._ensure_consistency(conf("good4"))["fiber"])

        self.assertLessEqual(
            {"raman_type": "agrawal"}.items(),
            init._ensure_consistency(conf("good2"))["simulation"].items(),
        )

        self.assertLessEqual(
            {"name": "no name"}.items(), init._ensure_consistency(conf("good3")).items()
        )

        self.assertLessEqual(
            {"capillary_nested": 0, "capillary_resonance_strengths": []}.items(),
            init._ensure_consistency(conf("good4"))["fiber"].items(),
        )

        self.assertLessEqual(
            dict(he_mode=(1, 1)).items(),
            init._ensure_consistency(conf("good5"))["fiber"].items(),
        )

        self.assertLessEqual(
            dict(temperature=300, pressure=1e5, gas_name="vacuum", plasma_density=0).items(),
            init._ensure_consistency(conf("good5"))["gas"].items(),
        )

        self.assertLessEqual(
            dict(
                t_num=16384,
                time_window=37e-12,
                lower_wavelength_interp_limit=0,
                upper_wavelength_interp_limit=1900e-9,
            ).items(),
            init._ensure_consistency(conf("good6"))["simulation"].items(),
        )

    # def test_compute_init_parameters(self):
    #     conf = lambda s: load_conf("compute_init_parameters/" + s)


if __name__ == "__main__":
    conf = conf_maker("validate_types")

    unittest.main()
