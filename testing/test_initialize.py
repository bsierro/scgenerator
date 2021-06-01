import unittest
from copy import deepcopy

import scgenerator.initialize as init
import numpy as np
import toml
from scgenerator import defaults, utils, math
from scgenerator.errors import *


def load_conf(name):
    with open("testing/configs/" + name + ".toml") as file:
        conf = toml.load(file)
    return conf


def conf_maker(folder):
    def conf(name):
        return load_conf(folder + "/" + name)

    return conf


class TestParamSequence(unittest.TestCase):
    def iterconf(self, files):
        conf = conf_maker("param_sequence")
        for path in files:
            yield init.ParamSequence(conf(path))

    def test_no_repeat_in_sub_folder_names(self):
        for param_seq in self.iterconf(["almost_equal", "equal", "no_variations"]):
            l = []
            s = []
            for vary_list, _ in utils.required_simulations(param_seq.config):
                self.assertNotIn(vary_list, l)
                self.assertNotIn(utils.format_variable_list(vary_list), s)
                l.append(vary_list)
                s.append(utils.format_variable_list(vary_list))

    def test_init_config_not_affected_by_iteration(self):
        for param_seq in self.iterconf(["almost_equal", "equal", "no_variations"]):
            config = deepcopy(param_seq.config)
            for _ in utils.required_simulations(param_seq.config):
                self.assertEqual(config.items(), param_seq.config.items())

    def test_no_variations_yields_only_num_and_id(self):
        for param_seq in self.iterconf(["no_variations"]):
            for vary_list, _ in utils.required_simulations(param_seq.config):
                self.assertEqual(vary_list[1][0], "num")
                self.assertEqual(vary_list[0][0], "id")
                self.assertEqual(2, len(vary_list))


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

        with self.assertRaisesRegex(TypeError, "Variable parameters should be specified in a list"):
            init._validate_types(conf("bad5"))

        with self.assertRaisesRegex(
            TypeError,
            "value '0' of type .*int.* for key 'repeat' is not valid, must be a strictly positive integer",
        ):
            init._validate_types(conf("bad6"))

        with self.assertRaisesRegex(
            ValueError,
            r"Variable parameters lists should contain at least 1 element",
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
            r"1 of '\['peak_power', 'mean_power', 'energy', 'width', 't0'\]' is required when 'soliton_num' is specified and no defaults have been set",
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
                lower_wavelength_interp_limit=defaults.default_parameters[
                    "lower_wavelength_interp_limit"
                ],
                upper_wavelength_interp_limit=defaults.default_parameters[
                    "upper_wavelength_interp_limit"
                ],
            ).items(),
            init._ensure_consistency(conf("good6"))["simulation"].items(),
        )

    def test_setup_custom_field(self):
        d = np.load("testing/configs/custom_field/init_field.npz")
        t = d["time"]
        field = d["field"]
        conf = load_conf("custom_field/no_change")
        conf = init._generate_sim_grid(conf)
        result = init.setup_custom_field(conf)
        self.assertAlmostEqual(conf["field_0"].real.max(), field.real.max(), 4)
        self.assertTrue(result)

        conf = load_conf("custom_field/peak_power")
        conf = init._generate_sim_grid(conf)
        result = init.setup_custom_field(conf)
        self.assertAlmostEqual(math.abs2(conf["field_0"]).max(), 20000, 4)
        self.assertTrue(result)

        conf = load_conf("custom_field/mean_power")
        conf = init._generate_sim_grid(conf)
        result = init.setup_custom_field(conf)
        self.assertAlmostEqual(np.trapz(math.abs2(conf["field_0"]), conf["t"]), 0.22 / 40e6, 4)
        self.assertTrue(result)

        conf = load_conf("custom_field/recover1")
        conf = init._generate_sim_grid(conf)
        result = init.setup_custom_field(conf)
        self.assertAlmostEqual(math.abs2(conf["field_0"] - field).sum(), 0)
        self.assertTrue(result)

        conf = load_conf("custom_field/recover2")
        conf = init._generate_sim_grid(conf)
        result = init.setup_custom_field(conf)
        self.assertAlmostEqual((math.abs2(conf["field_0"]) / 0.9 - math.abs2(field)).sum(), 0)
        self.assertTrue(result)


if __name__ == "__main__":
    conf = conf_maker("validate_types")

    unittest.main()
