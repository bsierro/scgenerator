import unittest
from copy import deepcopy

import scgenerator.initialize as init
import numpy as np
import toml
from scgenerator import defaults, utils, math
from scgenerator.errors import *
from scgenerator.physics import pulse, units
from scgenerator.utils.parameter import Config, Parameters


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

    def test_no_variations_yields_only_num_and_id(self):
        for param_seq in self.iterconf(["no_variations"]):
            for vary_list, _ in utils.required_simulations(param_seq.config):
                self.assertEqual(vary_list[1][0], "num")
                self.assertEqual(vary_list[0][0], "id")
                self.assertEqual(2, len(vary_list))


class TestInitializeMethods(unittest.TestCase):
    def test_validate_types(self):
        conf = lambda s: load_conf("validate_types/" + s)

        with self.assertRaisesRegex(ValueError, r"'behaviors\[3\]' must be a str in"):
            init.Config(**conf("bad2"))

        with self.assertRaisesRegex(TypeError, "value must be of type <class 'float'>"):
            init.Config(**conf("bad3"))

        with self.assertRaisesRegex(TypeError, "'parallel' is not a valid variable parameter"):
            init.Config(**conf("bad4"))

        with self.assertRaisesRegex(
            TypeError, "'variable intensity_noise' value must be of type <class 'list'>"
        ):
            init.Config(**conf("bad5"))

        with self.assertRaisesRegex(ValueError, "'repeat' must be positive"):
            init.Config(**conf("bad6"))

        with self.assertRaisesRegex(
            ValueError, "variable parameter 'intensity_noise' must not be empty"
        ):
            init.Config(**conf("bad7"))

        self.assertIsNone(init.Config(**conf("good")).hr_w)

    def test_ensure_consistency(self):
        conf = lambda s: load_conf("ensure_consistency/" + s)
        with self.assertRaisesRegex(
            MissingParameterError,
            r"1 of '\['t0', 'width'\]' is required and no defaults have been set",
        ):
            init.Config(**conf("bad1"))

        with self.assertRaisesRegex(
            MissingParameterError,
            r"1 of '\['peak_power', 'mean_power', 'energy', 'width', 't0'\]' is required when 'soliton_num' is specified and no defaults have been set",
        ):
            init.Config(**conf("bad2"))

        with self.assertRaisesRegex(
            MissingParameterError,
            r"2 of '\['dt', 't_num', 'time_window'\]' are required and no defaults have been set",
        ):
            init.Config(**conf("bad3"))

        with self.assertRaisesRegex(
            DuplicateParameterError,
            r"got multiple values for parameter 'width'",
        ):
            init.Config(**conf("bad4"))

        with self.assertRaisesRegex(
            MissingParameterError,
            r"'capillary_thickness' is a required parameter for fiber model 'hasan' and no defaults have been set",
        ):
            init.Config(**conf("bad5"))

        with self.assertRaisesRegex(
            MissingParameterError,
            r"1 of '\['capillary_spacing', 'capillary_outer_d'\]' is required for fiber model 'hasan' and no defaults have been set",
        ):
            init.Config(**conf("bad6"))

        self.assertLessEqual(
            {"model": "pcf"}.items(), init.Config(**conf("good1")).__dict__.items()
        )

        self.assertIsNone(init.Config(**conf("good4")).gamma)

        self.assertLessEqual(
            {"raman_type": "agrawal"}.items(),
            init.Config(**conf("good2")).__dict__.items(),
        )

        self.assertLessEqual(
            {"name": "no name"}.items(), init.Config(**conf("good3")).__dict__.items()
        )

        self.assertLessEqual(
            {"capillary_nested": 0, "capillary_resonance_strengths": []}.items(),
            init.Config(**conf("good4")).__dict__.items(),
        )

        self.assertLessEqual(
            dict(he_mode=(1, 1)).items(),
            init.Config(**conf("good5")).__dict__.items(),
        )

        self.assertLessEqual(
            dict(temperature=300, pressure=1e5, gas_name="vacuum", plasma_density=0).items(),
            init.Config(**conf("good5")).__dict__.items(),
        )

    def setup_conf_custom_field(self, path) -> Parameters:

        conf = load_conf(path)
        conf = Parameters(**conf)
        init.build_sim_grid_in_place(conf)
        return conf

    def test_setup_custom_field(self):
        d = np.load("testing/configs/custom_field/init_field.npz")
        t = d["time"]
        field = d["field"]
        conf = self.setup_conf_custom_field("custom_field/no_change")
        result, conf.width, conf.peak_power, conf.energy, conf.field_0 = pulse.setup_custom_field(
            conf
        )
        self.assertAlmostEqual(conf.field_0.real.max(), field.real.max(), 4)
        self.assertTrue(result)

        conf = self.setup_conf_custom_field("custom_field/peak_power")
        result, conf.width, conf.peak_power, conf.energy, conf.field_0 = pulse.setup_custom_field(
            conf
        )
        conf.wavelength = pulse.correct_wavelength(conf.wavelength, conf.w_c, conf.field_0)
        self.assertAlmostEqual(math.abs2(conf.field_0).max(), 20000, 4)
        self.assertTrue(result)
        self.assertNotAlmostEqual(conf.wavelength, 1593e-9)

        conf = self.setup_conf_custom_field("custom_field/mean_power")
        result, conf.width, conf.peak_power, conf.energy, conf.field_0 = pulse.setup_custom_field(
            conf
        )
        self.assertAlmostEqual(np.trapz(math.abs2(conf.field_0), conf.t), 0.22 / 40e6, 4)
        self.assertTrue(result)

        conf = self.setup_conf_custom_field("custom_field/recover1")
        result, conf.width, conf.peak_power, conf.energy, conf.field_0 = pulse.setup_custom_field(
            conf
        )
        self.assertAlmostEqual(math.abs2(conf.field_0 - field).sum(), 0)
        self.assertTrue(result)

        conf = self.setup_conf_custom_field("custom_field/recover2")
        result, conf.width, conf.peak_power, conf.energy, conf.field_0 = pulse.setup_custom_field(
            conf
        )
        self.assertAlmostEqual((math.abs2(conf.field_0) / 0.9 - math.abs2(field)).sum(), 0)
        self.assertTrue(result)

        conf = self.setup_conf_custom_field("custom_field/wavelength_shift1")
        result = Parameters(**conf)
        self.assertAlmostEqual(units.m.inv(result.w)[np.argmax(math.abs2(result.spec_0))], 1050e-9)

        conf = self.setup_conf_custom_field("custom_field/wavelength_shift1")
        conf.wavelength = 1593e-9
        result = Parameters(**conf)

        conf = load_conf("custom_field/wavelength_shift2")
        conf = init.Config(**conf)
        for target, (variable, config) in zip(
            [1050e-9, 1321e-9, 1593e-9], init.ParamSequence(conf)
        ):
            init.build_sim_grid_in_place(conf)
            self.assertAlmostEqual(
                units.m.inv(config.w)[np.argmax(math.abs2(config.spec_0))], target
            )
            print(config.wavelength, target)


if __name__ == "__main__":
    conf = conf_maker("validate_types")

    unittest.main()
