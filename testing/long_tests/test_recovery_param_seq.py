import shutil
import unittest

import toml
from scgenerator import initialize, io, logger
from send2trash import send2trash

TMP = "testing/.tmp"


class TestRecoveryParamSequence(unittest.TestCase):
    def setUp(self):
        shutil.copytree("/Users/benoitsierro/sc_tests/scgenerator_full anomalous55", TMP)
        self.conf = toml.load(TMP + "/initial_config.toml")
        logger.DEFAULT_LEVEL = logger.logging.FATAL
        io.set_data_folder(55, TMP)

    def test_remaining_simulations_count(self):
        param_seq = initialize.RecoveryParamSequence(self.conf, 55)
        self.assertEqual(5, len(param_seq))

    def test_only_one_to_complete(self):
        param_seq = initialize.RecoveryParamSequence(self.conf, 55)
        i = 0
        for expected, (vary_list, params) in zip([True, False, False, False, False], param_seq):
            i += 1
            self.assertEqual(expected, "recovery_last_stored" in params)

        self.assertEqual(5, i)

    def tearDown(self):
        send2trash(TMP)


if __name__ == "__main__":
    unittest.main()
