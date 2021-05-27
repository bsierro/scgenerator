import unittest
from scgenerator.physics.pulse import conform_pulse_params


class TestPulseMethods(unittest.TestCase):
    def test_conform_pulse_params(self):
        self.assertNotIn(None, conform_pulse_params("gaussian", t0=5, energy=6))
        self.assertNotIn(None, conform_pulse_params("gaussian", width=5, energy=6))
        self.assertNotIn(None, conform_pulse_params("gaussian", t0=5, peak_power=6))
        self.assertNotIn(None, conform_pulse_params("gaussian", width=5, peak_power=6))

        self.assertEqual(4, len(conform_pulse_params("gaussian", t0=5, energy=6)))
        self.assertEqual(4, len(conform_pulse_params("gaussian", width=5, energy=6)))
        self.assertEqual(4, len(conform_pulse_params("gaussian", t0=5, peak_power=6)))
        self.assertEqual(4, len(conform_pulse_params("gaussian", width=5, peak_power=6)))

        with self.assertRaisesRegex(
            TypeError, "when soliton number is desired, both gamma and beta2 must be specified"
        ):
            conform_pulse_params("gaussian", t0=5, energy=6, gamma=0.01)
        with self.assertRaisesRegex(
            TypeError, "when soliton number is desired, both gamma and beta2 must be specified"
        ):
            conform_pulse_params("gaussian", t0=5, energy=6, beta2=0.01)

        self.assertEqual(
            5, len(conform_pulse_params("gaussian", t0=5, energy=6, gamma=0.01, beta2=2e-6))
        )
        self.assertEqual(
            5, len(conform_pulse_params("gaussian", width=5, energy=6, gamma=0.01, beta2=2e-6))
        )
        self.assertEqual(
            5, len(conform_pulse_params("gaussian", t0=5, peak_power=6, gamma=0.01, beta2=2e-6))
        )
        self.assertEqual(
            5, len(conform_pulse_params("gaussian", width=5, peak_power=6, gamma=0.01, beta2=2e-6))
        )


if __name__ == "__main__":
    unittest.main()
