import importlib.metadata
from typing import Any

__version__ = importlib.metadata.version("scgenerator")


def pbar_format(worker_id: int) -> dict[str, Any]:
    if worker_id == 0:
        return dict(
            position=0,
            bar_format="{l_bar}{bar}" "|[{elapsed}<{remaining}, " "{rate_fmt}{postfix}]",
        )
    else:
        return dict(
            total=1,
            desc=f"Worker {worker_id}",
            position=worker_id,
            bar_format="{l_bar}{bar}" "|[{elapsed}<{remaining}, " "{rate_fmt}{postfix}]",
        )


SPEC1_FN = "spectrum_{}.npy"
SPECN_FN1 = "spectra_{}.npy"
SPEC1_FN_N = "spectrum_{}_{}.npy"
Z_FN = "z.npy"
PARAM_FN = "params.toml"
PARAM_SEPARATOR = " "

VALID_VARIABLE = {
    "dispersion_file",
    "prev_data_dir",
    "field_file",
    "loss_file",
    "A_eff_file",
    "beta2_coefficients",
    "gamma",
    "pitch",
    "pitch_ratio",
    "effective_mode_diameter",
    "core_radius",
    "model",
    "capillary_num",
    "capillary_radius",
    "capillary_thickness",
    "capillary_spacing",
    "capillary_resonance_strengths",
    "capillary_resonance_max_order",
    "capillary_nested",
    "he_mode",
    "fit_parameters",
    "input_transmission",
    "n2",
    "pressure",
    "temperature",
    "gas_name",
    "plasma_density",
    "peak_power",
    "mean_power",
    "peak_power",
    "energy",
    "quantum_noise",
    "shape",
    "wavelength",
    "intensity_noise",
    "width",
    "t0",
    "soliton_num",
    "raman_type",
    "tolerated_error",
    "photoionization",
    "step_size",
    "interpolation_degree",
    "ideal_gas",
    "length",
    "integration_scheme",
    "num",
}

INVALID_VARIABLE = {"repeat"}


MANDATORY_PARAMETERS = [
    "name",
    "w",
    "w0",
    "spec_0",
    "field_0",
    "mean_power",
    "input_transmission",
    "z_targets",
    "length",
    "adapt_step_size",
    "tolerated_error",
    "recovery_last_stored",
    "output_path",
    "repeat",
    "linear_operator",
    "nonlinear_operator",
]

ROOT_PARAMETERS = [
    "repeat",
    "num",
    "dt",
    "t_num",
    "time_window",
    "step_size",
    "tolerated_error",
    "width",
    "shape",
]
