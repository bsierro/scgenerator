__version__ = "0.1.0"


from typing import Any


def pbar_format(worker_id: int):
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


ENVIRON_KEY_BASE = "SCGENERATOR_"
TMP_FOLDER_KEY_BASE = ENVIRON_KEY_BASE + "SC_TMP_"
PREFIX_KEY_BASE = ENVIRON_KEY_BASE + "PREFIX_"
PARAM_SEPARATOR = " "

PBAR_POLICY = ENVIRON_KEY_BASE + "PBAR_POLICY"
LOG_FILE_LEVEL = ENVIRON_KEY_BASE + "LOG_FILE_LEVEL"
LOG_PRINT_LEVEL = ENVIRON_KEY_BASE + "LOG_PRINT_LEVEL"
START_RAY = ENVIRON_KEY_BASE + "START_RAY"
NO_RAY = ENVIRON_KEY_BASE + "NO_RAY"
OUTPUT_PATH = ENVIRON_KEY_BASE + "OUTPUT_PATH"


global_config: dict[str, dict[str, Any]] = {
    LOG_FILE_LEVEL: dict(
        help="minimum lvl of message to be saved in the log file",
        choices=["critical", "error", "warning", "info", "debug"],
        default=None,
        type=str,
    ),
    LOG_PRINT_LEVEL: dict(
        help="minimum lvl of message to be printed to the standard output",
        choices=["critical", "error", "warning", "info", "debug"],
        default="error",
        type=str,
    ),
    PBAR_POLICY: dict(
        help="what to do with progress pars (print them, make them a txt file or nothing), default is print",
        choices=["print", "file", "both", "none"],
        default=None,
        type=str,
    ),
    START_RAY: dict(action="store_true", help="initialize ray (ray must be installed)", type=bool),
    NO_RAY: dict(action="store_true", help="force not to use ray", type=bool),
    OUTPUT_PATH: dict(
        short_name="-o", help="path to the final output folder", default=None, type=str
    ),
}


SPEC1_FN = "spectrum_{}.npy"
SPECN_FN = "spectra_{}.npy"
Z_FN = "z.npy"
PARAM_FN = "params.toml"
