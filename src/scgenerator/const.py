__version__ = "0.1.0"


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
PBAR_POLICY = ENVIRON_KEY_BASE + "PBAR_POLICY"
LOG_POLICY = ENVIRON_KEY_BASE + "LOG_POLICY"
TMP_FOLDER_KEY_BASE = ENVIRON_KEY_BASE + "SC_TMP_"
PREFIX_KEY_BASE = ENVIRON_KEY_BASE + "PREFIX_"
PARAM_SEPARATOR = " "

SPEC1_FN = "spectrum_{}.npy"
SPECN_FN = "spectra_{}.npy"
Z_FN = "z.npy"
PARAM_FN = "params.toml"
