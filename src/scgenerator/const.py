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


SPEC1_FN = "spectrum_{}.npy"
SPECN_FN = "spectra_{}.npy"
Z_FN = "z.npy"
PARAM_FN = "params.toml"
PARAM_SEPARATOR = " "
