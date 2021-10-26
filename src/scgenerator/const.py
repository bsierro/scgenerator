__version__ = "0.2.4"


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
SPECN_FN1 = "spectra_{}.npy"
SPEC1_FN_N = "spectrum_{}_{}.npy"
Z_FN = "z.npy"
PARAM_FN = "params.toml"
PARAM_SEPARATOR = " "
