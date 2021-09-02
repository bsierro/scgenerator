import itertools
import os
from itertools import cycle
from pathlib import Path
from typing import Any, Iterable, Optional

import matplotlib.pyplot as plt
import numpy as np
from cycler import cycler
from tqdm import tqdm

from .. import env, math
from ..const import PARAM_FN, PARAM_SEPARATOR
from ..physics import fiber, units
from ..plotting import plot_setup
from ..spectra import Pulse
from ..utils import auto_crop, load_toml, save_toml, translate_parameters
from ..utils.parameter import (
    Configuration,
    Parameters,
    pretty_format_from_sim_name,
    pretty_format_value,
)


def fingerprint(params: Parameters):
    h1 = hash(params.field_0.tobytes())
    h2 = tuple(params.beta2_coefficients)
    return h1, h2


def plot_all(sim_dir: Path, limits: list[str], show=False, **opts):
    for k, v in opts.items():
        if k in ["skip"]:
            opts[k] = int(v)
    dir_list = list(p for p in sim_dir.glob("*") if p.is_dir())
    if len(dir_list) == 0:
        dir_list = [sim_dir]
    limits = [
        tuple(func(el) for func, el in zip([float, float, str], lim.split(","))) for lim in limits
    ]
    with tqdm(total=len(dir_list) * len(limits)) as bar:
        for p in dir_list:
            pulse = Pulse(p)
            for left, right, unit in limits:
                path, fig, ax = plot_setup(
                    pulse.path.parent
                    / (
                        pretty_format_from_sim_name(pulse.path.name)
                        + PARAM_SEPARATOR
                        + f"{left:.1f}{PARAM_SEPARATOR}{right:.1f}{PARAM_SEPARATOR}{unit}"
                    )
                )
                fig.suptitle(p.name)
                pulse.plot_2D(
                    left,
                    right,
                    unit,
                    ax,
                    **opts,
                )
                bar.update()
                if show:
                    plt.show()
                else:
                    fig.savefig(path, bbox_inches="tight")
                plt.close(fig)


def plot_init_field_spec(
    config_path: Path,
    lim_t: tuple[float, float] = None,
    lim_l: tuple[float, float] = None,
):
    fig, (left, right) = plt.subplots(1, 2, figsize=(12, 7))
    all_labels = []
    already_plotted = set()
    for style, lbl, params in plot_helper(config_path):
        if (bbb := hash(params.field_0.tobytes())) not in already_plotted:
            already_plotted.add(bbb)
        else:
            continue

        lbl = plot_1_init_spec_field(lim_t, lim_l, left, right, style, lbl, params)
        all_labels.append(lbl)
    finish_plot(fig, left, right, all_labels, params)


def plot_dispersion(config_path: Path, lim: tuple[float, float] = None):
    fig, (left, right) = plt.subplots(1, 2, figsize=(12, 7))
    left.grid()
    right.grid()
    all_labels = []
    already_plotted = set()
    loss_ax = None
    plt.sca(left)
    for style, lbl, params in plot_helper(config_path):
        if params.alpha_arr is not None and loss_ax is None:
            loss_ax = right.twinx()
        if (bbb := tuple(params.beta2_coefficients)) not in already_plotted:
            already_plotted.add(bbb)
        else:
            continue

        lbl = plot_1_dispersion(lim, left, right, style, lbl, params, loss_ax)
        all_labels.append(lbl)
    finish_plot(fig, right, all_labels, params)


def plot_init(
    config_path: Path,
    lim_field: tuple[float, float] = None,
    lim_spec: tuple[float, float] = None,
    lim_disp: tuple[float, float] = None,
):
    fig, ((tl, tr), (bl, br)) = plt.subplots(2, 2, figsize=(14, 10))
    loss_ax = None
    tl.grid()
    tr.grid()
    all_labels = []
    already_plotted = set()
    for style, lbl, params in plot_helper(config_path):
        if params.alpha_arr is not None and loss_ax is None:
            loss_ax = tr.twinx()
        if (fp := fingerprint(params)) not in already_plotted:
            already_plotted.add(fp)
        else:
            continue
        lbl = plot_1_dispersion(lim_disp, tl, tr, style, lbl, params, loss_ax)
        lbl = plot_1_init_spec_field(lim_field, lim_spec, bl, br, style, lbl, params)
        all_labels.append(lbl)
    finish_plot(fig, tr, all_labels, params)


def plot_1_init_spec_field(lim_t, lim_l, left, right, style, lbl, params):
    field = math.abs2(params.field_0)
    spec = math.abs2(params.spec_0)
    t = units.To.fs(params.t)
    wl = units.To.nm(params.w)

    lbl.append(f"max at {wl[spec.argmax()]:.1f} nm")

    mt = np.ones_like(t, dtype=bool)
    if lim_t is not None:
        mt &= t >= lim_t[0]
        mt &= t <= lim_t[1]
    else:
        mt = auto_crop(t, field)
    ml = np.ones_like(wl, dtype=bool)
    if lim_l is not None:
        ml &= wl >= lim_l[0]
        ml &= wl <= lim_l[1]
    else:
        ml = auto_crop(wl, spec)

    left.plot(t[mt], field[mt])
    right.plot(wl[ml], spec[ml], label=" ", **style)
    return lbl


def plot_1_dispersion(
    lim: Optional[tuple[float, float]],
    left: plt.Axes,
    right: plt.Axes,
    style: dict[str, Any],
    lbl: list[str],
    params: Parameters,
    loss: plt.Axes = None,
):
    beta_arr = fiber.dispersion_from_coefficients(params.w_c, params.beta2_coefficients)
    wl = units.m.inv(params.w)
    D = fiber.beta2_to_D(beta_arr, wl) * 1e6

    zdw = math.all_zeros(wl, beta_arr)
    if len(zdw) > 0:
        zdw = zdw[np.argmin(abs(zdw - params.wavelength))]
        lbl.append(f"ZDW at {zdw:.1f}nm")
    else:
        lbl.append("")

    m = np.ones_like(wl, dtype=bool)
    if lim is None:
        lim = params.interpolation_range
    m &= wl >= (lim[0] if lim[0] < 1 else lim[0] * 1e-9)
    m &= wl <= (lim[1] if lim[1] < 1 else lim[1] * 1e-9)

    info_str = (
        rf"$\lambda_{{\mathrm{{min}}}}={np.min(params.l[params.l>0])*1e9:.1f}$ nm"
        + f"\nlower interpolation limit : {params.interpolation_range[0]*1e9:.1f} nm\n"
        + f"max time delay : {params.t.max()*1e12:.1f} ps"
    )

    left.annotate(
        info_str,
        xy=(1, 1),
        xytext=(-12, -12),
        xycoords="axes fraction",
        textcoords="offset points",
        va="top",
        ha="right",
        backgroundcolor=(1, 1, 1, 0.4),
    )

    m = np.argwhere(m)[:, 0]
    m = np.array(sorted(m, key=lambda el: wl[el]))

    if len(m) == 0:
        raise ValueError(f"nothing to plot in the range {lim!r}")

    # plot D
    right.plot(1e9 * wl[m], D[m], label=" ", **style)
    right.set_ylabel(units.D_ps_nm_km.label)

    # plot beta2
    left.plot(units.To.nm(params.w[m]), units.beta2_fs_cm.inv(beta_arr[m]), label=" ", **style)
    left.set_ylabel(units.beta2_fs_cm.label)

    left.set_xlabel(units.nm.label)
    right.set_xlabel("wavelength (nm)")

    if params.alpha_arr is not None and loss is not None:
        loss.plot(1e9 * wl[m], params.alpha_arr[m], c="r", ls="--")
        loss.set_ylabel("loss (1/m)", color="r")
        loss.set_yscale("log")
        loss.tick_params(axis="y", labelcolor="r")

    return lbl


def finish_plot(fig, legend_axes, all_labels, params):
    fig.suptitle(params.name)
    plt.tight_layout()

    handles, _ = legend_axes.get_legend_handles_labels()
    lbl_lengths = [[len(l) for l in lbl] for lbl in all_labels]
    lengths = np.max(lbl_lengths, axis=0)
    labels = [
        " ".join(format(l, f">{int(s)}s") for s, l in zip(lengths, lab)) for lab in all_labels
    ]

    legend = legend_axes.legend(handles, labels, prop=dict(size=8, family="monospace"))

    out_path = env.output_path()

    show = out_path is None
    if not show:
        file_name = out_path.stem + ".pdf"
        out_path = out_path.parent / file_name
        if (
            out_path.exists()
            and input(f"{out_path.name} already exsits, overwrite ? (y/[n])\n > ") != "y"
        ):
            show = True
        else:
            fig.savefig(out_path, bbox_inches="tight")
    if show:
        plt.show()


def plot_helper(config_path: Path) -> Iterable[tuple[dict, list[str], Parameters]]:
    cc = cycler(color=[f"C{i}" for i in range(10)]) * cycler(ls=["-", "--"])
    pseq = Configuration(load_toml(config_path))
    for style, (variables, params) in zip(cc, pseq):
        lbl = [pretty_format_value(name, value) for name, value in variables[1:-1]]
        yield style, lbl, params


def convert_params(params_file: os.PathLike):
    p = Path(params_file)
    if p.name == PARAM_FN:
        d = load_toml(params_file)
        d = translate_parameters(d)
        save_toml(params_file, d)
        print(f"converted {p}")
    else:
        for pp in p.glob(PARAM_FN):
            convert_params(pp)
        for pp in p.glob("fiber*"):
            if pp.is_dir():
                convert_params(pp)
