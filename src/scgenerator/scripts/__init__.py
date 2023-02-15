import os
import re
from pathlib import Path
from typing import Any, Iterable, Optional

import matplotlib.pyplot as plt
import numpy as np
from cycler import cycler
from tqdm import tqdm

from scgenerator import env, math
from scgenerator.const import PARAM_FN, PARAM_SEPARATOR, SPEC1_FN
from scgenerator.parameter import FileConfiguration, Parameters
from scgenerator.physics import fiber, units
from scgenerator.plotting import plot_setup, transform_2D_propagation, get_extent
from scgenerator.spectra import SimulationSeries
from scgenerator.utils import _open_config, auto_crop, save_toml, simulations_list, load_toml, load_spectrum


def fingerprint(params: Parameters):
    h1 = hash(params.field_0.tobytes())
    h2 = tuple(params.beta2_coefficients)
    return h1, h2


def plot_all(sim_dir: Path, limits: list[str], show=False, **opts):
    for k, v in opts.items():
        if k in ["skip"]:
            opts[k] = int(v)
        if v == "True":
            opts[k] = True
        elif v == "False":
            opts[k] = False
    dir_list = simulations_list(sim_dir)
    if len(dir_list) == 0:
        dir_list = [sim_dir]
    limits = [
        tuple(func(el) for func, el in zip([float, float, str], lim.split(","))) for lim in limits
    ]
    with tqdm(total=len(dir_list) * max(1, len(limits))) as bar:
        for p in dir_list:
            pulse = SimulationSeries(p)
            if not limits:
                limits = [
                    (
                        pulse.params.interpolation_range[0] * 1e9,
                        pulse.params.interpolation_range[1] * 1e9,
                        "nm",
                    )
                ]
            for left, right, unit in limits:
                path, fig, ax = plot_setup(
                    pulse.path.parent
                    / (
                        pulse.path.name
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
        print(params.pretty_str(exclude="beta2_coefficients"))
    finish_plot(fig, tr, all_labels, params)


def plot_1_init_spec_field(
    lim_t: Optional[tuple[float, float]],
    lim_l: Optional[tuple[float, float]],
    left: plt.Axes,
    right: plt.Axes,
    style: dict[str, Any],
    lbl: str,
    params: Parameters,
):
    field = math.abs2(params.field_0)
    spec = math.abs2(params.spec_0)
    t = units.fs.inv(params.t)
    wl = units.nm.inv(params.w)

    lbl += f" max at {wl[spec.argmax()]:.1f} nm"

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
    zdw = zdw[(zdw >= params.interpolation_range[0]) & (zdw <= params.interpolation_range[1])]
    if len(zdw) > 0:
        zdw = zdw[np.argmin(abs(zdw - params.wavelength))]
        lbl += f" ZDW at {zdw*1e9:.1f}nm"
    else:
        lbl += ""

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
    left.plot(units.nm.inv(params.w[m]), units.beta2_fs_cm.inv(beta_arr[m]), label=" ", **style)
    left.set_ylabel(units.beta2_fs_cm.label)

    left.set_xlabel(units.nm.label)
    right.set_xlabel("wavelength (nm)")

    if params.alpha_arr is not None and loss is not None:
        loss.plot(1e9 * wl[m], params.alpha_arr[m], c="r", ls="--")
        loss.set_ylabel("loss (1/m)", color="r")
        loss.set_yscale("log")
        loss.tick_params(axis="y", labelcolor="r")

    return lbl


def finish_plot(fig: plt.Figure, legend_axes: plt.Axes, all_labels: list[str], params: Parameters):
    fig.suptitle(params.name)
    plt.tight_layout()

    handles, _ = legend_axes.get_legend_handles_labels()

    legend_axes.legend(handles, all_labels, prop=dict(size=8, family="monospace"))

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
    for style, (descriptor, params), _ in zip(cc, FileConfiguration(config_path), range(20)):
        yield style, descriptor.branch.formatted_descriptor(), params


def convert_params(params_file: os.PathLike):
    p = Path(params_file)
    if p.name == PARAM_FN:
        d = _open_config(params_file)
        save_toml(params_file, d)
        print(f"converted {p}")
    else:







        for pp in p.glob(PARAM_FN):
            convert_params(pp)
        for pp in p.glob("fiber*"):
            if pp.is_dir():
                convert_params(pp)


def partial_plot(root: os.PathLike, lim: str = None):
    path = Path(root)
    fig, ax = plt.subplots(figsize=(12, 8))
    fig.suptitle(path.name)
    spec_list = sorted(
        path.glob(SPEC1_FN.format("*")), key=lambda el: int(re.search("[0-9]+", el.name)[0])
    )





    params = Parameters(**load_toml(path / "params.toml"))
    params.z_targets = params.z_targets[: len(spec_list)]
    raw_values = np.array([load_spectrum(s) for s in spec_list])
    if lim is None:
        plot_range = units.PlotRange(
            0.5 * params.interpolation_range[0] * 1e9,
            1.1 * params.interpolation_range[1] * 1e9,
            "nm",
        )
    else:
        left_u, right_u, unit = lim.split(",")
        plot_range = units.PlotRange(float(left_u), float(right_u), unit)
    if plot_range.unit.type == "TIME":
        values = params.ifft(raw_values)
        log = False
        vmin = None
    else:
        values = raw_values
        log = "2D"
        vmin = -60

    x, y, values = transform_2D_propagation(
        values,
        plot_range,
        params,
        log=log,
    )
    ax.imshow(
        values,
        origin="lower",
        aspect="auto",
        vmin=vmin,
        interpolation="nearest",
        extent=get_extent(x, y),
    )

    return ax
