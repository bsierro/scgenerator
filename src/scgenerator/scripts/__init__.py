from itertools import cycle
import itertools
from pathlib import Path
from typing import Iterable
from cycler import cycler

import matplotlib.pyplot as plt
import numpy as np

from ..utils.parameter import BareParams

from ..initialize import ParamSequence
from ..physics import units, fiber
from ..spectra import Pulse
from ..utils import pretty_format_value
from .. import env, math


def plot_all(sim_dir: Path, limits: list[str]):
    for p in sim_dir.glob("*"):
        if not p.is_dir():
            continue

        pulse = Pulse(p)
        for lim in limits:
            left, right, unit = lim.split(",")
            left = float(left)
            right = float(right)
            pulse.plot_2D(left, right, unit, file_name=p.parent / f"{p.name}_{left}_{right}_{unit}")


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

        plot_1_init_spec_field(lim_t, lim_l, left, right, style, lbl, params)
        all_labels.append(lbl)
    finish_plot(fig, left, right, all_labels, params)


def plot_dispersion(config_path: Path, lim: tuple[float, float] = None):
    fig, (left, right) = plt.subplots(1, 2, figsize=(12, 7), sharex=True)
    left.grid()
    right.grid()
    all_labels = []
    already_plotted = set()
    for style, lbl, params in plot_helper(config_path):
        if (bbb := tuple(params.beta)) not in already_plotted:
            already_plotted.add(bbb)
        else:
            continue

        plot_1_dispersion(lim, left, right, style, lbl, params)
        all_labels.append(lbl)
    finish_plot(fig, left, right, all_labels, params)


def plot_init(
    config_path: Path,
    lim_field: tuple[float, float] = None,
    lim_spec: tuple[float, float] = None,
    lim_disp: tuple[float, float] = None,
):
    fig, ((tl, tr), (bl, br)) = plt.subplots(2, 2, figsize=(14, 10))
    tl.grid()
    tr.grid()
    all_labels = []
    already_plotted = set()
    for style, lbl, params in plot_helper(config_path):
        if (bbb := hash(params.field_0.tobytes())) not in already_plotted:
            already_plotted.add(bbb)
        else:
            continue
        lbl = plot_1_dispersion(lim_disp, tl, tr, style, lbl, params)
        lbl = plot_1_init_spec_field(lim_field, lim_spec, bl, br, style, lbl, params)
        all_labels.append(lbl)
    finish_plot(fig, tr, all_labels, params)


def plot_1_init_spec_field(lim_t, lim_l, left, right, style, lbl, params):
    field = math.abs2(params.field_0)
    spec = math.abs2(params.spec_0)
    t = units.fs.inv(params.t)
    wl = units.nm.inv(params.w)

    lbl.append(f"max at {wl[spec.argmax()]:.1f} nm")

    mt = np.ones_like(t, dtype=bool)
    if lim_t is not None:
        mt &= t >= lim_t[0]
        mt &= t <= lim_t[1]
    else:
        mt = find_lim(t, field)
    ml = np.ones_like(wl, dtype=bool)
    if lim_l is not None:
        ml &= t >= lim_l[0]
        ml &= t <= lim_l[1]
    else:
        ml = find_lim(wl, spec)

    left.plot(t[mt], field[mt])
    right.plot(wl[ml], spec[ml], label=" ", **style)
    return lbl


def plot_1_dispersion(lim, left, right, style, lbl, params):
    coef = params.beta / np.cumprod([1] + list(range(1, len(params.beta))))
    w_c = params.w_c

    beta_arr = np.zeros_like(w_c)
    for k, beta in reversed(list(enumerate(coef))):
        beta_arr = beta_arr + beta * w_c ** k
    wl = units.m.inv(params.w)

    zdw = math.all_zeros(wl, beta_arr)
    if len(zdw) > 0:
        zdw = zdw[np.argmin(abs(zdw - params.wavelength))]
        lbl.append(f"ZDW at {zdw*1e9:.1f}nm")
    else:
        lbl.append("")

    m = np.ones_like(wl, dtype=bool)
    if lim is None:
        lim = params.interp_range
    m &= wl >= lim[0]
    m &= wl <= lim[1]

    m = np.argwhere(m)[:, 0]
    m = np.array(sorted(m, key=lambda el: wl[el]))

    # plot D
    D = fiber.beta2_to_D(beta_arr, wl) * 1e6
    right.plot(1e9 * wl[m], D[m], label=" ", **style)
    right.set_ylabel(units.D_ps_nm_km.label)

    # plot beta
    left.plot(1e9 * wl[m], units.beta2_fs_cm.inv(beta_arr[m]), label=" ", **style)
    left.set_ylabel(units.beta2_fs_cm.label)

    left.set_xlabel("wavelength (nm)")
    right.set_xlabel("wavelength (nm)")
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


def plot_helper(config_path: Path) -> Iterable[tuple[dict, list[str], BareParams]]:
    cc = cycler(color=[f"C{i}" for i in range(10)]) * cycler(ls=["-", "--"])
    pseq = ParamSequence(config_path)
    for style, (variables, params) in zip(cc, pseq):
        lbl = [pretty_format_value(name, value) for name, value in variables[1:-1]]
        yield style, lbl, params


def find_lim(x: np.ndarray, y: np.ndarray, rel_thr: float = 0.01) -> int:
    threshold = y.min() + rel_thr * (y.max() - y.min())
    above_threshold = y > threshold
    ind = np.argsort(x)
    valid_ind = [
        np.array(list(g)) for k, g in itertools.groupby(ind, key=lambda i: above_threshold[i]) if k
    ]
    ind_above = sorted(valid_ind, key=lambda el: len(el), reverse=True)[0]
    width = len(ind_above)
    return np.concatenate(
        (
            np.arange(max(ind_above[0] - width, 0), ind_above[0]),
            ind_above,
            np.arange(ind_above[-1] + 1, min(len(y), ind_above[-1] + width)),
        )
    )
