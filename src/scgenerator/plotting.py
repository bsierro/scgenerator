import os
from pathlib import Path
from typing import Any, Callable, Literal, Optional, Union

import matplotlib.gridspec as gs
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from scipy.interpolate import UnivariateSpline
from scipy.interpolate.interpolate import interp1d

from . import math
from .const import PARAM_SEPARATOR
from .defaults import default_plotting as defaults
from .math import abs2, span
from .parameter import Parameters
from .physics import pulse, units
from .physics.units import PlotRange, sort_axis

RangeType = tuple[float, float, Union[str, Callable]]
NO_LIM = object()


def get_extent(x, y, facx=1, facy=1):
    """
    returns the extent 4-tuple needed for imshow, aligning each pixel
    center to the corresponding value assuming uniformly spaced axes
    multiplying values by a constant factor is optional
    """
    try:
        dx = (x[1] - x[0]) / 2
    except IndexError:
        dx = 1
    try:
        dy = (y[1] - y[0]) / 2
    except IndexError:
        dy = 1
    return (
        (np.min(x) - dx) * facx,
        (np.max(x) + dx) * facx,
        (np.min(y) - dy) * facy,
        (np.max(y) + dy) * facy,
    )


def plot_setup(
    out_path: Path,
    file_type: str = "png",
    figsize: tuple[float, float] = defaults["figsize"],
    mode: Literal["default", "coherence", "coherence_T"] = "default",
) -> tuple[Path, plt.Figure, Union[plt.Axes, tuple[plt.Axes]]]:
    out_path = defaults["name"] if out_path is None else out_path
    out_path = Path(out_path)
    plot_name = out_path.name.replace(f".{file_type}", "")
    out_dir = out_path.resolve().parent

    file_name = plot_name + "." + file_type
    out_path = out_dir / file_name

    os.makedirs(out_dir, exist_ok=True)

    # ensure no overwrite
    ind = 0
    while (full_path := (out_dir / (plot_name + f"{PARAM_SEPARATOR}{ind}." + file_type))).exists():
        ind += 1

    if mode == "default":
        fig, ax = plt.subplots(figsize=figsize)
    elif mode == "coherence":
        n = defaults["avg_main_to_coherence_ratio"]
        gs1 = plt.GridSpec(n + 1, 1, hspace=0.4)
        fig = plt.figure(figsize=defaults["figsize"])
        top = fig.add_subplot(gs1[:n])
        top.tick_params(labelbottom=False)
        bot = fig.add_subplot(gs1[n], sharex=top)

        bot.set_ylim(-0.1, 1.1)
        bot.set_ylabel(r"|$g_{12}$|")
        ax = (top, bot)
    elif mode == "coherence_T":
        n = defaults["avg_main_to_coherence_ratio"]
        gs1 = plt.GridSpec(1, n + 1, wspace=0.4)
        fig = plt.figure(figsize=defaults["default_figsize"])
        top = fig.add_subplot(gs1[:n])
        top.tick_params(labelleft=False, left=False, right=True)
        bot = fig.add_subplot(gs1[n], sharey=top)

        bot.set_xlim(1.1, -0.1)
        bot.set_xlabel(r"|$g_{12}$|")
        ax = (top, bot)
    else:
        raise ValueError(f"mode {mode} not understood")

    return full_path, fig, ax


def draw_across(ax1, xy1, ax2, xy2, clip_on=False, **kwargs):
    """draws a line across 2 axes
    Parameters
    ----------
        ax1, ax2 : axis objects
        xy1, xy2 : tupple (float, float)
            The end positions in data coordinates (from their respective axis)
        **kwargs : arrowprops kwargs
    Returns
    ----------
        None
    """
    ax1.annotate(
        "",
        xy=xy1,
        xytext=xy2,
        xycoords=ax1.transData,
        textcoords=ax2.transData,
        arrowprops=dict(arrowstyle="-", clip_on=clip_on, **kwargs),
    )


def zoom(ax, zoom_ax, clip_on=False, **kwargs):
    l, r = zoom_ax.get_xlim()
    b, t = zoom_ax.get_ylim()
    draw_across(ax, (l, b), zoom_ax, (l, b), clip_on=clip_on, **kwargs)
    draw_across(ax, (l, t), zoom_ax, (l, t), clip_on=clip_on, **kwargs)
    draw_across(ax, (r, l), zoom_ax, (r, l), clip_on=clip_on, **kwargs)
    draw_across(ax, (r, t), zoom_ax, (r, t), clip_on=clip_on, **kwargs)
    ax.plot([l, l, r, r], [b, t, t, b], **kwargs)


def create_zoom_axis(
    axis,
    xlim,
    ylim=None,
    width_ratios=[1, 1, 1],
    height_ratios=[1, 1, 1],
    frame_style=dict(c="k", lw=0.5),
    plot=True,
):
    """creates a zoomed in plot inside a plot. Should be called as a last step as parent axis limits will be locked
    Parameters
    ----------
        axis : parent axis object
        xlim : tupple
            limits in parent axis data coordinates
        ylim : tupple, optional
        width_ratios, height_ration : lists of len 3
            cut the parent axis in 3x3 cells with these ratios, the center one will be the new axis
        frame_style : dict, optional
        plot : bool, optional
            whether to copy the lines or return an empty axis

    Returns
    ----------
        the new axis
    """
    axis.set_xlim(axis.get_xlim())
    axis.set_ylim(axis.get_ylim())

    # set up the axis
    grid = gs.GridSpecFromSubplotSpec(
        3,
        3,
        subplot_spec=axis,
        width_ratios=width_ratios,
        height_ratios=height_ratios,
        hspace=0,
        wspace=0,
    )
    inset = axis.get_figure().add_subplot(grid[1, 1])
    width_ratios = np.cumsum(np.array(width_ratios) / np.sum(width_ratios))
    height_ratios = np.cumsum(np.array(height_ratios) / np.sum(height_ratios))

    # copy the plot content
    if plot:
        lines = axis.get_lines()
        for line in lines:
            xdata = line.get_xdata()
            xdata, ind, _ = sort_axis(xdata, (*xlim, units.s))
            ydata = line.get_ydata()[ind]
            inset.plot(
                xdata, ydata, c=line.get_color(), ls=line.get_linestyle(), lw=line.get_linewidth()
            )
        inset.set_xlim(xlim)
        if ylim is not None:
            inset.set_ylim(ylim)
        ylim = inset.get_ylim()
    elif ylim is None:
        raise ValueError("ylim is mandatory when not plotting")

    # draw the box in parent axis
    dx = math.length(axis.get_xlim())
    dy = math.length(axis.get_ylim())
    l, r = xlim
    b, t = ylim
    axis.plot([l, l, r, r, l], [b, t, t, b, b], **frame_style)

    # draw lines connecting the box to the new axis
    ll = axis.get_xlim()[0] + width_ratios[0] * dx
    rr = axis.get_xlim()[0] + width_ratios[1] * dx
    bb = axis.get_ylim()[1] - height_ratios[1] * dy
    tt = axis.get_ylim()[1] - height_ratios[0] * dy

    axis.plot([l, ll], [t, tt], **frame_style)
    axis.plot([l, ll], [b, bb], **frame_style)
    axis.plot([r, rr], [t, tt], **frame_style)
    axis.plot([r, rr], [b, bb], **frame_style)

    return inset


def corner_annotation(text, ax, position="tl", rel_x_offset=0.05, rel_y_offset=0.05, **text_kwargs):
    """puts an annotatin in a corner of an ax
    Parameters
    ----------
        text : str
            text to put in the corner
        ax : matplotlib axis object
        position : str {"tl", "tr", "bl", "br"}

    Returns
    ----------
        nothing
    """
    # xlim = ax.get_xlim()
    # ylim = ax.get_ylim()

    # xoff = length(xlim) * rel_x_offset
    # yoff = length(ylim) * rel_y_offset

    if position[0] == "t":
        y = 1 - rel_y_offset
        va = "top"
    else:
        y = 0 + rel_y_offset
        va = "bottom"
    if position[1] == "l":
        x = 0 + rel_x_offset
        ha = "left"
    else:
        x = 1 - rel_x_offset
        ha = "right"

    ax.annotate(
        text,
        (x, y),
        (x, y),
        xycoords="axes fraction",
        textcoords="axes fraction",
        verticalalignment=va,
        horizontalalignment=ha,
        **text_kwargs,
    )

    return None


def propagation_plot(
    values: np.ndarray,
    plt_range: Union[PlotRange, RangeType],
    params: Parameters,
    ax: plt.Axes,
    log: Union[int, float, bool, str] = "1D",
    renormalize: bool = False,
    vmin: float = None,
    vmax: float = None,
    transpose: bool = False,
    skip: int = 1,
    cbar_label: Optional[str] = "normalized intensity (dB)",
    cmap: str = None,
) -> tuple[plt.Figure, plt.Axes, plt.Line2D, np.ndarray, np.ndarray]:
    """transforms and plots a 2D propagation

    Parameters
    ----------
    values : np.ndarray
        raw values, either complex fields or complex spectra
    plt_range : Union[PlotRange, RangeType]
        time, wavelength or frequency range
    params : Parameters
        parameters of the simulation
    log : Union[int, float, bool, str], optional
        what kind of log to apply, see apply_log for details. by default "1D"
    vmin : float, optional
        minimum value, by default None
    vmax : float, optional
        maximum value, by default None
    transpose : bool, optional
        whether to transpose the plot (rotate the plot 90° counterclockwise), by default False
    skip : int, optional
        only plot one every skip values along the x axis (y if transposed), by default 1
    cbar_label : Optional[str], optional
        label of the colorbar. No colorbar is drawn if this is set to None, by default "normalized intensity (dB)"
    cmap : str, optional
        colormap, by default None
    ax : plt.Axes, optional
        Axes obj on which to draw, by default None

    """
    x_axis, y_axis, values = transform_2D_propagation(values, plt_range, params, log, skip)
    if renormalize and log is False:
        values = values / values.max()
    if log is not False:
        vmax = defaults["vmax"] if vmax is None else vmax
        vmin = defaults["vmin"] if vmin is None else vmin
    plot_2D(
        values,
        x_axis,
        y_axis,
        ax,
        plt_range.unit.label,
        "propagation distance (m)",
        vmin,
        vmax,
        transpose,
        cmap,
        cbar_label,
    )


def plot_2D(
    values: np.ndarray,
    x_axis: np.ndarray,
    y_axis: np.ndarray,
    ax: Union[plt.Axes, tuple[plt.Axes, plt.Axes]],
    x_label: str = None,
    y_label: str = None,
    vmin: float = None,
    vmax: float = None,
    transpose: bool = False,
    cmap: str = None,
    cbar_label: str = "",
) -> Union[tuple[plt.Axes, plt.Axes], plt.Axes]:
    """plots given 2D values in a standard

    Parameters
    ----------
    values : np.ndarray, shape (m, n)
        real values to plot
    x_axis : np.ndarray, shape (n,)
        x axis
    y_axis : np.ndarray, shape (m,)
        y axis
    ax : Union[plt.Axes, tuple[plt.Axes, plt.Axes]]
        the ax on which to draw, or a tuple (ax, cbar_ax) where cbar_ax is the ax for the color bar
    x_label : str, optional
        x label
    y_label : str, optional
        y label
    vmin : float, optional
        minimum value (values below are the same color as the minimum of the colormap)
    vmax : float, optional
        maximum value (values above are the same color as the maximum of the colormap)
    transpose : bool, optional
        whether to rotate the plot 90° counterclockwise
    cmap : str, optional
        color map name
    cbar_label : str, optional
        label of the color bar axes. No color bar is drawn if cbar_label = None

    Returns
    -------
    Union[tuple[plt.Axes, plt.Axes], plt.Axes]
        ax if no color bar is drawn, a tuple (ax, cbar_ax) otherwise
    """
    # apply log transform if required
    cmap = defaults["cmap"] if cmap is None else cmap

    cbar_ax = None
    if isinstance(ax, tuple) and len(ax) > 1:
        ax, cbar_ax = ax[0], ax[1]

    fig = ax.get_figure()

    # Determine grid extent and spacing to be able to center
    # each pixel since by default imshow draws values at the lower-left corner
    if transpose:
        extent = get_extent(y_axis, x_axis)
        values = values.T
        ax.set_xlabel(y_label)
        ax.set_ylabel(x_label)
    else:
        extent = get_extent(x_axis, y_axis)
        ax.set_ylabel(y_label)
        ax.set_xlabel(x_label)

    ax.set_xlim(*extent[:2])
    ax.set_ylim(*extent[2:])

    interpolation = defaults["interpolation_2D"]
    im = ax.imshow(
        values,
        extent=extent,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        origin="lower",
        interpolation=interpolation,
        aspect="auto",
    )

    cbar = None
    if cbar_label is not None:
        if cbar_ax is None:
            cbar = fig.colorbar(im, ax=ax, orientation="vertical")
        else:
            cbar = fig.colorbar(im, cax=cbar_ax, orientation="vertical")
        cbar.ax.set_ylabel(cbar_label)

    if cbar_label is not None:
        return ax, cbar.ax
    else:
        return ax


def transform_2D_propagation(
    values: np.ndarray,
    plt_range: Union[PlotRange, RangeType],
    params: Parameters,
    log: Union[int, float, bool, str] = "1D",
    skip: int = 1,
    y_axis=None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """transforms raws values into plottable values

    Parameters
    ----------
    values : np.ndarray, shape (n, nt)
        values to transform
    plt_range : Union[PlotRange, RangeType]
        range
    params : Parameters
        parameters of the simulation
    log : Union[int, float, bool, str], optional
        see apply_log, by default "1D"
    skip : int, optional
        take one every skip values, by default 1

    Returns
    -------
    np.ndarray
        x_axis
    np.ndarray
        y_axis
    np.ndarray
        values

    Raises
    ------
    ValueError
        incorrect shape
    """

    if values.ndim != 2:
        raise ValueError(f"shape was {values.shape}. Can only plot 2D array")
    is_complex, x_axis, plt_range = prep_plot_axis(values, plt_range, params)
    if is_complex or any(values.ravel() < 0):
        values = abs2(values)
    # if params.full_field and plt_range.unit.type == "TIME":
    #     values = envelope_2d(x_axis, values)
    if y_axis is None:
        y_axis = params.z_targets

    x_axis, values = uniform_axis(x_axis, values, plt_range)
    y_axis, values.T[:] = uniform_axis(y_axis, values.T, None)
    values = apply_log(values, log)
    return x_axis[::skip], y_axis, values[:, ::skip]


def mean_values_plot(
    values: np.ndarray,
    plt_range: Union[PlotRange, RangeType],
    params: Parameters,
    ax: plt.Axes,
    log: Union[float, int, str, bool] = False,
    vmin: float = None,
    vmax: float = None,
    transpose: bool = False,
    spacing: Union[float, int] = 1,
    renormalize: bool = True,
    y_label: str = None,
    line_labels: tuple[str, str] = None,
    mean_style: dict[str, Any] = None,
    individual_style: dict[str, Any] = None,
) -> tuple[plt.Line2D, list[plt.Line2D]]:

    x_axis, mean_values, values = transform_mean_values(values, plt_range, params, log, spacing)
    if renormalize and log is False:
        maxi = mean_values.max()
        mean_values = mean_values / maxi
        values = values / maxi

    if log is not False:
        vmax = defaults["vmax_with_headroom"] if vmax is None else vmax
        vmin = defaults["vmin"] if vmin is None else vmin
    return plot_mean(
        values,
        mean_values,
        x_axis,
        ax,
        plt_range.unit.label,
        y_label,
        line_labels,
        vmin,
        vmax,
        transpose,
        mean_style,
        individual_style,
    )


def transform_mean_values(
    values: np.ndarray,
    plt_range: Union[PlotRange, RangeType],
    params: Parameters,
    log: Union[bool, int, float] = False,
    spacing: Union[int, float] = 1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """transforms values similar to transform_1D_values but with a collection of lines, giving also the mean

    Parameters
    ----------
    values : np.ndarray, shape (m, n)
        values to transform
    plt_range : Union[PlotRange, RangeType]
        x axis specifications
    params : Parameters
        parameters of the simulation
    log : Union[bool, int, float], optional
        see transform_1D_values for details, by default False
    spacing : Union[int, float], optional
        see transform_1D_values for details, by default 1

    Returns
    -------
    np.ndarray, shape (n,)
        x axis
    np.ndarray, shape (n,)
        mean y values
    np.ndarray, shape (m, n)
        all the values
    """
    if values.ndim != 2:
        print(f"Shape was {values.shape}. Can only plot 2D arrays")
        return
    is_complex, x_axis, plt_range = prep_plot_axis(values, plt_range, params)
    if is_complex:
        values = abs2(values)
    new_axis, ind, ext = sort_axis(x_axis, plt_range)
    values = values[:, ind]
    if plt_range.unit.type == "WL" and plt_range.conserved_quantity:
        values = np.apply_along_axis(units.to_WL, -1, values, new_axis)

    if isinstance(spacing, (float, np.floating)):
        tmp_axis = np.linspace(*span(new_axis), int(len(new_axis) / spacing))
        values = np.array([UnivariateSpline(new_axis, v, k=4, s=0)(tmp_axis) for v in values])
        new_axis = tmp_axis
    elif isinstance(spacing, (int, np.integer)) and spacing > 1:
        values = values[:, ::spacing]
        new_axis = new_axis[::spacing]

    mean_values = np.mean(values, axis=0)

    if log is not False:
        if log is not True and isinstance(log, (int, float, np.integer, np.floating)):
            ref = log
        else:
            ref = float(mean_values.max())
        values = apply_log(values, ref)
        mean_values = apply_log(mean_values, ref)
    return new_axis, mean_values, values


def plot_mean(
    values: np.ndarray,
    mean_values: np.ndarray,
    x_axis: np.ndarray,
    ax: plt.Axes,
    x_label: str = None,
    y_label: str = None,
    line_labels: tuple[str, str] = None,
    vmin: float = None,
    vmax: float = None,
    transpose: bool = False,
    mean_style: dict[str, Any] = None,
    individual_style: dict[str, Any] = None,
) -> tuple[plt.Line2D, list[plt.Line2D]]:
    """plots already transformed 1D values

    Parameters
    ----------
    values : np.ndarray, shape (m, n)
        values to plot
    mean_values : np.ndarray, shape (n,)
        values to plot
    x_axis : np.ndarray, shape (n,)
        corresponding x axis
    ax : plt.Axes
        ax on which to plot
    x_label : str, optional
        x label, by default None
    y_label : str, optional
        y label, by default None
    line_labels: tuple[str, str]
        label of the mean line and the individual lines, by default None
    vmin : float, optional
        minimum y limit, by default None
    vmax : float, optional
        maximum y limit, by default None
    transpose : bool, optional
        rotate the plot 90° counterclockwise, by default False
    """
    individual_style = defaults["muted_style"] if individual_style is None else individual_style
    mean_style = defaults["highlighted_style"] if mean_style is None else mean_style
    labels = defaults["avg_line_labels"] if line_labels is None else line_labels
    lines = []
    if transpose:
        for value in values[:-1]:
            lines += ax.plot(value, x_axis, **individual_style)
        lines += ax.plot(values[-1], x_axis, **individual_style)
        (mean_line,) = ax.plot(mean_values, x_axis, **mean_style)
        ax.set_xlim(vmax, vmin)
        ax.yaxis.tick_right()
        ax.set_xlabel(y_label)
        ax.set_ylabel(x_label)
    else:
        for value in values[:-1]:
            lines += ax.plot(x_axis, value, **individual_style)
        lines += ax.plot(x_axis, values[-1], label=labels[0], **individual_style)
        (mean_line,) = ax.plot(x_axis, mean_values, label=labels[1], **mean_style)
        ax.set_ylim(vmin, vmax)
        ax.set_ylabel(y_label)
        ax.set_xlabel(x_label)

    return mean_line, lines


def single_position_plot(
    values: np.ndarray,
    plt_range: Union[PlotRange, RangeType],
    params: Parameters,
    ax: plt.Axes,
    log: Union[str, int, float, bool] = False,
    vmin: float = None,
    vmax: float = None,
    transpose: bool = False,
    spacing: Union[int, float] = 1,
    renormalize: bool = False,
    y_label: str = None,
    **line_kwargs,
) -> tuple[plt.Figure, plt.Axes, plt.Line2D, np.ndarray, np.ndarray]:

    x_axis, values = transform_1D_values(values, plt_range, params, log, spacing)
    if renormalize:
        values = values / values.max()

    if log is not False:
        vmax = defaults["vmax_with_headroom"] if vmax is None else vmax
        vmin = defaults["vmin"] if vmin is None else vmin

    return plot_1D(
        values, x_axis, ax, plt_range.unit.label, y_label, vmin, vmax, transpose, **line_kwargs
    )


def plot_1D(
    values: np.ndarray,
    x_axis: np.ndarray,
    ax: plt.Axes,
    x_label: str = None,
    y_label: str = None,
    vmin: float = None,
    vmax: float = None,
    transpose: bool = False,
    **line_kwargs,
) -> plt.Line2D:
    """plots already transformed 1D values

    Parameters
    ----------
    values : np.ndarray, shape (n,)
        values to plot
    x_axis : np.ndarray, shape (n,)
        corresponding x axis
    ax : plt.Axes,
        ax on which to plot
    x_label : str, optional
        x label
    y_label : str, optional
        y label
    vmin : float, optional
        minimum y limit, by default None
    vmax : float, optional
        maximum y limit, by default None
    transpose : bool, optional
        rotate the plot 90° counterclockwise, by default False
    """
    if transpose:
        (line,) = ax.plot(values, x_axis, **line_kwargs)
        ax.yaxis.tick_right()
        ax.yaxis.set_label_position("right")
        ax.set_xlim(vmax, vmin)
        ax.set_xlabel(y_label)
        ax.set_ylabel(x_label)
    else:
        (line,) = ax.plot(x_axis, values, **line_kwargs)
        ax.set_ylim(vmin, vmax)
        ax.set_ylabel(y_label)
        ax.set_xlabel(x_label)
    return line


def transform_1D_values(
    values: np.ndarray,
    plt_range: Union[PlotRange, RangeType],
    params: Parameters,
    log: Union[int, float, bool] = False,
    spacing: Union[int, float] = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """transforms raw values to be plotted

    Parameters
    ----------
    values : np.ndarray, shape (n,)
        values to plot, may be complex
    plt_range : Union[PlotRange, RangeType]
        plot range specification, either (min, max, unit) or a PlotRange obj
    params : Parameters
        parameters of the simulations
    log : Union[int, float, bool], optional
        if True, will convert to dB relative to max. If a float or int, whill
        convert to dB relative to that number, by default False
    spacing : Union[int, float], optional
        change the resolution by either taking only 1 every `spacing` value (int) or
        multiplying the original spacing between point by `spacing` and interpolating

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        x axis and values
    """
    if len(values.shape) != 1:
        raise ValueError("Can only plot 1D values")
    is_complex, x_axis, plt_range = prep_plot_axis(values, plt_range, params)
    if is_complex:
        values = abs2(values)
    new_axis, ind, ext = sort_axis(x_axis, plt_range)
    values = values[ind]
    if plt_range.unit.type == "WL" and plt_range.conserved_quantity:
        values = units.to_WL(values, new_axis)

    if isinstance(spacing, (float, np.floating)):
        tmp_axis = np.linspace(*span(new_axis), int(len(new_axis) / spacing))
        values = UnivariateSpline(new_axis, values, k=4, s=0)(tmp_axis)
        new_axis = tmp_axis
    elif isinstance(spacing, (int, np.integer)) and spacing > 1:
        values = values[::spacing]
        new_axis = new_axis[::spacing]

    if isinstance(log, str):
        log = True
    values = apply_log(values, log)
    return new_axis, values


def plot_spectrogram(
    values: np.ndarray,
    x_range: RangeType,
    y_range: RangeType,
    params: Parameters,
    t_res: int = None,
    gate_width: float = None,
    log: bool = "2D",
    vmin: float = None,
    vmax: float = None,
    cbar_label: str = "normalized intensity (dB)",
    cmap: str = None,
    ax: plt.Axes = None,
):
    """Plots a spectrogram given a complex field in the time domain
    Parameters
    ----------
    values : 2D array
        axis 0 defines the position in the fiber and axis 1 the position in time, frequency or wl
        example : [[1, 2, 3], [0, 1, 0]] describes a quantity at 3 different freq/time and at two locations in the fiber
    x_range, y_range : tupple (min, max, units)
        one of them must be time, the other one must be wl/freq
        min, max : int or float
            minimum and maximum values given in the desired units
        units : function to convert from the desired units to rad/s or to time.
                common functions are already defined in scgenerator.physics.units
                look there for more details
    params : Parameters
        parameters of the simulations
    log : bool, optional
        whether to compute the logarithm of the spectrogram
    vmin : float, optional
        min value of the colorbar
    vmax : float, optional
        max value of the colorbar
    cbar_label : str or None
        label of the colorbar. Will not draw colorbar if None
    file_type : str, optional
        usually pdf or png
    plt_name : str, optional
        special name to give to the plot. A name is automatically assigned anyway
    cmap : str, optional
        colormap to be used in matplotlib.pyplot.imshow
    ax : matplotlib.axes._subplots.AxesSubplot object or tupple of 2 axis objects, optional
        axis on which to draw the plot
        if only one is given, a new one will be created to draw the colorbar

    """
    if values.ndim != 1:
        print("plot_spectrogram can only plot 1D arrays")
        return
    x_range: PlotRange
    y_range: PlotRange
    _, x_axis, x_range = prep_plot_axis(values, x_range, params)
    _, y_axis, y_range = prep_plot_axis(values, y_range, params)

    if (x_range.unit.type == "TIME") == (y_range.unit.type == "TIME"):
        print("exactly one range must be a time range")
        return

    # 0 axis means x-axis -> determine final orientation of spectrogram
    time_axis = 0 if x_range.unit.type not in ["WL", "FREQ", "AFREQ"] else 1
    if time_axis == 0:
        t_range = x_range
    else:
        t_range = y_range

    # Actually compute the spectrogram
    t_win = 2 * np.max(t_range.unit(np.abs((t_range.left, t_range.right))))
    spec_kwargs = dict(t_res=t_res, t_win=t_win, gate_width=gate_width, shift=False)
    spec, new_t = pulse.spectrogram(
        params.t.copy(), values, **{k: v for k, v in spec_kwargs.items() if v is not None}
    )
    if time_axis == 0:
        x_axis = new_t
    else:
        y_axis = new_t

    x_axis, spec = uniform_axis(x_axis, spec, x_range)
    y_axis, spec.T[:] = uniform_axis(y_axis, spec.T, y_range)

    values = apply_log(spec, log)

    return plot_2D(
        values,
        x_axis,
        y_axis,
        ax,
        x_range.unit.label,
        y_range.unit.label,
        vmin,
        vmax,
        False,
        cmap,
        cbar_label,
    )


def uniform_axis(
    axis: np.ndarray, values: np.ndarray, new_axis_spec: Union[PlotRange, RangeType, str]
) -> tuple[np.ndarray, np.ndarray]:
    """given some values(axis), creates a new uniformly spaced axis and interpolates
    the values over it.

    Parameters
    ----------
    axis : np.ndarray, shape (n,)
        grid points to which values correspond
    values : np.ndarray, shape (n,) or (m, n)
        values as function of axis
    new_axis_spec : Union[PlotRange, RangeType, str]
        specifications of the new axis. May be None, a unit as a str,
        a tuple (min, max, unit) or a PlotRange obj

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        new axis and new values

    Raises
    ------
    TypeError
        invalid new_axis_spec
    """
    if new_axis_spec is None:
        new_axis_spec = "unity"
    if isinstance(new_axis_spec, str) or callable(new_axis_spec):
        unit = units.get_unit(new_axis_spec)
        plt_range = PlotRange(unit.inv(axis.min()), unit.inv(axis.max()), new_axis_spec)
    elif isinstance(new_axis_spec, tuple):
        plt_range = PlotRange(*new_axis_spec)
    elif isinstance(new_axis_spec, PlotRange):
        plt_range = new_axis_spec
    else:
        raise TypeError(f"Don't know how to interpret {new_axis_spec}")
    tmp_axis, ind, ext = sort_axis(axis, plt_range)
    values = np.atleast_2d(values)
    if np.allclose((diff := np.diff(tmp_axis))[0], diff):
        new_axis = tmp_axis
        values = values[:, ind]
    else:
        if plt_range.unit.type == "WL" and plt_range.conserved_quantity:
            values[:, ind] = np.apply_along_axis(units.to_WL, 1, values[:, ind], tmp_axis)
        new_axis = np.linspace(tmp_axis.min(), tmp_axis.max(), len(tmp_axis))
        values = np.array([interp1d(tmp_axis, v[ind])(new_axis) for v in values])
    return new_axis, values.squeeze()


def apply_log(values: np.ndarray, log: Union[str, bool, float, int]) -> np.ndarray:
    """apply log transform

    Parameters
    ----------
    values : np.ndarray
        input array
    log : Union[str, bool, float, int]
        True -> "1D"
        "1D" -> each row has its own reference value
        "smooth 1D" -> attempted compromise between 2D and 1D. Will clip the highest values
        "2D" -> same reference value for the whole 2D array
        float, int -> take this value as the reference
        False -> don't apply log

    Returns
    -------
    np.ndarray
        values with log applied

    Raises
    ------
    ValueError
        unrecognized log argument
    """

    if log is not False:
        if isinstance(log, (float, int, np.floating, np.integer)) and log is not True:
            values = units.to_log(values, ref=log)
        elif log == "2D":
            values = units.to_log2D(values)
        elif log == "1D" or log is True:
            values = np.apply_along_axis(units.to_log, -1, values)
        elif log == "smooth 1D":
            ref = np.max(values, axis=1)
            ind = np.argmax((ref[:-1] - ref[1:]) < 0)
            values = units.to_log(values, ref=np.max(ref[ind:]))
        else:
            raise ValueError(f"Log argument {log} not recognized")
    return values


def prep_plot_axis(
    values: np.ndarray, plt_range: Union[PlotRange, RangeType], params: Parameters
) -> tuple[bool, np.ndarray, PlotRange]:
    is_spectrum = values.dtype == "complex"
    if not isinstance(plt_range, PlotRange):
        plt_range = PlotRange(*plt_range)
    if plt_range.unit.type in ["WL", "FREQ", "AFREQ"]:
        x_axis = params.w.copy()
    else:
        x_axis = params.t.copy()
    return is_spectrum, x_axis, plt_range


def white_bottom_cmap(name, start=0, end=1, new_name="white_background", c_back=(1, 1, 1, 1)):
    """returns a new colormap based on "name" but that has a solid bacground (default=white)"""
    top = plt.get_cmap(name, 1024)
    n_bottom = 80
    bottom = np.ones((n_bottom, 4))
    for i in range(4):
        bottom[:, i] = np.linspace(c_back[i], top(start)[i], n_bottom)
    return ListedColormap(np.vstack((bottom, top(np.linspace(start, end, 1024)))), name=new_name)


def default_marker_style(k):
    """returns a style dictionary

    Parameters
    ----------
    k : int
        index in the cycle

    Returns
    -------
    dict
        style dictionnary
    """
    return dict(
        marker=defaults["markers"][k],
        markerfacecolor="none",
        linestyle=":",
        lw=1,
        c=defaults["color_cycle"][k],
    )


def arrowstyle(direction=1, color="white"):
    return dict(
        arrowprops=dict(arrowstyle="->", connectionstyle=f"arc3,rad={direction*0.3}", color=color),
        color=color,
        backgroundcolor=(0.5, 0.5, 0.5, 0.5),
    )


def measure_and_annotate_fwhm(
    ax: plt.Axes,
    t: np.ndarray,
    field: np.ndarray,
    side: Literal["left", "right"] = "right",
    unit="fs",
    arrow_length_pts: float = 20.0,
    arrow_props: dict[str, Any] = None,
) -> float:
    """measured the FWHM of a pulse and plots it

    Parameters
    ----------
    ax : plt.Axes
        ax on which to plot
    t : np.ndarray, shape (n,)
        time in s
    field : np.ndarray, shape (n,)
        complex field
    side : Literal["left", "right"]
        whether to write the text on the right or left side
    unit : str, optional
        units of the plot, by default "fs"
    arrow_length_pts : float, optional
        length of the arrows in pts, by default 20.0
    arrow_props : dict[str, Any], optional
        style of the arrow to be passed to plt.annotate, by default None

    Returns
    -------
    float
        FWHM in units
    """
    unit = units.get_unit(unit)
    if np.iscomplexobj(field):
        field = abs2(field)
    _, (left, right), *_ = pulse.find_lobe_limits(unit.inv(t), field)
    arrow_label = f"{right - left:.1f} {unit.name}"

    annotate_fwhm(ax, left, right, arrow_label, field.max(), side, arrow_length_pts, arrow_props)
    return right - left


def annotate_fwhm(
    ax, left, right, arrow_label, v_max=1, side="right", arrow_length_pts=20.0, arrow_props=None
):
    arrow_dict = dict(arrowstyle="->")
    if arrow_props is not None:
        arrow_dict |= arrow_props
    ax.annotate(
        "" if side == "right" else arrow_label,
        (left, v_max / 2),
        xytext=(-arrow_length_pts, 0),
        ha="right",
        va="center",
        textcoords="offset points",
        arrowprops=arrow_dict,
    )
    ax.annotate(
        "" if side == "left" else arrow_label,
        (right, v_max / 2),
        xytext=(arrow_length_pts, 0),
        textcoords="offset points",
        arrowprops=arrow_dict,
        va="center",
    )
