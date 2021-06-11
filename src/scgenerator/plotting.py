import os
from pathlib import Path
from typing import Any, Callable, Dict, Literal, Optional, Tuple, Union

import matplotlib.gridspec as gs
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from scipy.interpolate import UnivariateSpline

from . import io, math
from .defaults import default_plotting as defaults
from .math import abs2, make_uniform_1D, span
from .physics import pulse, units
from .utils.parameter import BareParams

RangeType = Tuple[float, float, Union[str, Callable]]


def plot_setup(
    out_path: Path,
    file_type: str = "png",
    figsize: Tuple[float, float] = defaults["figsize"],
    mode: Literal["default", "coherence", "coherence_T"] = "default",
) -> Tuple[Path, plt.Figure, Union[plt.Axes, Tuple[plt.Axes]]]:
    """It should return :
    - a folder_name
    - a file name
    - a fig
    - an axis
    """
    out_path = defaults["name"] if out_path is None else out_path
    plot_name = out_path.stem
    out_dir = out_path.resolve().parent

    file_name = plot_name + "." + file_type
    out_path = out_dir / file_name

    os.makedirs(out_dir, exist_ok=True)

    # ensure no overwrite
    ind = 0
    while (full_path := (out_dir / (plot_name + f"_{ind}." + file_type))).exists():
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
        ymin, ymax = 0, 0
        for line in lines:
            xdata = line.get_xdata()
            xdata, ind, _ = units.sort_axis(xdata, (*xlim, units.s))
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


def _finish_plot_2D(
    values,
    x_axis,
    x_label,
    y_axis,
    y_label,
    log,
    vmin,
    vmax,
    transpose,
    cmap,
    cbar_label,
    ax,
    file_name,
    file_type,
    params,
):

    # apply log transform if required
    if log != False:
        vmax = defaults["vmax"] if vmax is None else vmax
        vmin = defaults["vmin"] if vmin is None else vmin
        if isinstance(log, (float, int)) and log != True:
            values = units.to_log(values, ref=log)

        elif log == "2D":
            values = units.to_log2D(values)

        elif log == "1D":
            values = np.apply_along_axis(units.to_log, 1, values)

        elif log == "smooth 1D":
            ref = np.max(values, axis=1)
            ind = np.argmax((ref[:-1] - ref[1:]) < 0)
            values = units.to_log(values, ref=np.max(ref[ind:]))

        elif log == "unique 1D":
            try:
                ref = _finish_plot_2D.ref
                print(f"recovered reference value {ref} for log plot")
            except AttributeError:
                ref = np.max(values, axis=1)
                ind = np.argmax((ref[:-1] - ref[1:]) < 0)
                ref = np.max(ref[ind:])
                _finish_plot_2D.ref = ref

            values = units.to_log(values, ref=ref)
    cmap = defaults["cmap"] if cmap is None else cmap

    is_new_plot = ax is None
    cbar_ax = None
    if isinstance(ax, tuple) and len(ax) > 1:
        ax, cbar_ax = ax[0], ax[1]

    folder_name = ""
    if is_new_plot:
        out_path, fig, ax = plot_setup(out_path=Path(folder_name) / file_name, file_type=file_type)
    else:
        fig = ax.get_figure()

    # Determine grid extent and spacing to be able to center
    # each pixel since by default imshow draws values at the lower-left corner
    if transpose:
        dy = x_axis[1] - x_axis[0]
        ext_y = span(x_axis)
        dx = y_axis[1] - y_axis[0]
        ext_x = span(y_axis)
        values = values.T
        ax.set_xlabel(y_label)
        ax.set_ylabel(x_label)
    else:
        dx = x_axis[1] - x_axis[0]
        ext_x = span(x_axis)
        dy = y_axis[1] - y_axis[0]
        ext_y = span(y_axis)
        ax.set_ylabel(y_label)
        ax.set_xlabel(x_label)

    ax.set_xlim(*ext_x)
    ax.set_ylim(*ext_y)

    interpolation = defaults["interpolation_2D"]
    im = ax.imshow(
        values,
        extent=[ext_x[0] - dx / 2, ext_x[1] + dx / 2, ext_y[0] - dy / 2, ext_y[1] + dy / 2],
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

    if is_new_plot:
        fig.savefig(out_path, bbox_inches="tight", dpi=200)
        print(f"plot saved in {out_path}")
    if cbar_label is not None:
        return fig, ax, cbar.ax
    else:
        return fig, ax


def plot_spectrogram(
    values: np.ndarray,
    x_range: RangeType,
    y_range: RangeType,
    params: BareParams,
    t_res: int = None,
    gate_width: float = None,
    log: bool = True,
    vmin: float = None,
    vmax: float = None,
    cbar_label: str = "normalized intensity (dB)",
    file_type: str = "png",
    file_name: str = None,
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
        params : BareParams
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

    if (x_range[2].type == "TIME") == (y_range[2].type == "TIME"):
        print("exactly one range must be a time range")
        return

    log = "2D" if log in ["2D", True] else False

    # 0 axis means x-axis -> determine final orientation of spectrogram
    time_axis = 0 if x_range[2].type not in ["WL", "FREQ", "AFREQ"] else 1
    if time_axis == 0:
        t_range, f_range = x_range, y_range
    else:
        t_range, f_range = y_range, x_range

    # Actually compute the spectrogram
    t_win = 2 * np.max(t_range[2](np.abs(t_range[:2])))
    spec_kwargs = dict(t_res=t_res, t_win=t_win, gate_width=gate_width, shift=False)
    spec, new_t = pulse.spectrogram(
        params.t.copy(), values, **{k: v for k, v in spec_kwargs.items() if v is not None}
    )

    # Crop and reoder axis
    new_t, ind_t, _ = units.sort_axis(new_t, t_range)
    new_f, ind_f, _ = units.sort_axis(params.w, f_range)
    values = spec[ind_t][:, ind_f]
    if f_range[2].type == "WL":
        values = np.apply_along_axis(
            units.to_WL, 1, values, params.frep, units.m(f_range[2].inv(new_f))
        )
        values = np.apply_along_axis(make_uniform_1D, 1, values, new_f)

    if time_axis == 0:
        x_axis, y_axis = new_t, new_f
        values = values.T
    else:
        x_axis, y_axis = new_f, new_t

    return _finish_plot_2D(
        values,
        x_axis,
        x_range[2].label,
        y_axis,
        y_range[2].label,
        log,
        vmin,
        vmax,
        False,
        cmap,
        cbar_label,
        ax,
        file_name,
        file_type,
        params,
    )


def plot_results_2D(
    values: np.ndarray,
    plt_range: RangeType,
    params: BareParams,
    log: Union[int, float, bool, str] = "1D",
    skip: int = 16,
    vmin: float = None,
    vmax: float = None,
    transpose: bool = False,
    cbar_label: Optional[str] = "normalized intensity (dB)",
    file_type: str = "png",
    file_name: str = None,
    cmap: str = None,
    ax: plt.Axes = None,
):
    """
    plots 2D arrays and automatically saves the plots, as well as returns it

    Parameters
    ----------
        values : 2D array
            axis 0 defines the position in the fiber and axis 1 the position in time, frequency or wl
            example : [[1, 2, 3], [0, 1, 0]] describes a quantity at 3 different freq/time and at two locations in the fiber
        plt_range : tupple (min, max, units)
            min, max : int or float
                minimum and maximum values given in the desired units
            units : function to convert from the desired units to rad/s or to time.
                    common functions are already defined in scgenerator.physics.units
                    look there for more details
        params : dict
            parameters of the simulations
        log : str {"1D", "2D", "smooth 1D"} or int, float or bool, optional
            str : plot in dB
                1D : takes the log for every slice
                2D : takes the log for the whole 2D array
                smooth 1D : figures out a smart reference value for the whole 2D array
            int, float : plot in dB
                reference value
            bool : whether to use 1D variant or nothing
        skip : int, optional
            take 1 every skip values along the -1 axis
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
    returns
        fig, ax : matplotlib objects containing the plots
    example:
        if spectra is a (m, n, nt) array, one can plot a spectrum evolution as such:
        >>> fig, ax = plot_results_2D(spectra[:, -1], (600, 1600, nm), log=True, "Heidt2017")
    """

    if values.ndim != 2:
        print(f"Shape was {values.shape}. plot_results_2D can only plot 2D arrays")
        return

    is_spectrum, x_axis, plt_range = _prep_plot(values, plt_range, params)

    # crop and convert
    x_axis, ind, ext = units.sort_axis(x_axis[::skip], plt_range)
    values = values[:, ::skip][:, ind]
    if is_spectrum:
        values = abs2(values)

    # make uniform if converting to wavelength
    if plt_range[2].type == "WL":
        if is_spectrum:
            values = np.apply_along_axis(units.to_WL, 1, values, params.frep, x_axis)
        values = np.array(
            [make_uniform_1D(v, x_axis, n=len(x_axis), method="linear") for v in values]
        )

    lim_diff = 1e-5 * np.max(params.z_targets)
    dz_s = np.diff(params.z_targets)
    if not np.all(np.diff(dz_s) < lim_diff):
        new_z = np.linspace(
            *span(params.z_targets),
            int(
                np.floor(
                    (np.max(params.z_targets) - np.min(params.z_targets))
                    / np.min(dz_s[dz_s > lim_diff])
                )
            ),
        )
        values = np.array(
            [make_uniform_1D(v, params.z_targets, n=len(new_z), method="linear") for v in values.T]
        ).T
        params.z_targets = new_z
    return _finish_plot_2D(
        values,
        x_axis,
        plt_range[2].label,
        params.z_targets,
        "propagation distance (m)",
        log,
        vmin,
        vmax,
        transpose,
        cmap,
        cbar_label,
        ax,
        file_name,
        file_type,
        params,
    )


def plot_results_1D(
    values: np.ndarray,
    plt_range: RangeType,
    params: BareParams,
    log: Union[str, int, float, bool] = False,
    spacing: Union[int, float] = 1,
    vmin: float = None,
    vmax: float = None,
    ylabel: str = None,
    yscaling: float = 1,
    file_type: str = "pdf",
    file_name: str = None,
    ax: plt.Axes = None,
    line_label: str = None,
    transpose: bool = False,
    **line_kwargs,
):
    """

    Parameters
    ----------
        values : 1D array
            if values are complex, the abs^2 is computed before plotting
        plt_range : tupple (min, max, units)
            min, max : int or float
                minimum and maximum values given in the desired units
            units : function to convert from the desired units to rad/s or to time.
                    common functions are already defined in scgenerator.physics.units
                    look there for more details
        params : dict
            parameters of the simulations
        log : str {"1D"} or int, float or bool, optional
            str : plot in dB
                1D : takes the log for every slice
            int, float : plot in dB
                reference value
            bool : whether to use 1D variant or nothing
        spacing : int, float, optional
            tells the function to take one value every `spacing` one available. If a float is given, it will interpolate with a spline.
        vmin : float, optional
            min value of the colorbar
        vmax : float, optional
            max value of the colorbar
        ylabel : str, optional
            label of the y axis (x axis in transposed mode). Default is "normalized intensity (dB)" for log plots
        yscaling : float, optional
            scale the y values by this amount
        file_type : str, optional
            usually pdf or png
        plt_name : str, optional
            special name to give to the plot. A name is automatically assigned anyway
        ax : matplotlib.axes._subplots.AxesSubplot object, optional
            axis on which to draw the plot
        line_label : str, optional
            label of the line
        transpose : bool, optional
            transpose the plot
        line_kwargs : to be passed to plt.plot
    returns
        fig, ax : matplotlib objects containing the plots
    example:
        if spectra is a (m, n, nt) array, one can plot a spectrum evolution as such:
        >>> fig, ax = plot_results_2D(spectra[:, -1], (600, 1600, nm), log=True, "Heidt2017")
    """

    if len(values.shape) != 1:
        print(f"Shape was {values.shape}. plot_results_1D can only plot 1D arrays")
        return

    is_spectrum, x_axis, plt_range = _prep_plot(values, plt_range, params)

    # crop and convert
    x_axis, ind, ext = units.sort_axis(x_axis, plt_range)
    values = values[ind]
    if is_spectrum:
        values = abs2(values)
    values *= yscaling

    # make uniform if converting to wavelength
    if plt_range[2].type == "WL":
        if is_spectrum:
            values = units.to_WL(values, params.frep, units.m.inv(params.w[ind]))

    # change the resolution
    if isinstance(spacing, float):
        new_x_axis = np.linspace(*span(x_axis), int(len(x_axis) / spacing))
        values = UnivariateSpline(x_axis, values, k=4, s=0)(new_x_axis)
        x_axis = new_x_axis
    elif isinstance(spacing, int) and spacing > 1:
        values = values[::spacing]
        x_axis = x_axis[::spacing]

    # apply log transform if required
    if log == False:
        pass
    else:
        ylabel = "normalized intensity (dB)" if ylabel is None else ylabel
        vmax = defaults["vmax_with_headroom"] if vmax is None else vmax
        vmin = defaults["vmin"] if vmin is None else vmin
        if isinstance(log, (float, int)) and log != True:
            values = units.to_log(values, ref=log)
        else:
            values = units.to_log(values)

    is_new_plot = ax is None

    folder_name = ""
    if is_new_plot:
        out_path, fig, ax = plot_setup(out_path=Path(folder_name) / file_name, file_type=file_type)
    else:
        fig = ax.get_figure()
    if transpose:
        ax.plot(values, x_axis, label=line_label, **line_kwargs)
        ax.yaxis.tick_right()
        ax.yaxis.set_label_position("right")
        ax.set_xlim(vmax, vmin)
        ax.set_xlabel(ylabel)
        ax.set_ylabel(plt_range[2].label)
    else:
        ax.plot(x_axis, values, label=line_label, **line_kwargs)
        ax.set_ylim(vmin, vmax)
        ax.set_ylabel(ylabel)
        ax.set_xlabel(plt_range[2].label)

    if is_new_plot:
        fig.savefig(out_path, bbox_inches="tight", dpi=200)
        print(f"plot saved in {out_path}")
    return fig, ax, x_axis, values


def _prep_plot(values: np.ndarray, plt_range: RangeType, params: BareParams):
    is_spectrum = values.dtype == "complex"
    plt_range = (*plt_range[:2], units.get_unit(plt_range[2]))
    if plt_range[2].type in ["WL", "FREQ", "AFREQ"]:
        x_axis = params.w.copy()
    else:
        x_axis = params.t.copy()
    return is_spectrum, x_axis, plt_range


def plot_avg(
    values: np.ndarray,
    plt_range: RangeType,
    params: BareParams,
    log: Union[float, int, str, bool] = False,
    spacing: Union[float, int] = 1,
    vmin: float = None,
    vmax: float = None,
    ylabel: str = None,
    yscaling: float = 1,
    renormalize: bool = True,
    add_coherence: bool = False,
    file_type: str = "png",
    file_name: str = None,
    ax: plt.Axes = None,
    line_labels: Tuple[str, str] = None,
    legend: bool = True,
    legend_kwargs: Dict[str, Any] = {},
    transpose: bool = False,
):
    """
    plots 1D arrays and there mean and automatically saves the plots, as well as returns it

    Parameters
    ----------
        values : 2D array
            axis 0 defines the position in the fiber and axis 1 the position in time, frequency or wl
            example : [[1, 2, 3], [0, 1, 0]] describes a quantity at 3 different freq/time and at two locations in the fiber
        plt_range : tupple (min, max, units)
            min, max : int or float
                minimum and maximum values given in the desired units
            units : function to convert from the desired units to rad/s or to time.
                    common functions are already defined in scgenerator.physics.units
                    look there for more details
        params : dict
            parameters of the simulations
        log : str {"1D"} or int, float or bool, optional
            str : plot in dB
                1D : takes the log for every slice
            int, float : plot in dB
                reference value
            bool : whether to use 1D variant or nothing
        spacing : int, float, optional
            tells the function to take one value every `spacing` one available. If a float is given, it will interpolate with a spline.
        vmin : float, optional
            min value of the colorbar
        vmax : float, optional
            max value of the colorbar
        ylabel : str, optional
            label of the y axis (x axis in transposed mode). Default is 'normalized intensity (dB)' for log plots
        yscaling : float, optional
            scale the y values by this amount
        renormalize : bool, optional
            if converting to wl scale, renormalize with to_WL function to ensure energy is conserved
        add_coherence : bool, optional
            whether to add a subplot with coherence
        file_type : str, optional
            usually pdf or png
        plt_name : str, optional
            special name to give to the plot. A name is automatically assigned anyway
        ax : matplotlib.axes._subplots.AxesSubplot object, optional
            axis on which to draw the plot
        line_labels : tupple(str), optional
            label of the lines. line_labels[0] is the label of the mean and line_labels[1] is the label of the indiv. values
        legend : bool, optional
            whether to draw the legend
        transpose : bool, optional
            transpose the plot
    returns
        fig, ax : matplotlib objects containing the plots
    example:
        if spectra is a (m, n, nt) array, one can plot a spectrum evolution as such:
        >>> fig, ax = plot_results_2D(spectra[:, -1], (600, 1600, nm), log=True, "Heidt2017")
    """

    if len(values.shape) != 2:
        print(f"Shape was {values.shape}. plot_avg can only plot 2D arrays")
        return

    is_spectrum, x_axis, plt_range = _prep_plot(values, plt_range, params)

    # crop and convert
    x_axis, ind, ext = units.sort_axis(x_axis, plt_range)
    if add_coherence:
        coherence = pulse.g12(values)
        coherence = coherence[ind]
    else:
        coherence = None
    values = values[:, ind]

    is_new_plot = ax is None
    folder_name = ""
    original_lines = []

    # compute the mean spectrum
    if is_spectrum:
        values = abs2(values)
    values *= yscaling
    mean_values = np.mean(values, axis=0)
    if plt_range[2].type == "WL" and renormalize:
        values = np.apply_along_axis(units.to_WL, 1, values, params.frep, x_axis)
        mean_values = units.to_WL(mean_values, params.frep, x_axis)

    # change the resolution
    if isinstance(spacing, float):
        new_x_axis = np.linspace(*span(x_axis), int(len(x_axis) / spacing))
        values = np.array(
            [UnivariateSpline(x_axis, value, k=4, s=0)(new_x_axis) for value in values]
        )
        if add_coherence:
            coherence = UnivariateSpline(x_axis, coherence, k=4, s=0)(new_x_axis)
        mean_values = np.mean(values, axis=0)
        x_axis = new_x_axis
    elif isinstance(spacing, int) and spacing > 1:
        values = values[:, ::spacing]
        mean_values = mean_values[::spacing]
        x_axis = x_axis[::spacing]
        if add_coherence:
            coherence = coherence[::spacing]

    # apply log transform if required
    if log != False:
        ylabel = "normalized intensity (dB)" if ylabel is None else ylabel
        vmax = defaults["vmax_with_headroom"] if vmax is None else vmax
        vmin = defaults["vmin"] if vmin is None else vmin
        if isinstance(log, (float, int)) and log != True:
            ref = log
        else:
            ref = np.max(mean_values)
        values = units.to_log(values, ref=ref)
        mean_values = units.to_log(mean_values, ref=ref)

    if is_new_plot:
        if add_coherence:
            mode = "coherence_T" if transpose else "coherence"
            out_path, fig, (top, bot) = plot_setup(
                out_path=Path(folder_name) / file_name, file_type=file_type, mode=mode
            )
        else:
            out_path, fig, top = plot_setup(
                out_path=Path(folder_name) / file_name, file_type=file_type
            )
            bot = top
    else:
        if isinstance(ax, (tuple, list)):
            top, bot = ax
            if transpose:
                bot.set_xlim(1.1, -0.1)
                bot.set_xlabel(r"|$g_{12}$|")
            else:
                bot.set_ylim(-0.1, 1.1)
                bot.set_ylabel(r"|$g_{12}$|")
        else:
            bot, top = ax, ax

        fig = top.get_figure()
        original_lines = top.get_lines()

    # Actual Plotting

    gray_style = defaults["muted_style"]
    highlighted_style = defaults["highlighted_style"]

    if transpose:
        for value in values:
            top.plot(value, x_axis, **gray_style)
        top.plot(mean_values, x_axis, **highlighted_style)
        if add_coherence:
            bot.plot(coherence, x_axis, c=defaults["color_cycle"][0])

        top.set_xlim(left=vmax, right=vmin)
        top.yaxis.tick_right()
        top.set_xlabel(ylabel)
        top.set_ylim(*ext)
        bot.yaxis.tick_right()
        bot.yaxis.set_label_position("right")
        bot.set_ylabel(plt_range[2].label)
        bot.set_ylim(*ext)
    else:
        for value in values:
            top.plot(x_axis, value, **gray_style)
        top.plot(x_axis, mean_values, **highlighted_style)
        if add_coherence:
            bot.plot(x_axis, coherence, c=defaults["color_cycle"][0])

        top.set_ylim(bottom=vmin, top=vmax)
        top.set_ylabel(ylabel)
        top.set_xlim(*ext)
        bot.set_xlabel(plt_range[2].label)
        bot.set_xlim(*ext)

    custom_lines = [
        plt.Line2D([0], [0], lw=2, c=gray_style["c"]),
        plt.Line2D([0], [0], lw=2, c=highlighted_style["c"]),
    ]
    line_labels = defaults["avg_line_labels"] if line_labels is None else line_labels
    line_labels = list(line_labels)

    if not is_new_plot:
        custom_lines += original_lines
        line_labels += [l.get_label() for l in original_lines]

    if legend:
        top.legend(custom_lines, line_labels, **legend_kwargs)

    if is_new_plot:
        fig.savefig(out_path, bbox_inches="tight", dpi=200)
        print(f"plot saved in {out_path}")

    if top is bot:
        return fig, top
    else:
        return fig, (top, bot)


def prepare_plot_1D(values, plt_range, x_axis, yscaling=1, spacing=1, frep=80e6):
    """prepares the values for plotting
    Parameters
    ----------
        values : array
            the values to plot.
            if complex, will take the abs^2
            if 2D, will consider it a as a list of values, each corresponding to the same x_axis
        plt_range : tupple (float, float, fct)
            fct as defined in scgenerator.physics.units
        x_axis : 1D array
            the corresponding x_axis
        yscaling : float, optional
            scale the y values by this amount
        spacing : int, float, optional
            tells the function to take one value every `spacing` one available. If a float is given, it will interpolate with a spline.
        frep : float
            used for conversion between frequency and wavelength if necessary
    Returns
    ----------
        new_x_axis : array
        new_values : array
    """
    is_spectrum = values.dtype == "complex"

    unique = len(values.shape) == 1
    values = np.atleast_2d(values)

    x_axis, ind, ext = units.sort_axis(x_axis, plt_range)

    if is_spectrum:
        values = abs2(values)
    values *= yscaling

    values = values[:, ind]

    if plt_range[2].type == "WL":
        values = np.apply_along_axis(units.to_WL, -1, values, frep, x_axis)

    if isinstance(spacing, float):
        new_x_axis = np.linspace(*span(x_axis), int(len(x_axis) / spacing))
        values = np.array(
            [UnivariateSpline(x_axis, value, k=4, s=0)(new_x_axis) for value in values]
        )
        x_axis = new_x_axis
    elif isinstance(spacing, int) and spacing > 1:
        values = values[:, ::spacing]
        x_axis = x_axis[::spacing]

    return x_axis, np.squeeze(values)


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
