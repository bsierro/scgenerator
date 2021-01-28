import matplotlib.pyplot as plt

from . import utilities as util

"""
This File is used as a public global variable storage solutions. Main functions are having a centralised
    parameters index
    plotting parameters
    logger
This is not a solution when functions are accessing this module in parallel threads, as 
changes made by a thread are not reflected in other threads, which is why another solution should be used when dealing
with parameters or paths (if those paths are not stored in paths.json)
"""


class CurrentLogger:
    _current_logger = None

    @classmethod
    def focus_logger(cls, logger):
        cls._current_logger = logger

    @classmethod
    def log(cls, *args, **kwargs):
        if cls._current_logger is not None:
            util.ray_safe(cls._current_logger.log, *args, **kwargs)
        else:
            print(*args)


# WILL BREAK SIMULATION SAVING AND MERGING IF CHANGED
recorded_types = ["spectra", "params"]  # nickname of the objects saved and tracked when doing automatic simulations


# --------SIMULATION VARIABLES--------#
default_z_target_size = 128


# ---------PLOTTING VARIABLES---------#


def plot_arrowstyle(direction=1, color="white"):
    return dict(
        arrowprops=dict(arrowstyle="->", connectionstyle=f"arc3,rad={direction*0.3}", color=color),
        color=color,
        backgroundcolor=(0.5, 0.5, 0.5, 0.5),
    )


plot_default_figsize = (10, 7)
plot_default_2D_interpolation = "bicubic"
plot_default_vmin = -40
plot_default_vmax = 0
plot_default_vmax_with_headroom = 2
plot_default_name = "plot"

plot_avg_default_main_to_coherence_ratio = 4
plot_avg_default_line_labels = ["individual values", "mean"]

plot_muted_style = dict(linewidth=0.5, c=(0.8, 0.8, 0.8, 0.4))
plot_highlighted_style = dict(c="red")
plot_default_color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
plot_default_light_color = (1, 1, 1, 0.7)
plot_default_markers = ["*", "+", ".", "D", "x", "d", "v", "s", "1", "^"]

plot_default_cmap = "viridis"

plot_label_quality_factor = r"$F_\mathrm{Q}$"
plot_label_mean_g12 = r"$\langle | g_{12} |\rangle$"
plot_label_g12 = r"|$g_{12}$|"
plot_label_z = "propagation distance z (m)"
plot_label_fwhm = r"$T_\mathrm{FWHM}$ (fs)"
plot_label_wb_distance = r"$L_\mathrm{WB}$"
plot_label_t_jitter = "timing jitter (fs)"
plot_label_fwhm_noise = "FWHM noise (%)"
plot_label_int_noise = "RIN (%)"

plot_text_topright_style = dict(verticalalignment="top", horizontalalignment="right")
plot_text_topleft_style = dict(verticalalignment="top", horizontalalignment="left")

# ------------------------------------#

# plotting choices
dist = 1.5

# plotting variables


def style(k):
    return dict(
        marker=plot_default_markers[k], markerfacecolor="none", linestyle=":", lw=1, c=plot_default_color_cycle[k]
    )


default_width = 10


def fs(ratio):
    return (default_width, default_width * ratio)


# store global variables for debug purposes
_DEBUG = {}
