from pathlib import Path

import matplotlib.pyplot as plt

default_plotting = dict(
    figsize=(10, 7),
    interpolation_2D="bicubic",
    vmin=-40,
    vmax=0,
    vmax_with_headroom=2,
    out_path=Path("plot"),
    avg_main_to_coherence_ratio=4,
    avg_line_labels=["individual values", "mean"],
    muted_style=dict(linewidth=0.5, c=(0.8, 0.8, 0.8, 0.4)),
    highlighted_style=dict(c="red"),
    color_cycle=plt.rcParams["axes.prop_cycle"].by_key()["color"],
    light_color=(1, 1, 1, 0.7),
    markers=["*", "+", ".", "D", "x", "d", "v", "s", "1", "^"],
    cmap="viridis",
    label_quality_factor=r"$F_\mathrm{Q}$",
    label_mean_g12=r"$\langle | g_{12} |\rangle$",
    label_g12=r"|$g_{12}$|",
    label_z="propagation distance z (m)",
    label_fwhm=r"$T_\mathrm{FWHM}$ (fs)",
    label_wb_distance=r"$L_\mathrm{WB}$",
    label_t_jitter="timing jitter (fs)",
    label_fwhm_noise="FWHM noise (%)",
    label_int_noise="RIN (%)",
    text_topright_style=dict(verticalalignment="top", horizontalalignment="right"),
    text_topleft_style=dict(verticalalignment="top", horizontalalignment="left"),
)
