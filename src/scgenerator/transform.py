import numpy as np

import scgenerator.math as math
import scgenerator.physics.units as units


def prop_2d(
    values: np.ndarray,
    h_axis: np.ndarray,
    v_axis: np.ndarray,
    horizontal_range: tuple | units.PlotRange | None,
    vertical_range: tuple | units.PlotRange | None,
    h_num: int = 1024,
    v_num: int = 1024,
    z_lim: tuple[float, float] | None = None,
):
    if values.ndim != 2:
        raise TypeError("prop_2d can only transform 2d data")
    
    if horizontal_range is None:
        horizontal_range = units.PlotRange(h_axis.min(), h_axis.max(), units.no_unit)
    elif not isinstance(horizontal_range, units.PlotRange):
        horizontal_range = units.PlotRange(*horizontal_range)

    if vertical_range is None:
        vertical_range = units.PlotRange(h_axis.min(), h_axis.max(), units.no_unit)
    elif not isinstance(vertical_range, units.PlotRange):
        vertical_range = units.PlotRange(*vertical_range)

    if np.iscomplex(values):
        values = math.abs2(values)

    horizontal = np.linspace(horizontal_range[0], horizontal_range[1], h_num)
    vertical = np.linspace(vertical_range[0], vertical_range[1], v_num)

    values = math.interp_2d(h_axis, v_axis, values, horizontal_range.unit(horizontal), vertical_range.unit(vertical))
    return horizontal, vertical, values

