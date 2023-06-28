import numpy as np
import scgenerator.math as math
import scgenerator.physics.units as units


def normalize_range(
    axis: np.ndarray, _range: tuple | units.PlotRange | None, num: int
) -> tuple[units.PlotRange, np.ndarray]:
    if _range is None:
        _range = units.PlotRange(axis.min(), axis.max(), units.no_unit)
    elif not isinstance(_range, units.PlotRange):
        _range = units.PlotRange(*_range)
    new_axis = np.linspace(_range[0], _range[1], num)
    return _range, new_axis


def prop_2d(
    values: np.ndarray,
    h_axis: np.ndarray,
    v_axis: np.ndarray,
    h_range: tuple | units.PlotRange | None = None,
    v_range: tuple | units.PlotRange | None = None,
    h_num: int = 1024,
    v_num: int = 1024,
    z_lim: tuple[float, float] | None = None,
):
    if values.ndim != 2:
        raise TypeError("prop_2d can only transform 2d data")
    if np.iscomplexobj(values):
        values = math.abs2(values)

    horizontal_range, horizontal = normalize_range(h_axis, h_range, h_num)
    vertical_range, vertical = normalize_range(v_axis, v_range, v_num)

    values = math.interp_2d(
        h_axis, v_axis, values, horizontal_range.unit(horizontal), vertical_range.unit(vertical)
    )

    if horizontal_range.must_correct_wl:
        values = np.apply_along_axis(
            lambda x: units.to_WL(x, horizontal_range.unit.to.m(horizontal)), 1, values
        )
    elif vertical_range.must_correct_wl:
        values = np.apply_along_axis(
            lambda x: units.to_WL(x, vertical_range.unit.to.m(vertical)), 0, values
        )

    return horizontal, vertical, values
