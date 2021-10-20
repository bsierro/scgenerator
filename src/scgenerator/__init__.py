# flake8: noqa
from . import math
from .legacy import convert_sim_folder
from .math import abs2, argclosest, span
from .parameter import Configuration, Parameters
from .physics import fiber, materials, pulse, simulate, units
from .physics.simulate import RK4IP, parallel_RK4IP, run_simulation
from .physics.units import PlotRange
from .plotting import (
    get_extent,
    mean_values_plot,
    plot_spectrogram,
    propagation_plot,
    single_position_plot,
    transform_1D_values,
    transform_2D_propagation,
    transform_mean_values,
)
from .spectra import SimulationSeries, Spectrum
from .utils import Paths, _open_config, open_single_config
from .variationer import DescriptorDict, VariationDescriptor, Variationer, VariationSpecsError
