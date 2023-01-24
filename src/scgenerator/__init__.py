# flake8: noqa
from scgenerator import math, operators
from scgenerator.evaluator import Evaluator
from scgenerator.helpers import *
from scgenerator.legacy import convert_sim_folder
from scgenerator.math import abs2, argclosest, normalized, span, tspace, wspace
from scgenerator.parameter import FileConfiguration, Parameters
from scgenerator.physics import fiber, materials, plasma, pulse, simulate, units
from scgenerator.physics.simulate import RK4IP, parallel_RK4IP, run_simulation
from scgenerator.physics.units import PlotRange
from scgenerator.plotting import (
    get_extent,
    mean_values_plot,
    plot_spectrogram,
    propagation_plot,
    single_position_plot,
    transform_1D_values,
    transform_2D_propagation,
    transform_mean_values,
)
from scgenerator.spectra import SimulationSeries, Spectrum
from scgenerator.utils import Paths, _open_config, open_single_config, simulations_list
from scgenerator.variationer import (
    DescriptorDict,
    VariationDescriptor,
    Variationer,
    VariationSpecsError,
)
