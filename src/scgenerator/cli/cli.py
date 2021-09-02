import argparse
import os
import re
import subprocess
import sys
from collections import ChainMap
from pathlib import Path

import numpy as np

from .. import const, env, scripts, utils
from ..logger import get_logger
from ..physics.fiber import dispersion_coefficients
from ..physics.simulate import SequencialSimulations, run_simulation

try:
    import ray
except ImportError:
    ray = None


def set_env_variables(cmd_line_args: dict[str, str]):
    cm = ChainMap(cmd_line_args, os.environ)
    for env_key in env.global_config:
        k = env_key.replace(env.ENVIRON_KEY_BASE, "").lower()
        v = cm.get(k)
        if v is not None:
            os.environ[env_key] = str(v)


def create_parser():
    parser = argparse.ArgumentParser(description="scgenerator command", prog="scgenerator")
    subparsers = parser.add_subparsers(help="sub-command help")

    for key, args in env.global_config.items():
        names = ["--" + key.replace(env.ENVIRON_KEY_BASE, "").replace("_", "-").lower()]
        if "short_name" in args:
            names.append(args["short_name"])
        parser.add_argument(
            *names, **{k: v for k, v in args.items() if k not in {"short_name", "type"}}
        )
    parser.add_argument("--version", action="version", version=const.__version__)

    run_parser = subparsers.add_parser("run", help="run a simulation from a config file")
    run_parser.add_argument("config", help="path(s) to the toml configuration file(s)")
    run_parser.set_defaults(func=run_sim)

    merge_parser = subparsers.add_parser("merge", help="merge simulation results")
    merge_parser.add_argument(
        "path", help="path to the final simulation folder containing 'initial_config.toml'"
    )
    merge_parser.set_defaults(func=merge)

    plot_parser = subparsers.add_parser("plot", help="generate basic plots of a simulation")
    plot_parser.add_argument(
        "sim_dir",
        help="path to the root directory of the simulation (i.e. the "
        "directory directly containing 'initial_config0.toml'",
    )
    plot_parser.add_argument(
        "spectrum_limits",
        nargs=argparse.REMAINDER,
        help="comma-separated list of left limit, right limit and unit. "
        "One plot is made for each limit set provided. Example : 600,1200,nm or -2,2,ps",
    )
    plot_parser.add_argument("--options", "-o", default=None)
    plot_parser.add_argument(
        "--show", action="store_true", help="show the plots instead of saving them"
    )
    plot_parser.set_defaults(func=plot_all)

    dispersion_parser = subparsers.add_parser(
        "dispersion", help="show the dispersion of the given config"
    )
    dispersion_parser.add_argument("config", help="path to the config file")
    dispersion_parser.add_argument(
        "--limits", "-l", default=None, type=float, nargs=2, help="left and right limits in nm"
    )
    dispersion_parser.set_defaults(func=plot_dispersion)

    init_pulse_plot_parser = subparsers.add_parser(
        "plot-spec-field", help="plot the initial field and spectrum"
    )
    init_pulse_plot_parser.add_argument("config", help="path to the config file")
    init_pulse_plot_parser.add_argument(
        "--wavelength-limits",
        "-l",
        default=None,
        type=float,
        nargs=2,
        help="left and right limits in nm",
    )
    init_pulse_plot_parser.add_argument(
        "--time-limit", "-t", default=None, type=float, help="time axis limit in fs"
    )
    init_pulse_plot_parser.set_defaults(func=plot_init_field_spec)

    init_plot_parser = subparsers.add_parser("plot-init", help="plot initial values")
    init_plot_parser.add_argument("config", help="path to the config file")
    init_plot_parser.add_argument(
        "--dispersion-limits",
        "-d",
        default=None,
        type=float,
        nargs=2,
        help="left and right limits for dispersion plots in nm",
    )
    init_plot_parser.add_argument(
        "--time-limit", "-t", default=None, type=float, help="time axis limit in fs"
    )
    init_plot_parser.add_argument(
        "--wavelength-limits",
        "-l",
        default=None,
        nargs=2,
        type=float,
        help="wavelength axis limit in nm",
    )
    init_plot_parser.set_defaults(func=plot_init)

    convert_parser = subparsers.add_parser(
        "convert",
        help="convert parameter files that have been saved with an older version of the program",
    )
    convert_parser.add_argument("config", help="path to config/parameter file")
    convert_parser.set_defaults(func=translate_parameters)

    return parser


def main():
    parser = create_parser()
    args = parser.parse_args()

    set_env_variables({k: v for k, v in vars(args).items() if v is not None})

    args.func(args)

    logger = get_logger(__name__)
    logger.info(f"dispersion cache : {dispersion_coefficients.cache_info()}")


def run_sim(args):

    method = prep_ray()
    run_simulation(args.config, method=method)
    # if sys.platform == "darwin" and sys.stdout.isatty():
    #     subprocess.run(
    #         [
    #             "osascript",
    #             "-e",
    #             'tell app "System Events" to display dialog "simulation finished !"',
    #         ],
    #         stdout=subprocess.DEVNULL,
    #         stderr=subprocess.DEVNULL,
    #     )


def merge(args):
    path_trees = utils.build_path_trees(Path(args.path))

    output = env.output_path()
    if output is None:
        output = path_trees[0][-1][0].parent.name + " merged"

    utils.merge(output, path_trees)


def prep_ray():
    logger = get_logger(__name__)
    if ray:
        if env.get(env.START_RAY):
            init_str = ray.init()
        elif not env.get(env.NO_RAY):
            try:
                init_str = ray.init(
                    address="auto",
                    _redis_password=os.environ.get("redis_password", "caco1234"),
                )
                logger.info(init_str)
            except ConnectionError as e:
                logger.warning(e)
    return SequencialSimulations if env.get(env.NO_RAY) else None


def plot_all(args):
    opts = {}
    if args.options is not None:
        opts |= dict([o.split("=")[:2] for o in re.split("[, ]", args.options)])
    root = Path(args.sim_dir).resolve()
    scripts.plot_all(root, args.spectrum_limits, show=args.show, **opts)


def plot_init_field_spec(args):
    if args.wavelength_limits is None:
        l = None
    else:
        l = list(args.wavelength_limits)

    if args.time_limit is None:
        t = None
    else:
        t = [-args.time_limit, args.time_limit]

    scripts.plot_init_field_spec(args.config, t, l)


def plot_init(args):
    if args.wavelength_limits is None:
        l = None
    else:
        l = list(args.wavelength_limits)
    if args.dispersion_limits is None:
        d = None
    else:
        d = list(args.dispersion_limits)

    if args.time_limit is None:
        t = None
    else:
        t = [-args.time_limit, args.time_limit]

    scripts.plot_init(args.config, t, l, d)


def plot_dispersion(args):
    if args.limits is None:
        lims = None
    else:
        lims = 1e-9 * np.array(args.limits, dtype=float)
    scripts.plot_dispersion(args.config, lims)


def translate_parameters(args):
    path = args.config
    scripts.convert_params(path)


if __name__ == "__main__":
    main()
