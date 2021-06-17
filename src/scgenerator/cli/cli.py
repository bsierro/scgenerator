import argparse
import os
import random
from pathlib import Path
from collections import ChainMap

from ray.worker import get

from .. import io, env, const
from ..logger import get_logger
from ..physics.simulate import (
    SequencialSimulations,
    resume_simulations,
    run_simulation_sequence,
)
from ..physics.fiber import dispersion_coefficients
from pprint import pprint

try:
    import ray
except ImportError:
    ray = None


def set_env_variables(cmd_line_args: dict[str, str]):
    cm = ChainMap(cmd_line_args, os.environ)
    for env_key in const.global_config:
        k = env_key.replace(const.ENVIRON_KEY_BASE, "").lower()
        v = cm.get(k)
        if v is not None:
            os.environ[env_key] = str(v)


def create_parser():
    parser = argparse.ArgumentParser(description="scgenerator command", prog="scgenerator")
    subparsers = parser.add_subparsers(help="sub-command help")

    for key, args in const.global_config.items():
        names = ["--" + key.replace(const.ENVIRON_KEY_BASE, "").replace("_", "-").lower()]
        if "short_name" in args:
            names.append(args["short_name"])
        parser.add_argument(
            *names, **{k: v for k, v in args.items() if k not in {"short_name", "type"}}
        )

    run_parser = subparsers.add_parser("run", help="run a simulation from a config file")
    run_parser.add_argument("configs", help="path(s) to the toml configuration file(s)", nargs="+")
    run_parser.set_defaults(func=run_sim)

    resume_parser = subparsers.add_parser("resume", help="resume a simulation")
    resume_parser.add_argument(
        "sim_dir",
        help="path to the directory where the initial_config.toml and the partial data is stored",
    )
    resume_parser.add_argument(
        "configs",
        nargs="*",
        default=[],
        help="list of subsequent config files (excluding the resumed one)",
    )
    resume_parser.set_defaults(func=resume_sim)

    merge_parser = subparsers.add_parser("merge", help="merge simulation results")
    merge_parser.add_argument(
        "path", help="path to the final simulation folder containing 'initial_config.toml'"
    )
    merge_parser.set_defaults(func=merge)

    return parser


def main():
    parser = create_parser()
    args = parser.parse_args()

    set_env_variables({k: v for k, v in vars(args).items() if v is not None})

    args.func(args)

    logger = get_logger(__name__)
    logger.info(f"dispersion cache : {dispersion_coefficients.cache_info()}")


def run_sim(args):

    method = prep_ray(args)
    run_simulation_sequence(*args.configs, method=method)


def merge(args):
    path_trees = io.build_path_trees(Path(args.path))

    if args.output_name is None:
        args.output_name = path_trees[0][-1][0].parent.name + " merged"
    io.merge(args.output_name, path_trees)


def prep_ray(args):
    logger = get_logger(__name__)
    if ray:
        if env.get(const.START_RAY):
            init_str = ray.init()
        elif not env.get(const.NO_RAY):
            try:
                init_str = ray.init(
                    address="auto",
                    _redis_password=os.environ.get("redis_password", "caco1234"),
                )
                logger.info(init_str)
            except ConnectionError as e:
                logger.error(e)
    return SequencialSimulations if env.get(const.NO_RAY) else None


def resume_sim(args):

    method = prep_ray(args)
    sim = resume_simulations(Path(args.sim_dir), method=method)
    sim.run()
    run_simulation_sequence(*args.configs, method=method, prev_sim_dir=sim.sim_dir)


if __name__ == "__main__":
    main()
