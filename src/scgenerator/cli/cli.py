import argparse
import os
import random
import sys

import ray

from scgenerator import initialize
from ..physics.simulate import run_simulation_sequence, resume_simulations, SequencialSimulations
from .. import io


def create_parser():
    parser = argparse.ArgumentParser(description="scgenerator command", prog="scgenerator")

    subparsers = parser.add_subparsers(help="sub-command help")

    parser.add_argument(
        "--id",
        type=int,
        default=random.randint(0, 1e18),
        help="Unique id of the session. Only useful when running several processes at the same time.",
    )
    parser.add_argument(
        "--start-ray",
        action="store_true",
        help="assume no ray instance has been started beforehand",
    )

    parser.add_argument(
        "--no-ray",
        action="store_true",
        help="force not to use ray",
    )
    parser.add_argument("--output-name", "-o", help="path to the final output folder", default=None)

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
    args.func(args)


def run_sim(args):

    method = prep_ray(args)
    run_simulation_sequence(*args.configs, method=method, final_name=args.output_name)


def merge(args):
    io.append_and_merge(args.path, args.output_name)


def prep_ray(args):
    if args.start_ray:
        init_str = ray.init()
    elif not args.no_ray:
        try:
            init_str = ray.init(
                address="auto",
                # _node_ip_address=os.environ.get("ip_head", "127.0.0.1").split(":")[0],
                _redis_password=os.environ.get("redis_password", "caco1234"),
            )
            print(init_str)
        except ConnectionError:
            pass
    return SequencialSimulations if args.no_ray else None


def resume_sim(args):

    method = prep_ray(args)
    sim = resume_simulations(args.sim_dir, method=method)
    sim.run()
    run_simulation_sequence(
        *args.configs, method=method, prev_sim_dir=sim.data_folder, final_name=args.output_name
    )


if __name__ == "__main__":
    main()
