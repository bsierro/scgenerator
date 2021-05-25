import argparse
import os
import random
import sys

import ray

from scgenerator import initialize
from ..physics.simulate import new_simulations, resume_simulations, SequencialSimulations
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

    run_parser = subparsers.add_parser("run", help="run a simulation from a config file")

    run_parser.add_argument("configs", help="path(s) to the toml configuration file(s)", nargs="+")
    run_parser.add_argument(
        "appendto",
        help="optional directory where a compatible simulation has already been ran",
        nargs="?",
        default=None,
    )
    run_parser.set_defaults(func=run_sim)

    resume_parser = subparsers.add_parser("resume", help="resume a simulation")
    resume_parser.add_argument(
        "data_dir",
        help="path to the directory where the initial_config.toml and the data is stored",
    )
    resume_parser.set_defaults(func=resume_sim)

    merge_parser = subparsers.add_parser("merge", help="merge simulation results")
    merge_parser.add_argument(
        "paths",
        nargs="+",
        help="path(s) to simulation folder(s) containing 'initial_config.toml'. If more "
        "than one path is given, simulations are appended to each other as if they're "
        "physically the continuation of the previous one.",
    )
    merge_parser.set_defaults(func=merge)

    return parser


def main():
    parser = create_parser()
    args = parser.parse_args()
    args.func(args)


def run_sim(args):

    method = prep_ray(args)
    configs = args.configs.copy()
    first_config = configs.pop(0)
    if args.appendto is None:
        sim = new_simulations(first_config, args.id, method=method)
    else:
        sim = new_simulations(
            first_config, args.id, data_folder=args.appendto, method=method, initial=False
        )
    sim.run()
    data_folders = [sim.data_folder]
    for config in configs:
        print("launching", config)
        sim = new_simulations(
            config, args.id, data_folder=data_folders[-1], method=method, initial=False
        )
        sim.run()
        data_folders.append(sim.data_folder)
    io.merge(data_folders)


def merge(args):
    io.merge(args.paths)


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
    sim = resume_simulations(args.data_dir, args.id, method=method)
    sim.run()


if __name__ == "__main__":
    main()
