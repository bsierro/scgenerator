import argparse
import os
import re
import shutil
import subprocess
from datetime import datetime, timedelta
from pathlib import Path
from typing import Tuple

import numpy as np

from ..utils import Paths
from ..utils.parameter import Configuration


def primes(n):
    prime_factors = []
    d = 2
    while d * d <= n:
        while (n % d) == 0:
            prime_factors.append(d)
            n //= d
        d += 1
    if n > 1:
        prime_factors.append(n)
    return prime_factors


def balance(n, lim=(32, 32)):
    factors = primes(n)
    if len(factors) == 1:
        factors = primes(n + 1)
    a, b, x, y = 1, 1, 1, 1
    while len(factors) > 0 and x <= lim[0] and y <= lim[1]:
        a = x
        b = y
        if y >= x:
            x *= factors.pop(0)
        else:
            y *= factors.pop()
    return a, b


def distribute(
    num: int, nodes: int = None, cpus_per_node: int = None, lim=(16, 32)
) -> Tuple[int, int]:
    if nodes is None and cpus_per_node is None:
        balanced = balance(num, lim)
        if num > max(lim):
            while np.product(balanced) < min(lim):
                num += 1
                balanced = balance(num, lim)
        nodes = min(balanced)
        cpus_per_node = max(balanced)

    elif nodes is None:
        nodes = num // cpus_per_node
        while nodes > lim[0]:
            nodes //= 2
    elif cpus_per_node is None:
        cpus_per_node = num // nodes
        while cpus_per_node > lim[1]:
            cpus_per_node //= 2
    return nodes, cpus_per_node


def format_time(t):
    try:
        t = float(t)
        return timedelta(minutes=t)
    except ValueError:
        return t


def create_parser():
    parser = argparse.ArgumentParser(description="submit a job to a slurm cluster")
    parser.add_argument("config", help="path to the toml configuration file")
    parser.add_argument(
        "-t", "--time", required=True, type=str, help="time required for the job in hh:mm:ss"
    )
    parser.add_argument(
        "-c", "--cpus-per-node", default=None, type=int, help="number of cpus required per node"
    )
    parser.add_argument("-n", "--nodes", default=None, type=int, help="number of nodes required")
    parser.add_argument(
        "--environment-setup",
        required=False,
        default=f"source {os.path.expanduser('~/anaconda3/etc/profile.d/conda.sh')} && conda activate sc && "
        "export SCGENERATOR_PBAR_POLICY=file && export SCGENERATOR_LOG_PRINT_LEVEL=none && export SCGENERATOR_LOG_FILE_LEVEL=info",
        help="commands to run to setup the environement (default : activate the sc environment with conda)",
    )
    parser.add_argument(
        "--command", default="run", choices=["run", "resume", "merge"], help="command to run"
    )
    parser.add_argument("--dependency", default=None, help="sbatch dependency argument")
    return parser


def copy_starting_files():
    for name in ["start_worker", "start_head"]:
        path = Paths.get(name)
        file_name = os.path.split(path)[1]
        shutil.copy(path, file_name)
        mode = os.stat(file_name)
        os.chmod(file_name, 0o100 | mode.st_mode)


def main():

    command_map = dict(run="Propagate", resume="Resuming", merge="Merging")

    parser = create_parser()
    template = Paths.gets("submit_job_template")
    args = parser.parse_args()

    if args.dependency is None:
        args.dependency = ""
    else:
        args.dependency = f"#SBATCH --dependency={args.dependency}"

    if not re.match(r"^[0-9]{2}:[0-9]{2}:[0-9]{2}$", args.time) and not re.match(
        r"^[0-9]+$", args.time
    ):

        raise ValueError(
            "time format must be an integer number of minute or must match the pattern hh:mm:ss"
        )

    config = Configuration.load(args.config)
    final_name = config.name
    sim_num = config.num_sim

    if args.command == "merge":
        args.nodes = 1
        args.cpus_per_node = 1
    else:
        args.nodes, args.cpus_per_node = distribute(config.num_sim, args.nodes, args.cpus_per_node)

    submit_path = Path(
        "submit " + final_name.replace("/", "") + "-" + format(datetime.now(), "%Y%m%d%H%M") + ".sh"
    )
    tmp_path = Path("submit tmp.sh")

    job_name = f"supercontinuum {final_name}"
    submit_sh = template.format(
        job_name=job_name, configs_list=" ".join(f'"{c}"' for c in args.config), **vars(args)
    )

    tmp_path.write_text(submit_sh)
    subprocess.run(["sbatch", "--test-only", str(tmp_path)])
    submit = input(
        f"{command_map[args.command]} {sim_num} pulses from config {args.config} with {args.cpus_per_node} cpus"
        + f" per node on {args.nodes} nodes for {format_time(args.time)} ? (y/[n])\n"
    )
    if submit.lower() in ["y", "yes"]:
        submit_path.write_text(submit_sh)
        copy_starting_files()
        subprocess.run(["sbatch", submit_path])
    tmp_path.unlink()
