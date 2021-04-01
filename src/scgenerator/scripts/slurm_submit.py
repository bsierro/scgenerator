import argparse
import os
import re
import shutil
import subprocess
from datetime import datetime, timedelta

from scgenerator.initialize import validate
from scgenerator.io import Paths, load_toml
from scgenerator.utils import count_variations


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
        "-c", "--cpus-per-node", required=True, type=int, help="number of cpus required per node"
    )
    parser.add_argument("-n", "--nodes", required=True, type=int, help="number of nodes required")
    parser.add_argument(
        "--environment-setup",
        required=False,
        default=f"source {os.path.expanduser('~/anaconda3/etc/profile.d/conda.sh')} && conda activate sc",
        help="commands to run to setup the environement (default : activate the sc environment with conda)",
    )
    return parser


def copy_starting_files():
    for name in ["start_worker", "start_head"]:
        path = Paths.get(name)
        file_name = os.path.split(path)[1]
        shutil.copy(path, file_name)
        mode = os.stat(file_name)
        os.chmod(file_name, 0o100 | mode.st_mode)


def main():
    parser = create_parser()
    template = Paths.gets("submit_job_template")
    args = parser.parse_args()

    if not re.match(r"^[0-9]{2}:[0-9]{2}:[0-9]{2}$", args.time) and not re.match(
        r"^[0-9]+$", args.time
    ):

        raise ValueError(
            "time format must be an integer number of minute or must match the pattern hh:mm:ss"
        )

    config = load_toml(args.config)
    config = validate(config)

    sim_num, _ = count_variations(config)

    file_name = "submit " + config["name"] + "-" + format(datetime.now(), "%Y%m%d%H%M") + ".sh"
    submit_sh = template.format(**vars(args))
    with open(file_name, "w") as file:
        file.write(submit_sh)
    subprocess.run(["sbatch", "--test-only", file_name])
    submit = input(
        f"Propagate {sim_num} pulses from config {args.config} with {args.cpus_per_node} cpus"
        + f" per node on {args.nodes} nodes for {format_time(args.time)} ? (y/[n])\n"
    )
    if submit.lower() in ["y", "yes"]:
        copy_starting_files()
        subprocess.run(["sbatch", file_name])
