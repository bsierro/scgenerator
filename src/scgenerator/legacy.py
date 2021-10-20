from genericpath import exists
import os
import sys
from pathlib import Path
from typing import Any, Set

import numpy as np
import toml

from .const import SPEC1_FN, SPEC1_FN_N, SPECN_FN1
from .parameter import Configuration, Parameters
from .utils import save_parameters
from .pbar import PBars
from .variationer import VariationDescriptor


def load_config(path: os.PathLike) -> dict[str, Any]:
    with open(path) as file:
        d = toml.load(file)
    d.setdefault("variable", {})
    return d


def load_config_sequence(path: os.PathLike) -> tuple[list[Path], list[dict[str, Any]]]:
    paths = sorted(list(Path(path).glob("initial_config*.toml")))
    confs = [load_config(cfg) for cfg in paths]
    repeat = None
    for c in confs:
        nums = c["variable"].pop("num", None)
        if nums is not None:
            repeat = len(nums)
    if repeat is not None:
        confs[0]["repeat"] = repeat
    return paths, confs


def convert_sim_folder(path: os.PathLike):
    path = Path(path).resolve()
    new_root = path.parent / "sc_legagy_converter" / path.name
    os.makedirs(new_root, exist_ok=True)
    _, configs = load_config_sequence(path)
    master_config = dict(name=path.name, Fiber=configs)
    with open(new_root / "initial_config.toml", "w") as f:
        toml.dump(master_config, f, encoder=toml.TomlNumpyEncoder())
    configuration = Configuration(path, final_output_path=new_root)
    pbar = PBars(configuration.total_num_steps, "Converting")

    new_paths: dict[VariationDescriptor, Parameters] = dict(configuration)
    old_paths: Set[Path] = set()
    old2new: list[tuple[Path, VariationDescriptor, Parameters, tuple[int, int]]] = []
    for descriptor, params in configuration.iterate_single_fiber(-1):
        old_path = path / descriptor.branch.formatted_descriptor()
        if old_path in old_paths:
            continue
        if not Path(old_path).is_dir():
            raise FileNotFoundError(f"missing {old_path} from {path}. Aborting.")
        old_paths.add(old_path)
        for d in descriptor.iter_parents():
            z_num_start = sum(c["z_num"] for c in configs[: d.num_fibers - 1])
            z_limits = (z_num_start, z_num_start + params.z_num)
            old2new.append((old_path, d, new_paths[d], z_limits))

    processed_specs: Set[VariationDescriptor] = set()

    for old_path, descr, new_params, (start_z, end_z) in old2new:
        move_specs = descr not in processed_specs
        processed_specs.add(descr)
        if (parent := descr.parent) is not None:
            new_params.prev_data_dir = str(new_paths[parent].final_path)
        save_parameters(new_params.prepare_for_dump(), new_params.final_path)
        for spec_num in range(start_z, end_z):
            old_spec = old_path / SPECN_FN1.format(spec_num)
            if move_specs:
                _mv_specs(pbar, new_params, start_z, spec_num, old_spec)
    pbar.close()


def _mv_specs(pbar: PBars, new_params: Parameters, start_z: int, spec_num: int, old_spec: Path):
    os.makedirs(new_params.final_path, exist_ok=True)
    spec_data = np.load(old_spec)
    for j, spec1 in enumerate(spec_data):
        if j == 0:
            new_path = new_params.final_path / SPEC1_FN.format(spec_num - start_z)
        else:
            new_path = new_params.final_path / SPEC1_FN_N.format(spec_num - start_z, j)
        np.save(new_path, spec1)
        pbar.update()


def main():
    convert_sim_folder(sys.argv[1])


if __name__ == "__main__":
    main()