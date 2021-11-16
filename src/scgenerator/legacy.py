import os
import sys
from collections.abc import MutableMapping
from pathlib import Path
from typing import Any, Set

import numpy as np
import tomli
import tomli_w

from .const import SPEC1_FN, SPEC1_FN_N, SPECN_FN1
from .parameter import Configuration, Parameters
from .pbar import PBars
from .utils import save_parameters
from .variationer import VariationDescriptor


def load_config(path: os.PathLike) -> dict[str, Any]:
    with open(path, "rb") as file:
        d = tomli.load(file)
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
    with open(new_root / "initial_config.toml", "wb") as f:
        tomli_w.dump(Parameters.strip_params_dict(master_config), f)
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
        save_parameters(new_params.dump_dict(), new_params.final_path)
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


def translate_parameters(d: dict[str, Any]) -> dict[str, Any]:
    """translate parameters name and value from older versions of the program

    Parameters
    ----------
    d : dict[str, Any]
        any parameter dictionary WITHOUT "variable" part

    Returns
    -------
    dict[str, Any]
        translated parameters
    """
    if {"variable", "Fiber"} & d.keys():
        raise ValueError(
            "The dict to translate should be a single parameter set "
            "(no 'variable' nor 'Fiber' entry)"
        )
    old_names = dict(
        interp_degree="interpolation_degree",
        beta="beta2_coefficients",
        interp_range="interpolation_range",
    )
    to_delete = ["dynamic_dispersion"]
    wl_limits_old = ["lower_wavelength_interp_limit", "upper_wavelength_interp_limit"]
    defaults_to_add = dict(repeat=1)
    new = {}
    if len(set(wl_limits_old) & d.keys()) == 2:
        new["interpolation_range"] = (d[wl_limits_old[0]], d[wl_limits_old[1]])
    for k, v in d.items():
        if k in to_delete:
            continue
        if k == "error_ok":
            new["tolerated_error" if d.get("adapt_step_size", True) else "step_size"] = v
        elif k == "behaviors":
            beh = d["behaviors"]
            if "raman" in beh:
                new["raman_type"] = d["raman_type"]
            new["spm"] = "spm" in beh
            new["self_steepening"] = "ss" in beh
        elif isinstance(v, MutableMapping):
            new[k] = translate_parameters(v)
        else:
            new[old_names.get(k, k)] = v
    return defaults_to_add | new


def main():
    convert_sim_folder(sys.argv[1])


if __name__ == "__main__":
    main()
