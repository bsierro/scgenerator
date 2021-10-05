import os
import sys
from pathlib import Path
from pprint import pprint
from typing import Any, Set

import numpy as np
import toml

from ..const import PARAM_FN, SPEC1_FN, SPEC1_FN_N, SPECN_FN1, Z_FN
from .parameter import Configuration, Parameters
from .utils import fiber_folder, save_parameters
from .pbar import PBars
from .variationer import VariationDescriptor, Variationer


def load_config(path: os.PathLike) -> dict[str, Any]:
    with open(path) as file:
        d = toml.load(file)
    d.setdefault("variable", {})
    return d


def load_config_sequence(path: os.PathLike) -> tuple[list[Path], list[dict[str, Any]]]:
    paths = sorted(list(Path(path).glob("initial_config*.toml")))
    return paths, [load_config(cfg) for cfg in paths]


def convert_sim_folder(path: os.PathLike):
    path = Path(path)
    config_paths, configs = load_config_sequence(path)
    master_config = dict(name=path.name, Fiber=configs)
    with open(path / "initial_config.toml", "w") as f:
        toml.dump(master_config, f, encoder=toml.TomlNumpyEncoder())
    configuration = Configuration(path / "initial_config.toml")
    new_fiber_paths: list[Path] = [
        path / fiber_folder(i, path.name, cfg["name"]) for i, cfg in enumerate(configs)
    ]
    for p in new_fiber_paths:
        p.mkdir(exist_ok=True)
    repeat = configs[0].get("repeat", 1)

    pbar = PBars(configuration.total_num_steps, "Converting")

    old_paths: dict[Path, VariationDescriptor] = {
        path / descr.branch.formatted_descriptor(): (descr, param.final_path)
        for descr, param in configuration
    }

    # create map from old to new path

    pprint(old_paths)
    quit()
    for p in old_paths:
        if not p.is_dir():
            raise FileNotFoundError(f"missing {p} from {path}")
    processed_paths: Set[Path] = set()
    for old_variation_path, descriptor in old_paths.items():  # fiberA=0, fiber B=0
        vary_parts = old_variation_path.name.split("fiber")[1:]
        identifiers = [
            "".join("fiber" + el for el in vary_parts[: i + 1]).strip()
            for i in range(len(vary_parts))
        ]
        cum_z_num = 0
        for i, (fiber_path, new_identifier) in enumerate(zip(new_fiber_paths, identifiers)):
            config = descriptor.update_config(configs[i], i)
            new_variation_path = fiber_path / new_identifier
            z_num = config["z_num"]
            move = new_variation_path not in processed_paths
            os.makedirs(new_variation_path, exist_ok=True)
            processed_paths.add(new_variation_path)

            for spec_num in range(cum_z_num, cum_z_num + z_num):
                old_spec = old_variation_path / SPECN_FN1.format(spec_num)
                if move:
                    spec_data = np.load(old_spec)
                    for j, spec1 in enumerate(spec_data):
                        if j == 0:
                            np.save(
                                new_variation_path / SPEC1_FN.format(spec_num - cum_z_num), spec1
                            )
                        else:
                            np.save(
                                new_variation_path / SPEC1_FN_N.format(spec_num - cum_z_num, j),
                                spec1,
                            )
                        pbar.update()
                else:
                    pbar.update(value=repeat)
                old_spec.unlink()
            if move:
                if i > 0:
                    config["prev_data_dir"] = str(
                        (new_fiber_paths[i - 1] / identifiers[i - 1]).resolve()
                    )
                params = Parameters(**config)
                params.compute()
                save_parameters(params.prepare_for_dump(), new_variation_path)
            cum_z_num += z_num
        (old_variation_path / PARAM_FN).unlink()
        (old_variation_path / Z_FN).unlink()
        old_variation_path.rmdir()

    for cp in config_paths:
        cp.unlink()


def main():
    convert_sim_folder(sys.argv[1])


if __name__ == "__main__":
    main()
