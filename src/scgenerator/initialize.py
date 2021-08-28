import os
from collections import defaultdict
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterator, Union

import numpy as np

from . import utils
from .errors import *
from .logger import get_logger
from .utils.parameter import (
    Config,
    Parameters,
    override_config,
    required_simulations,
)
from scgenerator.utils import parameter


class ParamSequence:
    def __init__(self, config_dict: Union[dict[str, Any], os.PathLike, Config]):
        """creates a param sequence from a base config

        Parameters
        ----------
        config_dict : Union[dict[str, Any], os.PathLike, BareConfig]
            Can be either a dictionary, a path to a config toml file or BareConfig obj
        """
        if isinstance(config_dict, Config):
            self.config = config_dict
        elif isinstance(config_dict, Config):
            self.config = Config.from_bare(config_dict)
        else:
            if not isinstance(config_dict, Mapping):
                config_dict = utils.load_toml(config_dict)
            self.config = Config(**config_dict)
        self.name = self.config.name
        self.logger = get_logger(__name__)

        self.update_num_sim()

    def __iter__(self) -> Iterator[tuple[list[tuple[str, Any]], Parameters]]:
        """iterates through all possible parameters, yielding a config as well as a flattened
        computed parameters set each time"""
        for variable_list, params in required_simulations(self.config):
            yield variable_list, params

    def __len__(self):
        return self.num_sim

    def __repr__(self) -> str:
        return f"dispatcher generated from config {self.name}"

    def update_num_sim(self):
        num_sim = self.count_variations()
        self.num_sim = num_sim
        self.num_steps = self.num_sim * self.config.z_num
        self.single_sim = self.num_sim == 1

    def count_variations(self) -> int:
        return count_variations(self.config)


class ContinuationParamSequence(ParamSequence):
    def __init__(self, prev_sim_dir: os.PathLike, new_config: Config):
        """Parameter sequence that builds on a previous simulation but with a new configuration
        It is recommended that only the fiber and the number of points stored may be changed and
        changing other parameters could results in unexpected behaviors. The new config doesn't have to
        be a full configuration (i.e. you can specify only the parameters that change).

        Parameters
        ----------
        prev_sim_dir : PathLike
            path to the folder of the previous simulation containing 'initial_config.toml'
        new_config : dict[str, Any]
            new config
        """
        self.prev_sim_dir = Path(prev_sim_dir)
        self.bare_configs = Config.load_sequence(new_config.previous_config_file)
        self.bare_configs.append(new_config)
        self.bare_configs[0].check_validity()
        final_config = parameter.final_config_from_sequence(*self.bare_configs)
        super().__init__(final_config)

    def __iter__(self) -> Iterator[tuple[list[tuple[str, Any]], Parameters]]:
        """iterates through all possible parameters, yielding a config as well as a flattened
        computed parameters set each time"""
        for variable_list, params in required_simulations(*self.bare_configs):
            prev_data_dir = self.find_prev_data_dirs(variable_list)[0]
            params.prev_data_dir = str(prev_data_dir.resolve())
            yield variable_list, params

    def find_prev_data_dirs(self, new_variable_list: list[tuple[str, Any]]) -> list[Path]:
        """finds the previous simulation data that this new config should start from

        Parameters
        ----------
        new_variable_list : list[tuple[str, Any]]
            as yielded by required_simulations

        Returns
        -------
        Path
            path to the data folder

        Raises
        ------
        ValueError
            no data folder found
        """
        new_target = set(parameter.format_variable_list(new_variable_list).split()[2:])
        path_dic = defaultdict(list)
        max_in_common = 0
        for data_dir in self.prev_sim_dir.glob("id*"):
            candidate = set(data_dir.name.split()[2:])
            in_common = candidate & new_target
            num_in_common = len(in_common)
            max_in_common = max(num_in_common, max_in_common)
            path_dic[num_in_common].append(data_dir)

        return path_dic[max_in_common]

    def count_variations(self) -> int:
        return count_variations(*self.bare_configs)


def count_variations(*bare_configs: Config) -> int:
    sim_num = 1
    for conf in bare_configs:
        for l in conf.variable.values():
            sim_num *= len(l)
    return sim_num * (bare_configs[0].repeat or 1)


class RecoveryParamSequence(ParamSequence):
    def __init__(self, config_dict, task_id):
        super().__init__(config_dict)
        self.id = task_id
        self.num_steps = 0

        self.prev_sim_dir = None
        if self.config.prev_sim_dir is not None:
            self.prev_sim_dir = Path(self.config.prev_sim_dir)
            init_config = Config.load(self.prev_sim_dir / "initial_config.toml")
            self.prev_variable_lists = [
                (
                    set(variable_list[1:]),
                    self.prev_sim_dir / parameter.format_variable_list(variable_list),
                )
                for variable_list, _ in required_simulations(init_config)
            ]
            additional_sims_factor = int(
                np.prod(
                    [
                        len(init_config.variable[k])
                        for k in (self.config.variable.keys() & init_config.variable.keys())
                        if init_config.variable[k] != self.config.variable[k]
                    ]
                )
            )
            self.update_num_sim(self.num_sim * additional_sims_factor)
        not_started = self.num_sim
        sub_folders = utils.get_data_dirs(utils.get_sim_dir(self.id))

        for sub_folder in utils.PBars(
            sub_folders, "Initial recovery", head_kwargs=dict(unit="sim")
        ):
            num_left = utils.num_left_to_propagate(sub_folder, self.config.z_num)
            if num_left == 0:
                self.num_sim -= 1
            self.num_steps += num_left
            not_started -= 1

        self.num_steps += not_started * self.config.z_num
        self.single_sim = self.num_sim == 1

    def __iter__(self) -> Iterator[tuple[list[tuple[str, Any]], Parameters]]:
        for variable_list, params in required_simulations(self.config):

            data_dir = utils.get_sim_dir(self.id) / parameter.format_variable_list(variable_list)

            if not data_dir.is_dir() or utils.find_last_spectrum_num(data_dir) == 0:
                if (prev_data_dir := self.find_prev_data_dirs(variable_list)) is not None:
                    params.prev_data_dir = str(prev_data_dir)
                yield variable_list, params
            elif utils.num_left_to_propagate(data_dir, self.config.z_num) != 0:
                yield variable_list, params + "Needs to rethink recovery procedure"
            else:
                continue

    def find_prev_data_dirs(self, new_variable_list: list[tuple[str, Any]]) -> list[Path]:
        """finds the previous simulation data that this new config should start from

        Parameters
        ----------
        new_variable_list : list[tuple[str, Any]]
            as yielded by required_simulations

        Returns
        -------
        Path
            path to the data folder

        Raises
        ------
        ValueError
            no data folder found
        """
        new_set = set(new_variable_list[1:])
        path_dic = defaultdict(list)
        max_in_common = 0
        for stored_set, path in self.prev_variable_lists:
            in_common = stored_set & new_set
            num_in_common = len(in_common)
            max_in_common = max(num_in_common, max_in_common)
            path_dic[num_in_common].append(path)

        return path_dic[max_in_common]


def validate_config_sequence(*configs: os.PathLike) -> tuple[str, int]:
    """validates a sequence of configs where all but the first one may have
    parameters missing

    Parameters
    ----------
    configs : os.PathLike
        sequence of paths to toml config files. The first element may be a folder containing data intead

    Returns
    -------
    int
        total number of simulations
    """

    previous = None
    configs = Config.load_sequence(*configs)
    for config in configs:
        # if (p := Path(config)).is_dir():
        #     config = p / "initial_config.toml"
        new_conf = config
        previous = Config.from_bare(override_config(new_conf, previous))
    return previous.name, count_variations(*configs)
