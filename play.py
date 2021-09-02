from typing import Any, Generator
import scgenerator as sc
import itertools

import numpy as np


class DataPather:
    def __init__(self, dl: list[dict[str, Any]]):
        self.dict_list = dl
        self.n = len(self.dict_list)
        self.final_list = list(self.dico_iterator(self.n))

    def dico_iterator(self, index: int) -> Generator[list[list[tuple[str, Any]]], None, None]:
        d_tem_list = [el for d in self.dict_list[: index + 1] for el in d.items()]
        dict_pos = np.cumsum([0] + [len(d) for d in self.dict_list[: index + 1]])
        ranges = [range(len(l)) for _, l in d_tem_list]

        for r in itertools.product(*ranges):
            flat = [(d_tem_list[i][0], d_tem_list[i][1][j]) for i, j in enumerate(r)]
            out = [flat[left:right] for left, right in zip(dict_pos[:-1], dict_pos[1:])]
            yield out

    def all_vary_list(self, index):
        for l in self.dico_iterator(index):
            yield sc.utils.parameter.format_variable_list(
                sc.utils.parameter.reduce_all_variable(l[:index])
            ), sc.utils.parameter.format_variable_list(
                sc.utils.parameter.reduce_all_variable(l)
            ), l[
                index
            ]


configs, name = sc.utils.load_config_sequence(
    "/Users/benoitsierro/Nextcloud/PhD/Supercontinuum/PCF Simulations/Test/NewStyle.toml"
)

dp = DataPather([config["variable"] for config in configs])
# pprint(list(dp.dico_iterator(1)))
for i in range(3):
    for prev_path, this_path, this_vary in dp.all_vary_list(i):
        print(prev_path)
        print(this_path)
        print(this_vary)
        print()
    print()
