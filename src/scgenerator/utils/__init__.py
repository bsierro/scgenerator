"""
This files includes utility functions designed more or less to be used specifically with the
scgenerator module but some function may be used in any python program

"""

import itertools
import multiprocessing
import os
import random
import re
import threading
from collections import abc
from copy import deepcopy
from dataclasses import asdict, replace
from io import StringIO
from pathlib import Path
from typing import Any, Iterable, Iterator, TypeVar, Union

import numpy as np
from numpy.lib.arraysetops import isin
from tqdm import tqdm

from .. import env
from ..const import PARAM_SEPARATOR
from ..math import *
from .parameter import BareConfig, BareParams

T_ = TypeVar("T_")


class PBars:
    def __init__(
        self,
        task: Union[int, Iterable[T_]],
        desc: str,
        num_sub_bars: int = 0,
        head_kwargs=None,
        worker_kwargs=None,
    ) -> "PBars":

        self.id = random.randint(100000, 999999)
        try:
            self.width = os.get_terminal_size().columns
        except OSError:
            self.width = 80
        if isinstance(task, abc.Iterable):
            self.iterator: Iterable[T_] = iter(task)
            self.num_tot: int = len(task)
        else:
            self.num_tot: int = task
            self.iterator = None

        self.policy = env.pbar_policy()
        if head_kwargs is None:
            head_kwargs = dict()
        if worker_kwargs is None:
            worker_kwargs = dict(
                total=1,
                desc="Worker {worker_id}",
                bar_format="{l_bar}{bar}" "|[{elapsed}<{remaining}, " "{rate_fmt}{postfix}]",
            )
        if "print" not in env.pbar_policy():
            head_kwargs["file"] = worker_kwargs["file"] = StringIO()
            self.width = 80
        head_kwargs["desc"] = desc
        self.pbars = [tqdm(total=self.num_tot, ncols=self.width, ascii=False, **head_kwargs)]
        for i in range(1, num_sub_bars + 1):
            kwargs = {k: v for k, v in worker_kwargs.items()}
            if "desc" in kwargs:
                kwargs["desc"] = kwargs["desc"].format(worker_id=i)
            self.append(tqdm(position=i, ncols=self.width, ascii=False, **kwargs))
        self.print_path = Path(
            f"progress {self.pbars[0].desc.replace('/', '')} {self.id}"
        ).resolve()
        self.close_ev = threading.Event()
        if "file" in self.policy:
            self.thread = threading.Thread(target=self.print_worker, daemon=True)
            self.thread.start()

    def print(self):
        if "file" not in self.policy:
            return
        s = []
        for pbar in self.pbars:
            s.append(str(pbar))
        self.print_path.write_text("\n".join(s))

    def print_worker(self):
        while True:
            if self.close_ev.wait(2.0):
                return
            self.print()

    def __iter__(self):
        with self as pb:
            for thing in self.iterator:
                yield thing
                pb.update()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def __getitem__(self, key):
        return self.pbars[key]

    def update(self, i=None, value=1):
        if i is None:
            for pbar in self.pbars[1:]:
                pbar.update(value)
        elif i > 0:
            self.pbars[i].update(value)
        self.pbars[0].update()

    def append(self, pbar: tqdm):
        self.pbars.append(pbar)

    def reset(self, i):
        self.pbars[i].update(-self.pbars[i].n)
        self.print()

    def close(self):
        self.print()
        self.close_ev.set()
        if "file" in self.policy:
            self.thread.join()
        for pbar in self.pbars:
            pbar.close()


class ProgressBarActor:
    def __init__(self, name: str, num_workers: int, num_steps: int) -> None:
        self.counters = [0 for _ in range(num_workers + 1)]
        self.p_bars = PBars(
            num_steps, "Simulating " + name, num_workers, head_kwargs=dict(unit="step")
        )

    def update(self, worker_id: int, rel_pos: float = None) -> None:
        """update a counter

        Parameters
        ----------
        worker_id : int
            id of the worker
        rel_pos : float, optional
            if None, increase the counter by one, if set, will set
            the counter to the specified value (instead of incrementing it), by default None
        """
        if rel_pos is None:
            self.counters[worker_id] += 1
        else:
            self.counters[worker_id] = rel_pos

    def update_pbars(self):
        for counter, pbar in zip(self.counters, self.p_bars.pbars):
            pbar.update(counter - pbar.n)

    def close(self):
        self.p_bars.close()


def progress_worker(
    name: str, num_workers: int, num_steps: int, progress_queue: multiprocessing.Queue
):
    """keeps track of progress on a separate thread

    Parameters
    ----------
    num_steps : int
        total number of steps, used for the main progress bar (position 0)
    progress_queue : multiprocessing.Queue
        values are either
            Literal[0] : stop the worker and close the progress bars
            tuple[int, float] : worker id and relative progress between 0 and 1
    """
    with PBars(
        num_steps, "Simulating " + name, num_workers, head_kwargs=dict(unit="step")
    ) as pbars:
        while True:
            raw = progress_queue.get()
            if raw == 0:
                return
            i, rel_pos = raw
            pbars[i].update(rel_pos - pbars[i].n)
            pbars[0].update()


def format_variable_list(l: list[tuple[str, Any]]):
    joints = 2 * PARAM_SEPARATOR
    str_list = []
    for p_name, p_value in l:
        ps = p_name.replace("/", "").replace(joints[0], "").replace(joints[1], "")
        vs = (
            format_value(p_name, p_value)
            .replace("/", "")
            .replace(joints[0], "")
            .replace(joints[1], "")
        )
        str_list.append(ps + joints[1] + vs)
    return joints[0].join(str_list)


def branch_id(branch: tuple[Path, ...]) -> str:
    return "".join("".join(re.sub(r"id\d+\S*num\d+", "", b.name).split()[2:-2]) for b in branch)


def format_value(name: str, value) -> str:
    if value is True or value is False:
        return str(value)
    elif isinstance(value, (float, int)):
        try:
            return getattr(BareParams, name).display(value)
        except AttributeError:
            return format(value, ".9g")
    elif isinstance(value, (list, tuple, np.ndarray)):
        return "-".join([format_value(v) for v in value])
    elif isinstance(value, str):
        p = Path(value)
        if p.exists():
            return p.stem
    return str(value)


def pretty_format_value(name: str, value) -> str:
    try:
        return getattr(BareParams, name).display(value)
    except AttributeError:
        return name + PARAM_SEPARATOR + str(value)


def pretty_format_from_sim_name(name: str) -> str:
    """formats a pretty version of a simulation directory

    Parameters
    ----------
    name : str
        name of the simulation (directory name)

    Returns
    -------
    str
        prettier name
    """
    s = name.split(PARAM_SEPARATOR)
    out = []
    for key, value in zip(s[::2], s[1::2]):
        try:
            out += [key.replace("_", " "), getattr(BareParams, key).display(float(value))]
        except (AttributeError, ValueError):
            out.append(key + PARAM_SEPARATOR + value)
    return PARAM_SEPARATOR.join(out)


def variable_iterator(config: BareConfig) -> Iterator[tuple[list[tuple[str, Any]], dict[str, Any]]]:
    """given a config with "variable" parameters, iterates through every possible combination,
    yielding a a list of (parameter_name, value) tuples and a full config dictionary.

    Parameters
    ----------
    config : BareConfig
        initial config obj

    Yields
    -------
    Iterator[tuple[list[tuple[str, Any]], dict[str, Any]]]
        variable_list : a list of (name, value) tuple of parameter name and value that are variable.

        params : a dict[str, Any] to be fed to Params
    """
    possible_keys = []
    possible_ranges = []

    for key, values in config.variable.items():
        possible_keys.append(key)
        possible_ranges.append(range(len(values)))

    combinations = itertools.product(*possible_ranges)

    for combination in combinations:
        indiv_config = {}
        variable_list = []
        for i, key in enumerate(possible_keys):
            parameter_value = config.variable[key][combination[i]]
            indiv_config[key] = parameter_value
            variable_list.append((key, parameter_value))
        param_dict = asdict(config)
        param_dict.pop("variable")
        param_dict.update(indiv_config)
        yield variable_list, param_dict


def required_simulations(
    *configs: BareConfig,
) -> Iterator[tuple[list[tuple[str, Any]], BareParams]]:
    """takes the output of `scgenerator.utils.variable_iterator` which is a new dict per different
    parameter set and iterates through every single necessary simulation

    Yields
    -------
    Iterator[tuple[list[tuple[str, Any]], dict]]
        variable_ind : a list of (name, value) tuple of parameter name and value that are variable. The parameter
        "num" (how many times this specific parameter set has been yielded already) and "id" (how many parameter sets
        have been exhausted already) are added to the list to make sure every yielded list is unique.

        dict : a config dictionary for one simulation
    """
    i = 0  # unique sim id
    for data in itertools.product(*[variable_iterator(config) for config in configs]):
        all_variable_only, all_params_dict = list(zip(*data))
        params_dict = all_params_dict[0]
        for p in all_params_dict[1:]:
            params_dict.update({k: v for k, v in p.items() if v is not None})
        variable_only = reduce_all_variable(all_variable_only)
        for j in range(configs[0].repeat or 1):
            variable_ind = [("id", i)] + variable_only + [("num", j)]
            i += 1
            yield variable_ind, BareParams(**params_dict)


def reduce_all_variable(all_variable: list[list[tuple[str, Any]]]) -> list[tuple[str, Any]]:
    out = []
    for n, variable_list in enumerate(all_variable):
        out += [("fiber", "ABCDEFGHIJKLMNOPQRSTUVWXYZ"[n % 26] * (n // 26 + 1)), *variable_list]
    return out


def override_config(new: BareConfig, old: BareConfig = None) -> BareConfig:
    """makes sure all the parameters set in new are there, leaves untouched parameters in old"""
    new_dict = asdict(new)
    if old is None:
        return BareConfig(**new_dict)
    variable = deepcopy(old.variable)
    new_dict = {k: v for k, v in new_dict.items() if v is not None}

    for k, v in new_dict.pop("variable", {}).items():
        variable[k] = v
    for k in variable:
        new_dict[k] = None
    return replace(old, variable=variable, **new_dict)


def final_config_from_sequence(*configs: BareConfig) -> BareConfig:
    if len(configs) == 0:
        raise ValueError("Must provide at least one config")
    if len(configs) == 1:
        return configs[0]
    elif len(configs) == 2:
        return override_config(*configs[::-1])
    else:
        return override_config(configs[-1], final_config_from_sequence(*configs[:-1]))


def auto_crop(x: np.ndarray, y: np.ndarray, rel_thr: float = 0.01) -> np.ndarray:
    threshold = y.min() + rel_thr * (y.max() - y.min())
    above_threshold = y > threshold
    ind = np.argsort(x)
    valid_ind = [
        np.array(list(g)) for k, g in itertools.groupby(ind, key=lambda i: above_threshold[i]) if k
    ]
    ind_above = sorted(valid_ind, key=lambda el: len(el), reverse=True)[0]
    width = len(ind_above)
    return np.concatenate(
        (
            np.arange(max(ind_above[0] - width, 0), ind_above[0]),
            ind_above,
            np.arange(ind_above[-1] + 1, min(len(y), ind_above[-1] + width)),
        )
    )
