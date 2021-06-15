"""
This files includes utility functions designed more or less to be used specifically with the
scgenerator module but some function may be used in any python program

"""

import functools
import itertools
import multiprocessing
import threading
from functools import update_wrapper
from collections import abc
from copy import deepcopy
from dataclasses import asdict, replace
from io import StringIO
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Tuple, TypeVar, Union
from copy import copy

import numpy as np
from numpy.lib.index_tricks import nd_grid
from tqdm import tqdm

from ..logger import get_logger

from .. import env
from ..const import PARAM_SEPARATOR
from ..math import *
from .parameter import BareConfig, BareParams
from scgenerator import logger

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
        head_kwargs["desc"] = desc
        self.pbars = [tqdm(total=self.num_tot, ncols=100, ascii=False, **head_kwargs)]
        for i in range(1, num_sub_bars + 1):
            kwargs = {k: v for k, v in worker_kwargs.items()}
            if "desc" in kwargs:
                kwargs["desc"] = kwargs["desc"].format(worker_id=i)
            self.append(tqdm(position=i, ncols=100, ascii=False, **kwargs))
        self.print_path = Path("progress " + self.pbars[0].desc).resolve()
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
            Tuple[int, float] : worker id and relative progress between 0 and 1
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


def count_variations(config: BareConfig) -> int:
    """returns (sim_num, variable_params_num) where sim_num is the total number of simulations required and
    variable_params_num is the number of distinct parameters that will vary."""
    sim_num = np.prod([len(l) for l in config.variable.values()]) * config.repeat
    return sim_num


def format_variable_list(l: List[Tuple[str, Any]]):
    joints = 2 * PARAM_SEPARATOR
    str_list = []
    for p_name, p_value in l:
        ps = p_name.replace("/", "").replace(joints[0], "").replace(joints[1], "")
        vs = format_value(p_value).replace("/", "").replace(joints[0], "").replace(joints[1], "")
        str_list.append(ps + joints[1] + vs)
    return joints[0].join(str_list)


def branch_id(branch: Tuple[Path, ...]) -> str:
    return "".join("".join(b.name.split()[2:-2]) for b in branch)


def format_value(value):
    if type(value) == type(False):
        return str(value)
    elif isinstance(value, (float, int)):
        return format(value, ".9g")
    elif isinstance(value, (list, tuple, np.ndarray)):
        return "-".join([format_value(v) for v in value])
    else:
        return str(value)


def variable_iterator(config: BareConfig) -> Iterator[Tuple[List[Tuple[str, Any]], BareParams]]:
    """given a config with "variable" parameters, iterates through every possible combination,
    yielding a a list of (parameter_name, value) tuples and a full config dictionary.

    Parameters
    ----------
    config : BareConfig
        initial config obj

    Yields
    -------
    Iterator[Tuple[List[Tuple[str, Any]], BareParams]]
        variable_list : a list of (name, value) tuple of parameter name and value that are variable.

        params : a BareParams obj for one simulation
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
        yield variable_list, BareParams(**param_dict)


def required_simulations(config: BareConfig) -> Iterator[Tuple[List[Tuple[str, Any]], BareParams]]:
    """takes the output of `scgenerator.utils.variable_iterator` which is a new dict per different
    parameter set and iterates through every single necessary simulation

    Yields
    -------
    Iterator[Tuple[List[Tuple[str, Any]], dict]]
        variable_ind : a list of (name, value) tuple of parameter name and value that are variable. The parameter
        "num" (how many times this specific parameter set has been yielded already) and "id" (how many parameter sets
        have been exhausted already) are added to the list to make sure every yielded list is unique.

        dict : a config dictionary for one simulation
    """
    i = 0  # unique sim id
    for variable_only, bare_params in variable_iterator(config):
        for j in range(config.repeat):
            variable_ind = [("id", i)] + variable_only + [("num", j)]
            i += 1
            yield variable_ind, bare_params


def override_config(new: Dict[str, Any], old: BareConfig = None) -> BareConfig:
    """makes sure all the parameters set in new are there, leaves untouched parameters in old"""
    if old is None:
        return BareConfig(**new)
    variable = deepcopy(old.variable)
    variable.update(new.pop("variable", {}))  # add new variable
    for k in new:
        variable.pop(k, None)  # remove old ones
    return replace(old, variable=variable, **{k: None for k in variable}, **new)


# def np_cache(function):
#     """applies functools.cache to function that take numpy arrays as input"""

#     @cache
#     def cached_wrapper(*hashable_args, **hashable_kwargs):
#         args = tuple(np.array(arg) if isinstance(arg, tuple) else arg for arg in hashable_args)
#         kwargs = {
#             k: np.array(kwarg) if isinstance(kwarg, tuple) else kwarg
#             for k, kwarg in hashable_kwargs.items()
#         }
#         return function(*args, **kwargs)

#     @wraps(function)
#     def wrapper(*args, **kwargs):
#         hashable_args = tuple(tuple(arg) if isinstance(arg, np.ndarray) else arg for arg in args)
#         hashable_kwargs = {
#             k: tuple(kwarg) if isinstance(kwarg, np.ndarray) else kwarg
#             for k, kwarg in kwargs.items()
#         }
#         return cached_wrapper(*hashable_args, **hashable_kwargs)

#     # copy lru_cache attributes over too
#     wrapper.cache_info = cached_wrapper.cache_info
#     wrapper.cache_clear = cached_wrapper.cache_clear

#     return wrapper


class np_cache:
    def __init__(self, function):
        self.logger = get_logger(__name__)
        self.func = function
        self.cache = {}
        self.hits = 0
        self.misses = 0
        update_wrapper(self, function)

    def __call__(self, *args, **kwargs):
        hashable_args = tuple(
            tuple(arg) if isinstance(arg, (np.ndarray, list)) else arg for arg in args
        )
        hashable_kwargs = tuple(
            {
                k: tuple(kwarg) if isinstance(kwarg, (np.ndarray, list)) else kwarg
                for k, kwarg in kwargs.items()
            }.items()
        )
        key = hash((hashable_args, hashable_kwargs))
        if key not in self.cache:
            self.logger.debug("cache miss")
            self.misses += 1
            self.cache[key] = self.func(*args, **kwargs)
        else:
            self.hits += 1
            self.logger.debug("cache hit")
        return copy(self.cache[key])
