"""
This files includes utility functions designed more or less to be used specifically with the
scgenerator module but some function may be used in any python program

"""


import collections
import datetime as dt
import itertools
import logging
import multiprocessing
import re
import socket
from typing import Any, Callable, Dict, Iterator, List, Mapping, Tuple, Union
from asyncio import Event

import numpy as np
import ray
from copy import deepcopy

from tqdm import tqdm

from .const import PARAM_SEPARATOR, PREFIX_KEY_BASE, valid_variable, pbar_format
from .logger import get_logger
from .math import *

# XXX ############################################
# XXX ############### Pure Python ################
# XXX ############################################


class PBars:
    def __init__(self, pbars: Union[tqdm, List[tqdm]]) -> None:
        if isinstance(pbars, tqdm):
            self.pbars = [pbars]
        else:
            self.pbars = pbars
        self.logger = get_logger(__name__)

    def print(self):
        if len(self.pbars) > 1:
            s = [""]
        else:
            s = []
        for pbar in self.pbars:
            s.append(str(pbar))
        self.logger.info("\n".join(s))

    def __iter__(self):
        yield from self.pbars

    def __getitem__(self, key):
        return self.pbars[key]

    def update(self):
        for pbar in self:
            pbar.update()
        self.print()

    def append(self, pbar: tqdm):
        self.pbars.append(pbar)

    def close(self):
        for pbar in self.pbars:
            pbar.close()


class ProgressTracker:
    def __init__(
        self,
        max: Union[int, float],
        prefix: str = "",
        suffix: str = "",
        logger: logging.Logger = None,
        auto_print: bool = True,
        percent_incr: Union[int, float] = 5,
        default_update: Union[int, float] = 1,
    ):
        self.max = max
        self.current = 0
        self.prefix = prefix
        self.suffix = suffix
        self.start_time = dt.datetime.now()
        self.auto_print = auto_print
        self.next_percent = percent_incr
        self.percent_incr = percent_incr
        self.default_update = default_update
        self.logger = logger if logger is not None else get_logger()

    def _update(self):
        if self.auto_print and self.current / self.max >= self.next_percent / 100:
            self.next_percent += self.percent_incr
            self.logger.info(self.prefix + self.ETA + self.suffix)

    def update(self, num=None):
        if num is None:
            num = self.default_update
        self.current += num
        self._update()

    def set(self, value):
        self.current = value
        self._update()

    @property
    def ETA(self):
        if self.current <= 0:
            return "\033[31mETA : unknown\033[0m"
        eta = (
            (dt.datetime.now() - self.start_time).seconds / self.current * (self.max - self.current)
        )
        H = eta // 3600
        M = (eta - H * 3600) // 60
        S = eta % 60
        percent = int(100 * self.current / self.max)
        return "\033[34mremaining : {:.0f}h {:.0f}min {:.0f}s ({:.0f}% in total). \033[31mETA : {:%Y-%m-%d %H:%M:%S}\033[0m".format(
            H, M, S, percent, dt.datetime.now() + dt.timedelta(seconds=eta)
        )

    def get_eta(self):
        return self.ETA

    def __str__(self):
        return "{}/{}".format(self.current, self.max)


class ProgressBarActor:
    counter: int
    delta: int
    event: Event

    def __init__(self, num_workers: int) -> None:
        self.counters = [0 for _ in range(num_workers + 1)]
        self.event = Event()

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
        self.event.set()

    async def wait_for_update(self) -> List[float]:
        """Blocking call.

        Waits until somebody calls `update`, then returns a tuple of
        the number of updates since the last call to
        `wait_for_update`, and the total number of completed items.
        """
        await self.event.wait()
        self.event.clear()
        return self.counters


def progress_worker(num_steps: int, progress_queue: multiprocessing.Queue):
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
    pbars: Dict[int, tqdm] = {}
    with tqdm(total=num_steps, desc="Simulating", unit="step", position=0) as tq:
        while True:
            raw = progress_queue.get()
            if raw == 0:
                for pbar in pbars.values():
                    pbar.close()
                return
            i, rel_pos = raw
            if i not in pbars:
                pbars[i] = tqdm(**pbar_format(i))
            pbars[i].update(rel_pos - pbars[i].n)
            tq.update()


def count_variations(config: dict) -> Tuple[int, int]:
    """returns (sim_num, variable_params_num) where sim_num is the total number of simulations required and
    variable_params_num is the number of distinct parameters that will vary."""
    sim_num = 1
    variable_params_num = 0

    for section_name in valid_variable:
        for array in config.get(section_name, {}).get("variable", {}).values():
            sim_num *= len(array)
            variable_params_num += 1

    sim_num *= config["simulation"].get("repeat", 1)
    return sim_num, variable_params_num


def format_variable_list(l: List[tuple]):
    joints = 2 * PARAM_SEPARATOR
    str_list = []
    for p_name, p_value in l:
        ps = p_name.replace("/", "").replace(joints[0], "").replace(joints[1], "")
        vs = format_value(p_value).replace("/", "").replace(joints[0], "").replace(joints[1], "")
        str_list.append(ps + joints[1] + vs)
    return joints[0].join(str_list)


def format_value(value):
    if type(value) == type(False):
        return str(value)
    elif isinstance(value, (float, int)):
        return format(value, ".9g")
    elif isinstance(value, (list, tuple, np.ndarray)):
        return "-".join([format_value(v) for v in value])
    else:
        return str(value)


# def variable_list_from_path(s: str) -> List[tuple]:
#     s = s.replace("/", "")
#     str_list = s.split(PARAM_SEPARATOR)
#     out = []
#     for i in range(0, len(str_list) // 2 * 2, 2):
#         out.append((str_list[i], get_value(str_list[i + 1])))
#     return out


# def get_value(s: str):
#     if s.lower() == "true":
#         return True
#     if s.lower() == "false":
#         return False

#     try:
#         return int(s)
#     except ValueError:
#         pass

#     try:
#         return float(s)
#     except ValueError:
#         pass

#     return s


def variable_iterator(config) -> Iterator[Tuple[List[Tuple[str, Any]], dict]]:
    """given a config with "variable" parameters, iterates through every possible combination,
    yielding a a list of (parameter_name, value) tuples and a full config dictionary.

    Parameters
    ----------
    config : dict
        initial config dictionary

    Yields
    -------
    Iterator[Tuple[List[Tuple[str, Any]], dict]]
        variable_list : a list of (name, value) tuple of parameter name and value that are variable.

        dict : a config dictionary for one simulation
    """
    indiv_config = deepcopy(config)
    variable_dict = {
        section_name: indiv_config.get(section_name, {}).pop("variable", {})
        for section_name in valid_variable
    }

    possible_keys = []
    possible_ranges = []

    for section_name, section in variable_dict.items():
        for key in section:
            arr = variable_dict[section_name][key]
            possible_keys.append((section_name, key))
            possible_ranges.append(range(len(arr)))

    combinations = itertools.product(*possible_ranges)

    for combination in combinations:
        variable_list = []
        for i, key in enumerate(possible_keys):
            parameter_value = variable_dict[key[0]][key[1]][combination[i]]
            indiv_config[key[0]][key[1]] = parameter_value
            variable_list.append((key[1], parameter_value))
        yield variable_list, indiv_config


def required_simulations(config) -> Iterator[Tuple[List[Tuple[str, Any]], dict]]:
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
    for variable_only, full_config in variable_iterator(config):
        for j in range(config["simulation"]["repeat"]):
            variable_ind = [("id", i)] + variable_only + [("num", j)]
            i += 1
            yield variable_ind, full_config


def parallelize(func, arg_iter, sim_jobs=4, progress_tracker_kwargs=None, const_kwarg={}):
    """given a function and an iterable of arguments, runs the function in parallel
    Parameters
    ----------
        func : a function
        arg_iter : an iterable that yields a tuple to be unpacked to the function as argument(s)
        sim_jobs : number of parallel runs
        progress_tracker_kwargs : key word arguments to be passed to the ProgressTracker
        const_kwarg : keyword arguments to be passed to the function on every run

    Returns
    ----------
        a list of the result ordered like arg_iter
    """
    pt = None
    if progress_tracker_kwargs is not None:
        progress_tracker_kwargs["auto_print"] = True
        pt = ray.remote(ProgressTracker).remote(**progress_tracker_kwargs)

    # Initial setup
    func = ray.remote(func)
    jobs = []
    results = []
    dico = {}  # to keep track of the order, as tasks may no finish in order
    for k, args in enumerate(arg_iter):
        if not isinstance(args, tuple):
            print("iterator must return a tuple")
            quit()
        # as we got through the iterator, wait for first one to finish before
        # adding a new job
        if len(jobs) >= sim_jobs:
            res, jobs = ray.wait(jobs)
            results[dico[res[0].task_id()]] = ray.get(res[0])
            if pt is not None:
                ray.get(pt.update.remote())
        newJob = func.remote(*args, **const_kwarg)
        jobs.append(newJob)
        dico[newJob.task_id()] = k
        results.append(None)

    # still have to wait for the last few jobs when there is no more new jobs
    for j in jobs:
        results[dico[j.task_id()]] = ray.get(j)
        if pt is not None:
            ray.get(pt.update.remote())

    return np.array(results)


def deep_update(d: Mapping, u: Mapping):
    for k, v in u.items():
        if isinstance(v, collections.abc.Mapping):
            d[k] = deep_update(d.get(k, {}), v)
        else:
            d[k] = v
    return d


def override_config(new: Dict[str, Any], old: Dict[str, Any] = None) -> Dict[str, Any]:
    """makes sure all the parameters set in new are there, leaves untouched parameters in old"""
    if old is None:
        return new
    out = deepcopy(old)
    for section_name, section in new.items():
        if isinstance(section, Mapping):
            for param_name, value in section.items():
                if param_name == "variable" and isinstance(value, Mapping):
                    out[section_name].setdefault("variable", {})
                    for p, v in value.items():
                        # override previously unvariable param
                        if p in old[section_name]:
                            del out[section_name][p]
                        out[section_name]["variable"][p] = v
                else:
                    # override previously variable param
                    if (
                        "variable" in old[section_name]
                        and isinstance(old[section_name]["variable"], Mapping)
                        and param_name in old[section_name]["variable"]
                    ):
                        del out[section_name]["variable"][param_name]
                        if len(out[section_name]["variable"]) == 0:
                            del out[section_name["variable"]]
                    out[section_name][param_name] = value
        else:
            out[section_name] = section
    return out


def formatted_hostname():
    s = socket.gethostname().replace(".", "_")
    return (PREFIX_KEY_BASE + s).upper()