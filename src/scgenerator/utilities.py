"""
This files includes utility functions designed more or less to be used specifically with the
scgenerator module but some function may be used in any python program

"""


import datetime as dt
import itertools
from typing import Callable, List, Tuple, Union, Any

import numpy as np
import ray

from .math import *
from .const import valid_varying

# XXX ############################################
# XXX ############### Pure Python ################
# XXX ############################################


class ProgressTracker:
    def __init__(
        self,
        max: Union[int, float],
        auto_print: bool = False,
        percent_incr: Union[int, float] = 5,
        default_update: Union[int, float] = 1,
        callback: Callable[[str, Any], None] = None,
    ):
        self.max = max
        self.current = 0
        self.start_time = dt.datetime.now()
        self.auto_print = auto_print
        self.next_percent = percent_incr
        self.percent_incr = percent_incr
        self.default_update = default_update
        self.callback = callback

    def _update(self, callback_args):
        if self.auto_print and self.current / self.max >= self.next_percent / 100:
            self.next_percent += self.percent_incr
            if self.callback is None:
                print(self.ETA)
            else:
                self.callback(self.ETA, *callback_args)

    def update(self, num=None, callback_args=[]):
        if num is None:
            num = self.default_update
        self.current += num
        self._update(callback_args)

    def set(self, value, callback_args=[]):
        self.current = value
        self._update(callback_args)

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


# def ray_safe(func, *args, **kwargs):
#     """evaluates functions that return None whether they are Ray workers or normal functions
#     Parameters
#     ----------
#         func : the function or Worker id
#         args : arguments to give to the functions
#     Returns
#     ----------
#         nothing
#     """
#     if hasattr(func, "remote"):
#         ray.get(func.remote(*args, **kwargs))
#     else:
#         func(*args, **kwargs)


def count_variations(config: dict) -> Tuple[int, int]:
    """returns True if the config specified by the config dict requires only on simulation run"""
    num = 1
    varying_params = 0

    for section_name in valid_varying:
        for array in config.get(section_name, {}).get("varying", {}).values():
            num *= len(array)
            varying_params += 1

    num *= config["simulation"].get("repeat", 1)
    return num, varying_params


def format_varying_list(l: List[tuple], joints: List[str] = ""):
    while len(joints) < 2:
        joints += "_"
    str_list = []
    for p_name, p_value in l:
        ps = p_name.replace("/", "").replace(joints[0], "").replace(joints[1], "")
        vs = format_value(p_value).replace("/", "").replace(joints[0], "").replace(joints[1], "")
        str_list.append(ps + joints[1] + vs)
    return joints[0].join(str_list)


def varying_list_from_path(s: str) -> List[tuple]:
    s = s.replace("/", "")
    str_list = s.split("_")
    out = []
    for i in range(0, len(str_list) // 2 * 2, 2):
        out.append((str_list[i], get_value(str_list[i + 1])))
    return out


def format_value(value):
    if type(value) == type(False):
        return str(value)
    elif isinstance(value, (float, int)):
        return format(value, ".5g")
    elif isinstance(value, (list, tuple, np.ndarray)):
        return "-".join([format_value(v) for v in value])
    else:
        return str(value)


def get_value(s: str):
    if s.lower() == "true":
        return True
    if s.lower() == "false":
        return False

    try:
        return int(s)
    except ValueError:
        pass

    try:
        return float(s)
    except ValueError:
        pass

    return s


def varying_iterator(config):
    varying_dict = {
        section_name: config.get(section_name, {}).pop("varying", {})
        for section_name in valid_varying
    }

    possible_keys = []
    possible_ranges = []

    for section_name, section in varying_dict.items():
        for key in section:
            arr = np.atleast_1d(varying_dict[section_name][key])
            varying_dict[section_name][key] = arr
            possible_keys.append((section_name, key))
            possible_ranges.append(range(len(arr)))

    combinations = itertools.product(*possible_ranges)

    for combination in combinations:
        out = config.copy()
        only_varying = []
        for i, key in enumerate(possible_keys):
            parameter_value = varying_dict[key[0]][key[1]][combination[i]]
            out[key[0]][key[1]] = parameter_value
            only_varying.append((key[1], parameter_value))
        yield only_varying, out


def parallelize(func, arg_iter, sim_jobs=4, progress_tracker_kwargs=None, const_kwarg={}):
    """given a function and an iterator of arguments, runs the function in parallel
    Parameters
    ----------
        func : a function
        arg_iter : an iterator that yields a tuple to be unpacked to the function as argument(s)
        sim_jobs : number of simultaneous runs
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
