from collections import defaultdict
from typing import Any, Callable, Union
from typing import TypeVar, Optional
from dataclasses import dataclass
import numpy as np
import itertools
from functools import wraps
import re

from ..physics import fiber, pulse, materials
from .. import math

T = TypeVar("T")
import inspect


class Rule:
    def __init__(
        self,
        target: Union[str, list[Optional[str]]],
        func: Callable,
        args: list[str] = None,
        priorities: Union[int, list[int]] = None,
    ):
        targets = list(target) if isinstance(target, (list, tuple)) else [target]
        self.func = func
        if priorities is None:
            priorities = [1] * len(targets)
        elif isinstance(priorities, (int, float, np.integer, np.floating)):
            priorities = [priorities]
        self.targets = dict(zip(targets, priorities))
        if args is None:
            args = get_arg_names(func)
        self.args = args

    def __repr__(self) -> str:
        return f"Rule(targets={self.targets!r}, func={self.func!r}, args={self.args!r})"

    @classmethod
    def deduce(
        cls,
        target: Union[str, list[Optional[str]]],
        func: Callable,
        kwarg_names: list[str],
        n_var: int,
        args_const: list[str] = None,
    ) -> list["Rule"]:
        """given a function that doesn't need all its keyword arguemtn specified, will
        return a list of Rule obj, one for each combination of n_var specified kwargs

        Parameters
        ----------
        target : str | list[str | None]
            name of the variable(s) that func returns
        func : Callable
            function to work with
        kwarg_names : list[str]
            list of all kwargs of the function to be used
        n_var : int
            how many shoulf be used per rule
        arg_const : list[str], optional
            override the name of the positional arguments

        Returns
        -------
        list[Rule]
            list of all possible rules

        Example
        -------
        >> def lol(a, b=None, c=None):
            pass
        >> print(Rule.deduce(["d"], lol, ["b", "c"], 1))
        [
            Rule(targets={'d': 1}, func=<function lol_0 at 0x7f9bce31d0d0>, args=['a', 'b']),
            Rule(targets={'d': 1}, func=<function lol_0 at 0x7f9bce31d160>, args=['a', 'c'])
        ]
        """
        rules: list[cls] = []
        for var_possibility in itertools.combinations(kwarg_names, n_var):

            new_func = func_rewrite(func, list(var_possibility), args_const)

            rules.append(cls(target, new_func))
        return rules


@dataclass
class EvalStat:
    priority: float = np.inf


class Evaluator:
    def __init__(self):
        self.rules: dict[str, list[Rule]] = defaultdict(list)
        self.params = {}
        self.__curent_lookup = set()
        self.eval_stats: dict[str, EvalStat] = defaultdict(EvalStat)

    def append(self, *rule: Rule):
        for r in rule:
            for t in r.targets:
                if t is not None:
                    self.rules[t].append(r)
                    self.rules[t].sort(key=lambda el: el.targets[t], reverse=True)

    def update(self, **params: Any):
        self.params.update(params)
        for k in params:
            self.eval_stats[k].priority = np.inf

    def reset(self):
        self.params = {}
        self.eval_stats = defaultdict(EvalStat)

    def compute(self, target: str) -> Any:
        """computes a target

        Parameters
        ----------
        target : str
            name of the target

        Returns
        -------
        Any
            return type of the target function

        Raises
        ------
        RecursionError
            a cyclic dependence exists
        KeyError
            there is no saved rule for the target
        """
        value = self.params.get(target)
        if value is None:
            if target in self.__curent_lookup:
                raise RecursionError(
                    "cyclic dependency detected : "
                    f"{target!r} seems to depend on itself, "
                    f"please provide a value for at least one variable in {self.__curent_lookup}"
                )
            else:
                self.__curent_lookup.add(target)

            error = None
            for rule in reversed(self.rules[target]):
                try:
                    args = [self.compute(k) for k in rule.args]
                    returned_values = rule.func(*args)
                    if len(rule.targets) == 1:
                        self.params[target] = returned_values
                        self.eval_stats[target].priority = rule.targets[target]
                        value = returned_values
                    else:
                        for ((k, p), v) in zip(rule.targets.items(), returned_values):
                            if (
                                k == target
                                or k not in self.params
                                or self.eval_stats[k].priority < p
                            ):
                                self.params[k] = v
                                self.eval_stats[k] = p
                            if k == target:
                                value = v
                    break
                except (RecursionError, KeyError) as e:
                    error = e
                    continue

            if value is None and error is not None:
                raise error

            self.__curent_lookup.remove(target)
        return value

    def __call__(self, target: str, args: list[str] = None):
        """creates a wrapper that adds decorated functions to the set of rules

        Parameters
        ----------
        target : str
            name of the target
        args : list[str], optional
            list of name of arguments. Automatically deduced from function signature if
            not provided, by default None
        """

        def wrapper(func):
            self.append(Rule(target, func, args))
            return func

        return wrapper


def get_arg_names(func: Callable) -> list[str]:
    spec = inspect.getfullargspec(func)
    args = spec.args
    if spec.defaults is not None and len(spec.defaults) > 0:
        args = args[: -len(spec.defaults)]
    return args


def validate_arg_names(names: list[str]):
    for n in names:
        if re.match(r"^[^\s\-'\(\)\"\d][^\(\)\-\s'\"]*$", n) is None:
            raise ValueError(f"{n} is an invalid parameter name")


def func_rewrite(func: Callable, kwarg_names: list[str], arg_names: list[str] = None):
    if arg_names is None:
        arg_names = get_arg_names(func)
    else:
        validate_arg_names(arg_names)
    validate_arg_names(kwarg_names)
    sign_arg_str = ", ".join(arg_names + kwarg_names)
    call_arg_str = ", ".join(arg_names + [f"{s}={s}" for s in kwarg_names])
    tmp_name = f"{func.__name__}_0"
    func_str = f"def {tmp_name}({sign_arg_str}):\n    return __func__({call_arg_str})"
    scope = dict(__func__=func)
    exec(func_str, scope)
    return scope[tmp_name]


default_rules: list[Rule] = [
    *Rule.deduce(
        ["z_targets", "t", "time_window", "t_num", "dt", "w_c", "w0", "w", "w_power_fact", "l"],
        math.build_sim_grid,
        ["time_window", "t_num", "dt"],
        2,
    )
]
"""
    Rule("gamma", fiber.gamma_parameter),
    Rule("gamma", lambda gamma_arr: gamma_arr[0]),
    Rule(["beta", "gamma", "interp_range"], fiber.PCF_dispersion),
    Rule("n2"),
    Rule("loss"),
    Rule("loss_file"),
    Rule("effective_mode_diameter"),
    Rule("A_eff"),
    Rule("A_eff_file"),
    Rule("pitch"),
    Rule("pitch_ratio"),
    Rule("core_radius"),
    Rule("he_mode"),
    Rule("fit_parameters"),
    Rule("beta"),
    Rule("dispersion_file"),
    Rule("model"),
    Rule("length"),
    Rule("capillary_num"),
    Rule("capillary_outer_d"),
    Rule("capillary_thickness"),
    Rule("capillary_spacing"),
    Rule("capillary_resonance_strengths"),
    Rule("capillary_nested"),
    Rule("gas_name"),
    Rule("pressure"),
    Rule("temperature"),
    Rule("plasma_density"),
    Rule("field_file"),
    Rule("repetition_rate"),
    Rule("peak_power"),
    Rule("mean_power"),
    Rule("energy"),
    Rule("soliton_num"),
    Rule("quantum_noise"),
    Rule("shape"),
    Rule("wavelength"),
    Rule("intensity_noise"),
    Rule("width"),
    Rule("t0"),
    Rule("behaviors"),
    Rule("parallel"),
    Rule("raman_type"),
    Rule("ideal_gas"),
    Rule("repeat"),
    Rule("t_num"),
    Rule("z_num"),
    Rule("time_window"),
    Rule("dt"),
    Rule("tolerated_error"),
    Rule("step_size"),
    Rule("lower_wavelength_interp_limit"),
    Rule("upper_wavelength_interp_limit"),
    Rule("interpolation_degree"),
    Rule("prev_sim_dir"),
    Rule("recovery_last_stored"),
    Rule("worker_num"),
    Rule("field_0"),
    Rule("spec_0"),
    Rule("alpha"),
    Rule("gamma_arr"),
    Rule("A_eff_arr"),
    Rule("w"),
    Rule("l"),
    Rule("w_c"),
    Rule("w0"),
    Rule("w_power_fact"),
    Rule("t"),
    Rule("L_D"),
    Rule("L_NL"),
    Rule("L_sol"),
    Rule("dynamic_dispersion"),
    Rule("adapt_step_size"),
    Rule("error_ok"),
    Rule("hr_w"),
    Rule("z_targets"),
    Rule("const_qty"),
    Rule("beta_func"),
    Rule("gamma_func"),
    Rule("interp_range"),
    Rule("datetime"),
    Rule("version"),
]
"""


def main():

    evalor = Evaluator()
    evalor.append(*default_rules)
    evalor.update(
        **{
            "length": 1,
            "z_num": 128,
            "wavelength": 1500e-9,
            "interpolation_degree": 8,
            "t_num": 16384,
            "dt": 1e-15,
        }
    )
    evalor.compute("z_targets")
    print(evalor.params.keys())
    print(evalor.params["l"][evalor.params["l"] > 0].min())


if __name__ == "__main__":
    main()
