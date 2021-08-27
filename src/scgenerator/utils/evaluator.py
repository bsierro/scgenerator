import itertools
import re
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Callable, Optional, TypeVar, Union

import numpy as np

from .. import math
from ..logger import get_logger
from ..physics import fiber, materials, pulse, units

T = TypeVar("T")
import inspect


class EvaluatorError(Exception):
    pass


class Rule:
    def __init__(
        self,
        target: Union[str, list[Optional[str]]],
        func: Callable,
        args: list[str] = None,
        priorities: Union[int, list[int]] = None,
        conditions: dict[str, str] = None,
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
        self.conditions = conditions or {}

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
    @classmethod
    def default(cls) -> "Evaluator":
        evaluator = cls()
        evaluator.append(*default_rules)
        return evaluator

    def __init__(self):
        self.rules: dict[str, list[Rule]] = defaultdict(list)
        self.params = {}
        self.__curent_lookup = set()
        self.eval_stats: dict[str, EvalStat] = defaultdict(EvalStat)
        self.logger = get_logger(__name__)

    def append(self, *rule: Rule):
        for r in rule:
            for t in r.targets:
                if t is not None:
                    self.rules[t].append(r)
                    self.rules[t].sort(key=lambda el: el.targets[t], reverse=True)

    def set(self, **params: Any):
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
        EvaluatorError
            a cyclic dependence exists
        KeyError
            there is no saved rule for the target
        """
        value = self.params.get(target)
        if value is None:
            if target in self.__curent_lookup:
                raise EvaluatorError(
                    "cyclic dependency detected : "
                    f"{target!r} seems to depend on itself, "
                    f"please provide a value for at least one variable in {self.__curent_lookup}"
                )
            else:
                self.__curent_lookup.add(target)

            if len(self.rules[target]) == 0:
                raise EvaluatorError(f"no rule for {target}")

            error = None
            for ii, rule in enumerate(
                filter(lambda r: self.validate_condition(r), reversed(self.rules[target]))
            ):
                self.logger.debug(f"attempt {ii+1} to compute {target}, this time using {rule!r}")
                try:
                    args = [self.compute(k) for k in rule.args]
                    returned_values = rule.func(*args)
                    if len(rule.targets) == 1:
                        returned_values = [returned_values]
                    for ((param_name, param_priority), returned_value) in zip(
                        rule.targets.items(), returned_values
                    ):
                        if (
                            param_name == target
                            or param_name not in self.params
                            or self.eval_stats[param_name].priority < param_priority
                        ):
                            self.logger.info(
                                f"computed {param_name}={returned_value} using {rule.func.__name__} from {rule.func.__module__}"
                            )
                            self.params[param_name] = returned_value
                            self.eval_stats[param_name] = param_priority
                        if param_name == target:
                            value = returned_value
                    break
                except (EvaluatorError, KeyError) as e:
                    error = e
                    continue

            if value is None and error is not None:
                raise error

            self.__curent_lookup.remove(target)
        return value

    def validate_condition(self, rule: Rule) -> bool:
        return all(self.compute(k) == v for k, v in rule.conditions.items())

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
    out_func = scope[tmp_name]
    out_func.__module__ = "evaluator"
    return out_func


default_rules: list[Rule] = [
    # Grid
    *Rule.deduce(
        ["z_targets", "t", "time_window", "t_num", "dt", "w_c", "w0", "w", "w_power_fact", "l"],
        math.build_sim_grid,
        ["time_window", "t_num", "dt"],
        2,
    ),
    # Pulse
    Rule("spec_0", np.fft.fft, ["field_0"]),
    Rule("field_0", np.fft.ifft, ["spec_0"]),
    Rule("spec_0", pulse.load_previous_spectrum, priorities=3),
    Rule(
        ["pre_field_0", "peak_power", "energy", "width"],
        pulse.load_field_file,
        [
            "field_file",
            "t",
            "peak_power",
            "energy",
            "intensity_noise",
            "noise_correlation",
            "quantum_noise",
            "w_c",
            "w0",
            "time_window",
            "dt",
        ],
        priorities=[2, 1, 1, 1],
    ),
    Rule("pre_field_0", pulse.initial_field, priorities=1),
    Rule(
        "field_0",
        pulse.add_shot_noise,
        ["pre_field_0", "quantum_noise", "w_c", "w0", "time_window", "dt"],
    ),
    Rule("peak_power", pulse.E0_to_P0, ["energy", "t0", "shape"]),
    Rule("peak_power", pulse.soliton_num_to_peak_power),
    Rule("energy", pulse.P0_to_E0, ["peak_power", "t0", "shape"]),
    Rule("energy", pulse.mean_power_to_energy),
    Rule("t0", pulse.width_to_t0),
    Rule("t0", pulse.soliton_num_to_t0),
    Rule("width", pulse.t0_to_width),
    Rule("soliton_num", pulse.soliton_num),
    Rule("L_D", pulse.L_D),
    Rule("L_NL", pulse.L_NL),
    Rule("L_sol", pulse.L_sol),
    # Fiber Dispersion
    Rule("wl_for_disp", fiber.lambda_for_dispersion),
    Rule("w_for_disp", units.m, ["wl_for_disp"]),
    Rule(
        "beta2_coefficients",
        fiber.dispersion_coefficients,
        ["wl_for_disp", "beta2_arr", "w0", "interpolation_range", "interpolation_degree"],
    ),
    Rule("beta2_arr", fiber.beta2),
    Rule("beta2_arr", fiber.dispersion_from_coefficients),
    Rule("beta2", lambda beta2_coefficients: beta2_coefficients[0]),
    Rule(
        ["wl_for_disp", "beta2_arr", "interpolation_range"],
        fiber.load_custom_dispersion,
        priorities=[2, 2, 2],
    ),
    Rule("hr_w", fiber.delayed_raman_w),
    Rule("n_eff", fiber.n_eff_hasan, conditions=dict(model="hasan")),
    Rule("n_eff", fiber.n_eff_marcatili, conditions=dict(model="marcatili")),
    Rule("n_eff", fiber.n_eff_marcatili_adjusted, conditions=dict(model="marcatili_adjusted")),
    Rule(
        "n_eff",
        fiber.n_eff_pcf,
        ["wl_for_disp", "pitch", "pitch_ratio"],
        conditions=dict(model="pcf"),
    ),
    Rule("capillary_spacing", fiber.HCARF_gap),
    # Fiber nonlinearity
    Rule("A_eff", fiber.A_eff_from_V),
    Rule("A_eff", fiber.A_eff_from_diam),
    Rule("A_eff", fiber.A_eff_hasan, conditions=dict(model="hasan")),
    Rule("A_eff", fiber.A_eff_from_gamma, priorities=-1),
    Rule("A_eff_arr", fiber.A_eff_from_V, ["core_radius", "V_eff_arr"]),
    Rule("A_eff_arr", fiber.load_custom_A_eff),
    Rule("A_eff_arr", fiber.constant_A_eff_arr, priorities=-1),
    Rule(
        "V_eff",
        fiber.V_parameter_koshiba,
        ["wavelength", "pitch", "pitch_ratio"],
        conditions=dict(model="pcf"),
    ),
    Rule("V_eff", fiber.V_eff_marcuse, ["wavelength", "core_radius", "numerical_aperture"]),
    Rule("V_eff_arr", fiber.V_parameter_koshiba, conditions=dict(model="pcf")),
    Rule("V_eff_arr", fiber.V_eff_marcuse),
    Rule("gamma", lambda gamma_arr: gamma_arr[0]),
    Rule("gamma_arr", fiber.gamma_parameter, ["n2", "w0", "A_eff_arr"]),
    # Fiber loss
    Rule("alpha", fiber.compute_capillary_loss),
    Rule("alpha", fiber.load_custom_loss),
    # gas
    Rule("n_gas_2", materials.n_gas_2),
]


def main():
    import matplotlib.pyplot as plt

    evalor = Evaluator()
    evalor.append(*default_rules)
    evalor.set(
        **{
            "length": 1,
            "z_num": 128,
            "wavelength": 1500e-9,
            "interpolation_degree": 8,
            "interpolation_range": (500e-9, 2200e-9),
            "t_num": 16384,
            "dt": 1e-15,
            "shape": "gaussian",
            "repetition_rate": 40e6,
            "width": 30e-15,
            "mean_power": 100e-3,
            "n2": 2.4e-20,
            "A_eff_file": "/Users/benoitsierro/Nextcloud/PhD/Supercontinuum/PCF Simulations/PM2000D/PM2000D_A_eff_marcuse.npz",
            "model": "pcf",
            "quantum_noise": True,
            "pitch": 1.2e-6,
            "pitch_ratio": 0.5,
        }
    )
    evalor.compute("z_targets")
    print(evalor.params.keys())
    print(evalor.params["l"][evalor.params["l"] > 0].min())
    evalor.compute("spec_0")
    plt.plot(evalor.params["l"], abs(evalor.params["spec_0"]) ** 2)
    plt.yscale("log")
    plt.show()
    print(evalor.compute("gamma"))
    print(evalor.compute("beta2"))
    from pprint import pprint


if __name__ == "__main__":
    main()
