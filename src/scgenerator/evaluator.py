import itertools
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Callable, Optional, Union

import numpy as np


from . import math, operators, utils
from .const import MANDATORY_PARAMETERS
from .errors import EvaluatorError, NoDefaultError
from .physics import fiber, materials, pulse, units
from .utils import _mock_function, func_rewrite, get_arg_names, get_logger


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
            priorities = [0] * len(targets)
        elif isinstance(priorities, (int, float, np.integer, np.floating)):
            priorities = [priorities] * len(targets)
        self.targets = dict(zip(targets, priorities))
        if args is None:
            args = get_arg_names(func)
        self.args = args
        self.mock_func = _mock_function(len(self.args), len(self.targets))
        self.conditions = conditions or {}

    def __repr__(self) -> str:
        return f"Rule(targets={self.targets!r}, func={self.func_name}, args={self.args!r})"

    def __str__(self) -> str:
        return f"[{', '.join(self.args)}] -- {self.func.__module__}.{self.func.__name__} --> [{', '.join(self.targets)}]"

    @property
    def func_name(self) -> str:
        return f"{self.func.__module__}.{self.func.__name__}"

    @classmethod
    def deduce(
        cls,
        target: Union[str, list[Optional[str]]],
        func: Callable,
        kwarg_names: list[str],
        n_var: int,
        args_const: list[str] = None,
        priorities: Union[int, list[int]] = None,
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

            rules.append(cls(target, new_func, priorities=priorities))
        return rules


@dataclass
class EvalStat:
    priority: float = np.inf
    rule: Rule = None


class Evaluator:
    defaults: dict[str, Any] = {}

    @classmethod
    def default(cls, full_field: bool = False) -> "Evaluator":
        evaluator = cls()
        logger = get_logger(__name__)
        if full_field:
            logger.debug("Full field simulation")
            evaluator.append(*full_field_rules)
        else:
            logger.debug("Envelope simulation")
            evaluator.append(*envelope_rules)
        return evaluator

    @classmethod
    def evaluate_default(cls, params: dict[str, Any], check_only=False) -> dict[str, Any]:
        evaluator = cls.default(params.get("full_field", False))
        evaluator.set(**params)
        for target in MANDATORY_PARAMETERS:
            evaluator.compute(target, check_only=check_only)
        return evaluator.params

    @classmethod
    def register_default_param(cls, key, value):
        cls.defaults[key] = value

    def __init__(self):
        self.rules: dict[str, list[Rule]] = defaultdict(list)
        self.params = {}
        self.__curent_lookup: list[str] = []
        self.__failed_rules: dict[str, list[Rule]] = defaultdict(list)
        self.eval_stats: dict[str, EvalStat] = defaultdict(EvalStat)
        self.logger = get_logger(__name__)

    def append(self, *rule: Rule):
        for r in rule:
            for t in r.targets:
                if t is not None:
                    self.rules[t].append(r)
                    self.rules[t].sort(key=lambda el: el.targets[t], reverse=True)

    def set(self, dico: dict[str, Any] = None, **params: Any):
        """sets the internal set of parameters

        Parameters
        ----------
        dico : dict, optional
            if given, replace current dict of parameters with this one
            (not a copy of it), by default None
        params : Any
            if dico is None, update the internal dict of parameters with params
        """
        if dico is None:
            dico = params
            self.params.update(dico)
        else:
            self.reset()
            self.params = dico
        for k in dico:
            self.eval_stats[k].priority = np.inf

    def reset(self):
        self.params = {}
        self.eval_stats = defaultdict(EvalStat)

    def compute(self, target: str, check_only=False) -> Any:
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
            prefix = "\t" * len(self.__curent_lookup)
            # Avoid cycles
            if target in self.__curent_lookup:
                raise EvaluatorError(
                    "cyclic dependency detected : "
                    f"{target!r} seems to depend on itself, "
                    f"please provide a value for at least one variable in {self.__curent_lookup!r}. "
                    + self.attempted_rules_str(target)
                )
            else:
                self.__curent_lookup.append(target)

            if len(self.rules[target]) == 0:
                error = EvaluatorError(
                    f"no rule for {target}, trying to evaluate {self.__curent_lookup!r}"
                )
            else:
                error = None

            # try every rule until one succeeds
            for ii, rule in enumerate(filter(self.validate_condition, self.rules[target])):
                self.logger.debug(
                    prefix + f"attempt {ii+1} to compute {target}, this time using {rule!r}"
                )
                try:
                    args = [self.compute(k, check_only=check_only) for k in rule.args]
                    if check_only:
                        returned_values = rule.mock_func(*args)
                    else:
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
                            if check_only:
                                success_str = f"able to compute {param_name} "
                            else:
                                v_str = format(returned_value).replace("\n", "")
                                success_str = f"computed {param_name}={v_str} "
                            self.logger.debug(
                                prefix
                                + success_str
                                + f"using {rule.func.__name__} from {rule.func.__module__}"
                            )
                            self.set_value(param_name, returned_value, param_priority, rule)
                        if param_name == target:
                            value = returned_value
                    break
                except EvaluatorError as e:
                    error = e
                    self.logger.debug(
                        prefix + f"error using {rule.func.__name__} : {str(error).strip()}"
                    )
                    self.__failed_rules[target].append(rule)
                    continue
                except Exception as e:
                    raise type(e)(f"error while evaluating {target!r}")
            else:
                default = self.defaults.get(target)
                if default is None:
                    error = error or NoDefaultError(
                        prefix
                        + f"No default provided for {target}. Current lookup cycle : {self.__curent_lookup!r}. "
                        + self.attempted_rules_str(target)
                    )
                else:
                    value = default
                    self.logger.info(prefix + f"using default value of {value} for {target}")
                    self.set_value(target, value, 0, None)
            last_target = self.__curent_lookup.pop()
            assert target == last_target
            self.__failed_rules[target] = []

            if value is None and error is not None:
                raise error

        return value

    def __getitem__(self, key: str) -> Any:
        return self.params[key]

    def set_value(self, key: str, value: Any, priority: int, rule: Rule):
        self.params[key] = value
        self.eval_stats[key].priority = priority
        self.eval_stats[key].rule = rule

    def validate_condition(self, rule: Rule) -> bool:
        try:
            return all(self.compute(k) == v for k, v in rule.conditions.items())
        except EvaluatorError:
            return False

    def attempted_rules_str(self, target: str) -> str:
        rules = ", ".join(str(r) for r in self.__failed_rules[target])
        if len(rules) == 0:
            return ""
        return "attempted rules : " + rules


default_rules: list[Rule] = [
    # Grid
    *Rule.deduce(
        ["t", "time_window", "dt", "t_num"], math.build_t_grid, ["time_window", "t_num", "dt"], 2
    ),
    Rule("z_targets", math.build_z_grid),
    Rule("adapt_step_size", lambda step_size: step_size == 0),
    Rule("dynamic_dispersion", lambda pressure: isinstance(pressure, (list, tuple, np.ndarray))),
    Rule("w0", units.m, ["wavelength"]),
    Rule("l", units.m.inv, ["w"]),
    Rule("w0_ind", math.argclosest, ["w_for_disp", "w0"]),
    Rule("w_num", len, ["w"]),
    Rule("dw", lambda w: w[1] - w[0]),
    Rule(["fft", "ifft"], utils.fft_functions, priorities=1),
    # Pulse
    Rule("field_0", pulse.finalize_pulse),
    Rule("spec_0", utils.load_previous_spectrum, ["recovery_data_dir"], priorities=4),
    Rule("spec_0", utils.load_previous_spectrum, priorities=3),
    *Rule.deduce(
        ["pre_field_0", "peak_power", "energy", "width"],
        pulse.load_and_adjust_field_file,
        ["energy", "peak_power"],
        1,
        priorities=[2, 1, 1, 1],
    ),
    Rule("peak_power", pulse.E0_to_P0, ["energy", "t0", "shape"]),
    Rule("peak_power", pulse.soliton_num_to_peak_power),
    Rule("mean_power", pulse.energy_to_mean_power),
    Rule("energy", pulse.P0_to_E0, ["peak_power", "t0", "shape"]),
    Rule("energy", pulse.mean_power_to_energy, priorities=2),
    Rule("t0", pulse.width_to_t0),
    Rule("t0", pulse.soliton_num_to_t0),
    Rule("width", pulse.t0_to_width),
    Rule("soliton_num", pulse.soliton_num),
    Rule("L_D", pulse.L_D),
    Rule("L_NL", pulse.L_NL),
    Rule("L_sol", pulse.L_sol),
    Rule("c_to_a_factor", lambda: 1.0, priorities=-1),
    # Fiber Dispersion
    Rule("w_for_disp", units.m, ["wl_for_disp"]),
    Rule("hr_w", fiber.delayed_raman_w),
    Rule("gas_info", materials.GasInfo),
    Rule("chi_gas", lambda gas_info, wl_for_disp: gas_info.sellmeier.chi(wl_for_disp)),
    Rule("n_gas_2", materials.n_gas_2),
    Rule("n_eff", fiber.n_eff_hasan, conditions=dict(model="hasan")),
    Rule("n_eff", fiber.n_eff_marcatili, conditions=dict(model="marcatili")),
    Rule("n_eff", fiber.n_eff_marcatili_adjusted, conditions=dict(model="marcatili_adjusted")),
    Rule(
        "n_eff",
        fiber.n_eff_pcf,
        ["wl_for_disp", "pitch", "pitch_ratio"],
        conditions=dict(model="pcf"),
    ),
    Rule("n0", lambda w0_ind, n_eff: n_eff[w0_ind]),
    Rule("capillary_spacing", fiber.capillary_spacing_hasan),
    Rule("capillary_resonance_strengths", fiber.capillary_resonance_strengths),
    Rule("capillary_resonance_strengths", lambda: [], priorities=-1),
    Rule("beta_arr", fiber.beta),
    Rule("beta1_arr", fiber.beta1),
    Rule("beta2_arr", fiber.beta2),
    Rule(
        "zero_dispersion_wavelength",
        lambda beta2_arr, wl_for_disp: wl_for_disp[math.argclosest(beta2_arr, 0)],
    ),
    # Fiber nonlinearity
    Rule("A_eff", fiber.A_eff_from_V),
    Rule("A_eff", fiber.A_eff_from_diam),
    Rule("A_eff", fiber.A_eff_hasan, conditions=dict(model="hasan")),
    Rule("A_eff", fiber.A_eff_from_gamma, priorities=-1),
    Rule("A_eff", fiber.A_eff_marcatili, priorities=-2),
    Rule("A_eff_arr", fiber.A_eff_from_V, ["core_radius", "V_eff_arr"]),
    Rule("A_eff_arr", fiber.load_custom_A_eff),
    # Rule("A_eff_arr", fiber.constant_A_eff_arr, priorities=-1),
    Rule(
        "V_eff",
        fiber.V_parameter_koshiba,
        ["wavelength", "pitch", "pitch_ratio"],
        conditions=dict(model="pcf"),
    ),
    Rule("V_eff", fiber.V_eff_step_index, ["wavelength", "core_radius", "numerical_aperture"]),
    Rule("V_eff_arr", fiber.V_parameter_koshiba, conditions=dict(model="pcf")),
    Rule(
        "V_eff_arr",
        fiber.V_eff_step_index,
        ["l", "core_radius", "numerical_aperture", "interpolation_range"],
    ),
    Rule("n2", materials.gas_n2),
    Rule("n2", lambda: 2.2e-20, priorities=-1),
    Rule("gamma", lambda gamma_arr: gamma_arr[0], priorities=-1),
    Rule("gamma", fiber.gamma_parameter),
    Rule("gamma_arr", fiber.gamma_parameter, ["n2", "w0", "A_eff_arr"]),
    # operators
    Rule("n_op", operators.ConstantRefractiveIndex),
    Rule("n_op", operators.MarcatiliRefractiveIndex),
    Rule("n_op", operators.MarcatiliAdjustedRefractiveIndex),
    Rule("n_op", operators.HasanRefractiveIndex),
    Rule("gas_op", operators.ConstantGas),
    Rule("gas_op", operators.PressureGradientGas),
    Rule("loss_op", operators.NoLoss, priorities=-1),
    Rule("conserved_quantity", operators.NoConservedQuantity, priorities=-1),
]

envelope_rules = default_rules + [
    # Grid
    Rule(["w_c", "w", "w_order"], math.build_envelope_w_grid),
    # Pulse
    Rule("pre_field_0", pulse.initial_field_envelope, priorities=1),
    Rule("spec_0", np.fft.fft, ["field_0"]),
    Rule("field_0", np.fft.ifft, ["spec_0"]),
    Rule("c_to_a_factor", pulse.c_to_a_factor),
    # Dispersion
    Rule(["wl_for_disp", "dispersion_ind"], fiber.lambda_for_envelope_dispersion),
    Rule("beta2_coefficients", fiber.dispersion_coefficients),
    Rule("beta2_arr", fiber.dispersion_from_coefficients),
    Rule("beta2", lambda beta2_coefficients: beta2_coefficients[0]),
    Rule(
        ["wl_for_disp", "beta2_arr", "interpolation_range"],
        fiber.load_custom_dispersion,
        priorities=[2, 2, 2],
    ),
    # Operators
    Rule("gamma_op", operators.ConstantGamma, priorities=1),
    Rule("gamma_op", operators.ConstantScalarGamma),
    Rule("gamma_op", operators.NoGamma, priorities=-1),
    Rule("gamma_op", operators.VariableScalarGamma, priorities=2),
    Rule("ss_op", operators.SelfSteepening),
    Rule("ss_op", operators.NoSelfSteepening, priorities=-1),
    Rule("spm_op", operators.NoEnvelopeSPM, priorities=-1),
    Rule("spm_op", operators.EnvelopeSPM),
    Rule("raman_op", operators.EnvelopeRaman),
    Rule("raman_op", operators.NoEnvelopeRaman, priorities=-1),
    Rule("nonlinear_operator", operators.EnvelopeNonLinearOperator),
    Rule("loss_op", operators.CustomLoss, priorities=3),
    Rule("loss_op", operators.CapillaryLoss, priorities=2, conditions=dict(loss="capillary")),
    Rule("loss_op", operators.ConstantLoss, priorities=1),
    Rule("dispersion_op", operators.ConstantPolyDispersion),
    Rule("dispersion_op", operators.DirectDispersion),
    Rule("linear_operator", operators.EnvelopeLinearOperator),
    Rule("conserved_quantity", operators.conserved_quantity),
]

full_field_rules = default_rules + [
    # Grid
    Rule(["w", "w_order", "l"], math.build_full_field_w_grid, priorities=1),
    # Pulse
    Rule("spec_0", np.fft.rfft, ["field_0"]),
    Rule("field_0", np.fft.irfft, ["spec_0"]),
    Rule("pre_field_0", pulse.initial_full_field),
    # Dispersion
    Rule(["wl_for_disp", "dispersion_ind"], fiber.lambda_for_full_field_dispersion),
    Rule("frame_velocity", fiber.frame_velocity),
    Rule("beta2", lambda beta2_arr, w0_ind: beta2_arr[w0_ind]),
    # Nonlinearity
    Rule("chi3", materials.gas_chi3),
    # Operators
    Rule("spm_op", operators.FullFieldSPM),
    Rule("spm_op", operators.NoFullFieldSPM, priorities=-1),
    Rule("beta_op", operators.ConstantWaveVector),
    Rule(
        "linear_operator",
        operators.FullFieldLinearOperator,
    ),
    Rule("plasma_op", operators.Plasma, conditions=dict(photoionization=True)),
    Rule("plasma_op", operators.NoPlasma, priorities=-1),
    Rule("raman_op", operators.NoFullFieldRaman, priorities=-1),
    Rule("nonlinear_operator", operators.FullFieldNonLinearOperator),
]
