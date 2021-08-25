from typing import Callable
import inspect
import re


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


def lol(a, b=None, c=None):
    print(f"{a=}, {b=}, {c=}")


def main():
    lol1 = func_rewrite(lol, ["c"])
    print(inspect.getfullargspec(lol1))
    lol2 = func_rewrite(lol, ["b"])
    print(inspect.getfullargspec(lol2))


if __name__ == "__main__":
    main()
