from __future__ import annotations
from collections import defaultdict


class Parameter:
    registered_params = defaultdict(dict)

    def __init__(self, default_value, display_suffix=""):
        self.value = default_value
        self.display_suffix = display_suffix

    def __set_name__(self, owner, name):
        self.name = name
        self.registered_params[owner.__name__][name] = self

    def __get__(self, instance, owner):
        return self.value

    def __set__(self, instance, value):
        self.value = value

    def display(self):
        return str(self.value) + " " + self.display_suffix


class A:
    x = Parameter("lol")
    y = Parameter(56.2)


class B:
    x = Parameter(slice(None))
    opt = None


def main():
    print(Parameter.registered_params["A"])
    print(Parameter.registered_params["B"])
    a = A()
    a.x = 5
    print(a.x)


if __name__ == "__main__":
    main()
