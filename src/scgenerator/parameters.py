class Parameter:
    """base class for parameters"""

    all = dict(fiber=dict(), pulse=dict(), gas=dict(), simulation=dict())
    help_message = "no help message lol"

    def __init_subclass__(cls, section):
        Parameter.all[section][cls.__name__.lower()] = cls

    def __init__(self, s):
        self.s = s
        valid = True
        try:
            self.value = self._convert()
            valid = self.valid()
        except ValueError:
            valid = False

        if not valid:
            raise ValueError(
                f"{self.__class__.__name__} {self.__class__.help_message}. input : {self.s}"
            )

    def _convert(self):
        value = self.conversion_func(self.s)
        return value


class Wavelength(Parameter, section="pulse"):
    help_message = "must be a strictly positive real number"

    def valid(self):
        return self.value > 0

    def conversion_func(self, s: str) -> float:
        return float(s)
