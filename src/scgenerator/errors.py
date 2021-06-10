class MissingParameterError(Exception):
    def __init__(self, param, **kwargs):
        self.param = param

        # initial message
        message = ""
        if isinstance(param, str):
            message += f"'{self.param}' is a required parameter "
        elif isinstance(param, (tuple, list)):
            num = kwargs["num_required"]
            message += f"{num} of '{self.param}' {'is' if num == 1 else 'are'} required "
        else:
            raise TypeError(f"don't know what to do with param as {param}")

        # complementary information
        if "fiber_model" in kwargs:
            message += f"for fiber model '{kwargs['fiber_model']}' "
        if "specified_parameters" in kwargs:
            if len(kwargs["specified_parameters"]) == 0:
                pass
            elif len(kwargs["specified_parameters"]) == 1:
                message += f"when '{kwargs['specified_parameters'][0]}' is specified "
            else:
                message += f"when {kwargs['specified_parameters']} are specified "

        # closing statement
        message += "and no defaults have been set"
        super().__init__(message)


class DuplicateParameterError(Exception):
    pass


class IncompleteDataFolderError(FileNotFoundError):
    pass
