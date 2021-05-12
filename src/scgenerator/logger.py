import logging
from typing import Optional


class DebugOnlyFileHandler(logging.FileHandler):
    def __init__(
        self, filename, mode: str, encoding: Optional[str] = None, delay: bool = False
    ) -> None:
        super().__init__(filename, mode=mode, encoding=encoding, delay=delay)
        self.setLevel(logging.DEBUG)

    def emit(self, record: logging.LogRecord) -> None:
        if not record.levelno == logging.DEBUG:
            return
        return super().emit(record)


DEFAULT_LEVEL = logging.INFO

lvl_map = dict(
    debug=logging.DEBUG,
    info=logging.INFO,
    warning=logging.WARNING,
    error=logging.ERROR,
    fatal=logging.FATAL,
    critical=logging.CRITICAL,
)

loggers = []


def _set_debug():
    DEFAULT_LEVEL = logging.DEBUG


def get_logger(name=None):
    """returns a logging.Logger instance. This function is there because if scgenerator
    is used with ray, workers are not aware of any configuration done with the logging
    and so it must be reconfigured.

    Parameters
    ----------
    name : str, optional
        name of the logger, by default None

    Returns
    -------
    logging.Logger obj
        logger
    """
    name = __name__ if name is None else name
    logger = logging.getLogger(name)
    if name not in loggers:
        loggers.append(logger)
    return configure_logger(logger)


# def set_level_all(lvl):
#     _default_lvl =
#     logging.basicConfig(level=lvl_map[lvl])
#     for logger in loggers:
#         logger.setLevel(lvl_map[lvl])
#         for handler in logger.handlers:
#             handler.setLevel(lvl_map[lvl])


def configure_logger(logger):
    """configures a logging.Logger obj

    Parameters
    ----------
    logger : logging.Logger
        logger to configure
    logfile : str or None, optional
        path to log file

    Returns
    -------
    logging.Logger obj
        updated logger
    """
    if not hasattr(logger, "already_configured"):
        formatter = logging.Formatter("{levelname}: {name}: {message}", style="{")
        file_handler1 = DebugOnlyFileHandler("sc-DEBUG.log", "a+")
        file_handler1.setFormatter(formatter)
        logger.addHandler(file_handler1)

        file_handler2 = logging.FileHandler("sc-INFO.log", "a+")
        file_handler2.setFormatter(formatter)
        file_handler2.setLevel(logging.INFO)
        logger.addHandler(file_handler2)

        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(logging.WARNING)
        logger.addHandler(stream_handler)
        logger.setLevel(logging.DEBUG)

        logger.already_configured = True
    return logger