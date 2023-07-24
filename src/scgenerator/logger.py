import logging

from scgenerator.env import log_file_level, log_print_level


lvl_map: dict[str, int] = dict(
    debug=logging.DEBUG,
    info=logging.INFO,
    warning=logging.WARNING,
    error=logging.ERROR,
    critical=logging.CRITICAL,
)


def get_logger(name=None):
    """returns a logging.Logger instance. This function is there because if scgenerator
    is used with some multiprocessing library, workers are not aware of any configuration done
    with the logging and so it must be reconfigured.

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
    return configure_logger(logger)


def configure_logger(logger: logging.Logger):
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
        print_lvl = lvl_map.get(log_print_level(), logging.NOTSET)
        file_lvl = lvl_map.get(log_file_level(), logging.NOTSET)

        if file_lvl > logging.NOTSET:
            formatter = logging.Formatter("{levelname}: {name}: {message}", style="{")
            file_handler1 = logging.FileHandler("scgenerator.log", "a+")
            file_handler1.setFormatter(formatter)
            file_handler1.setLevel(file_lvl)
            logger.addHandler(file_handler1)
        if print_lvl > logging.NOTSET:
            stream_handler = logging.StreamHandler()
            stream_handler.setLevel(print_lvl)
            logger.addHandler(stream_handler)

        logger.setLevel(logging.DEBUG)
        logger.already_configured = True
    return logger
