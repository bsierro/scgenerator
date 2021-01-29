import logging


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
    return configure_logger(logger)


def configure_logger(logger, logfile="scgenerator.log"):
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
        if logfile is not None:
            file_handler = logging.FileHandler("scgenerator.log", "a+")
            file_handler.setFormatter(logging.Formatter("{name}: {message}", style="{"))
            logger.addHandler(file_handler)

        stream_handler = logging.StreamHandler()
        logger.addHandler(stream_handler)
        logger.setLevel(logging.INFO)

        logger.already_configured = True
    return logger