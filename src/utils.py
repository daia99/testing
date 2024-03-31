import os
import logging


def create_logger(
    name: str, log_dir: str = None, debug: bool = False
) -> logging.Logger:
    """Create a logger.

    Args:
        name - Name of the logger.
        log_dir - The logger will also log to an external file in the specified
                  directory if specified.
        debug - If we should log in DEBUG mode.

    Returns:
        logging.RootLogger.
    """

    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir)
    log_format = "%(name)s: %(asctime)s [%(levelname)s] %(message)s"
    logging.basicConfig(
        level=logging.DEBUG if debug else logging.INFO, format=log_format
    )
    logger = logging.getLogger(name)
    if log_dir:
        log_file = os.path.join(log_dir, "{}.txt".format(name))
        file_hdl = logging.FileHandler(log_file)
        formatter = logging.Formatter(fmt=log_format)
        file_hdl.setFormatter(formatter)
        logger.addHandler(file_hdl)
    # Set level explicitly, otherwise the logger does not output.
    logger.setLevel(logging.DEBUG if debug else logging.INFO)
    return logger
