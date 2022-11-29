import logging
import sys

def set_logger(
    name: str = None, log_path: str = None,
    formatter: str = '%(asctime)s: %(pathname)s-%(lineno)d - %(levelname)s\n%(message)s',
    to_console=True
):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter(formatter)
    # create console handler with a higher log level
    if to_console:
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        # create formatter and add it to the handlers
        ch.setFormatter(formatter)
        # add the handlers to the logger
        logger.addHandler(ch)

    # create file handler which logs even debug messages
    if log_path is not None:
        fh = logging.FileHandler(log_path)
        fh.setLevel(logging.INFO)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    # set error traceback logged
    # https://stackoverflow.com/questions/6234405/logging-uncaught-exceptions-in-python
    def handle_exception(exc_type, exc_value, exc_traceback):
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
            return

        logging.error("Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))

    sys.excepthook = handle_exception
    return logger

class DummyLogger:
    """
    Do nothing on logging
    """
    def __init__(*args, **kwargs):
        pass
    def __call__(self, *args, **kwargs):
        return self
    def __getattr__(self, *args, **kwargs):
        return self
