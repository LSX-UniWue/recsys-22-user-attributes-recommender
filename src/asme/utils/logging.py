import logging
import sys
import threading
from typing import Optional

_lock = threading.Lock()
_default_handler: Optional[logging.Handler] = None


def _config_logging():

    global _default_handler

    with _lock:
        if _default_handler:
            # logger already configured
            return
        _default_handler = logging.StreamHandler()  # Set sys.stderr as stream.
        _default_handler.flush = sys.stderr.flush

        # Apply our default configuration to the library root logger.
        library_root_logger = _get_asme_root_logger()
        library_root_logger.addHandler(_default_handler)
        library_root_logger.setLevel(logging.INFO)
        library_root_logger.propagate = False


def _get_asme_root_logger() -> logging.Logger:
    return logging.getLogger(_get_lib_name())


def _get_lib_name() -> str:
    return __name__.split(".")[0]


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """
    Return a logger with the specified name.
    """

    _config_logging()
    return logging.getLogger(name)
