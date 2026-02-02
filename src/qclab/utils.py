"""
This module contains utility helpers for optional jit compilation and in-memory logging.
"""

import logging
from io import StringIO

DISABLE_NUMBA = False
try:
    import numba as _
except ImportError:
    DISABLE_NUMBA = True

DISABLE_H5PY = False
try:
    import h5py as _
except ImportError:
    DISABLE_H5PY = True

DISABLE_ASE = False
try:
    import ase as _
except ImportError:
    DISABLE_ASE = True


logger = logging.getLogger(__name__)


def qc_lab_custom_jit(func=None, **kwargs):
    """
    Dummy jit decorator that does nothing if numba is not available.
    """

    def decorator(func):
        return func

    if func is None:
        return decorator
    return decorator(func)


def qc_lab_custom_njit(func=None, **kwargs):
    """
    Dummy njit decorator that does nothing if numba is not available.
    """

    def decorator(func):
        return func

    if func is None:
        return decorator
    else:
        return decorator(func)


if DISABLE_NUMBA:
    logger.info(
        "Numba is disabled; using dummy jit decorator.",
    )
    njit = qc_lab_custom_njit
    jit = qc_lab_custom_jit
else:
    try:
        from numba import jit, njit
    except ImportError:
        jit = qc_lab_custom_jit
        njit = qc_lab_custom_njit


_log_stream = StringIO()


class QCDataHandler(logging.Handler):
    """
    Logging handler that stores logs in memory.
    """

    def emit(self, record):
        msg = self.format(record)
        _log_stream.write(msg + "\n")


def configure_memory_logger(level=logging.INFO):
    """
    Configure root logger to store logs without printing.
    """
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    for handler in list(root_logger.handlers):
        root_logger.removeHandler(handler)
    handler = QCDataHandler()
    formatter = logging.Formatter("%(levelname)s:%(name)s:%(message)s")
    handler.setFormatter(formatter)
    root_logger.addHandler(handler)
    return handler


def get_log_output() -> str:
    """
    Return all collected log messages.
    """
    return _log_stream.getvalue()


def reset_log_output() -> None:
    """
    Clear all collected log messages.

    Resets the in-memory log stream so that subsequent calls to
    :func:`get_log_output` return only the messages emitted after this
    function is called.
    """
    _log_stream.seek(0)
    _log_stream.truncate(0)


__all__ = [
    "jit",
    "njit",
    "configure_memory_logger",
    "get_log_output",
    "reset_log_output",
    "DISABLE_NUMBA",
    "DISABLE_H5PY",
]
