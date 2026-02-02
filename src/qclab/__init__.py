"""
This module contains the QC Lab package initialization.
"""

from qclab.simulation import Simulation
from qclab.data import Data
from qclab.model import Model
from qclab.algorithm import Algorithm
from qclab.constants import Constants
from qclab.utils import configure_memory_logger
import qclab.models
import qclab.algorithms
import qclab.dynamics
import qclab.interfaces

configure_memory_logger()

try:
    from importlib.metadata import version as _pkg_version
except Exception:
    from importlib_metadata import version as _pkg_version

try:
    __version__ = _pkg_version("qclab")
except Exception:
    __version__ = "0+unknown"
