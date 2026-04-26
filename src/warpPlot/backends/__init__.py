"""Backend plug-in package for warpPlot.

Importing this package does NOT import any heavy optional dependencies
(pyvista, vispy, etc.).  Each backend module is loaded lazily by
:mod:`warpPlot.backends.factory` only when the user actually requests it.
"""

from .base import AbstractBackend
from .factory import get_backend

__all__ = ["AbstractBackend", "get_backend"]
