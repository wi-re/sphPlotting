"""Backend factory for warpPlot.

Call :func:`get_backend` with a backend name (or :class:`~warpPlot.Backend`
enum member) to receive a fully initialised backend instance ready for use
by :func:`~warpPlot.visualize`.

Adding a new backend
--------------------
1. Create ``src/warpPlot/backends/<name>_backend.py`` subclassing
   :class:`~warpPlot.backends.base.AbstractBackend`.
2. Add a branch in :func:`get_backend` below.
3. Add the corresponding extras entry in ``pyproject.toml``.
"""

from typing import Optional

from .base import AbstractBackend

# Canonical string names (lower-cased) recognised by the factory.
_MATPLOTLIB = "matplotlib"
_PYVISTA    = "pyvista"
_VISPY      = "vispy"

_KNOWN_BACKENDS = {_MATPLOTLIB, _PYVISTA, _VISPY}


def get_backend(backend: str, backendOptions: Optional[dict] = None) -> AbstractBackend:
    """Return an ``AbstractBackend`` instance for *backend*.

    Parameters
    ----------
    backend:
        One of ``"matplotlib"``, ``"pyvista"``, or ``"vispy"``.
        A :class:`~warpPlot.Backend` enum value is also accepted (it
        compares equal to its string value because the enum inherits ``str``).
    backendOptions:
        Forwarded verbatim to the backend; currently unused by the factory
        itself but reserved for future use (e.g. choosing a pyvista renderer).

    Raises
    ------
    ValueError
        If *backend* is not one of the recognised names.
    ImportError
        If the optional dependency for the chosen backend is not installed.
        The error message includes the corresponding ``pip install`` hint.
    """
    name = str(backend).lower()

    if name not in _KNOWN_BACKENDS:
        raise ValueError(
            f"Unknown backend '{backend}'.  "
            f"Choose one of: {sorted(_KNOWN_BACKENDS)}."
        )

    if name == _MATPLOTLIB:
        from .matplotlib_backend import MatplotlibBackend  # noqa: PLC0415
        return MatplotlibBackend()

    if name == _PYVISTA:
        try:
            import pyvista  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "The pyvista backend requires pyvista (and optionally trame).  "
                "Install it with:  pip install 'sphWarpPlotting[plot-pyvista]'"
            ) from exc
        from .pyvista_backend import PyVistaBackend  # noqa: PLC0415
        return PyVistaBackend()

    if name == _VISPY:
        try:
            import vispy  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "The vispy backend requires vispy (and optionally jupyter_rfb).  "
                "Install it with:  pip install 'sphWarpPlotting[plot-vispy]'"
            ) from exc
        from .vispy_backend import VispyBackend  # noqa: PLC0415
        return VispyBackend()

    # Unreachable – kept as a safety net.
    raise ValueError(f"Unhandled backend name '{name}'.")  # pragma: no cover
