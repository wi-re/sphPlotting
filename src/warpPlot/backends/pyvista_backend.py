"""PyVista rendering backend (stub).

This module is a placeholder for the Phase 3 implementation.
Attempting to instantiate :class:`PyVistaBackend` before the implementation
is complete will raise ``NotImplementedError`` with a clear message.

Phase 3 tasks (when implementation begins)
-----------------------------------------
- Use ``pv.Plotter(shape=...)`` for mosaic-like multi-panel layout.
- Represent particle data as ``pv.PolyData`` point clouds.
- Attach scalar arrays via ``point_data[...] = values`` for colour mapping.
- Implement efficient scalar-only updates (no full scene rebuild) via
  ``point_data`` mutation + ``render()`` / ``update_scalar_bar_range()``.
- Notebook support via backend option ``jupyter_mode``:
    - ``"trame"``  — inline via trame (requires ``trame`` package)
    - ``"static"`` — static screenshot
    - ``"none"``   — pop-out window (default)
- Grid visualisation: map grid output to ``pv.ImageData``.
- Streamlines: use VTK streamline filter when vector fields are available
  on ImageData; document as unsupported otherwise.

Risks noted in task plan
------------------------
- Trame setup differences across environments.
- VTK render-loop behaviour in remote/headless notebook sessions.
"""

from .base import AbstractBackend


class PyVistaBackend(AbstractBackend):
    """PyVista / VTK-backed rendering backend (not yet implemented)."""

    def create_figure(self, mosaic, figsize, sharex, sharey, figTitle, backendOptions):
        raise NotImplementedError(
            "The pyvista backend is not yet implemented (Phase 3).  "
            "Track progress in TASKS_BACKEND_PLAN.md."
        )

    def get_axes(self):
        raise NotImplementedError("PyVista backend not yet implemented.")

    def render_panel(self, panel_key, particleState, domain, quantity, options):
        raise NotImplementedError("PyVista backend not yet implemented.")

    def update_panel(self, panel_key, panel_state, particleState, domain, quantity, options, **kwargs):
        raise NotImplementedError("PyVista backend not yet implemented.")

    def show(self):
        raise NotImplementedError("PyVista backend not yet implemented.")

    @property
    def supports_streamlines(self) -> bool:
        return False  # will be True once Phase 3 is complete

    @property
    def supports_grid(self) -> bool:
        return False  # will be True once Phase 3 is complete

    @property
    def supports_notebook_inline(self) -> bool:
        return False  # will be True (via trame) once Phase 3 is complete
