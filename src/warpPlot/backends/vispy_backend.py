"""Vispy rendering backend (stub).

This module is a placeholder for the Phase 4 implementation.

Phase 4 tasks (when implementation begins)
-----------------------------------------
- Use a scene canvas with subviews for mosaic panels.
- Render particles via ``visuals.Markers``.
- Maintain GPU buffers for positions / colours and update in-place.
- Colormap pipeline: convert existing normalisation + colourmap outputs
  to RGBA arrays for vispy.
- Notebook / pop-out mode:
    - pop-out native window by default.
    - optional inline with ``jupyter_rfb`` when installed.
- Grid mode (first iteration): render mapped grid with an image visual.
- Streamlines: initial implementation optional; if deferred, define a
  clear TODO and capability flag.

Risks noted in task plan
------------------------
- Event-loop integration in notebook kernels.
- More manual colour and interaction plumbing than pyvista.
"""

from .base import AbstractBackend


class VispyBackend(AbstractBackend):
    """Vispy / OpenGL rendering backend (not yet implemented)."""

    def create_figure(self, mosaic, figsize, sharex, sharey, figTitle, backendOptions):
        raise NotImplementedError(
            "The vispy backend is not yet implemented (Phase 4).  "
            "Track progress in TASKS_BACKEND_PLAN.md."
        )

    def get_axes(self):
        raise NotImplementedError("Vispy backend not yet implemented.")

    def render_panel(self, panel_key, particleState, domain, quantity, options):
        raise NotImplementedError("Vispy backend not yet implemented.")

    def update_panel(self, panel_key, panel_state, particleState, domain, quantity, options, **kwargs):
        raise NotImplementedError("Vispy backend not yet implemented.")

    def show(self):
        raise NotImplementedError("Vispy backend not yet implemented.")

    @property
    def supports_notebook_inline(self) -> bool:
        return False  # will be True (via jupyter_rfb) once Phase 4 is complete
