"""Abstract backend interface for warpPlot rendering.

All concrete backends must subclass :class:`AbstractBackend` and implement
every ``@abstractmethod``.  Optional capabilities are declared via the
``supports_*`` properties so callers can gate features without catching
``NotImplementedError`` at runtime.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple


class AbstractBackend(ABC):
    """Common interface for all rendering backends.

    Lifecycle
    ---------
    1. :meth:`create_figure` — allocate the figure/scene and panels.
    2. :meth:`render_panel`  — once per panel at first draw.
    3. :meth:`show`          — flush / display after all panels are ready.
    4. :meth:`update_panel`  — called on each animation/update step.
    """

    # ------------------------------------------------------------------
    # Figure / scene management
    # ------------------------------------------------------------------

    @abstractmethod
    def create_figure(
        self,
        mosaic: str,
        figsize: Tuple[float, float],
        sharex: bool,
        sharey: bool,
        figTitle: Optional[str],
        backendOptions: Optional[dict],
    ) -> Any:
        """Create the top-level figure container and return it.

        The return value is stored in ``PlotState.fig`` and is intentionally
        backend-opaque.
        """

    @abstractmethod
    def get_axes(self) -> Dict[str, Any]:
        """Return a dict mapping panel label → backend-specific axes handle.

        The dict is iterated to drive per-panel rendering.  Keys must match
        the labels produced by the ``mosaic`` string passed to
        :meth:`create_figure`.
        """

    # ------------------------------------------------------------------
    # Per-panel rendering
    # ------------------------------------------------------------------

    @abstractmethod
    def render_panel(
        self,
        panel_key: str,
        particleState: Any,
        domain: Any,
        quantity: Any,
        options: Any,
    ) -> Any:
        """Render panel *panel_key* for the first time.

        Returns a backend-specific panel state object that is kept in
        ``PlotState.plotStates[panel_key]`` and passed back to
        :meth:`update_panel` on subsequent frames.
        """

    @abstractmethod
    def update_panel(
        self,
        panel_key: str,
        panel_state: Any,
        particleState: Any,
        domain: Any,
        quantity: Any,
        options: Any,
        **kwargs: Any,
    ) -> Any:
        """Update an already-rendered panel in place.

        Must return the (possibly mutated) panel state so the caller can
        store it back into ``PlotState.plotStates``.
        """

    # ------------------------------------------------------------------
    # Display
    # ------------------------------------------------------------------

    @abstractmethod
    def show(self) -> None:
        """Flush / display the figure after all panels are rendered.

        For matplotlib this is ``tight_layout()``.  For interactive backends
        this may start an event loop or attach to a notebook widget.
        """

    # ------------------------------------------------------------------
    # Optional: figure title updates
    # ------------------------------------------------------------------

    def update_figure_title(self, title: Optional[str]) -> None:
        """Update the figure title in-place.

        Backends that support runtime title edits should override this
        method. The default implementation is a no-op.
        """
        _ = title

    # ------------------------------------------------------------------
    # Optional: file export
    # ------------------------------------------------------------------

    def export(self, filepath: str, **kwargs: Any) -> None:
        """Export the current scene to *filepath*.

        The file format is inferred from the extension of *filepath*
        (e.g. ``".png"``, ``".pdf"``, ``".svg"``).  Keyword arguments are
        forwarded to the underlying export call; common ones include ``dpi``
        (matplotlib / vispy) and ``transparent`` (matplotlib).

        Concrete backends that support export must override this method.
        The base implementation always raises :exc:`NotImplementedError`.
        """
        raise NotImplementedError(
            f"The '{type(self).__name__}' backend does not support export."
        )

    # ------------------------------------------------------------------
    # Capability flags
    # ------------------------------------------------------------------

    @property
    def supports_streamlines(self) -> bool:
        """Whether this backend can render streamlines."""
        return False

    @property
    def supports_grid(self) -> bool:
        """Whether this backend supports grid/image-data visualization."""
        return False

    @property
    def supports_notebook_inline(self) -> bool:
        """Whether this backend can render inline inside a Jupyter notebook."""
        return False
