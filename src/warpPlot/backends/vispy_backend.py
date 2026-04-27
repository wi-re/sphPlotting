"""Vispy rendering backend (Phase 4).

Uses a vispy scene canvas with per-panel ``ViewBox`` widgets arranged in a
grid to replicate matplotlib's subplot-mosaic layout.  Particles are rendered
via ``visuals.Markers`` point clouds.  Grid data is rendered as a coloured
``visuals.Image`` visual.  Scalar colours are computed with matplotlib norms
so that all norm types (Linear, Centered, Log, SymLog) produce the correct
colour distribution.

Display modes (``backendOptions={'jupyter_backend': ...}``)
-----------------------------------------------------------
``"native"`` (default outside Jupyter)
    Opens a pop-out window via the native OS widget.
``"notebook"`` (default inside Jupyter kernels when ``jupyter_rfb`` is
    installed)
    Renders inline as a Jupyter widget.  Falls back to ``"native"``
    silently when ``jupyter_rfb`` is not installed.

Update pattern
--------------
After the initial :func:`~warpPlot.visualize.visualize` call, call
``plotState.updateQuantities(...)`` which dispatches to
:meth:`VispyBackend.update_panel`.  Existing ``Markers`` visuals are updated
in-place via ``set_data`` — no scene teardown — so position and colour
changes are fast.

Current limitations (Phase 4)
------------------------------
- Streamlines are not yet supported (``supports_streamlines = False``).
- Axis tick-label widgets are not wired (vispy does not replicate
  matplotlib-style annotated axes).
- ``sharex`` / ``sharey`` camera linking between panels is not yet wired.
- No colour-bar widget; the norm/cmap used is the same as the matplotlib
  backend so scalar ranges are still deterministic.
- ``matplotlib`` must be installed for the colormap / norm pipeline
  (same implicit dependency as the pyvista backend).
"""

from __future__ import annotations

import contextlib
import io as _io
import os as _os
import sys as _sys
import numpy as np
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

from .base import AbstractBackend
from ..enumTypes import VisualizeOptions
from ..options import PlottingOptions


# ─────────────────────────────────────────────────────────────────────────────
# Panel state dataclass
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class VispyVisualizationState:
    """Per-panel rendering state kept by the Vispy backend."""

    canvas: Any                    # shared vispy.scene.SceneCanvas reference
    view: Any                      # ViewBox for this panel

    # Marker visuals; None if the particle type is hidden or absent
    markers_fluid: Optional[Any]
    markers_boundary: Optional[Any]

    # Line visual for the domain bounding box; None when plotDomain=False
    domain_lines: Optional[Any]

    # Image visual for grid mode; None when gridVisualization is off
    image_grid: Optional[Any]

    # ColorBarWidget placed to the right of the view; None if showColorBar=False
    colorbar: Optional[Any]

    domain: Any                    # DomainDescription (detached copy)
    options: PlottingOptions

    # Per-type particle states (for update bookkeeping)
    fluidParticles: Any
    boundaryParticles: Any
    ghostParticles: Any
    assembledQuantity: Any
    rotatedQuantities: Any


# ─────────────────────────────────────────────────────────────────────────────
# Private helpers
# ─────────────────────────────────────────────────────────────────────────────

def _to_3d(positions) -> np.ndarray:
    """Convert (N, 2-or-3) float tensor → (N, 3) float32 numpy array."""
    pts = positions.detach().cpu().numpy().astype(np.float32)
    if pts.ndim == 1:
        pts = pts[:, None]
    if pts.shape[1] == 2:
        pts = np.column_stack([pts, np.zeros(len(pts), dtype=np.float32)])
    return pts


def _parse_mosaic(mosaic: str) -> Tuple[Tuple[int, int], Dict[str, Tuple[int, int]]]:
    """Parse a matplotlib mosaic string → (nrows, ncols), {label: (row, col)}.

    Handles simple rectangular mosaics like ``'AB'`` or ``'AB\\nCD'``.
    """
    rows = [r for r in mosaic.strip().split("\n") if r.strip()]
    nrows = len(rows)
    ncols = max(len(r) for r in rows)
    label_to_pos: Dict[str, Tuple[int, int]] = {}
    for r, row_str in enumerate(rows):
        for c, label in enumerate(row_str):
            if label.strip() and label not in label_to_pos:
                label_to_pos[label] = (r, c)
    return (nrows, ncols), label_to_pos


@contextlib.contextmanager
def _suppress_vispy_init_warnings():
    """Suppress libEGL/DRI3 and DPI warnings emitted during vispy canvas creation.

    The libEGL warnings are written directly to file descriptor 2 (bypassing
    Python's ``sys.stderr``), so we redirect the OS-level fd while the canvas
    is initialised.  ``sys.stderr`` is replaced simultaneously to also capture
    the Python-level "could not determine DPI" message.
    """
    devnull_fd = _os.open(_os.devnull, _os.O_WRONLY)
    saved_fd2 = _os.dup(2)
    _os.dup2(devnull_fd, 2)
    saved_stderr = _sys.stderr
    _sys.stderr = _io.StringIO()
    try:
        yield
    finally:
        _os.dup2(saved_fd2, 2)
        _os.close(devnull_fd)
        _os.close(saved_fd2)
        _sys.stderr = saved_stderr


def _get_cmap(name: str):
    """Return a matplotlib colormap by name; tries seaborn registration if needed."""
    import matplotlib as mpl  # noqa: PLC0415

    try:
        return mpl.colormaps[name]
    except KeyError:
        # Some colormaps (e.g. 'rocket', 'mako') require seaborn to be imported
        # first so they are registered with matplotlib.
        try:
            import seaborn  # noqa: F401, PLC0415
        except ImportError:
            pass
        try:
            return mpl.colormaps[name]
        except KeyError:
            pass
    # Final fallback for very old matplotlib
    import matplotlib.cm as mcm  # noqa: PLC0415
    return mcm.get_cmap(name)


def _mpl_to_vispy_cmap(cmap_name: str):
    """Build a vispy ``Colormap`` from a 256-sample matplotlib LUT.

    This bridges matplotlib colormaps (including seaborn ones) into the vispy
    ``Colormap`` object expected by ``ColorBarWidget``.
    """
    from vispy.color import Colormap as VispyColormap  # noqa: PLC0415

    mpl_cmap = _get_cmap(cmap_name)
    lut = mpl_cmap(np.linspace(0.0, 1.0, 256)).astype(np.float32)
    return VispyColormap(lut)


def _scalars_to_rgba(q_clipped: np.ndarray, norm, cmap_name: str) -> np.ndarray:
    """Convert a scalar array → (N, 4) RGBA float32 via matplotlib norm + cmap.

    Applying the matplotlib norm directly (rather than a plain linear rescale
    to ``clim``) ensures all norm types — ``Normalize``, ``CenteredNorm``,
    ``LogNorm``, ``SymLogNorm`` — produce the correct colour distribution.

    The input is ravelled to 1-D before processing so that quantities stored
    as column tensors of shape ``(N, 1)`` still produce an ``(N, 4)`` output
    rather than ``(N, 1, 4)`` which vispy ``Markers`` cannot interpret
    correctly.
    """
    q_flat = np.asarray(q_clipped).ravel()  # guarantee 1-D → output is (N, 4)
    cmap = _get_cmap(cmap_name)
    normed = norm(q_flat)
    if hasattr(normed, "data"):   # masked array returned by some norms
        normed = normed.data
    normed = np.clip(normed, 0.0, 1.0)
    return cmap(normed).astype(np.float32)


def _cmap_name(options: PlottingOptions) -> str:
    return options.colorMap.value + ("_r" if options.flipColorMap else "")


_MPL_PTS2_TO_VISPY_PX = 2.0 * (1.0 / np.pi) ** 0.5 * (96.0 / 72.0)
"""Conversion factor from matplotlib scatter ``s`` (pts²) to vispy pixel diameter.

Derivation:
  diameter_pts = 2 * sqrt(s / π)
  diameter_px  = diameter_pts * (screen_dpi / 72)  # 96 DPI assumed
"""


def _point_size(
    options: PlottingOptions,
    default_size: float,
) -> float:
    """Resolve marker size in vispy pixel diameter.

    When ``options.markerSize`` is ``None``, return *default_size*.
    When explicitly set, convert from matplotlib scatter area units (pts²) to
    vispy pixel diameter so that the same ``PlottingOptions.markerSize`` value
    produces visually comparable results across backends.
    """
    if options.markerSize is None:
        return float(default_size)
    # matplotlib s = area in pts²; vispy size = diameter in px
    return max(1.0, float(options.markerSize) ** 0.5 * _MPL_PTS2_TO_VISPY_PX)


def _auto_point_size(
    domain_,
    n_particles: int,
    panel_w_px: float,
    panel_h_px: float,
) -> float:
    """Estimate a vispy pixel diameter so each particle covers roughly one cell.

    Computes the inter-particle spacing from the domain area and particle count,
    then converts to pixels using the panel size and domain extent.  Mirrors
    the logic of ``scatter_util.computeMarkerSize`` but avoids a matplotlib
    axes object.
    """
    mn = domain_.min.cpu().numpy()
    mx = domain_.max.cpu().numpy()
    domain_w = max(float(mx[0] - mn[0]), 1e-9)
    domain_h = max(float(mx[1] - mn[1]), 1e-9)
    if n_particles > 1:
        # Estimated square-packing spacing
        dx = (domain_w * domain_h / n_particles) ** 0.5
    else:
        dx = domain_w / 32.0
    # Pixels per domain unit — use the more constrained axis
    px_per_unit = min(panel_w_px / domain_w, panel_h_px / domain_h)
    size_px = dx * px_per_unit * 0.8
    return max(2.0, float(size_px))


def _domain_line_positions_2d(domain_) -> np.ndarray:
    """Return (5, 3) float32 positions for a closed 2-D bounding-box rectangle."""
    mn = domain_.min.cpu().numpy()
    mx = domain_.max.cpu().numpy()
    return np.array([
        [mn[0], mn[1], 0.0],
        [mx[0], mn[1], 0.0],
        [mx[0], mx[1], 0.0],
        [mn[0], mx[1], 0.0],
        [mn[0], mn[1], 0.0],   # close the loop
    ], dtype=np.float32)


def _domain_line_positions_3d(domain_) -> np.ndarray:
    """Return (24, 3) float32 positions for all 12 edges of a 3-D bbox.

    Arranged as 12 disjoint segments (use ``connect='segments'``).
    """
    mn = domain_.min.cpu().numpy()
    mx = domain_.max.cpu().numpy()
    x0, y0, z0 = float(mn[0]), float(mn[1]), float(mn[2])
    x1, y1, z1 = float(mx[0]), float(mx[1]), float(mx[2])
    return np.array([
        # bottom face
        [x0, y0, z0], [x1, y0, z0],
        [x1, y0, z0], [x1, y1, z0],
        [x1, y1, z0], [x0, y1, z0],
        [x0, y1, z0], [x0, y0, z0],
        # top face
        [x0, y0, z1], [x1, y0, z1],
        [x1, y0, z1], [x1, y1, z1],
        [x1, y1, z1], [x0, y1, z1],
        [x0, y1, z1], [x0, y0, z1],
        # verticals
        [x0, y0, z0], [x0, y0, z1],
        [x1, y0, z0], [x1, y0, z1],
        [x1, y1, z0], [x1, y1, z1],
        [x0, y1, z0], [x0, y1, z1],
    ], dtype=np.float32)


def _fit_camera_to_domain(view: Any, domain_, margin: float = 0.05) -> None:
    """Set PanZoomCamera range to the domain extent with a proportional margin."""
    mn = domain_.min.cpu().numpy()
    mx = domain_.max.cpu().numpy()
    dx = float(mx[0] - mn[0])
    dy = float(mx[1] - mn[1])
    pad_x = dx * margin
    pad_y = dy * margin
    if domain_.dim == 2:
        view.camera.set_range(
            x=(float(mn[0]) - pad_x, float(mx[0]) + pad_x),
            y=(float(mn[1]) - pad_y, float(mx[1]) + pad_y),
        )
    # For 3-D, allow vispy's default auto-fit; an explicit extent-based fit
    # is deferred as a Phase 4 follow-up.


def _grid_to_rgba_image(
    gridState: Any,
    gridQuantity: Any,
    nxs: Tuple[int, int],
    gridExtent: Any,
    opts: PlottingOptions,
    cmap_name: str,
) -> Tuple[np.ndarray, Tuple[float, float, float, float]]:
    """Build a (ny, nx, 4) RGBA float32 image from ``mapToGrid`` output.

    Returns
    -------
    rgba_img : np.ndarray  shape ``(ny, nx, 4)``
    (ox, oy, dx, dy) : world-coordinate origin and per-cell spacing

    Notes
    -----
    ``generateGrid`` uses ``indexing='ij'`` (C-order), so the flat index is
    ``k = ny * ix + iy``.  Reshaping to ``(nx, ny)`` and transposing gives
    ``(ny, nx)`` = ``(H, W)`` where row corresponds to y-index and column to
    x-index — the correct layout for a vispy ``Image`` visual whose column
    axis is x and row axis is y.
    """
    from ..math import getBounds  # noqa: PLC0415

    qs, norm = getBounds(gridQuantity, opts)
    nx, ny = nxs[0], nxs[1]

    # Reshape: ij-order (nx, ny) → transpose → (ny, nx) = (H, W)
    q_img = qs.reshape(nx, ny).T
    normed = norm(q_img)
    if hasattr(normed, "data"):
        normed = normed.data
    normed = np.clip(normed, 0.0, 1.0)
    rgba_img = _get_cmap(cmap_name)(normed).astype(np.float32)  # (H, W, 4)

    # Infer world-space origin and cell spacing from grid point positions.
    # The grid uses ij-indexing so pts[ny * ix + iy, dim]:
    #   dx = pts[ny, 0] - pts[0, 0]  (change in x for ix 0→1, same iy)
    #   dy = pts[1,  1] - pts[0, 1]  (change in y for iy 0→1, same ix)
    pts = gridState.positions.detach().cpu().numpy()
    ox = float(pts[0, 0])
    oy = float(pts[0, 1])
    dx = (
        float(pts[ny, 0] - pts[0, 0])
        if nx > 1
        else float(gridExtent["max"][0] - gridExtent["min"][0])
    )
    dy = (
        float(pts[1, 1] - pts[0, 1])
        if ny > 1
        else float(gridExtent["max"][1] - gridExtent["min"][1])
    )

    return rgba_img, (ox, oy, dx, dy)


def _image_transform(ox: float, oy: float, dx: float, dy: float):
    """Return a vispy ``STTransform`` placing image pixels in world coordinates.

    A vispy ``Image`` of shape ``(H, W)`` has pixel ``(col, row)`` at canvas
    position ``(col, row)``.  The returned transform maps this to world
    ``(ox + col * dx,  oy + row * dy)``, matching the grid layout where
    column = x-index and row = y-index.
    """
    from vispy.visuals.transforms import STTransform  # noqa: PLC0415

    return STTransform(scale=(dx, dy), translate=(ox, oy))


def _in_ipykernel() -> bool:
    """Best-effort detection of a running Jupyter kernel."""
    try:
        from IPython import get_ipython  # noqa: PLC0415

        ip = get_ipython()
        return ip is not None and "IPKernelApp" in getattr(ip, "config", {})
    except Exception:
        return False


# ─────────────────────────────────────────────────────────────────────────────
# Backend implementation
# ─────────────────────────────────────────────────────────────────────────────

class VispyBackend(AbstractBackend):
    """Vispy / OpenGL rendering backend (Phase 4).

    Renders particles as ``visuals.Markers`` point clouds on a
    ``SceneCanvas`` divided into one ``ViewBox`` per mosaic panel.  Scalar
    colours are computed with matplotlib norms + colormaps so all norm types
    (Linear, Centered, Log, SymLog) render correctly.

    Pass ``backendOptions={'jupyter_backend': 'notebook'}`` to enable inline
    Jupyter display (requires ``jupyter_rfb``).
    """

    def __init__(self) -> None:
        self._canvas: Any = None
        self._grid_widget: Any = None       # top-level vispy Grid
        self._subgrids: Dict[str, Any] = {}  # label → per-panel sub-Grid
        self._axes: Dict[str, Tuple[int, int]] = {}
        self._views: Dict[str, Any] = {}    # label → ViewBox (inside sub-grid col 0)
        self._jupyter_mode: str = "native"
        self._point_size: float = 10.0
        self._shown: bool = False
        self._canvas_size: Tuple[int, int] = (960, 480)  # (px_w, px_h)
        self._shape: Tuple[int, int] = (1, 1)            # (nrows, ncols)

    # -------------------------------------------------------------------------
    # Figure management
    # -------------------------------------------------------------------------

    def create_figure(
        self,
        mosaic: str,
        figsize: Tuple[float, float],
        sharex: bool,
        sharey: bool,
        figTitle: Optional[str],
        backendOptions: Optional[dict],
    ) -> Any:
        import vispy.scene as vs  # noqa: PLC0415

        opts = backendOptions or {}

        # Resolve display mode, with graceful fallback when jupyter_rfb is absent
        default_mode = "notebook" if _in_ipykernel() else "native"
        self._jupyter_mode = opts.get("jupyter_backend", default_mode)
        if self._jupyter_mode == "notebook":
            try:
                import jupyter_rfb  # noqa: F401
            except ImportError:
                self._jupyter_mode = "native"

        self._point_size = float(opts.get("point_size", 10.0))
        self._shown = False

        shape, label_to_pos = _parse_mosaic(mosaic)
        self._axes = label_to_pos
        self._shape = shape

        px_w = int(figsize[0] * 96)
        px_h = int(figsize[1] * 96)
        self._canvas_size = (px_w, px_h)

        with _suppress_vispy_init_warnings():
            self._canvas = vs.SceneCanvas(
                title=figTitle or "warpPlot",
                size=(px_w, px_h),
                keys="interactive",
                show=False,
                bgcolor="white",
            )

        self._grid_widget = self._canvas.central_widget.add_grid(spacing=2)
        self._views = {}
        self._subgrids = {}
        for label, (r, c) in label_to_pos.items():
            # Each mosaic panel gets a sub-grid so the colorbar column is
            # fully contained within the panel's allocated space.  Within the
            # sub-grid: col 0 = ViewBox, col 1 = ColorBarWidget slot (added
            # lazily in render_panel when showColorBar is True).
            sg = self._grid_widget.add_grid(row=r, col=c, spacing=0)
            view = sg.add_view(row=0, col=0, bgcolor="white", border_color="gray")
            self._subgrids[label] = sg
            self._views[label] = view

        return self._canvas

    def get_axes(self) -> Dict[str, Tuple[int, int]]:
        return self._axes

    # -------------------------------------------------------------------------
    # Per-panel rendering
    # -------------------------------------------------------------------------

    def render_panel(
        self,
        panel_key: str,
        particleState: Any,
        domain: Any,
        quantity: Any,
        options: PlottingOptions,
    ) -> VispyVisualizationState:
        import vispy.scene.visuals as visuals  # noqa: PLC0415

        from ._render_util import prepare_particle_states  # noqa: PLC0415
        from ..math import getBounds  # noqa: PLC0415

        view = self._views[panel_key]

        # ── preprocessing ────────────────────────────────────────────────────
        domain_, opts, assembled, rotated_q, fluid, boundary, ghost = prepare_particle_states(
            particleState, domain, quantity, options
        )

        cmap = _cmap_name(opts)

        # ── auto marker size ──────────────────────────────────────────────────
        nrows, ncols = self._shape
        pw, ph = self._canvas_size
        # Approximate panel pixel size (colorbars take ~65 px per column)
        cb_px = 65 if opts.showColorBar else 0
        panel_w_px = max(1.0, pw / max(ncols, 1) - cb_px)
        panel_h_px = max(1.0, ph / max(nrows, 1))
        n_total = int(fluid.positions.shape[0]) + int(boundary.positions.shape[0])
        if opts.markerSize is None:
            point_size = _auto_point_size(domain_, n_total, panel_w_px, panel_h_px)
        else:
            point_size = _point_size(opts, self._point_size)

        # ── camera ────────────────────────────────────────────────────────────
        view.camera = "panzoom" if domain_.dim == 2 else "turntable"
        if domain_.dim == 2:
            view.camera.aspect = 1.0  # enforce equal x/y scale

        # ── panel title ───────────────────────────────────────────────────────
        if opts.plotTitle is not None:
            mn = domain_.min.cpu().numpy()
            mx = domain_.max.cpu().numpy()
            cx = 0.5 * (float(mn[0]) + float(mx[0]))
            pad = (float(mx[1]) - float(mn[1])) * 0.04
            visuals.Text(
                opts.plotTitle,
                color="black",
                font_size=10,
                pos=(cx, float(mx[1]) + pad, 0.0),
                anchor_x="center",
                anchor_y="bottom",
                parent=view.scene,
            )

        # ── domain bounding box ────────────────────────────────────────────────
        domain_lines = None
        if opts.plotDomain:
            if domain_.dim == 2:
                line_pts = _domain_line_positions_2d(domain_)
                domain_lines = visuals.Line(
                    line_pts, color="blue", connect="strip", parent=view.scene
                )
            else:
                line_pts = _domain_line_positions_3d(domain_)
                domain_lines = visuals.Line(
                    line_pts, color="blue", connect="segments", parent=view.scene
                )

        # ── grid or scatter path ───────────────────────────────────────────────
        markers_fluid, markers_boundary, image_grid = None, None, None

        if opts.gridVisualization is not None:
            # ── grid (Image visual) ───────────────────────────────────────────
            from ..grid import mapToGrid  # noqa: PLC0415

            gridState, gridQuantity, nxs, gridExtent = mapToGrid(
                particleState=particleState,
                quantity=rotated_q,
                domain=domain_,
                nx=opts.gridVisualization.resolution,
                targetNeighbors=50,
                kernel=opts.plottingKernel,
                alignment="center",
                includeFluid=opts.fluidVisualization != VisualizeOptions.Hide,
                includeBoundary=opts.boundaryVisualization != VisualizeOptions.Hide,
                gridMode=opts.gridVisualization.gridSupport,
            )

            rgba_img, (ox, oy, dx, dy) = _grid_to_rgba_image(
                gridState, gridQuantity, nxs, gridExtent, opts, cmap
            )
            image_grid = visuals.Image(rgba_img, parent=view.scene)
            image_grid.transform = _image_transform(ox, oy, dx, dy)

        else:
            # ── fluid markers ─────────────────────────────────────────────────
            if opts.fluidVisualization != VisualizeOptions.Hide and fluid.positions.shape[0] > 0:
                pts = _to_3d(fluid.positions)
                if opts.fluidVisualization == VisualizeOptions.Passive:
                    face_color = np.tile(
                        np.array([0.5, 0.5, 0.5, 0.5], dtype=np.float32), (len(pts), 1)
                    )
                else:
                    q, norm = getBounds(fluid.quantities, opts)
                    face_color = _scalars_to_rgba(q, norm, cmap)
                markers_fluid = visuals.Markers(antialias=0, parent=view.scene)
                markers_fluid.set_data(
                    pts,
                    face_color=face_color,
                    edge_color=face_color,
                    size=point_size,
                    edge_width=0,
                )

            # ── boundary markers ──────────────────────────────────────────────
            if opts.boundaryVisualization != VisualizeOptions.Hide and boundary.positions.shape[0] > 0:
                pts = _to_3d(boundary.positions)
                if opts.boundaryVisualization == VisualizeOptions.Passive:
                    face_color = np.tile(
                        np.array([0.5, 0.5, 0.5, 0.5], dtype=np.float32), (len(pts), 1)
                    )
                else:
                    q, norm = getBounds(boundary.quantities, opts)
                    face_color = _scalars_to_rgba(q, norm, cmap)
                markers_boundary = visuals.Markers(antialias=0, parent=view.scene)
                markers_boundary.set_data(
                    pts,
                    face_color=face_color,
                    edge_color=face_color,
                    size=point_size,
                    edge_width=0,
                )

        # ── camera fit ────────────────────────────────────────────────────────
        _fit_camera_to_domain(view, domain_)

        # ── colorbar ──────────────────────────────────────────────────────────
        colorbar_widget = None
        if opts.showColorBar:
            from vispy.scene.widgets.colorbar import ColorBarWidget  # noqa: PLC0415

            # Determine which scalar range and cmap to use for the colorbar.
            # Prefer the fluid (active) range; fall back to grid range.
            from ..math import getBounds  # noqa: PLC0415

            cb_range: Optional[Tuple[float, float]] = None
            cb_cmap = cmap
            if opts.gridVisualization is not None:
                # Recompute range from the grid quantity if we have it
                if image_grid is not None:
                    # We need the scalar range; re-derive from opts vMin/vMax
                    # stored on opts (getBounds side-effects fill them)
                    cb_range = (float(opts.vMin) if opts.vMin is not None else 0.0,
                                float(opts.vMax) if opts.vMax is not None else 1.0)
            elif (
                opts.fluidVisualization not in (VisualizeOptions.Hide, VisualizeOptions.Passive)
                and fluid.positions.shape[0] > 0
            ):
                q, norm = getBounds(fluid.quantities, opts)
                cb_range = (float(np.nanmin(q)), float(np.nanmax(q)))
            elif (
                opts.boundaryVisualization not in (VisualizeOptions.Hide, VisualizeOptions.Passive)
                and boundary.positions.shape[0] > 0
            ):
                q, norm = getBounds(boundary.quantities, opts)
                cb_range = (float(np.nanmin(q)), float(np.nanmax(q)))

            if cb_range is not None:
                vmin, vmax = cb_range
                colorbar_widget = ColorBarWidget(
                    cmap=_mpl_to_vispy_cmap(cb_cmap),
                    orientation="right",
                    label=panel_key,
                    label_color="black",
                    clim=(f"{vmin:.3g}", f"{vmax:.3g}"),
                )
                # Add colorbar into the panel's sub-grid at col 1.
                # Fix its width so the view (col 0) gets all remaining space.
                self._subgrids[panel_key].add_widget(colorbar_widget, row=0, col=1)
                colorbar_widget.width_min = 65
                colorbar_widget.width_max = 65
                # Rotate tick labels so they read vertically (-90° CCW, same
                # as the main colorbar label).  Horizontal numbers like
                # "0.0123" are wider than the 65-px column; rotated they fit
                # comfortably.
                colorbar_widget._colorbar._ticks[0].rotation = -90
                colorbar_widget._colorbar._ticks[1].rotation = -90

        return VispyVisualizationState(
            canvas=self._canvas,
            view=view,
            markers_fluid=markers_fluid,
            markers_boundary=markers_boundary,
            domain_lines=domain_lines,
            image_grid=image_grid,
            colorbar=colorbar_widget,
            domain=domain_,
            options=opts,
            fluidParticles=fluid,
            boundaryParticles=boundary,
            ghostParticles=ghost,
            assembledQuantity=assembled,
            rotatedQuantities=rotated_q,
        )

    def update_panel(
        self,
        panel_key: str,
        panel_state: VispyVisualizationState,
        particleState: Any,
        domain: Any,
        quantity: Any,
        options: PlottingOptions,
        **kwargs: Any,
    ) -> VispyVisualizationState:
        import copy as _copy  # noqa: PLC0415
        import vispy.scene.visuals as visuals  # noqa: PLC0415

        from ._render_util import prepare_particle_states  # noqa: PLC0415
        from ..math import getBounds  # noqa: PLC0415

        # Apply any option overrides passed as keyword arguments
        if kwargs:
            options = _copy.deepcopy(options)
            for k, v in kwargs.items():
                if hasattr(options, k):
                    setattr(options, k, v)

        domain_, opts, assembled, rotated_q, fluid, boundary, ghost = prepare_particle_states(
            particleState, domain, quantity, options
        )

        cmap = _cmap_name(opts)

        # ── auto marker size ──────────────────────────────────────────────────
        nrows, ncols = self._shape
        pw, ph = self._canvas_size
        cb_px = 65 if opts.showColorBar else 0
        panel_w_px = max(1.0, pw / max(ncols, 1) - cb_px)
        panel_h_px = max(1.0, ph / max(nrows, 1))
        n_total = int(fluid.positions.shape[0]) + int(boundary.positions.shape[0])
        if opts.markerSize is None:
            point_size = _auto_point_size(domain_, n_total, panel_w_px, panel_h_px)
        else:
            point_size = _point_size(opts, self._point_size)

        view = panel_state.view

        if opts.gridVisualization is not None:
            # ── grid update ────────────────────────────────────────────────────
            from ..grid import mapToGrid  # noqa: PLC0415

            gridState, gridQuantity, nxs, gridExtent = mapToGrid(
                particleState=particleState,
                quantity=rotated_q,
                domain=domain_,
                nx=opts.gridVisualization.resolution,
                targetNeighbors=50,
                kernel=opts.plottingKernel,
                alignment="center",
                includeFluid=opts.fluidVisualization != VisualizeOptions.Hide,
                includeBoundary=opts.boundaryVisualization != VisualizeOptions.Hide,
                gridMode=opts.gridVisualization.gridSupport,
            )

            rgba_img, (ox, oy, dx, dy) = _grid_to_rgba_image(
                gridState, gridQuantity, nxs, gridExtent, opts, cmap
            )

            if panel_state.image_grid is not None:
                panel_state.image_grid.set_data(rgba_img)
            else:
                panel_state.image_grid = visuals.Image(rgba_img, parent=view.scene)
                panel_state.image_grid.transform = _image_transform(ox, oy, dx, dy)

        else:
            # ── fluid markers update ───────────────────────────────────────────
            if opts.fluidVisualization != VisualizeOptions.Hide and fluid.positions.shape[0] > 0:
                pts = _to_3d(fluid.positions)
                if opts.fluidVisualization == VisualizeOptions.Passive:
                    face_color = np.tile(
                        np.array([0.5, 0.5, 0.5, 0.5], dtype=np.float32), (len(pts), 1)
                    )
                else:
                    q, norm = getBounds(fluid.quantities, opts)
                    face_color = _scalars_to_rgba(q, norm, cmap)

                if panel_state.markers_fluid is not None:
                    panel_state.markers_fluid.set_data(
                        pts,
                        face_color=face_color,
                        edge_color=face_color,
                        size=point_size,
                        edge_width=0,
                    )
                else:
                    panel_state.markers_fluid = visuals.Markers(antialias=0, parent=view.scene)
                    panel_state.markers_fluid.set_data(
                        pts,
                        face_color=face_color,
                        edge_color=face_color,
                        size=point_size,
                        edge_width=0,
                    )
            elif panel_state.markers_fluid is not None:
                panel_state.markers_fluid.visible = False

            # ── boundary markers update ────────────────────────────────────────
            if opts.boundaryVisualization != VisualizeOptions.Hide and boundary.positions.shape[0] > 0:
                pts = _to_3d(boundary.positions)
                if opts.boundaryVisualization == VisualizeOptions.Passive:
                    face_color = np.tile(
                        np.array([0.5, 0.5, 0.5, 0.5], dtype=np.float32), (len(pts), 1)
                    )
                else:
                    q, norm = getBounds(boundary.quantities, opts)
                    face_color = _scalars_to_rgba(q, norm, cmap)

                if panel_state.markers_boundary is not None:
                    panel_state.markers_boundary.set_data(
                        pts,
                        face_color=face_color,
                        edge_color=face_color,
                        size=point_size,
                        edge_width=0,
                    )
                else:
                    panel_state.markers_boundary = visuals.Markers(antialias=0, parent=view.scene)
                    panel_state.markers_boundary.set_data(
                        pts,
                        face_color=face_color,
                        edge_color=face_color,
                        size=point_size,
                        edge_width=0,
                    )
            elif panel_state.markers_boundary is not None:
                panel_state.markers_boundary.visible = False

        panel_state.fluidParticles = fluid
        panel_state.boundaryParticles = boundary
        panel_state.ghostParticles = ghost
        panel_state.assembledQuantity = assembled
        panel_state.rotatedQuantities = rotated_q
        panel_state.options = opts

        # ── update colorbar clim / cmap ────────────────────────────────────────
        if panel_state.colorbar is not None and opts.showColorBar:
            from vispy.color import Colormap as VispyColormap  # noqa: PLC0415

            new_range: Optional[Tuple[float, float]] = None
            if opts.gridVisualization is None:
                if (
                    opts.fluidVisualization not in (VisualizeOptions.Hide, VisualizeOptions.Passive)
                    and fluid.positions.shape[0] > 0
                ):
                    q, _ = getBounds(fluid.quantities, opts)
                    new_range = (float(np.nanmin(q)), float(np.nanmax(q)))
                elif (
                    opts.boundaryVisualization not in (VisualizeOptions.Hide, VisualizeOptions.Passive)
                    and boundary.positions.shape[0] > 0
                ):
                    q, _ = getBounds(boundary.quantities, opts)
                    new_range = (float(np.nanmin(q)), float(np.nanmax(q)))
            if new_range is not None:
                vmin, vmax = new_range
                panel_state.colorbar.clim = (f"{vmin:.3g}", f"{vmax:.3g}")
            panel_state.colorbar.cmap = _mpl_to_vispy_cmap(cmap)

        # Trigger a canvas redraw so the updated data is displayed
        self._canvas.update()

        return panel_state

    # -------------------------------------------------------------------------
    # Display
    # -------------------------------------------------------------------------

    def show(self) -> None:
        """Flush / display the canvas.

        First call
            Shows the canvas (native OS window or inline Jupyter widget).
        Subsequent calls
            For ``"native"`` mode: pumps the app event loop so pending redraws
            are processed immediately — required for live updates inside a
            synchronous ``for`` loop.
            For ``"notebook"`` mode: directly calls the jupyter_rfb draw
            pipeline without asyncio scheduling, so frames are pushed
            synchronously via ZMQ and appear in the widget mid-loop.
            (``nest_asyncio`` is intentionally avoided: it is broken in
            Python 3.13 due to tightened ``contextvars.Context.run``
            re-entrancy guards.)
        """
        if self._shown:
            # Subsequent call: flush pending redraws
            self._canvas.update()
            if self._jupyter_mode == "native":
                try:
                    self._canvas.app.process_events()
                except Exception:
                    pass
            elif self._jupyter_mode == "notebook":
                self._flush_notebook_frame()
            return
        self._canvas.show()
        if self._jupyter_mode == "notebook":
            try:
                from IPython.display import display as ip_display  # noqa: PLC0415

                ip_display(self._canvas)
            except Exception:
                pass
        self._shown = True

    def _flush_notebook_frame(self) -> None:
        """Push the current frame to the notebook widget synchronously.

        jupyter_rfb throttles rendering via a ``max_buffered_frames`` (= 2)
        check: if more frames are *in-flight* (sent but not yet ACK-ed by the
        browser) it skips drawing.  The browser can only ACK frames between
        kernel messages, so inside a synchronous ``for`` loop every frame
        after the second would be silently dropped.

        We bypass the throttle by calling ``get_frame()`` and
        ``_rfb_send_frame()`` directly and resetting ``_frame_feedback`` to
        pretend all previous frames were acknowledged before each call.  The
        render and ZMQ send are fully synchronous; no asyncio is involved.
        """
        try:
            backend = self._canvas._backend
            if not (hasattr(backend, "get_frame") and hasattr(backend, "_rfb_send_frame")):
                return
            # Pretend all in-flight frames were acknowledged so the throttle
            # never blocks us.
            backend._frame_feedback = {"index": backend._rfb_frame_index}
            with backend._output_context:
                array = backend.get_frame()
                if array is not None:
                    backend._rfb_send_frame(array)
        except Exception:
            pass

    # -------------------------------------------------------------------------
    # Capability flags
    # -------------------------------------------------------------------------

    @property
    def supports_streamlines(self) -> bool:
        # TODO (Phase 4 follow-up): implement with vispy Line visuals + seed
        # integration; leave False until that work lands.
        return False

    @property
    def supports_grid(self) -> bool:
        return True

    @property
    def supports_notebook_inline(self) -> bool:
        try:
            import jupyter_rfb  # noqa: F401

            return True
        except ImportError:
            return False
