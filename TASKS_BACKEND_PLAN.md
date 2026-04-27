# Backend Expansion Task Plan

## Goals

- Add backend selection to `visualize(...)` and update paths without using global state.
- Keep all new backends optional so users install only what they need.
- Implement two interactive rendering backends:
  - `pyvista` (VTK-backed, high-level, ParaView-friendly ecosystem)
  - `vispy` (OpenGL-first, high performance)
- Add a lightweight VTK export path for external ParaView workflows.

## Non-Goals

- No requirement to make old demo internals compatible with all new backends.
- No forced migration away from the current matplotlib backend.
- No hard dependency on ParaView or full VTK Python bindings for file export.

---

## Phase 0: API Contract and Scope Lock

### Tasks

- Define backend enum/string contract:
  - `"matplotlib"` (default)
  - `"pyvista"`
  - `"vispy"`
- Add backend parameter to top-level API:
  - `visualize(..., backend: str = "matplotlib", backendOptions: Optional[dict] = None)`
- Add backend parameter to update entry points where needed:
  - `PlotState.updateQuantities(...)`
  - backend-aware internal update dispatcher
- Define minimal cross-backend feature matrix:
  - required: notebook display + fast updates
  - optional: grid visualization, streamlines, export quality

### Acceptance Criteria

- Public API docs show backend selection at call site (not global variable).
- Existing matplotlib users run unchanged unless they pass a backend.
- A single `plotter` object still supports update operations for its backend.

---

## Phase 1: Optional Dependency Packaging

### Tasks

- Extend `pyproject.toml` extras:
  - `plot-matplotlib`: `matplotlib`, `ipympl`, `ipywidgets`
  - `plot-pyvista`: `pyvista`, `trame` (optional but recommended for notebook inline)
  - `plot-vispy`: `vispy`, `jupyter_rfb` (optional inline support)
  - `vtk-export`: lightweight writer dependency (prefer `pyevtk`)
  - `plot-all`: union of all optional plotting extras
- Keep base dependencies unchanged (no backend-specific heavy packages in core install).
- Implement import guards with actionable error messages.

### Acceptance Criteria

- `pip install -e .` still works without graphics extras.
- `pip install -e .[plot-pyvista]` enables pyvista backend only.
- Calling unavailable backend raises clear install hint.

---

## Phase 2: Internal Backend Abstraction Layer

### Tasks

- Create internal backend package:
  - `src/warpPlot/backends/base.py`
  - `src/warpPlot/backends/matplotlib_backend.py`
  - `src/warpPlot/backends/pyvista_backend.py`
  - `src/warpPlot/backends/vispy_backend.py`
  - `src/warpPlot/backends/factory.py`
- Define backend interface methods:
  - `create_figure(mosaic, figsize, sharex, sharey, figTitle, backendOptions)`
  - `render_panel(panel_key, particleState, domain, quantity, options)`
  - `update_panel(panel_key, particleState, domain, quantity, options)`
  - `show()` and/or notebook attach method
  - `export(...)` (optional per backend)
- Refactor current matplotlib-specific logic behind the interface with minimal behavior changes.

### Acceptance Criteria

- Matplotlib path runs through the backend interface and keeps current outputs.
- `visualize(...)` only decides backend and delegates rendering.
- `updateQuantities(...)` dispatches to backend implementation without branching in user code.

---

## Phase 3: PyVista Backend Implementation

### Tasks

- Implement `PyVistaBackend`:
  - Use `pv.Plotter(shape=...)` for mosaic-like layout.
  - Represent particles as `pv.PolyData` point clouds.
  - Attach scalar arrays for color mapping.
  - Support efficient scalar updates (`point_data[...] = new_values` + render/update call).
- Notebook support:
  - backend option for `jupyter_mode`: `"none" | "trame" | "static"`.
  - default to non-blocking pop-out where available.
- Grid visualization:
  - map grid output to `pv.ImageData` or structured surface mesh.
- Streamline support (initial scope):
  - if vector fields available on grid/image data, use VTK streamline filter.
  - if not available, document temporary fallback/unsupported state.

### Acceptance Criteria

- User can call:
  - `visualize(..., backend="pyvista")`
- Plot appears in notebook (trame) or pop-out window based on backend options.
- `plotter.updateQuantities(...)` updates colors without recreating full scene.

### Risks

- Trame setup differences across environments.
- VTK render loop behavior in remote/headless notebook sessions.

---

## Phase 4: Vispy Backend Implementation ✅ COMPLETE

### Tasks

- [x] Implement `VispyBackend` (`src/warpPlot/backends/vispy_backend.py`):
  - `SceneCanvas` + vispy `Grid` widget divides the canvas into per-panel `ViewBox`es.
  - Particles rendered via `visuals.Markers` with per-particle RGBA arrays.
  - In-place updates via `markers.set_data(...)` — no scene teardown.
- [x] Colormap pipeline:
  - `_scalars_to_rgba`: applies matplotlib norm (all norm types) → `_get_cmap` → RGBA float32.
- [x] Notebook/pop-out mode:
  - Default `"native"` (pop-out window) outside Jupyter.
  - Default `"notebook"` (inline `jupyter_rfb` widget) inside Jupyter kernels.
  - Graceful silent fallback to `"native"` when `jupyter_rfb` is not installed.
- [x] Grid mode: `visuals.Image` visual with `STTransform` for world-coordinate placement.
- [x] Streamlines: deferred — `supports_streamlines = False`, clear TODO in source.

### Acceptance Criteria

- [x] User can call `visualize(..., backend="vispy")`.
- [x] Updates are fast for particle color changes and position updates.
- [x] Missing optional inline dependency does not break pop-out mode.

### Known Limitations (post-Phase-4 follow-up)

- Streamlines not yet implemented.
- No colour-bar widget (norm/cmap is correct; bar is cosmetic).
- Axis tick-label widgets not wired.
- `sharex`/`sharey` camera linking not yet implemented.

### Risks

- Event loop integration in notebook kernels.
- More manual color and interaction plumbing than pyvista.

---

## Phase 5: Lightweight VTK Export for ParaView

### Tasks

- Add exporter module:
  - `src/warpPlot/export/vtk_export.py`
- Implement functions:
  - `export_particles_vtp(path, particleState, quantities: dict, domain=None, metadata=None)`
  - `export_grid_vti(path, gridState, gridQuantity, resolution, extent, metadata=None)`
  - optional `write_time_series_pvd(...)` index helper
- File format design:
  - point cloud export as VTK XML PolyData (`.vtp`)
  - grid export as VTK XML ImageData (`.vti`) when regular grid exists
- Keep exporter independent from pyvista/vispy runtime.
- Prefer lightweight writer dependency (`pyevtk`) or minimal internal XML writer.

### Acceptance Criteria

- Files open directly in ParaView with scalar arrays and coordinates intact.
- Users can export from existing matplotlib backend workflows.
- Export path works without installing pyvista/vispy.

### Risks

- Choosing writer implementation that balances simplicity and long-term maintenance.

---

## Phase 6: PlotState and Update Flow Adjustments

### Tasks

- Update `PlotState` to store backend handle and panel mapping in a backend-neutral way.
- Ensure `updateQuantities(...)` path does:
  - quantity preprocess (existing mapping/ops)
  - backend update dispatch
  - optional redraw/flush per backend
- Add capability flags to avoid runtime surprises:
  - `supports_streamlines`
  - `supports_grid`
  - `supports_notebook_inline`

### Acceptance Criteria

- Single `plotter` interface works for all enabled backends.
- Unsupported feature requests return readable errors/warnings with alternatives.

---

## Phase 7: Documentation and Examples

### Tasks

- Update README install section with extras table and examples.
- Add one minimal example per backend.
- Add one export-to-ParaView example.
- Add backend capability matrix (scatter/grid/streamline/inline/pop-out/export).

### Acceptance Criteria

- New user can choose a backend and install only required extras in one command.
- Users understand how to export to ParaView even if they stay on matplotlib.

---

## Phase 8: Validation and Performance Checks

### Tasks

- Add smoke tests that skip when optional deps are absent.
- Add backend selection tests for dispatcher behavior and import guards.
- Add basic performance benchmark notebook/script:
  - 30k, 100k particles
  - initial render time
  - update latency
- Validate on CPU-only environment and CUDA-enabled environment.

### Acceptance Criteria

- CI passes with core install and optional dependency test jobs.
- Measured update performance improves for high-particle scenarios on non-matplotlib backends.

---

## Suggested File-Level Work Queue

1. `pyproject.toml` extras and dependency guards.
2. `src/warpPlot/backends/*` base, factory, matplotlib adapter.
3. `src/warpPlot/visualize.py` backend parameter + dispatch.
4. `src/warpPlot/update.py` backend-aware update dispatch.
5. `src/warpPlot/state.py` backend-neutral state extensions.
6. `src/warpPlot/backends/pyvista_backend.py` first implementation.
7. `src/warpPlot/backends/vispy_backend.py` first implementation.
8. `src/warpPlot/export/vtk_export.py` lightweight export path.
9. `README.md` docs + usage matrix.

---

## Initial Milestone Split

### Milestone A (Foundation)

- Phase 0, 1, 2 complete.
- Matplotlib behavior unchanged but now routed through backend abstraction.

### Milestone B (PyVista + Export)

- Phase 3 and 5 complete.
- Users get high-value pop-out/inline option plus ParaView file output.

### Milestone C (Vispy + Perf)

- Phase 4 and 8 complete.
- Users get highest-performance GPU rendering option.

---

## Open Decisions to Resolve Early

- Whether streamlines are mandatory in v1 for vispy backend.
- Whether VTK exporter should use `pyevtk` or an internal writer for `.vtp/.vti`.
- Default backend behavior in headless/notebook-remote sessions.
- Whether to expose backend-specific kwargs through typed dataclasses instead of raw dict.
