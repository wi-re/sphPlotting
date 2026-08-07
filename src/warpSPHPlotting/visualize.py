from matplotlib.collections import PathCollection
from matplotlib.colorbar import Colorbar
from matplotlib.streamplot import StreamplotSet
from warpSPHCore import *
from warpSPHPlotting.update import updatePlot
from .domain import processDomain
from .grid import generateGrid, mapToGrid, gridVisualize
from .scatter import scatterVisualize
from .streamLines import streamLinePlot, updateStreamLinePlot
from .state import PlottingParticleState, rotateState, filterState
from .options import PlottingOptions
from .math import getBounds, buildRotationMatrix, mapQuantity
from .enumTypes import VisualizeOptions, Mapping, StreamLineLocation
from typing import Optional, Tuple
import torch
import matplotlib.colors as colors
from matplotlib.collections import PathCollection
from matplotlib.colorbar import Colorbar
from .util import verbosePrint
from .state import VisualizationState, assembleQuantity
import copy

def visualizeParticlesNew(
    fig, axis,
    particleState,
    domain: DomainDescription,
    quantity: torch.Tensor,
    options: PlottingOptions = PlottingOptions(),
    verbose = False,
    **kwargs
):
    """
    High-level visualization orchestrator for SPH particle data.
    
    Pipeline: Domain Setup → Quantity Assembly → Rotation → 
              Operation → Mapping → Filtering → Rendering
    
    Args:
        quantity: Per-particle scalar or per-particle-type tuple of scalars
        options: Controls all visualization aspects (see PlottingOptions)
        
    Returns:
        VisualizationState with matplotlib artists and intermediate data
        for use in animations/updates
    """
    options = copy.deepcopy(options)
    for arg in kwargs:
        if hasattr(options, arg):
            setattr(options, arg, kwargs[arg])

    verbosePrint(verbose, "Processing domain and setting up axis...")
    domain_, rotMat, invRotMat = processDomain(fig, axis, domain, options, (particleState.masses / particleState.densities).mean().item())  

    verbosePrint(verbose, "Assembling quantity for visualization...", "Current quantity: ", quantity.shape if isinstance(quantity, torch.Tensor) else [q.shape if q is not None else None for q in quantity])
    assembled = assembleQuantity(particleState, quantity)
    verbosePrint(verbose, "Assembled quantity shape: ", assembled.shape)

    verbosePrint(verbose, "Rotating particle state for filtering...")
    plotState = PlottingParticleState(
        positions = particleState.positions,
        supports = particleState.supports,
        masses = particleState.masses,
        densities = particleState.densities,
        kinds = particleState.kinds,
        quantities = assembled
    )
    # We are rotating the state back to the original orientation for filtering thus we are using the inverse rotation matrix
    rotatedState = rotateState(plotState, invRotMat)
    streamLineQuantity = None

    if options.gridVisualization is not None and options.gridVisualization.streamLines and options.gridVisualization.streamLineOperationLocation == StreamLineLocation.BeforeOperation:
        streamLineQuantity = rotatedState.quantities.clone()

    # Apply the operation properties to the quantities if specified in the options
    if options.plottingOperation is not None:
        if not isinstance(options.plottingOperation, list):
            operationProperties = [options.plottingOperation]
        else:
            operationProperties = options.plottingOperation
        for o, op in enumerate(operationProperties):
            verbosePrint(verbose, f"Applying plotting operation {o}...")
            verbosePrint(verbose, "Quantity shape before plotting operation: ", rotatedState.quantities.shape, 'min: ', torch.min(rotatedState.quantities).item(), 'max: ', torch.max(rotatedState.quantities).item())
            rotatedState.quantities = warpOperation(
                queryParticles = rotatedState,
                queryValues = rotatedState.quantities,
                operationProperties = op,
                adjacency = None, # We can consider adding adjacency-based operations in the future
                domain = domain_
            )
            verbosePrint(verbose, "Plotting operation applied. New quantity shape: ", rotatedState.quantities.shape, 'min: ', torch.min(rotatedState.quantities).item(), 'max: ', torch.max(rotatedState.quantities).item())

    if options.gridVisualization is not None and options.gridVisualization.streamLines and options.gridVisualization.streamLineOperationLocation == StreamLineLocation.BeforeMapping:
        streamLineQuantity = rotatedState.quantities.clone()

    # Apply mapping to the quantities if specified in the options
    if options.mapping != Mapping.none:
        verbosePrint(verbose, f"Applying mapping {options.mapping} to quantities...")
        verbosePrint(verbose, "Quantity shape before mapping: ", rotatedState.quantities.shape, 'min: ', torch.min(rotatedState.quantities).item(), 'max: ', torch.max(rotatedState.quantities).item())
        rotatedState.quantities = mapQuantity(rotatedState.quantities, options.mapping)
        verbosePrint(verbose, "Mapping applied. New quantity shape: ", rotatedState.quantities.shape, 'min: ', torch.min(rotatedState.quantities).item(), 'max: ', torch.max(rotatedState.quantities).item())

    if options.gridVisualization is not None and options.gridVisualization.streamLines and options.gridVisualization.streamLineOperationLocation == StreamLineLocation.AfterMapping:
        streamLineQuantity = rotatedState.quantities.clone()

    # Checking quantity shape
    if rotatedState.quantities.shape[0] != rotatedState.positions.shape[0]:
        raise ValueError(f"Quantity length {rotatedState.quantities.shape[0]} does not match number of particles {rotatedState.positions.shape[0]} after applying plotting operation.")
    if rotatedState.quantities.ndim == 2:
        raise ValueError(f"Quantity has more than 1 component per particle (shape: {rotatedState.quantities.shape}). Please specify how to map the quantity to a scalar value for visualization (e.g., using options.mapping) or ensure the plotting operation returns a single scalar value per particle.")

    # Disentangle the different particle types for separate visualization and easier handling of color mapping based on particle type
    verbosePrint(verbose, "Filtering particle state by type...")
    fluidParticles = filterState(rotatedState, rotatedState.quantities, kind=ParticleType.Fluid) 
    boundaryParticles = filterState(rotatedState, rotatedState.quantities, kind=ParticleType.Boundary)
    ghostParticles = filterState(rotatedState, rotatedState.quantities, kind=ParticleType.Ghost)

    if options.gridVisualization is None:
        verbosePrint(verbose, "Creating scatter visualizations for fluid, boundary, and ghost particles...")
        verbosePrint(verbose, f"Fluid particles: {fluidParticles.positions.shape[0]}, Boundary particles: {boundaryParticles.positions.shape[0]}, Ghost particles: {ghostParticles.positions.shape[0]}")
        fluidVisualized = options.fluidVisualization == VisualizeOptions.Visualize and fluidParticles.positions.shape[0] > 0
        boundaryVisualized = options.boundaryVisualization == VisualizeOptions.Visualize and boundaryParticles.positions.shape[0] > 0

        fluidQs = None
        boundaryQs = None
        sharedNorm = None
        if fluidVisualized and boundaryVisualized:
            combinedQuantity = torch.cat((fluidParticles.quantities, boundaryParticles.quantities), dim=0)
            combinedQs, sharedNorm = getBounds(combinedQuantity, options)
            fluidCount = fluidParticles.quantities.shape[0]
            fluidQs = combinedQs[:fluidCount]
            boundaryQs = combinedQs[fluidCount:]

        colorBarOnFluid = fluidVisualized
        colorBarOnBoundary = (not fluidVisualized) and boundaryVisualized

        verbosePrint(verbose, "Plotting Fluid Particles...")
        fluidSc = scatterVisualize(
            fig,
            axis,
            fluidParticles,
            domain_,
            options,
            variant=options.fluidVisualization,
            precomputedQs=fluidQs,
            precomputedNorm=sharedNorm,
            attachColorBar=colorBarOnFluid,
        )
        verbosePrint(verbose, "Plotting Boundary Particles...")
        boundarySc = scatterVisualize(
            fig,
            axis,
            boundaryParticles,
            domain_,
            options,
            variant=options.boundaryVisualization,
            precomputedQs=boundaryQs,
            precomputedNorm=sharedNorm,
            attachColorBar=colorBarOnBoundary,
        )
        verbosePrint(verbose, "Plotting Ghost Particles...")
        ghostSc = scatterVisualize(fig, axis, ghostParticles, domain_, options, variant=VisualizeOptions.Hide)
        grid = None
        streamLines = None
        gridState = None
        gridQuantity = None
    else:
        gridState, gridQuantity, nxs, gridExtent = mapToGrid(
            particleState = particleState,
            quantity = rotatedState.quantities,
            domain = domain_,
            nx = options.gridVisualization.resolution,
            targetNeighbors = 50,
            kernel = options.plottingKernel,
            alignment = 'center',
            includeFluid = options.fluidVisualization != VisualizeOptions.Hide,
            includeBoundary = options.boundaryVisualization != VisualizeOptions.Hide,
            gridMode = options.gridVisualization.gridSupport
        )

        grid = gridVisualize(fig, axis, gridState, gridQuantity, nxs, gridExtent, options)
        fluidSc, boundarySc, ghostSc = (None, None, None)

        if options.gridVisualization.streamLines:
            verbosePrint(verbose, "Computing streamlines...")
            if options.gridVisualization.streamLineOperation is not None:
                verbosePrint(verbose, "Applying streamline operation to quantity...")
                streamLineQuantity = warpOperation(
                    queryParticles = rotatedState,
                    queryValues = streamLineQuantity,
                    operationProperties = OperationProperties(
                        kernel = options.plottingKernel,
                        operation = options.gridVisualization.streamLineOperation,
                        gradientMode = options.gridVisualization.streamLineGradientMode,
                        laplacianMode = options.gridVisualization.streamLineLaplaceMode,  
                        operationMode = OperationDirection.NoGhost
                    ),
                    adjacency = None, # We can consider adding adjacency-based operations in the future
                    domain = domain_
                )

            verbosePrint(verbose, "Generating streamlines...")
            verbosePrint(verbose, "Streamline quantity shape: ", streamLineQuantity.shape)
            verbosePrint(verbose, "Mapping to Grid for streamlines...")

            streamLineGridState, streamLineGridQuantity, streamLinenxs, streamLineGridExtent = mapToGrid(
                particleState = particleState,
                quantity = streamLineQuantity,
                domain = domain_,
                nx = options.gridVisualization.resolution,
                targetNeighbors = 50,
                kernel = options.plottingKernel,
                alignment = 'center',
                includeFluid = options.fluidVisualization != VisualizeOptions.Hide,
                includeBoundary = options.boundaryVisualization != VisualizeOptions.Hide,
                gridMode = options.gridVisualization.gridSupport
            )
            verbosePrint(verbose, 'Mapped streamline quantity to grid. Shape: ', streamLineGridQuantity.shape)
            verbosePrint(verbose, "Plotting streamlines...")

            streamLines = streamLinePlot(fig, axis, streamLineGridState, streamLineGridQuantity, streamLinenxs, streamLineGridExtent, options)
        else:
            streamLines = None

    if options.plotTitle is not None:
        if options.plotTitleGap is not None:
            axis.set_title(options.plotTitle, pad=float(options.plotTitleGap))
        else:
            axis.set_title(options.plotTitle)
    verbosePrint(verbose, "Visualization setup complete.")
    return VisualizationState(
        fig = fig,
        axis = axis,
        domain = domain_,
        options = options,
        assembledQuantity = assembled,
        fluidParticles = fluidParticles,
        boundaryParticles = boundaryParticles,
        ghostParticles = ghostParticles,
        fluidScatterResult = fluidSc,
        boundaryScatterResult = boundarySc,
        ghostScatterResult = ghostSc,
        gridResult = grid,
        streamLines = streamLines
    )


from typing import Union, Dict, Tuple, Optional, Any, List
from .state import VisualizationState
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
import copy 
import time


def _yield_notebook_events(seconds: float = 0.0) -> None:
    """Yield control to the notebook event loop from sync code.

    This keeps widget-based backends (e.g. pyvista+trame) responsive inside
    plain ``for`` loops without forcing users to write async notebook cells.
    """
    delay = max(float(seconds), 0.0)

    try:
        from IPython import get_ipython  # noqa: PLC0415

        ip = get_ipython()
        in_notebook = ip is not None and "IPKernelApp" in getattr(ip, "config", {})
    except Exception:
        in_notebook = False

    if not in_notebook:
        if delay > 0.0:
            time.sleep(delay)
        return

    try:
        import asyncio  # noqa: PLC0415

        loop = asyncio.get_event_loop()
        if loop.is_running():
            try:
                import nest_asyncio  # noqa: PLC0415

                nest_asyncio.apply(loop)
                loop.run_until_complete(asyncio.sleep(delay))
            except Exception:
                # Last resort: pause briefly if we cannot re-enter the loop.
                if delay > 0.0:
                    time.sleep(delay)
        else:
            loop.run_until_complete(asyncio.sleep(delay))
    except Exception:
        if delay > 0.0:
            time.sleep(delay)
import asyncio
@dataclass
class PlotState:
    fig: Any  # backend-specific figure handle (matplotlib Figure for default backend)
    axes: Dict[str, Any]  # backend-specific per-panel handles
    domain: DomainDescription
    options: PlottingOptions
    quantities: Union[torch.Tensor, Dict[str, torch.Tensor]]
    particleState: Any
    mosaic: str
    sharex: bool
    sharey: bool
    figTitle: Optional[str]

    plotStates: Dict[str, VisualizationState]

    backend: str = "matplotlib"
    backendOptions: Optional[dict] = None
    _backend_instance: Any = field(default=None, repr=False, compare=False)
    _update_counter: int = field(default=0, repr=False, compare=False)
    
    def updateTitle(self, newTitle: str):
        self.figTitle = newTitle
        if self._backend_instance is not None:
            self._backend_instance.update_figure_title(newTitle)
            return

        # Fallback for legacy PlotState objects created without a backend.
        if hasattr(self.fig, "suptitle"):
            self.fig.suptitle(newTitle, fontsize=16)

    def updateDomain(self, newDomain: DomainDescription):
        self.domain = newDomain
        if self._backend_instance is not None:
            self._backend_instance.update_domain(newDomain)
            return

        # # Fallback for legacy PlotState objects created without a backend.
        # for key, panel_state in self.plotStates.items():
        #     panel_state.domain = newDomain

    def show(self) -> None:
        """Display or re-display the figure.

        For the matplotlib backend this is equivalent to calling
        ``fig.tight_layout()`` and is typically a no-op after the initial
        ``visualize()`` call.

        For the pyvista *static* backend, this re-captures an offscreen
        screenshot and displays it inline in the current Jupyter cell —
        useful for showing the updated scene after
        ``updateQuantities(...)``.

        For pyvista *trame* and the pop-out window modes, the scene updates
        are reflected live and this call is also a no-op.
        """
        if self._backend_instance is not None:
            self._backend_instance.show()

    def export(self, filepath: str, **kwargs) -> None:
        """Export the current visualization to *filepath*.

        The file format is inferred from the extension (e.g. ``".png"``,
        ``".pdf"``, ``".svg"``).

        Parameters
        ----------
        filepath:
            Full path including extension.
        dpi:
            Dots per inch (supported by the matplotlib and vispy backends).
        **kwargs:
            Additional keyword arguments forwarded verbatim to the active
            backend.  Common options:

            * ``transparent`` / ``transparent_background`` — transparent
              background (matplotlib / pyvista-raster).
            * ``bbox_inches`` — e.g. ``"tight"`` (matplotlib only).

        Raises
        ------
        NotImplementedError
            If the active backend does not support file export.
        RuntimeError
            If no backend instance is available.
        """
        if self._backend_instance is None:
            raise RuntimeError("No backend instance available.")
        self._backend_instance.export(filepath, **kwargs)

    def updateQuantities(self, newQuantities: Union[torch.Tensor, Dict[str, torch.Tensor]], key: Optional[str] = None, newParticleState: Optional[Any] = None, newDomain: Optional[DomainDescription] = None, newOptions: Optional[Dict[str, Any]] = None, redraw: bool = True, redrawEvery: int = 1, yieldNotebookEvents: bool = False, yieldSeconds: float = 0.02, **kwargs):
        if newDomain is not None:
            self.domain = newDomain
        if newParticleState is not None or newDomain is not None:
            self.particleState = copy.deepcopy(newParticleState) if newParticleState is not None else self.particleState
            
            positions = self.particleState.positions
            minD = self.domain.min.detach()
            maxD = self.domain.max.detach()
            periodicity = self.domain.periodic

            pos = [(torch.remainder(positions[:, i] - minD[i], maxD[i] - minD[i]) + minD[i]) if periodicity[i] else positions[:,i] for i in range(self.domain.dim)]
            self.particleState.positions = torch.stack(pos, dim = -1)
            # modPos = torch.stack(pos, dim = -1).detach().cpu().numpy()

        if key is not None and isinstance(newQuantities, dict):
            self.quantities[key] = newQuantities[key]
            self.updatePlot(key, newOptions= newOptions[key] if newOptions is not None and key in newOptions else None, **kwargs)
        elif key is not None and isinstance(newQuantities, torch.Tensor):
            self.quantities[key] = newQuantities
            self.updatePlot(key, newOptions= newOptions[key] if newOptions is not None and key in newOptions else None, **kwargs)
        elif key is None and isinstance(self.quantities, dict) and isinstance(newQuantities, dict):
            for k in newQuantities:
                self.quantities[k] = newQuantities[k]
                self.updatePlot(k, newOptions= newOptions[k] if newOptions is not None and k in newOptions else None, **kwargs)
        else:
            if len(self.plotStates) == 1:
                self.quantities = newQuantities
                self.updatePlot(list(self.plotStates.keys())[0], newOptions= newOptions[list(self.plotStates.keys())[0]] if newOptions is not None and list(self.plotStates.keys())[0] in newOptions else None, **kwargs)

        # Flush the display after all panels are updated.
        # For pyvista-static this captures a new screenshot; for trame it
        # calls render(); for matplotlib it calls tight_layout() (no-op).
        if self._backend_instance is not None and redraw:
            n = max(int(redrawEvery), 1)
            self._update_counter += 1
            if self._update_counter % n == 0:
                self._backend_instance.show()

        if yieldNotebookEvents or self.backend == 'pyvista':
            _yield_notebook_events(yieldSeconds)
        # if self.backend == 'matplotlib':
        #     self.fig.canvas.draw()
        #     self.fig.canvas.flush_events()


    def updatePlot(self, key: Union[str, List[str]], newOptions: Optional[Dict[str, Any]] = None, **kwargs):
        if isinstance(key, list):
            for k in key:
                self.updatePlot(k, newOptions, **kwargs)
            return

        panel_state = self.plotStates[key]
        particleState = self.particleState
        quantity = self.quantities[key] if isinstance(self.quantities, dict) else self.quantities

        if self._backend_instance is not None:
            updatedState = self._backend_instance.update_panel(
                key,
                panel_state,
                particleState=particleState,
                domain=self.domain,
                quantity=quantity,
                options=panel_state.options,
                **newOptions if newOptions is not None else {},
                **kwargs
            )
        else:
            # Fallback: direct matplotlib updatePlot (keeps backward-compat if
            # PlotState was constructed without a backend instance).
            updatedState = updatePlot(
                panel_state,
                particles=particleState,
                domain=self.domain,
                quantity=quantity,
                options=panel_state.options,
                **newOptions if newOptions is not None else {},
                **kwargs
            )
        self.plotStates[key] = updatedState


def visualize(
    particleState: Any,
    domain: DomainDescription,
    quantities: Union[torch.Tensor, Dict[str, torch.Tensor]],
    plotOptions: Union[PlottingOptions, Dict[str, PlottingOptions]],
    mosaic: str = 'A',
    figsize: Tuple[float, float] = (7, 6),
    sharex: bool = True,
    sharey: bool = True,
    figTitle: Optional[str] = None,
    backend: str = "matplotlib",
    backendOptions: Optional[dict] = None,
) -> 'PlotState':
    """Create a multi-panel SPH visualization.

    Args:
        backend: One of ``"matplotlib"`` (default), ``"pyvista"``, or
            ``"vispy"``.  Pass the :class:`~warpSPHPlotting.Backend` enum or a plain
            string — both work.
        backendOptions: Optional dict of keyword arguments forwarded verbatim to
            the chosen backend's ``create_figure`` call.
    """
    from .backends.factory import get_backend
    be = get_backend(backend, backendOptions)

    fig = be.create_figure(
        mosaic=mosaic,
        figsize=figsize,
        sharex=sharex,
        sharey=sharey,
        figTitle=figTitle,
        backendOptions=backendOptions,
    )
    axis = be.get_axes()

    visParticleState = copy.deepcopy(particleState)
    
    positions = particleState.positions
    minD = domain.min.detach()
    maxD = domain.max.detach()
    periodicity = domain.periodic

    pos = [(torch.remainder(positions[:, i] - minD[i], maxD[i] - minD[i]) + minD[i]) if periodicity[i] else positions[:,i] for i in range(domain.dim)]
    visParticleState.positions = torch.stack(pos, dim = -1)

    plotStates = {}
    for key in axis:
        panel_state = be.render_panel(
            key,
            particleState=visParticleState,
            domain=domain,
            quantity=quantities[key] if isinstance(quantities, dict) else quantities,
            options=plotOptions[key] if isinstance(plotOptions, dict) else plotOptions,
        )
        plotStates[key] = panel_state

    if backend == 'matplotlib':
        fig.tight_layout()
    be.show()

    return PlotState(
        fig=fig,
        axes=axis,
        domain=domain,
        options=plotOptions,
        quantities=quantities,
        mosaic=mosaic,
        sharex=sharex,
        sharey=sharey,
        figTitle=figTitle,
        plotStates=plotStates,
        particleState=visParticleState,
        backend=str(backend),
        backendOptions=backendOptions,
        _backend_instance=be,
    )

