from matplotlib.collections import PathCollection
from matplotlib.colorbar import Colorbar
from matplotlib.streamplot import StreamplotSet
from sphWarpCore import DomainDescription, ParticleState, volumeToSupport, warpOperation
from sphWarpCore import OperationProperties, WarpOperation, OperationDirection, KernelFunctions, SupportScheme, ParticleType
from warpPlot.update import updatePlot
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
        verbosePrint(verbose, "Plotting Fluid Particles...")
        fluidSc = scatterVisualize(fig, axis, fluidParticles, domain_, options, variant=options.fluidVisualization)
        verbosePrint(verbose, "Plotting Boundary Particles...")
        boundarySc = scatterVisualize(fig, axis, boundaryParticles, domain_, options, variant=options.boundaryVisualization)
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


from typing import Union, Dict, Tuple, Optional, Any
from .state import VisualizationState
from sphWarpCore.radiusSearch import DomainDescription
import matplotlib.pyplot as plt
from dataclasses import dataclass

from typing import Union, Dict, Tuple, Optional, Any, List
from warpPlot.state import VisualizationState
from sphWarpCore.radiusSearch import DomainDescription
import matplotlib.pyplot as plt
from dataclasses import dataclass

@dataclass
class PlotState:
    fig: plt.Figure
    axes: Dict[str, plt.Axes]
    domain: DomainDescription
    options: PlottingOptions
    quantities: Union[torch.Tensor, Dict[str, torch.Tensor]]
    particleState: Any
    mosaic : str
    sharex: bool
    sharey: bool
    figTitle: Optional[str]
    
    plotStates: Dict[str, VisualizationState]
    
    def updateTitle(self, newTitle: str):
        self.fig.suptitle(newTitle, fontsize=16)
        
    def updateQuantities(self, newQuantities: Union[torch.Tensor, Dict[str, torch.Tensor]], key: Optional[str] = None, newParticleState: Optional[Any] = None, newDomain: Optional[DomainDescription] = None, newOptions: Optional[Dict[str, Any]] = None, **kwargs):
        if newParticleState is not None:
            self.particleState = newParticleState
        if newDomain is not None:
            self.domain = newDomain
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
        
            
    def updatePlot(self, key: Union[str, List[str]], newOptions: Optional[Dict[str, Any]] = None, **kwargs):
        if isinstance(key, list):
            for k in key:
                self.updatePlot(k, newOptions, **kwargs)
            return

        plotState = self.plotStates[key]
        particleState = self.particleState
        domain = self.domain
        # options = plotState.options if newOptions is None else PlottingOptions(**{**plotState.options.__dict__, **newOptions})
        quantity = self.quantities[key] if isinstance(self.quantities, dict) else self.quantities

        updatedState = updatePlot(
            plotState,
            particles = particleState,
            domain = domain,
            quantity = quantity,
            options = plotState.options,
            **newOptions if newOptions is not None else {},
            **kwargs
        )
        self.plotStates[key] = updatedState


def visualize(
    particleState: Any,
    domain: DomainDescription,
    quantities: Union[torch.Tensor, Dict[str, torch.Tensor]],
    plotOptions: Union[torch.Tensor, Dict[str, torch.Tensor]],
    mosaic: str = 'A',
    figsize: Tuple[float, float] = (7,6),
    sharex: bool = True,
    sharey: bool = True,
    figTitle: Optional[str] = None,
):
    fig, axis = plt.subplot_mosaic(mosaic, figsize=figsize, sharex=sharex, sharey=sharey)
    plotStates = {}
    for key in axis:
        plotState = visualizeParticlesNew(
            fig, axis[key],
            particleState = particleState,
            domain = domain,
            quantity = quantities[key] if isinstance(quantities, dict) else quantities,
            options = plotOptions[key] if isinstance(plotOptions, dict) else plotOptions,
            plotTitle = f"Visualization of {key}",
        )
        plotStates[key] = plotState
        if figTitle is not None:
            fig.suptitle(figTitle, fontsize=16)
    fig.tight_layout()

    return PlotState(
        fig = fig,
        axes = axis,
        domain = domain,
        options = plotOptions,
        quantities = quantities,
        mosaic = mosaic,
        sharex = sharex,
        sharey = sharey,
        figTitle = figTitle,
        plotStates = plotStates,
        particleState=particleState,
    )

