from matplotlib.collections import PathCollection
from matplotlib.colorbar import Colorbar
from matplotlib.streamplot import StreamplotSet
from sphWarpCore import DomainDescription, ParticleState, volumeToSupport, warpOperation
from sphWarpCore import OperationProperties, WarpOperation, OperationDirection, KernelFunctions, SupportScheme, ParticleType
from .domain import processDomain
from .grid import generateGrid, mapToGrid, updateGridVisualize
from .scatter import updateScatterVisualize
from .streamLines import updateStreamLinePlot
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
    
def updatePlot(plotState, 
        particles: ParticleState,
        quantity: Optional[torch.Tensor] = None, 
        **kwargs):
    options = plotState.options
    for arg in kwargs:
        if hasattr(options, arg):
            setattr(options, arg, kwargs[arg])

    fig, axis = plotState.fig, plotState.axis
    domain = plotState.domain
    domain_, rotMat, invRotMat = processDomain(fig, axis, domain, options, (particles.masses / particles.densities).mean().item())  

    assembledQuantity = plotState.assembledQuantity if quantity is None else assembleQuantity(particles, quantity)
    particlePlotState = PlottingParticleState(
        positions = particles.positions,
        supports = particles.supports,
        masses = particles.masses,
        densities = particles.densities,
        kinds = particles.kinds,
        quantities = assembledQuantity
    )
    # We are rotating the state back to the original orientation for filtering thus we are using the inverse rotation matrix
    rotatedState = rotateState(particlePlotState, invRotMat)
    rotatedState.quantities = assembledQuantity if quantity is None else rotatedState.quantities
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
            rotatedState.quantities = warpOperation(
                queryParticles = rotatedState,
                queryValues = rotatedState.quantities,
                operationProperties = op,
                adjacency = None, # We can consider adding adjacency-based operations in the future
                domain = domain_
            )
            
    if options.gridVisualization is not None and options.gridVisualization.streamLines and options.gridVisualization.streamLineOperationLocation == StreamLineLocation.BeforeMapping:
        streamLineQuantity = rotatedState.quantities.clone()

    # Apply mapping to the quantities if specified in the options
    if options.mapping != Mapping.none:
        rotatedState.quantities = mapQuantity(rotatedState.quantities, options.mapping)
        
    if options.gridVisualization is not None and options.gridVisualization.streamLines and options.gridVisualization.streamLineOperationLocation == StreamLineLocation.AfterMapping:
        streamLineQuantity = rotatedState.quantities.clone()

    # Checking quantity shape
    if rotatedState.quantities.shape[0] != rotatedState.positions.shape[0]:
        raise ValueError(f"Quantity length {rotatedState.quantities.shape[0]} does not match number of particles {rotatedState.positions.shape[0]} after applying plotting operation.")
    if rotatedState.quantities.ndim == 2:
        raise ValueError(f"Quantity has more than 1 component per particle (shape: {rotatedState.quantities.shape}). Please specify how to map the quantity to a scalar value for visualization (e.g., using options.mapping) or ensure the plotting operation returns a single scalar value per particle.")

    fluidParticles = filterState(rotatedState, rotatedState.quantities, kind=ParticleType.Fluid) 
    boundaryParticles = filterState(rotatedState, rotatedState.quantities, kind=ParticleType.Boundary)
    ghostParticles = filterState(rotatedState, rotatedState.quantities, kind=ParticleType.Ghost)

    if options.gridVisualization is None:
        updateScatterVisualize(plotState.fluidScatterResult, fig, axis, fluidParticles, domain_, options, variant=options.fluidVisualization)
        updateScatterVisualize(plotState.boundaryScatterResult, fig, axis, boundaryParticles, domain_, options, variant=options.boundaryVisualization)
        updateScatterVisualize(plotState.ghostScatterResult, fig, axis, ghostParticles, domain_, options, variant=VisualizeOptions.Hide)
    else:
        gridState, gridQuantity, nxs, gridExtent = mapToGrid(
            particleState = rotatedState,
            quantity = rotatedState.quantities,
            domain = domain_,
            nx = options.gridVisualization.resolution,
            targetNeighbors = 50,
            kernel = options.plottingKernel,
            alignment = 'center',
            includeFluid = options.fluidVisualization != VisualizeOptions.Hide,
            includeBoundary = options.boundaryVisualization != VisualizeOptions.Hide
        )

        updateGridVisualize(plotState.gridResult, fig, axis, gridState, gridQuantity, nxs, gridExtent, options)
        
        if options.gridVisualization.streamLines:
            if options.gridVisualization.streamLineOperation is not None:
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

            streamLineGridState, streamLineGridQuantity, streamLinenxs, streamLineGridExtent = mapToGrid(
                particleState = rotatedState,
                quantity = streamLineQuantity,
                domain = domain_,
                nx = options.gridVisualization.resolution,
                targetNeighbors = 50,
                kernel = options.plottingKernel,
                alignment = 'center',
                includeFluid = options.fluidVisualization != VisualizeOptions.Hide,
                includeBoundary = options.boundaryVisualization != VisualizeOptions.Hide
            )

            streamLines = updateStreamLinePlot(plotState.streamLines, fig, axis, streamLineGridState, streamLineGridQuantity, streamLinenxs, streamLineGridExtent, options)
            plotState.streamLines = streamLines
        else:
            streamLines = None

    plotState.fluidParticles = fluidParticles
    plotState.boundaryParticles = boundaryParticles
    plotState.ghostParticles = ghostParticles
    plotState.assembledQuantity = assembledQuantity

    if options.plotTitle is not None:
        axis.set_title(options.plotTitle)
        
    return plotState

