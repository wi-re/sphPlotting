from matplotlib.collections import PathCollection
from matplotlib.colorbar import Colorbar
from matplotlib.streamplot import StreamplotSet
from sphWarpCore import DomainDescription, ParticleState, volumeToSupport, warpOperation
from sphWarpCore import OperationProperties, WarpOperation, OperationDirection, KernelFunctions, SupportScheme, ParticleType
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

def visualizeParticlesNew(
    fig, axis,
    particleState,
    domain: DomainDescription,
    quantity: torch.Tensor,
    options: PlottingOptions = PlottingOptions(),
    verbose = False,
):
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
        verbosePrint(verbose, "Applying plotting operation...")
        verbosePrint(verbose, "Quantity shape before plotting operation: ", rotatedState.quantities.shape, 'min: ', torch.min(rotatedState.quantities).item(), 'max: ', torch.max(rotatedState.quantities).item())
        rotatedState.quantities = warpOperation(
            queryParticles = rotatedState,
            queryValues = rotatedState.quantities,
            operationProperties = options.plottingOperation,
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
            includeBoundary = options.boundaryVisualization != VisualizeOptions.Hide
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
                includeBoundary = options.boundaryVisualization != VisualizeOptions.Hide
            )
            verbosePrint(verbose, 'Mapped streamline quantity to grid. Shape: ', streamLineGridQuantity.shape)
            verbosePrint(verbose, "Plotting streamlines...")

            streamLines = streamLinePlot(fig, axis, streamLineGridState, streamLineGridQuantity, streamLinenxs, streamLineGridExtent, options)
        else:
            streamLines = None


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


