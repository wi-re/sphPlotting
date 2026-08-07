from warpSPHCore import *
import torch
from .math import getBounds

def generateGrid(
    domain: DomainDescription,
    resolution: int,
    device: torch.device = 'cpu',
    dtype: torch.dtype = torch.float32,
    alignment: str = 'center',
    shortEdge: bool = False
):
    # Generate a grid of points within the specified domain and resolution
    # Alignment can be 'center', 'edge', 'left', or 'right'
    
    ns = [resolution] * domain.dim
    if shortEdge:
        edgeLengths = domain.max - domain.min
        minEdge = torch.min(edgeLengths)
        ns = [int((edgeLengths[i] / minEdge) * resolution) for i in range(domain.dim)]
    else:
        edgeLengths = domain.max - domain.min
        maxEdge = torch.max(edgeLengths)
        ns = [int((edgeLengths[i] / maxEdge) * resolution) for i in range(domain.dim)]

    # print(ns)

    dx = (domain.max - domain.min) / torch.tensor(ns, device=device, dtype=dtype)
    if alignment == 'center':
        start = domain.min + dx / 2
        end = domain.max - dx / 2
    elif alignment == 'edge':
        start = domain.min
        end = domain.max
    elif alignment == 'left':
        start = domain.min
        end = domain.max - dx
    elif alignment == 'right':
        start = domain.min + dx
        end = domain.max
    else:
        raise ValueError(f"Unsupported alignment: {alignment}")
    
    grids = [torch.linspace(start[i], end[i], ns[i], device=device, dtype=dtype) for i in range(domain.dim)]
    mesh = torch.meshgrid(*grids, indexing='ij')
    gridPoints = torch.stack(mesh, dim=-1).reshape(-1, domain.dim)

    gridArea = torch.prod(dx)

    gridExtent = {
        'min': domain.min,
        'max': domain.max,
    }

    return gridPoints, dx, ns, gridArea, gridExtent

def mapToGrid(
    particleState: ParticleState,
    quantity: torch.Tensor,
    domain: DomainDescription,
    nx: int,
    targetNeighbors: int,
    kernel: KernelFunctions,
    alignment: str = 'center',
    includeFluid: bool = True,
    includeBoundary: bool = False,
    gridMode: SupportScheme = SupportScheme.Scatter
):
    # Map particle quantities to a grid using SPH-like interpolation
    device = particleState.positions.device
    dtype = particleState.positions.dtype

    grid, dx, nxs, gridArea, gridExtent = generateGrid(domain, nx, device=device, dtype=dtype, alignment=alignment)

    h = volumeToSupport(gridArea, targetNeighbors, particleState.positions.shape[1])

    gridState = ParticleState(
        positions = grid,
        supports = torch.full((grid.shape[0],), h, device=device, dtype=dtype),
        masses = torch.full((grid.shape[0],), gridArea, device=device, dtype=dtype), 
        densities = torch.ones((grid.shape[0],), device=device, dtype=dtype),
        kinds = torch.full((grid.shape[0],), ParticleType.Fluid.value, device=device, dtype=torch.int32)
    )
    gridQuantity = torch.zeros((grid.shape[0],) + quantity.shape[1:], device=device, dtype=dtype)

    if gridMode is None:
        if grid.shape[0] < particleState.positions.shape[0]:
            gridMode = SupportScheme.Gather
        else:
            gridMode = SupportScheme.Scatter

    if includeFluid:
        gridQuantity += warpOperation(
            queryParticles = gridState, referenceParticles = particleState,
            queryValues = gridQuantity, referenceValues = quantity,
            operationProperties = OperationProperties(
                kernel = kernel,
                operation = WarpOperation.Interpolate,
                operationMode = OperationDirection.FluidToFluid,
                supportMode = gridMode
            ),
            domain = domain,
        )
    if includeBoundary:
        gridQuantity += warpOperation(
            queryParticles = gridState, referenceParticles = particleState,
            queryValues = gridQuantity, referenceValues = quantity,
            operationProperties = OperationProperties(
                kernel = kernel,
                operation = WarpOperation.Interpolate,
                operationMode = OperationDirection.BoundaryToFluid,
                supportMode = gridMode
            ),
            domain = domain,
        )
    return gridState, gridQuantity, nxs, gridExtent


def gridVisualize(fig, axis, gridState, gridQuantity, gridResolution, gridExtent, options):
    
    quantity = gridQuantity
    qs, norm = getBounds(quantity, options)

    # print("Grid Quantity: ", quantity)
    # print("qs: ", qs)
    # print()

    # sc = axis.imshow(
    #     qs.reshape(gridResolution[0], gridResolution[1]).T,
    #     extent=(gridExtent['min'][0].cpu().item(), gridExtent['max'][0].cpu().item(), gridExtent['min'][1].cpu().item(), gridExtent['max'][1].cpu().item()),
    #     origin='lower', 
    #     cmap = options.colorMap.value + ('_r' if options.flipColorMap else ''),
    #     norm = norm,
    # )

    sc = axis.pcolormesh(
        gridState.positions[:,0].reshape(gridResolution[0], gridResolution[1]).cpu().detach(),
        gridState.positions[:,1].reshape(gridResolution[0], gridResolution[1]).cpu().detach(),
        qs.reshape(gridResolution[0], gridResolution[1]),
        shading='auto',
        cmap = options.colorMap.value + ('_r' if options.flipColorMap else ''),
        norm = norm,
    )
    if options.showColorBar:
        cb = fig.colorbar(sc, ax=axis)
    else:
        cb = None
    return sc, cb, norm

def updateGridVisualize(priorResult, fig, axis, gridState, gridQuantity, gridResolution, gridExtent, options):
    
    sc, cb, norm = priorResult if priorResult is not None else (None, None, None)

    quantity = gridQuantity
    qs, norm = getBounds(quantity, options)

    sc.set_array(qs)
    sc.set_cmap(options.colorMap.value + ('_r' if options.flipColorMap else ''))
    sc.set_norm(norm)
    return

    