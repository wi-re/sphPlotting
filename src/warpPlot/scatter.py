from sphWarpCore import DomainDescription, ParticleState, volumeToSupport, warpOperation
from sphWarpCore import OperationProperties, WarpOperation, OperationDirection, KernelFunctions, SupportScheme, ParticleType
import torch
from .math import getBounds

from .state import PlottingParticleState
from .options import PlottingOptions
from .enumTypes import VisualizeOptions
from typing import Optional, Tuple
import matplotlib.colors as colors
from matplotlib.collections import PathCollection
from matplotlib.colorbar import Colorbar

def scatterVisualize(
    fig, axis,
    particleState: PlottingParticleState,
    domain: DomainDescription,
    options: PlottingOptions,
    variant: VisualizeOptions = VisualizeOptions.Visualize
):
    
    if variant == VisualizeOptions.Hide:
        return None, None, None
    
    positions = particleState.positions
    minD = domain.min.cpu().detach()
    maxD = domain.max.cpu().detach()
    periodicity = domain.periodic

    pos = [(torch.remainder(positions[:, i] - minD[i], maxD[i] - minD[i]) + minD[i]) if periodicity[i] else positions[:,i] for i in range(domain.dim)]
    modPos = torch.stack(pos, dim = -1).detach().cpu().numpy()

    if variant == VisualizeOptions.Passive:
        sc = axis.scatter(modPos[:,0], modPos[:,1], s = options.markerSize, color = 'gray', alpha = 0.5, marker = 'x')
        return sc, None, None
    
    quantity = particleState.quantities

    qs, norm = getBounds(quantity, options)

    sc = axis.scatter(modPos[:,0], modPos[:,1], s = options.markerSize, c = qs, cmap = options.colorMap.value + ('_r' if options.flipColorMap else ''), norm = norm)
    cb = None

    if options.showColorBar:
        cb = fig.colorbar(sc, ax=axis)
    
    return sc, cb, norm

    


def updateScatterVisualize(
    priorResult: Optional[Tuple[Optional[PathCollection], Optional[Colorbar], Optional[colors.Normalize]]],
    fig, axis,
    particleState: PlottingParticleState,
    domain: DomainDescription,
    options: PlottingOptions,
    variant: VisualizeOptions = VisualizeOptions.Visualize        
):
    
    if variant == VisualizeOptions.Hide:
        return None, None, None
    
    positions = particleState.positions
    minD = domain.min.cpu().detach()
    maxD = domain.max.cpu().detach()
    periodicity = domain.periodic

    pos = [(torch.remainder(positions[:, i] - minD[i], maxD[i] - minD[i]) + minD[i]) if periodicity[i] else positions[:,i] for i in range(domain.dim)]
    modPos = torch.stack(pos, dim = -1).detach().cpu().numpy()

    sc, cb, norm = priorResult if priorResult is not None else (None, None, None)

    if variant == VisualizeOptions.Passive:
        sc.set_offsets(modPos)
        return
    
    quantity = particleState.quantities

    qs, norm = getBounds(quantity, options)

    sc.set_offsets(modPos)
    sc.set_array(qs)
    sc.set_norm(norm)
    sc.set_cmap(options.colorMap.value + ('_r' if options.flipColorMap else ''))
    return
