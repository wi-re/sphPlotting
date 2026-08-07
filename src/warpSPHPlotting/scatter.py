from warpSPHCore import *
import torch
from .math import getBounds

from .state import PlottingParticleState
from .options import PlottingOptions
from .enumTypes import VisualizeOptions
from typing import Optional, Tuple
import matplotlib.colors as colors
from matplotlib.collections import PathCollection
from matplotlib.colorbar import Colorbar
import numpy as np

def scatterVisualize(
    fig, axis,
    particleState: PlottingParticleState,
    domain: DomainDescription,
    options: PlottingOptions,
    variant: VisualizeOptions = VisualizeOptions.Visualize,
    precomputedQs: Optional[np.ndarray] = None,
    precomputedNorm: Optional[colors.Normalize] = None,
    attachColorBar: Optional[bool] = None,
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

    if precomputedQs is None or precomputedNorm is None:
        qs, norm = getBounds(quantity, options)
    else:
        qs = precomputedQs
        norm = precomputedNorm

    sc = axis.scatter(modPos[:,0], modPos[:,1], s = options.markerSize, c = qs, cmap = options.colorMap.value + ('_r' if options.flipColorMap else ''), norm = norm)
    cb = None

    shouldAttachColorBar = options.showColorBar if attachColorBar is None else (options.showColorBar and attachColorBar)
    if shouldAttachColorBar:
        cb = fig.colorbar(sc, ax=axis)
    
    return sc, cb, norm

    


def updateScatterVisualize(
    priorResult: Optional[Tuple[Optional[PathCollection], Optional[Colorbar], Optional[colors.Normalize]]],
    fig, axis,
    particleState: PlottingParticleState,
    domain: DomainDescription,
    options: PlottingOptions,
    variant: VisualizeOptions = VisualizeOptions.Visualize,
    precomputedQs: Optional[np.ndarray] = None,
    precomputedNorm: Optional[colors.Normalize] = None,
    attachColorBar: Optional[bool] = None,
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
    shouldAttachColorBar = options.showColorBar if attachColorBar is None else (options.showColorBar and attachColorBar)

    if sc is None:
        return scatterVisualize(
            fig,
            axis,
            particleState,
            domain,
            options,
            variant=variant,
            precomputedQs=precomputedQs,
            precomputedNorm=precomputedNorm,
            attachColorBar=attachColorBar,
        )

    if variant == VisualizeOptions.Passive:
        if cb is not None:
            cb.remove()
            cb = None
        sc.set_offsets(modPos)
        return sc, None, None
    
    quantity = particleState.quantities

    if precomputedQs is None or precomputedNorm is None:
        qs, norm = getBounds(quantity, options)
    else:
        qs = precomputedQs
        norm = precomputedNorm

    sc.set_offsets(modPos)
    sc.set_array(qs)
    sc.set_norm(norm)
    sc.set_cmap(options.colorMap.value + ('_r' if options.flipColorMap else ''))

    if not shouldAttachColorBar and cb is not None:
        cb.remove()
        cb = None
    elif shouldAttachColorBar and cb is None:
        cb = fig.colorbar(sc, ax=axis)

    if cb is not None:
        cb.update_normal(sc)

    return sc, cb, norm
