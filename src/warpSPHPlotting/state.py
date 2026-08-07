
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Union
import matplotlib.pyplot as plt
import torch
from warpSPHCore import *
from .options import PlottingOptions
from matplotlib.collections import PathCollection
from matplotlib.colorbar import Colorbar
import matplotlib.colors as colors
from matplotlib.streamplot import StreamplotSet

@dataclass
class PlottingParticleState:
    positions: torch.Tensor
    supports: torch.Tensor
    masses: torch.Tensor
    densities: torch.Tensor
    kinds: torch.Tensor
    quantities: torch.Tensor


def rotateState(
        state: PlottingParticleState,
        rotMat: Optional[torch.Tensor]
) -> PlottingParticleState:
    if rotMat is None:
        return state
    else:
        rotatedPositions = torch.einsum('ij, ni->nj', rotMat, state.positions)
        rotatedQuantities = None
        if state.quantities.ndim == 2 and state.quantities.shape[1] == rotMat.shape[0]:
            rotatedQuantities = torch.einsum('ij, ni->nj', rotMat, state.quantities)
        else:            
            rotatedQuantities = state.quantities
        return PlottingParticleState(
            positions = rotatedPositions,
            supports = state.supports,
            masses = state.masses,
            densities = state.densities,
            kinds = state.kinds,
            quantities = rotatedQuantities
        )
    
def filterState(
    particleState: ParticleState,
    quantity: Union[torch.Tensor, Tuple[torch.Tensor, ...]],

    kind: ParticleType = ParticleType.Fluid,
    batch = None,
    rotMat: Optional[torch.Tensor] = None
):
    mask = particleState.kinds == kind.value
    if batch is not None:
        mask = torch.logical_and(mask, particleState.batches == batch)
    state = PlottingParticleState(
        positions = particleState.positions[mask],
        supports = particleState.supports[mask],
        masses = particleState.masses[mask],
        densities = particleState.densities[mask],
        kinds = particleState.kinds[mask],
        quantities=quantity[mask] if isinstance(quantity, torch.Tensor) else quantity[kind.value]
    )
    if rotMat is not None and torch.sum(mask) > 0:
        state = rotateState(state, rotMat)
    return state


def assembleQuantity(
    particleState: ParticleState,
    quantity: Union[torch.Tensor, Tuple[torch.Tensor, ...]] 
):
    if isinstance(quantity, torch.Tensor):
        return quantity
    else:
        firstNonNull = next(q for q in quantity if q is not None)
        assembled = torch.zeros((particleState.positions.shape[0],) + firstNonNull.shape[1:], device=firstNonNull.device, dtype=firstNonNull.dtype)
        for kind in ParticleType:
            mask = particleState.kinds == kind.value
            if quantity[kind.value] is not None:
                assembled[mask] = quantity[kind.value][mask]
        return assembled


@dataclass 
class VisualizationState:
    fig: plt.Figure
    axis: plt.Axes
    domain: DomainDescription
    options: PlottingOptions

    assembledQuantity: torch.Tensor

    fluidParticles: Optional[PlottingParticleState] = None
    boundaryParticles: Optional[PlottingParticleState] = None
    ghostParticles: Optional[PlottingParticleState] = None

    fluidScatterResult: Optional[Tuple[Optional[PathCollection], Optional[Colorbar], Optional[colors.Normalize]]] = None
    boundaryScatterResult: Optional[Tuple[Optional[PathCollection], Optional[Colorbar], Optional[colors.Normalize]]] = None
    ghostScatterResult: Optional[Tuple[Optional[PathCollection], Optional[Colorbar], Optional[colors.Normalize]]] = None
    gridResult: Optional[Tuple[Optional[PathCollection], Optional[Colorbar], Optional[colors.Normalize]]] = None

    streamLines: Optional[StreamplotSet] = None
