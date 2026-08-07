

from matplotlib import patches
from warpSPHCore import DomainDescription
from .options import PlottingOptions
from .math import buildRotationMatrix
from .scatter_util import computeMarkerSize
import torch

def processDomain(
        fig, axis,
        domain: DomainDescription,
        options: PlottingOptions,
        dx: float
):
    domain_ = DomainDescription(
        min = domain.min.detach(),
        max = domain.max.detach(),
        periodic = domain.periodic.detach(),
        dim = domain.dim
    )
    if hasattr(domain, 'angles'):
        rotMat = buildRotationMatrix(torch.tensor(domain.angles, dtype = domain.min.dtype, device = domain.min.device), domain.dim, device=domain.min.device, dtype=domain.min.dtype)
        invRotMat = rotMat.inverse()
    else:
        rotMat = None
        invRotMat = None

    # Set up the axis
    eps = (domain_.max.cpu() - domain_.min.cpu()) * options.domainEpsilon
    axis.set_xlim(domain_.min.cpu()[0] - eps[0], domain_.max.cpu()[0] + eps[0])
    axis.set_ylim(domain_.min.cpu()[1] - eps[1], domain_.max.cpu()[1] + eps[1])
    if options.plotDomain:
        square = patches.Rectangle((domain_.min.cpu()[0], domain_.min.cpu()[1]), domain_.max.cpu()[0] - domain_.min.cpu()[0], domain_.max.cpu()[1] - domain_.min.cpu()[1],    linewidth=1, edgecolor='b', facecolor='none',ls='--')
        axis.add_patch(square)
    axis.set_aspect('equal')

    if options.markerSize is None:
        options.markerSize = computeMarkerSize(axis, (domain_.min.cpu()[0] - eps[0], domain_.max.cpu()[0] + eps[0]), (domain_.min.cpu()[1] - eps[1], domain_.max.cpu()[1] + eps[1]), dx)

    return domain_, rotMat, invRotMat