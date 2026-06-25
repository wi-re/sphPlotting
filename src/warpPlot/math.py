from typing import List
import matplotlib.colors as colors
from matplotlib.collections import PathCollection
from matplotlib.colorbar import Colorbar
import torch
import numpy as np
from .options import PlottingOptions
from .enumTypes import Mapping, PlotScaling

def buildRotationMatrix(angles : List[float], dim: int, device: torch.device = None, dtype: torch.dtype = None):
    if dim == 1:
        return torch.tensor([[1.0]], device=device, dtype=dtype)
    elif dim == 2:
        return torch.tensor([[torch.cos(angles), -torch.sin(angles)],
                             [torch.sin(angles), torch.cos(angles)]], device=device, dtype=dtype)
    elif dim == 3:
        angle_phi = angles[0]
        angle_theta = angles[1]
        return torch.tensor([
            [torch.cos(angle_phi) * torch.sin(angle_theta), -torch.sin(angle_phi), torch.cos(angle_phi) * torch.cos(angle_theta)],
            [torch.sin(angle_phi) * torch.sin(angle_theta), torch.cos(angle_phi), torch.sin(angle_phi) * torch.cos(angle_theta)],
            [torch.cos(angle_theta), 0, -torch.sin(angle_theta)]
        ], device=device, dtype=dtype)
    else:
        raise ValueError(f"Unsupported dimension: {dim}")
    


def getBounds(values: torch.Tensor, options: PlottingOptions):
    q = values.detach().cpu().numpy()
    minScale = np.min(q) if options.vMin is None else options.vMin
    maxScale = np.max(q) if options.vMax is None else options.vMax
    if options.quantityScaling == PlotScaling.Symmetric or options.quantityScaling == PlotScaling.SymmetricLog:
        minScale = -np.max(np.abs(q)) if options.vMin is None else options.vMin
        maxScale = np.max(np.abs(q)) if options.vMax is None else options.vMax
        if options.midPoint is not None:
            if isinstance(options.midPoint, str):
                options.midPoint = np.median(q)
            minScale = options.midPoint-np.max(np.abs(q - options.midPoint)) if options.vMin is None else options.vMin
            maxScale = options.midPoint+np.max(np.abs(q - options.midPoint)) if options.vMax is None else options.vMax
        if options.quantityScaling == PlotScaling.SymmetricLog:
            minElement = np.min(np.abs(q)[np.abs(q)>0.0]) if np.sum(np.abs(q) > 0.0) > 0 else 1.0
            maxElement = np.max(np.abs(q)) if np.sum(np.abs(q) > 0.0) > 0 else 10.0
            maxScale = maxScale if maxScale > 0.0 else maxElement
            minScale = minScale if minScale > 0.0 else minElement
            norm = colors.SymLogNorm(linthresh=options.quantityLogThreshold, linscale=0.03, vmin=minScale, vmax=maxScale)
        else:
            norm = colors.CenteredNorm(vcenter=options.midPoint, halfrange = maxScale)
    else:
        if options.quantityScaling == PlotScaling.Logarithmic:
            minElement = np.min(np.abs(q)[np.abs(q)>0.0]) if np.sum(np.abs(q) > 0.0) > 0 else 1.0
            maxElement = np.max(np.abs(q)) if np.sum(np.abs(q) > 0.0) > 0 else 10.0
            maxScale = maxScale if maxScale > 0.0 else maxElement
            minScale = minScale if minScale > 0.0 else minElement
            norm = colors.LogNorm(vmin=minScale, vmax=maxScale)
        else:
            norm = colors.Normalize(vmin=minScale, vmax=maxScale)
    
    qs = q.clip(minScale, maxScale)
    return qs, norm


def mapQuantity(
    quantity: torch.Tensor,
    mapping: Mapping,
):
    if mapping == Mapping.none:
        return quantity
    elif mapping == Mapping.x:
        return quantity[:,0] if quantity.ndim > 1 and quantity.shape[1] > 0 else quantity.flatten()
    elif mapping == Mapping.y:
        return quantity[:,1]
    elif mapping == Mapping.z:
        return quantity[:,2]
    elif mapping == Mapping.L2Norm or mapping == Mapping.magnitude or mapping == Mapping.L2:
        return torch.linalg.norm(quantity, dim=-1)
    elif mapping == Mapping.L1:
        return torch.linalg.norm(quantity, dim=-1, ord=1)
    else:
        raise ValueError(f"Unsupported mapping: {mapping}")