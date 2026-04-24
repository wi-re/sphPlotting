from dataclasses import dataclass
from enum import Enum, auto
from .colorMaps import ColorMap
from .enumTypes import VisualizeOptions, PlotScaling, Mapping, StreamLineLocation
from sphWarpCore import GradientScheme, LaplacianScheme, OperationProperties, WarpOperation, OperationDirection, KernelFunctions, SupportScheme, ParticleType
import torch
from typing import Optional, Union


@dataclass
class GridVisualization:
    resolution: int = 128
    streamLines: bool = False

    streamLineOperation: Optional[WarpOperation] = None
    streamLineGradientMode: GradientScheme = GradientScheme.Difference
    streamLineLaplaceMode: LaplacianScheme = LaplacianScheme.Brookshaw
    streamLinePositiveDivergence: bool = False
    streamLineOperationLocation: StreamLineLocation = StreamLineLocation.BeforeOperation # If true, the mapping will be applied to the quantity before computing the streamlines

@dataclass
class PlottingOptions:
    plottingKernel: KernelFunctions = KernelFunctions.Wendland4
    plottingOperation: Optional[OperationProperties] = None
    
    fluidVisualization: VisualizeOptions = VisualizeOptions.Visualize
    boundaryVisualization: VisualizeOptions = VisualizeOptions.Hide

    showColorBar: bool = True
    colorMap: ColorMap = ColorMap.viridis
    flipColorMap: bool = False # adds _r to the colormap name

    quantityScaling: PlotScaling = PlotScaling.Linear
    quantityLogThreshold: float = 1e-3 # Only used if quantityScaling is set to SymmetricLog
    vMin: Optional[float] = None
    vMax: Optional[float] = None
    midPoint: Optional[Union[float,str]] = None # Used to center the scaling around a value with equal positive and negative bands

    domainEpsilon: float = 0.05
    markerSize: Optional[float] = None # If None, marker size will be automatically computed based on the grid resolution and domain size

    mapping : Mapping = Mapping.none
    gridVisualization: Optional[GridVisualization] = None

    plotDomain: bool = True
    plotTitle : Optional[str] = None
