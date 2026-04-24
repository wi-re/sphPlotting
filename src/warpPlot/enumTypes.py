from .colorMaps import ColorMap
from enum import Enum

class VisualizeOptions(Enum):
    Hide = 0
    Visualize = 1
    Passive = 2


class PlotScaling(Enum):
    Linear = 'linear'
    Logarithmic = 'log'
    SymmetricLog = 'symlog'
    Symmetric = 'sym'

class Mapping(Enum):
    none = 'none'
    x = 'x'
    y = 'y'
    z = 'z'
    L2Norm = 'L2Norm'
    L2 = 'L2'
    L1 = 'L1'
    magnitude = 'magnitude'
    
class StreamLineLocation(Enum):
    BeforeOperation = 0
    BeforeMapping = 1
    AfterMapping = 2