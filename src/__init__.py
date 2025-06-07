"""
DH-SNN (Dendritic Heterogeneity Spiking Neural Networks) 核心库
"""

__version__ = "1.0.0"
__author__ = "Htz"

# 导入核心组件
from .core.neurons import DH_LIFNode, ParametricLIFNode
from .core.layers import DendriticDenseLayer, ReadoutIntegrator
from .core.models import DH_SNN, DH_SFNN, DH_SRNN, create_dh_snn
from .core.surrogate import MultiGaussianSurrogate
from .core.utils import *

__all__ = [
    'DH_LIFNode', 'ParametricLIFNode',
    'DendriticDenseLayer', 'ReadoutIntegrator', 
    'DH_SNN', 'DH_SFNN', 'DH_SRNN', 'create_dh_snn',
    'MultiGaussianSurrogate'
]
