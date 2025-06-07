"""
DH-SNN核心模块

包含DH-SNN的所有核心组件：神经元、层、模型、替代函数和工具函数。
"""

from .neurons import DH_LIFNode, ParametricLIFNode, ReadoutNeuron
from .layers import DendriticDenseLayer, ReadoutIntegrator, ConnectionMask
from .models import DH_SNN, DH_SFNN, DH_SRNN, create_dh_snn
from .surrogate import MultiGaussianSurrogate, AdaptiveMultiGaussianSurrogate
from .utils import StateManager, MetricsCalculator, DataProcessor, ConfigManager

__all__ = [
    # 神经元
    'DH_LIFNode', 'ParametricLIFNode', 'ReadoutNeuron',
    # 层
    'DendriticDenseLayer', 'ReadoutIntegrator', 'ConnectionMask',
    # 模型
    'DH_SNN', 'DH_SFNN', 'DH_SRNN', 'create_dh_snn',
    # 替代函数
    'MultiGaussianSurrogate', 'AdaptiveMultiGaussianSurrogate',
    # 工具
    'StateManager', 'MetricsCalculator', 'DataProcessor', 'ConfigManager'
]
