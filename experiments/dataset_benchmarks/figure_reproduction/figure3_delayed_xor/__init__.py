"""
SpikingJelly版本的DH-SNN (Dendritic Heterogeneity Spiking Neural Networks)

基于论文: "Temporal dendritic heterogeneity incorporated with spiking neural networks 
for learning multi-timescale dynamics" (Nature Communications, 2024)

这个包提供了完整的DH-SNN实现，包括：
- DH-LIF神经元模型
- 树突异质性密集层
- 多高斯替代函数
- 读出积分器
- 完整的网络架构
- 多种任务的训练脚本

主要模块：
- neurons: DH-LIF神经元和相关组件
- layers: 树突异质性层和读出层
- surrogate: 多高斯替代函数
- models: 完整的DH-SNN网络模型
- tasks: 各种任务的实现（SHD, SSC, GSC, TIMIT, DEAP等）
- utils: 工具函数和辅助类
"""

__version__ = "1.0.0"
__author__ = "DH-SNN SpikingJelly Implementation"

# 导入主要组件
from .neurons import *
from .layers import *
from .surrogate import *
from .models import *
from .utils import *

# 版本信息
__all__ = [
    # 神经元
    'DH_LIFNode',
    'ParametricLIFNode',
    
    # 层
    'DendriticDenseLayer', 
    'ReadoutIntegrator',
    'DendriticRNNLayer',
    
    # 替代函数
    'MultiGaussianSurrogate',
    
    # 模型
    'DH_SNN',
    'DH_SFNN', 
    'DH_SRNN',
    
    # 工具
    'ConnectionMask',
    'StateManager',
]
