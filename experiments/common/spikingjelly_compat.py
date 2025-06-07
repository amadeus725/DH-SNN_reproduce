"""
SpikingJelly兼容性层
为legacy代码提供向后兼容支持
"""

# 重新导出常用组件，保持向后兼容
from dh_snn.core.models import DH_SFNN as PaperEquivalentDH_SFNN
from dh_snn.core.models import VanillaSFNN as PaperEquivalentVanillaSFNN  
from dh_snn.core.neurons import DH_LIFNode as PaperEquivalentLIFNode
from dh_snn.core.neurons import ParametricLIFNode as CustomParametricLIFNode
from dh_snn.core.layers import ReadoutIntegrator as PaperEquivalentReadoutIntegrator
from dh_snn.core.layers import DendriticDenseLayer
from dh_snn.core.surrogate import MultiGaussianSurrogate

# 导出所有兼容性别名
__all__ = [
    'PaperEquivalentDH_SFNN',
    'PaperEquivalentVanillaSFNN', 
    'PaperEquivalentLIFNode',
    'CustomParametricLIFNode',
    'PaperEquivalentReadoutIntegrator',
    'DendriticDenseLayer',
    'MultiGaussianSurrogate'
]
