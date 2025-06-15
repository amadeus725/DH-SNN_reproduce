"""
DH-SNN: 树突异质性脉冲神经网络

最小实现版本 - 只包含核心功能
支持胞体异质性(SH-SNN)和树突异质性(DH-SNN)对比
"""

__version__ = "1.0.0-minimal"

# 核心导入
try:
    from .utils import setup_seed
except ImportError:
    def setup_seed(seed=42):
        """简单的随机种子设置函数"""
        import random
        import numpy as np
        import torch
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

__all__ = ['setup_seed']
