# DH-SNN Core Package

def setup_seed(seed=42):
    """设置随机种子以保证实验可重现性"""
    import random
    import numpy as np
    import torch
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # 为了保证可重现性
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

__all__ = ['setup_seed']
