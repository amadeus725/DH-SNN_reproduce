"""
DH-SNN 默认配置文件
通用配置，适用于基础实验
"""

import torch

# 设备配置
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 模型配置 - 通用设置
MODEL_CONFIG = {
    'input_size': 64,
    'hidden_size': 128,
    'output_size': 10,
    'num_branches': 4,
    'v_threshold': 1.0,
    'tau_m_init': (0.0, 4.0),  # Medium初始化
    'tau_n_init': (0.0, 4.0)   # Medium初始化
}

# 训练配置 - 通用设置
TRAINING_CONFIG = {
    'batch_size': 32,
    'learning_rate': 0.001,
    'epochs': 100,
    'weight_decay': 0.0
}

# DH-SNN配置 - 通用设置
DH_CONFIG = {
    'sparsity': 0.25,  # 1.0/num_branches
    'mask_share': 1,
    'bias': True,
    'reset_mode': 'soft',
    'step_mode': 's'
}

# 数据配置 - 通用设置
DATA_CONFIG = {
    'train_ratio': 0.8,
    'val_ratio': 0.1,
    'test_ratio': 0.1
}
