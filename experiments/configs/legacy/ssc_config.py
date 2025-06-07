#!/usr/bin/env python3
"""
SSC数据集配置文件
"""

import torch

# 基础配置
BASE_CONFIG = {
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'seed': 42,
    'num_workers': 4,
    'pin_memory': True
}

# 网络配置
NETWORK_CONFIG = {
    'input_size': 700,
    'hidden_size': 200,
    'output_size': 35,
    'v_threshold': 1.0,
    'dt': 1.0
}

# 训练配置
TRAINING_CONFIG = {
    'batch_size': 32,  # 减小批次大小
    'learning_rate': 0.01,
    'epochs': 10,  # 减少训练轮数用于测试
    'weight_decay': 0.0,
    'grad_clip': None
}

# 时间因子配置
TIMING_CONFIG = {
    'small': (-4.0, 0.0),
    'medium': (0.0, 4.0),
    'large': (2.0, 6.0)
}

# DH-SNN配置
DH_CONFIG = {
    'num_branches': 8,
    'timing_init': 'large',
    'learnable_timing': True,
    'beneficial_init': True,
    'tau_m_init': TIMING_CONFIG['large'],  # 膜电位时间常数初始化
    'tau_n_init': TIMING_CONFIG['large']   # 树突时间常数初始化
}

# 数据配置
DATA_CONFIG = {
    'data_path': '../../../datasets/ssc/data/',
    'train_samples': None,  # None表示使用全部数据
    'test_samples': None,
    'validation_split': 0.1
}

# 实验配置
EXPERIMENT_CONFIG = {
    'num_trials': 5,
    'save_models': True,
    'save_results': True,
    'output_dir': 'outputs/ssc/'
}
