#!/usr/bin/env python3
"""
Sequential MNIST实验配置文件
"""

import torch
import os

# 基础配置
BASE_CONFIG = {
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'seed': 42,
    'num_workers': 4,
    'pin_memory': True
}

# 网络配置
NETWORK_CONFIG = {
    'input_size': 1,
    'hidden_size': 256,
    'output_size': 10,
    'v_threshold': 1.0,
    'dt': 1.0
}

# 训练配置
TRAINING_CONFIG = {
    'batch_size': 128,
    'learning_rate': 0.001,
    'epochs': 150,
    'weight_decay': 0.0,
    'grad_clip': None,
    'lr_decay': 0.1,
    'lr_step': 50
}

# 数据配置
DATA_CONFIG = {
    'data_path': './mnist_data/',
    'sequence_length': 784
}

# DH-SNN配置
DH_CONFIG = {
    'num_branches': 4,  # 根据sequential_mnist实验
    'timing_init': 'large',
    'learnable_timing': True,
    'beneficial_init': True,
    'tau_m_init': (2.0, 6.0),  # Large配置
    'tau_n_init': (2.0, 6.0)   # Large配置
}

# 时间因子配置
TIMING_CONFIG = {
    'small': (-4.0, 0.0),
    'medium': (0.0, 4.0), 
    'large': (2.0, 6.0)
}

# 实验配置
EXPERIMENT_CONFIG = {
    'output_dir': './results',
    'save_model': True,
    'save_logs': True,
    'log_interval': 10
}

# 合并所有配置
ALL_CONFIG = {
    **BASE_CONFIG,
    **NETWORK_CONFIG, 
    **TRAINING_CONFIG,
    **DATA_CONFIG,
    **DH_CONFIG,
    **EXPERIMENT_CONFIG
}
