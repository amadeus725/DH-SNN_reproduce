#!/usr/bin/env python3
"""
NeuroVPR 视觉位置识别 实验配置文件
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
    'input_size': 346,  # NeuroVPR特征维度
    'hidden_size': 256,
    'output_size': 100, # 位置类别数
    'v_threshold': 1.0,
    'dt': 1.0
}

# 训练配置
TRAINING_CONFIG = {
    'batch_size': 32,
    'learning_rate': 0.001,
    'epochs': 200,
    'weight_decay': 1e-4,
    'grad_clip': None
}

# 数据配置
DATA_CONFIG = {
    'data_path': '../../../datasets/neurovpr/',
    'image_size': (346, 240),
    'sequence_length': 100,
    'time_window': 50  # ms
}

# DH-SNN配置
DH_CONFIG = {
    'num_branches': 2,  # 根据原论文
    'timing_init': 'large',  # NeuroVPR使用large配置
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
