#!/usr/bin/env python3
"""
SSC (Spiking Speech Commands) 实验配置文件
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
    'input_size': 700,  # SSC特征维度
    'hidden_size': 200,
    'output_size': 35,  # SSC类别数
    'v_threshold': 1.0,
    'dt': 1.0
}

# 训练配置
TRAINING_CONFIG = {
    'batch_size': 100,
    'learning_rate': 0.01,
    'epochs': 100,
    'weight_decay': 0.0,
    'grad_clip': None
}

# 数据配置
DATA_CONFIG = {
    'data_path': '../../../datasets/ssc/',
    'max_time': 1.0,  # 最大时间长度(秒)
    'dt': 0.001,      # 时间步长
    'num_bins': 1000  # 时间bins数量
}

# DH-SNN配置
DH_CONFIG = {
    'num_branches': 2,  # 根据原论文
    'timing_init': 'medium',  # SSC使用medium配置
    'learnable_timing': True,
    'beneficial_init': True,
    'tau_m_init': (0.0, 4.0),  # Medium配置
    'tau_n_init': (0.0, 4.0)   # Medium配置
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
