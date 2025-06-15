"""
DH-SNN 语音识别实验配置
专门用于SHD和SSC数据集的配置
"""

import torch

# 设备配置
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# SHD数据集配置
SHD_CONFIG = {
    'input_size': 700,
    'hidden_size': 256,
    'output_size': 20,
    'num_branches': 8,
    'v_threshold': 0.5,
    'tau_m_init': (0.0, 4.0),  # Medium初始化
    'tau_n_init': (2.0, 6.0),  # Large初始化
    'sequence_length': 100
}

# SSC数据集配置
SSC_CONFIG = {
    'input_size': 700,
    'hidden_size': 128,
    'output_size': 35,
    'num_branches': 4,
    'v_threshold': 0.5,
    'tau_m_init': (0.0, 4.0),  # Medium初始化
    'tau_n_init': (2.0, 6.0),  # Large初始化
    'sequence_length': 100
}

# 训练配置
TRAINING_CONFIG = {
    'batch_size': 64,
    'learning_rate': 0.001,
    'epochs': 200,
    'weight_decay': 1e-4,
    'lr_scheduler': 'cosine',
    'warmup_epochs': 10
}

# 数据配置
DATA_CONFIG = {
    'train_ratio': 0.8,
    'val_ratio': 0.1,
    'test_ratio': 0.1,
    'dt': 1.0,  # 采样时间间隔(ms)
    'augmentation': True
}