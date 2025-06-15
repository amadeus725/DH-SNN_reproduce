"""
DH-SNN 序列学习实验配置
专门用于Sequential MNIST和Permuted MNIST的配置
"""

import torch

# 设备配置
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Sequential MNIST配置
SEQUENTIAL_MNIST_CONFIG = {
    'input_size': 1,
    'hidden_size': 128,
    'output_size': 10,
    'num_branches': 4,
    'v_threshold': 0.5,
    'tau_m_init': (0.0, 4.0),  # Medium初始化
    'tau_n_init': (2.0, 6.0),  # Large初始化，长期记忆
    'sequence_length': 784  # 28*28像素序列
}

# Permuted MNIST配置
PERMUTED_MNIST_CONFIG = {
    'input_size': 1,
    'hidden_size': 256,
    'output_size': 10,
    'num_branches': 8,
    'v_threshold': 0.5,
    'tau_m_init': (0.0, 4.0),  # Medium初始化
    'tau_n_init': (2.0, 6.0),  # Large初始化，长期记忆
    'sequence_length': 784,
    'num_permutations': 10  # 排列数量
}

# 训练配置
TRAINING_CONFIG = {
    'batch_size': 128,
    'learning_rate': 0.001,
    'tau_lr_multiplier': 2.0,  # 时间常数使用2倍学习率
    'epochs': 100,
    'weight_decay': 1e-4,
    'lr_scheduler': 'cosine',
    'warmup_epochs': 5
}

# 数据配置
DATA_CONFIG = {
    'normalize': True,
    'shuffle': True,
    'num_workers': 4,
    'pin_memory': True
}