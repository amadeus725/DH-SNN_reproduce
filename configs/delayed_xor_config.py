"""
DH-SNN 延迟XOR实验配置
专门用于Figure 3延迟XOR问题的配置
"""

import torch

# 设备配置
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 模型配置
MODEL_CONFIG = {
    'input_size': 2,
    'hidden_size': 64,
    'output_size': 1,
    'num_branches': 1,
    'v_threshold': 1.0,
    'tau_m': 20.0
}

# 训练配置
TRAINING_CONFIG = {
    'batch_size': 100,
    'learning_rate': 0.01,
    'epochs': 100,
    'weight_decay': 0.0
}

# DH-SNN时间因子配置 - 三种初始化方式
TIMING_CONFIGS = {
    'Small': {
        'tau_m_init': (-4.0, 0.0),
        'tau_n_init': (-4.0, 0.0)
    },
    'Medium': {
        'tau_m_init': (0.0, 4.0),
        'tau_n_init': (0.0, 4.0)
    },
    'Large': {
        'tau_m_init': (2.0, 6.0),
        'tau_n_init': (2.0, 6.0)
    }
}

# 延迟XOR任务配置
TASK_CONFIG = {
    'sequence_length': 20,
    'delay_length': 10,
    'num_samples': 1000
}