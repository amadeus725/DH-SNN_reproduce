"""
DH-SNN 多时间尺度XOR实验配置
专门用于Figure 4多时间尺度XOR问题的配置
"""

import torch

# 设备配置
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 双分支DH-SFNN配置
TWO_BRANCH_CONFIG = {
    'input_size': 40,  # 20 + 20 (Signal 1 + Signal 2)
    'hidden_size': 64,
    'output_size': 1,
    'num_branches': 2,
    'v_threshold': 1.0,
    'beneficial_init': True,  # 有益初始化
    'learnable_timing': True
}

# 有益初始化配置
BENEFICIAL_INIT_CONFIG = {
    'branch1_tau_n': (2.0, 6.0),   # Large初始化 - 长期记忆Signal 1
    'branch2_tau_n': (-4.0, 0.0),  # Small初始化 - 快速响应Signal 2
    'tau_m': (0.0, 4.0)            # Medium初始化
}

# 随机初始化配置
RANDOM_INIT_CONFIG = {
    'branch1_tau_n': (0.0, 4.0),   # Medium初始化
    'branch2_tau_n': (0.0, 4.0),   # Medium初始化
    'tau_m': (0.0, 4.0)            # Medium初始化
}

# 训练配置
TRAINING_CONFIG = {
    'batch_size': 100,
    'learning_rate': 0.01,
    'epochs': 100,
    'weight_decay': 0.0
}

# 任务配置
TASK_CONFIG = {
    'sequence_length': 50,
    'signal1_freq': 0.1,    # 低频信号
    'signal2_freq': 0.5,    # 高频信号
    'num_samples': 2000
}