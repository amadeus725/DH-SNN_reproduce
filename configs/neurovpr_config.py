"""
DH-SNN NeuroVPR实验配置
专门用于视觉位置识别任务的配置
"""

import torch

# 设备配置
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# NeuroVPR模型配置
MODEL_CONFIG = {
    'input_size': 2752,  # 2 * 32 * 43 (极性 * 高度 * 宽度)
    'hidden_size': 512,
    'output_size': 25,   # 类别数
    'num_branches': 2,   # 减少分支数以适应短序列
    'v_threshold': 0.3,  # 适中阈值
    'tau_m_init': (0.1, 1.0),  # 短时间常数，适合短序列
    'tau_n_init': (0.1, 1.0),  # 短时间常数，适合短序列
}

# 训练配置 - 优化版本
TRAINING_CONFIG = {
    'batch_size': 32,
    'learning_rate': 0.001,
    'tau_lr_multiplier': 2.0,  # 时间常数使用2倍学习率
    'epochs': 50,
    'weight_decay': 1e-5,
    'gradient_clip': 1.0,
    'lr_scheduler': 'step',
    'lr_step_size': 20,
    'lr_gamma': 0.5
}

# 数据配置
DATA_CONFIG = {
    'sequence_length': 3,  # DVS序列长度
    'downsample_factor': 2,  # 下采样因子
    'train_exp_idx': [1, 2, 3, 5, 6, 7],
    'test_exp_idx': [4],
    'data_path': '/data/room/room_v'
}

# DH配置 - 针对短序列优化
DH_CONFIG = {
    'sparsity': 0.5,  # 1.0/num_branches
    'mask_share': 1,
    'reset_mode': 'soft',
    'step_mode': 's'
}