#!/usr/bin/env python3
"""
基础配置类
定义所有实验的通用配置
"""

import torch
import os
from typing import Dict, Any, Optional
from dataclasses import dataclass, field

@dataclass
class BaseConfig:
    """基础配置类"""
    
    # 设备配置
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    seed: int = 42
    
    # 路径配置
    project_root: str = '/root/DH-SNN_reproduce'
    data_root: str = field(default_factory=lambda: os.path.join('/root/DH-SNN_reproduce'))
    output_root: str = field(default_factory=lambda: os.path.join('/root/DH-SNN_reproduce/spikingjelly_delayed_xor/outputs'))
    
    # 数据路径
    shd_data_path: str = field(default_factory=lambda: '../../../datasets/shd/data')
    ssc_data_path: str = field(default_factory=lambda: '../../../datasets/ssc/data')
    
    # 日志配置
    log_level: str = 'INFO'
    log_to_file: bool = True
    log_to_console: bool = True
    
    # 可视化配置
    figure_format: str = 'png'
    figure_dpi: int = 300
    figure_size: tuple = (10, 6)
    
    def __post_init__(self):
        """初始化后处理"""
        # 创建输出目录
        os.makedirs(self.output_root, exist_ok=True)
        os.makedirs(os.path.join(self.output_root, 'results'), exist_ok=True)
        os.makedirs(os.path.join(self.output_root, 'figures'), exist_ok=True)
        os.makedirs(os.path.join(self.output_root, 'models'), exist_ok=True)
        os.makedirs(os.path.join(self.output_root, 'logs'), exist_ok=True)
        
        # 设置随机种子
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.seed)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'device': self.device,
            'seed': self.seed,
            'project_root': self.project_root,
            'data_root': self.data_root,
            'output_root': self.output_root,
            'shd_data_path': self.shd_data_path,
            'ssc_data_path': self.ssc_data_path,
            'log_level': self.log_level,
            'log_to_file': self.log_to_file,
            'log_to_console': self.log_to_console,
            'figure_format': self.figure_format,
            'figure_dpi': self.figure_dpi,
            'figure_size': self.figure_size
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'BaseConfig':
        """从字典创建配置"""
        return cls(**config_dict)

@dataclass
class TimingFactorConfig:
    """时间因子配置"""
    
    # 三种标准配置 - 按照论文Table S3
    small: tuple = (-4.0, 0.0)      # β̂,α̂ ~ U(-4,0)
    medium: tuple = (0.0, 4.0)      # β̂,α̂ ~ U(0,4)
    large: tuple = (2.0, 6.0)       # β̂,α̂ ~ U(2,6)
    
    # 有益初始化配置
    beneficial_branch1: tuple = (2.0, 6.0)   # 大时间常数用于长期记忆
    beneficial_branch2: tuple = (-4.0, 0.0)  # 小时间常数用于快速响应
    
    def get_range(self, config_name: str) -> tuple:
        """获取时间因子范围"""
        if hasattr(self, config_name):
            return getattr(self, config_name)
        else:
            raise ValueError(f"Unknown timing factor config: {config_name}")

@dataclass
class NetworkConfig:
    """网络架构配置"""
    
    # SHD配置
    shd_input_size: int = 700
    shd_hidden_size: int = 64
    shd_output_size: int = 20
    
    # SSC配置
    ssc_input_size: int = 700
    ssc_hidden_size: int = 200
    ssc_output_size: int = 35
    
    # 多时间尺度XOR配置
    xor_input_size: int = 100
    xor_hidden_size: int = 64
    xor_output_size: int = 1
    
    # 神经元参数
    v_threshold: float = 1.0
    dt: float = 1.0
    
    # 分支配置
    max_branches: int = 8
    default_branches: int = 4

@dataclass
class TrainingConfig:
    """训练配置"""
    
    # 基础训练参数
    learning_rate: float = 1e-2
    batch_size: int = 100
    epochs: int = 100
    
    # 优化器配置
    optimizer: str = 'adam'
    weight_decay: float = 0.0
    
    # 学习率调度
    scheduler: str = 'step'
    step_size: int = 20
    gamma: float = 0.5
    
    # 分组学习率 - 按照原论文
    tau_lr_multiplier: float = 2.0  # 时间常数使用2倍学习率
    
    # 早停配置
    early_stopping: bool = False
    patience: int = 20
    min_delta: float = 0.001
    
    # 梯度裁剪
    grad_clip: Optional[float] = None
    
    # 验证配置
    validation_split: float = 0.1
    validation_freq: int = 1

@dataclass
class ExperimentConfig:
    """实验配置"""
    
    # 重复实验
    num_trials: int = 5
    
    # 数据配置
    train_samples: int = 2000
    test_samples: int = 500
    
    # 统计分析
    confidence_level: float = 0.95
    significance_level: float = 0.05
    
    # 可视化
    save_figures: bool = True
    show_figures: bool = False
    
    # 模型保存
    save_models: bool = True
    save_best_only: bool = True
    
    # 结果保存
    save_results: bool = True
    result_format: str = 'pth'  # 'pth', 'json', 'csv'

# 预定义配置组合
class ConfigPresets:
    """预定义配置组合"""
    
    @staticmethod
    def get_shd_config() -> Dict[str, Any]:
        """获取SHD实验配置"""
        base = BaseConfig()
        network = NetworkConfig()
        training = TrainingConfig()
        experiment = ExperimentConfig()
        timing = TimingFactorConfig()
        
        return {
            'base': base,
            'network': {
                'input_size': network.shd_input_size,
                'hidden_size': network.shd_hidden_size,
                'output_size': network.shd_output_size,
                'v_threshold': network.v_threshold,
                'dt': network.dt
            },
            'training': training,
            'experiment': experiment,
            'timing': timing
        }
    
    @staticmethod
    def get_ssc_config() -> Dict[str, Any]:
        """获取SSC实验配置"""
        base = BaseConfig()
        network = NetworkConfig()
        training = TrainingConfig()
        experiment = ExperimentConfig()
        timing = TimingFactorConfig()
        
        # SSC特定调整
        training.batch_size = 200  # SSC使用更大的批次
        
        return {
            'base': base,
            'network': {
                'input_size': network.ssc_input_size,
                'hidden_size': network.ssc_hidden_size,
                'output_size': network.ssc_output_size,
                'v_threshold': network.v_threshold,
                'dt': network.dt
            },
            'training': training,
            'experiment': experiment,
            'timing': timing
        }
    
    @staticmethod
    def get_multi_timescale_xor_config() -> Dict[str, Any]:
        """获取多时间尺度XOR实验配置"""
        base = BaseConfig()
        network = NetworkConfig()
        training = TrainingConfig()
        experiment = ExperimentConfig()
        timing = TimingFactorConfig()
        
        # 多时间尺度XOR特定调整
        training.learning_rate = 1e-3
        training.batch_size = 32
        training.epochs = 200
        experiment.num_trials = 10
        
        return {
            'base': base,
            'network': {
                'input_size': network.xor_input_size,
                'hidden_size': network.xor_hidden_size,
                'output_size': network.xor_output_size,
                'v_threshold': network.v_threshold,
                'dt': network.dt
            },
            'training': training,
            'experiment': experiment,
            'timing': timing
        }
