"""
DH-SNN工具函数

包含训练、评估、数据处理等辅助功能。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from spikingjelly.activation_based import functional


class StateManager:
    """
    状态管理器
    
    用于管理DH-SNN网络的状态变量。
    """
    
    def __init__(self, model: nn.Module):
        self.model = model
    
    def reset_states(self):
        """
        重置所有状态
        """
        functional.reset_net(self.model)
        
        # 特殊处理SRNN的隐藏状态
        if hasattr(self.model, 'hidden_state'):
            self.model.hidden_state = None
    
    def get_state_dict(self) -> Dict[str, Any]:
        """
        获取状态字典
        """
        state_dict = {}
        
        def collect_states(module, prefix=''):
            for name, child in module.named_children():
                child_prefix = f"{prefix}.{name}" if prefix else name
                
                # 收集内存模块的状态
                if hasattr(child, '_memories'):
                    module_states = {}
                    for mem_name in child._memories:
                        value = getattr(child, mem_name, None)
                        if value is not None and isinstance(value, torch.Tensor):
                            module_states[mem_name] = value.clone()
                    if module_states:
                        state_dict[child_prefix] = module_states
                
                # 递归处理子模块
                collect_states(child, child_prefix)
        
        collect_states(self.model)
        return state_dict
    
    def load_state_dict(self, state_dict: Dict[str, Any]):
        """
        加载状态字典
        """
        def restore_states(module, prefix=''):
            for name, child in module.named_children():
                child_prefix = f"{prefix}.{name}" if prefix else name
                
                if child_prefix in state_dict:
                    module_states = state_dict[child_prefix]
                    for mem_name, value in module_states.items():
                        if hasattr(child, mem_name):
                            setattr(child, mem_name, value.clone())
                
                restore_states(child, child_prefix)
        
        restore_states(self.model)


class MetricsCalculator:
    """
    指标计算器
    
    计算各种评估指标。
    """
    
    @staticmethod
    def accuracy(outputs: torch.Tensor, targets: torch.Tensor) -> float:
        """
        计算准确率
        """
        if outputs.dim() == 3:  # [time_steps, batch_size, num_classes]
            # 对时间维度求平均
            outputs = outputs.mean(dim=0)
        
        predicted = torch.argmax(outputs, dim=1)
        correct = (predicted == targets).float()
        return correct.mean().item()
    
    @staticmethod
    def spike_rate(spikes: torch.Tensor) -> float:
        """
        计算脉冲发放率
        """
        return spikes.float().mean().item()
    
    @staticmethod
    def energy_consumption(model: nn.Module, spikes_dict: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        计算能耗指标
        """
        energy_dict = {}
        
        for name, spikes in spikes_dict.items():
            # 计算脉冲数量
            total_spikes = spikes.sum().item()
            # 计算神经元数量
            num_neurons = np.prod(spikes.shape[1:])  # 除了batch维度
            # 计算时间步数
            time_steps = spikes.shape[0] if spikes.dim() > 2 else 1
            
            # 归一化能耗 (脉冲数 / (神经元数 * 时间步))
            energy_dict[name] = total_spikes / (num_neurons * time_steps)
        
        return energy_dict


class DataProcessor:
    """
    数据处理器
    
    处理各种数据格式和预处理。
    """
    
    @staticmethod
    def encode_rate(data: torch.Tensor, time_steps: int, max_rate: float = 1.0) -> torch.Tensor:
        """
        速率编码
        
        Args:
            data: 输入数据 [batch_size, features]
            time_steps: 时间步数
            max_rate: 最大发放率
            
        Returns:
            脉冲序列 [time_steps, batch_size, features]
        """
        batch_size, features = data.shape
        
        # 归一化到[0, max_rate]
        normalized_data = torch.clamp(data, 0, 1) * max_rate
        
        # 生成随机数并比较
        random_vals = torch.rand(time_steps, batch_size, features, device=data.device)
        spikes = (random_vals < normalized_data.unsqueeze(0)).float()
        
        return spikes
    
    @staticmethod
    def encode_temporal(data: torch.Tensor, time_steps: int, encoding_type: str = 'linear') -> torch.Tensor:
        """
        时间编码
        
        Args:
            data: 输入数据 [batch_size, features]
            time_steps: 时间步数
            encoding_type: 编码类型 ('linear', 'exponential')
            
        Returns:
            脉冲序列 [time_steps, batch_size, features]
        """
        batch_size, features = data.shape
        spikes = torch.zeros(time_steps, batch_size, features, device=data.device)
        
        # 归一化到[0, time_steps-1]
        normalized_data = torch.clamp(data, 0, 1) * (time_steps - 1)
        
        if encoding_type == 'linear':
            spike_times = normalized_data.long()
        elif encoding_type == 'exponential':
            spike_times = (time_steps - 1 - torch.exp(-normalized_data * 3) * (time_steps - 1)).long()
        else:
            raise ValueError(f"Unsupported encoding_type: {encoding_type}")
        
        # 设置脉冲
        for t in range(time_steps):
            mask = (spike_times == t)
            spikes[t][mask] = 1.0
        
        return spikes
    
    @staticmethod
    def decode_rate(spikes: torch.Tensor, time_window: Optional[int] = None) -> torch.Tensor:
        """
        速率解码
        
        Args:
            spikes: 脉冲序列 [time_steps, batch_size, features]
            time_window: 时间窗口 (None表示使用全部时间)
            
        Returns:
            解码后的数据 [batch_size, features]
        """
        if time_window is not None:
            spikes = spikes[-time_window:]
        
        return spikes.mean(dim=0)
    
    @staticmethod
    def add_noise(spikes: torch.Tensor, noise_rate: float = 0.01) -> torch.Tensor:
        """
        添加噪声
        
        Args:
            spikes: 脉冲序列
            noise_rate: 噪声率
            
        Returns:
            添加噪声后的脉冲序列
        """
        noise = torch.rand_like(spikes) < noise_rate
        return torch.clamp(spikes + noise.float(), 0, 1)


class ConfigManager:
    """
    配置管理器
    
    管理模型配置和超参数。
    """
    
    @staticmethod
    def get_default_config() -> Dict[str, Any]:
        """
        获取默认配置
        """
        return {
            # 模型架构
            'model_type': 'feedforward',
            'input_dim': 700,
            'hidden_dims': [256, 128],
            'output_dim': 20,
            'num_branches': 4,
            
            # 神经元参数
            'v_threshold': 0.5,
            'tau_m_init': (0.0, 4.0),
            'tau_n_init': (0.0, 4.0),
            'tau_initializer': 'uniform',
            'reset_mode': 'soft',
            
            # 连接参数
            'sparsity': None,  # 默认为1/num_branches
            'mask_share': 1,
            'bias': True,
            
            # 读出层参数
            'readout_tau_init': (0.0, 4.0),
            
            # 训练参数
            'learning_rate': 1e-3,
            'batch_size': 32,
            'num_epochs': 100,
            'time_steps': 40,
            
            # 正则化
            'weight_decay': 1e-4,
            'dropout': 0.0,
            
            # 其他
            'step_mode': 's',
            'device': 'cuda' if torch.cuda.is_available() else 'cpu',
            'seed': 42,
        }
    
    @staticmethod
    def update_config(base_config: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
        """
        更新配置
        """
        config = base_config.copy()
        config.update(updates)
        return config
    
    @staticmethod
    def validate_config(config: Dict[str, Any]) -> bool:
        """
        验证配置
        """
        required_keys = ['input_dim', 'output_dim', 'hidden_dims']
        
        for key in required_keys:
            if key not in config:
                raise ValueError(f"Missing required config key: {key}")
        
        # 验证维度
        if config['input_dim'] <= 0 or config['output_dim'] <= 0:
            raise ValueError("Input and output dimensions must be positive")
        
        if not isinstance(config['hidden_dims'], list) or len(config['hidden_dims']) == 0:
            raise ValueError("hidden_dims must be a non-empty list")
        
        return True


def set_seed(seed: int):
    """
    设置随机种子
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def count_parameters(model: nn.Module) -> int:
    """
    计算模型参数数量
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_model_info(model: nn.Module) -> Dict[str, Any]:
    """
    获取模型信息
    """
    info = {
        'total_parameters': count_parameters(model),
        'model_type': getattr(model, 'network_type', 'unknown'),
        'architecture': str(model),
    }
    
    # 获取时间常数信息
    if hasattr(model, 'get_tau_parameters'):
        tau_m_list, tau_n_list, readout_tau = model.get_tau_parameters()
        info['tau_ranges'] = {
            'tau_m': [(t.min().item(), t.max().item()) for t in tau_m_list],
            'tau_n': [(t.min().item(), t.max().item()) for t in tau_n_list],
            'readout_tau': (readout_tau.min().item(), readout_tau.max().item())
        }
    
    return info
