"""
胞体异质性模块 - DH-SNN核心创新
Soma Heterogeneity Module - Core Innovation of DH-SNN


"""

import torch
import torch.nn as nn
import numpy as np
from spikingjelly.clock_driven import neuron, functional
from typing import Tuple, Dict, List, Optional
import random


class SomaHeterogeneousLIF(neuron.LIFNode):
    """
    胞体异质性LIF神经元 - 核心创新
    Soma Heterogeneous LIF Neuron - Core Innovation
    
    实现胞体的多种异质性特征：
    1. 膜时间常数异质性 (τ_m) - 快速vs慢速整合
    2. 阈值异质性 (V_th) - 敏感vs迟钝神经元
    3. 重置电位异质性 (V_reset) - 强vs弱重置
    4. 适应性异质性 - 短期vs长期适应
    """
    
    def __init__(self, 
                 n_neurons: int,
                 tau_range: Tuple[float, float] = (2.0, 20.0),
                 v_th_range: Tuple[float, float] = (0.8, 1.2),
                 v_reset_range: Tuple[float, float] = (-0.2, 0.0),
                 adaptation_ratio: float = 0.3,
                 surrogate_function=None,
                 **kwargs):
        super().__init__(surrogate_function=surrogate_function, **kwargs)
        
        self.n_neurons = n_neurons
        
        # 生成异质性参数
        self.tau_m_hetero = self._sample_parameters(tau_range, n_neurons)
        self.v_th_hetero = self._sample_parameters(v_th_range, n_neurons) 
        self.v_reset_hetero = self._sample_parameters(v_reset_range, n_neurons)
        
        # 适应性掩码 (部分神经元具有适应性)
        self.adaptation_mask = torch.rand(n_neurons) < adaptation_ratio
        self.adaptation_strength = torch.where(
            self.adaptation_mask,
            torch.rand(n_neurons) * 0.1 + 0.05,  # 适应强度 0.05-0.15
            torch.zeros(n_neurons)
        )
        
        # 注册为缓冲区，使其能够在设备间移动
        self.register_buffer('tau_m_buf', self.tau_m_hetero)
        self.register_buffer('v_th_buf', self.v_th_hetero)
        self.register_buffer('v_reset_buf', self.v_reset_hetero)
        self.register_buffer('adaptation_mask_buf', self.adaptation_mask)
        self.register_buffer('adaptation_strength_buf', self.adaptation_strength)
        
        # 适应性状态
        self.v_th_adapted = None
        
    def _sample_parameters(self, param_range: Tuple[float, float], size: int) -> torch.Tensor:
        """从范围内采样异质性参数"""
        low, high = param_range
        return torch.rand(size) * (high - low) + low
    
    def neuronal_charge(self, x: torch.Tensor):
        """
        重写神经元充电过程，实现异质性膜时间常数
        核心创新：不同神经元具有不同的时间尺度整合能力
        """
        if self.v is None:
            self.v = torch.zeros_like(x.data)
            if self.v_th_adapted is None:
                self.v_th_adapted = self.v_th_buf.clone()
        
        # 使用异质性膜时间常数
        tau_m = self.tau_m_buf.to(x.device)
        decay_factor = 1.0 / tau_m
        
        # 异质性膜动力学: v[t] = v[t-1] * (1 - 1/τ) + x[t]
        # 快速神经元（小τ）：快速响应短时特征
        # 慢速神经元（大τ）：整合长时特征
        self.v = self.v * (1.0 - decay_factor) + x
    
    def neuronal_fire(self) -> torch.Tensor:
        """
        重写发放过程，实现异质性阈值和重置
        核心创新：不同敏感度的神经元处理不同强度的信号
        """
        # 使用适应性阈值
        v_th = self.v_th_adapted.to(self.v.device)
        spike = (self.v >= v_th).float()
        
        # 异质性重置 - 不同重置强度影响后续动态
        v_reset = self.v_reset_buf.to(self.v.device)
        self.v = torch.where(spike.bool(), v_reset, self.v)
        
        # 阈值适应 - 发放后阈值升高，模拟神经元疲劳
        adaptation_strength = self.adaptation_strength_buf.to(self.v.device)
        self.v_th_adapted = torch.where(
            spike.bool() & self.adaptation_mask_buf.to(self.v.device),
            self.v_th_adapted + adaptation_strength,  # 发放后阈值升高
            self.v_th_adapted * 0.99  # 缓慢恢复
        )
        
        return spike
    
    def get_heterogeneity_info(self) -> Dict[str, torch.Tensor]:
        """获取异质性参数信息，用于分析和可视化"""
        return {
            'tau_m': self.tau_m_buf,
            'v_th': self.v_th_buf,
            'v_reset': self.v_reset_buf,
            'adaptation_mask': self.adaptation_mask_buf,
            'adaptation_strength': self.adaptation_strength_buf
        }


class SomaHeterogeneousSNN(nn.Module):
    """
    胞体异质性脉冲神经网络 - 完整架构
    Soma Heterogeneous SNN - Complete Architecture
    
    专门设计用于多时间尺度任务
    """
    
    def __init__(self, 
                 input_size: int = 2,
                 hidden_size: int = 128,
                 output_size: int = 2,
                 tau_range: Tuple[float, float] = (2.0, 20.0),
                 v_th_range: Tuple[float, float] = (0.8, 1.2),
                 adaptation_ratio: float = 0.3):
        super().__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        # 输入投影层
        self.fc1 = nn.Linear(input_size, hidden_size)
        
        # 核心创新：胞体异质性层
        self.hetero_lif = SomaHeterogeneousLIF(
            n_neurons=hidden_size,
            tau_range=tau_range,
            v_th_range=v_th_range,
            adaptation_ratio=adaptation_ratio
        )
        
        # 输出层
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.lif_out = neuron.LIFNode(tau=10.0)
        
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        前向传播
        
        Args:
            x: (batch_size, time_steps, input_size)
        Returns:
            dict with:
                - output: (batch_size, output_size) - 累积输出
                - hidden_spikes: (batch_size, time_steps, hidden_size) - 隐藏层脉冲
        """
        batch_size, time_steps, _ = x.shape
        
        # 重置所有神经元状态
        functional.reset_net(self)
        
        output_spikes = []
        hidden_spikes = []
        
        for t in range(time_steps):
            # 输入到隐藏层
            h1 = self.fc1(x[:, t, :])
            s1 = self.hetero_lif(h1)  # 核心异质性处理
            hidden_spikes.append(s1)
            
            # 隐藏层到输出层
            h2 = self.fc2(s1)
            s2 = self.lif_out(h2)
            output_spikes.append(s2)
        
        # 累积时间维度的脉冲
        output = torch.stack(output_spikes, dim=1).sum(dim=1)
        hidden_spikes_tensor = torch.stack(hidden_spikes, dim=1)
        
        return {
            'output': output,
            'hidden_spikes': hidden_spikes_tensor
        }
    
    def get_heterogeneity_analysis(self) -> Dict[str, any]:
        """获取网络异质性分析"""
        hetero_info = self.hetero_lif.get_heterogeneity_info()
        
        analysis = {
            'n_neurons': self.hidden_size,
            'tau_stats': {
                'min': hetero_info['tau_m'].min().item(),
                'max': hetero_info['tau_m'].max().item(),
                'mean': hetero_info['tau_m'].mean().item(),
                'std': hetero_info['tau_m'].std().item()
            },
            'threshold_stats': {
                'min': hetero_info['v_th'].min().item(),
                'max': hetero_info['v_th'].max().item(),
                'mean': hetero_info['v_th'].mean().item(),
                'std': hetero_info['v_th'].std().item()
            },
            'adaptation_ratio': hetero_info['adaptation_mask'].float().mean().item()
        }
        
        return analysis


# 实用函数
def create_soma_heterogeneous_model(config: Dict) -> SomaHeterogeneousSNN:
    """
    根据配置创建胞体异质性模型
    """
    return SomaHeterogeneousSNN(
        input_size=config.get('input_size', 2),
        hidden_size=config.get('hidden_size', 128),
        output_size=config.get('output_size', 2),
        tau_range=config.get('tau_range', (2.0, 20.0)),
        v_th_range=config.get('v_th_range', (0.8, 1.2)),
        adaptation_ratio=config.get('adaptation_ratio', 0.3)
    )


def test_soma_heterogeneity_integration():
    """测试胞体异质性集成"""
    print("🧠 测试胞体异质性集成...")
    
    # 测试异质性神经元
    print("1. 测试SomaHeterogeneousLIF...")
    lif = SomaHeterogeneousLIF(n_neurons=10)
    x = torch.randn(5, 10)
    output = lif(x)
    hetero_info = lif.get_heterogeneity_info()
    print(f"   ✅ 异质性神经元测试通过")
    print(f"   膜时间常数范围: {hetero_info['tau_m'].min():.2f} - {hetero_info['tau_m'].max():.2f}")
    
    # 测试完整网络
    print("2. 测试SomaHeterogeneousSNN...")
    model = SomaHeterogeneousSNN(hidden_size=64)
    x = torch.randn(4, 50, 2)  # (batch=4, time=50, input=2)
    result = model(x)
    analysis = model.get_heterogeneity_analysis()
    print(f"   ✅ 网络测试通过")
    print(f"   输出形状: {result['output'].shape}")
    print(f"   异质性分析: {analysis['n_neurons']}个神经元")
    
    print("✅ 胞体异质性集成测试全部通过!")
    return True


if __name__ == "__main__":
    test_soma_heterogeneity_integration()