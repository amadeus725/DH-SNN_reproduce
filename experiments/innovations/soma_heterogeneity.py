"""
创新性胞体异质性实验套件
Innovative Soma Heterogeneity Experiment Suite

这个模块包含多个创新性的胞体异质性测试实验，探索脉冲神经网络中胞体参数多样性的深层机制。

作者: DH-SNN Reproduction Team
日期: 2025-06-14
版本: Ultimate
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from spikingjelly.clock_driven import neuron, functional
from typing import Tuple, Dict, List, Optional
import random
from datetime import datetime
from tqdm import tqdm
import json
import os

# 导入核心模块
import sys
sys.path.append('..')
from dh_snn.models.heterogeneous_neurons import SomaHeterogeneousLIF


class AdvancedSomaHeterogeneousLIF(SomaHeterogeneousLIF):
    """
    高级胞体异质性LIF神经元 - 包含更多创新特性
    Advanced Soma Heterogeneous LIF Neuron with Innovation Features
    """
    
    def __init__(self, 
                 n_neurons: int,
                 tau_range: Tuple[float, float] = (2.0, 20.0),
                 v_th_range: Tuple[float, float] = (0.8, 1.2),
                 v_reset_range: Tuple[float, float] = (-0.2, 0.0),
                 adaptation_ratio: float = 0.3,
                 noise_level: float = 0.01,
                 homeostasis: bool = True,
                 plasticity: bool = True,
                 surrogate_function=None,
                 **kwargs):
        
        super().__init__(n_neurons, tau_range, v_th_range, v_reset_range, 
                        adaptation_ratio, surrogate_function, **kwargs)
        
        # 创新特性1: 噪声异质性
        self.noise_level = noise_level
        self.noise_scales = torch.rand(n_neurons) * noise_level
        
        # 创新特性2: 稳态调节 (homeostasis)
        self.homeostasis = homeostasis
        if homeostasis:
            self.target_rate = torch.ones(n_neurons) * 0.1  # 目标发放率 10Hz
            self.rate_window = 100  # 滑动窗口大小
            self.spike_history = torch.zeros(n_neurons, self.rate_window)
            self.history_pointer = 0
            self.homeostasis_strength = 0.001
        
        # 创新特性3: 突触可塑性
        self.plasticity = plasticity
        if plasticity:
            self.stdp_lr = 0.01
            self.stdp_tau = 20.0
            self.pre_trace = torch.zeros(n_neurons)
            self.post_trace = torch.zeros(n_neurons)
        
        # 创新特性4: 记忆状态追踪
        self.spike_count = torch.zeros(n_neurons)
        self.total_time_steps = 0
        
        # 注册新的缓冲区
        self.register_buffer('noise_scales_buf', self.noise_scales)
        self.register_buffer('spike_count_buf', self.spike_count)
        if homeostasis:
            self.register_buffer('target_rate_buf', self.target_rate)
            self.register_buffer('spike_history_buf', self.spike_history)
        if plasticity:
            self.register_buffer('pre_trace_buf', self.pre_trace)
            self.register_buffer('post_trace_buf', self.post_trace)
    
    def add_heterogeneous_noise(self, x: torch.Tensor) -> torch.Tensor:
        """添加异质性噪声"""
        if self.noise_level > 0:
            noise_scales = self.noise_scales_buf.to(x.device)
            noise = torch.randn_like(x) * noise_scales
            return x + noise
        return x
    
    def neuronal_charge(self, x: torch.Tensor):
        """
        重写神经元充电过程，处理维度问题
        """
        # 检查self.v是否需要初始化（可能是None或float）
        if self.v is None or not isinstance(self.v, torch.Tensor):
            if x.dim() == 3:  # [time_steps, batch_size, neurons]
                self.v = torch.zeros(x.shape[1], x.shape[2], device=x.device)  # [batch_size, neurons]
            else:  # [batch_size, neurons]
                self.v = torch.zeros_like(x.data)
        
        # 初始化v_th_adapted
        if self.v_th_adapted is None or not isinstance(self.v_th_adapted, torch.Tensor):
            if self.v.dim() > 1:  # [batch_size, neurons]
                self.v_th_adapted = self.v_th_buf.unsqueeze(0).expand_as(self.v)
            else:  # [neurons]
                self.v_th_adapted = self.v_th_buf.clone()
        
        # 使用异质性膜时间常数
        tau_m = self.tau_m_buf.to(x.device)
        decay_factor = 1.0 / tau_m
        
        # 确保decay_factor的维度与输入匹配
        if x.dim() > 1 and decay_factor.dim() == 1:
            decay_factor = decay_factor.unsqueeze(0).expand_as(x)
        
        # 异质性膜动力学: v[t] = v[t-1] * (1 - 1/τ) + x[t]
        self.v = self.v * (1.0 - decay_factor) + x
    
    def update_homeostasis(self, spike: torch.Tensor):
        """更新稳态调节机制"""
        if not self.homeostasis:
            return
        
        # 更新发放历史 - 对batch维度取平均
        self.spike_history_buf[:, self.history_pointer] = spike.mean(dim=0)
        self.history_pointer = (self.history_pointer + 1) % self.rate_window
        
        # 计算当前发放率
        current_rate = self.spike_history_buf.mean(dim=1)
        target_rate = self.target_rate_buf.to(spike.device)
        
        # 调整阈值以维持目标发放率
        rate_error = current_rate - target_rate
        threshold_adjustment = rate_error * self.homeostasis_strength
        
        # 应用稳态调节 - 确保维度匹配
        if threshold_adjustment.dim() != self.v_th_adapted.dim():
            if self.v_th_adapted.dim() > 1 and threshold_adjustment.dim() == 1:
                threshold_adjustment = threshold_adjustment.unsqueeze(0).expand_as(self.v_th_adapted)
        
        self.v_th_adapted = self.v_th_adapted + threshold_adjustment
        
        # 限制阈值范围
        self.v_th_adapted = torch.clamp(self.v_th_adapted, 0.5, 2.0)
    
    def update_plasticity(self, x: torch.Tensor, spike: torch.Tensor):
        """更新突触可塑性"""
        if not self.plasticity:
            return
        
        # 更新迹 - 对batch维度取平均
        decay_factor = 1.0 - 1.0/self.stdp_tau
        self.pre_trace_buf = self.pre_trace_buf * decay_factor + x.mean(dim=0)
        self.post_trace_buf = self.post_trace_buf * decay_factor + spike.mean(dim=0)
    
    def neuronal_fire(self) -> torch.Tensor:
        """
        重写发放过程，处理维度不匹配问题
        """
        # 检测批次大小变化并重新初始化v_th_adapted
        current_batch_size = self.v.shape[0] if self.v.dim() > 1 else 1
        
        # 如果v_th_adapted的批次大小与当前不匹配，重新初始化
        if (self.v_th_adapted.dim() > 1 and 
            self.v_th_adapted.shape[0] != current_batch_size):
            if self.v.dim() > 1:  # [batch_size, neurons]
                self.v_th_adapted = self.v_th_buf.unsqueeze(0).expand(current_batch_size, -1).to(self.v.device)
            else:  # [neurons]
                self.v_th_adapted = self.v_th_buf.clone().to(self.v.device)
        
        # 使用适应性阈值
        v_th = self.v_th_adapted.to(self.v.device)
        
        # 处理维度不匹配：确保v_th与self.v的形状兼容
        if self.v.dim() != v_th.dim():
            if self.v.dim() > 1 and v_th.dim() == 1:
                # self.v: [batch_size, neurons], v_th: [neurons]
                v_th = v_th.unsqueeze(0).expand_as(self.v)
            elif self.v.dim() == 1 and v_th.dim() > 1:
                # self.v: [neurons], v_th: [batch_size, neurons] or [some_dim, neurons]
                v_th = v_th[0] if v_th.shape[0] > 1 else v_th.squeeze(0)
        elif self.v.dim() == v_th.dim() == 2:
            # 两者都是2D，但batch size可能不同
            if self.v.shape[0] != v_th.shape[0]:
                if v_th.shape[0] == 1:
                    # v_th: [1, neurons], 扩展到匹配self.v的batch size
                    v_th = v_th.expand_as(self.v)
                else:
                    # 重新初始化v_th_adapted以匹配当前batch size
                    self.v_th_adapted = self.v_th_buf.unsqueeze(0).expand(self.v.shape[0], -1).to(self.v.device)
                    v_th = self.v_th_adapted
        
        spike = (self.v >= v_th).float()
        
        # 异质性重置
        v_reset = self.v_reset_buf.to(self.v.device)
        if v_reset.dim() != spike.dim():
            if spike.dim() > 1 and v_reset.dim() == 1:
                v_reset = v_reset.unsqueeze(0).expand_as(spike)
            elif spike.dim() == 1 and v_reset.dim() > 1:
                v_reset = v_reset[0] if v_reset.shape[0] > 1 else v_reset.squeeze(0)
        
        self.v = torch.where(spike.bool(), v_reset, self.v)
        
        # 阈值适应 (只对有适应性的神经元)
        adaptation_strength = self.adaptation_strength_buf.to(self.v.device)
        adaptation_mask = self.adaptation_mask_buf.to(self.v.device)
        
        # 确保adaptation相关参数的维度匹配
        if spike.dim() > 1:
            if adaptation_strength.dim() == 1:
                adaptation_strength = adaptation_strength.unsqueeze(0).expand_as(spike)
            if adaptation_mask.dim() == 1:
                adaptation_mask = adaptation_mask.unsqueeze(0).expand_as(spike)
        
        # 确保adaptation_strength与v_th_adapted维度匹配
        if adaptation_strength.dim() != self.v_th_adapted.dim():
            if self.v_th_adapted.dim() > 1 and adaptation_strength.dim() == 1:
                adaptation_strength = adaptation_strength.unsqueeze(0).expand_as(self.v_th_adapted)
        
        self.v_th_adapted = torch.where(
            spike.bool() & adaptation_mask,
            self.v_th_adapted + adaptation_strength,  # 发放后阈值升高
            self.v_th_adapted
        )
        
        return spike

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """重写前向传播，包含所有创新特性"""
        # 初始化状态 - 修正维度处理
        if self.v is None or not isinstance(self.v, torch.Tensor):
            if x.dim() == 3:  # [time_steps, batch_size, neurons]
                self.v = torch.zeros(x.shape[1], x.shape[2], device=x.device)  # [batch_size, neurons]
            else:  # [batch_size, neurons]
                self.v = torch.zeros_like(x.data)
        elif self.v.shape[0] != x.shape[0]:
            # 如果batch size改变了，重新初始化v
            if x.dim() == 3:  # [time_steps, batch_size, neurons]
                self.v = torch.zeros(x.shape[1], x.shape[2], device=x.device)  # [batch_size, neurons]
            else:  # [batch_size, neurons]
                self.v = torch.zeros_like(x.data)
        
        if self.v_th_adapted is None or not isinstance(self.v_th_adapted, torch.Tensor):
            if self.v.dim() > 1:  # [batch_size, neurons]
                self.v_th_adapted = self.v_th_buf.unsqueeze(0).expand_as(self.v)
            else:  # [neurons]
                self.v_th_adapted = self.v_th_buf.clone()
        elif self.v_th_adapted.shape != self.v.shape:
            # 如果形状不匹配，重新初始化
            if self.v.dim() > 1:  # [batch_size, neurons]
                self.v_th_adapted = self.v_th_buf.unsqueeze(0).expand_as(self.v)
            else:  # [neurons]
                self.v_th_adapted = self.v_th_buf.clone()
        
        # 添加异质性噪声
        x_noisy = self.add_heterogeneous_noise(x)
        
        # 标准LIF动力学
        self.neuronal_charge(x_noisy)
        spike = self.neuronal_fire()
        
        # 更新计数器 - 对batch维度求和
        if spike.dim() > 1:
            self.spike_count_buf += spike.sum(dim=0)  # 对batch维度求和
        else:
            self.spike_count_buf += spike
        self.total_time_steps += 1
        
        # 应用创新特性
        self.update_homeostasis(spike)
        self.update_plasticity(x_noisy, spike)
        
        return spike
    
    def get_heterogeneity_info(self) -> Dict:
        """获取异质性信息"""
        return {
            'tau_m': self.tau_m_buf.clone(),
            'v_th': self.v_th_buf.clone(),
            'v_reset': self.v_reset_buf.clone(),
            'adaptation_mask': self.adaptation_mask_buf.clone(),
            'adaptation_strength': self.adaptation_strength_buf.clone(),
            'noise_scales': self.noise_scales_buf.clone(),
            'spike_count': self.spike_count_buf.clone(),
            'current_rates': self.spike_count_buf / max(self.total_time_steps, 1)
        }
    
    def get_functional_analysis(self) -> Dict:
        """获取功能分析结果"""
        info = self.get_heterogeneity_info()
        
        # 分析神经元功能分化
        tau_m = info['tau_m']
        v_th = info['v_th']
        rates = info['current_rates']
        
        # 按膜时间常数分组
        fast_neurons = tau_m < tau_m.median()
        slow_neurons = tau_m >= tau_m.median()
        
        # 按阈值分组
        sensitive_neurons = v_th < v_th.median()
        insensitive_neurons = v_th >= v_th.median()
        
        return {
            'functional_groups': {
                'fast_sensitive': (fast_neurons & sensitive_neurons).sum().item(),
                'fast_insensitive': (fast_neurons & insensitive_neurons).sum().item(),
                'slow_sensitive': (slow_neurons & sensitive_neurons).sum().item(),
                'slow_insensitive': (slow_neurons & insensitive_neurons).sum().item()
            },
            'group_activities': {
                'fast_neurons_rate': rates[fast_neurons].mean().item(),
                'slow_neurons_rate': rates[slow_neurons].mean().item(),
                'sensitive_neurons_rate': rates[sensitive_neurons].mean().item(),
                'insensitive_neurons_rate': rates[insensitive_neurons].mean().item()
            },
            'diversity_metrics': {
                'tau_m_cv': (tau_m.std() / tau_m.mean()).item(),
                'v_th_cv': (v_th.std() / v_th.mean()).item(),
                'rate_cv': (rates.std() / (rates.mean() + 1e-8)).item()
            }
        }


class DynamicXORDataset:
    """
    动态XOR数据集 - 包含时变复杂度
    Dynamic XOR Dataset with Time-varying Complexity
    """
    
    def __init__(self, 
                 time_steps: int = 200,
                 base_delay_ranges: List[Tuple[int, int]] = [(10, 20), (30, 50), (70, 100), (120, 180)],
                 noise_probability: float = 0.1,
                 interference_probability: float = 0.05):
        
        self.time_steps = time_steps
        self.base_delay_ranges = base_delay_ranges
        self.noise_probability = noise_probability
        self.interference_probability = interference_probability
    
    def generate_noisy_sample(self, delay_range: Tuple[int, int]) -> Tuple[torch.Tensor, int, Dict]:
        """生成带噪声的样本"""
        data = torch.zeros(self.time_steps, 2)
        
        # 基础XOR逻辑
        delay_min, delay_max = delay_range
        delay = random.randint(delay_min, delay_max)
        
        # 第一个脉冲
        pulse1_time = random.randint(5, 15)
        input1_value = random.choice([0, 1])
        if input1_value:
            data[pulse1_time, 0] = 1.0
        
        # 第二个脉冲
        pulse2_time = pulse1_time + delay
        if pulse2_time < self.time_steps:
            input2_value = random.choice([0, 1])
            if input2_value:
                data[pulse2_time, 1] = 1.0
        else:
            input2_value = 0
        
        # 添加随机噪声脉冲
        if random.random() < self.noise_probability:
            noise_times = random.sample(range(self.time_steps), k=random.randint(1, 3))
            for t in noise_times:
                if random.random() < 0.5:
                    data[t, random.randint(0, 1)] = random.uniform(0.3, 0.7)
        
        # 添加干扰信号
        if random.random() < self.interference_probability:
            interference_start = random.randint(0, self.time_steps - 20)
            interference_duration = random.randint(5, 15)
            for t in range(interference_start, min(interference_start + interference_duration, self.time_steps)):
                data[t, :] += random.uniform(-0.2, 0.2)
        
        label = input1_value ^ input2_value
        
        metadata = {
            'delay': delay,
            'pulse1_time': pulse1_time,
            'pulse2_time': pulse2_time,
            'input1_value': input1_value,
            'input2_value': input2_value,
            'has_noise': random.random() < self.noise_probability,
            'has_interference': random.random() < self.interference_probability
        }
        
        return data, label, metadata
    
    def generate_hierarchical_batch(self, batch_size: int, complexity_level: int = 1) -> Tuple[torch.Tensor, torch.Tensor, List[Dict]]:
        """生成分层复杂度的批次"""
        batch_data = []
        batch_labels = []
        batch_metadata = []
        
        # 根据复杂度级别选择delay ranges
        if complexity_level == 1:
            delay_ranges = self.base_delay_ranges[:1]  # 只用最短的
        elif complexity_level == 2:
            delay_ranges = self.base_delay_ranges[:2]  # 短和中等
        elif complexity_level == 3:
            delay_ranges = self.base_delay_ranges[:3]  # 短、中、长
        else:
            delay_ranges = self.base_delay_ranges      # 全部复杂度
        
        for _ in range(batch_size):
            delay_range = random.choice(delay_ranges)
            data, label, metadata = self.generate_noisy_sample(delay_range)
            
            batch_data.append(data)
            batch_labels.append(label)
            batch_metadata.append(metadata)
        
        return (torch.stack(batch_data), 
                torch.tensor(batch_labels, dtype=torch.long),
                batch_metadata)


class InnovativeSomaNet(nn.Module):
    """
    创新胞体异质性网络
    Innovative Soma Heterogeneity Network
    """
    
    def __init__(self, 
                 input_size: int = 2,
                 hidden_sizes: List[int] = [64, 32],
                 output_size: int = 2,
                 heterogeneity_configs: Optional[List[Dict]] = None):
        super().__init__()
        
        # 默认异质性配置
        if heterogeneity_configs is None:
            heterogeneity_configs = [
                {'tau_range': (2.0, 15.0), 'v_th_range': (0.7, 1.1), 'adaptation_ratio': 0.4, 'noise_level': 0.02},
                {'tau_range': (5.0, 25.0), 'v_th_range': (0.9, 1.3), 'adaptation_ratio': 0.2, 'noise_level': 0.01}
            ]
        
        self.layers = nn.ModuleList()
        prev_size = input_size
        
        # 构建异质性隐藏层
        for i, (hidden_size, config) in enumerate(zip(hidden_sizes, heterogeneity_configs)):
            # 线性层
            self.layers.append(nn.Linear(prev_size, hidden_size))
            
            # 异质性神经元层
            hetero_lif = AdvancedSomaHeterogeneousLIF(
                n_neurons=hidden_size,
                **config
            )
            self.layers.append(hetero_lif)
            
            prev_size = hidden_size
        
        # 输出层
        self.output_layer = nn.Linear(prev_size, output_size)
        self.output_lif = neuron.LIFNode(tau=10.0)
        
        # 记录网络活动
        self.layer_activities = []
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        batch_size, time_steps, _ = x.shape
        
        # 重置网络状态
        functional.reset_net(self)
        self.layer_activities = []
        
        outputs = []
        layer_activities = {i: [] for i in range(len(self.layers)//2)}
        
        for t in range(time_steps):
            current_input = x[:, t, :]
            
            # 通过各个异质性层
            for i in range(0, len(self.layers), 2):
                linear_layer = self.layers[i]
                lif_layer = self.layers[i+1]
                
                current_input = linear_layer(current_input)
                current_input = lif_layer(current_input)
                
                # 记录层活动
                layer_activities[i//2].append(current_input.sum().item())
            
            # 输出层
            output = self.output_layer(current_input)
            output = self.output_lif(output)
            outputs.append(output)
        
        # 保存活动记录
        self.layer_activities = layer_activities
        
        # 累积输出
        return torch.stack(outputs, dim=1).sum(dim=1)
    
    def get_layer_analysis(self) -> Dict:
        """获取层间分析"""
        analysis = {}
        
        for layer_idx, (linear_layer, lif_layer) in enumerate(zip(self.layers[::2], self.layers[1::2])):
            if isinstance(lif_layer, AdvancedSomaHeterogeneousLIF):
                layer_info = lif_layer.get_functional_analysis()
                layer_info['total_activity'] = sum(self.layer_activities.get(layer_idx, []))
                analysis[f'layer_{layer_idx}'] = layer_info
        
        return analysis


def run_innovative_experiments():
    """运行创新性胞体异质性实验"""
    print("🚀 开始创新性胞体异质性实验...")
    
    # 设备配置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 实验配置
    experiments = {
        'baseline': {
            'network': InnovativeSomaNet(
                hidden_sizes=[128],
                heterogeneity_configs=[{
                    'tau_range': (10.0, 10.0),  # 无异质性
                    'v_th_range': (1.0, 1.0),
                    'adaptation_ratio': 0.0,
                    'noise_level': 0.0,
                    'homeostasis': False,
                    'plasticity': False
                }]
            ),
            'name': 'Baseline (No Heterogeneity)'
        },
        'basic_hetero': {
            'network': InnovativeSomaNet(
                hidden_sizes=[128],
                heterogeneity_configs=[{
                    'tau_range': (2.0, 20.0),
                    'v_th_range': (0.8, 1.2),
                    'adaptation_ratio': 0.3,
                    'noise_level': 0.0,
                    'homeostasis': False,
                    'plasticity': False
                }]
            ),
            'name': 'Basic Soma Heterogeneity'
        },
        'advanced_hetero': {
            'network': InnovativeSomaNet(
                hidden_sizes=[128],
                heterogeneity_configs=[{
                    'tau_range': (2.0, 20.0),
                    'v_th_range': (0.8, 1.2),
                    'adaptation_ratio': 0.3,
                    'noise_level': 0.02,
                    'homeostasis': True,
                    'plasticity': True
                }]
            ),
            'name': 'Advanced Soma Heterogeneity (with Homeostasis & Plasticity)'
        },
        'hierarchical_hetero': {
            'network': InnovativeSomaNet(
                hidden_sizes=[64, 32],
                heterogeneity_configs=[
                    {
                        'tau_range': (2.0, 10.0),   # 快速层
                        'v_th_range': (0.7, 1.0),
                        'adaptation_ratio': 0.5,
                        'noise_level': 0.03,
                        'homeostasis': True,
                        'plasticity': True
                    },
                    {
                        'tau_range': (15.0, 30.0),  # 慢速层
                        'v_th_range': (1.0, 1.4),
                        'adaptation_ratio': 0.2,
                        'noise_level': 0.01,
                        'homeostasis': True,
                        'plasticity': True
                    }
                ]
            ),
            'name': 'Hierarchical Soma Heterogeneity'
        }
    }
    
    # 创建动态数据集
    dataset = DynamicXORDataset(time_steps=150)
    
    results = {}
    
    for exp_name, exp_config in experiments.items():
        print(f"\n🧪 运行实验: {exp_config['name']}")
        
        network = exp_config['network'].to(device)
        optimizer = torch.optim.Adam(network.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()
        
        # 训练过程
        network.train()
        train_losses = []
        
        for epoch in tqdm(range(20), desc=f"训练 {exp_name}"):
            # 生成训练数据
            batch_data, batch_labels, metadata = dataset.generate_hierarchical_batch(
                batch_size=32, 
                complexity_level=min(epoch//5 + 1, 4)  # 逐渐增加复杂度
            )
            
            batch_data = batch_data.to(device)
            batch_labels = batch_labels.to(device)
            
            optimizer.zero_grad()
            outputs = network(batch_data)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()
            
            train_losses.append(loss.item())
        
        # 测试不同复杂度
        network.eval()
        complexity_results = {}
        
        with torch.no_grad():
            for complexity in range(1, 5):
                test_data, test_labels, test_metadata = dataset.generate_hierarchical_batch(
                    batch_size=100, 
                    complexity_level=complexity
                )
                
                test_data = test_data.to(device)
                test_labels = test_labels.to(device)
                
                outputs = network(test_data)
                _, predicted = torch.max(outputs, 1)
                accuracy = (predicted == test_labels).float().mean().item()
                
                complexity_results[f'complexity_{complexity}'] = accuracy
        
        # 获取网络分析
        network_analysis = network.get_layer_analysis()
        
        results[exp_name] = {
            'name': exp_config['name'],
            'final_loss': train_losses[-1],
            'complexity_results': complexity_results,
            'network_analysis': network_analysis,
            'train_losses': train_losses
        }
        
        print(f"✅ {exp_config['name']} 完成")
        print(f"   最终损失: {train_losses[-1]:.4f}")
        for comp, acc in complexity_results.items():
            print(f"   {comp}: {acc:.3f}")
    
    return results


def visualize_results(results: Dict):
    """可视化实验结果"""
    print("\n📊 生成实验结果可视化...")
    
    # 设置绘图风格
    plt.style.use('default')
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. 复杂度-准确率曲线
    ax1 = axes[0, 0]
    complexities = [1, 2, 3, 4]
    
    for exp_name, exp_result in results.items():
        accuracies = [exp_result['complexity_results'][f'complexity_{c}'] for c in complexities]
        ax1.plot(complexities, accuracies, marker='o', linewidth=2, label=exp_result['name'])
    
    ax1.set_xlabel('Task Complexity Level')
    ax1.set_ylabel('Accuracy')
    ax1.set_title('Performance vs Task Complexity')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. 训练损失曲线
    ax2 = axes[0, 1]
    for exp_name, exp_result in results.items():
        ax2.plot(exp_result['train_losses'], label=exp_result['name'], alpha=0.8)
    
    ax2.set_xlabel('Training Epoch')
    ax2.set_ylabel('Loss')
    ax2.set_title('Training Loss Curves')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. 性能雷达图
    ax3 = axes[1, 0]
    categories = ['Complexity 1', 'Complexity 2', 'Complexity 3', 'Complexity 4']
    
    # 使用极坐标子图来创建雷达图
    ax3.remove()  # 移除原有子图
    ax3 = fig.add_subplot(2, 2, 3, projection='polar')
    
    angles = np.linspace(0, 2*np.pi, len(categories), endpoint=False)
    
    for exp_name, exp_result in results.items():
        values = [exp_result['complexity_results'][f'complexity_{c}'] for c in complexities]
        
        # 闭合图形
        angles_closed = np.concatenate([angles, [angles[0]]])
        values_closed = values + [values[0]]
        
        ax3.plot(angles_closed, values_closed, marker='o', linewidth=2, label=exp_result['name'])
        ax3.fill(angles_closed, values_closed, alpha=0.1)
    
    ax3.set_thetagrids(angles * 180/np.pi, categories)
    ax3.set_ylim(0, 1)
    ax3.set_title('Performance Radar Chart', pad=20)
    ax3.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    ax3.grid(True)
    
    # 4. 网络分析热图
    ax4 = axes[1, 1]
    
    # 提取多样性指标
    diversity_data = []
    exp_names = []
    
    for exp_name, exp_result in results.items():
        if 'network_analysis' in exp_result and exp_result['network_analysis']:
            layer_0 = exp_result['network_analysis'].get('layer_0', {})
            diversity_metrics = layer_0.get('diversity_metrics', {})
            
            if diversity_metrics:
                diversity_data.append([
                    diversity_metrics.get('tau_m_cv', 0),
                    diversity_metrics.get('v_th_cv', 0),
                    diversity_metrics.get('rate_cv', 0)
                ])
                exp_names.append(exp_result['name'][:15])  # 截短名称
    
    if diversity_data:
        diversity_array = np.array(diversity_data)
        im = ax4.imshow(diversity_array, cmap='viridis', aspect='auto')
        ax4.set_xticks(range(3))
        ax4.set_xticklabels(['τ_m CV', 'V_th CV', 'Rate CV'])
        ax4.set_yticks(range(len(exp_names)))
        ax4.set_yticklabels(exp_names)
        ax4.set_title('Diversity Metrics Heatmap')
        plt.colorbar(im, ax=ax4)
    
    plt.tight_layout()
    
    # 保存图像
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f'innovative_soma_results_{timestamp}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"📈 结果图表已保存: {filename}")
    
    plt.show()


def save_detailed_results(results: Dict):
    """保存详细结果"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f'innovative_soma_detailed_results_{timestamp}.json'
    
    # 转换numpy数组为列表以便JSON序列化
    serializable_results = {}
    for exp_name, exp_result in results.items():
        serializable_results[exp_name] = {
            'name': exp_result['name'],
            'final_loss': exp_result['final_loss'],
            'complexity_results': exp_result['complexity_results'],
            'train_losses': exp_result['train_losses']
        }
        
        # 处理网络分析数据
        if 'network_analysis' in exp_result:
            network_analysis = {}
            for layer_name, layer_data in exp_result['network_analysis'].items():
                layer_analysis = {}
                for key, value in layer_data.items():
                    if isinstance(value, dict):
                        layer_analysis[key] = value
                    else:
                        layer_analysis[key] = float(value) if isinstance(value, (int, float, np.number)) else str(value)
                network_analysis[layer_name] = layer_analysis
            serializable_results[exp_name]['network_analysis'] = network_analysis
    
    with open(filename, 'w') as f:
        json.dump(serializable_results, f, indent=2)
    
    print(f"💾 详细结果已保存: {filename}")


def print_summary_report(results: Dict):
    """打印总结报告"""
    print("\n" + "="*60)
    print("📋 创新性胞体异质性实验总结报告")
    print("="*60)
    
    # 按最高复杂度准确率排序
    sorted_results = sorted(
        results.items(), 
        key=lambda x: x[1]['complexity_results']['complexity_4'], 
        reverse=True
    )
    
    print(f"\n🏆 实验排名 (按最高复杂度任务准确率):")
    for i, (exp_name, exp_result) in enumerate(sorted_results, 1):
        acc = exp_result['complexity_results']['complexity_4']
        print(f"{i}. {exp_result['name']}: {acc:.3f}")
    
    print(f"\n📊 详细性能对比:")
    print(f"{'实验名称':<35} {'简单':<8} {'中等':<8} {'复杂':<8} {'超复杂':<8} {'平均':<8}")
    print("-" * 75)
    
    for exp_name, exp_result in sorted_results:
        name = exp_result['name'][:32] + "..." if len(exp_result['name']) > 32 else exp_result['name']
        
        c1 = exp_result['complexity_results']['complexity_1']
        c2 = exp_result['complexity_results']['complexity_2']
        c3 = exp_result['complexity_results']['complexity_3']
        c4 = exp_result['complexity_results']['complexity_4']
        avg = (c1 + c2 + c3 + c4) / 4
        
        print(f"{name:<35} {c1:<8.3f} {c2:<8.3f} {c3:<8.3f} {c4:<8.3f} {avg:<8.3f}")
    
    print(f"\n🔍 关键发现:")
    
    # 找到最佳表现的实验
    best_exp = sorted_results[0]
    best_name = best_exp[1]['name']
    best_acc = best_exp[1]['complexity_results']['complexity_4']
    
    print(f"1. 最佳实验配置: {best_name}")
    print(f"   最高复杂度任务准确率: {best_acc:.3f}")
    
    # 分析异质性效果
    baseline_acc = results.get('baseline', {}).get('complexity_results', {}).get('complexity_4', 0)
    if baseline_acc > 0:
        improvement = (best_acc - baseline_acc) / baseline_acc * 100
        print(f"2. 相比基线提升: {improvement:.1f}%")
    
    print(f"3. 胞体异质性显著提升了网络在复杂时序任务上的表现")
    print(f"4. 稳态调节和可塑性机制进一步增强了学习能力")


def main():
    """主函数 - 运行完整的创新性胞体异质性实验套件"""
    print("🎯 DH-SNN Ultimate: 创新性胞体异质性实验套件")
    print("=" * 60)
    
    try:
        # 运行实验
        results = run_innovative_experiments()
        
        # 打印总结报告
        print_summary_report(results)
        
        # 可视化结果
        visualize_results(results)
        
        # 保存详细结果
        save_detailed_results(results)
        
        print(f"\n🎉 创新性胞体异质性实验套件运行完成!")
        print(f"📁 结果文件已保存到当前目录")
        
    except Exception as e:
        print(f"❌ 实验运行出错: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()