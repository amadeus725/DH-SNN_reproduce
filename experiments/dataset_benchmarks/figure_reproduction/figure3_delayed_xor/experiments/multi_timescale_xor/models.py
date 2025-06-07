#!/usr/bin/env python3
"""
多时间尺度XOR模型
实现Figure 4中的双分支和多分支DH-SFNN模型
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from typing import Optional, Tuple, List

# 导入核心组件
import sys
import os
# sys.path.append removed during restructure

from dh_snn.core.surrogate import MultiGaussianSurrogate
from dh_snn.core.neurons import DH_LIFNode, ParametricLIFNode, ReadoutNeuron

class TwoBranchDH_SFNN(nn.Module):
    """
    双分支DH-SFNN模型
    专门用于多时间尺度XOR任务

    按照Figure 4的设计：
    - Branch 1: 处理低频Signal 1，使用大时间常数
    - Branch 2: 处理高频Signal 2，使用小时间常数
    """

    def __init__(self,
                 input_size: int = 100,
                 hidden_size: int = 64,
                 output_size: int = 1,
                 tau_m_range: Tuple[float, float] = (0.0, 4.0),
                 tau_n_branch1_range: Tuple[float, float] = (2.0, 6.0),  # 大时间常数
                 tau_n_branch2_range: Tuple[float, float] = (-4.0, 0.0), # 小时间常数
                 v_threshold: float = 1.0,
                 beneficial_init: bool = True,
                 learnable_timing: bool = True,
                 device: str = 'cpu'):
        """
        初始化双分支DH-SFNN

        Args:
            input_size: 输入维度
            hidden_size: 隐藏层大小
            output_size: 输出维度
            tau_m_range: 膜时间常数范围
            tau_n_branch1_range: 分支1树突时间常数范围
            tau_n_branch2_range: 分支2树突时间常数范围
            v_threshold: 脉冲阈值
            beneficial_init: 是否使用有益初始化
            learnable_timing: 时间常数是否可学习
            device: 设备
        """
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.v_threshold = v_threshold
        self.beneficial_init = beneficial_init
        self.learnable_timing = learnable_timing
        self.device = device

        # 创建双分支DH-LIF层
        self.dh_layer = self._create_two_branch_layer(
            input_size, hidden_size,
            tau_m_range, tau_n_branch1_range, tau_n_branch2_range
        )

        # 读出层
        self.readout = nn.Linear(hidden_size, output_size)

        # 初始化权重
        self._initialize_weights()

    def _create_two_branch_layer(self,
                                input_size: int,
                                output_size: int,
                                tau_m_range: Tuple[float, float],
                                tau_n_branch1_range: Tuple[float, float],
                                tau_n_branch2_range: Tuple[float, float]) -> nn.Module:
        """创建双分支DH-LIF层"""

        layer = nn.Module()

        # 连接权重 - 分别连接到两个分支
        layer.dense_branch1 = nn.Linear(input_size, output_size)
        layer.dense_branch2 = nn.Linear(input_size, output_size)

        # 时间常数参数
        layer.tau_m = nn.Parameter(torch.empty(output_size))
        layer.tau_n_branch1 = nn.Parameter(torch.empty(output_size))
        layer.tau_n_branch2 = nn.Parameter(torch.empty(output_size))

        # 初始化时间常数
        nn.init.uniform_(layer.tau_m, tau_m_range[0], tau_m_range[1])

        if self.beneficial_init:
            # 有益初始化：Branch 1大时间常数，Branch 2小时间常数
            nn.init.uniform_(layer.tau_n_branch1, tau_n_branch1_range[0], tau_n_branch1_range[1])
            nn.init.uniform_(layer.tau_n_branch2, tau_n_branch2_range[0], tau_n_branch2_range[1])
        else:
            # 随机初始化
            nn.init.uniform_(layer.tau_n_branch1, -2.0, 2.0)
            nn.init.uniform_(layer.tau_n_branch2, -2.0, 2.0)

        # 设置是否可学习
        layer.tau_m.requires_grad = self.learnable_timing
        layer.tau_n_branch1.requires_grad = self.learnable_timing
        layer.tau_n_branch2.requires_grad = self.learnable_timing

        # 神经元状态
        layer.register_buffer('mem', torch.zeros(1, output_size))
        layer.register_buffer('spike', torch.zeros(1, output_size))
        layer.register_buffer('d_input_branch1', torch.zeros(1, output_size))
        layer.register_buffer('d_input_branch2', torch.zeros(1, output_size))

        def set_neuron_state(batch_size):
            layer.mem = torch.rand(batch_size, output_size).to(self.device)
            layer.spike = torch.rand(batch_size, output_size).to(self.device)
            layer.d_input_branch1 = torch.zeros(batch_size, output_size).to(self.device)
            layer.d_input_branch2 = torch.zeros(batch_size, output_size).to(self.device)

        def forward(branch1_input, branch2_input):
            # 分支1处理（长时间记忆）
            beta1 = torch.sigmoid(layer.tau_n_branch1)
            d1_input = layer.dense_branch1(branch1_input.float())
            layer.d_input_branch1 = beta1 * layer.d_input_branch1 + (1 - beta1) * d1_input

            # 分支2处理（快速响应）
            beta2 = torch.sigmoid(layer.tau_n_branch2)
            d2_input = layer.dense_branch2(branch2_input.float())
            layer.d_input_branch2 = beta2 * layer.d_input_branch2 + (1 - beta2) * d2_input

            # 整合两个分支的输入
            total_input = layer.d_input_branch1 + layer.d_input_branch2

            # 膜电位更新
            alpha = torch.sigmoid(layer.tau_m)
            layer.mem = layer.mem * alpha + (1 - alpha) * total_input - self.v_threshold * layer.spike

            # 脉冲生成
            inputs_ = layer.mem - self.v_threshold
            layer.spike = MultiGaussianSurrogate.apply(inputs_)

            return layer.mem, layer.spike, layer.d_input_branch1, layer.d_input_branch2

        layer.set_neuron_state = set_neuron_state
        layer.forward = forward

        return layer

    def _initialize_weights(self):
        """初始化权重"""
        # Xavier初始化
        nn.init.xavier_normal_(self.dh_layer.dense_branch1.weight)
        nn.init.xavier_normal_(self.dh_layer.dense_branch2.weight)
        nn.init.constant_(self.dh_layer.dense_branch1.bias, 0)
        nn.init.constant_(self.dh_layer.dense_branch2.bias, 0)

        nn.init.xavier_normal_(self.readout.dense.weight)
        nn.init.constant_(self.readout.dense.bias, 0)

    def forward(self,
                input_data: torch.Tensor,
                branch1_data: Optional[torch.Tensor] = None,
                branch2_data: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, dict]:
        """
        前向传播

        Args:
            input_data: 输入数据 [batch_size, seq_len, input_size]
            branch1_data: 分支1数据（可选）
            branch2_data: 分支2数据（可选）

        Returns:
            output: 输出 [batch_size, seq_len, output_size]
            states: 内部状态字典
        """
        batch_size, seq_len, _ = input_data.shape

        # 设置神经元状态
        self.dh_layer.set_neuron_state(batch_size)
        self.readout.set_neuron_state(batch_size)

        # 如果没有提供分支数据，使用完整输入
        if branch1_data is None:
            branch1_data = input_data
        if branch2_data is None:
            branch2_data = input_data

        outputs = []
        mem_traces = []
        spike_traces = []
        d1_traces = []
        d2_traces = []

        for t in range(seq_len):
            # 获取当前时间步的输入
            b1_input = branch1_data[:, t, :]
            b2_input = branch2_data[:, t, :]

            # DH层前向传播
            mem, spike, d1, d2 = self.dh_layer.forward(b1_input, b2_input)

            # 读出层
            output = self.readout.forward(spike)

            outputs.append(output)
            mem_traces.append(mem.clone())
            spike_traces.append(spike.clone())
            d1_traces.append(d1.clone())
            d2_traces.append(d2.clone())

        # 堆叠输出
        output_tensor = torch.stack(outputs, dim=1)

        # 收集状态信息
        states = {
            'membrane_potential': torch.stack(mem_traces, dim=1),
            'spikes': torch.stack(spike_traces, dim=1),
            'dendritic_current_branch1': torch.stack(d1_traces, dim=1),
            'dendritic_current_branch2': torch.stack(d2_traces, dim=1),
            'tau_m': self.dh_layer.tau_m.clone(),
            'tau_n_branch1': self.dh_layer.tau_n_branch1.clone(),
            'tau_n_branch2': self.dh_layer.tau_n_branch2.clone()
        }

        return output_tensor, states

class MultiBranchDH_SFNN(nn.Module):
    """
    多分支DH-SFNN模型
    支持任意数量的树突分支
    """

    def __init__(self,
                 input_size: int = 100,
                 hidden_size: int = 64,
                 output_size: int = 1,
                 num_branches: int = 4,
                 tau_m_range: Tuple[float, float] = (0.0, 4.0),
                 tau_n_range: Tuple[float, float] = (2.0, 6.0),
                 v_threshold: float = 1.0,
                 learnable_timing: bool = True,
                 device: str = 'cpu'):
        """
        初始化多分支DH-SFNN

        Args:
            input_size: 输入维度
            hidden_size: 隐藏层大小
            output_size: 输出维度
            num_branches: 分支数量
            tau_m_range: 膜时间常数范围
            tau_n_range: 树突时间常数范围
            v_threshold: 脉冲阈值
            learnable_timing: 时间常数是否可学习
            device: 设备
        """
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_branches = num_branches
        self.v_threshold = v_threshold
        self.learnable_timing = learnable_timing
        self.device = device

        # 创建多分支DH-LIF层
        self.dh_layer = DH_LIFNode(
            input_size, hidden_size,
            tau_m_range, tau_n_range,
            num_branches, v_threshold, device
        )

        # 读出层
        self.readout = ReadoutIntegrator(
            hidden_size, output_size, tau_m_range, device
        )

        # 设置时间常数可学习性
        self.dh_layer.tau_m.requires_grad = learnable_timing
        self.dh_layer.tau_n.requires_grad = learnable_timing

        # 初始化权重
        self._initialize_weights()

    def _initialize_weights(self):
        """初始化权重"""
        nn.init.xavier_normal_(self.dh_layer.dense.weight)
        nn.init.constant_(self.dh_layer.dense.bias, 0)
        nn.init.xavier_normal_(self.readout.dense.weight)
        nn.init.constant_(self.readout.dense.bias, 0)

    def forward(self, input_data: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        """
        前向传播

        Args:
            input_data: 输入数据 [batch_size, seq_len, input_size]

        Returns:
            output: 输出 [batch_size, seq_len, output_size]
            states: 内部状态字典
        """
        batch_size, seq_len, _ = input_data.shape

        # 设置神经元状态
        self.dh_layer.set_neuron_state(batch_size)
        self.readout.set_neuron_state(batch_size)

        outputs = []
        mem_traces = []
        spike_traces = []
        d_input_traces = []

        for t in range(seq_len):
            # 获取当前时间步的输入
            current_input = input_data[:, t, :]

            # 应用连接掩码
            self.dh_layer.apply_mask()

            # DH层前向传播
            mem, spike = self.dh_layer.forward(current_input)

            # 读出层
            output = self.readout.forward(spike)

            outputs.append(output)
            mem_traces.append(mem.clone())
            spike_traces.append(spike.clone())
            d_input_traces.append(self.dh_layer.d_input.clone())

        # 堆叠输出
        output_tensor = torch.stack(outputs, dim=1)

        # 收集状态信息
        states = {
            'membrane_potential': torch.stack(mem_traces, dim=1),
            'spikes': torch.stack(spike_traces, dim=1),
            'dendritic_currents': torch.stack(d_input_traces, dim=1),
            'tau_m': self.dh_layer.tau_m.clone(),
            'tau_n': self.dh_layer.tau_n.clone()
        }

        return output_tensor, states

# 测试代码
if __name__ == '__main__':
    # 测试双分支模型
    model = TwoBranchDH_SFNN(
        input_size=100,
        hidden_size=64,
        output_size=1,
        beneficial_init=True,
        learnable_timing=True
    )

    # 创建测试数据
    batch_size = 10
    seq_len = 1000
    input_size = 100

    input_data = torch.randn(batch_size, seq_len, input_size)
    branch1_data = torch.randn(batch_size, seq_len, input_size)
    branch2_data = torch.randn(batch_size, seq_len, input_size)

    # 前向传播
    output, states = model(input_data, branch1_data, branch2_data)

    print(f"Model output shape: {output.shape}")
    print(f"States keys: {list(states.keys())}")
    print(f"Tau_m range: {states['tau_m'].min():.3f} - {states['tau_m'].max():.3f}")
    print(f"Tau_n_branch1 range: {states['tau_n_branch1'].min():.3f} - {states['tau_n_branch1'].max():.3f}")
    print(f"Tau_n_branch2 range: {states['tau_n_branch2'].min():.3f} - {states['tau_n_branch2'].max():.3f}")
