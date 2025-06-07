#!/usr/bin/env python3
"""
DH-SNN模型定义 - 基于SpikingJelly框架
用于SHD数据集的实验
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
try:
    from spikingjelly.activation_based import neuron, functional, surrogate, layer
    HAS_SPIKINGJELLY = True
except ImportError:
    HAS_SPIKINGJELLY = False
    print("Warning: SpikingJelly not available, using fallback implementations")
from typing import Dict, List, Optional, Tuple
import numpy as np


class MultiGaussianSurrogate(torch.autograd.Function):
    """多高斯替代函数 - 基于论文实现"""

    @staticmethod
    def forward(ctx, input, alpha=2.0):
        ctx.save_for_backward(input)
        ctx.alpha = alpha
        return (input >= 0).float()

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        alpha = ctx.alpha

        # 多高斯替代梯度
        grad_input = alpha * torch.exp(-alpha * input.pow(2)) * grad_output
        return grad_input, None


class DH_LIFNode(nn.Module):
    """树突异质性LIF神经元 - 基于SpikingJelly框架"""

    def __init__(self, input_size: int, hidden_size: int, num_branches: int = 4,
                 tau_m_init: Tuple[float, float] = (0.0, 4.0),
                 tau_n_init: Tuple[float, float] = (2.0, 6.0),
                 v_threshold: float = 1.0, v_reset: float = 0.0,
                 surrogate_function=None, detach_reset: bool = False):
        """
        初始化DH-LIF神经元

        Args:
            input_size: 输入维度
            hidden_size: 隐藏层维度
            num_branches: 树突分支数量
            tau_m_init: 膜电位时间常数初始化范围
            tau_n_init: 树突时间常数初始化范围
            v_threshold: 脉冲阈值
            v_reset: 重置电位
            surrogate_function: 替代函数
            detach_reset: 是否分离重置
        """
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_branches = num_branches
        self.v_threshold = v_threshold
        self.v_reset = v_reset
        self.detach_reset = detach_reset

        # 替代函数
        if surrogate_function is None:
            self.surrogate_function = MultiGaussianSurrogate.apply
        else:
            self.surrogate_function = surrogate_function

        # 线性层
        self.linear = nn.Linear(input_size, hidden_size)

        # 膜电位时间常数 (可学习)
        self.tau_m = nn.Parameter(torch.empty(hidden_size))
        nn.init.uniform_(self.tau_m, tau_m_init[0], tau_m_init[1])

        # 树突时间常数 (可学习)
        self.tau_n = nn.Parameter(torch.empty(hidden_size, num_branches))
        nn.init.uniform_(self.tau_n, tau_n_init[0], tau_n_init[1])

        # 连接掩码 - 实现稀疏连接
        self.register_buffer('connection_mask', self._create_connection_mask())

        # 状态变量
        self.register_memory('v', 0.)  # 膜电位
        self.register_memory('d_current', 0.)  # 树突电流

    def _create_connection_mask(self) -> torch.Tensor:
        """创建连接掩码实现稀疏连接"""
        mask = torch.zeros(self.hidden_size, self.input_size, self.num_branches)

        # 每个神经元随机连接到不同分支
        for i in range(self.hidden_size):
            # 随机选择输入连接到不同分支
            indices = torch.randperm(self.input_size)
            branch_size = self.input_size // self.num_branches

            for b in range(self.num_branches):
                start_idx = b * branch_size
                end_idx = min((b + 1) * branch_size, self.input_size)
                if start_idx < len(indices):
                    selected_indices = indices[start_idx:end_idx]
                    mask[i, selected_indices, b] = 1.0

        return mask

    def register_memory(self, name: str, value):
        """注册状态变量"""
        self.register_buffer(name + '_init', torch.as_tensor(value))

    def reset_memory(self, batch_size: int, device: torch.device):
        """重置状态变量"""
        self.v = torch.zeros(batch_size, self.hidden_size, device=device)
        self.d_current = torch.zeros(batch_size, self.hidden_size, self.num_branches, device=device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        Args:
            x: [batch_size, input_size] 输入脉冲

        Returns:
            spike: [batch_size, hidden_size] 输出脉冲
        """
        batch_size = x.shape[0]
        device = x.device

        # 初始化状态变量
        if not hasattr(self, 'v') or self.v.shape[0] != batch_size:
            self.reset_memory(batch_size, device)

        # 计算输入电流
        input_current = self.linear(x)  # [batch_size, hidden_size]

        # 更新树突电流
        alpha = torch.sigmoid(self.tau_n)  # [hidden_size, num_branches]

        # 计算每个分支的输入
        branch_inputs = torch.zeros(batch_size, self.hidden_size, self.num_branches, device=device)
        for b in range(self.num_branches):
            # 应用连接掩码
            masked_input = x.unsqueeze(1) * self.connection_mask[:, :, b].unsqueeze(0)  # [batch_size, hidden_size, input_size]
            branch_inputs[:, :, b] = masked_input.sum(dim=2)

        # 更新树突电流 (关键：脉冲后不重置)
        self.d_current = self.d_current * alpha.unsqueeze(0) + (1 - alpha.unsqueeze(0)) * branch_inputs

        # 计算总的树突贡献
        dendritic_contribution = self.d_current.sum(dim=2)  # [batch_size, hidden_size]

        # 更新膜电位
        beta = torch.sigmoid(self.tau_m)  # [hidden_size]
        total_current = input_current + 0.5 * dendritic_contribution  # 树突贡献权重

        # 膜电位动态
        self.v = self.v * beta + (1 - beta) * total_current

        # 生成脉冲
        spike = self.surrogate_function(self.v - self.v_threshold)

        # 重置膜电位 (只重置膜电位，树突电流保持)
        if self.detach_reset:
            spike_detach = spike.detach()
        else:
            spike_detach = spike

        self.v = self.v - spike_detach * (self.v - self.v_reset)

        return spike


class ReadoutIntegrator(nn.Module):
    """读出积分器 - 无脉冲的泄漏积分器"""

    def __init__(self, input_size: int, output_size: int,
                 tau_m_init: Tuple[float, float] = (0.0, 4.0)):
        """
        初始化读出积分器

        Args:
            input_size: 输入维度
            output_size: 输出维度
            tau_m_init: 时间常数初始化范围
        """
        super().__init__()

        self.input_size = input_size
        self.output_size = output_size

        # 线性层
        self.linear = nn.Linear(input_size, output_size)

        # 时间常数 (可学习)
        self.tau_m = nn.Parameter(torch.empty(output_size))
        nn.init.uniform_(self.tau_m, tau_m_init[0], tau_m_init[1])

        # 状态变量
        self.register_buffer('v_init', torch.tensor(0.))

    def reset_memory(self, batch_size: int, device: torch.device):
        """重置状态变量"""
        self.v = torch.zeros(batch_size, self.output_size, device=device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        Args:
            x: [batch_size, input_size] 输入脉冲

        Returns:
            v: [batch_size, output_size] 膜电位
        """
        batch_size = x.shape[0]
        device = x.device

        # 初始化状态变量
        if not hasattr(self, 'v') or self.v.shape[0] != batch_size:
            self.reset_memory(batch_size, device)

        # 计算输入电流
        input_current = self.linear(x)

        # 更新膜电位
        beta = torch.sigmoid(self.tau_m)
        self.v = self.v * beta + (1 - beta) * input_current

        return self.v


class VanillaSFNN(nn.Module):
    """Vanilla脉冲前馈神经网络 - 用于对比实验"""

    def __init__(self, input_size: int = 700, hidden_size: int = 64, output_size: int = 20,
                 tau_m_init: Tuple[float, float] = (0.0, 4.0),
                 v_threshold: float = 1.0, surrogate_function=None):
        """
        初始化Vanilla SFNN

        Args:
            input_size: 输入维度
            hidden_size: 隐藏层维度
            output_size: 输出维度
            tau_m_init: 时间常数初始化范围
            v_threshold: 脉冲阈值
            surrogate_function: 替代函数
        """
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # 替代函数
        if surrogate_function is None:
            surrogate_function = surrogate.ATan()

        # 隐藏层LIF神经元
        self.lif_hidden = neuron.LIFNode(
            tau=2.0, v_threshold=v_threshold, v_reset=0.0,
            surrogate_function=surrogate_function, detach_reset=False
        )

        # 线性层
        self.fc_hidden = nn.Linear(input_size, hidden_size)

        # 读出积分器
        self.readout = ReadoutIntegrator(hidden_size, output_size, tau_m_init)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        Args:
            x: [batch_size, seq_length, input_size] 输入序列

        Returns:
            output: [batch_size, output_size] 输出
        """
        batch_size, seq_length, _ = x.shape
        device = x.device

        # 重置状态
        functional.reset_net(self)
        self.readout.reset_memory(batch_size, device)

        output_sum = torch.zeros(batch_size, self.output_size, device=device)

        # 时间步循环
        for t in range(seq_length):
            # 隐藏层
            hidden_current = self.fc_hidden(x[:, t])
            hidden_spike = self.lif_hidden(hidden_current)

            # 读出层
            readout_v = self.readout(hidden_spike)

            # 累积输出 (跳过前几个时间步)
            if t > 10:
                output_sum += F.softmax(readout_v, dim=1)

        return output_sum


class DH_SFNN(nn.Module):
    """树突异质性脉冲前馈神经网络"""

    def __init__(self, input_size: int = 700, hidden_size: int = 64, output_size: int = 20,
                 num_branches: int = 4, tau_m_init: Tuple[float, float] = (0.0, 4.0),
                 tau_n_init: Tuple[float, float] = (2.0, 6.0),
                 v_threshold: float = 1.0, surrogate_function=None):
        """
        初始化DH-SFNN

        Args:
            input_size: 输入维度
            hidden_size: 隐藏层维度
            output_size: 输出维度
            num_branches: 树突分支数量
            tau_m_init: 膜电位时间常数初始化范围
            tau_n_init: 树突时间常数初始化范围
            v_threshold: 脉冲阈值
            surrogate_function: 替代函数
        """
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_branches = num_branches

        # DH-LIF隐藏层
        self.dh_lif_hidden = DH_LIFNode(
            input_size, hidden_size, num_branches,
            tau_m_init, tau_n_init, v_threshold,
            surrogate_function=surrogate_function
        )

        # 读出积分器
        self.readout = ReadoutIntegrator(hidden_size, output_size, tau_m_init)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        Args:
            x: [batch_size, seq_length, input_size] 输入序列

        Returns:
            output: [batch_size, output_size] 输出
        """
        batch_size, seq_length, _ = x.shape
        device = x.device

        # 重置状态
        self.dh_lif_hidden.reset_memory(batch_size, device)
        self.readout.reset_memory(batch_size, device)

        output_sum = torch.zeros(batch_size, self.output_size, device=device)

        # 时间步循环
        for t in range(seq_length):
            # DH-LIF隐藏层
            hidden_spike = self.dh_lif_hidden(x[:, t])

            # 读出层
            readout_v = self.readout(hidden_spike)

            # 累积输出 (跳过前几个时间步)
            if t > 10:
                output_sum += F.softmax(readout_v, dim=1)

        return output_sum


if __name__ == "__main__":
    # 测试模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 创建测试数据
    batch_size = 4
    seq_length = 100
    input_size = 700

    x = torch.randn(batch_size, seq_length, input_size).to(device)

    # 测试Vanilla SFNN
    print("Testing Vanilla SFNN...")
    vanilla_model = VanillaSFNN().to(device)
    vanilla_output = vanilla_model(x)
    print(f"Vanilla SFNN output shape: {vanilla_output.shape}")

    # 测试DH-SFNN
    print("Testing DH-SFNN...")
    dh_model = DH_SFNN(num_branches=4).to(device)
    dh_output = dh_model(x)
    print(f"DH-SFNN output shape: {dh_output.shape}")

    print("Model tests completed successfully!")
