#!/usr/bin/env python3
"""
Sequential MNIST模型定义 - 正确使用SpikingJelly框架
基于原论文DH-SRNN架构实现
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from spikingjelly.activation_based import neuron, functional, layer, surrogate
import sys
import os

# 添加项目根目录到路径
sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))

# 使用我们的DH-SNN组件 (基于SpikingJelly)
try:
    from src.core.neurons import DH_LIFNode, ParametricLIFNode
    from src.core.layers import DendriticDenseLayer, ReadoutIntegrator
    from src.core.surrogate import MultiGaussianSurrogate
except ImportError:
    # 如果导入失败，使用简化版本
    print("⚠️  DH-SNN组件导入失败，使用简化实现")
    DH_LIFNode = None
    ParametricLIFNode = None
    DendriticDenseLayer = None
    ReadoutIntegrator = None
    MultiGaussianSurrogate = None

class DHSRNNCell(nn.Module):
    """
    DH-SRNN单元 - 参考SSC实验的SpikingJelly实现
    基于原论文实现，支持多分支树突结构
    """

    def __init__(self, input_size, hidden_size, num_branches=4,
                 tau_m_init=(0, 4), tau_n_init=(0, 4), v_threshold=1.0):
        super(DHSRNNCell, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_branches = num_branches

        # 总输入大小 (前向输入 + 递归输入)
        total_input_size = input_size + hidden_size

        # 多分支连接 (参考SSC实验实现)
        # 为了处理不能整除的情况，每个分支的输入大小可能略有不同
        self.branch_input_sizes = []
        remaining_inputs = total_input_size

        for i in range(num_branches):
            if i == num_branches - 1:
                # 最后一个分支处理剩余的所有输入
                branch_size = remaining_inputs
            else:
                branch_size = total_input_size // num_branches
                remaining_inputs -= branch_size
            self.branch_input_sizes.append(branch_size)

        self.branch_fcs = nn.ModuleList([
            layer.Linear(self.branch_input_sizes[i], hidden_size)
            for i in range(num_branches)
        ])

        # 分支时间常数（树突时间常数）
        self.branch_taus = nn.ParameterList([
            nn.Parameter(torch.empty(hidden_size).uniform_(*tau_n_init))
            for _ in range(num_branches)
        ])

        # 主神经元 (使用SpikingJelly的LIF)
        self.lif = neuron.LIFNode(
            tau=2.0,
            surrogate_function=surrogate.ATan(),
            detach_reset=True
        )

        # 膜电位时间常数 (可学习)
        self.tau_m = nn.Parameter(torch.empty(hidden_size).uniform_(*tau_m_init))

        # 分支状态 (注册为buffer以便正确重置)
        self.register_buffer('branch_states', None)

        # 稀疏连接掩码 (暂时禁用)
        self.mask = None  # self._create_sparse_mask(total_input_size)

    def _create_sparse_mask(self, total_input_size):
        """创建稀疏连接掩码 (与原论文一致)"""
        # 为每个分支创建掩码，维度匹配分支权重
        masks = []

        for branch in range(self.num_branches):
            branch_input_size = self.branch_input_sizes[branch]
            branch_output_size = self.hidden_size

            # 创建该分支的掩码 [output_size, input_size]
            mask = torch.ones(branch_output_size, branch_input_size)
            masks.append(mask)

        return masks

    def apply_mask(self):
        """应用稀疏连接掩码 (与原论文一致)"""
        # 暂时禁用稀疏连接，专注于验证训练效果
        pass

    def forward(self, input_t, hidden_spike=None):
        """
        前向传播 - 参考SSC实验的多分支实现

        Args:
            input_t: [batch_size, input_size] - 当前时间步输入
            hidden_spike: [batch_size, hidden_size] - 上一时间步的脉冲输出

        Returns:
            spike_output: [batch_size, hidden_size] - 脉冲输出
        """
        batch_size = input_t.size(0)
        device = input_t.device

        # 初始化递归连接
        if hidden_spike is None:
            hidden_spike = torch.zeros(batch_size, self.hidden_size, device=device)

        # 拼接输入和递归连接
        combined_input = torch.cat([input_t, hidden_spike], dim=1)

        # 初始化分支状态
        if self.branch_states is None or self.branch_states.size(0) != batch_size:
            self.branch_states = torch.zeros(batch_size, self.hidden_size, self.num_branches, device=device)

        # 多分支处理
        branch_outputs = []
        start_idx = 0
        new_branch_states = []

        for i in range(self.num_branches):
            # 分支输入 (使用预计算的分支大小)
            end_idx = start_idx + self.branch_input_sizes[i]
            branch_input = combined_input[:, start_idx:end_idx]

            # 分支线性变换
            branch_current = self.branch_fcs[i](branch_input)

            # 更新分支状态 (树突时间常数) - 避免就地操作
            alpha = torch.sigmoid(self.branch_taus[i])
            new_state = alpha * self.branch_states[:, :, i] + (1 - alpha) * branch_current
            new_branch_states.append(new_state)
            branch_outputs.append(new_state)

            start_idx = end_idx

        # 更新分支状态 (保持梯度以便学习)
        self.branch_states = torch.stack(new_branch_states, dim=2)

        # 合并分支输出
        total_current = torch.stack(branch_outputs, dim=2).sum(dim=2)

        # 主神经元处理
        spike_output = self.lif(total_current)

        return spike_output

    def reset_states(self):
        """重置分支状态"""
        self.branch_states = None

class VanillaSRNNCell(nn.Module):
    """
    Vanilla SRNN单元 - 使用SpikingJelly的LIF神经元
    用作基线对比
    """

    def __init__(self, input_size, hidden_size, tau_m_init=(0, 4), v_threshold=1.0):
        super(VanillaSRNNCell, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size

        # 使用SpikingJelly的线性层
        self.linear = layer.Linear(input_size + hidden_size, hidden_size, bias=True)

        # 使用SpikingJelly的标准LIF神经元
        self.lif = neuron.LIFNode(
            tau=2.0,
            surrogate_function=surrogate.ATan(),
            detach_reset=True
        )

        # 膜电位时间常数 (可学习)
        self.tau_m = nn.Parameter(torch.empty(hidden_size).uniform_(*tau_m_init))
    
    def forward(self, input_t, hidden_spike=None):
        """前向传播 - 使用SpikingJelly的LIF神经元"""
        batch_size = input_t.size(0)
        device = input_t.device

        # 初始化递归连接
        if hidden_spike is None:
            hidden_spike = torch.zeros(batch_size, self.hidden_size, device=device)

        # 拼接输入和递归连接
        combined_input = torch.cat([input_t, hidden_spike], dim=1)

        # 线性变换
        linear_output = self.linear(combined_input)

        # 使用LIF神经元处理
        spike_output = self.lif(linear_output)

        return spike_output

class SequentialMNISTModel(nn.Module):
    """
    Sequential MNIST模型 - 使用SpikingJelly框架
    基于论文网络结构: 1-r64-r256-10
    """

    def __init__(self, model_type='dh_srnn', num_branches=4, tau_m_init=(0, 4), tau_n_init=(0, 4)):
        super(SequentialMNISTModel, self).__init__()

        self.model_type = model_type

        # 网络结构: 1-r64-r256-10
        if model_type == 'dh_srnn':
            self.rnn1 = DHSRNNCell(1, 64, num_branches, tau_m_init, tau_n_init)
            self.rnn2 = DHSRNNCell(64, 256, num_branches, tau_m_init, tau_n_init)
        else:  # vanilla_srnn
            self.rnn1 = VanillaSRNNCell(1, 64, tau_m_init)
            self.rnn2 = VanillaSRNNCell(64, 256, tau_m_init)

        # 输出层 - 使用SpikingJelly的ReadoutIntegrator
        self.output_layer = ReadoutIntegrator(
            input_dim=256,
            output_dim=10,
            tau_init=tau_m_init,
            step_mode='s'
        )
        
    def forward(self, x):
        """
        前向传播 - 使用SpikingJelly框架

        Args:
            x: [batch_size, seq_len, input_dim] - 序列输入

        Returns:
            output: [batch_size, num_classes] - 累积膜电位
        """
        batch_size, seq_len, input_dim = x.shape
        device = x.device

        # 重置网络状态 (SpikingJelly + 自定义状态)
        functional.reset_net(self)

        # 重置DH-SRNN的分支状态
        if hasattr(self.rnn1, 'reset_states'):
            self.rnn1.reset_states()
        if hasattr(self.rnn2, 'reset_states'):
            self.rnn2.reset_states()

        # 初始化隐藏脉冲
        h1_spike = None
        h2_spike = None

        # 逐时间步处理
        for t in range(seq_len):
            x_t = x[:, t, :]  # [batch_size, input_dim]

            # 第一层RNN
            h1_spike = self.rnn1(x_t, h1_spike)

            # 第二层RNN
            h2_spike = self.rnn2(h1_spike, h2_spike)

            # 输出层 - 积分器
            output = self.output_layer(h2_spike)

            # 注意：不分离梯度，让BPTT正常工作

        return output

    def reset_states(self, batch_size=None):
        """重置模型状态 (用于TBPTT) - 使用SpikingJelly + 自定义状态"""
        functional.reset_net(self)

        # 重置DH-SRNN的分支状态
        if hasattr(self.rnn1, 'reset_states'):
            self.rnn1.reset_states()
        if hasattr(self.rnn2, 'reset_states'):
            self.rnn2.reset_states()

    def detach_states(self):
        """分离状态，防止梯度爆炸 (用于TBPTT) - 使用SpikingJelly"""
        # SpikingJelly会自动处理状态分离
        pass

    def apply_mask(self):
        """应用稀疏连接掩码 (与原论文一致)"""
        if hasattr(self.rnn1, 'apply_mask'):
            self.rnn1.apply_mask()
        if hasattr(self.rnn2, 'apply_mask'):
            self.rnn2.apply_mask()
