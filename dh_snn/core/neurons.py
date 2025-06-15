"""
DH-SNN神经元模型实现

包含DH-LIF (Dendritic Heterogeneity Leaky Integrate-and-Fire)神经元
和相关的神经元组件。
"""

import torch
import torch.nn as nn
from typing import Callable, Optional, Union
from spikingjelly.activation_based import neuron, base
try:
    from .surrogate import MultiGaussianSurrogate
except ImportError:
    from surrogate import MultiGaussianSurrogate


class DH_LIFNode(neuron.LIFNode):
    """
    树突异质性LIF神经元 (DH-LIF)

    这是DH-SNN的核心组件，在标准LIF神经元基础上增加了树突异质性。
    每个神经元有多个树突分支，每个分支有独立的时间常数，
    实现多时间尺度的信息处理。

    Args:
        tau_m: 膜电位时间常数 (可学习参数)
        tau_n: 树突分支时间常数 (可学习参数)
        num_branches: 树突分支数量
        v_threshold: 脉冲阈值
        v_reset: 重置电位
        surrogate_function: 替代函数
        reset_mode: 重置模式 ('soft', 'hard', 'none')
        detach_reset: 是否分离重置梯度
        step_mode: 步进模式 ('s' 或 'm')
        backend: 后端类型
        store_v_seq: 是否存储电压序列
    """

    def __init__(self,
                 tau_m: float = 2.0,
                 tau_n: Optional[torch.Tensor] = None,
                 num_branches: int = 4,
                 v_threshold: float = 1.0,
                 v_reset: float = 0.0,
                 surrogate_function: Callable = None,
                 reset_mode: str = 'soft',
                 detach_reset: bool = False,
                 step_mode: str = 's',
                 backend: str = 'torch',
                 store_v_seq: bool = False):

        # 使用多高斯替代函数作为默认
        if surrogate_function is None:
            surrogate_function = MultiGaussianSurrogate()

        super().__init__(
            tau=tau_m,
            v_threshold=v_threshold,
            v_reset=v_reset,
            surrogate_function=surrogate_function,
            detach_reset=detach_reset,
            step_mode=step_mode,
            backend=backend,
            store_v_seq=store_v_seq
        )

        self.num_branches = num_branches
        self.reset_mode = reset_mode

        # 可学习的时间常数参数
        self.tau_m = nn.Parameter(torch.tensor(tau_m, dtype=torch.float))

        if tau_n is None:
            # 默认初始化树突时间常数
            self.tau_n = nn.Parameter(torch.rand(num_branches) * 4.0)
        else:
            self.tau_n = nn.Parameter(tau_n.clone())

        # 树突电流状态 (将在forward中动态创建)
        self.register_memory('d_current', None)

        # 脉冲状态
        self.register_memory('spike', 0.0)

    def neuronal_charge(self, x: torch.Tensor) -> torch.Tensor:
        """
        神经元充电过程

        Args:
            x: 输入电流 [batch_size, num_neurons, num_branches]

        Returns:
            更新后的膜电位
        """
        if x.dim() == 2:
            # 如果输入是2D，扩展为3D
            batch_size, num_neurons = x.shape
            x = x.unsqueeze(-1).expand(batch_size, num_neurons, self.num_branches)

        batch_size, num_neurons, num_branches = x.shape

        # 初始化树突电流状态
        if self.d_current is None or self.d_current.shape != (batch_size, num_neurons, num_branches):
            self.d_current = torch.zeros(batch_size, num_neurons, num_branches,
                                       device=x.device, dtype=x.dtype)

        # 更新树突电流 (每个分支独立的时间常数)
        beta = torch.sigmoid(self.tau_n)  # [num_branches]
        beta = beta.view(1, 1, -1)  # 广播到 [1, 1, num_branches]

        self.d_current = beta * self.d_current + (1 - beta) * x

        # 求和到体细胞
        somatic_input = self.d_current.sum(dim=-1)  # [batch_size, num_neurons]

        # 更新膜电位
        alpha = torch.sigmoid(self.tau_m)

        if self.reset_mode == 'soft':
            # 软重置：减去阈值电压
            self.v = self.v * alpha + (1 - alpha) * somatic_input - self.v_threshold * self.spike
        elif self.reset_mode == 'hard':
            # 硬重置：脉冲后重置为0
            self.v = self.v * alpha * (1 - self.spike) + (1 - alpha) * somatic_input
        else:  # 'none'
            # 无重置：不进行重置
            self.v = self.v * alpha + (1 - alpha) * somatic_input

        return self.v

    def neuronal_fire(self) -> torch.Tensor:
        """
        神经元放电过程
        """
        self.spike = self.surrogate_function(self.v - self.v_threshold)
        return self.spike

    def extra_repr(self) -> str:
        """
        额外的字符串表示
        """
        return (f'num_branches={self.num_branches}, '
                f'v_threshold={self.v_threshold}, '
                f'reset_mode={self.reset_mode}, '
                f'tau_m={self.tau_m.item():.3f}, '
                f'tau_n_range=[{self.tau_n.min().item():.3f}, {self.tau_n.max().item():.3f}]')


class ParametricLIFNode(neuron.LIFNode):
    """
    参数化LIF神经元

    标准LIF神经元的可学习参数版本，用于对比实验。

    Args:
        tau: 时间常数 (可学习)
        v_threshold: 脉冲阈值
        v_reset: 重置电位
        surrogate_function: 替代函数
        reset_mode: 重置模式
        detach_reset: 是否分离重置梯度
        step_mode: 步进模式
        backend: 后端类型
        store_v_seq: 是否存储电压序列
    """

    def __init__(self,
                 tau: float = 2.0,
                 v_threshold: float = 1.0,
                 v_reset: float = 0.0,
                 surrogate_function: Callable = None,
                 reset_mode: str = 'soft',
                 detach_reset: bool = False,
                 step_mode: str = 's',
                 backend: str = 'torch',
                 store_v_seq: bool = False):

        if surrogate_function is None:
            surrogate_function = MultiGaussianSurrogate()

        super().__init__(
            tau=tau,
            v_threshold=v_threshold,
            v_reset=v_reset,
            surrogate_function=surrogate_function,
            detach_reset=detach_reset,
            step_mode=step_mode,
            backend=backend,
            store_v_seq=store_v_seq
        )

        self.reset_mode = reset_mode

        # 将tau转换为可学习参数
        self.tau = nn.Parameter(torch.tensor(tau, dtype=torch.float))

        # 脉冲状态
        self.register_memory('spike', 0.0)

    def neuronal_charge(self, x: torch.Tensor) -> torch.Tensor:
        """
        神经元充电过程
        """
        alpha = torch.sigmoid(self.tau)

        if self.reset_mode == 'soft':
            self.v = self.v * alpha + (1 - alpha) * x - self.v_threshold * self.spike
        elif self.reset_mode == 'hard':
            self.v = self.v * alpha * (1 - self.spike) + (1 - alpha) * x
        else:  # 'none'
            self.v = self.v * alpha + (1 - alpha) * x

        return self.v

    def extra_repr(self) -> str:
        return (f'tau={self.tau.item():.3f}, '
                f'v_threshold={self.v_threshold}, '
                f'reset_mode={self.reset_mode}')


class ReadoutNeuron(base.MemoryModule):
    """
    读出神经元 (无脉冲的泄漏积分器)

    用于网络的输出层，进行膜电位积分但不产生脉冲。

    Args:
        tau: 时间常数 (可学习)
        step_mode: 步进模式
        backend: 后端类型
        store_v_seq: 是否存储电压序列
    """

    def __init__(self,
                 tau: float = 2.0,
                 step_mode: str = 's',
                 backend: str = 'torch',
                 store_v_seq: bool = False):
        super().__init__()

        self.step_mode = step_mode
        self.backend = backend
        self.store_v_seq = store_v_seq

        # 可学习的时间常数
        self.tau = nn.Parameter(torch.tensor(tau, dtype=torch.float))

        # 膜电位
        self.register_memory('v', 0.0)

        if store_v_seq:
            self.register_memory('v_seq', [])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        Args:
            x: 输入电流

        Returns:
            膜电位 (无脉冲)
        """
        alpha = torch.sigmoid(self.tau)
        self.v = self.v * alpha + (1 - alpha) * x

        if self.store_v_seq:
            if self.v_seq is None:
                self.v_seq = []
            self.v_seq.append(self.v.clone())

        return self.v

    def extra_repr(self) -> str:
        return f'tau={self.tau.item():.3f}'
