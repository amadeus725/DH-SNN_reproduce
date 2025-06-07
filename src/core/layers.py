"""
DH-SNN层实现

包含树突异质性密集层、读出积分器和相关的网络层组件。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Union
from spikingjelly.activation_based import base
try:
    from .neurons import DH_LIFNode, ReadoutNeuron
    from .surrogate import MultiGaussianSurrogate
except ImportError:
    from neurons import DH_LIFNode, ReadoutNeuron
    from surrogate import MultiGaussianSurrogate


class ConnectionMask:
    """
    连接模式掩码生成器

    用于生成树突分支的连接模式，支持稀疏连接和分支特定连接。
    """

    @staticmethod
    def create_dendritic_mask(input_dim: int,
                            output_dim: int,
                            num_branches: int,
                            sparsity: float = None,
                            mask_share: int = 1,
                            device: torch.device = None) -> torch.Tensor:
        """
        创建树突连接掩码

        Args:
            input_dim: 输入维度
            output_dim: 输出维度
            num_branches: 分支数量
            sparsity: 稀疏度 (None表示使用默认1/num_branches)
            mask_share: 共享掩码的神经元数量
            device: 设备

        Returns:
            连接掩码 [output_dim * num_branches, input_dim]
        """
        if device is None:
            device = torch.device('cpu')

        if sparsity is None:
            sparsity = 1.0 / num_branches

        # 计算填充
        pad = ((input_dim // num_branches * num_branches + num_branches - input_dim) % num_branches)
        padded_input_dim = input_dim + pad

        mask = torch.zeros(output_dim * num_branches, padded_input_dim, device=device)

        for i in range(output_dim // mask_share):
            # 随机排列输入连接
            seq = torch.randperm(padded_input_dim, device=device)

            for j in range(num_branches):
                # 计算每个分支的连接范围
                start_idx = j * padded_input_dim // num_branches
                end_idx = start_idx + int(padded_input_dim * sparsity)

                if end_idx > padded_input_dim:
                    # 处理环绕情况
                    indices1 = seq[start_idx:]
                    indices2 = seq[:end_idx - padded_input_dim]
                    indices = torch.cat([indices1, indices2])
                else:
                    indices = seq[start_idx:end_idx]

                # 为共享掩码的神经元设置连接
                for k in range(mask_share):
                    neuron_idx = i * mask_share + k
                    if neuron_idx < output_dim:
                        mask[neuron_idx * num_branches + j, indices] = 1

        # 移除填充部分
        if pad > 0:
            mask = mask[:, :-pad]

        return mask


class DendriticDenseLayer(base.MemoryModule):
    """
    树突异质性密集层

    DH-SNN的核心层，实现多分支树突结构和异质性时间动态。
    每个神经元有多个树突分支，每个分支有独立的时间常数。

    Args:
        input_dim: 输入维度
        output_dim: 输出神经元数量
        num_branches: 树突分支数量
        v_threshold: 脉冲阈值
        tau_m_init: 膜电位时间常数初始化范围 (low, high)
        tau_n_init: 树突时间常数初始化范围 (low, high)
        tau_initializer: 初始化方法 ('uniform' 或 'constant')
        sparsity: 连接稀疏度
        mask_share: 共享连接模式的神经元数量
        bias: 是否使用偏置
        surrogate_function: 替代函数
        reset_mode: 重置模式
        step_mode: 步进模式
    """

    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 num_branches: int = 4,
                 v_threshold: float = 0.5,
                 tau_m_init: Tuple[float, float] = (0.0, 4.0),
                 tau_n_init: Tuple[float, float] = (0.0, 4.0),
                 tau_initializer: str = 'uniform',
                 sparsity: Optional[float] = None,
                 mask_share: int = 1,
                 bias: bool = True,
                 surrogate_function: Optional[nn.Module] = None,
                 reset_mode: str = 'soft',
                 step_mode: str = 's'):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_branches = num_branches
        self.v_threshold = v_threshold
        self.sparsity = sparsity if sparsity is not None else (1.0 / num_branches)
        self.mask_share = mask_share
        self.reset_mode = reset_mode
        self.step_mode = step_mode

        # 计算填充
        self.pad = ((input_dim // num_branches * num_branches + num_branches - input_dim) % num_branches)

        # 线性层 (输入到所有分支)
        self.linear = nn.Linear(input_dim + self.pad, output_dim * num_branches, bias=bias)

        # 可学习的时间常数参数
        self.tau_m = nn.Parameter(torch.empty(output_dim))
        self.tau_n = nn.Parameter(torch.empty(output_dim, num_branches))

        # 初始化时间常数 - 使用多样化初始化策略避免sigmoid饱和
        if tau_initializer == 'uniform':
            nn.init.uniform_(self.tau_m, tau_m_init[0], tau_m_init[1])
            # 为不同的树突分支使用不同的初始化范围以增加多样性
            self._initialize_tau_n_diverse(tau_n_init)
        elif tau_initializer == 'constant':
            nn.init.constant_(self.tau_m, tau_m_init[0])
            nn.init.constant_(self.tau_n, tau_n_init[0])

        # 替代函数
        if surrogate_function is None:
            self.surrogate_function = MultiGaussianSurrogate()
        else:
            self.surrogate_function = surrogate_function

        # 状态变量
        self.register_memory('v', 0.0)
        self.register_memory('spike', 0.0)
        self.register_memory('d_current', 0.0)

        # 连接掩码 (延迟创建)
        self.mask = None

    def create_connection_mask(self, device: torch.device):
        """
        创建连接掩码
        """
        self.mask = ConnectionMask.create_dendritic_mask(
            input_dim=self.input_dim,
            output_dim=self.output_dim,
            num_branches=self.num_branches,
            sparsity=self.sparsity,
            mask_share=self.mask_share,
            device=device
        )

    def apply_connection_mask(self):
        """
        应用连接掩码到权重
        """
        if self.mask is not None:
            # 扩展掩码以匹配填充后的输入维度
            if self.pad > 0:
                pad_mask = torch.zeros(self.mask.size(0), self.pad, device=self.mask.device)
                extended_mask = torch.cat([self.mask, pad_mask], dim=1)
            else:
                extended_mask = self.mask

            self.linear.weight.data = self.linear.weight.data * extended_mask

    def single_step_forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        单步前向传播

        Args:
            x: 输入脉冲 [batch_size, input_dim]

        Returns:
            输出脉冲 [batch_size, output_dim]
        """
        batch_size = x.size(0)

        # 初始化状态 - 使用GSC成功模式的Variable()方法确保梯度跟踪
        if self.v is None:
            from torch.autograd import Variable
            # 注意：Variable()创建的变量默认requires_grad=False，但参与计算后会自动获得梯度
            self.v = Variable(torch.rand(batch_size, self.output_dim)).to(x.device)
            self.spike = Variable(torch.rand(batch_size, self.output_dim)).to(x.device)
            if self.num_branches == 1:
                self.d_current = Variable(torch.rand(batch_size, self.output_dim, self.num_branches)).to(x.device)
            else:
                self.d_current = Variable(torch.zeros(batch_size, self.output_dim, self.num_branches)).to(x.device)

        # 创建连接掩码 (如果需要)
        if self.mask is None:
            self.create_connection_mask(x.device)

        # 添加填充
        if self.pad > 0:
            padding = torch.zeros(batch_size, self.pad, device=x.device)
            x_padded = torch.cat([x.float(), padding], dim=1)
        else:
            x_padded = x.float()

        # 线性变换到所有分支
        linear_output = self.linear(x_padded)  # [batch_size, output_dim * num_branches]
        dendritic_inputs = linear_output.reshape(batch_size, self.output_dim, self.num_branches)

        # 更新树突电流 (每个分支独立的时间常数) - 这里会创建新的变量并获得梯度
        beta = torch.sigmoid(self.tau_n)  # [output_dim, num_branches]
        self.d_current = beta * self.d_current + (1 - beta) * dendritic_inputs

        # 求和到体细胞
        somatic_input = self.d_current.sum(dim=-1)  # [batch_size, output_dim]

        # 更新膜电位 - 这里会创建新的变量并获得梯度
        alpha = torch.sigmoid(self.tau_m)  # [output_dim]

        if self.reset_mode == 'soft':
            self.v = self.v * alpha + (1 - alpha) * somatic_input - self.v_threshold * self.spike
        elif self.reset_mode == 'hard':
            self.v = self.v * alpha * (1 - self.spike) + (1 - alpha) * somatic_input
        else:  # 'none'
            self.v = self.v * alpha + (1 - alpha) * somatic_input

        # 生成脉冲 - 使用原论文的激活函数确保梯度传递
        class ActFun_adp(torch.autograd.Function):
            @staticmethod
            def forward(ctx, input):
                ctx.save_for_backward(input)
                return input.gt(0.).float()

            @staticmethod
            def backward(ctx, grad_output):
                input, = ctx.saved_tensors
                grad_input = grad_output.clone()
                temp = abs(input) < 0.5
                return grad_input * temp.float()

        # 使用原论文的激活函数代替复杂的替代函数
        self.spike = ActFun_adp.apply(self.v - self.v_threshold)

        return self.spike

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        """
        if self.step_mode == 's':
            return self.single_step_forward(x)
        elif self.step_mode == 'm':
            # 多步模式 (批处理时间步)
            outputs = []
            for t in range(x.size(0)):
                output = self.single_step_forward(x[t])
                outputs.append(output)
            return torch.stack(outputs, dim=0)
        else:
            raise ValueError(f"Unsupported step_mode: {self.step_mode}")

    def extra_repr(self) -> str:
        return (f'input_dim={self.input_dim}, output_dim={self.output_dim}, '
                f'num_branches={self.num_branches}, v_threshold={self.v_threshold}, '
                f'sparsity={self.sparsity:.3f}, reset_mode={self.reset_mode}')

    def _initialize_tau_n_diverse(self, tau_n_init: Tuple[float, float]):
        """
        多样化初始化树突时间常数以避免sigmoid饱和和促进功能多样性

        基于原论文的策略，为不同分支使用不同的初始化范围：
        - 快速分支：负值范围，sigmoid后得到小时间常数 (0.1-0.4)
        - 慢速分支：正值范围，sigmoid后得到大时间常数 (0.6-0.9)
        - 中等分支：零附近，sigmoid后得到中等时间常数 (0.4-0.6)
        """
        output_dim, num_branches = self.tau_n.shape

        if num_branches == 1:
            # 单分支情况，使用原始范围
            nn.init.uniform_(self.tau_n, tau_n_init[0], tau_n_init[1])
        elif num_branches == 2:
            # 双分支：一快一慢
            nn.init.uniform_(self.tau_n[:, 0], -3.0, -1.0)  # 快速分支
            nn.init.uniform_(self.tau_n[:, 1], 1.0, 3.0)    # 慢速分支
        elif num_branches == 4:
            # 四分支：多样化时间尺度
            nn.init.uniform_(self.tau_n[:, 0], -4.0, -2.0)  # 最快分支 (tau~0.02-0.12)
            nn.init.uniform_(self.tau_n[:, 1], -1.5, -0.5)  # 快速分支 (tau~0.18-0.38)
            nn.init.uniform_(self.tau_n[:, 2], 0.5, 1.5)    # 慢速分支 (tau~0.62-0.82)
            nn.init.uniform_(self.tau_n[:, 3], 2.0, 4.0)    # 最慢分支 (tau~0.88-0.98)
        else:
            # 通用情况：将分支分配到不同的时间尺度范围
            for i in range(num_branches):
                # 从快到慢线性分配
                low = -4.0 + (6.0 * i / (num_branches - 1))
                high = low + 1.5
                nn.init.uniform_(self.tau_n[:, i], low, high)


class ReadoutIntegrator(base.MemoryModule):
    """
    读出积分器

    网络的输出层，使用泄漏积分器进行膜电位积分但不产生脉冲。

    Args:
        input_dim: 输入维度
        output_dim: 输出维度
        tau_initializer: 时间常数初始化方法
        tau_init: 时间常数初始化范围 (low, high)
        bias: 是否使用偏置
        step_mode: 步进模式
    """

    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 tau_initializer: str = 'uniform',
                 tau_init: Tuple[float, float] = (0.0, 4.0),
                 bias: bool = True,
                 step_mode: str = 's'):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.step_mode = step_mode

        # 线性层
        self.linear = nn.Linear(input_dim, output_dim, bias=bias)

        # 可学习的时间常数
        self.tau_m = nn.Parameter(torch.empty(output_dim))

        # 初始化时间常数
        if tau_initializer == 'uniform':
            nn.init.uniform_(self.tau_m, tau_init[0], tau_init[1])
        elif tau_initializer == 'constant':
            nn.init.constant_(self.tau_m, tau_init[0])

        # 膜电位状态
        self.register_memory('v', 0.0)

    def single_step_forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        单步前向传播
        """
        batch_size = x.size(0)

        # 初始化状态 - 使用GSC成功模式确保梯度跟踪
        if self.v is None:
            from torch.autograd import Variable
            self.v = Variable(torch.rand(batch_size, self.output_dim)).to(x.device)

        # 线性变换
        synaptic_input = self.linear(x.float())

        # 泄漏积分
        alpha = torch.sigmoid(self.tau_m)
        self.v = self.v * alpha + (1 - alpha) * synaptic_input

        return self.v

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        """
        if self.step_mode == 's':
            return self.single_step_forward(x)
        elif self.step_mode == 'm':
            outputs = []
            for t in range(x.size(0)):
                output = self.single_step_forward(x[t])
                outputs.append(output)
            return torch.stack(outputs, dim=0)
        else:
            raise ValueError(f"Unsupported step_mode: {self.step_mode}")

    def extra_repr(self) -> str:
        return f'input_dim={self.input_dim}, output_dim={self.output_dim}'
