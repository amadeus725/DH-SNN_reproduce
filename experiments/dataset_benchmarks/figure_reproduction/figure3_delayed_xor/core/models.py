"""
DH-SNN模型架构

包含完整的DH-SNN网络模型，支持多种任务和配置。
"""

import torch
import torch.nn as nn
from typing import List, Optional, Tuple, Union
from spikingjelly.activation_based import functional
try:
    from .layers import DendriticDenseLayer, ReadoutIntegrator
    from .surrogate import MultiGaussianSurrogate
except ImportError:
    from layers import DendriticDenseLayer, ReadoutIntegrator
    from surrogate import MultiGaussianSurrogate


class DH_SNN(nn.Module):
    """
    DH-SNN (Dendritic Heterogeneity Spiking Neural Network)

    完整的DH-SNN网络架构，支持多层树突异质性层和读出积分器。

    Args:
        input_dim: 输入维度
        hidden_dims: 隐藏层维度列表
        output_dim: 输出维度
        num_branches: 树突分支数量
        v_threshold: 脉冲阈值
        tau_m_init: 膜电位时间常数初始化范围
        tau_n_init: 树突时间常数初始化范围
        tau_initializer: 时间常数初始化方法
        sparsity: 连接稀疏度
        mask_share: 共享连接模式的神经元数量
        bias: 是否使用偏置
        surrogate_function: 替代函数
        reset_mode: 重置模式
        readout_tau_init: 读出层时间常数初始化范围
        step_mode: 步进模式
    """

    def __init__(self,
                 input_dim: int,
                 hidden_dims: List[int],
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
                 readout_tau_init: Tuple[float, float] = (0.0, 4.0),
                 step_mode: str = 's'):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        self.num_branches = num_branches
        self.step_mode = step_mode

        if surrogate_function is None:
            surrogate_function = MultiGaussianSurrogate()

        # 构建隐藏层
        self.hidden_layers = nn.ModuleList()

        # 第一个隐藏层
        if hidden_dims:
            first_layer = DendriticDenseLayer(
                input_dim=input_dim,
                output_dim=hidden_dims[0],
                num_branches=num_branches,
                v_threshold=v_threshold,
                tau_m_init=tau_m_init,
                tau_n_init=tau_n_init,
                tau_initializer=tau_initializer,
                sparsity=sparsity,
                mask_share=mask_share,
                bias=bias,
                surrogate_function=surrogate_function,
                reset_mode=reset_mode,
                step_mode=step_mode
            )
            self.hidden_layers.append(first_layer)

            # 其余隐藏层
            for i in range(1, len(hidden_dims)):
                layer = DendriticDenseLayer(
                    input_dim=hidden_dims[i-1],
                    output_dim=hidden_dims[i],
                    num_branches=num_branches,
                    v_threshold=v_threshold,
                    tau_m_init=tau_m_init,
                    tau_n_init=tau_n_init,
                    tau_initializer=tau_initializer,
                    sparsity=sparsity,
                    mask_share=mask_share,
                    bias=bias,
                    surrogate_function=surrogate_function,
                    reset_mode=reset_mode,
                    step_mode=step_mode
                )
                self.hidden_layers.append(layer)

        # 读出层
        readout_input_dim = hidden_dims[-1] if hidden_dims else input_dim
        self.readout = ReadoutIntegrator(
            input_dim=readout_input_dim,
            output_dim=output_dim,
            tau_initializer=tau_initializer,
            tau_init=readout_tau_init,
            bias=bias,
            step_mode=step_mode
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        Args:
            x: 输入脉冲序列 [time_steps, batch_size, input_dim] 或 [batch_size, input_dim]

        Returns:
            输出 [time_steps, batch_size, output_dim] 或 [batch_size, output_dim]
        """
        # 通过隐藏层
        for layer in self.hidden_layers:
            x = layer(x)

        # 通过读出层
        output = self.readout(x)

        return output

    def reset_states(self):
        """
        重置所有层的状态
        """
        functional.reset_net(self)

    def get_tau_parameters(self) -> Tuple[List[torch.Tensor], List[torch.Tensor], torch.Tensor]:
        """
        获取所有时间常数参数

        Returns:
            (tau_m_list, tau_n_list, readout_tau)
        """
        tau_m_list = []
        tau_n_list = []

        for layer in self.hidden_layers:
            tau_m_list.append(layer.tau_m)
            tau_n_list.append(layer.tau_n)

        readout_tau = self.readout.tau_m

        return tau_m_list, tau_n_list, readout_tau

    def apply_connection_masks(self):
        """
        应用所有层的连接掩码
        """
        for layer in self.hidden_layers:
            layer.apply_connection_mask()

    def extra_repr(self) -> str:
        return (f'input_dim={self.input_dim}, hidden_dims={self.hidden_dims}, '
                f'output_dim={self.output_dim}, num_branches={self.num_branches}')


class DH_SFNN(DH_SNN):
    """
    DH-SFNN (DH Spiking Feedforward Neural Network)

    前馈版本的DH-SNN，专门用于前馈任务。
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.network_type = 'feedforward'


class DH_SRNN(nn.Module):
    """
    DH-SRNN (DH Spiking Recurrent Neural Network)

    循环版本的DH-SNN，包含循环连接。

    Args:
        input_dim: 输入维度
        hidden_dim: 隐藏层维度
        output_dim: 输出维度
        num_branches: 树突分支数量
        v_threshold: 脉冲阈值
        tau_m_init: 膜电位时间常数初始化范围
        tau_n_init: 树突时间常数初始化范围
        tau_initializer: 时间常数初始化方法
        sparsity: 连接稀疏度
        bias: 是否使用偏置
        surrogate_function: 替代函数
        reset_mode: 重置模式
        readout_tau_init: 读出层时间常数初始化范围
        step_mode: 步进模式
    """

    def __init__(self,
                 input_dim: int,
                 hidden_dim: int,
                 output_dim: int,
                 num_branches: int = 4,
                 v_threshold: float = 0.5,
                 tau_m_init: Tuple[float, float] = (0.0, 4.0),
                 tau_n_init: Tuple[float, float] = (0.0, 4.0),
                 tau_initializer: str = 'uniform',
                 sparsity: Optional[float] = None,
                 bias: bool = True,
                 surrogate_function: Optional[nn.Module] = None,
                 reset_mode: str = 'soft',
                 readout_tau_init: Tuple[float, float] = (0.0, 4.0),
                 step_mode: str = 's'):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_branches = num_branches
        self.step_mode = step_mode
        self.network_type = 'recurrent'

        if surrogate_function is None:
            surrogate_function = MultiGaussianSurrogate()

        # 输入到隐藏层的连接
        self.input_layer = DendriticDenseLayer(
            input_dim=input_dim,
            output_dim=hidden_dim,
            num_branches=num_branches,
            v_threshold=v_threshold,
            tau_m_init=tau_m_init,
            tau_n_init=tau_n_init,
            tau_initializer=tau_initializer,
            sparsity=sparsity,
            bias=bias,
            surrogate_function=surrogate_function,
            reset_mode=reset_mode,
            step_mode=step_mode
        )

        # 循环连接
        self.recurrent_layer = DendriticDenseLayer(
            input_dim=hidden_dim,
            output_dim=hidden_dim,
            num_branches=num_branches,
            v_threshold=v_threshold,
            tau_m_init=tau_m_init,
            tau_n_init=tau_n_init,
            tau_initializer=tau_initializer,
            sparsity=sparsity,
            bias=False,  # 循环连接通常不使用偏置
            surrogate_function=surrogate_function,
            reset_mode=reset_mode,
            step_mode=step_mode
        )

        # 读出层
        self.readout = ReadoutIntegrator(
            input_dim=hidden_dim,
            output_dim=output_dim,
            tau_initializer=tau_initializer,
            tau_init=readout_tau_init,
            bias=bias,
            step_mode=step_mode
        )

        # 隐藏状态
        self.hidden_state = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        """
        if self.step_mode == 's':
            return self.single_step_forward(x)
        elif self.step_mode == 'm':
            return self.multi_step_forward(x)
        else:
            raise ValueError(f"Unsupported step_mode: {self.step_mode}")

    def single_step_forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        单步前向传播
        """
        # 输入层
        input_output = self.input_layer(x)

        # 循环连接
        if self.hidden_state is None:
            self.hidden_state = torch.zeros_like(input_output)

        recurrent_output = self.recurrent_layer(self.hidden_state)

        # 组合输入和循环信号
        self.hidden_state = input_output + recurrent_output

        # 读出层
        output = self.readout(self.hidden_state)

        return output

    def multi_step_forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        多步前向传播
        """
        outputs = []
        for t in range(x.size(0)):
            output = self.single_step_forward(x[t])
            outputs.append(output)
        return torch.stack(outputs, dim=0)

    def reset_states(self):
        """
        重置所有状态
        """
        functional.reset_net(self)
        self.hidden_state = None

    def extra_repr(self) -> str:
        return (f'input_dim={self.input_dim}, hidden_dim={self.hidden_dim}, '
                f'output_dim={self.output_dim}, num_branches={self.num_branches}')


def create_dh_snn(config: dict) -> Union[DH_SNN, DH_SRNN]:
    """
    根据配置创建DH-SNN模型

    Args:
        config: 模型配置字典

    Returns:
        DH-SNN模型实例
    """
    model_type = config.get('model_type', 'feedforward')

    if model_type == 'feedforward':
        return DH_SNN(**config)
    elif model_type == 'recurrent':
        return DH_SRNN(**config)
    else:
        raise ValueError(f"Unsupported model_type: {model_type}")
