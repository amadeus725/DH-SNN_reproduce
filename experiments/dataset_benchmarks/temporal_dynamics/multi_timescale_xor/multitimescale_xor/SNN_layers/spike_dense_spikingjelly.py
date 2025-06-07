# SpikingJelly重构版本的DH-SNN架构
import torch
import torch.nn as nn
import numpy as np
from typing import Callable, Optional
from spikingjelly.activation_based import neuron, functional, surrogate, layer
from spikingjelly.activation_based.base import MemoryModule


class MultiGaussianSurrogate(surrogate.SurrogateFunctionBase):
    """多高斯替代函数，与原始实现保持一致"""
    
    def __init__(self, alpha: float = 0.5, sigma: float = 0.5, scale: float = 6.0, height: float = 0.15):
        super().__init__(alpha)
        self.sigma = sigma
        self.scale = scale
        self.height = height
    
    def forward(self, x: torch.Tensor):
        return self.spiking_function(x, self.alpha, self.sigma, self.scale, self.height)
    
    @staticmethod
    def spiking_function(x: torch.Tensor, alpha: float, sigma: float, scale: float, height: float):
        return (x >= 0).float()
    
    @staticmethod
    def primitive_function(x: torch.Tensor, alpha: float, sigma: float, scale: float, height: float):
        def gaussian(x, mu=0., sigma=0.5):
            return torch.exp(-((x - mu) ** 2) / (2 * sigma ** 2)) / torch.sqrt(2 * torch.tensor(torch.pi)) / sigma
        
        temp = gaussian(x, mu=0., sigma=sigma) * (1. + height) \
               - gaussian(x, mu=sigma, sigma=scale * sigma) * height \
               - gaussian(x, mu=-sigma, sigma=scale * sigma) * height
        return temp * alpha


class CustomParametricLIFNode(neuron.BaseNode):
    """自定义参数化LIF神经元，支持可训练的tau参数"""
    
    def __init__(self, 
                 tau_m: float = 2.0,
                 v_threshold: float = 1.0,
                 v_reset: Optional[float] = 0.0,
                 surrogate_function: Callable = surrogate.ATan(),
                 detach_reset: bool = False,
                 step_mode: str = 's',
                 backend: str = 'torch',
                 store_v_seq: bool = False,
                 tau_learnable: bool = True,
                 reset_mode: str = 'soft'):
        """
        Args:
            tau_m: 膜时间常数
            v_threshold: 阈值
            v_reset: 重置电位
            surrogate_function: 替代函数
            detach_reset: 是否分离重置
            step_mode: 步进模式
            backend: 后端
            store_v_seq: 是否存储电压序列
            tau_learnable: tau是否可学习
            reset_mode: 重置模式 ('soft', 'hard', 'none')
        """
        
        super().__init__(v_threshold, v_reset, surrogate_function, detach_reset, step_mode, backend, store_v_seq)
        
        if tau_learnable:
            self.tau_m = nn.Parameter(torch.tensor(tau_m))
        else:
            self.register_buffer('tau_m', torch.tensor(tau_m))
        
        self.reset_mode = reset_mode
    
    def neuronal_charge(self, x: torch.Tensor):
        alpha = torch.sigmoid(self.tau_m)
        if self.reset_mode == 'soft':
            self.v = self.v * alpha + (1 - alpha) * x - self.v_threshold * self.spike
        elif self.reset_mode == 'hard':
            self.v = self.v * alpha * (1 - self.spike) + (1 - alpha) * x
        elif self.reset_mode == 'none':
            self.v = self.v * alpha + (1 - alpha) * x
        else:
            raise ValueError(f"Unknown reset mode: {self.reset_mode}")


class ReadoutIntegrator(MemoryModule):
    """读出层积分器 - 使用SpikingJelly重构"""
    
    def __init__(self, 
                 input_dim: int, 
                 output_dim: int,
                 tau_initializer: str = 'uniform',
                 low_m: float = 0.0,
                 high_m: float = 4.0,
                 bias: bool = True,
                 step_mode: str = 's'):
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.step_mode = step_mode
        
        # 线性层
        self.linear = nn.Linear(input_dim, output_dim, bias=bias)
        
        # 可训练的tau参数
        self.tau_m = nn.Parameter(torch.empty(output_dim))
        
        # 初始化tau参数
        if tau_initializer == 'uniform':
            nn.init.uniform_(self.tau_m, low_m, high_m)
        elif tau_initializer == 'constant':
            nn.init.constant_(self.tau_m, low_m)
        
        # 膜电位
        self.register_memory('v', 0.0)
    
    def single_step_forward(self, x: torch.Tensor):
        """单步前向传播"""
        synaptic_input = self.linear(x.float())
        alpha = torch.sigmoid(self.tau_m)
        self.v = self.v * alpha + (1 - alpha) * synaptic_input
        return self.v
    
    def forward(self, x: torch.Tensor):
        if self.step_mode == 's':
            return self.single_step_forward(x)
        elif self.step_mode == 'm':
            return functional.seq_to_ann_forward(x, self.single_step_forward)


class DendriticLIF(CustomParametricLIFNode):
    """支持树突分支的LIF神经元"""
    
    def __init__(self,
                 num_branches: int = 4,
                 tau_m: float = 2.0,
                 tau_n_init: tuple = (0.0, 4.0),
                 v_threshold: float = 0.5,
                 surrogate_function: Callable = None,
                 tau_initializer: str = 'uniform',
                 **kwargs):
        
        if surrogate_function is None:
            surrogate_function = MultiGaussianSurrogate()
        
        super().__init__(tau_m=tau_m, v_threshold=v_threshold, 
                        surrogate_function=surrogate_function, **kwargs)
        
        self.num_branches = num_branches
        
        # 树突分支的时间常数
        self.tau_n = nn.Parameter(torch.empty(num_branches))
        
        if tau_initializer == 'uniform':
            nn.init.uniform_(self.tau_n, tau_n_init[0], tau_n_init[1])
        elif tau_initializer == 'constant':
            nn.init.constant_(self.tau_n, tau_n_init[0])
        
        # 树突电流状态 - 初始化为None，在第一次前向传播时创建
        self.d_current = None
    
    def single_step_forward(self, dendritic_inputs: torch.Tensor):
        """重写单步前向传播以处理树突输入"""
        self.neuronal_charge(dendritic_inputs)
        
        # 确保spike变量有正确的形状
        somatic_input_shape = dendritic_inputs.sum(dim=-1).shape
        if not hasattr(self, 'spike') or self.spike is None or self.spike.shape != somatic_input_shape:
            self.spike = torch.zeros(somatic_input_shape, device=dendritic_inputs.device)
        
        self.spike = self.surrogate_function(self.v - self.v_threshold)
        return self.spike
    
    def extra_repr(self):
        return f'num_branches={self.num_branches}, tau_m={self.tau_m}, tau_n={self.tau_n}'
    
    def neuronal_charge(self, dendritic_inputs: torch.Tensor):
        """
        Args:
            dendritic_inputs: shape (batch_size, num_branches) 或 (batch_size, output_dim, num_branches)
        """
        # 初始化d_current如果必要
        if self.d_current is None or self.d_current.shape != dendritic_inputs.shape:
            self.d_current = torch.zeros_like(dendritic_inputs)
        
        # 更新树突电流
        beta = torch.sigmoid(self.tau_n)
        
        # 确保beta的维度匹配
        if dendritic_inputs.dim() == 3:  # (batch_size, output_dim, num_branches)
            beta = beta.view(1, 1, -1)
        elif dendritic_inputs.dim() == 2:  # (batch_size, num_branches)
            beta = beta.view(1, -1)
            
        self.d_current = beta * self.d_current + (1 - beta) * dendritic_inputs
        somatic_input = self.d_current.sum(dim=-1)
        
        # 初始化膜电位如果必要
        if self.v is None or self.v.shape != somatic_input.shape:
            self.v = torch.zeros_like(somatic_input)
        
        # 更新膜电位
        alpha = torch.sigmoid(self.tau_m)
        if self.reset_mode == 'soft':
            self.v = self.v * alpha + (1 - alpha) * somatic_input - self.v_threshold * self.spike
        elif self.reset_mode == 'hard':
            self.v = self.v * alpha * (1 - self.spike) + (1 - alpha) * somatic_input
        elif self.reset_mode == 'none':
            self.v = self.v * alpha + (1 - alpha) * somatic_input


class DendriticDenseLayer(MemoryModule):
    """树突密集层 - DH-SFNN的核心组件"""
    
    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 num_branches: int = 4,
                 tau_m_init: tuple = (0.0, 4.0),
                 tau_n_init: tuple = (0.0, 4.0),
                 v_threshold: float = 0.5,
                 tau_initializer: str = 'uniform',
                 test_sparsity: bool = False,
                 sparsity: float = 0.5,
                 mask_share: int = 1,
                 bias: bool = True,
                 surrogate_function: Callable = None,
                 step_mode: str = 's'):
        
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_branches = num_branches
        self.test_sparsity = test_sparsity
        self.sparsity = sparsity if test_sparsity else 1.0 / num_branches
        self.mask_share = mask_share
        self.step_mode = step_mode
        
        # 计算填充大小
        self.pad = ((input_dim // num_branches * num_branches + num_branches - input_dim) % num_branches)
        
        # 密集连接层
        self.linear = nn.Linear(input_dim + self.pad, output_dim * num_branches, bias=bias)
        
        # 树突LIF神经元
        self.neurons = DendriticLIF(
            num_branches=num_branches,
            tau_m=tau_m_init[0] if tau_initializer == 'constant' else 2.0,
            tau_n_init=tau_n_init,
            v_threshold=v_threshold,
            surrogate_function=surrogate_function,
            tau_initializer=tau_initializer,
            step_mode=step_mode
        )
        
        # 初始化神经元的tau参数
        if tau_initializer == 'uniform':
            if hasattr(self.neurons, 'tau_m'):
                nn.init.uniform_(self.neurons.tau_m, tau_m_init[0], tau_m_init[1])
        elif tau_initializer == 'constant':
            if hasattr(self.neurons, 'tau_m'):
                nn.init.constant_(self.neurons.tau_m, tau_m_init[0])
        
        # 创建连接掩码
        self._create_mask()
    
    def _create_mask(self):
        """创建稀疏连接掩码"""
        input_size = self.input_dim + self.pad
        mask = torch.zeros(self.output_dim * self.num_branches, input_size)
        
        for i in range(self.output_dim // self.mask_share):
            seq = torch.randperm(input_size)
            
            for j in range(self.num_branches):
                if self.test_sparsity:
                    # 稀疏连接模式
                    num_connections = int(input_size * self.sparsity)
                    start_idx = j * input_size // self.num_branches
                    end_idx = start_idx + num_connections
                    
                    if end_idx > input_size:
                        # 处理越界情况
                        for k in range(self.mask_share):
                            neuron_idx = (i * self.mask_share + k) * self.num_branches + j
                            mask[neuron_idx, seq[start_idx:]] = 1
                            mask[neuron_idx, seq[:end_idx - input_size]] = 1
                    else:
                        for k in range(self.mask_share):
                            neuron_idx = (i * self.mask_share + k) * self.num_branches + j
                            mask[neuron_idx, seq[start_idx:end_idx]] = 1
                else:
                    # 均匀分布连接模式
                    start_idx = j * input_size // self.num_branches
                    end_idx = (j + 1) * input_size // self.num_branches
                    
                    for k in range(self.mask_share):
                        neuron_idx = (i * self.mask_share + k) * self.num_branches + j
                        mask[neuron_idx, seq[start_idx:end_idx]] = 1
        
        self.register_buffer('mask', mask)
    
    def apply_mask(self):
        """应用连接掩码"""
        self.linear.weight.data = self.linear.weight.data * self.mask
    
    def single_step_forward(self, x: torch.Tensor):
        """单步前向传播"""
        # 添加填充
        if self.pad > 0:
            padding = torch.zeros(x.size(0), self.pad, device=x.device, dtype=x.dtype)
            x_padded = torch.cat([x.float(), padding], dim=1)
        else:
            x_padded = x.float()
        
        # 线性变换
        linear_output = self.linear(x_padded)
        
        # 重塑为树突分支形式
        dendritic_inputs = linear_output.reshape(-1, self.output_dim, self.num_branches)
        
        # 神经元更新
        return self.neurons(dendritic_inputs)
    
    def forward(self, x: torch.Tensor):
        if self.step_mode == 's':
            return self.single_step_forward(x)
        elif self.step_mode == 'm':
            return functional.seq_to_ann_forward(x, self.single_step_forward)


class VanillaSFNN(MemoryModule):
    """原版SFNN层 - 使用SpikingJelly重构"""
    
    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 tau_m_init: tuple = (0.0, 4.0),
                 v_threshold: float = 0.5,
                 tau_initializer: str = 'uniform',
                 bias: bool = True,
                 reset_mode: str = 'soft',
                 surrogate_function: Callable = None,
                 step_mode: str = 's'):
        
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.step_mode = step_mode
        
        # 线性层
        self.linear = nn.Linear(input_dim, output_dim, bias=bias)
        
        # LIF神经元
        if surrogate_function is None:
            surrogate_function = MultiGaussianSurrogate()
        
        self.neurons = CustomParametricLIFNode(
            tau_m=tau_m_init[0] if tau_initializer == 'constant' else 2.0,
            v_threshold=v_threshold,
            surrogate_function=surrogate_function,
            reset_mode=reset_mode,
            step_mode=step_mode
        )
        
        # 初始化tau参数
        if tau_initializer == 'uniform':
            nn.init.uniform_(self.neurons.tau_m, tau_m_init[0], tau_m_init[1])
        elif tau_initializer == 'constant':
            nn.init.constant_(self.neurons.tau_m, tau_m_init[0])
    
    def single_step_forward(self, x: torch.Tensor):
        """单步前向传播"""
        synaptic_input = self.linear(x.float())
        return self.neurons(synaptic_input)
    
    def forward(self, x: torch.Tensor):
        if self.step_mode == 's':
            return self.single_step_forward(x)
        elif self.step_mode == 'm':
            return functional.seq_to_ann_forward(x, self.single_step_forward)


class DH_SNN_Network(nn.Module):
    """完整的DH-SNN网络"""
    
    def __init__(self,
                 input_dim: int,
                 hidden_dims: list,
                 output_dim: int,
                 num_branches: int = 4,
                 use_dendritic: bool = True,
                 step_mode: str = 's'):
        
        super().__init__()
        
        self.step_mode = step_mode
        self.use_dendritic = use_dendritic
        
        # 构建隐藏层
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            if use_dendritic:
                layer = DendriticDenseLayer(
                    input_dim=prev_dim,
                    output_dim=hidden_dim,
                    num_branches=num_branches,
                    step_mode=step_mode
                )
            else:
                layer = VanillaSFNN(
                    input_dim=prev_dim,
                    output_dim=hidden_dim,
                    step_mode=step_mode
                )
            layers.append(layer)
            prev_dim = hidden_dim
        
        self.hidden_layers = nn.ModuleList(layers)
        
        # 读出层
        self.readout = ReadoutIntegrator(
            input_dim=prev_dim,
            output_dim=output_dim,
            step_mode=step_mode
        )
    
    def forward(self, x: torch.Tensor):
        # 隐藏层前向传播
        for layer in self.hidden_layers:
            x = layer(x)
        
        # 读出层
        output = self.readout(x)
        
        return output


# 使用示例
def create_dh_snn_model(input_dim=2, hidden_dims=[200], output_dim=1, num_branches=4):
    """创建DH-SNN模型的便捷函数"""
    return DH_SNN_Network(
        input_dim=input_dim,
        hidden_dims=hidden_dims,
        output_dim=output_dim,
        num_branches=num_branches,
        use_dendritic=True,
        step_mode='s'
    )


def create_vanilla_snn_model(input_dim=2, hidden_dims=[200], output_dim=1):
    """创建原版SNN模型的便捷函数"""
    return DH_SNN_Network(
        input_dim=input_dim,
        hidden_dims=hidden_dims,
        output_dim=output_dim,
        use_dendritic=False,
        step_mode='s'
    )


if __name__ == "__main__":
    # 测试模型
    model = create_dh_snn_model(input_dim=2, hidden_dims=[200], output_dim=1)
    
    # 创建测试输入
    batch_size = 32
    seq_len = 100
    input_dim = 2
    
    # 重置网络状态
    functional.reset_net(model)
    
    # 测试单步模式
    x = torch.randn(batch_size, input_dim)
    output = model(x)
    print(f"Single step output shape: {output.shape}")
    
    # 测试多步模式
    model.step_mode = 'm'
    for layer in model.modules():
        if hasattr(layer, 'step_mode'):
            layer.step_mode = 'm'
    
    x_seq = torch.randn(seq_len, batch_size, input_dim)
    functional.reset_net(model)
    output_seq = model(x_seq)
    print(f"Multi step output shape: {output_seq.shape}")
