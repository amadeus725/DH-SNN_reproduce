"""
多高斯替代函数实现

基于原论文中的Multi-Gaussian (MG)替代函数，用于脉冲函数的梯度计算。
这是DH-SNN中的关键组件，提供了比标准替代函数更好的梯度特性。
"""

import torch
import torch.nn as nn
import math
from typing import Callable
from spikingjelly.activation_based import surrogate


def gaussian(x: torch.Tensor, mu: float = 0.0, sigma: float = 0.5) -> torch.Tensor:
    """
    高斯函数
    
    Args:
        x: 输入张量
        mu: 均值
        sigma: 标准差
    
    Returns:
        高斯函数值
    """
    return torch.exp(-((x - mu) ** 2) / (2 * sigma ** 2)) / torch.sqrt(2 * torch.tensor(math.pi)) / sigma


class MultiGaussianSurrogate(surrogate.SurrogateFunctionBase):
    """
    多高斯替代函数
    
    基于论文中的MG替代函数实现，使用多个高斯函数的组合来近似脉冲函数的梯度。
    相比单一高斯函数，提供了更好的梯度特性和训练稳定性。
    
    Args:
        alpha: 控制函数宽度的参数 (默认: 0.5)
        sigma: 主高斯函数的标准差 (默认: 0.5) 
        scale: 侧翼高斯函数的尺度因子 (默认: 6.0)
        height: 侧翼高斯函数的高度因子 (默认: 0.15)
        gamma: 梯度缩放因子 (默认: 0.5)
        spiking: 是否启用脉冲模式 (默认: True)
    """
    
    def __init__(self, 
                 alpha: float = 0.5,
                 sigma: float = 0.5, 
                 scale: float = 6.0,
                 height: float = 0.15,
                 gamma: float = 0.5,
                 spiking: bool = True):
        super().__init__(alpha, spiking)
        self.sigma = sigma
        self.scale = scale
        self.height = height
        self.gamma = gamma
    
    @staticmethod
    def spiking_function(x: torch.Tensor, alpha: float) -> torch.Tensor:
        """
        脉冲函数：当输入大于0时输出1，否则输出0
        """
        return (x >= 0.0).to(x)
    
    @staticmethod 
    def primitive_function(x: torch.Tensor, alpha: float) -> torch.Tensor:
        """
        原函数：Heaviside阶跃函数
        """
        return (x >= 0.0).to(x)
    
    def backward(self, grad_output: torch.Tensor, x: torch.Tensor, alpha: float) -> torch.Tensor:
        """
        反向传播：计算多高斯替代梯度
        
        使用三个高斯函数的组合：
        - 中心高斯函数：提供主要的梯度信号
        - 两个侧翼高斯函数：提供更宽的梯度支持
        """
        # 主高斯函数 (中心在0)
        main_gaussian = gaussian(x, mu=0.0, sigma=self.sigma) * (1.0 + self.height)
        
        # 侧翼高斯函数 (中心在±sigma)
        left_gaussian = gaussian(x, mu=-self.sigma, sigma=self.scale * self.sigma) * self.height
        right_gaussian = gaussian(x, mu=self.sigma, sigma=self.scale * self.sigma) * self.height
        
        # 组合多高斯函数
        surrogate_grad = main_gaussian - left_gaussian - right_gaussian
        
        return grad_output * surrogate_grad * self.gamma


class AdaptiveMultiGaussianSurrogate(MultiGaussianSurrogate):
    """
    自适应多高斯替代函数
    
    在训练过程中自动调整参数，提供更好的训练动态。
    """
    
    def __init__(self, 
                 alpha: float = 0.5,
                 sigma: float = 0.5,
                 scale: float = 6.0, 
                 height: float = 0.15,
                 gamma: float = 0.5,
                 adaptive: bool = True,
                 spiking: bool = True):
        super().__init__(alpha, sigma, scale, height, gamma, spiking)
        self.adaptive = adaptive
        self.step_count = 0
        
    def backward(self, grad_output: torch.Tensor, x: torch.Tensor, alpha: float) -> torch.Tensor:
        """
        自适应反向传播
        """
        if self.adaptive and self.training:
            # 根据训练步数调整参数
            self.step_count += 1
            adaptive_gamma = self.gamma * (1.0 + 0.1 * torch.cos(torch.tensor(self.step_count / 1000.0)))
        else:
            adaptive_gamma = self.gamma
            
        # 计算多高斯梯度
        main_gaussian = gaussian(x, mu=0.0, sigma=self.sigma) * (1.0 + self.height)
        left_gaussian = gaussian(x, mu=-self.sigma, sigma=self.scale * self.sigma) * self.height  
        right_gaussian = gaussian(x, mu=self.sigma, sigma=self.scale * self.sigma) * self.height
        
        surrogate_grad = main_gaussian - left_gaussian - right_gaussian
        
        return grad_output * surrogate_grad * adaptive_gamma


# 便捷的函数接口
def multi_gaussian_surrogate(alpha: float = 0.5, 
                           sigma: float = 0.5,
                           scale: float = 6.0, 
                           height: float = 0.15,
                           gamma: float = 0.5) -> Callable:
    """
    创建多高斯替代函数
    
    Returns:
        多高斯替代函数实例
    """
    return MultiGaussianSurrogate(alpha, sigma, scale, height, gamma)


def adaptive_multi_gaussian_surrogate(alpha: float = 0.5,
                                    sigma: float = 0.5, 
                                    scale: float = 6.0,
                                    height: float = 0.15,
                                    gamma: float = 0.5) -> Callable:
    """
    创建自适应多高斯替代函数
    
    Returns:
        自适应多高斯替代函数实例
    """
    return AdaptiveMultiGaussianSurrogate(alpha, sigma, scale, height, gamma, adaptive=True)


# 默认替代函数实例
default_multi_gaussian = MultiGaussianSurrogate()
default_adaptive_multi_gaussian = AdaptiveMultiGaussianSurrogate()
