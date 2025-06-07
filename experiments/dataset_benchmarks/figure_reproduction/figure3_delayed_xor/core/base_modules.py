"""
基础模块定义

当SpikingJelly不可用时，提供基本的内存管理和神经元基类。
"""

import torch
import torch.nn as nn
from typing import Any, Dict, Optional, Union


class MemoryModule(nn.Module):
    """
    内存管理模块
    
    提供状态变量的注册和管理功能，类似于SpikingJelly的MemoryModule。
    """
    
    def __init__(self):
        super().__init__()
        self._memories = {}
        self._memory_shapes = {}
    
    def register_memory(self, name: str, value: Union[float, torch.Tensor]):
        """
        注册内存变量
        
        Args:
            name: 变量名
            value: 初始值
        """
        if isinstance(value, (int, float)):
            self._memories[name] = value
            self._memory_shapes[name] = None
        else:
            self._memories[name] = value.clone() if isinstance(value, torch.Tensor) else value
            self._memory_shapes[name] = value.shape if hasattr(value, 'shape') else None
        
        # 将内存变量设置为模块属性
        setattr(self, name, None)
    
    def reset_memory(self, name: str, batch_size: Optional[int] = None, device: Optional[torch.device] = None):
        """
        重置指定的内存变量
        
        Args:
            name: 变量名
            batch_size: 批次大小
            device: 设备
        """
        if name not in self._memories:
            return
            
        init_value = self._memories[name]
        
        if isinstance(init_value, (int, float)):
            if batch_size is not None and self._memory_shapes[name] is not None:
                shape = (batch_size,) + self._memory_shapes[name]
                setattr(self, name, torch.full(shape, init_value, device=device))
            else:
                setattr(self, name, torch.tensor(init_value, device=device))
        else:
            if batch_size is not None:
                if hasattr(init_value, 'shape'):
                    shape = (batch_size,) + init_value.shape
                    setattr(self, name, torch.zeros(shape, device=device, dtype=init_value.dtype))
                else:
                    setattr(self, name, init_value)
            else:
                setattr(self, name, init_value.clone() if hasattr(init_value, 'clone') else init_value)
    
    def reset_all_memories(self, batch_size: Optional[int] = None, device: Optional[torch.device] = None):
        """
        重置所有内存变量
        
        Args:
            batch_size: 批次大小
            device: 设备
        """
        for name in self._memories:
            self.reset_memory(name, batch_size, device)


class BaseLIFNode(MemoryModule):
    """
    基础LIF神经元
    
    当SpikingJelly不可用时的简化LIF实现。
    """
    
    def __init__(self,
                 tau: float = 2.0,
                 v_threshold: float = 1.0,
                 v_reset: float = 0.0,
                 surrogate_function: Optional[Any] = None,
                 detach_reset: bool = False,
                 step_mode: str = 's',
                 backend: str = 'torch',
                 store_v_seq: bool = False):
        super().__init__()
        
        self.tau = tau
        self.v_threshold = v_threshold
        self.v_reset = v_reset
        self.surrogate_function = surrogate_function
        self.detach_reset = detach_reset
        self.step_mode = step_mode
        self.backend = backend
        self.store_v_seq = store_v_seq
        
        # 注册内存变量
        self.register_memory('v', v_reset)
        self.register_memory('spike', 0.0)
        
        if store_v_seq:
            self.register_memory('v_seq', [])
    
    def neuronal_charge(self, x: torch.Tensor) -> torch.Tensor:
        """
        神经元充电过程
        """
        if self.v is None:
            self.reset_all_memories(x.size(0), x.device)
            
        # 简化的LIF动态
        alpha = 1.0 / self.tau if isinstance(self.tau, (int, float)) else torch.sigmoid(self.tau)
        self.v = self.v * (1 - alpha) + alpha * x
        
        return self.v
    
    def neuronal_fire(self) -> torch.Tensor:
        """
        神经元放电过程
        """
        if self.surrogate_function is not None:
            self.spike = self.surrogate_function(self.v - self.v_threshold)
        else:
            self.spike = (self.v >= self.v_threshold).float()
        
        return self.spike
    
    def neuronal_reset(self) -> torch.Tensor:
        """
        神经元重置过程
        """
        if not self.detach_reset:
            self.v = self.v - self.spike * (self.v - self.v_reset)
        else:
            self.v = self.v - self.spike.detach() * (self.v - self.v_reset)
        
        return self.v
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        """
        self.neuronal_charge(x)
        self.neuronal_fire()
        self.neuronal_reset()
        
        if self.store_v_seq:
            if self.v_seq is None:
                self.v_seq = []
            self.v_seq.append(self.v.clone())
        
        return self.spike


# 创建兼容性别名
if 'neuron' not in globals():
    class neuron:
        LIFNode = BaseLIFNode


class StateManager:
    """
    状态管理器
    
    用于管理网络中所有模块的状态变量。
    """
    
    def __init__(self, model: nn.Module):
        self.model = model
        self.memory_modules = []
        self._find_memory_modules()
    
    def _find_memory_modules(self):
        """
        查找所有内存模块
        """
        for module in self.model.modules():
            if isinstance(module, MemoryModule):
                self.memory_modules.append(module)
    
    def reset_all_states(self, batch_size: Optional[int] = None, device: Optional[torch.device] = None):
        """
        重置所有状态
        """
        for module in self.memory_modules:
            module.reset_all_memories(batch_size, device)
    
    def get_state_dict(self) -> Dict[str, Any]:
        """
        获取状态字典
        """
        state_dict = {}
        for i, module in enumerate(self.memory_modules):
            module_states = {}
            for name in module._memories:
                value = getattr(module, name)
                if value is not None:
                    module_states[name] = value.clone() if hasattr(value, 'clone') else value
            state_dict[f'module_{i}'] = module_states
        return state_dict
    
    def load_state_dict(self, state_dict: Dict[str, Any]):
        """
        加载状态字典
        """
        for i, module in enumerate(self.memory_modules):
            if f'module_{i}' in state_dict:
                module_states = state_dict[f'module_{i}']
                for name, value in module_states.items():
                    if hasattr(module, name):
                        setattr(module, name, value)
