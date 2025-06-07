#!/usr/bin/env python3
"""
多时间尺度XOR数据生成器
实现Figure 4a中描述的多时间尺度脉冲XOR问题
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List, Optional

class MultiTimescaleXORGenerator:
    """
    多时间尺度XOR数据生成器
    
    根据论文Figure 4a的描述：
    1. Signal 1: 低频信号，单个脉冲模式，低/高发放率
    2. Signal 2: 高频信号，多个快速脉冲模式序列
    3. 目标: 记住Signal 1并与每个Signal 2进行XOR运算
    """
    
    def __init__(self, 
                 dt: float = 1.0,
                 total_time: int = 1000,
                 signal1_duration: int = 100,
                 signal2_duration: int = 50,
                 signal2_interval: int = 100,
                 num_signal2: int = 5,
                 input_size: int = 100,
                 device: str = 'cpu'):
        """
        初始化多时间尺度XOR生成器
        
        Args:
            dt: 时间步长
            total_time: 总时间长度
            signal1_duration: Signal 1持续时间
            signal2_duration: Signal 2持续时间
            signal2_interval: Signal 2之间的间隔
            num_signal2: Signal 2的数量
            input_size: 输入维度
            device: 设备
        """
        self.dt = dt
        self.total_time = total_time
        self.signal1_duration = signal1_duration
        self.signal2_duration = signal2_duration
        self.signal2_interval = signal2_interval
        self.num_signal2 = num_signal2
        self.input_size = input_size
        self.device = device
        
        # 计算时间点
        self.signal1_start = 50
        self.signal1_end = self.signal1_start + signal1_duration
        
        self.signal2_starts = []
        self.signal2_ends = []
        for i in range(num_signal2):
            start = self.signal1_end + 100 + i * (signal2_duration + signal2_interval)
            end = start + signal2_duration
            self.signal2_starts.append(start)
            self.signal2_ends.append(end)
    
    def generate_signal_pattern(self, 
                              firing_rate: str, 
                              duration: int, 
                              pattern_type: str = 'random') -> torch.Tensor:
        """
        生成脉冲模式
        
        Args:
            firing_rate: 'low' 或 'high'
            duration: 持续时间
            pattern_type: 模式类型
            
        Returns:
            脉冲模式张量 [duration, input_size]
        """
        if firing_rate == 'low':
            prob = 0.1  # 低发放率
        elif firing_rate == 'high':
            prob = 0.3  # 高发放率
        else:
            raise ValueError(f"Unknown firing rate: {firing_rate}")
        
        # 生成随机脉冲模式
        pattern = torch.rand(duration, self.input_size) < prob
        return pattern.float()
    
    def generate_sample(self, 
                       signal1_type: str, 
                       signal2_types: List[str]) -> Tuple[torch.Tensor, torch.Tensor, List[int]]:
        """
        生成单个多时间尺度XOR样本
        
        Args:
            signal1_type: Signal 1类型 ('low' 或 'high')
            signal2_types: Signal 2类型列表
            
        Returns:
            input_data: 输入数据 [total_time, input_size]
            target_data: 目标数据 [total_time, 1]
            xor_results: XOR结果列表
        """
        # 初始化输入和目标
        input_data = torch.zeros(self.total_time, self.input_size)
        target_data = torch.zeros(self.total_time, 1)
        
        # 生成Signal 1
        signal1_pattern = self.generate_signal_pattern(signal1_type, self.signal1_duration)
        input_data[self.signal1_start:self.signal1_end, :] = signal1_pattern
        
        # 生成Signal 2序列和XOR结果
        xor_results = []
        signal1_value = 1 if signal1_type == 'high' else 0
        
        for i, signal2_type in enumerate(signal2_types):
            # 生成Signal 2
            signal2_pattern = self.generate_signal_pattern(signal2_type, self.signal2_duration)
            start_idx = self.signal2_starts[i]
            end_idx = self.signal2_ends[i]
            input_data[start_idx:end_idx, :] = signal2_pattern
            
            # 计算XOR结果
            signal2_value = 1 if signal2_type == 'high' else 0
            xor_result = signal1_value ^ signal2_value
            xor_results.append(xor_result)
            
            # 设置目标输出（在Signal 2结束后的一段时间内）
            output_start = end_idx
            output_end = min(output_start + 20, self.total_time)
            target_data[output_start:output_end, 0] = xor_result
        
        return input_data.to(self.device), target_data.to(self.device), xor_results
    
    def generate_dataset(self, 
                        num_samples: int, 
                        split_by_branch: bool = True) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        生成多时间尺度XOR数据集
        
        Args:
            num_samples: 样本数量
            split_by_branch: 是否按分支分割输入
            
        Returns:
            input_data: 输入数据 [num_samples, total_time, input_size]
            target_data: 目标数据 [num_samples, total_time, 1]
            branch1_data: 分支1数据（如果split_by_branch=True）
            branch2_data: 分支2数据（如果split_by_branch=True）
        """
        all_inputs = []
        all_targets = []
        all_branch1 = []
        all_branch2 = []
        
        for _ in range(num_samples):
            # 随机选择Signal 1类型
            signal1_type = np.random.choice(['low', 'high'])
            
            # 随机选择Signal 2类型序列
            signal2_types = [np.random.choice(['low', 'high']) for _ in range(self.num_signal2)]
            
            # 生成样本
            input_data, target_data, _ = self.generate_sample(signal1_type, signal2_types)
            
            all_inputs.append(input_data)
            all_targets.append(target_data)
            
            if split_by_branch:
                # 分离Signal 1和Signal 2到不同分支
                branch1_data = torch.zeros_like(input_data)
                branch2_data = torch.zeros_like(input_data)
                
                # Branch 1: Signal 1
                branch1_data[self.signal1_start:self.signal1_end, :] = \
                    input_data[self.signal1_start:self.signal1_end, :]
                
                # Branch 2: Signal 2序列
                for i in range(self.num_signal2):
                    start_idx = self.signal2_starts[i]
                    end_idx = self.signal2_ends[i]
                    branch2_data[start_idx:end_idx, :] = input_data[start_idx:end_idx, :]
                
                all_branch1.append(branch1_data)
                all_branch2.append(branch2_data)
        
        # 转换为张量
        input_tensor = torch.stack(all_inputs)
        target_tensor = torch.stack(all_targets)
        
        if split_by_branch:
            branch1_tensor = torch.stack(all_branch1)
            branch2_tensor = torch.stack(all_branch2)
            return input_tensor, target_tensor, branch1_tensor, branch2_tensor
        else:
            return input_tensor, target_tensor, None, None
    
    def visualize_sample(self, 
                        signal1_type: str = 'low', 
                        signal2_types: List[str] = None,
                        save_path: Optional[str] = None):
        """
        可视化多时间尺度XOR样本
        
        Args:
            signal1_type: Signal 1类型
            signal2_types: Signal 2类型列表
            save_path: 保存路径
        """
        if signal2_types is None:
            signal2_types = ['high', 'low', 'high', 'low', 'high']
        
        # 生成样本
        input_data, target_data, xor_results = self.generate_sample(signal1_type, signal2_types)
        
        # 创建图表
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 8))
        
        # 绘制输入脉冲
        time_axis = np.arange(self.total_time) * self.dt
        spike_raster = input_data.cpu().numpy()
        
        # 只显示部分神经元的脉冲
        neurons_to_show = min(20, self.input_size)
        for i in range(neurons_to_show):
            spike_times = time_axis[spike_raster[:, i] > 0]
            ax1.scatter(spike_times, [i] * len(spike_times), s=1, c='black', alpha=0.7)
        
        ax1.set_ylabel('Neuron ID')
        ax1.set_title('Input Spike Patterns')
        ax1.set_xlim(0, self.total_time * self.dt)
        
        # 标记Signal 1和Signal 2区域
        ax1.axvspan(self.signal1_start * self.dt, self.signal1_end * self.dt, 
                   alpha=0.3, color='blue', label='Signal 1')
        for i in range(self.num_signal2):
            ax1.axvspan(self.signal2_starts[i] * self.dt, self.signal2_ends[i] * self.dt,
                       alpha=0.3, color='red', label='Signal 2' if i == 0 else '')
        ax1.legend()
        
        # 绘制目标输出
        target_line = target_data.cpu().numpy().flatten()
        ax2.plot(time_axis, target_line, 'g-', linewidth=2)
        ax2.set_ylabel('Target Output')
        ax2.set_title('Expected XOR Output')
        ax2.set_ylim(-0.1, 1.1)
        
        # 绘制XOR结果
        xor_times = []
        xor_values = []
        for i, xor_result in enumerate(xor_results):
            xor_times.append(self.signal2_ends[i] * self.dt)
            xor_values.append(xor_result)
        
        ax3.stem(xor_times, xor_values, basefmt=' ')
        ax3.set_ylabel('XOR Result')
        ax3.set_xlabel('Time (ms)')
        ax3.set_title(f'XOR Results: Signal1={signal1_type}, Signal2={signal2_types}')
        ax3.set_ylim(-0.1, 1.1)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Sample visualization saved to {save_path}")
        
        plt.show()
        
        return fig

# 测试代码
if __name__ == '__main__':
    # 创建生成器
    generator = MultiTimescaleXORGenerator()
    
    # 生成并可视化样本
    generator.visualize_sample(
        signal1_type='low',
        signal2_types=['high', 'low', 'high', 'low', 'high'],
        save_path='multi_timescale_xor_sample.png'
    )
    
    # 生成数据集
    input_data, target_data, branch1_data, branch2_data = generator.generate_dataset(
        num_samples=100, split_by_branch=True
    )
    
    print(f"Generated dataset:")
    print(f"  Input shape: {input_data.shape}")
    print(f"  Target shape: {target_data.shape}")
    print(f"  Branch1 shape: {branch1_data.shape}")
    print(f"  Branch2 shape: {branch2_data.shape}")
