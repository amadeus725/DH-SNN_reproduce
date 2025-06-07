#!/usr/bin/env python3
"""
更具挑战性的多时间尺度XOR数据生成器
基于原论文的精确实现，增加任务难度
"""

import torch
import numpy as np
import matplotlib.pyplot as plt

class ChallengingMultiTimescaleXORGenerator:
    """更具挑战性的多时间尺度XOR数据生成器"""

    def __init__(self, device='cpu'):
        self.device = device

        # 基于原论文参数
        self.time_steps = 100  # 总时间步
        self.channel_size = 20  # 每个信号的通道数
        self.input_size = self.channel_size * 2  # 40个输入通道

        # 时间参数
        self.coding_time = 10   # 信号编码时间
        self.remain_time = 5    # 信号间隔时间
        self.start_time = 10    # 开始时间

        # 发放率设置 - 更接近的发放率增加难度
        self.channel_rates = [0.2, 0.6]  # 低发放率和高发放率
        self.noise_rate = 0.01  # 背景噪声

        # XOR标签
        self.label = torch.zeros(len(self.channel_rates), len(self.channel_rates))
        self.label[1][0] = 1  # 高-低 = 1
        self.label[0][1] = 1  # 低-高 = 1

    def generate_sample(self, batch_size=1):
        """生成单个或批量样本"""

        # 初始化数据
        input_data = torch.zeros(batch_size, self.time_steps, self.input_size)
        target_data = torch.zeros(batch_size, self.time_steps, dtype=torch.long)

        # 添加背景噪声
        noise_mask = torch.rand(batch_size, self.time_steps, self.input_size) <= self.noise_rate
        input_data[noise_mask] = 1.0

        # 生成Signal 1 (长期记忆信号)
        signal1_patterns = torch.randint(len(self.channel_rates), size=(batch_size,))

        for b in range(batch_size):
            pattern_idx = signal1_patterns[b].item()
            prob = self.channel_rates[pattern_idx]

            # 生成Signal 1的脉冲
            prob_matrix = torch.ones(self.start_time, self.channel_size) * prob
            spikes = torch.bernoulli(prob_matrix)
            input_data[b, :self.start_time, :self.channel_size] = torch.maximum(
                input_data[b, :self.start_time, :self.channel_size], spikes)

        # 生成Signal 2序列 (短期快速信号)
        num_signal2 = (self.time_steps - self.start_time) // (self.coding_time + self.remain_time)

        for i in range(num_signal2):
            signal2_patterns = torch.randint(len(self.channel_rates), size=(batch_size,))

            for b in range(batch_size):
                signal1_idx = signal1_patterns[b].item()
                signal2_idx = signal2_patterns[b].item()

                # 计算XOR结果
                xor_result = self.label[signal1_idx, signal2_idx].int().item()

                # Signal 2的时间窗口
                start_idx = self.start_time + i * (self.coding_time + self.remain_time) + self.remain_time
                end_idx = start_idx + self.coding_time

                if end_idx <= self.time_steps:
                    # 生成Signal 2的脉冲
                    prob = self.channel_rates[signal2_idx]
                    prob_matrix = torch.ones(self.coding_time, self.channel_size) * prob
                    spikes = torch.bernoulli(prob_matrix)
                    input_data[b, start_idx:end_idx, self.channel_size:] = torch.maximum(
                        input_data[b, start_idx:end_idx, self.channel_size:], spikes)

                    # 设置目标标签
                    target_start = self.start_time + i * (self.coding_time + self.remain_time)
                    target_end = target_start + (self.coding_time + self.remain_time)
                    if target_end <= self.time_steps:
                        target_data[b, target_start:target_end] = xor_result

        return input_data.to(self.device), target_data.to(self.device)

    def generate_dataset(self, num_samples, batch_size=32):
        """生成数据集"""
        all_inputs = []
        all_targets = []

        num_batches = (num_samples + batch_size - 1) // batch_size

        for _ in range(num_batches):
            current_batch_size = min(batch_size, num_samples - len(all_inputs) * batch_size)
            if current_batch_size <= 0:
                break

            batch_inputs, batch_targets = self.generate_sample(current_batch_size)
            all_inputs.append(batch_inputs)
            all_targets.append(batch_targets)

        # 合并所有批次
        inputs = torch.cat(all_inputs, dim=0)[:num_samples]
        targets = torch.cat(all_targets, dim=0)[:num_samples]

        return inputs, targets

class AdaptiveDifficultyGenerator:
    """自适应难度的多时间尺度XOR生成器"""

    def __init__(self, device='cpu', difficulty_level='medium'):
        self.device = device
        self.difficulty_level = difficulty_level

        # 根据难度级别设置参数
        if difficulty_level == 'easy':
            self.low_rate = 0.1
            self.high_rate = 0.5
            self.noise_rate = 0.01
            self.signal_duration_var = 0.1
        elif difficulty_level == 'medium':
            self.low_rate = 0.15
            self.high_rate = 0.35
            self.noise_rate = 0.02
            self.signal_duration_var = 0.2
        else:  # hard
            self.low_rate = 0.2
            self.high_rate = 0.3
            self.noise_rate = 0.03
            self.signal_duration_var = 0.3

        # 基本参数
        self.total_time = 400
        self.input_size = 40
        self.signal1_base_duration = 80
        self.signal2_base_duration = 20
        self.num_signal2 = 4

    def generate_sample(self):
        """生成自适应难度的样本"""
        input_data = torch.zeros(self.total_time, self.input_size)
        target_data = torch.zeros(self.total_time, 1)

        # 添加背景噪声
        noise_mask = torch.rand(self.total_time, self.input_size) < self.noise_rate
        input_data[noise_mask] = 1.0

        # Signal 1: 可变长度的长期信号
        signal1_duration = int(self.signal1_base_duration * (1 + np.random.uniform(-self.signal_duration_var, self.signal_duration_var)))
        signal1_start = 20
        signal1_end = signal1_start + signal1_duration

        signal1_type = np.random.choice([0, 1])
        signal1_rate = self.low_rate if signal1_type == 0 else self.high_rate

        # 添加时间变化的发放率
        for t in range(signal1_start, min(signal1_end, self.total_time)):
            # 发放率随时间略有变化
            time_factor = 1 + 0.1 * np.sin(2 * np.pi * (t - signal1_start) / 20)
            current_rate = signal1_rate * time_factor
            spike_mask = torch.rand(20) < current_rate
            input_data[t, :20][spike_mask] = 1.0

        # Signal 2序列: 可变时间间隔
        base_interval = (self.total_time - signal1_end - 50) // self.num_signal2

        for i in range(self.num_signal2):
            # 可变的开始时间
            interval_var = int(base_interval * 0.2 * np.random.uniform(-1, 1))
            signal2_start = signal1_end + 20 + i * base_interval + interval_var

            signal2_duration = int(self.signal2_base_duration * (1 + np.random.uniform(-0.3, 0.3)))
            signal2_end = signal2_start + signal2_duration

            if signal2_end >= self.total_time:
                break

            signal2_type = np.random.choice([0, 1])
            signal2_rate = self.low_rate if signal2_type == 0 else self.high_rate

            # 生成Signal 2
            for t in range(signal2_start, signal2_end):
                spike_mask = torch.rand(20) < signal2_rate
                input_data[t, 20:][spike_mask] = 1.0

            # XOR结果
            xor_result = signal1_type ^ signal2_type
            response_start = signal2_end
            response_end = min(response_start + 15, self.total_time)
            target_data[response_start:response_end, 0] = float(xor_result)

        return input_data.to(self.device), target_data.to(self.device)

    def generate_dataset(self, num_samples):
        """生成数据集"""
        inputs, targets = [], []

        for _ in range(num_samples):
            input_data, target_data = self.generate_sample()
            inputs.append(input_data)
            targets.append(target_data)

        return torch.stack(inputs), torch.stack(targets)

def test_generators():
    """测试不同的生成器"""
    print("🧪 测试不同难度的数据生成器...")

    # 测试原始挑战性生成器
    print("\n1. 原始挑战性生成器:")
    gen1 = ChallengingMultiTimescaleXORGenerator()
    inputs1, targets1 = gen1.generate_sample(batch_size=5)
    print(f"   输入形状: {inputs1.shape}")
    print(f"   目标形状: {targets1.shape}")
    print(f"   平均发放率: {inputs1.mean():.4f}")

    # 测试自适应难度生成器
    for difficulty in ['easy', 'medium', 'hard']:
        print(f"\n2. 自适应难度生成器 ({difficulty}):")
        gen2 = AdaptiveDifficultyGenerator(difficulty_level=difficulty)
        inputs2, targets2 = gen2.generate_dataset(5)
        print(f"   输入形状: {inputs2.shape}")
        print(f"   目标形状: {targets2.shape}")
        print(f"   平均发放率: {inputs2.mean():.4f}")
        print(f"   目标覆盖率: {(targets2 > 0).float().mean():.4f}")

if __name__ == '__main__':
    test_generators()
