#!/usr/bin/env python3
"""
最终挑战性多时间尺度XOR实验
基于原论文精确参数，设计更具挑战性的任务
"""

import torch
import torch.nn as nn
import numpy as np
import os
from challenging_data_generator import ChallengingMultiTimescaleXORGenerator

class UltraChallengingXORGenerator:
    """超级挑战性XOR生成器 - 基于原论文参数"""
    
    def __init__(self, device='cpu'):
        self.device = device
        
        # 原论文精确参数
        self.time_steps = 100
        self.channel_size = 20
        self.coding_time = 10
        self.remain_time = 5
        self.start_time = 10
        
        # 更具挑战性的发放率设置
        self.channel_rates = [0.2, 0.6]  # 保持原论文设置
        self.noise_rate = 0.01
        
        # 增加任务复杂度的参数
        self.interference_rate = 0.05  # 干扰信号
        self.rate_jitter = 0.1  # 发放率抖动
        self.temporal_jitter = 2  # 时间抖动
        
    def generate_sample(self, batch_size=1):
        """生成超级挑战性样本"""
        input_data = torch.zeros(batch_size, self.time_steps, self.channel_size * 2)
        target_data = torch.zeros(batch_size, self.time_steps, dtype=torch.long)
        
        # 添加背景噪声
        noise_mask = torch.rand(batch_size, self.time_steps, self.channel_size * 2) <= self.noise_rate
        input_data[noise_mask] = 1.0
        
        for b in range(batch_size):
            # Signal 1生成 (带时间抖动)
            signal1_pattern = np.random.randint(len(self.channel_rates))
            signal1_start = self.start_time + np.random.randint(-self.temporal_jitter, self.temporal_jitter + 1)
            signal1_start = max(0, signal1_start)
            
            # 添加发放率抖动
            base_rate = self.channel_rates[signal1_pattern]
            actual_rate = base_rate + np.random.uniform(-self.rate_jitter, self.rate_jitter)
            actual_rate = np.clip(actual_rate, 0.05, 0.95)
            
            # 生成Signal 1
            for t in range(signal1_start, min(signal1_start + self.start_time, self.time_steps)):
                spikes = torch.rand(self.channel_size) < actual_rate
                input_data[b, t, :self.channel_size] = torch.maximum(
                    input_data[b, t, :self.channel_size], spikes.float())
            
            # 添加干扰信号 (在Signal 1和Signal 2之间)
            interference_start = signal1_start + self.start_time + 2
            interference_end = min(interference_start + 3, self.time_steps)
            for t in range(interference_start, interference_end):
                interference_spikes = torch.rand(self.channel_size * 2) < self.interference_rate
                input_data[b, t, :] = torch.maximum(
                    input_data[b, t, :], interference_spikes.float())
            
            # Signal 2序列生成
            num_signal2 = (self.time_steps - self.start_time) // (self.coding_time + self.remain_time)
            
            for i in range(num_signal2):
                signal2_pattern = np.random.randint(len(self.channel_rates))
                
                # 时间窗口计算 (带抖动)
                base_start = self.start_time + i * (self.coding_time + self.remain_time) + self.remain_time
                signal2_start = base_start + np.random.randint(-self.temporal_jitter, self.temporal_jitter + 1)
                signal2_start = max(base_start - 2, signal2_start)
                signal2_end = signal2_start + self.coding_time
                
                if signal2_end >= self.time_steps:
                    break
                
                # 发放率抖动
                base_rate = self.channel_rates[signal2_pattern]
                actual_rate = base_rate + np.random.uniform(-self.rate_jitter, self.rate_jitter)
                actual_rate = np.clip(actual_rate, 0.05, 0.95)
                
                # 生成Signal 2
                for t in range(signal2_start, signal2_end):
                    spikes = torch.rand(self.channel_size) < actual_rate
                    input_data[b, t, self.channel_size:] = torch.maximum(
                        input_data[b, t, self.channel_size:], spikes.float())
                
                # XOR标签 (只在部分时间窗口有标签，增加难度)
                xor_result = 1 if signal1_pattern != signal2_pattern else 0
                
                # 随机决定是否在这个窗口提供标签 (增加稀疏性)
                if np.random.random() > 0.3:  # 70%概率有标签
                    target_start = self.start_time + i * (self.coding_time + self.remain_time)
                    target_end = target_start + (self.coding_time + self.remain_time)
                    target_end = min(target_end, self.time_steps)
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
        
        inputs = torch.cat(all_inputs, dim=0)[:num_samples]
        targets = torch.cat(all_targets, dim=0)[:num_samples]
        
        return inputs, targets

def test_task_difficulty():
    """测试任务难度"""
    print("🧪 测试超级挑战性任务难度...")
    
    generator = UltraChallengingXORGenerator()
    
    # 生成测试数据
    test_inputs, test_targets = generator.generate_dataset(50)
    
    print(f"数据形状: {test_inputs.shape}")
    print(f"平均发放率: {test_inputs.mean():.4f}")
    print(f"目标覆盖率: {(test_targets > 0).float().mean():.4f}")
    
    # 分析Signal 1和Signal 2的可区分性
    signal1_rates = []
    signal2_rates = []
    
    for i in range(10):  # 分析前10个样本
        sample = test_inputs[i]
        
        # Signal 1区域 (时间步8-18左右)
        signal1_region = sample[8:18, :20]
        signal1_rate = signal1_region.mean().item()
        signal1_rates.append(signal1_rate)
        
        # Signal 2区域 (时间步25-35左右)
        if sample.shape[0] > 35:
            signal2_region = sample[25:35, 20:]
            signal2_rate = signal2_region.mean().item()
            signal2_rates.append(signal2_rate)
    
    print(f"Signal 1发放率范围: {min(signal1_rates):.3f} - {max(signal1_rates):.3f}")
    print(f"Signal 2发放率范围: {min(signal2_rates):.3f} - {max(signal2_rates):.3f}")
    print(f"发放率重叠度: {len([r for r in signal1_rates if min(signal2_rates) <= r <= max(signal2_rates)])/len(signal1_rates)*100:.1f}%")
    
    # 简单基线测试
    print("\n🤖 简单基线模型测试...")
    
    # 基于发放率的简单分类器
    correct_predictions = 0
    total_predictions = 0
    
    for i in range(len(test_inputs)):
        sample = test_inputs[i]
        target = test_targets[i]
        
        # 找到有标签的时间步
        labeled_steps = torch.where(target >= 0)[0]
        
        if len(labeled_steps) > 0:
            # 简单策略：比较Signal 1和Signal 2的平均发放率
            signal1_rate = sample[8:18, :20].mean().item()
            signal2_rate = sample[25:35, 20:].mean().item() if sample.shape[0] > 35 else 0
            
            # 简单XOR预测
            signal1_high = signal1_rate > 0.4
            signal2_high = signal2_rate > 0.4
            predicted_xor = int(signal1_high != signal2_high)
            
            actual_xor = target[labeled_steps[0]].item()
            
            if predicted_xor == actual_xor:
                correct_predictions += 1
            total_predictions += 1
    
    baseline_acc = correct_predictions / total_predictions * 100 if total_predictions > 0 else 0
    print(f"基线准确率: {baseline_acc:.1f}%")
    
    if baseline_acc > 80:
        print("⚠️  任务仍然过于简单，需要进一步增加难度")
    elif baseline_acc < 60:
        print("✅ 任务难度适中，适合区分不同模型性能")
    else:
        print("🔄 任务难度中等，可能需要微调")
    
    return baseline_acc

def create_final_experiment_config():
    """创建最终实验配置建议"""
    
    config = {
        "data_generation": {
            "generator_class": "UltraChallengingXORGenerator",
            "parameters": {
                "interference_rate": 0.05,
                "rate_jitter": 0.1,
                "temporal_jitter": 2,
                "label_sparsity": 0.7
            }
        },
        "model_architectures": {
            "vanilla_sfnn": {"tau_init": "medium"},
            "1branch_dh_sfnn_small": {"tau_init": "small"},
            "1branch_dh_sfnn_large": {"tau_init": "large"},
            "2branch_dh_sfnn_beneficial": {"beneficial_init": True, "learnable": True},
            "2branch_dh_sfnn_fixed": {"beneficial_init": True, "learnable": False},
            "2branch_dh_sfnn_random": {"beneficial_init": False, "learnable": True}
        },
        "training": {
            "epochs": 100,
            "learning_rate": 5e-4,
            "batch_size": 16,
            "weight_decay": 1e-5,
            "scheduler": "StepLR",
            "trials_per_model": 5
        },
        "expected_results": {
            "vanilla_sfnn": "60-70%",
            "1branch_small": "65-75%", 
            "1branch_large": "70-80%",
            "2branch_beneficial": "80-90%",
            "2branch_fixed": "75-85%",
            "2branch_random": "70-80%"
        }
    }
    
    return config

def main():
    """主函数"""
    print("🎯 最终挑战性多时间尺度XOR实验设计")
    print("="*60)
    
    # 测试任务难度
    baseline_acc = test_task_difficulty()
    
    # 创建实验配置
    config = create_final_experiment_config()
    
    print(f"\n📋 推荐的实验配置:")
    print(f"基线准确率: {baseline_acc:.1f}%")
    print(f"数据生成器: {config['data_generation']['generator_class']}")
    print(f"训练轮数: {config['training']['epochs']}")
    print(f"每模型试验次数: {config['training']['trials_per_model']}")
    
    print(f"\n🎯 期望结果:")
    for model, expected in config['expected_results'].items():
        print(f"  {model}: {expected}")
    
    # 保存配置
    torch.save(config, "results/final_experiment_config.pth")
    print(f"\n💾 实验配置已保存到: results/final_experiment_config.pth")

if __name__ == '__main__':
    main()
