#!/usr/bin/env python3
"""
数据分析脚本 - 分析多时间尺度XOR数据的特性
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from main_experiment import MultiTimescaleXORGenerator

def analyze_data_patterns():
    """分析数据模式"""
    print("📊 分析多时间尺度XOR数据模式...")
    
    # 生成数据
    generator = MultiTimescaleXORGenerator()
    
    # 生成几个样本进行分析
    samples = []
    targets = []
    for i in range(10):
        input_data, target_data = generator.generate_sample()
        samples.append(input_data)
        targets.append(target_data)
    
    # 分析第一个样本
    sample_input = samples[0]  # [600, 40]
    sample_target = targets[0]  # [600, 1]
    
    print(f"输入数据形状: {sample_input.shape}")
    print(f"目标数据形状: {sample_target.shape}")
    
    # 分析Signal 1和Signal 2的发放率
    signal1_data = sample_input[50:150, :20]  # Signal 1区域
    signal2_regions = [
        sample_input[200:230, 20:],  # Signal 2-1
        sample_input[280:310, 20:],  # Signal 2-2
        sample_input[360:390, 20:],  # Signal 2-3
        sample_input[440:470, 20:],  # Signal 2-4
        sample_input[520:550, 20:],  # Signal 2-5
    ]
    
    print(f"\nSignal 1发放率: {signal1_data.mean():.4f}")
    for i, signal2 in enumerate(signal2_regions):
        if signal2.numel() > 0:
            print(f"Signal 2-{i+1}发放率: {signal2.mean():.4f}")
    
    # 分析目标分布
    target_nonzero = sample_target[sample_target > 0]
    print(f"\n目标值分布:")
    print(f"  总时间步: {sample_target.shape[0]}")
    print(f"  有目标的时间步: {(sample_target > 0).sum().item()}")
    print(f"  目标值为1的比例: {(target_nonzero == 1).float().mean():.3f}")
    
    return sample_input, sample_target

def visualize_sample():
    """可视化样本数据"""
    print("\n🎨 可视化数据样本...")
    
    generator = MultiTimescaleXORGenerator()
    input_data, target_data = generator.generate_sample()
    
    # 创建图形
    fig, axes = plt.subplots(3, 1, figsize=(15, 10))
    
    # 1. 输入脉冲模式
    axes[0].imshow(input_data.T, aspect='auto', cmap='Blues', interpolation='nearest')
    axes[0].set_title('输入脉冲模式 (上半部分: Signal 1, 下半部分: Signal 2)', fontweight='bold')
    axes[0].set_ylabel('输入通道')
    axes[0].axhline(y=19.5, color='red', linestyle='--', alpha=0.7, label='Signal分界线')
    
    # 标记重要时间区域
    axes[0].axvspan(50, 150, alpha=0.2, color='green', label='Signal 1')
    signal2_starts = [200, 280, 360, 440, 520]
    for i, start in enumerate(signal2_starts):
        if start + 30 < input_data.shape[0]:
            axes[0].axvspan(start, start+30, alpha=0.2, color='orange', 
                          label='Signal 2' if i == 0 else '')
    axes[0].legend()
    
    # 2. 目标输出
    target_line = target_data.squeeze().numpy()
    axes[1].plot(target_line, 'r-', linewidth=2, label='目标输出')
    axes[1].fill_between(range(len(target_line)), target_line, alpha=0.3, color='red')
    axes[1].set_title('目标输出 (XOR结果)', fontweight='bold')
    axes[1].set_ylabel('目标值')
    axes[1].set_ylim(-0.1, 1.1)
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    
    # 3. 发放率分析
    window_size = 20
    firing_rates = []
    time_points = []
    
    for t in range(0, input_data.shape[0] - window_size, 10):
        window_data = input_data[t:t+window_size, :]
        firing_rate = window_data.mean().item()
        firing_rates.append(firing_rate)
        time_points.append(t + window_size // 2)
    
    axes[2].plot(time_points, firing_rates, 'b-', linewidth=2, label='整体发放率')
    
    # 分别计算Signal 1和Signal 2的发放率
    signal1_rates = []
    signal2_rates = []
    
    for t in range(0, input_data.shape[0] - window_size, 10):
        window_data = input_data[t:t+window_size, :]
        signal1_rate = window_data[:, :20].mean().item()
        signal2_rate = window_data[:, 20:].mean().item()
        signal1_rates.append(signal1_rate)
        signal2_rates.append(signal2_rate)
    
    axes[2].plot(time_points, signal1_rates, 'g--', linewidth=2, label='Signal 1发放率')
    axes[2].plot(time_points, signal2_rates, 'orange', linestyle='--', linewidth=2, label='Signal 2发放率')
    
    axes[2].set_title('时间窗口发放率分析', fontweight='bold')
    axes[2].set_xlabel('时间步')
    axes[2].set_ylabel('发放率')
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()
    
    plt.tight_layout()
    plt.savefig('results/data_pattern_analysis.png', dpi=300, bbox_inches='tight')
    print("✅ 数据模式分析图已保存到 results/data_pattern_analysis.png")
    
    return fig

def analyze_task_difficulty():
    """分析任务难度"""
    print("\n🔍 分析任务难度...")
    
    generator = MultiTimescaleXORGenerator()
    
    # 生成多个样本分析
    num_samples = 100
    xor_results = []
    signal1_types = []
    signal2_types = []
    
    for _ in range(num_samples):
        input_data, target_data = generator.generate_sample()
        
        # 分析Signal 1类型
        signal1_region = input_data[50:150, :20]
        signal1_rate = signal1_region.mean().item()
        signal1_type = 1 if signal1_rate > 0.15 else 0  # 阈值判断
        signal1_types.append(signal1_type)
        
        # 分析每个Signal 2和对应的XOR结果
        signal2_starts = [200, 280, 360, 440, 520]
        for start in signal2_starts:
            if start + 30 < input_data.shape[0]:
                signal2_region = input_data[start:start+30, 20:]
                signal2_rate = signal2_region.mean().item()
                signal2_type = 1 if signal2_rate > 0.15 else 0
                signal2_types.append(signal2_type)
                
                # 对应的XOR结果
                response_start = start + 30
                response_end = min(response_start + 20, input_data.shape[0])
                if response_end > response_start:
                    target_value = target_data[response_start:response_end, 0].max().item()
                    expected_xor = signal1_type ^ signal2_type
                    xor_results.append((expected_xor, target_value))
    
    # 统计分析
    print(f"Signal 1类型分布: 0={signal1_types.count(0)}, 1={signal1_types.count(1)}")
    print(f"Signal 2类型分布: 0={signal2_types.count(0)}, 1={signal2_types.count(1)}")
    
    correct_xor = sum(1 for expected, actual in xor_results if abs(expected - actual) < 0.1)
    total_xor = len(xor_results)
    print(f"XOR逻辑正确率: {correct_xor}/{total_xor} = {correct_xor/total_xor*100:.1f}%")
    
    # 分析类别平衡性
    xor_0_count = sum(1 for expected, _ in xor_results if expected == 0)
    xor_1_count = sum(1 for expected, _ in xor_results if expected == 1)
    print(f"XOR结果分布: 0={xor_0_count}, 1={xor_1_count}")
    print(f"类别平衡性: {min(xor_0_count, xor_1_count) / max(xor_0_count, xor_1_count):.3f}")
    
    return {
        'signal1_types': signal1_types,
        'signal2_types': signal2_types,
        'xor_results': xor_results
    }

def main():
    """主函数"""
    print("🔍 多时间尺度XOR数据分析")
    print("="*50)
    
    # 创建结果目录
    import os
    os.makedirs("results", exist_ok=True)
    
    # 分析数据模式
    sample_input, sample_target = analyze_data_patterns()
    
    # 可视化样本
    visualize_sample()
    
    # 分析任务难度
    difficulty_analysis = analyze_task_difficulty()
    
    print(f"\n🎉 数据分析完成!")
    print(f"📁 结果保存在 results/ 目录下")

if __name__ == '__main__':
    main()
