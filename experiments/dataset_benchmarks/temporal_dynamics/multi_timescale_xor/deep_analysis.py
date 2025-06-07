#!/usr/bin/env python3
"""
深入分析实验结果
理解为什么随机初始化比有益初始化更好
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from fixed_experiment import FixedXORGenerator, FixedTwoBranchDH_SFNN

print("🔍 深入分析多时间尺度XOR实验结果")
print("="*60)

def analyze_time_constants():
    """分析不同初始化的时间常数分布"""
    print("\n📊 分析时间常数分布...")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 创建不同初始化的模型
    models = {
        'Beneficial': FixedTwoBranchDH_SFNN(beneficial_init=True, learnable=False),
        'Random': FixedTwoBranchDH_SFNN(beneficial_init=False, learnable=False)
    }
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle('时间常数初始化分布对比', fontsize=16, fontweight='bold')
    
    for i, (name, model) in enumerate(models.items()):
        # 获取时间常数
        tau_m = torch.sigmoid(model.tau_m).detach().cpu().numpy()
        tau_n1 = torch.sigmoid(model.tau_n_branch1).detach().cpu().numpy()
        tau_n2 = torch.sigmoid(model.tau_n_branch2).detach().cpu().numpy()
        
        print(f"\n{name}初始化:")
        print(f"  膜电位τ_m: {tau_m.mean():.3f} ± {tau_m.std():.3f}")
        print(f"  分支1 τ_n1: {tau_n1.mean():.3f} ± {tau_n1.std():.3f}")
        print(f"  分支2 τ_n2: {tau_n2.mean():.3f} ± {tau_n2.std():.3f}")
        
        # 绘制分布
        axes[i, 0].hist(tau_n1, bins=20, alpha=0.7, color='blue', label='Branch 1')
        axes[i, 0].hist(tau_n2, bins=20, alpha=0.7, color='red', label='Branch 2')
        axes[i, 0].set_title(f'{name} - 树突时间常数')
        axes[i, 0].set_xlabel('时间常数值')
        axes[i, 0].set_ylabel('频次')
        axes[i, 0].legend()
        axes[i, 0].grid(True, alpha=0.3)
        
        # 绘制膜电位时间常数
        axes[i, 1].hist(tau_m, bins=20, alpha=0.7, color='green')
        axes[i, 1].set_title(f'{name} - 膜电位时间常数')
        axes[i, 1].set_xlabel('时间常数值')
        axes[i, 1].set_ylabel('频次')
        axes[i, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/time_constant_analysis.png', dpi=300, bbox_inches='tight')
    print("✅ 时间常数分析图已保存")
    
    return models

def analyze_data_patterns():
    """分析数据模式，理解任务特性"""
    print("\n📊 分析数据模式...")
    
    generator = FixedXORGenerator()
    
    # 生成多个样本分析
    samples = []
    targets = []
    signal1_types = []
    signal2_types = []
    xor_results = []
    
    for _ in range(100):
        input_data, target_data = generator.generate_sample()
        samples.append(input_data)
        targets.append(target_data)
        
        # 分析Signal 1类型
        signal1_region = input_data[10:20, :20]
        signal1_rate = signal1_region.mean().item()
        signal1_type = 1 if signal1_rate > 0.4 else 0
        signal1_types.append(signal1_type)
        
        # 分析Signal 2类型 (取第一个Signal 2)
        signal2_region = input_data[30:38, 20:]
        signal2_rate = signal2_region.mean().item()
        signal2_type = 1 if signal2_rate > 0.4 else 0
        signal2_types.append(signal2_type)
        
        # XOR结果
        xor_result = signal1_type ^ signal2_type
        xor_results.append(xor_result)
    
    print(f"Signal 1类型分布: 0={signal1_types.count(0)}, 1={signal1_types.count(1)}")
    print(f"Signal 2类型分布: 0={signal2_types.count(0)}, 1={signal2_types.count(1)}")
    print(f"XOR结果分布: 0={xor_results.count(0)}, 1={xor_results.count(1)}")
    
    # 分析信号的时间特性
    sample_input = samples[0]
    sample_target = targets[0]
    
    fig, axes = plt.subplots(3, 1, figsize=(15, 10))
    
    # 1. 输入模式可视化
    axes[0].imshow(sample_input.T, aspect='auto', cmap='Blues', interpolation='nearest')
    axes[0].set_title('输入脉冲模式 (上半部分: Signal 1, 下半部分: Signal 2)', fontweight='bold')
    axes[0].set_ylabel('输入通道')
    axes[0].axhline(y=19.5, color='red', linestyle='--', alpha=0.7, label='Signal分界线')
    
    # 标记重要时间区域
    axes[0].axvspan(10, 20, alpha=0.2, color='green', label='Signal 1')
    axes[0].axvspan(30, 38, alpha=0.2, color='orange', label='Signal 2-1')
    axes[0].axvspan(40, 48, alpha=0.2, color='orange', label='Signal 2-2')
    axes[0].axvspan(50, 58, alpha=0.2, color='orange', label='Signal 2-3')
    axes[0].legend()
    
    # 2. 目标输出
    target_line = sample_target.squeeze().numpy()
    axes[1].plot(target_line, 'r-', linewidth=2, label='目标输出')
    axes[1].fill_between(range(len(target_line)), target_line, alpha=0.3, color='red')
    axes[1].set_title('目标输出 (XOR结果)', fontweight='bold')
    axes[1].set_ylabel('目标值')
    axes[1].set_ylim(-0.1, 1.1)
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    
    # 3. 发放率分析
    signal1_rates = []
    signal2_rates = []
    time_points = []
    
    for t in range(sample_input.shape[0]):
        signal1_rate = sample_input[t, :20].mean().item()
        signal2_rate = sample_input[t, 20:].mean().item()
        signal1_rates.append(signal1_rate)
        signal2_rates.append(signal2_rate)
        time_points.append(t)
    
    axes[2].plot(time_points, signal1_rates, 'g-', linewidth=2, label='Signal 1发放率')
    axes[2].plot(time_points, signal2_rates, 'orange', linewidth=2, label='Signal 2发放率')
    axes[2].set_title('时间序列发放率分析', fontweight='bold')
    axes[2].set_xlabel('时间步')
    axes[2].set_ylabel('发放率')
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()
    
    plt.tight_layout()
    plt.savefig('results/data_pattern_analysis.png', dpi=300, bbox_inches='tight')
    print("✅ 数据模式分析图已保存")

def train_with_monitoring(model, train_data, train_targets, model_name, epochs=50):
    """带监控的训练，记录时间常数变化"""
    print(f"\n🏋️ 监控训练: {model_name}")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
    criterion = nn.BCELoss()
    
    # 记录历史
    history = {
        'losses': [],
        'accuracies': [],
        'tau_m': [],
        'tau_n1': [],
        'tau_n2': []
    }
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        num_batches = 0
        
        batch_size = 16
        for i in range(0, len(train_data), batch_size):
            batch_data = train_data[i:i+batch_size].to(device)
            batch_targets = train_targets[i:i+batch_size].to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_data)
            
            valid_mask = (batch_targets >= 0)
            if valid_mask.sum() > 0:
                valid_outputs = outputs[valid_mask]
                valid_targets = batch_targets[valid_mask]
                
                loss = criterion(valid_outputs, valid_targets)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
        
        # 记录时间常数
        if hasattr(model, 'tau_m'):
            history['tau_m'].append(torch.sigmoid(model.tau_m).detach().cpu().mean().item())
            history['tau_n1'].append(torch.sigmoid(model.tau_n_branch1).detach().cpu().mean().item())
            history['tau_n2'].append(torch.sigmoid(model.tau_n_branch2).detach().cpu().mean().item())
        
        # 计算准确率
        model.eval()
        with torch.no_grad():
            outputs = model(train_data.to(device))
            valid_mask = (train_targets >= 0)
            if valid_mask.sum() > 0:
                valid_outputs = outputs[valid_mask]
                valid_targets = train_targets[valid_mask].to(device)
                pred_binary = (valid_outputs > 0.5).float()
                acc = (pred_binary == valid_targets).float().mean().item() * 100
            else:
                acc = 0
        
        history['losses'].append(total_loss / max(num_batches, 1))
        history['accuracies'].append(acc)
        
        if epoch % 10 == 0:
            print(f"  Epoch {epoch+1:2d}: Loss={history['losses'][-1]:.4f}, Acc={acc:.1f}%")
    
    return history

def compare_training_dynamics():
    """比较不同初始化的训练动态"""
    print("\n🔄 比较训练动态...")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    generator = FixedXORGenerator(device=device)
    train_data, train_targets = generator.generate_dataset(200)
    
    # 训练不同模型
    models = {
        'Beneficial': FixedTwoBranchDH_SFNN(beneficial_init=True, learnable=True),
        'Random': FixedTwoBranchDH_SFNN(beneficial_init=False, learnable=True)
    }
    
    histories = {}
    for name, model in models.items():
        histories[name] = train_with_monitoring(model, train_data, train_targets, name, epochs=50)
    
    # 绘制训练动态
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('训练动态对比', fontsize=16, fontweight='bold')
    
    epochs = range(1, 51)
    
    # 损失对比
    for name, history in histories.items():
        axes[0, 0].plot(epochs, history['losses'], label=name, linewidth=2)
    axes[0, 0].set_title('训练损失')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 准确率对比
    for name, history in histories.items():
        axes[0, 1].plot(epochs, history['accuracies'], label=name, linewidth=2)
    axes[0, 1].set_title('训练准确率')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy (%)')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 时间常数变化 - Branch 1
    for name, history in histories.items():
        if history['tau_n1']:
            axes[1, 0].plot(epochs, history['tau_n1'], label=f'{name} Branch1', linewidth=2)
    axes[1, 0].set_title('Branch 1 时间常数变化')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('τ_n1')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 时间常数变化 - Branch 2
    for name, history in histories.items():
        if history['tau_n2']:
            axes[1, 1].plot(epochs, history['tau_n2'], label=f'{name} Branch2', linewidth=2)
    axes[1, 1].set_title('Branch 2 时间常数变化')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('τ_n2')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/training_dynamics_comparison.png', dpi=300, bbox_inches='tight')
    print("✅ 训练动态对比图已保存")
    
    return histories

def main():
    """主分析函数"""
    import os
    os.makedirs('results', exist_ok=True)
    
    # 1. 分析时间常数分布
    models = analyze_time_constants()
    
    # 2. 分析数据模式
    analyze_data_patterns()
    
    # 3. 比较训练动态
    histories = compare_training_dynamics()
    
    print(f"\n🎯 深入分析总结:")
    print("="*50)
    
    # 分析最终时间常数
    for name, history in histories.items():
        if history['tau_n1'] and history['tau_n2']:
            final_tau1 = history['tau_n1'][-1]
            final_tau2 = history['tau_n2'][-1]
            final_acc = history['accuracies'][-1]
            
            print(f"{name}:")
            print(f"  最终准确率: {final_acc:.1f}%")
            print(f"  最终Branch1 τ: {final_tau1:.3f}")
            print(f"  最终Branch2 τ: {final_tau2:.3f}")
            print(f"  时间常数差异: {abs(final_tau1 - final_tau2):.3f}")
    
    print(f"\n💡 可能的解释:")
    print("1. 随机初始化可能提供了更好的探索空间")
    print("2. 当前任务可能不需要极端的时间常数差异")
    print("3. Medium范围的时间常数可能更适合这个特定任务")
    print("4. 有益初始化的假设可能需要针对任务调整")

if __name__ == '__main__':
    main()
