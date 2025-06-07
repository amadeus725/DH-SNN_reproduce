#!/usr/bin/env python3
"""
简化版Figure 4c复现 - 时间常数分布分析
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import os

# 导入之前的模型
from paper_reproduction import Paper2BranchDH_SFNN, get_batch, channel_size, hidden_dims, output_dim, learning_rate

print("📊 简化版Figure 4c - 时间常数分布分析")
print("="*50)

def quick_train_and_analyze():
    """快速训练并分析时间常数"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🔧 使用设备: {device}")
    
    # 创建模型
    model = Paper2BranchDH_SFNN(channel_size*2, hidden_dims, output_dim, learnable=True, device=device)
    model.to(device)
    
    # 记录初始时间常数
    print(f"\n📊 记录初始时间常数...")
    initial_tau_m = torch.sigmoid(model.dense_1.tau_m).detach().cpu().numpy()
    initial_tau_n1 = torch.sigmoid(model.dense_1.tau_n1).detach().cpu().numpy()
    initial_tau_n2 = torch.sigmoid(model.dense_1.tau_n2).detach().cpu().numpy()
    
    print(f"初始Branch 1: {initial_tau_n1.mean():.3f} ± {initial_tau_n1.std():.3f}")
    print(f"初始Branch 2: {initial_tau_n2.mean():.3f} ± {initial_tau_n2.std():.3f}")
    print(f"初始Membrane: {initial_tau_m.mean():.3f} ± {initial_tau_m.std():.3f}")
    
    # 简化训练
    print(f"\n🏋️ 开始训练...")
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    for epoch in range(20):  # 减少训练轮数
        model.train()
        train_loss_sum = 0
        sum_correct = 0
        sum_sample = 0
        
        for _ in range(10):  # 减少每轮的批次数
            model.init()
            
            data, target = get_batch(device)
            optimizer.zero_grad()
            
            loss, output, correct, total = model(data, target)
            
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 20)
            train_loss_sum += loss.item()
            optimizer.step()
            sum_correct += correct.item()
            sum_sample += total
        
        acc = sum_correct / sum_sample * 100 if sum_sample > 0 else 0
        if epoch % 5 == 0:
            print(f'Epoch {epoch+1:2d}: Loss={train_loss_sum/10:.4f}, Acc={acc:.1f}%')
    
    # 记录最终时间常数
    print(f"\n📊 记录最终时间常数...")
    final_tau_m = torch.sigmoid(model.dense_1.tau_m).detach().cpu().numpy()
    final_tau_n1 = torch.sigmoid(model.dense_1.tau_n1).detach().cpu().numpy()
    final_tau_n2 = torch.sigmoid(model.dense_1.tau_n2).detach().cpu().numpy()
    
    print(f"最终Branch 1: {final_tau_n1.mean():.3f} ± {final_tau_n1.std():.3f}")
    print(f"最终Branch 2: {final_tau_n2.mean():.3f} ± {final_tau_n2.std():.3f}")
    print(f"最终Membrane: {final_tau_m.mean():.3f} ± {final_tau_m.std():.3f}")
    
    return {
        'initial': {'tau_m': initial_tau_m, 'tau_n1': initial_tau_n1, 'tau_n2': initial_tau_n2},
        'final': {'tau_m': final_tau_m, 'tau_n1': final_tau_n1, 'tau_n2': final_tau_n2}
    }

def plot_simple_figure4c(results):
    """绘制简化的Figure 4c"""
    print(f"\n🎨 绘制简化版Figure 4c...")
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Figure 4c: Time Constant Distribution Analysis\n(Before and After Training)', 
                 fontsize=14, fontweight='bold')
    
    # 颜色设置
    colors = {'branch1': '#2E86AB', 'branch2': '#A23B72', 'membrane': '#F18F01'}
    
    # 训练前
    axes[0, 0].hist(results['initial']['tau_n1'], bins=15, alpha=0.7, 
                    color=colors['branch1'], density=True)
    axes[0, 0].set_title('Branch 1 (Before Training)\nLarge Initialization', fontweight='bold')
    axes[0, 0].set_xlabel('Time Constant τ')
    axes[0, 0].set_ylabel('Density')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_xlim(0, 1)
    
    axes[0, 1].hist(results['initial']['tau_n2'], bins=15, alpha=0.7, 
                    color=colors['branch2'], density=True)
    axes[0, 1].set_title('Branch 2 (Before Training)\nSmall Initialization', fontweight='bold')
    axes[0, 1].set_xlabel('Time Constant τ')
    axes[0, 1].set_ylabel('Density')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_xlim(0, 1)
    
    axes[0, 2].hist(results['initial']['tau_m'], bins=15, alpha=0.7, 
                    color=colors['membrane'], density=True)
    axes[0, 2].set_title('Membrane (Before Training)\nMedium Initialization', fontweight='bold')
    axes[0, 2].set_xlabel('Time Constant τ')
    axes[0, 2].set_ylabel('Density')
    axes[0, 2].grid(True, alpha=0.3)
    axes[0, 2].set_xlim(0, 1)
    
    # 训练后
    axes[1, 0].hist(results['final']['tau_n1'], bins=15, alpha=0.7, 
                    color=colors['branch1'], density=True)
    axes[1, 0].set_title('Branch 1 (After Training)\nLearned Distribution', fontweight='bold')
    axes[1, 0].set_xlabel('Time Constant τ')
    axes[1, 0].set_ylabel('Density')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_xlim(0, 1)
    
    axes[1, 1].hist(results['final']['tau_n2'], bins=15, alpha=0.7, 
                    color=colors['branch2'], density=True)
    axes[1, 1].set_title('Branch 2 (After Training)\nLearned Distribution', fontweight='bold')
    axes[1, 1].set_xlabel('Time Constant τ')
    axes[1, 1].set_ylabel('Density')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_xlim(0, 1)
    
    axes[1, 2].hist(results['final']['tau_m'], bins=15, alpha=0.7, 
                    color=colors['membrane'], density=True)
    axes[1, 2].set_title('Membrane (After Training)\nLearned Distribution', fontweight='bold')
    axes[1, 2].set_xlabel('Time Constant τ')
    axes[1, 2].set_ylabel('Density')
    axes[1, 2].grid(True, alpha=0.3)
    axes[1, 2].set_xlim(0, 1)
    
    plt.tight_layout()
    
    # 保存图片
    os.makedirs("results", exist_ok=True)
    plt.savefig("results/simple_figure4c.png", dpi=300, bbox_inches='tight')
    plt.savefig("results/simple_figure4c.pdf", bbox_inches='tight')
    print(f"✅ 简化版Figure 4c已保存到: results/simple_figure4c.png")
    
    return fig

def analyze_changes(results):
    """分析时间常数变化"""
    print(f"\n📊 时间常数变化分析:")
    print("="*40)
    
    # 计算变化
    delta_n1 = results['final']['tau_n1'].mean() - results['initial']['tau_n1'].mean()
    delta_n2 = results['final']['tau_n2'].mean() - results['initial']['tau_n2'].mean()
    delta_m = results['final']['tau_m'].mean() - results['initial']['tau_m'].mean()
    
    print(f"Branch 1变化: {delta_n1:+.3f}")
    print(f"Branch 2变化: {delta_n2:+.3f}")
    print(f"Membrane变化: {delta_m:+.3f}")
    
    # 分化程度
    initial_diff = abs(results['initial']['tau_n1'].mean() - results['initial']['tau_n2'].mean())
    final_diff = abs(results['final']['tau_n1'].mean() - results['final']['tau_n2'].mean())
    
    print(f"\n分支分化分析:")
    print(f"训练前分支差异: {initial_diff:.3f}")
    print(f"训练后分支差异: {final_diff:.3f}")
    print(f"分化变化: {final_diff - initial_diff:+.3f}")
    
    if final_diff > initial_diff:
        print("✅ 分支分化增强 - 符合论文预期")
    else:
        print("⚠️ 分支分化减弱 - 需要进一步分析")

def create_summary_plot(results):
    """创建总结图"""
    print(f"\n🎨 创建总结对比图...")
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    # 数据准备
    categories = ['Branch 1\n(Long-term)', 'Branch 2\n(Short-term)', 'Membrane\n(Integration)']
    initial_means = [
        results['initial']['tau_n1'].mean(),
        results['initial']['tau_n2'].mean(),
        results['initial']['tau_m'].mean()
    ]
    final_means = [
        results['final']['tau_n1'].mean(),
        results['final']['tau_n2'].mean(),
        results['final']['tau_m'].mean()
    ]
    
    x = np.arange(len(categories))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, initial_means, width, label='Before Training', alpha=0.8, color='lightblue')
    bars2 = ax.bar(x + width/2, final_means, width, label='After Training', alpha=0.8, color='darkblue')
    
    ax.set_xlabel('Time Constant Type')
    ax.set_ylabel('Time Constant Value')
    ax.set_title('Time Constant Changes During Training')
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 添加数值标签
    for bar in bars1:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.3f}', ha='center', va='bottom')
    
    for bar in bars2:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig("results/time_constant_summary.png", dpi=300, bbox_inches='tight')
    print(f"✅ 总结图已保存到: results/time_constant_summary.png")
    
    return fig

def main():
    """主函数"""
    # 训练并分析
    results = quick_train_and_analyze()
    
    # 绘制Figure 4c
    plot_simple_figure4c(results)
    
    # 分析变化
    analyze_changes(results)
    
    # 创建总结图
    create_summary_plot(results)
    
    # 保存结果
    torch.save(results, "results/simple_figure4c_results.pth")
    print(f"\n💾 结果已保存到: results/simple_figure4c_results.pth")
    
    print(f"\n🎉 简化版Figure 4c复现完成!")

if __name__ == '__main__':
    main()
