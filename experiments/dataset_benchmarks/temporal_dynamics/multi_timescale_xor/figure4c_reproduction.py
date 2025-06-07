#!/usr/bin/env python3
"""
复现论文Figure 4c - 时间常数分布分析
训练前后的时间常数分布对比，包含KDE曲线和直方图
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import os

# 简单的KDE实现
def simple_kde(data, x_range=None, bandwidth=0.1):
    """简单的KDE实现"""
    if x_range is None:
        x_range = np.linspace(data.min(), data.max(), 100)

    kde_values = np.zeros_like(x_range)
    for i, x in enumerate(x_range):
        kde_values[i] = np.mean(np.exp(-0.5 * ((data - x) / bandwidth) ** 2))

    # 归一化
    kde_values = kde_values / (bandwidth * np.sqrt(2 * np.pi))
    return x_range, kde_values

# 导入之前的模型
from paper_reproduction import (
    Paper2BranchDH_SFNN, get_batch,
    time_steps, channel_size, hidden_dims, output_dim, learning_rate
)

print("📊 复现Figure 4c - 时间常数分布分析")
print("="*60)

def train_and_track_time_constants(model, epochs=50, device='cpu'):
    """训练模型并跟踪时间常数变化"""
    print(f"🏋️ 训练并跟踪时间常数变化...")

    model.to(device)

    # 记录初始时间常数
    initial_tau_m = torch.sigmoid(model.dense_1.tau_m).detach().cpu().numpy()
    initial_tau_n1 = torch.sigmoid(model.dense_1.tau_n1).detach().cpu().numpy()
    initial_tau_n2 = torch.sigmoid(model.dense_1.tau_n2).detach().cpu().numpy()

    # 优化器设置 (按照原论文)
    base_params = [model.dense_2.weight, model.dense_2.bias,
                   model.dense_1.dense1.weight, model.dense_1.dense2.weight]

    optimizer_params = [
        {'params': base_params},
        {'params': model.dense_1.tau_m, 'lr': learning_rate},
        {'params': model.dense_1.tau_n1, 'lr': learning_rate},
        {'params': model.dense_1.tau_n2, 'lr': learning_rate}
    ]

    optimizer = torch.optim.Adam(optimizer_params, lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)

    # 训练历史
    history = {
        'tau_m': [],
        'tau_n1': [],
        'tau_n2': [],
        'losses': [],
        'accuracies': []
    }

    log_interval = 20

    for epoch in range(epochs):
        model.train()
        train_loss_sum = 0
        sum_correct = 0
        sum_sample = 0

        for _ in range(log_interval):
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

        scheduler.step()

        # 记录时间常数
        current_tau_m = torch.sigmoid(model.dense_1.tau_m).detach().cpu().numpy()
        current_tau_n1 = torch.sigmoid(model.dense_1.tau_n1).detach().cpu().numpy()
        current_tau_n2 = torch.sigmoid(model.dense_1.tau_n2).detach().cpu().numpy()

        history['tau_m'].append(current_tau_m.copy())
        history['tau_n1'].append(current_tau_n1.copy())
        history['tau_n2'].append(current_tau_n2.copy())
        history['losses'].append(train_loss_sum / log_interval)

        acc = sum_correct / sum_sample * 100 if sum_sample > 0 else 0
        history['accuracies'].append(acc)

        if epoch % 10 == 0:
            print(f'Epoch {epoch+1:3d}: Loss={train_loss_sum/log_interval:.4f}, Acc={acc:.1f}%')

    # 记录最终时间常数
    final_tau_m = torch.sigmoid(model.dense_1.tau_m).detach().cpu().numpy()
    final_tau_n1 = torch.sigmoid(model.dense_1.tau_n1).detach().cpu().numpy()
    final_tau_n2 = torch.sigmoid(model.dense_1.tau_n2).detach().cpu().numpy()

    return {
        'initial': {'tau_m': initial_tau_m, 'tau_n1': initial_tau_n1, 'tau_n2': initial_tau_n2},
        'final': {'tau_m': final_tau_m, 'tau_n1': final_tau_n1, 'tau_n2': final_tau_n2},
        'history': history
    }

def plot_figure4c(results, save_path="results/figure4c_reproduction.png"):
    """绘制Figure 4c - 时间常数分布"""
    print(f"🎨 绘制Figure 4c...")

    # 设置绘图风格
    plt.style.use('default')

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Figure 4c: Time Constant Distribution Analysis\n(Before and After Training)',
                 fontsize=16, fontweight='bold')

    # 颜色设置
    colors = {'branch1': '#2E86AB', 'branch2': '#A23B72', 'membrane': '#F18F01'}

    # 上排：训练前
    # Branch 1 (Large initialization)
    tau_n1_initial = results['initial']['tau_n1'].flatten()
    axes[0, 0].hist(tau_n1_initial, bins=20, alpha=0.7,
                    color=colors['branch1'], density=True, label='Branch 1')

    # 添加KDE曲线
    if len(tau_n1_initial) > 1:
        kde_x, kde_y = simple_kde(tau_n1_initial, bandwidth=0.05)
        axes[0, 0].plot(kde_x, kde_y, color=colors['branch1'], linewidth=2)

    axes[0, 0].set_title('Branch 1 (Before Training)\nLarge Initialization', fontweight='bold')
    axes[0, 0].set_xlabel('Time Constant τ')
    axes[0, 0].set_ylabel('Density')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_xlim(0, 1)

    # Branch 2 (Small initialization)
    tau_n2_initial = results['initial']['tau_n2'].flatten()
    axes[0, 1].hist(tau_n2_initial, bins=20, alpha=0.7,
                    color=colors['branch2'], density=True, label='Branch 2')

    if len(tau_n2_initial) > 1:
        kde_x, kde_y = simple_kde(tau_n2_initial, bandwidth=0.05)
        axes[0, 1].plot(kde_x, kde_y, color=colors['branch2'], linewidth=2)

    axes[0, 1].set_title('Branch 2 (Before Training)\nSmall Initialization', fontweight='bold')
    axes[0, 1].set_xlabel('Time Constant τ')
    axes[0, 1].set_ylabel('Density')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_xlim(0, 1)

    # Membrane potential
    tau_m_initial = results['initial']['tau_m'].flatten()
    axes[0, 2].hist(tau_m_initial, bins=20, alpha=0.7,
                    color=colors['membrane'], density=True, label='Membrane')

    if len(tau_m_initial) > 1:
        kde_x, kde_y = simple_kde(tau_m_initial, bandwidth=0.05)
        axes[0, 2].plot(kde_x, kde_y, color=colors['membrane'], linewidth=2)

    axes[0, 2].set_title('Membrane Potential (Before Training)\nMedium Initialization', fontweight='bold')
    axes[0, 2].set_xlabel('Time Constant τ')
    axes[0, 2].set_ylabel('Density')
    axes[0, 2].grid(True, alpha=0.3)
    axes[0, 2].set_xlim(0, 1)

    # 下排：训练后
    # Branch 1 (After training)
    tau_n1_final = results['final']['tau_n1'].flatten()
    axes[1, 0].hist(tau_n1_final, bins=20, alpha=0.7,
                    color=colors['branch1'], density=True, label='Branch 1')

    if len(tau_n1_final) > 1:
        kde_x, kde_y = simple_kde(tau_n1_final, bandwidth=0.05)
        axes[1, 0].plot(kde_x, kde_y, color=colors['branch1'], linewidth=2)

    axes[1, 0].set_title('Branch 1 (After Training)\nLearned Distribution', fontweight='bold')
    axes[1, 0].set_xlabel('Time Constant τ')
    axes[1, 0].set_ylabel('Density')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_xlim(0, 1)

    # Branch 2 (After training)
    tau_n2_final = results['final']['tau_n2'].flatten()
    axes[1, 1].hist(tau_n2_final, bins=20, alpha=0.7,
                    color=colors['branch2'], density=True, label='Branch 2')

    if len(tau_n2_final) > 1:
        kde_x, kde_y = simple_kde(tau_n2_final, bandwidth=0.05)
        axes[1, 1].plot(kde_x, kde_y, color=colors['branch2'], linewidth=2)

    axes[1, 1].set_title('Branch 2 (After Training)\nLearned Distribution', fontweight='bold')
    axes[1, 1].set_xlabel('Time Constant τ')
    axes[1, 1].set_ylabel('Density')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_xlim(0, 1)

    # Membrane potential (After training)
    tau_m_final = results['final']['tau_m'].flatten()
    axes[1, 2].hist(tau_m_final, bins=20, alpha=0.7,
                    color=colors['membrane'], density=True, label='Membrane')

    if len(tau_m_final) > 1:
        kde_x, kde_y = simple_kde(tau_m_final, bandwidth=0.05)
        axes[1, 2].plot(kde_x, kde_y, color=colors['membrane'], linewidth=2)

    axes[1, 2].set_title('Membrane Potential (After Training)\nLearned Distribution', fontweight='bold')
    axes[1, 2].set_xlabel('Time Constant τ')
    axes[1, 2].set_ylabel('Density')
    axes[1, 2].grid(True, alpha=0.3)
    axes[1, 2].set_xlim(0, 1)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.savefig(save_path.replace('.png', '.pdf'), bbox_inches='tight')
    print(f"✅ Figure 4c已保存到: {save_path}")

    return fig

def plot_time_constant_evolution(results, save_path="results/time_constant_evolution.png"):
    """绘制时间常数演化过程"""
    print(f"🎨 绘制时间常数演化...")

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Time Constant Evolution During Training', fontsize=16, fontweight='bold')

    epochs = range(1, len(results['history']['tau_m']) + 1)

    # 计算平均值和标准差
    tau_m_mean = [np.mean(tau) for tau in results['history']['tau_m']]
    tau_m_std = [np.std(tau) for tau in results['history']['tau_m']]

    tau_n1_mean = [np.mean(tau) for tau in results['history']['tau_n1']]
    tau_n1_std = [np.std(tau) for tau in results['history']['tau_n1']]

    tau_n2_mean = [np.mean(tau) for tau in results['history']['tau_n2']]
    tau_n2_std = [np.std(tau) for tau in results['history']['tau_n2']]

    # 膜电位时间常数
    axes[0, 0].plot(epochs, tau_m_mean, 'g-', linewidth=2, label='Mean')
    axes[0, 0].fill_between(epochs,
                           np.array(tau_m_mean) - np.array(tau_m_std),
                           np.array(tau_m_mean) + np.array(tau_m_std),
                           alpha=0.3, color='green')
    axes[0, 0].set_title('Membrane Time Constant τ_m')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Time Constant')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend()

    # Branch 1时间常数
    axes[0, 1].plot(epochs, tau_n1_mean, 'b-', linewidth=2, label='Branch 1 Mean')
    axes[0, 1].fill_between(epochs,
                           np.array(tau_n1_mean) - np.array(tau_n1_std),
                           np.array(tau_n1_mean) + np.array(tau_n1_std),
                           alpha=0.3, color='blue')
    axes[0, 1].set_title('Branch 1 Time Constant τ_n1 (Long-term)')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Time Constant')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend()

    # Branch 2时间常数
    axes[1, 0].plot(epochs, tau_n2_mean, 'r-', linewidth=2, label='Branch 2 Mean')
    axes[1, 0].fill_between(epochs,
                           np.array(tau_n2_mean) - np.array(tau_n2_std),
                           np.array(tau_n2_mean) + np.array(tau_n2_std),
                           alpha=0.3, color='red')
    axes[1, 0].set_title('Branch 2 Time Constant τ_n2 (Short-term)')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Time Constant')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].legend()

    # 训练准确率
    axes[1, 1].plot(epochs, results['history']['accuracies'], 'purple', linewidth=2)
    axes[1, 1].set_title('Training Accuracy')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Accuracy (%)')
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ 时间常数演化图已保存到: {save_path}")

    return fig

def analyze_time_constant_statistics(results):
    """分析时间常数统计信息"""
    print(f"\n📊 时间常数统计分析:")
    print("="*50)

    # 训练前统计
    print("训练前:")
    print(f"  Branch 1 (Large): {results['initial']['tau_n1'].mean():.3f} ± {results['initial']['tau_n1'].std():.3f}")
    print(f"  Branch 2 (Small): {results['initial']['tau_n2'].mean():.3f} ± {results['initial']['tau_n2'].std():.3f}")
    print(f"  Membrane (Medium): {results['initial']['tau_m'].mean():.3f} ± {results['initial']['tau_m'].std():.3f}")

    # 训练后统计
    print("\n训练后:")
    print(f"  Branch 1: {results['final']['tau_n1'].mean():.3f} ± {results['final']['tau_n1'].std():.3f}")
    print(f"  Branch 2: {results['final']['tau_n2'].mean():.3f} ± {results['final']['tau_n2'].std():.3f}")
    print(f"  Membrane: {results['final']['tau_m'].mean():.3f} ± {results['final']['tau_m'].std():.3f}")

    # 变化分析
    print("\n变化分析:")
    delta_n1 = results['final']['tau_n1'].mean() - results['initial']['tau_n1'].mean()
    delta_n2 = results['final']['tau_n2'].mean() - results['initial']['tau_n2'].mean()
    delta_m = results['final']['tau_m'].mean() - results['initial']['tau_m'].mean()

    print(f"  Branch 1变化: {delta_n1:+.3f}")
    print(f"  Branch 2变化: {delta_n2:+.3f}")
    print(f"  Membrane变化: {delta_m:+.3f}")

    # 分化程度
    final_diff = abs(results['final']['tau_n1'].mean() - results['final']['tau_n2'].mean())
    initial_diff = abs(results['initial']['tau_n1'].mean() - results['initial']['tau_n2'].mean())

    print(f"\n分支分化:")
    print(f"  训练前分支差异: {initial_diff:.3f}")
    print(f"  训练后分支差异: {final_diff:.3f}")
    print(f"  分化增强: {final_diff - initial_diff:+.3f}")

def main():
    """主函数"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🔧 使用设备: {device}")

    # 创建结果目录
    os.makedirs("results", exist_ok=True)

    # 创建2分支DH-SFNN模型
    print(f"\n🧠 创建2-Branch DH-SFNN模型...")
    model = Paper2BranchDH_SFNN(channel_size*2, hidden_dims, output_dim, learnable=True, device=device)

    # 训练并跟踪时间常数
    results = train_and_track_time_constants(model, epochs=50, device=device)

    # 分析统计信息
    analyze_time_constant_statistics(results)

    # 绘制Figure 4c
    plot_figure4c(results)

    # 绘制时间常数演化
    plot_time_constant_evolution(results)

    # 保存结果
    torch.save(results, "results/figure4c_time_constant_analysis.pth")
    print(f"\n💾 时间常数分析结果已保存到: results/figure4c_time_constant_analysis.pth")

    print(f"\n🎉 Figure 4c复现完成!")

if __name__ == '__main__':
    main()
