#!/usr/bin/env python3
"""
可视化最终实验结果
生成Figure 4b和Figure 4c的复现图
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from scipy import stats

# 设置matplotlib
plt.rcParams['font.size'] = 12
plt.rcParams['figure.figsize'] = (12, 8)

def load_final_results():
    """加载最终实验结果"""
    try:
        results = torch.load("results/final_experiment_results.pth", map_location='cpu')
        return results
    except FileNotFoundError:
        print("❌ 未找到最终实验结果文件")
        return None

def plot_figure4b_reproduction(results):
    """绘制Figure 4b复现图"""
    
    fig, ax = plt.subplots(1, 1, figsize=(14, 8))
    
    # 提取数据
    exp_names = []
    means = []
    stds = []
    colors = []
    
    # 定义颜色方案
    color_map = {
        'Vanilla SFNN': '#1f77b4',
        '1-Branch DH-SFNN (Small)': '#ff7f0e', 
        '1-Branch DH-SFNN (Large)': '#2ca02c',
        '2-Branch DH-SFNN (Beneficial)': '#d62728',
        '2-Branch DH-SFNN (Fixed)': '#9467bd',
        '2-Branch DH-SFNN (Random)': '#8c564b'
    }
    
    for exp_name, exp_data in results.items():
        # 简化名称用于显示
        display_name = exp_name.replace('DH-SFNN', 'DH-SFNN\n').replace('(', '\n(')
        exp_names.append(display_name)
        means.append(exp_data['mean'])
        stds.append(exp_data['std'])
        colors.append(color_map.get(exp_name, '#gray'))
    
    # 创建条形图
    x_pos = np.arange(len(exp_names))
    bars = ax.bar(x_pos, means, yerr=stds, capsize=8, alpha=0.8, 
                  color=colors, edgecolor='black', linewidth=1.5)
    
    # 设置标签和标题
    ax.set_xlabel('Model Architecture', fontsize=14, fontweight='bold')
    ax.set_ylabel('Accuracy (%)', fontsize=14, fontweight='bold')
    ax.set_title('Figure 4b: Multi-timescale Spiking XOR Task Performance\n(Intra-neuron Heterogeneous Feature Integration)', 
                 fontsize=16, fontweight='bold', pad=20)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(exp_names, rotation=0, ha='center', fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, 100)
    
    # 添加数值标签
    for i, (mean, std, bar) in enumerate(zip(means, stds, bars)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + std + 1,
                f'{mean:.1f}±{std:.1f}%', 
                ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    # 添加显著性分析
    if len(means) >= 2:
        # 找到最佳性能
        best_idx = np.argmax(means)
        best_name = list(results.keys())[best_idx]
        
        # 添加文本说明
        textstr = f'Best Performance: {best_name}\n'
        textstr += f'Improvement over Vanilla: +{means[best_idx] - means[0]:.1f}%'
        
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
        ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=11,
                verticalalignment='top', bbox=props)
    
    plt.tight_layout()
    
    # 保存图片
    os.makedirs("results", exist_ok=True)
    plt.savefig("results/figure4b_final_reproduction.png", dpi=300, bbox_inches='tight')
    plt.savefig("results/figure4b_final_reproduction.pdf", bbox_inches='tight')
    
    print("✅ Figure 4b已保存到 results/figure4b_final_reproduction.png")
    
    return fig

def analyze_performance_gains(results):
    """分析性能提升"""
    
    print("\n📊 性能分析:")
    print("="*50)
    
    # 获取基线性能 (Vanilla SFNN)
    baseline_name = "Vanilla SFNN"
    if baseline_name in results:
        baseline_acc = results[baseline_name]['mean']
        print(f"基线性能 ({baseline_name}): {baseline_acc:.1f}%")
        
        print(f"\n相对于基线的性能提升:")
        for exp_name, exp_data in results.items():
            if exp_name != baseline_name:
                improvement = exp_data['mean'] - baseline_acc
                print(f"  {exp_name:30s}: +{improvement:5.1f}%")
    
    # 找到最佳模型
    best_model = max(results.items(), key=lambda x: x[1]['mean'])
    print(f"\n🏆 最佳模型: {best_model[0]}")
    print(f"   准确率: {best_model[1]['mean']:.1f}% ± {best_model[1]['std']:.1f}%")
    
    # 分析分支数量的影响
    print(f"\n🌿 分支数量影响分析:")
    
    # 1分支模型
    one_branch_models = [name for name in results.keys() if '1-Branch' in name]
    if one_branch_models:
        one_branch_accs = [results[name]['mean'] for name in one_branch_models]
        print(f"  1分支DH-SFNN平均: {np.mean(one_branch_accs):.1f}%")
    
    # 2分支模型
    two_branch_models = [name for name in results.keys() if '2-Branch' in name]
    if two_branch_models:
        two_branch_accs = [results[name]['mean'] for name in two_branch_models]
        print(f"  2分支DH-SFNN平均: {np.mean(two_branch_accs):.1f}%")
    
    # 有益初始化的影响
    if "2-Branch DH-SFNN (Beneficial)" in results and "2-Branch DH-SFNN (Random)" in results:
        beneficial_acc = results["2-Branch DH-SFNN (Beneficial)"]['mean']
        random_acc = results["2-Branch DH-SFNN (Random)"]['mean']
        init_benefit = beneficial_acc - random_acc
        print(f"\n🎯 有益初始化效果: +{init_benefit:.1f}%")
    
    return True

def create_summary_table(results):
    """创建结果总结表"""
    
    print(f"\n📋 实验结果总结表:")
    print("="*70)
    print(f"{'模型架构':<30} {'准确率 (%)':<12} {'标准差':<8} {'试验结果'}")
    print("="*70)
    
    for exp_name, exp_data in results.items():
        mean_acc = exp_data['mean']
        std_acc = exp_data['std']
        trials = exp_data['trials']
        trials_str = ', '.join([f"{t:.1f}" for t in trials])
        
        print(f"{exp_name:<30} {mean_acc:>6.1f}      {std_acc:>6.1f}    [{trials_str}]")
    
    print("="*70)

def plot_trial_variability(results):
    """绘制试验变异性分析"""
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    
    # 准备数据
    all_trials = []
    model_names = []
    
    for exp_name, exp_data in results.items():
        trials = exp_data['trials']
        for trial_acc in trials:
            all_trials.append(trial_acc)
            model_names.append(exp_name.replace('DH-SFNN', 'DH-SFNN\n'))
    
    # 创建箱线图
    unique_models = list(results.keys())
    trial_data = [results[model]['trials'] for model in unique_models]
    model_labels = [name.replace('DH-SFNN', 'DH-SFNN\n') for name in unique_models]
    
    bp = ax.boxplot(trial_data, labels=model_labels, patch_artist=True)
    
    # 设置颜色
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    for patch, color in zip(bp['boxes'], colors[:len(bp['boxes'])]):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
    ax.set_title('Trial Variability Analysis', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    plt.savefig("results/trial_variability_analysis.png", dpi=300, bbox_inches='tight')
    print("✅ 试验变异性分析图已保存到 results/trial_variability_analysis.png")
    
    return fig

def main():
    """主函数"""
    print("📊 可视化最终实验结果")
    print("="*50)
    
    # 加载结果
    results = load_final_results()
    if results is None:
        print("⏳ 等待实验完成...")
        return
    
    # 创建结果总结表
    create_summary_table(results)
    
    # 分析性能提升
    analyze_performance_gains(results)
    
    # 绘制Figure 4b复现图
    print(f"\n🎨 绘制Figure 4b复现图...")
    plot_figure4b_reproduction(results)
    
    # 绘制试验变异性分析
    print(f"\n🎨 绘制试验变异性分析...")
    plot_trial_variability(results)
    
    print(f"\n🎉 结果可视化完成!")
    print(f"📁 所有图片保存在 results/ 目录下")

if __name__ == '__main__':
    main()
