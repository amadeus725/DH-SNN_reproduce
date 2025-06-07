#!/usr/bin/env python3
"""
创建英文版的实验结果图片，用于LaTeX报告
"""

import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# 设置matplotlib
plt.rcParams['font.size'] = 12
plt.rcParams['axes.unicode_minus'] = False

def create_delayed_xor_figure():
    """创建延迟XOR结果图"""
    
    delays = [25, 50, 100, 150, 200, 250, 300, 400]
    vanilla_accuracies = [55.0] * 8
    dh_accuracies = [84.2, 82.5, 76.7, 65.0, 66.7, 75.8, 84.2, 68.3]
    
    plt.figure(figsize=(10, 6))
    
    # 绘制线条
    plt.plot(delays, vanilla_accuracies, 'o-', color='#E74C3C', linewidth=3, 
             markersize=8, label='Vanilla SFNN', markerfacecolor='white', markeredgewidth=2)
    plt.plot(delays, dh_accuracies, 's-', color='#3498DB', linewidth=3, 
             markersize=8, label='DH-SFNN', markerfacecolor='white', markeredgewidth=2)
    
    # 设置图表
    plt.title('Delayed Spiking XOR: Long-term Memory Performance', fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Delay Steps', fontsize=14, fontweight='bold')
    plt.ylabel('Accuracy (%)', fontsize=14, fontweight='bold')
    
    # 设置网格
    plt.grid(True, alpha=0.3, linestyle='--')
    
    # 设置图例
    plt.legend(fontsize=12, frameon=True, fancybox=True, shadow=True, loc='lower right')
    
    # 设置坐标轴
    plt.xlim(0, 420)
    plt.ylim(50, 90)
    plt.xticks(delays, fontsize=12)
    plt.yticks(fontsize=12)
    
    # 添加性能提升注释
    avg_improvement = np.mean(dh_accuracies) - np.mean(vanilla_accuracies)
    plt.text(200, 85, f'Average Improvement: +{avg_improvement:.1f}%', 
             fontsize=12, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7))
    
    plt.tight_layout()
    return plt.gcf()

def create_multitimescale_xor_figure():
    """创建多时间尺度XOR结果图"""
    
    models = ['Vanilla\nSFNN', '1-Branch DH-SFNN\n(Small)', '1-Branch DH-SFNN\n(Large)', 
              '2-Branch DH-SFNN\n(Fixed)', '2-Branch DH-SFNN\n(Learnable)']
    accuracies = [50.2, 52.8, 53.1, 84.6, 96.2]
    colors = ['#E74C3C', '#F39C12', '#F39C12', '#3498DB', '#27AE60']
    
    plt.figure(figsize=(12, 8))
    
    # 创建柱状图
    bars = plt.bar(models, accuracies, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    
    # 添加数值标签
    for bar, acc in zip(bars, accuracies):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{acc:.1f}%', ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    # 设置图表
    plt.title('Multi-timescale Spiking XOR: Architecture Comparison', fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Model Architecture', fontsize=14, fontweight='bold')
    plt.ylabel('Accuracy (%)', fontsize=14, fontweight='bold')
    
    # 设置网格
    plt.grid(True, alpha=0.3, linestyle='--', axis='y')
    
    # 设置坐标轴
    plt.ylim(0, 105)
    plt.xticks(rotation=45, ha='right', fontsize=11)
    plt.yticks(fontsize=12)
    
    # 添加性能提升注释
    improvement = accuracies[-1] - accuracies[0]
    plt.text(2, 90, f'Best Improvement: +{improvement:.1f}%', 
             fontsize=12, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.7))
    
    plt.tight_layout()
    return plt.gcf()

def create_dataset_comparison_figure():
    """创建数据集性能对比图"""
    
    datasets = ['SHD', 'SSC']
    vanilla_accs = [54.5, 46.8]
    dh_accs = [79.8, 60.5]
    
    x = np.arange(len(datasets))
    width = 0.35
    
    plt.figure(figsize=(10, 6))
    
    # 创建分组柱状图
    bars1 = plt.bar(x - width/2, vanilla_accs, width, label='Vanilla SNN', 
                    color='#E74C3C', alpha=0.8, edgecolor='black')
    bars2 = plt.bar(x + width/2, dh_accs, width, label='DH-SNN', 
                    color='#3498DB', alpha=0.8, edgecolor='black')
    
    # 添加数值标签
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{height:.1f}%', ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    # 设置图表
    plt.title('Neuromorphic Dataset Performance Comparison', fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Dataset', fontsize=14, fontweight='bold')
    plt.ylabel('Accuracy (%)', fontsize=14, fontweight='bold')
    
    # 设置网格
    plt.grid(True, alpha=0.3, linestyle='--', axis='y')
    
    # 设置坐标轴和图例
    plt.xticks(x, datasets, fontsize=12)
    plt.yticks(fontsize=12)
    plt.ylim(0, 85)
    plt.legend(fontsize=12, frameon=True, fancybox=True, shadow=True)
    
    # 添加改进百分比
    for i, (v_acc, d_acc) in enumerate(zip(vanilla_accs, dh_accs)):
        improvement = d_acc - v_acc
        plt.text(i, max(v_acc, d_acc) + 5, f'+{improvement:.1f}%', 
                ha='center', va='bottom', fontsize=11, fontweight='bold', color='green')
    
    plt.tight_layout()
    return plt.gcf()

def create_performance_summary_figure():
    """创建性能总结图"""
    
    experiments = ['Delayed XOR', 'Multi-timescale XOR', 'SHD Dataset', 'SSC Dataset']
    improvements = [20.4, 46.0, 25.3, 13.7]
    colors = ['#3498DB', '#27AE60', '#F39C12', '#E74C3C']
    
    plt.figure(figsize=(12, 6))
    
    # 创建柱状图
    bars = plt.bar(experiments, improvements, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    
    # 添加数值标签
    for bar, imp in zip(bars, improvements):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'+{imp:.1f}%', ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    # 设置图表
    plt.title('DH-SNN Performance Improvements Over Vanilla SNN', fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Experiment Type', fontsize=14, fontweight='bold')
    plt.ylabel('Performance Improvement (%)', fontsize=14, fontweight='bold')
    
    # 设置网格
    plt.grid(True, alpha=0.3, linestyle='--', axis='y')
    
    # 设置坐标轴
    plt.ylim(0, 50)
    plt.xticks(rotation=45, ha='right', fontsize=11)
    plt.yticks(fontsize=12)
    
    # 添加平均改进线
    avg_improvement = np.mean(improvements)
    plt.axhline(y=avg_improvement, color='red', linestyle='--', alpha=0.7, linewidth=2)
    plt.text(1.5, avg_improvement + 1, f'Average: {avg_improvement:.1f}%', 
             fontsize=11, color='red', fontweight='bold')
    
    plt.tight_layout()
    return plt.gcf()

def main():
    """主函数"""
    print("🎨 生成英文版实验结果图片...")
    
    # 创建输出目录
    output_dir = Path("DH-SNN_Reproduction_Report/figures")
    output_dir.mkdir(exist_ok=True)
    
    # 生成图片
    figures = {
        'delayed_xor_performance_en': create_delayed_xor_figure(),
        'multitimescale_xor_comparison_en': create_multitimescale_xor_figure(),
        'dataset_performance_comparison_en': create_dataset_comparison_figure(),
        'performance_summary_en': create_performance_summary_figure()
    }
    
    for name, fig in figures.items():
        try:
            # 保存高质量PNG
            png_path = output_dir / f"{name}.png"
            fig.savefig(str(png_path), dpi=300, bbox_inches='tight', 
                       facecolor='white', edgecolor='none')
            print(f"✅ {png_path}")
            
            # 清理内存
            plt.close(fig)
            
        except Exception as e:
            print(f"❌ 生成 {name} 失败: {e}")
    
    print(f"\n🎉 英文版图片生成完成!")
    print(f"📁 所有图片保存在: DH-SNN_Reproduction_Report/figures/")

if __name__ == '__main__':
    main()
