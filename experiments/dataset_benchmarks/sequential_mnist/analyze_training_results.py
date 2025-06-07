#!/usr/bin/env python3
"""
Sequential MNIST训练结果分析和可视化
"""

import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import pandas as pd

# 设置中文字体和样式
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")

def load_training_results():
    """加载所有训练结果"""
    results = {}
    
    # 加载主要结果文件
    result_files = {
        'S-MNIST_Vanilla_SRNN': 'results/S-MNIST_vanilla_srnn_results.pth',
        'PS-MNIST_Vanilla_SRNN': 'results/PS-MNIST_vanilla_srnn_results.pth', 
        'S-MNIST_DH_SRNN': 'results/S-MNIST_dh_srnn_results.pth'
    }
    
    for name, file_path in result_files.items():
        if os.path.exists(file_path):
            try:
                data = torch.load(file_path, map_location='cpu')
                results[name] = data
                print(f"✅ 加载 {name}: {file_path}")
            except Exception as e:
                print(f"❌ 加载失败 {name}: {e}")
        else:
            print(f"⚠️  文件不存在: {file_path}")
    
    return results

def load_parallel_results():
    """加载并行训练结果"""
    parallel_results = {}
    
    # 检查并行结果目录
    parallel_dirs = [
        'results_parallel_4/S-MNIST_Vanilla-SRNN',
        'results_parallel_4/PS-MNIST_Vanilla-SRNN',
        'results_parallel_4/S-MNIST_DH-SRNN',
        'results_parallel_4/PS-MNIST_DH-SRNN'
    ]
    
    for dir_path in parallel_dirs:
        if os.path.exists(dir_path):
            # 查找结果文件
            for file in os.listdir(dir_path):
                if file.endswith('.pth') and 'results' in file:
                    full_path = os.path.join(dir_path, file)
                    try:
                        data = torch.load(full_path, map_location='cpu')
                        exp_name = os.path.basename(dir_path)
                        parallel_results[exp_name] = data
                        print(f"✅ 加载并行结果 {exp_name}: {file}")
                    except Exception as e:
                        print(f"❌ 加载并行结果失败 {exp_name}: {e}")
    
    return parallel_results

def analyze_training_curves(results):
    """分析训练曲线"""
    print("\n📈 训练曲线分析")
    print("=" * 50)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Sequential MNIST 训练过程分析', fontsize=16, fontweight='bold')
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    for idx, (name, data) in enumerate(results.items()):
        if 'train_accuracies' in data and 'test_accuracies' in data:
            epochs = range(1, len(data['train_accuracies']) + 1)
            color = colors[idx % len(colors)]
            
            # 训练准确率
            axes[0, 0].plot(epochs, data['train_accuracies'], 
                           label=name, color=color, linewidth=2)
            
            # 测试准确率
            axes[0, 1].plot(epochs, data['test_accuracies'], 
                           label=name, color=color, linewidth=2)
            
            # 训练损失
            if 'train_losses' in data:
                axes[1, 0].plot(epochs, data['train_losses'], 
                               label=name, color=color, linewidth=2)
            
            # 学习率
            if 'learning_rates' in data:
                axes[1, 1].plot(epochs, data['learning_rates'], 
                               label=name, color=color, linewidth=2)
    
    # 设置子图
    axes[0, 0].set_title('训练准确率')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Accuracy (%)')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].set_title('测试准确率')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy (%)')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    axes[1, 0].set_title('训练损失')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Loss')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    axes[1, 1].set_title('学习率')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Learning Rate')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/sequential_mnist_detailed_training_curves.png', 
                dpi=300, bbox_inches='tight')
    print("💾 保存训练曲线图: results/sequential_mnist_detailed_training_curves.png")
    plt.show()

def create_performance_comparison(results, parallel_results):
    """创建性能对比图"""
    print("\n📊 性能对比分析")
    print("=" * 50)
    
    # 收集所有结果
    all_results = {}
    all_results.update(results)
    all_results.update(parallel_results)
    
    # 提取性能数据
    performance_data = []
    
    for name, data in all_results.items():
        if isinstance(data, dict):
            # 解析实验名称
            if 'S-MNIST' in name and 'Vanilla' in name:
                task, model = 'S-MNIST', 'Vanilla SRNN'
            elif 'PS-MNIST' in name and 'Vanilla' in name:
                task, model = 'PS-MNIST', 'Vanilla SRNN'
            elif 'S-MNIST' in name and 'DH' in name:
                task, model = 'S-MNIST', 'DH-SRNN'
            elif 'PS-MNIST' in name and 'DH' in name:
                task, model = 'PS-MNIST', 'DH-SRNN'
            else:
                continue
            
            # 获取最佳准确率
            best_acc = data.get('best_test_accuracy', 0)
            final_acc = data.get('final_test_accuracy', 0)
            training_time = data.get('training_time_hours', 0)
            epochs = data.get('epochs', 0)
            
            performance_data.append({
                'Task': task,
                'Model': model,
                'Best_Accuracy': best_acc,
                'Final_Accuracy': final_acc,
                'Training_Time_Hours': training_time,
                'Epochs': epochs,
                'Source': 'Parallel' if name in parallel_results else 'Main'
            })
    
    if not performance_data:
        print("❌ 未找到有效的性能数据")
        return
    
    df = pd.DataFrame(performance_data)
    print(f"📋 性能数据表:")
    print(df.to_string(index=False))
    
    # 创建对比图
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Sequential MNIST 性能对比', fontsize=16, fontweight='bold')
    
    # 最佳准确率对比
    sns.barplot(data=df, x='Task', y='Best_Accuracy', hue='Model', ax=axes[0, 0])
    axes[0, 0].set_title('最佳测试准确率')
    axes[0, 0].set_ylabel('Accuracy (%)')
    
    # 最终准确率对比
    sns.barplot(data=df, x='Task', y='Final_Accuracy', hue='Model', ax=axes[0, 1])
    axes[0, 1].set_title('最终测试准确率')
    axes[0, 1].set_ylabel('Accuracy (%)')
    
    # 训练时间对比
    sns.barplot(data=df, x='Task', y='Training_Time_Hours', hue='Model', ax=axes[1, 0])
    axes[1, 0].set_title('训练时间')
    axes[1, 0].set_ylabel('Hours')
    
    # Epochs对比
    sns.barplot(data=df, x='Task', y='Epochs', hue='Model', ax=axes[1, 1])
    axes[1, 1].set_title('训练轮数')
    axes[1, 1].set_ylabel('Epochs')
    
    plt.tight_layout()
    plt.savefig('results/sequential_mnist_performance_detailed_comparison.png', 
                dpi=300, bbox_inches='tight')
    print("💾 保存性能对比图: results/sequential_mnist_performance_detailed_comparison.png")
    plt.show()
    
    return df

def analyze_convergence(results):
    """分析收敛性"""
    print("\n🎯 收敛性分析")
    print("=" * 50)
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle('模型收敛性分析', fontsize=16, fontweight='bold')
    
    for name, data in results.items():
        if 'test_accuracies' in data:
            accuracies = data['test_accuracies']
            epochs = range(1, len(accuracies) + 1)
            
            # 计算收敛点（准确率变化小于0.1%的连续10个epoch）
            convergence_point = None
            for i in range(10, len(accuracies)):
                recent_std = np.std(accuracies[i-10:i])
                if recent_std < 0.1:
                    convergence_point = i
                    break
            
            # 绘制准确率曲线
            axes[0].plot(epochs, accuracies, label=f"{name}", linewidth=2)
            if convergence_point:
                axes[0].axvline(x=convergence_point, linestyle='--', alpha=0.7,
                               label=f"{name} 收敛点: Epoch {convergence_point}")
            
            # 计算准确率改进率
            if len(accuracies) > 1:
                improvement_rate = np.diff(accuracies)
                axes[1].plot(epochs[1:], improvement_rate, label=f"{name}", linewidth=2)
    
    axes[0].set_title('测试准确率收敛曲线')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Test Accuracy (%)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].set_title('准确率改进率')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy Improvement (%)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[1].axhline(y=0, color='black', linestyle='-', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/sequential_mnist_convergence_analysis.png', 
                dpi=300, bbox_inches='tight')
    print("💾 保存收敛性分析图: results/sequential_mnist_convergence_analysis.png")
    plt.show()

def main():
    """主函数"""
    print("🔍 Sequential MNIST 训练结果分析")
    print("=" * 60)
    
    # 加载结果
    print("\n📂 加载训练结果...")
    results = load_training_results()
    parallel_results = load_parallel_results()
    
    print(f"\n📊 找到 {len(results)} 个主要结果")
    print(f"📊 找到 {len(parallel_results)} 个并行结果")
    
    if not results and not parallel_results:
        print("❌ 未找到任何训练结果")
        return
    
    # 分析训练曲线
    if results:
        analyze_training_curves(results)
    
    # 创建性能对比
    df = create_performance_comparison(results, parallel_results)
    
    # 分析收敛性
    if results:
        analyze_convergence(results)
    
    # 保存分析结果
    if df is not None:
        analysis_summary = {
            'timestamp': datetime.now().isoformat(),
            'analysis_type': 'detailed_training_analysis',
            'total_experiments': len(df),
            'performance_summary': df.to_dict('records')
        }
        
        with open('results/detailed_analysis_summary.json', 'w') as f:
            json.dump(analysis_summary, f, indent=2)
        print("💾 保存分析总结: results/detailed_analysis_summary.json")
    
    print("\n🎉 分析完成！")

if __name__ == "__main__":
    main()
