#!/usr/bin/env python3
"""
Sequential MNIST结果可视化 - 生成论文格式的图表
"""

import os
import sys
import torch
import numpy as np
import plotly.graph_objects as go
import plotly.subplots as sp
from plotly.subplots import make_subplots
import plotly.express as px
from pathlib import Path
import json
from datetime import datetime

# 添加项目根目录到路径
sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))

def load_results(results_dir='./results'):
    """加载训练结果"""
    results = {}
    
    # 查找结果文件
    for file_path in Path(results_dir).glob('*_results.pth'):
        try:
            data = torch.load(file_path, map_location='cpu')
            
            # 提取模型类型和任务名称
            filename = file_path.stem
            if 'S-MNIST' in filename:
                task = 'S-MNIST'
            elif 'PS-MNIST' in filename:
                task = 'PS-MNIST'
            else:
                task = 'Unknown'
            
            if 'dh_srnn' in filename:
                model_type = 'DH-SRNN'
            elif 'vanilla_srnn' in filename:
                model_type = 'Vanilla SRNN'
            else:
                model_type = 'Unknown'
            
            key = f"{task}_{model_type}"
            results[key] = data
            
        except Exception as e:
            print(f"⚠️ 无法加载 {file_path}: {e}")
    
    return results

def plot_training_curves(results, save_dir='./results'):
    """绘制训练曲线对比"""
    
    # 创建子图
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('训练损失', '训练准确率', '测试准确率', '学习率变化'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    colors = px.colors.qualitative.Set1
    
    for i, (name, data) in enumerate(results.items()):
        color = colors[i % len(colors)]
        epochs = list(range(1, len(data['train_losses']) + 1))
        
        # 训练损失
        fig.add_trace(
            go.Scatter(x=epochs, y=data['train_losses'],
                      name=f'{name} - 训练损失',
                      line=dict(color=color),
                      showlegend=True),
            row=1, col=1
        )
        
        # 训练准确率
        fig.add_trace(
            go.Scatter(x=epochs, y=data['train_accs'],
                      name=f'{name} - 训练准确率',
                      line=dict(color=color),
                      showlegend=False),
            row=1, col=2
        )
        
        # 测试准确率
        fig.add_trace(
            go.Scatter(x=epochs, y=data['test_accs'],
                      name=f'{name} - 测试准确率',
                      line=dict(color=color, dash='dash'),
                      showlegend=False),
            row=2, col=1
        )
        
        # 学习率变化 (如果有的话)
        if 'lr_history' in data:
            fig.add_trace(
                go.Scatter(x=epochs, y=data['lr_history'],
                          name=f'{name} - 学习率',
                          line=dict(color=color, dash='dot'),
                          showlegend=False),
                row=2, col=2
            )
    
    # 更新布局
    fig.update_layout(
        title='Sequential MNIST 训练曲线对比',
        height=800,
        showlegend=True,
        legend=dict(x=1.05, y=1)
    )
    
    # 更新坐标轴标签
    fig.update_xaxes(title_text="Epoch", row=1, col=1)
    fig.update_xaxes(title_text="Epoch", row=1, col=2)
    fig.update_xaxes(title_text="Epoch", row=2, col=1)
    fig.update_xaxes(title_text="Epoch", row=2, col=2)
    
    fig.update_yaxes(title_text="Loss", row=1, col=1)
    fig.update_yaxes(title_text="Accuracy (%)", row=1, col=2)
    fig.update_yaxes(title_text="Accuracy (%)", row=2, col=1)
    fig.update_yaxes(title_text="Learning Rate", row=2, col=2)
    
    # 保存图表
    os.makedirs(save_dir, exist_ok=True)
    html_path = os.path.join(save_dir, 'sequential_mnist_training_curves.html')
    png_path = os.path.join(save_dir, 'sequential_mnist_training_curves.png')
    
    fig.write_html(html_path)
    fig.write_image(png_path, width=1200, height=800)
    
    print(f"📈 训练曲线已保存: {html_path}")
    return fig

def plot_performance_comparison(results, save_dir='./results'):
    """绘制性能对比柱状图"""
    
    # 提取最佳准确率
    model_names = []
    s_mnist_accs = []
    ps_mnist_accs = []
    
    for name, data in results.items():
        if 'S-MNIST' in name:
            model_type = name.replace('S-MNIST_', '')
            model_names.append(model_type)
            s_mnist_accs.append(data['best_acc'])
        elif 'PS-MNIST' in name:
            model_type = name.replace('PS-MNIST_', '')
            if model_type not in model_names:
                model_names.append(model_type)
                ps_mnist_accs.append(data['best_acc'])
            else:
                idx = model_names.index(model_type)
                if len(ps_mnist_accs) <= idx:
                    ps_mnist_accs.append(data['best_acc'])
                else:
                    ps_mnist_accs[idx] = data['best_acc']
    
    # 确保两个列表长度一致
    while len(s_mnist_accs) < len(model_names):
        s_mnist_accs.append(0)
    while len(ps_mnist_accs) < len(model_names):
        ps_mnist_accs.append(0)
    
    # 创建柱状图
    fig = go.Figure(data=[
        go.Bar(name='S-MNIST', x=model_names, y=s_mnist_accs,
               text=[f'{acc:.1f}%' for acc in s_mnist_accs],
               textposition='auto',
               marker_color='#1f77b4'),
        go.Bar(name='PS-MNIST', x=model_names, y=ps_mnist_accs,
               text=[f'{acc:.1f}%' for acc in ps_mnist_accs],
               textposition='auto',
               marker_color='#ff7f0e')
    ])
    
    fig.update_layout(
        title='Sequential MNIST 性能对比',
        xaxis_title='模型类型',
        yaxis_title='准确率 (%)',
        yaxis=dict(range=[0, 100]),
        barmode='group',
        height=500,
        showlegend=True
    )
    
    # 保存图表
    html_path = os.path.join(save_dir, 'sequential_mnist_performance_comparison.html')
    png_path = os.path.join(save_dir, 'sequential_mnist_performance_comparison.png')
    
    fig.write_html(html_path)
    fig.write_image(png_path, width=800, height=500)
    
    print(f"📊 性能对比图已保存: {html_path}")
    return fig

def create_summary_table(results, save_dir='./results'):
    """创建结果总结表"""
    
    print("\n📋 Sequential MNIST 实验结果总结")
    print("="*80)
    print(f"{'任务':<15} {'模型':<15} {'最佳准确率':<12} {'训练时间':<12} {'参数数量':<12}")
    print("-"*80)
    
    summary_data = []
    
    for name, data in results.items():
        task, model = name.split('_', 1)
        best_acc = data['best_acc']
        total_time = data.get('total_time', 0) / 3600  # 转换为小时
        
        # 尝试获取参数数量
        if 'args' in data and 'total_params' in data['args']:
            params = data['args']['total_params']
        else:
            params = 'N/A'
        
        print(f"{task:<15} {model:<15} {best_acc:>8.2f}%    {total_time:>8.2f}h    {params}")
        
        summary_data.append({
            'Task': task,
            'Model': model,
            'Best_Accuracy': best_acc,
            'Training_Time_Hours': total_time,
            'Parameters': params
        })
    
    # 保存为JSON
    json_path = os.path.join(save_dir, 'sequential_mnist_summary.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(summary_data, f, indent=2, ensure_ascii=False)
    
    print(f"\n📄 结果总结已保存: {json_path}")
    
    return summary_data

def analyze_performance_gains(results):
    """分析性能提升"""
    
    print("\n📊 性能分析:")
    print("="*50)
    
    # S-MNIST分析
    s_mnist_results = {k.replace('S-MNIST_', ''): v for k, v in results.items() if 'S-MNIST' in k}
    if 'Vanilla SRNN' in s_mnist_results and 'DH-SRNN' in s_mnist_results:
        baseline_acc = s_mnist_results['Vanilla SRNN']['best_acc']
        dh_acc = s_mnist_results['DH-SRNN']['best_acc']
        improvement = dh_acc - baseline_acc
        print(f"S-MNIST:")
        print(f"  Vanilla SRNN: {baseline_acc:.2f}%")
        print(f"  DH-SRNN:      {dh_acc:.2f}%")
        print(f"  提升:         +{improvement:.2f}%")
    
    # PS-MNIST分析
    ps_mnist_results = {k.replace('PS-MNIST_', ''): v for k, v in results.items() if 'PS-MNIST' in k}
    if 'Vanilla SRNN' in ps_mnist_results and 'DH-SRNN' in ps_mnist_results:
        baseline_acc = ps_mnist_results['Vanilla SRNN']['best_acc']
        dh_acc = ps_mnist_results['DH-SRNN']['best_acc']
        improvement = dh_acc - baseline_acc
        print(f"\nPS-MNIST:")
        print(f"  Vanilla SRNN: {baseline_acc:.2f}%")
        print(f"  DH-SRNN:      {dh_acc:.2f}%")
        print(f"  提升:         +{improvement:.2f}%")

def main():
    """主函数"""
    print("📊 Sequential MNIST 结果可视化")
    print("="*50)
    
    # 加载结果
    results = load_results()
    if not results:
        print("❌ 未找到训练结果文件")
        print("请先运行训练脚本生成结果")
        return
    
    print(f"✅ 加载了 {len(results)} 个实验结果")
    
    # 创建保存目录
    save_dir = './results'
    os.makedirs(save_dir, exist_ok=True)
    
    # 创建结果总结表
    create_summary_table(results, save_dir)
    
    # 分析性能提升
    analyze_performance_gains(results)
    
    # 绘制训练曲线
    print(f"\n🎨 绘制训练曲线...")
    plot_training_curves(results, save_dir)
    
    # 绘制性能对比
    print(f"\n🎨 绘制性能对比...")
    plot_performance_comparison(results, save_dir)
    
    print(f"\n🎉 可视化完成!")
    print(f"📁 所有图表保存在 {save_dir}/ 目录下")

if __name__ == '__main__':
    main()
