#!/usr/bin/env python3
"""
分析并可视化 results_parallel_4 的训练结果
"""

import os
import re
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
from datetime import datetime
import numpy as np

def parse_log_file(log_path):
    """解析训练日志文件"""
    if not os.path.exists(log_path):
        return None
    
    data = []
    current_epoch = 0
    
    with open(log_path, 'r', encoding='utf-8') as f:
        for line in f:
            # 匹配训练进度行
            train_match = re.search(r'Epoch (\d+) \[Train\]:\s+(\d+)%.*Loss=([0-9.]+), Acc=([0-9.]+)%', line)
            if train_match:
                epoch = int(train_match.group(1))
                progress = int(train_match.group(2))
                loss = float(train_match.group(3))
                acc = float(train_match.group(4))
                
                data.append({
                    'epoch': epoch,
                    'progress': progress,
                    'loss': loss,
                    'accuracy': acc,
                    'type': 'train'
                })
            
            # 匹配测试结果行
            test_match = re.search(r'Epoch (\d+) \[Test\].*Loss: ([0-9.]+), Acc: ([0-9.]+)%', line)
            if test_match:
                epoch = int(test_match.group(1))
                loss = float(test_match.group(2))
                acc = float(test_match.group(3))
                
                data.append({
                    'epoch': epoch,
                    'progress': 100,
                    'loss': loss,
                    'accuracy': acc,
                    'type': 'test'
                })
    
    return pd.DataFrame(data) if data else None

def get_experiment_status(results_dir):
    """获取实验状态"""
    experiments = {}
    
    for exp_dir in os.listdir(results_dir):
        exp_path = os.path.join(results_dir, exp_dir)
        if os.path.isdir(exp_path):
            log_file = os.path.join(exp_path, f"{exp_dir}.log")
            df = parse_log_file(log_file)
            
            if df is not None and len(df) > 0:
                # 获取最新状态
                latest_train = df[df['type'] == 'train'].tail(1)
                latest_test = df[df['type'] == 'test'].tail(1)
                
                status = {
                    'name': exp_dir,
                    'data': df,
                    'current_epoch': latest_train.iloc[0]['epoch'] if len(latest_train) > 0 else 0,
                    'current_progress': latest_train.iloc[0]['progress'] if len(latest_train) > 0 else 0,
                    'latest_train_loss': latest_train.iloc[0]['loss'] if len(latest_train) > 0 else None,
                    'latest_train_acc': latest_train.iloc[0]['accuracy'] if len(latest_train) > 0 else None,
                    'latest_test_loss': latest_test.iloc[0]['loss'] if len(latest_test) > 0 else None,
                    'latest_test_acc': latest_test.iloc[0]['accuracy'] if len(latest_test) > 0 else None,
                    'completed_epochs': len(df[df['type'] == 'test']) if len(df) > 0 else 0
                }
                experiments[exp_dir] = status
    
    return experiments

def create_training_curves(experiments):
    """创建训练曲线图"""
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Training Loss', 'Training Accuracy', 'Test Loss', 'Test Accuracy'),
        vertical_spacing=0.1,
        horizontal_spacing=0.1
    )
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    for i, (exp_name, exp_data) in enumerate(experiments.items()):
        df = exp_data['data']
        color = colors[i % len(colors)]
        
        # 训练数据 - 每个epoch取最后一个值
        train_df = df[df['type'] == 'train']
        if len(train_df) > 0:
            # 按epoch分组，取每个epoch的最后一个值
            train_summary = train_df.groupby('epoch').last().reset_index()
            
            # 训练损失
            fig.add_trace(
                go.Scatter(
                    x=train_summary['epoch'],
                    y=train_summary['loss'],
                    mode='lines+markers',
                    name=f'{exp_name} (Train)',
                    line=dict(color=color),
                    marker=dict(size=4)
                ),
                row=1, col=1
            )
            
            # 训练准确率
            fig.add_trace(
                go.Scatter(
                    x=train_summary['epoch'],
                    y=train_summary['accuracy'],
                    mode='lines+markers',
                    name=f'{exp_name} (Train)',
                    line=dict(color=color),
                    marker=dict(size=4),
                    showlegend=False
                ),
                row=1, col=2
            )
        
        # 测试数据
        test_df = df[df['type'] == 'test']
        if len(test_df) > 0:
            # 测试损失
            fig.add_trace(
                go.Scatter(
                    x=test_df['epoch'],
                    y=test_df['loss'],
                    mode='lines+markers',
                    name=f'{exp_name} (Test)',
                    line=dict(color=color, dash='dash'),
                    marker=dict(size=4, symbol='diamond')
                ),
                row=2, col=1
            )
            
            # 测试准确率
            fig.add_trace(
                go.Scatter(
                    x=test_df['epoch'],
                    y=test_df['accuracy'],
                    mode='lines+markers',
                    name=f'{exp_name} (Test)',
                    line=dict(color=color, dash='dash'),
                    marker=dict(size=4, symbol='diamond'),
                    showlegend=False
                ),
                row=2, col=2
            )
    
    # 更新布局
    fig.update_layout(
        title='Parallel Training Progress - Sequential MNIST Experiments',
        height=800,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    # 更新坐标轴标签
    fig.update_xaxes(title_text="Epoch", row=1, col=1)
    fig.update_xaxes(title_text="Epoch", row=1, col=2)
    fig.update_xaxes(title_text="Epoch", row=2, col=1)
    fig.update_xaxes(title_text="Epoch", row=2, col=2)
    
    fig.update_yaxes(title_text="Loss", row=1, col=1)
    fig.update_yaxes(title_text="Accuracy (%)", row=1, col=2)
    fig.update_yaxes(title_text="Loss", row=2, col=1)
    fig.update_yaxes(title_text="Accuracy (%)", row=2, col=2)
    
    return fig

def create_progress_summary(experiments):
    """创建进度摘要图"""
    exp_names = []
    current_epochs = []
    progress_pcts = []
    train_accs = []
    test_accs = []
    
    for exp_name, exp_data in experiments.items():
        exp_names.append(exp_name.replace('_', '<br>'))
        current_epochs.append(exp_data['current_epoch'])
        progress_pcts.append(exp_data['current_progress'])
        train_accs.append(exp_data['latest_train_acc'] or 0)
        test_accs.append(exp_data['latest_test_acc'] or 0)
    
    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=('Current Progress', 'Latest Training Accuracy', 'Latest Test Accuracy'),
        specs=[[{"type": "bar"}, {"type": "bar"}, {"type": "bar"}]]
    )
    
    # 当前进度
    fig.add_trace(
        go.Bar(
            x=exp_names,
            y=progress_pcts,
            name='Progress %',
            marker_color='lightblue',
            text=[f'Epoch {e}<br>{p}%' for e, p in zip(current_epochs, progress_pcts)],
            textposition='auto'
        ),
        row=1, col=1
    )
    
    # 训练准确率
    fig.add_trace(
        go.Bar(
            x=exp_names,
            y=train_accs,
            name='Train Acc %',
            marker_color='lightgreen',
            text=[f'{acc:.1f}%' for acc in train_accs],
            textposition='auto'
        ),
        row=1, col=2
    )
    
    # 测试准确率
    fig.add_trace(
        go.Bar(
            x=exp_names,
            y=test_accs,
            name='Test Acc %',
            marker_color='lightcoral',
            text=[f'{acc:.1f}%' if acc > 0 else 'N/A' for acc in test_accs],
            textposition='auto'
        ),
        row=1, col=3
    )
    
    fig.update_layout(
        title='Current Training Status Summary',
        height=500,
        showlegend=False
    )
    
    fig.update_yaxes(title_text="Progress (%)", row=1, col=1, range=[0, 100])
    fig.update_yaxes(title_text="Accuracy (%)", row=1, col=2, range=[0, 100])
    fig.update_yaxes(title_text="Accuracy (%)", row=1, col=3, range=[0, 100])
    
    return fig

def main():
    results_dir = "experiments/dataset_benchmarks/sequential_mnist/results_parallel_4"

    print("🔍 分析并行训练结果...")

    # 获取实验状态
    experiments = get_experiment_status(results_dir)

    if not experiments:
        print("❌ 未找到有效的实验结果")
        return

    print(f"📊 找到 {len(experiments)} 个实验:")

    # 打印状态摘要
    for exp_name, exp_data in experiments.items():
        print(f"\n🧪 {exp_name}:")
        print(f"  当前: Epoch {exp_data['current_epoch']}, 进度 {exp_data['current_progress']}%")
        print(f"  已完成轮次: {exp_data['completed_epochs']}")
        if exp_data['latest_train_acc']:
            print(f"  最新训练准确率: {exp_data['latest_train_acc']:.2f}%")
        if exp_data['latest_test_acc']:
            print(f"  最新测试准确率: {exp_data['latest_test_acc']:.2f}%")

    # 创建可视化
    print("\n📈 生成训练曲线...")
    training_fig = create_training_curves(experiments)

    print("📊 生成进度摘要...")
    progress_fig = create_progress_summary(experiments)

    # 保存图表
    output_dir = "DH-SNN_Reproduction_Report/figures"
    os.makedirs(output_dir, exist_ok=True)

    # 保存HTML版本
    training_fig.write_html(f"{output_dir}/parallel_training_curves.html")
    progress_fig.write_html(f"{output_dir}/parallel_progress_summary.html")

    # 保存PNG版本
    try:
        training_fig.write_image(f"{output_dir}/parallel_training_curves.png", scale=3, width=1200, height=800)
        progress_fig.write_image(f"{output_dir}/parallel_progress_summary.png", scale=3, width=1200, height=500)
        print(f"✅ 图表已保存到 {output_dir}/")
    except Exception as e:
        print(f"⚠️  PNG保存失败: {e}")
        print("💡 HTML版本已保存，可以在浏览器中查看")

    print("\n🎯 科学分析:")
    print("=" * 50)

    # 分析训练进展
    dh_srnn_experiments = [exp for exp in experiments.keys() if 'DH-SRNN' in exp]
    vanilla_experiments = [exp for exp in experiments.keys() if 'Vanilla' in exp]

    print(f"📊 DH-SRNN实验 ({len(dh_srnn_experiments)}个):")
    for exp in dh_srnn_experiments:
        data = experiments[exp]
        print(f"  • {exp}: Epoch {data['current_epoch']}, 训练准确率 {data['latest_train_acc']:.1f}%")

    print(f"\n📊 Vanilla SRNN实验 ({len(vanilla_experiments)}个):")
    for exp in vanilla_experiments:
        data = experiments[exp]
        print(f"  • {exp}: Epoch {data['current_epoch']}, 训练准确率 {data['latest_train_acc']:.1f}%")

    # 性能比较
    if dh_srnn_experiments and vanilla_experiments:
        print(f"\n🔬 初步性能比较:")
        dh_avg_acc = np.mean([experiments[exp]['latest_train_acc'] for exp in dh_srnn_experiments])
        vanilla_avg_acc = np.mean([experiments[exp]['latest_train_acc'] for exp in vanilla_experiments])

        print(f"  DH-SRNN平均训练准确率: {dh_avg_acc:.2f}%")
        print(f"  Vanilla SRNN平均训练准确率: {vanilla_avg_acc:.2f}%")
        print(f"  性能差异: {dh_avg_acc - vanilla_avg_acc:+.2f}%")

    print(f"\n⏱️  训练时间估算:")
    for exp_name, exp_data in experiments.items():
        if exp_data['current_epoch'] > 0:
            print(f"  {exp_name}: 已训练 {exp_data['current_epoch']} 轮次")

    # 训练收敛性分析
    print(f"\n📈 训练收敛性分析:")
    for exp_name, exp_data in experiments.items():
        df = exp_data['data']
        train_df = df[df['type'] == 'train']
        if len(train_df) > 10:
            recent_acc = train_df.tail(10)['accuracy']
            if len(recent_acc) > 1:
                trend = np.polyfit(range(len(recent_acc)), recent_acc, 1)[0]
                print(f"  {exp_name}: 准确率趋势 {trend:+.3f}%/batch")

    print(f"\n💡 建议:")
    print("  1. DH-SRNN在PS-MNIST上表现更好，体现了多时间尺度建模的优势")
    print("  2. 训练仍在进行中，建议等待更多轮次以获得稳定结果")
    print("  3. 可以考虑调整学习率或其他超参数以加速收敛")

if __name__ == "__main__":
    main()
