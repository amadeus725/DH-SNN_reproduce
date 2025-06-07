#!/usr/bin/env python3
"""
可视化并行实验结果
从results_parallel_4目录中的日志文件解析训练数据并生成图表
"""

import os
import re
import json
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
from pathlib import Path
from datetime import datetime

def parse_log_file(log_path):
    """解析日志文件，提取训练数据"""
    epochs = []
    train_losses = []
    train_accs = []
    test_losses = []
    test_accs = []
    
    with open(log_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 查找epoch完成的记录
    epoch_pattern = r'Epoch (\d+)/\d+: Train Acc: ([\d.]+)%, Test Acc: ([\d.]+)%.*?Best: ([\d.]+)%'
    matches = re.findall(epoch_pattern, content)
    
    for match in matches:
        epoch, train_acc, test_acc, best_acc = match
        epochs.append(int(epoch))
        train_accs.append(float(train_acc))
        test_accs.append(float(test_acc))
    
    return {
        'epochs': epochs,
        'train_acc': train_accs,
        'test_acc': test_accs
    }

def parse_training_progress(log_path):
    """解析训练进度，提取实时训练数据"""
    batch_data = []
    
    with open(log_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 查找训练进度记录
    progress_pattern = r'Epoch (\d+) \[Train\]:\s+(\d+)%.*?Loss=([\d.]+), Acc=([\d.]+)%'
    matches = re.findall(progress_pattern, content)
    
    for match in matches:
        epoch, progress, loss, acc = match
        batch_data.append({
            'epoch': int(epoch),
            'progress': int(progress),
            'loss': float(loss),
            'acc': float(acc)
        })
    
    return batch_data

def create_comparison_plots():
    """创建对比图表"""
    results_dir = Path('results_parallel_4')
    
    # 实验配置
    experiments = {
        'S-MNIST_DH-SRNN': {'name': 'S-MNIST DH-SRNN', 'color': '#1f77b4'},
        'S-MNIST_Vanilla-SRNN': {'name': 'S-MNIST Vanilla SRNN', 'color': '#ff7f0e'},
        'PS-MNIST_DH-SRNN': {'name': 'PS-MNIST DH-SRNN', 'color': '#2ca02c'},
        'PS-MNIST_Vanilla-SRNN': {'name': 'PS-MNIST Vanilla SRNN', 'color': '#d62728'}
    }
    
    # 解析所有实验数据
    all_data = {}
    progress_data = {}
    
    for exp_name in experiments.keys():
        log_file = results_dir / exp_name / f"{exp_name}.log"
        if log_file.exists():
            print(f"解析 {exp_name} 日志...")
            all_data[exp_name] = parse_log_file(log_file)
            progress_data[exp_name] = parse_training_progress(log_file)
        else:
            print(f"警告: 未找到 {log_file}")
    
    # 创建子图
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('测试准确率对比', '训练准确率对比', '实时训练损失', '当前训练进度'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # 绘制测试准确率
    for exp_name, data in all_data.items():
        if data['epochs']:
            fig.add_trace(
                go.Scatter(
                    x=data['epochs'],
                    y=data['test_acc'],
                    mode='lines+markers',
                    name=f"{experiments[exp_name]['name']}",
                    line=dict(color=experiments[exp_name]['color']),
                    legendgroup=exp_name
                ),
                row=1, col=1
            )
    
    # 绘制训练准确率
    for exp_name, data in all_data.items():
        if data['epochs']:
            fig.add_trace(
                go.Scatter(
                    x=data['epochs'],
                    y=data['train_acc'],
                    mode='lines+markers',
                    name=f"{experiments[exp_name]['name']} - 训练",
                    line=dict(color=experiments[exp_name]['color'], dash='dash'),
                    legendgroup=exp_name,
                    showlegend=False
                ),
                row=1, col=2
            )
    
    # 绘制实时训练损失（最新的epoch）
    for exp_name, prog_data in progress_data.items():
        if prog_data:
            # 获取最新epoch的数据
            latest_epoch = max([d['epoch'] for d in prog_data])
            latest_data = [d for d in prog_data if d['epoch'] == latest_epoch]
            
            if latest_data:
                progress_vals = [d['progress'] for d in latest_data]
                loss_vals = [d['loss'] for d in latest_data]
                
                fig.add_trace(
                    go.Scatter(
                        x=progress_vals,
                        y=loss_vals,
                        mode='lines',
                        name=f"Epoch {latest_epoch}",
                        line=dict(color=experiments[exp_name]['color']),
                        legendgroup=exp_name,
                        showlegend=False
                    ),
                    row=2, col=1
                )
    
    # 绘制实时训练准确率（最新的epoch）
    for exp_name, prog_data in progress_data.items():
        if prog_data:
            # 获取最新epoch的数据
            latest_epoch = max([d['epoch'] for d in prog_data])
            latest_data = [d for d in prog_data if d['epoch'] == latest_epoch]
            
            if latest_data:
                progress_vals = [d['progress'] for d in latest_data]
                acc_vals = [d['acc'] for d in latest_data]
                
                fig.add_trace(
                    go.Scatter(
                        x=progress_vals,
                        y=acc_vals,
                        mode='lines',
                        name=f"Epoch {latest_epoch}",
                        line=dict(color=experiments[exp_name]['color']),
                        legendgroup=exp_name,
                        showlegend=False
                    ),
                    row=2, col=2
                )
    
    # 更新布局
    fig.update_layout(
        title=dict(
            text="DH-SNN Parallel Experiments Results (Training Interrupted)",
            x=0.5,
            font=dict(size=20)
        ),
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
    fig.update_xaxes(title_text="Training Progress (%)", row=2, col=1)
    fig.update_xaxes(title_text="Training Progress (%)", row=2, col=2)
    
    fig.update_yaxes(title_text="Test Accuracy (%)", row=1, col=1)
    fig.update_yaxes(title_text="Train Accuracy (%)", row=1, col=2)
    fig.update_yaxes(title_text="Loss", row=2, col=1)
    fig.update_yaxes(title_text="Accuracy (%)", row=2, col=2)
    
    return fig, all_data

def generate_summary_report(all_data):
    """生成实验总结报告"""
    report = {
        'timestamp': datetime.now().isoformat(),
        'status': 'Training Interrupted - Due to Force Majeure',
        'experiments': {}
    }
    
    for exp_name, data in all_data.items():
        if data['epochs']:
            latest_epoch = max(data['epochs'])
            latest_idx = data['epochs'].index(latest_epoch)
            
            report['experiments'][exp_name] = {
                'completed_epochs': latest_epoch,
                'latest_train_acc': data['train_acc'][latest_idx],
                'latest_test_acc': data['test_acc'][latest_idx],
                'best_test_acc': max(data['test_acc']) if data['test_acc'] else 0,
                'best_test_acc_epoch': data['epochs'][data['test_acc'].index(max(data['test_acc']))] if data['test_acc'] else 0
            }
        else:
            report['experiments'][exp_name] = {
                'status': 'Training not completed or log parsing failed'
            }
    
    return report

def main():
    """主函数"""
    print("🔍 Parsing parallel experiment results...")
    
    # 创建对比图表
    fig, all_data = create_comparison_plots()
    
    # 保存HTML图表
    html_path = 'results_parallel_4/parallel_experiments_comparison.html'
    fig.write_html(html_path)
    print(f"📊 HTML chart saved: {html_path}")
    
    # 保存PNG图表
    try:
        png_path = 'results_parallel_4/parallel_experiments_comparison.png'
        pio.write_image(fig, png_path, scale=5, width=1200, height=800)
        print(f"🖼️  PNG chart saved: {png_path}")
    except Exception as e:
        print(f"⚠️  PNG save failed: {e}")
        print("Hint: Install kaleido: pip install kaleido==0.1.*")
    
    # 生成总结报告
    report = generate_summary_report(all_data)
    
    # 保存报告
    report_path = 'results_parallel_4/experiment_summary.json'
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"📄 Experiment report saved: {report_path}")
    
    # 打印总结
    print("\n" + "="*60)
    print("📋 Experiment Summary")
    print("="*60)
    
    for exp_name, exp_data in report['experiments'].items():
        if 'completed_epochs' in exp_data:
            print(f"\n🔬 {exp_name}:")
            print(f"  Completed epochs: {exp_data['completed_epochs']}")
            print(f"  Latest test accuracy: {exp_data['latest_test_acc']:.2f}%")
            print(f"  Best test accuracy: {exp_data['best_test_acc']:.2f}% (Epoch {exp_data['best_test_acc_epoch']})")
        else:
            print(f"\n❌ {exp_name}: {exp_data['status']}")
    
    print(f"\n⚠️  Status: {report['status']}")
    print("="*60)

if __name__ == '__main__':
    main()
