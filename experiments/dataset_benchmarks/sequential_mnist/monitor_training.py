#!/usr/bin/env python3
"""
Sequential MNIST训练监控脚本
实时监控训练进度和结果
"""

import os
import time
import torch
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px

def load_results(results_dir='./results'):
    """加载训练结果"""
    results = {}
    
    # 检查结果文件
    files_to_check = [
        'S-MNIST_vanilla_srnn_results.pth',
        'S-MNIST_dh_srnn_results.pth',
        'PS-MNIST_vanilla_srnn_results.pth', 
        'PS-MNIST_dh_srnn_results.pth'
    ]
    
    for filename in files_to_check:
        filepath = os.path.join(results_dir, filename)
        if os.path.exists(filepath):
            try:
                data = torch.load(filepath, map_location='cpu')
                results[filename] = data
                print(f"✅ 加载结果: {filename}")
            except Exception as e:
                print(f"❌ 加载失败 {filename}: {e}")
        else:
            print(f"⏳ 等待结果: {filename}")
    
    return results

def plot_training_curves(results):
    """绘制训练曲线"""
    if not results:
        print("📊 暂无结果可绘制")
        return
    
    # 创建子图
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('训练损失', '测试损失', '训练准确率', '测试准确率'),
        vertical_spacing=0.1
    )
    
    colors = {
        'vanilla_srnn': '#1f77b4',
        'dh_srnn': '#ff7f0e'
    }
    
    for filename, data in results.items():
        # 解析文件名
        if 'vanilla_srnn' in filename:
            model_name = 'Vanilla SRNN'
            color = colors['vanilla_srnn']
        elif 'dh_srnn' in filename:
            model_name = 'DH-SRNN'
            color = colors['dh_srnn']
        else:
            continue
            
        if 'PS-MNIST' in filename:
            task = 'PS-MNIST'
            line_dash = 'dash'
        else:
            task = 'S-MNIST'
            line_dash = 'solid'
        
        legend_name = f"{model_name} ({task})"
        
        # 获取数据
        epochs = list(range(1, len(data['train_losses']) + 1))
        
        # 训练损失
        fig.add_trace(
            go.Scatter(
                x=epochs, y=data['train_losses'],
                name=legend_name,
                line=dict(color=color, dash=line_dash),
                showlegend=True
            ),
            row=1, col=1
        )
        
        # 测试损失
        fig.add_trace(
            go.Scatter(
                x=epochs, y=data['test_losses'],
                name=legend_name,
                line=dict(color=color, dash=line_dash),
                showlegend=False
            ),
            row=1, col=2
        )
        
        # 训练准确率
        fig.add_trace(
            go.Scatter(
                x=epochs, y=data['train_accs'],
                name=legend_name,
                line=dict(color=color, dash=line_dash),
                showlegend=False
            ),
            row=2, col=1
        )
        
        # 测试准确率
        fig.add_trace(
            go.Scatter(
                x=epochs, y=data['test_accs'],
                name=legend_name,
                line=dict(color=color, dash=line_dash),
                showlegend=False
            ),
            row=2, col=2
        )
    
    # 更新布局
    fig.update_layout(
        title='Sequential MNIST 训练进度监控',
        height=600,
        showlegend=True
    )
    
    # 更新坐标轴
    fig.update_xaxes(title_text="Epoch")
    fig.update_yaxes(title_text="Loss", row=1, col=1)
    fig.update_yaxes(title_text="Loss", row=1, col=2)
    fig.update_yaxes(title_text="Accuracy (%)", row=2, col=1)
    fig.update_yaxes(title_text="Accuracy (%)", row=2, col=2)
    
    # 保存图表
    fig.write_html('./results/training_monitor.html')
    print("📈 训练曲线已保存: ./results/training_monitor.html")
    
    return fig

def print_summary(results):
    """打印结果摘要"""
    if not results:
        return
    
    print("\n" + "="*60)
    print("📊 Sequential MNIST 训练结果摘要")
    print("="*60)
    
    for filename, data in results.items():
        # 解析文件名
        if 'vanilla_srnn' in filename:
            model_name = 'Vanilla SRNN'
        elif 'dh_srnn' in filename:
            model_name = 'DH-SRNN'
        else:
            continue
            
        if 'PS-MNIST' in filename:
            task = 'PS-MNIST'
        else:
            task = 'S-MNIST'
        
        # 获取最佳结果
        best_acc = data['best_acc']
        total_time = data.get('total_time', 0)
        epochs_completed = len(data['train_losses'])
        
        print(f"\n🧠 {model_name} - {task}:")
        print(f"   最佳准确率: {best_acc:.2f}%")
        print(f"   完成轮数: {epochs_completed}/150")
        print(f"   训练时间: {total_time/3600:.2f} 小时")
        
        if epochs_completed > 0:
            latest_train_acc = data['train_accs'][-1]
            latest_test_acc = data['test_accs'][-1]
            print(f"   最新训练准确率: {latest_train_acc:.2f}%")
            print(f"   最新测试准确率: {latest_test_acc:.2f}%")

def monitor_loop(interval=60):
    """监控循环"""
    print("🔍 开始监控 Sequential MNIST 训练...")
    print(f"⏱️  监控间隔: {interval} 秒")
    print("按 Ctrl+C 停止监控")
    
    try:
        while True:
            print(f"\n⏰ {time.strftime('%Y-%m-%d %H:%M:%S')}")
            
            # 加载结果
            results = load_results()
            
            # 绘制图表
            if results:
                plot_training_curves(results)
                print_summary(results)
            
            # 等待
            time.sleep(interval)
            
    except KeyboardInterrupt:
        print("\n🛑 监控已停止")

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Sequential MNIST训练监控')
    parser.add_argument('--interval', type=int, default=300,
                       help='监控间隔(秒)')
    parser.add_argument('--once', action='store_true',
                       help='只运行一次，不循环监控')
    
    args = parser.parse_args()
    
    if args.once:
        print("📊 生成当前训练结果...")
        results = load_results()
        if results:
            plot_training_curves(results)
            print_summary(results)
        else:
            print("❌ 暂无训练结果")
    else:
        monitor_loop(args.interval)
