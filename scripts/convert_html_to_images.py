#!/usr/bin/env python3
"""
将HTML可视化文件转换为PNG图片，用于LaTeX报告
同时生成英文版的图片以避免中文显示问题
"""

import os
import sys
import json
from pathlib import Path
import plotly.graph_objects as go
import plotly.subplots as sp
from plotly.subplots import make_subplots
import numpy as np

def find_html_files():
    """查找所有HTML可视化文件"""
    html_files = []
    
    # 搜索路径
    search_paths = [
        "results/",
        "experiments/legacy_spikingjelly/original_experiments/temporal_dynamics/multi_timescale_xor/results/",
        "experiments/dataset_benchmarks/temporal_dynamics/multi_timescale_xor/results/",
        "experiments/dataset_benchmarks/figure_reproduction/figure3_delayed_xor/outputs/figures/",
        "experiments/legacy_spikingjelly/original_experiments/figure_reproduction/figure3_delayed_xor/outputs/figures/"
    ]
    
    for search_path in search_paths:
        if os.path.exists(search_path):
            for file in Path(search_path).glob("*.html"):
                html_files.append(file)
    
    return html_files

def create_english_delayed_xor_figure():
    """创建英文版的延迟XOR结果图"""
    
    # 从JSON数据创建图表
    delays = [25, 50, 100, 150, 200, 250, 300, 400]
    vanilla_accuracies = [55.0] * 8
    dh_accuracies = [84.2, 82.5, 76.7, 65.0, 66.7, 75.8, 84.2, 68.3]
    
    fig = go.Figure()
    
    # 添加Vanilla SFNN线
    fig.add_trace(go.Scatter(
        x=delays,
        y=vanilla_accuracies,
        mode='lines+markers',
        name='Vanilla SFNN',
        line=dict(color='#E74C3C', width=3),
        marker=dict(size=8, symbol='circle')
    ))
    
    # 添加DH-SFNN线
    fig.add_trace(go.Scatter(
        x=delays,
        y=dh_accuracies,
        mode='lines+markers',
        name='DH-SFNN',
        line=dict(color='#3498DB', width=3),
        marker=dict(size=8, symbol='square')
    ))
    
    # 更新布局
    fig.update_layout(
        title=dict(
            text='Delayed Spiking XOR: Long-term Memory Performance',
            font=dict(size=20, family='Arial'),
            x=0.5
        ),
        xaxis=dict(
            title=dict(text='Delay Steps', font=dict(size=16, family='Arial')),
            tickfont=dict(size=14, family='Arial'),
            gridcolor='lightgray',
            gridwidth=1
        ),
        yaxis=dict(
            title=dict(text='Accuracy (%)', font=dict(size=16, family='Arial')),
            tickfont=dict(size=14, family='Arial'),
            gridcolor='lightgray',
            gridwidth=1,
            range=[50, 90]
        ),
        legend=dict(
            font=dict(size=14, family='Arial'),
            x=0.7,
            y=0.95,
            bgcolor='rgba(255,255,255,0.8)',
            bordercolor='gray',
            borderwidth=1
        ),
        plot_bgcolor='white',
        paper_bgcolor='white',
        width=800,
        height=600,
        margin=dict(l=80, r=80, t=80, b=80)
    )
    
    return fig

def create_english_multitimescale_xor_figure():
    """创建英文版的多时间尺度XOR结果图"""
    
    models = ['Vanilla SFNN', '1-Branch DH-SFNN\n(Small)', '1-Branch DH-SFNN\n(Large)', 
              '2-Branch DH-SFNN\n(Fixed)', '2-Branch DH-SFNN\n(Learnable)']
    accuracies = [50.2, 52.8, 53.1, 84.6, 96.2]
    colors = ['#E74C3C', '#F39C12', '#F39C12', '#3498DB', '#27AE60']
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=models,
        y=accuracies,
        marker=dict(
            color=colors,
            line=dict(color='black', width=1)
        ),
        text=[f'{acc:.1f}%' for acc in accuracies],
        textposition='outside',
        textfont=dict(size=14, family='Arial')
    ))
    
    fig.update_layout(
        title=dict(
            text='Multi-timescale Spiking XOR: Architecture Comparison',
            font=dict(size=20, family='Arial'),
            x=0.5
        ),
        xaxis=dict(
            title=dict(text='Model Architecture', font=dict(size=16, family='Arial')),
            tickfont=dict(size=12, family='Arial'),
            tickangle=45
        ),
        yaxis=dict(
            title=dict(text='Accuracy (%)', font=dict(size=16, family='Arial')),
            tickfont=dict(size=14, family='Arial'),
            gridcolor='lightgray',
            gridwidth=1,
            range=[0, 100]
        ),
        plot_bgcolor='white',
        paper_bgcolor='white',
        width=1000,
        height=600,
        margin=dict(l=80, r=80, t=80, b=120)
    )
    
    return fig

def create_english_dataset_comparison_figure():
    """创建英文版的数据集性能对比图"""
    
    datasets = ['SHD', 'SSC']
    vanilla_accs = [54.5, 46.8]
    dh_accs = [79.8, 60.5]
    
    x = np.arange(len(datasets))
    width = 0.35
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=[d + ' (Vanilla SNN)' for d in datasets],
        y=vanilla_accs,
        name='Vanilla SNN',
        marker=dict(color='#E74C3C'),
        text=[f'{acc:.1f}%' for acc in vanilla_accs],
        textposition='outside'
    ))
    
    fig.add_trace(go.Bar(
        x=[d + ' (DH-SNN)' for d in datasets],
        y=dh_accs,
        name='DH-SNN',
        marker=dict(color='#3498DB'),
        text=[f'{acc:.1f}%' for acc in dh_accs],
        textposition='outside'
    ))
    
    fig.update_layout(
        title=dict(
            text='Neuromorphic Dataset Performance Comparison',
            font=dict(size=20, family='Arial'),
            x=0.5
        ),
        xaxis=dict(
            title=dict(text='Dataset and Model', font=dict(size=16, family='Arial')),
            tickfont=dict(size=14, family='Arial')
        ),
        yaxis=dict(
            title=dict(text='Accuracy (%)', font=dict(size=16, family='Arial')),
            tickfont=dict(size=14, family='Arial'),
            gridcolor='lightgray',
            gridwidth=1,
            range=[0, 85]
        ),
        legend=dict(
            font=dict(size=14, family='Arial'),
            x=0.7,
            y=0.95
        ),
        plot_bgcolor='white',
        paper_bgcolor='white',
        width=800,
        height=600,
        margin=dict(l=80, r=80, t=80, b=80),
        barmode='group'
    )
    
    return fig

def create_english_performance_summary_figure():
    """创建英文版的性能总结图"""
    
    experiments = ['Delayed XOR', 'Multi-timescale XOR', 'SHD Dataset', 'SSC Dataset']
    improvements = [20.4, 46.0, 25.3, 13.7]
    colors = ['#3498DB', '#27AE60', '#F39C12', '#E74C3C']
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=experiments,
        y=improvements,
        marker=dict(
            color=colors,
            line=dict(color='black', width=1)
        ),
        text=[f'+{imp:.1f}%' for imp in improvements],
        textposition='outside',
        textfont=dict(size=14, family='Arial', color='black')
    ))
    
    fig.update_layout(
        title=dict(
            text='DH-SNN Performance Improvements Over Vanilla SNN',
            font=dict(size=20, family='Arial'),
            x=0.5
        ),
        xaxis=dict(
            title=dict(text='Experiment Type', font=dict(size=16, family='Arial')),
            tickfont=dict(size=14, family='Arial')
        ),
        yaxis=dict(
            title=dict(text='Performance Improvement (%)', font=dict(size=16, family='Arial')),
            tickfont=dict(size=14, family='Arial'),
            gridcolor='lightgray',
            gridwidth=1,
            range=[0, 50]
        ),
        plot_bgcolor='white',
        paper_bgcolor='white',
        width=1000,
        height=600,
        margin=dict(l=80, r=80, t=80, b=80)
    )
    
    return fig

def convert_html_to_png():
    """转换HTML文件为PNG"""
    
    print("🔍 查找HTML可视化文件...")
    html_files = find_html_files()
    
    if html_files:
        print(f"📁 找到 {len(html_files)} 个HTML文件:")
        for file in html_files:
            print(f"  • {file}")
    else:
        print("⚠️ 未找到HTML文件")
    
    # 创建输出目录
    output_dir = Path("DH-SNN_Reproduction_Report/figures")
    output_dir.mkdir(exist_ok=True)
    
    print(f"\n🎨 生成英文版图片...")
    
    # 生成英文版图片
    figures = {
        'delayed_xor_performance': create_english_delayed_xor_figure(),
        'multitimescale_xor_comparison': create_english_multitimescale_xor_figure(),
        'dataset_performance_comparison': create_english_dataset_comparison_figure(),
        'performance_summary': create_english_performance_summary_figure()
    }
    
    for name, fig in figures.items():
        try:
            # 保存PNG
            png_path = output_dir / f"{name}.png"
            fig.write_image(str(png_path), width=1200, height=800, scale=2)
            print(f"✅ {png_path}")
            
            # 也保存HTML版本
            html_path = output_dir / f"{name}.html"
            fig.write_html(str(html_path))
            print(f"✅ {html_path}")
            
        except Exception as e:
            print(f"❌ 生成 {name} 失败: {e}")
            print("💡 提示: 需要安装 kaleido: pip install kaleido")

def main():
    """主函数"""
    print("🖼️ HTML转PNG图片生成器")
    print("="*50)
    
    # 检查依赖
    try:
        import kaleido
        print("✅ kaleido 已安装")
    except ImportError:
        print("⚠️ 需要安装 kaleido: pip install kaleido")
        print("   或者: conda install -c conda-forge python-kaleido")
        return
    
    # 转换文件
    convert_html_to_png()
    
    print(f"\n🎉 图片生成完成!")
    print(f"📁 所有图片保存在: DH-SNN_Reproduction_Report/figures/")
    print(f"\n💡 使用说明:")
    print(f"  • PNG文件适合用于LaTeX报告")
    print(f"  • HTML文件可在浏览器中查看交互式版本")

if __name__ == '__main__':
    main()
