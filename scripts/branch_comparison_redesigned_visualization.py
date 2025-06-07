#!/usr/bin/env python3
"""
重新设计的分支数量对比可视化
采用新颖的可视化形式，避免重复的四面板格式
"""

import plotly.graph_objects as go
import plotly.subplots as sp
from plotly.subplots import make_subplots
import numpy as np
import json

def create_branch_comparison_redesigned():
    """创建重新设计的分支数量对比可视化"""
    
    # 实验数据
    branch_nums = [1, 2, 4, 8]
    mean_accs = [59.1, 97.3, 96.5, 95.1]
    std_accs = [1.5, 0.3, 0.4, 0.9]
    
    # 创建1x2布局的子图
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=[
            'Multi-timescale XOR Task: Branch Number Analysis',
            'Performance-Complexity Trade-off Analysis'
        ],
        specs=[[{"secondary_y": False}, {"secondary_y": True}]],
        horizontal_spacing=0.12
    )
    
    # 左图：性能曲线与最优点标注
    # 主要性能曲线
    fig.add_trace(go.Scatter(
        x=branch_nums,
        y=mean_accs,
        error_y=dict(type='data', array=std_accs, visible=True, thickness=2),
        mode='lines+markers',
        name='Test Accuracy',
        line=dict(color='#2E86AB', width=4),
        marker=dict(size=12, symbol='circle'),
        hovertemplate='<b>%{x}B</b><br>Accuracy: %{y:.1f}%<br>Std: ±%{error_y.array:.1f}%<extra></extra>'
    ), row=1, col=1)
    
    # 突出显示最优点
    fig.add_trace(go.Scatter(
        x=[2],
        y=[97.3],
        mode='markers',
        name='Optimal Configuration',
        marker=dict(
            size=20,
            color='#FF6B6B',
            symbol='star',
            line=dict(width=3, color='white')
        ),
        hovertemplate='<b>Optimal: 2B</b><br>Accuracy: 97.3%<br>Std: ±0.3%<extra></extra>'
    ), row=1, col=1)
    
    # 添加性能区间标注
    fig.add_shape(
        type="rect",
        x0=1.5, y0=95, x1=2.5, y1=100,
        fillcolor="rgba(255, 107, 107, 0.2)",
        line=dict(color="rgba(255, 107, 107, 0.5)", width=2),
        row=1, col=1
    )
    
    # 添加文字标注 - 调整位置避免遮挡线条
    fig.add_annotation(
        x=1.2, y=88,
        text="<b>Optimal Zone</b><br>97.3% ± 0.3%<br>Best stability",
        showarrow=True,
        arrowhead=2,
        arrowcolor="#FF6B6B",
        arrowwidth=2,
        ax=2, ay=97.3,  # 箭头指向最优点
        bgcolor="rgba(255,255,255,0.9)",
        bordercolor="#FF6B6B",
        borderwidth=1,
        font=dict(size=11, color="#2C3E50"),
        row=1, col=1
    )
    
    # 右图：性能-复杂度权衡分析
    # 计算参数数量（简化估算）
    param_counts = []
    for num_branches in branch_nums:
        input_per_branch = 40 // num_branches  # 总输入40维
        branch_params = num_branches * input_per_branch * 16  # 隐藏层16维
        output_params = 16 * 2  # 输出层
        tau_params = 16 * (1 + num_branches)  # 时间常数
        total_params = branch_params + output_params + tau_params
        param_counts.append(total_params)
    
    # 归一化参数数量
    param_ratios = [p / param_counts[0] for p in param_counts]
    
    # 参数数量柱状图
    fig.add_trace(go.Bar(
        x=branch_nums,
        y=param_ratios,
        name='Parameter Ratio',
        marker_color='#4BC0C0',
        opacity=0.7,
        text=[f'{ratio:.1f}x' for ratio in param_ratios],
        textposition='outside',
        hovertemplate='<b>%{x}B</b><br>Parameters: %{y:.1f}x baseline<extra></extra>'
    ), row=1, col=2)
    
    # 性能提升曲线（右轴）
    baseline_acc = mean_accs[0]
    improvements = [acc - baseline_acc for acc in mean_accs]
    
    fig.add_trace(go.Scatter(
        x=branch_nums,
        y=improvements,
        mode='lines+markers',
        name='Performance Gain',
        line=dict(color='#9966FF', width=4, dash='dash'),
        marker=dict(size=10, symbol='diamond'),
        yaxis='y2',
        hovertemplate='<b>%{x}B</b><br>Gain: +%{y:.1f}%<extra></extra>'
    ), row=1, col=2, secondary_y=True)
    
    # 添加效率最优点标注
    fig.add_annotation(
        x=2, y=1.8,
        text="<b>Efficiency Sweet Spot</b><br>Max gain with<br>minimal overhead",
        showarrow=True,
        arrowhead=2,
        arrowcolor="#9966FF",
        arrowwidth=2,
        bgcolor="rgba(255,255,255,0.9)",
        bordercolor="#9966FF",
        borderwidth=1,
        font=dict(size=10, color="#2C3E50"),
        row=1, col=2
    )
    
    # 更新布局
    fig.update_layout(
        title={
            'text': "Branch Number Analysis for Multi-timescale XOR Task",
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 18, 'color': '#2E86AB', 'family': 'Arial Black'}
        },
        height=500,
        width=1000,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.2,
            xanchor="center",
            x=0.5,
            font=dict(size=11)
        ),
        font=dict(size=12, family='Arial'),
        plot_bgcolor='rgba(248,249,250,0.8)',
        paper_bgcolor='white'
    )
    
    # 更新左图坐标轴
    fig.update_xaxes(
        title_text="Number of Branches",
        tickvals=branch_nums,
        ticktext=[f'{n}B' for n in branch_nums],
        row=1, col=1
    )
    fig.update_yaxes(
        title_text="Test Accuracy (%)",
        range=[50, 100],
        row=1, col=1
    )
    
    # 更新右图坐标轴
    fig.update_xaxes(
        title_text="Number of Branches",
        tickvals=branch_nums,
        ticktext=[f'{n}B' for n in branch_nums],
        row=1, col=2
    )
    fig.update_yaxes(
        title_text="Parameter Ratio (×baseline)",
        row=1, col=2
    )
    fig.update_yaxes(
        title_text="Performance Gain (%)",
        side="right",
        row=1, col=2,
        secondary_y=True
    )
    
    return fig

def create_branch_mechanism_diagram():
    """创建分支机制示意图"""
    
    fig = go.Figure()
    
    # 创建不同分支配置的示意图
    configurations = [
        {"name": "1B (Vanilla SNN)", "x": 1, "y": 3, "color": "#95A5A6", "performance": "59.1%"},
        {"name": "2B (Optimal)", "x": 2, "y": 3, "color": "#2E86AB", "performance": "97.3%"},
        {"name": "4B", "x": 3, "y": 3, "color": "#F39C12", "performance": "96.5%"},
        {"name": "8B", "x": 4, "y": 3, "color": "#E74C3C", "performance": "95.1%"}
    ]
    
    # 绘制配置节点
    for config in configurations:
        # 主节点
        fig.add_trace(go.Scatter(
            x=[config["x"]],
            y=[config["y"]],
            mode='markers+text',
            marker=dict(
                size=60,
                color=config["color"],
                line=dict(width=3, color='white'),
                symbol='circle'
            ),
            text=config["performance"],
            textposition="middle center",
            textfont=dict(size=12, color='white', family='Arial Black'),
            name=config["name"],
            hovertemplate=f'<b>{config["name"]}</b><br>Performance: {config["performance"]}<extra></extra>'
        ))
        
        # 分支示意
        num_branches = int(config["name"].split('B')[0])
        if num_branches > 1:
            for i in range(num_branches):
                angle = (i - (num_branches-1)/2) * 0.3
                branch_x = config["x"] + 0.3 * np.sin(angle)
                branch_y = config["y"] - 0.5 + 0.1 * np.cos(angle)
                
                fig.add_trace(go.Scatter(
                    x=[config["x"], branch_x],
                    y=[config["y"], branch_y],
                    mode='lines',
                    line=dict(width=3, color=config["color"]),
                    showlegend=False,
                    hoverinfo='skip'
                ))
    
    # 添加性能趋势线
    x_vals = [1, 2, 3, 4]
    y_vals = [59.1, 97.3, 96.5, 95.1]
    
    fig.add_trace(go.Scatter(
        x=x_vals,
        y=[2.5] * len(x_vals),
        mode='lines',
        line=dict(width=3, color='#34495E', dash='dot'),
        name='Performance Trend',
        showlegend=False,
        hoverinfo='skip'
    ))
    
    # 添加标注
    fig.add_annotation(
        x=2, y=2,
        text="<b>Optimal for Multi-timescale Tasks</b><br>• Best performance (97.3%)<br>• Lowest variance (±0.3%)<br>• Efficient parameter usage",
        showarrow=True,
        arrowhead=2,
        arrowcolor="#2E86AB",
        arrowwidth=2,
        bgcolor="rgba(255,255,255,0.95)",
        bordercolor="#2E86AB",
        borderwidth=2,
        font=dict(size=11, color="#2C3E50")
    )
    
    fig.add_annotation(
        x=3.5, y=1.5,
        text="<b>Diminishing Returns</b><br>More branches ≠ Better performance",
        showarrow=True,
        arrowhead=2,
        arrowcolor="#E74C3C",
        arrowwidth=2,
        bgcolor="rgba(255,255,255,0.95)",
        bordercolor="#E74C3C",
        borderwidth=2,
        font=dict(size=10, color="#2C3E50")
    )
    
    # 更新布局
    fig.update_layout(
        title={
            'text': "DH-SNN Branch Configuration Analysis",
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 16, 'color': '#2E86AB', 'family': 'Arial Black'}
        },
        height=400,
        width=800,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.15,
            xanchor="center",
            x=0.5
        ),
        xaxis=dict(
            title="Branch Configuration",
            range=[0.5, 4.5],
            tickvals=[1, 2, 3, 4],
            ticktext=["1B", "2B", "4B", "8B"]
        ),
        yaxis=dict(
            title="",
            range=[1, 4],
            showticklabels=False
        ),
        plot_bgcolor='rgba(248,249,250,0.8)',
        paper_bgcolor='white'
    )
    
    return fig

def main():
    """主函数"""
    
    print("🎨 创建重新设计的分支数量对比可视化...")
    
    # 创建主要分析图
    fig1 = create_branch_comparison_redesigned()
    
    # 保存图表
    html_path1 = "branch_comparison_redesigned.html"
    fig1.write_html(html_path1)
    print(f"✅ HTML图表已保存: {html_path1}")
    
    try:
        png_path1 = "branch_comparison_redesigned.png"
        fig1.write_image(png_path1, width=1000, height=500, scale=2)
        print(f"✅ PNG图表已保存: {png_path1}")
    except Exception as e:
        print(f"⚠️ PNG保存失败: {e}")
    
    # 创建机制示意图
    fig2 = create_branch_mechanism_diagram()
    
    html_path2 = "branch_mechanism_diagram.html"
    fig2.write_html(html_path2)
    print(f"✅ 机制图HTML已保存: {html_path2}")
    
    try:
        png_path2 = "branch_mechanism_diagram.png"
        fig2.write_image(png_path2, width=800, height=400, scale=2)
        print(f"✅ 机制图PNG已保存: {png_path2}")
    except Exception as e:
        print(f"⚠️ 机制图PNG保存失败: {e}")
    
    print("🎉 重新设计的可视化创建完成!")

if __name__ == "__main__":
    main()
