#!/usr/bin/env python3
"""
时间因子特化分析实验
深入分析DH-SNN中不同分支的时间常数如何特化以处理不同时间尺度的信息
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import plotly.graph_objects as go
import plotly.subplots as sp
from plotly.subplots import make_subplots
import os
import json
import time
from datetime import datetime

print("🔬 时间因子特化分析实验")
print("="*60)

# 实验参数 (基于原论文)
torch.manual_seed(42)
time_steps = 100
channel = 2
channel_rate = [0.2, 0.6]  # 原论文精确发放率
noise_rate = 0.01
channel_size = 20
coding_time = 10
remain_time = 5
start_time = 10
batch_size = 500
hidden_dims = 16
output_dim = 2
learning_rate = 1e-2

# XOR标签
label = torch.zeros(len(channel_rate), len(channel_rate))
label[1][0] = 1
label[0][1] = 1

def get_batch(device='cpu'):
    """原论文的数据生成函数"""
    # 构建第一个序列
    values = torch.rand(batch_size, time_steps, channel_size*2, requires_grad=False) <= noise_rate
    targets = torch.zeros(time_steps, batch_size, requires_grad=False).int()

    # 构建Signal 1
    init_pattern = torch.randint(len(channel_rate), size=(batch_size,))
    # 生成脉冲
    prob_matrix = torch.ones(start_time, channel_size, batch_size) * torch.tensor(channel_rate)[init_pattern]
    add_patterns = torch.bernoulli(prob_matrix).permute(2, 0, 1).bool()
    values[:, :start_time, :channel_size] = values[:, :start_time, :channel_size] | add_patterns

    # 构建Signal 2
    for i in range((time_steps - start_time) // (coding_time + remain_time)):
        pattern = torch.randint(len(channel_rate), size=(batch_size,))
        label_t = label[init_pattern, pattern].int()
        # 生成脉冲
        prob = torch.tensor(channel_rate)[pattern]
        prob_matrix = torch.ones(coding_time, channel_size, batch_size) * prob
        add_patterns = torch.bernoulli(prob_matrix).permute(2, 0, 1).bool()

        start_idx = start_time + i * (coding_time + remain_time) + remain_time
        end_idx = start_time + (i + 1) * (coding_time + remain_time)
        values[:, start_idx:end_idx, channel_size:] = values[:, start_idx:end_idx, channel_size:] | add_patterns
        targets[start_time + i * (coding_time + remain_time):start_time + (i + 1) * (coding_time + remain_time)] = label_t

    return values.float().to(device), targets.transpose(0, 1).contiguous().to(device)

class TemporalFactorDH_LIF(nn.Module):
    """时间因子特化分析的DH-LIF神经元"""

    def __init__(self, input_size, hidden_size, learnable=True, device='cpu'):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.device = device

        # 分支线性层
        self.dense1 = nn.Linear(input_size // 2, hidden_size, bias=False)
        self.dense2 = nn.Linear(input_size // 2, hidden_size, bias=False)

        # 膜电位时间常数
        self.tau_m = nn.Parameter(torch.ones(hidden_size) * 2.0)

        # 分支时间常数 - 按照原论文初始化
        # Branch 1: Large时间常数 (长期记忆)
        self.tau_n1 = nn.Parameter(torch.empty(hidden_size).uniform_(2, 6))
        # Branch 2: Small时间常数 (快速响应)
        self.tau_n2 = nn.Parameter(torch.empty(hidden_size).uniform_(-4, 0))

        # 设置可学习性
        self.tau_m.requires_grad = learnable
        self.tau_n1.requires_grad = learnable
        self.tau_n2.requires_grad = learnable

        # 神经元状态
        self.mem = None
        self.d_input1 = None
        self.d_input2 = None

    def set_neuron_state(self, batch_size):
        """重置神经元状态"""
        self.mem = torch.zeros(batch_size, self.hidden_size).to(self.device)
        self.d_input1 = torch.zeros(batch_size, self.hidden_size).to(self.device)
        self.d_input2 = torch.zeros(batch_size, self.hidden_size).to(self.device)

    def forward(self, x):
        """前向传播"""
        # 分割输入到两个分支
        x1 = x[:, :self.input_size // 2]
        x2 = x[:, self.input_size // 2:]

        # 通过分支线性层
        d_input1 = self.dense1(x1)
        d_input2 = self.dense2(x2)

        # 更新分支电流
        beta1 = torch.sigmoid(self.tau_n1)
        beta2 = torch.sigmoid(self.tau_n2)
        
        self.d_input1 = beta1 * self.d_input1 + (1 - beta1) * d_input1
        self.d_input2 = beta2 * self.d_input2 + (1 - beta2) * d_input2

        # 合并分支输入
        total_d_current = self.d_input1 + self.d_input2

        # 更新膜电位
        alpha = torch.sigmoid(self.tau_m)
        self.mem = alpha * self.mem + (1 - alpha) * total_d_current

        # 脉冲生成 (简化版，使用sigmoid)
        spike = torch.sigmoid(self.mem)

        return self.mem, spike

class TemporalFactorDH_SFNN(nn.Module):
    """时间因子特化分析的DH-SFNN"""

    def __init__(self, input_size, hidden_dims, output_dim, learnable=True, device='cpu'):
        super().__init__()
        self.input_size = input_size
        self.device = device

        self.dense_1 = TemporalFactorDH_LIF(input_size, hidden_dims, learnable, device)
        self.dense_2 = nn.Linear(hidden_dims, output_dim)
        self.criterion = nn.CrossEntropyLoss()

    def init(self):
        self.dense_1.set_neuron_state(batch_size)

    def forward(self, input_data, target):
        batch_size, seq_num, input_size = input_data.shape

        d2_output = torch.zeros(batch_size, seq_num, 2)
        loss = 0
        total = 0
        correct = 0

        for i in range(seq_num):
            input_x = input_data[:, i, :]
            mem_layer1, spike_layer1 = self.dense_1.forward(input_x)
            l2_output = self.dense_2(spike_layer1)
            d2_output[:, i, :] = l2_output.cpu()

            if (((i - start_time) % (coding_time + remain_time)) > remain_time) and (i > start_time):
                output = F.softmax(l2_output, dim=1)
                loss += self.criterion(output, target[:, i].long())
                _, predicted = torch.max(output.data, 1)
                labels = target[:, i].cpu()
                predicted = predicted.cpu()
                correct += (predicted == labels).sum()
                total += labels.size()[0]

        return loss, d2_output, correct, total

def train_and_track_temporal_factors(model, epochs=50, device='cpu'):
    """训练模型并跟踪时间因子变化"""
    
    print(f"🏋️ 训练并跟踪时间因子特化过程...")
    
    model.to(device)
    
    # 记录初始时间常数
    initial_tau_m = torch.sigmoid(model.dense_1.tau_m).detach().cpu().numpy()
    initial_tau_n1 = torch.sigmoid(model.dense_1.tau_n1).detach().cpu().numpy()
    initial_tau_n2 = torch.sigmoid(model.dense_1.tau_n2).detach().cpu().numpy()
    
    # 优化器设置
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
    
    # 训练历史记录
    history = {
        'tau_m': [],
        'tau_n1': [],
        'tau_n2': [],
        'accuracies': [],
        'losses': [],
        'specialization_index': [],  # 特化指数
        'branch_correlation': []     # 分支相关性
    }
    
    best_acc = 0
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
        
        acc = sum_correct / sum_sample * 100 if sum_sample > 0 else 0
        if acc > best_acc:
            best_acc = acc
        
        # 记录时间常数
        with torch.no_grad():
            tau_m = torch.sigmoid(model.dense_1.tau_m).cpu().numpy()
            tau_n1 = torch.sigmoid(model.dense_1.tau_n1).cpu().numpy()
            tau_n2 = torch.sigmoid(model.dense_1.tau_n2).cpu().numpy()
            
            history['tau_m'].append(tau_m.copy())
            history['tau_n1'].append(tau_n1.copy())
            history['tau_n2'].append(tau_n2.copy())
            history['accuracies'].append(acc)
            history['losses'].append(train_loss_sum / log_interval)
            
            # 计算特化指数 (分支间差异)
            specialization = np.abs(tau_n1.mean() - tau_n2.mean())
            history['specialization_index'].append(specialization)
            
            # 计算分支相关性
            correlation = np.corrcoef(tau_n1, tau_n2)[0, 1]
            history['branch_correlation'].append(correlation)
        
        if epoch % 5 == 0:
            print(f'  Epoch {epoch+1:3d}: Loss={train_loss_sum/log_interval:.4f}, '
                  f'Acc={acc:.1f}%, Best={best_acc:.1f}%, '
                  f'Spec={specialization:.3f}')
    
    # 记录最终状态
    final_tau_m = torch.sigmoid(model.dense_1.tau_m).detach().cpu().numpy()
    final_tau_n1 = torch.sigmoid(model.dense_1.tau_n1).detach().cpu().numpy()
    final_tau_n2 = torch.sigmoid(model.dense_1.tau_n2).detach().cpu().numpy()
    
    results = {
        'initial': {
            'tau_m': initial_tau_m,
            'tau_n1': initial_tau_n1,
            'tau_n2': initial_tau_n2
        },
        'final': {
            'tau_m': final_tau_m,
            'tau_n1': final_tau_n1,
            'tau_n2': final_tau_n2
        },
        'history': history,
        'best_accuracy': best_acc,
        'final_accuracy': acc
    }
    
    return results

def create_temporal_factor_visualization(results):
    """创建时间因子特化分析的综合可视化"""

    print("🎨 创建时间因子特化分析可视化...")

    # 创建2x3子图布局
    fig = make_subplots(
        rows=2, cols=3,
        subplot_titles=[
            'Temporal Factor Evolution',
            'Specialization Development',
            'Branch Differentiation',
            'Distribution Analysis',
            'Correlation Analysis',
            'Performance vs Specialization'
        ],
        specs=[[{"secondary_y": False}, {"secondary_y": True}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}, {"secondary_y": True}]]
    )

    epochs = list(range(1, len(results['history']['tau_n1']) + 1))

    # Panel A: 时间因子演化
    tau_n1_mean = [np.mean(tau) for tau in results['history']['tau_n1']]
    tau_n2_mean = [np.mean(tau) for tau in results['history']['tau_n2']]
    tau_m_mean = [np.mean(tau) for tau in results['history']['tau_m']]

    fig.add_trace(go.Scatter(
        x=epochs, y=tau_n1_mean,
        mode='lines+markers',
        name='Branch 1 (Long-term)',
        line=dict(color='#2E86AB', width=3),
        marker=dict(size=6)
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=epochs, y=tau_n2_mean,
        mode='lines+markers',
        name='Branch 2 (Short-term)',
        line=dict(color='#FF6B6B', width=3),
        marker=dict(size=6)
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=epochs, y=tau_m_mean,
        mode='lines+markers',
        name='Membrane (Integration)',
        line=dict(color='#4BC0C0', width=3, dash='dash'),
        marker=dict(size=6)
    ), row=1, col=1)

    # Panel B: 特化发展过程
    specialization = results['history']['specialization_index']
    accuracies = results['history']['accuracies']

    fig.add_trace(go.Scatter(
        x=epochs, y=specialization,
        mode='lines+markers',
        name='Specialization Index',
        line=dict(color='#9966FF', width=3),
        marker=dict(size=6)
    ), row=1, col=2)

    fig.add_trace(go.Scatter(
        x=epochs, y=accuracies,
        mode='lines+markers',
        name='Accuracy',
        line=dict(color='#FF9F40', width=3),
        marker=dict(size=6),
        yaxis='y2'
    ), row=1, col=2, secondary_y=True)

    # Panel C: 分支分化分析
    tau_diff = [abs(t1 - t2) for t1, t2 in zip(tau_n1_mean, tau_n2_mean)]

    fig.add_trace(go.Scatter(
        x=epochs, y=tau_diff,
        mode='lines+markers',
        name='Branch Difference',
        line=dict(color='#E74C3C', width=3),
        marker=dict(size=6),
        fill='tonexty'
    ), row=1, col=3)

    # Panel D: 分布分析 (初始 vs 最终)
    fig.add_trace(go.Histogram(
        x=results['initial']['tau_n1'],
        name='Branch 1 Initial',
        marker_color='rgba(46, 134, 171, 0.6)',
        nbinsx=15,
        showlegend=False
    ), row=2, col=1)

    fig.add_trace(go.Histogram(
        x=results['final']['tau_n1'],
        name='Branch 1 Final',
        marker_color='rgba(46, 134, 171, 1.0)',
        nbinsx=15,
        showlegend=False
    ), row=2, col=1)

    # Panel E: 相关性分析
    correlations = results['history']['branch_correlation']

    fig.add_trace(go.Scatter(
        x=epochs, y=correlations,
        mode='lines+markers',
        name='Branch Correlation',
        line=dict(color='#8E44AD', width=3),
        marker=dict(size=6)
    ), row=2, col=2)

    # Panel F: 性能与特化关系
    fig.add_trace(go.Scatter(
        x=specialization, y=accuracies,
        mode='markers',
        name='Performance vs Specialization',
        marker=dict(
            size=8,
            color=epochs,
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(title="Epoch")
        )
    ), row=2, col=3)

    # 添加趋势线
    z = np.polyfit(specialization, accuracies, 1)
    p = np.poly1d(z)
    fig.add_trace(go.Scatter(
        x=specialization,
        y=p(specialization),
        mode='lines',
        name='Trend',
        line=dict(color='red', width=2, dash='dash'),
        showlegend=False
    ), row=2, col=3)

    # 更新布局
    fig.update_layout(
        title={
            'text': "Temporal Factor Specialization Analysis in DH-SNN",
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 18, 'color': '#2E86AB', 'family': 'Arial Black'}
        },
        height=800,
        width=1400,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.15,
            xanchor="center",
            x=0.5
        ),
        font=dict(size=11, family='Arial')
    )

    # 更新坐标轴
    fig.update_xaxes(title_text="Training Epoch", row=1, col=1)
    fig.update_yaxes(title_text="Time Constant (τ)", row=1, col=1)

    fig.update_xaxes(title_text="Training Epoch", row=1, col=2)
    fig.update_yaxes(title_text="Specialization Index", row=1, col=2)
    fig.update_yaxes(title_text="Accuracy (%)", row=1, col=2, secondary_y=True)

    fig.update_xaxes(title_text="Training Epoch", row=1, col=3)
    fig.update_yaxes(title_text="Branch Difference |τ₁-τ₂|", row=1, col=3)

    fig.update_xaxes(title_text="Time Constant Value", row=2, col=1)
    fig.update_yaxes(title_text="Frequency", row=2, col=1)

    fig.update_xaxes(title_text="Training Epoch", row=2, col=2)
    fig.update_yaxes(title_text="Correlation Coefficient", row=2, col=2)

    fig.update_xaxes(title_text="Specialization Index", row=2, col=3)
    fig.update_yaxes(title_text="Accuracy (%)", row=2, col=3)

    return fig

def analyze_temporal_specialization(results):
    """分析时间特化机制"""

    print(f"\n📊 时间因子特化分析:")
    print("="*50)

    # 初始状态分析
    initial_tau1 = results['initial']['tau_n1'].mean()
    initial_tau2 = results['initial']['tau_n2'].mean()
    initial_diff = abs(initial_tau1 - initial_tau2)

    # 最终状态分析
    final_tau1 = results['final']['tau_n1'].mean()
    final_tau2 = results['final']['tau_n2'].mean()
    final_diff = abs(final_tau1 - final_tau2)

    # 变化分析
    tau1_change = final_tau1 - initial_tau1
    tau2_change = final_tau2 - initial_tau2

    print(f"初始状态:")
    print(f"  Branch 1 (Long-term): {initial_tau1:.3f}")
    print(f"  Branch 2 (Short-term): {initial_tau2:.3f}")
    print(f"  初始差异: {initial_diff:.3f}")

    print(f"\n最终状态:")
    print(f"  Branch 1 (Long-term): {final_tau1:.3f}")
    print(f"  Branch 2 (Short-term): {final_tau2:.3f}")
    print(f"  最终差异: {final_diff:.3f}")

    print(f"\n特化过程:")
    print(f"  Branch 1 变化: {tau1_change:+.3f}")
    print(f"  Branch 2 变化: {tau2_change:+.3f}")
    print(f"  分化增强: {final_diff - initial_diff:+.3f}")

    # 特化效果评估
    specialization_improvement = (final_diff - initial_diff) / initial_diff * 100
    print(f"  特化改善: {specialization_improvement:+.1f}%")

    # 性能关联分析
    final_acc = results['final_accuracy']
    best_acc = results['best_accuracy']

    print(f"\n性能分析:")
    print(f"  最终准确率: {final_acc:.1f}%")
    print(f"  最佳准确率: {best_acc:.1f}%")

    # 特化质量评估
    if final_tau1 > final_tau2:
        print(f"  ✅ 特化方向正确: Branch 1 > Branch 2")
        specialization_quality = "优秀"
    else:
        print(f"  ⚠️ 特化方向异常: Branch 1 < Branch 2")
        specialization_quality = "需要改进"

    print(f"  特化质量: {specialization_quality}")

    return {
        'initial_diff': initial_diff,
        'final_diff': final_diff,
        'specialization_improvement': specialization_improvement,
        'specialization_quality': specialization_quality,
        'final_accuracy': final_acc
    }

def main():
    """主函数"""

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🔧 使用设备: {device}")

    # 创建结果目录
    os.makedirs("experiments/temporal_factor_specialization/results", exist_ok=True)

    # 创建模型
    print(f"\n🧠 创建时间因子特化分析模型...")
    model = TemporalFactorDH_SFNN(
        input_size=channel_size*2,
        hidden_dims=hidden_dims,
        output_dim=output_dim,
        learnable=True,
        device=device
    )

    # 训练并跟踪时间因子
    print(f"\n🏋️ 开始训练和时间因子跟踪...")
    start_time = time.time()

    results = train_and_track_temporal_factors(model, epochs=50, device=device)

    training_time = time.time() - start_time
    print(f"\n⏱️ 训练完成，用时: {training_time/60:.1f}分钟")

    # 分析时间特化
    specialization_analysis = analyze_temporal_specialization(results)

    # 创建可视化
    print(f"\n🎨 创建可视化...")
    fig = create_temporal_factor_visualization(results)

    # 保存结果
    results_dir = "experiments/temporal_factor_specialization/results"

    # 保存图表
    html_path = f"{results_dir}/temporal_factor_specialization.html"
    fig.write_html(html_path)
    print(f"✅ HTML图表已保存: {html_path}")

    try:
        png_path = f"{results_dir}/temporal_factor_specialization.png"
        fig.write_image(png_path, width=1400, height=800, scale=2)
        print(f"✅ PNG图表已保存: {png_path}")
    except Exception as e:
        print(f"⚠️ PNG保存失败: {e}")

    # 保存数据
    results_with_analysis = {
        **results,
        'specialization_analysis': specialization_analysis,
        'training_time': training_time,
        'timestamp': datetime.now().isoformat()
    }

    # 保存为JSON
    json_path = f"{results_dir}/temporal_factor_results.json"
    with open(json_path, 'w') as f:
        # 转换numpy数组为列表以便JSON序列化
        json_results = {}
        for key, value in results_with_analysis.items():
            if key in ['initial', 'final']:
                json_results[key] = {k: v.tolist() for k, v in value.items()}
            elif key == 'history':
                json_results[key] = {k: [arr.tolist() if isinstance(arr, np.ndarray) else arr
                                       for arr in v] for k, v in value.items()}
            else:
                json_results[key] = value

        json.dump(json_results, f, indent=2)
    print(f"✅ JSON结果已保存: {json_path}")

    # 保存为PyTorch格式
    torch_path = f"{results_dir}/temporal_factor_results.pth"
    torch.save(results_with_analysis, torch_path)
    print(f"✅ PyTorch结果已保存: {torch_path}")

    # 打印最终总结
    print(f"\n🎉 时间因子特化分析实验完成!")
    print("="*60)
    print(f"📊 实验总结:")
    print(f"  最终准确率: {results['final_accuracy']:.1f}%")
    print(f"  最佳准确率: {results['best_accuracy']:.1f}%")
    print(f"  特化改善: {specialization_analysis['specialization_improvement']:+.1f}%")
    print(f"  特化质量: {specialization_analysis['specialization_quality']}")
    print(f"  训练时间: {training_time/60:.1f}分钟")
    print("="*60)

    return results_with_analysis

if __name__ == "__main__":
    main()
