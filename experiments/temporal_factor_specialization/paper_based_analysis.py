#!/usr/bin/env python3
"""
基于原论文的时间因子特化分析实验
复现Figure 4c: 时间常数分布变化分析
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

print("📊 基于原论文的时间因子特化分析")
print("="*60)

# 原论文参数设置
torch.manual_seed(42)
time_steps = 100
channel = 2
channel_rate = [0.2, 0.6]
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
    values = torch.rand(batch_size, time_steps, channel_size*2, requires_grad=False) <= noise_rate
    targets = torch.zeros(time_steps, batch_size, requires_grad=False).int()

    # 构建Signal 1
    init_pattern = torch.randint(len(channel_rate), size=(batch_size,))
    prob_matrix = torch.ones(start_time, channel_size, batch_size) * torch.tensor(channel_rate)[init_pattern]
    add_patterns = torch.bernoulli(prob_matrix).permute(2, 0, 1).bool()
    values[:, :start_time, :channel_size] = values[:, :start_time, :channel_size] | add_patterns

    # 构建Signal 2
    for i in range((time_steps - start_time) // (coding_time + remain_time)):
        pattern = torch.randint(len(channel_rate), size=(batch_size,))
        label_t = label[init_pattern, pattern].int()
        prob = torch.tensor(channel_rate)[pattern]
        prob_matrix = torch.ones(coding_time, channel_size, batch_size) * prob
        add_patterns = torch.bernoulli(prob_matrix).permute(2, 0, 1).bool()

        start_idx = start_time + i * (coding_time + remain_time) + remain_time
        end_idx = start_time + (i + 1) * (coding_time + remain_time)
        values[:, start_idx:end_idx, channel_size:] = values[:, start_idx:end_idx, channel_size:] | add_patterns
        targets[start_time + i * (coding_time + remain_time):start_time + (i + 1) * (coding_time + remain_time)] = label_t

    return values.float().to(device), targets.transpose(0, 1).contiguous().to(device)

class Paper2BranchDH_LIF(nn.Module):
    """原论文的2分支DH-LIF神经元"""

    def __init__(self, input_size, hidden_size, learnable=True, device='cpu'):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.device = device

        # 分支线性层
        self.dense1 = nn.Linear(input_size // 2, hidden_size, bias=False)
        self.dense2 = nn.Linear(input_size // 2, hidden_size, bias=False)

        # 膜电位时间常数 (Medium初始化: U(0,4))
        self.tau_m = nn.Parameter(torch.empty(hidden_size).uniform_(0, 4))

        # 分支时间常数 - 按照原论文Table S3
        # Branch 1: Large时间常数 U(2,6) - 长期记忆
        self.tau_n1 = nn.Parameter(torch.empty(hidden_size).uniform_(2, 6))
        # Branch 2: Small时间常数 U(-4,0) - 快速响应
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

        # 脉冲生成
        spike = torch.sigmoid(self.mem)

        return self.mem, spike

class Paper2BranchDH_SFNN(nn.Module):
    """原论文的2分支DH-SFNN"""

    def __init__(self, input_size, hidden_dims, output_dim, learnable=True, device='cpu'):
        super().__init__()
        self.input_size = input_size
        self.device = device

        self.dense_1 = Paper2BranchDH_LIF(input_size, hidden_dims, learnable, device)
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

def train_and_track_time_constants(model, epochs=50, device='cpu'):
    """训练模型并跟踪时间常数变化 - 按照原论文方法"""
    
    print(f"🏋️ 训练并跟踪时间常数变化...")
    
    model.to(device)
    
    # 记录初始时间常数
    initial_tau_m = torch.sigmoid(model.dense_1.tau_m).detach().cpu().numpy()
    initial_tau_n1 = torch.sigmoid(model.dense_1.tau_n1).detach().cpu().numpy()
    initial_tau_n2 = torch.sigmoid(model.dense_1.tau_n2).detach().cpu().numpy()
    
    # 优化器设置 (按照原论文)
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
        'losses': []
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
        
        if epoch % 5 == 0:
            print(f'  Epoch {epoch+1:3d}: Loss={train_loss_sum/log_interval:.4f}, '
                  f'Acc={acc:.1f}%, Best={best_acc:.1f}%')
    
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

def create_figure4c_reproduction(results):
    """复现Figure 4c: 时间常数分布变化"""

    print("🎨 创建Figure 4c复现图...")

    # 创建2x2子图布局 (按照原论文Figure 4c)
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[
            'Branch 1 (Long-term Memory) - Before Training',
            'Branch 1 (Long-term Memory) - After Training',
            'Branch 2 (Short-term Response) - Before Training',
            'Branch 2 (Short-term Response) - After Training'
        ],
        vertical_spacing=0.12,
        horizontal_spacing=0.1
    )

    # 提取数据
    initial_tau_n1 = results['initial']['tau_n1']
    final_tau_n1 = results['final']['tau_n1']
    initial_tau_n2 = results['initial']['tau_n2']
    final_tau_n2 = results['final']['tau_n2']

    # Branch 1 - 训练前
    fig.add_trace(go.Histogram(
        x=initial_tau_n1,
        nbinsx=20,
        name='Branch 1 Initial',
        marker_color='rgba(46, 134, 171, 0.7)',
        showlegend=False,
        histnorm='probability density'
    ), row=1, col=1)

    # Branch 1 - 训练后
    fig.add_trace(go.Histogram(
        x=final_tau_n1,
        nbinsx=20,
        name='Branch 1 Final',
        marker_color='rgba(46, 134, 171, 1.0)',
        showlegend=False,
        histnorm='probability density'
    ), row=1, col=2)

    # Branch 2 - 训练前
    fig.add_trace(go.Histogram(
        x=initial_tau_n2,
        nbinsx=20,
        name='Branch 2 Initial',
        marker_color='rgba(255, 107, 107, 0.7)',
        showlegend=False,
        histnorm='probability density'
    ), row=2, col=1)

    # Branch 2 - 训练后
    fig.add_trace(go.Histogram(
        x=final_tau_n2,
        nbinsx=20,
        name='Branch 2 Final',
        marker_color='rgba(255, 107, 107, 1.0)',
        showlegend=False,
        histnorm='probability density'
    ), row=2, col=2)

    # 添加统计信息标注
    # Branch 1 统计
    fig.add_annotation(
        x=0.95, y=0.95,
        text=f"μ = {initial_tau_n1.mean():.3f}<br>σ = {initial_tau_n1.std():.3f}",
        xref="x domain", yref="y domain",
        showarrow=False,
        bgcolor="rgba(255,255,255,0.8)",
        bordercolor="rgba(46, 134, 171, 0.8)",
        borderwidth=1,
        font=dict(size=10),
        row=1, col=1
    )

    fig.add_annotation(
        x=0.95, y=0.95,
        text=f"μ = {final_tau_n1.mean():.3f}<br>σ = {final_tau_n1.std():.3f}",
        xref="x domain", yref="y domain",
        showarrow=False,
        bgcolor="rgba(255,255,255,0.8)",
        bordercolor="rgba(46, 134, 171, 0.8)",
        borderwidth=1,
        font=dict(size=10),
        row=1, col=2
    )

    # Branch 2 统计
    fig.add_annotation(
        x=0.95, y=0.95,
        text=f"μ = {initial_tau_n2.mean():.3f}<br>σ = {initial_tau_n2.std():.3f}",
        xref="x domain", yref="y domain",
        showarrow=False,
        bgcolor="rgba(255,255,255,0.8)",
        bordercolor="rgba(255, 107, 107, 0.8)",
        borderwidth=1,
        font=dict(size=10),
        row=2, col=1
    )

    fig.add_annotation(
        x=0.95, y=0.95,
        text=f"μ = {final_tau_n2.mean():.3f}<br>σ = {final_tau_n2.std():.3f}",
        xref="x domain", yref="y domain",
        showarrow=False,
        bgcolor="rgba(255,255,255,0.8)",
        bordercolor="rgba(255, 107, 107, 0.8)",
        borderwidth=1,
        font=dict(size=10),
        row=2, col=2
    )

    # 更新布局
    fig.update_layout(
        title={
            'text': "Figure 4c Reproduction: Dendritic Time Constant Distribution Changes",
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 16, 'color': '#2E86AB', 'family': 'Arial Black'}
        },
        height=600,
        width=1000,
        showlegend=False,
        font=dict(size=11, family='Arial')
    )

    # 更新坐标轴
    fig.update_xaxes(title_text="Time Constant (τ)", row=1, col=1)
    fig.update_yaxes(title_text="Density", row=1, col=1)
    fig.update_xaxes(title_text="Time Constant (τ)", row=1, col=2)
    fig.update_yaxes(title_text="Density", row=1, col=2)
    fig.update_xaxes(title_text="Time Constant (τ)", row=2, col=1)
    fig.update_yaxes(title_text="Density", row=2, col=1)
    fig.update_xaxes(title_text="Time Constant (τ)", row=2, col=2)
    fig.update_yaxes(title_text="Density", row=2, col=2)

    return fig

def analyze_specialization_statistics(results):
    """分析时间因子特化的统计信息"""

    print(f"\n📊 时间因子特化统计分析:")
    print("="*50)

    # 初始状态
    initial_tau1_mean = results['initial']['tau_n1'].mean()
    initial_tau1_std = results['initial']['tau_n1'].std()
    initial_tau2_mean = results['initial']['tau_n2'].mean()
    initial_tau2_std = results['initial']['tau_n2'].std()

    # 最终状态
    final_tau1_mean = results['final']['tau_n1'].mean()
    final_tau1_std = results['final']['tau_n1'].std()
    final_tau2_mean = results['final']['tau_n2'].mean()
    final_tau2_std = results['final']['tau_n2'].std()

    print(f"初始状态 (按照论文Table S3初始化):")
    print(f"  Branch 1 (Large Init): μ={initial_tau1_mean:.3f}, σ={initial_tau1_std:.3f}")
    print(f"  Branch 2 (Small Init): μ={initial_tau2_mean:.3f}, σ={initial_tau2_std:.3f}")
    print(f"  初始分化程度: {abs(initial_tau1_mean - initial_tau2_mean):.3f}")

    print(f"\n最终状态 (训练后):")
    print(f"  Branch 1 (Long-term): μ={final_tau1_mean:.3f}, σ={final_tau1_std:.3f}")
    print(f"  Branch 2 (Short-term): μ={final_tau2_mean:.3f}, σ={final_tau2_std:.3f}")
    print(f"  最终分化程度: {abs(final_tau1_mean - final_tau2_mean):.3f}")

    # 变化分析
    tau1_change = final_tau1_mean - initial_tau1_mean
    tau2_change = final_tau2_mean - initial_tau2_mean

    print(f"\n特化过程:")
    print(f"  Branch 1 变化: {tau1_change:+.3f}")
    print(f"  Branch 2 变化: {tau2_change:+.3f}")

    # 特化质量评估
    if final_tau1_mean > final_tau2_mean:
        specialization_quality = "✅ 正确特化"
        print(f"  特化方向: {specialization_quality} (Branch 1 > Branch 2)")
    else:
        specialization_quality = "⚠️ 特化异常"
        print(f"  特化方向: {specialization_quality} (Branch 1 < Branch 2)")

    # 性能关联
    print(f"\n性能指标:")
    print(f"  最终准确率: {results['final_accuracy']:.1f}%")
    print(f"  最佳准确率: {results['best_accuracy']:.1f}%")

    return {
        'initial_differentiation': abs(initial_tau1_mean - initial_tau2_mean),
        'final_differentiation': abs(final_tau1_mean - final_tau2_mean),
        'specialization_quality': specialization_quality,
        'tau1_change': tau1_change,
        'tau2_change': tau2_change,
        'final_accuracy': results['final_accuracy']
    }

def main():
    """主函数"""

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🔧 使用设备: {device}")

    # 创建结果目录
    os.makedirs("experiments/temporal_factor_specialization/results", exist_ok=True)

    # 创建模型 (按照原论文设置)
    print(f"\n🧠 创建2-Branch DH-SFNN模型 (原论文配置)...")
    model = Paper2BranchDH_SFNN(
        input_size=channel_size*2,
        hidden_dims=hidden_dims,
        output_dim=output_dim,
        learnable=True,
        device=device
    )

    # 训练并跟踪时间常数
    print(f"\n🏋️ 开始训练和时间常数跟踪...")
    start_time = time.time()

    results = train_and_track_time_constants(model, epochs=50, device=device)

    training_time = time.time() - start_time
    print(f"\n⏱️ 训练完成，用时: {training_time/60:.1f}分钟")

    # 分析特化统计
    specialization_stats = analyze_specialization_statistics(results)

    # 创建Figure 4c复现图
    print(f"\n🎨 创建Figure 4c复现图...")
    fig = create_figure4c_reproduction(results)

    # 保存结果
    results_dir = "experiments/temporal_factor_specialization/results"

    # 保存图表
    html_path = f"{results_dir}/figure4c_reproduction.html"
    fig.write_html(html_path)
    print(f"✅ HTML图表已保存: {html_path}")

    try:
        png_path = f"{results_dir}/figure4c_reproduction.png"
        fig.write_image(png_path, width=1000, height=600, scale=2)
        print(f"✅ PNG图表已保存: {png_path}")
    except Exception as e:
        print(f"⚠️ PNG保存失败: {e}")

    # 保存数据
    final_results = {
        **results,
        'specialization_stats': specialization_stats,
        'training_time': training_time,
        'timestamp': datetime.now().isoformat()
    }

    # 保存为JSON
    json_path = f"{results_dir}/figure4c_results.json"
    with open(json_path, 'w') as f:
        json_results = {}
        for key, value in final_results.items():
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
    torch_path = f"{results_dir}/figure4c_results.pth"
    torch.save(final_results, torch_path)
    print(f"✅ PyTorch结果已保存: {torch_path}")

    # 打印最终总结
    print(f"\n🎉 Figure 4c复现实验完成!")
    print("="*60)
    print(f"📊 实验总结:")
    print(f"  最终准确率: {results['final_accuracy']:.1f}%")
    print(f"  最佳准确率: {results['best_accuracy']:.1f}%")
    print(f"  特化质量: {specialization_stats['specialization_quality']}")
    print(f"  训练时间: {training_time/60:.1f}分钟")
    print("="*60)

    return final_results

if __name__ == "__main__":
    main()
