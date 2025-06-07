#!/usr/bin/env python3
"""
分支数量对比实验
基于现有的multi_timescale_xor实验代码，系统性地比较不同分支数量对DH-SNN性能的影响
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
import os
import json
import plotly.graph_objects as go
import plotly.subplots as sp
from plotly.subplots import make_subplots

print("🎯 分支数量对比实验")
print("="*60)

# 基于原论文的精确参数
torch.manual_seed(42)
time_steps = 100
channel = 2
channel_rate = [0.2, 0.6]  # 原论文精确发放率
noise_rate = 0.01
channel_size = 20
coding_time = 10
remain_time = 5
start_time = 10
batch_size = 500  # 原论文批次大小
hidden_dims = 16  # 原论文隐藏层大小
output_dim = 2
learning_rate = 1e-2  # 原论文学习率

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

class MultiBranchDH_LIF(nn.Module):
    """多分支DH-LIF神经元 - 支持任意分支数量"""

    def __init__(self, input_size, hidden_size, num_branches=2, learnable=True, device='cpu'):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_branches = num_branches
        self.device = device

        # 为每个分支创建线性层
        self.branch_layers = nn.ModuleList()
        input_per_branch = input_size // num_branches
        
        for i in range(num_branches):
            self.branch_layers.append(nn.Linear(input_per_branch, hidden_size, bias=False))

        # 膜电位时间常数
        self.tau_m = nn.Parameter(torch.ones(hidden_size) * 2.0)

        # 为每个分支创建时间常数
        self.branch_taus = nn.ParameterList()
        for i in range(num_branches):
            if i == 0:
                # 第一个分支使用Large时间常数 (适合长期记忆)
                tau = torch.empty(hidden_size).uniform_(2, 6)
            else:
                # 其他分支使用Small时间常数 (适合快速响应)
                tau = torch.empty(hidden_size).uniform_(-4, 0)
            self.branch_taus.append(nn.Parameter(tau))

        # 设置可学习性
        self.tau_m.requires_grad = learnable
        for tau in self.branch_taus:
            tau.requires_grad = learnable

        # 神经元状态
        self.mem = None
        self.branch_currents = None

    def set_neuron_state(self, batch_size):
        """重置神经元状态"""
        self.mem = torch.zeros(batch_size, self.hidden_size).to(self.device)
        self.branch_currents = [torch.zeros(batch_size, self.hidden_size).to(self.device) 
                               for _ in range(self.num_branches)]

    def forward(self, x):
        """前向传播"""
        input_per_branch = self.input_size // self.num_branches
        
        # 处理每个分支
        branch_inputs = []
        for i in range(self.num_branches):
            start_idx = i * input_per_branch
            end_idx = (i + 1) * input_per_branch
            branch_input = x[:, start_idx:end_idx]
            
            # 通过分支的线性层
            d_input = self.branch_layers[i](branch_input)
            
            # 更新分支电流
            beta = torch.sigmoid(self.branch_taus[i])
            self.branch_currents[i] = beta * self.branch_currents[i] + (1 - beta) * d_input
            
            branch_inputs.append(self.branch_currents[i])

        # 合并所有分支的输入
        total_d_current = sum(branch_inputs)

        # 更新膜电位
        alpha = torch.sigmoid(self.tau_m)
        self.mem = alpha * self.mem + (1 - alpha) * total_d_current

        # 脉冲生成 (简化版，使用sigmoid)
        spike = torch.sigmoid(self.mem)

        return self.mem, spike

class BranchNumberDH_SFNN(nn.Module):
    """支持任意分支数量的DH-SFNN"""

    def __init__(self, input_size, hidden_dims, output_dim, num_branches=2, learnable=True, device='cpu'):
        super().__init__()
        self.input_size = input_size
        self.device = device
        self.num_branches = num_branches

        self.dense_1 = MultiBranchDH_LIF(input_size, hidden_dims, num_branches, learnable, device)
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

def train_model(model, epochs, device):
    """训练模型"""
    
    # 设置优化器
    base_params = [model.dense_2.weight, model.dense_2.bias]
    for branch_layer in model.dense_1.branch_layers:
        base_params.append(branch_layer.weight)

    optimizer_params = [{'params': base_params}]

    # 添加时间常数参数
    optimizer_params.append({'params': model.dense_1.tau_m, 'lr': learning_rate})
    for tau in model.dense_1.branch_taus:
        optimizer_params.append({'params': tau, 'lr': learning_rate})

    optimizer = torch.optim.Adam(optimizer_params, lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)

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

        if epoch % 5 == 0:
            print(f'  Epoch {epoch+1:3d}: Loss={train_loss_sum/log_interval:.4f}, Acc={acc:.1f}%, Best={best_acc:.1f}%')

    return best_acc

def run_branch_comparison_experiment():
    """运行分支数量对比实验"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🔧 使用设备: {device}")

    # 要测试的分支数量
    branch_numbers = [1, 2, 4, 8]
    num_trials = 3
    epochs = 25  # 减少epoch以加快实验

    results = {}
    start_time_exp = time.time()

    print(f"\n🧪 开始分支数量对比实验...")
    print(f"分支数量: {branch_numbers}")
    print(f"参数: batch_size={batch_size}, hidden_dims={hidden_dims}, lr={learning_rate}")

    for num_branches in branch_numbers:
        print(f"\n📊 测试 {num_branches} 分支")
        print("="*50)

        trial_results = []

        for trial in range(num_trials):
            print(f"  🔄 试验 {trial+1}/{num_trials}")

            model = BranchNumberDH_SFNN(
                input_size=channel_size*2,
                hidden_dims=hidden_dims,
                output_dim=output_dim,
                num_branches=num_branches,
                learnable=True,
                device=device
            )
            model.to(device)

            acc = train_model(model, epochs=epochs, device=device)
            trial_results.append(acc)

            print(f"    ✅ 试验{trial+1}完成: {acc:.1f}%")

        mean_acc = np.mean(trial_results)
        std_acc = np.std(trial_results)
        results[f"{num_branches}-Branch"] = {
            'num_branches': num_branches,
            'mean': mean_acc,
            'std': std_acc,
            'trials': trial_results
        }

        print(f"  📈 {num_branches}分支结果: {mean_acc:.1f}% ± {std_acc:.1f}%")

    # 显示最终结果
    total_time = time.time() - start_time_exp
    print(f"\n🎉 分支数量对比实验完成! 总用时: {total_time/60:.1f}分钟")
    print("="*60)
    print("分支数量对比结果:")
    print("="*60)

    for exp_name, result in results.items():
        mean_acc = result['mean']
        std_acc = result['std']
        num_branches = result['num_branches']
        print(f"{num_branches:2d} 分支: {mean_acc:5.1f}% ± {std_acc:4.1f}%")

    # 保存结果
    os.makedirs("results", exist_ok=True)

    # 保存为JSON格式
    results_json = {}
    for key, value in results.items():
        results_json[key] = {
            'num_branches': int(value['num_branches']),
            'mean': float(value['mean']),
            'std': float(value['std']),
            'trials': [float(x) for x in value['trials']]
        }

    with open("results/branch_comparison_results.json", "w") as f:
        json.dump(results_json, f, indent=2)

    # 也保存为torch格式
    torch.save(results, "results/branch_comparison_results.pth")
    print(f"\n💾 结果已保存到: results/branch_comparison_results.json")

    return results

def create_visualization(results):
    """创建可视化图表"""

    # 提取数据
    branch_nums = []
    mean_accs = []
    std_accs = []

    for key in sorted(results.keys(), key=lambda x: results[x]['num_branches']):
        result = results[key]
        branch_nums.append(result['num_branches'])
        mean_accs.append(result['mean'])
        std_accs.append(result['std'])

    # 创建2x2子图布局
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[
            'a) Performance vs Branch Number',
            'b) Performance Improvement Analysis',
            'c) Parameter Efficiency',
            'd) Computational Complexity'
        ],
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )

    # Panel A: 性能对比
    fig.add_trace(go.Scatter(
        x=branch_nums,
        y=mean_accs,
        error_y=dict(type='data', array=std_accs, visible=True),
        mode='lines+markers',
        name='Test Accuracy',
        line=dict(color='#2E86AB', width=3),
        marker=dict(size=10)
    ), row=1, col=1)

    # Panel B: 性能提升分析
    baseline_acc = mean_accs[0]  # 1分支作为基线
    improvements = [acc - baseline_acc for acc in mean_accs]

    fig.add_trace(go.Bar(
        x=branch_nums,
        y=improvements,
        name='Improvement over 1-Branch',
        marker_color='#FF6B6B',
        text=[f'+{imp:.1f}%' if imp > 0 else f'{imp:.1f}%' for imp in improvements],
        textposition='outside'
    ), row=1, col=2)

    # Panel C: 参数效率
    # 估算参数数量 (简化计算)
    param_counts = []
    for num_branches in branch_nums:
        # 每个分支: input_per_branch * hidden_dims
        input_per_branch = (channel_size * 2) // num_branches
        branch_params = num_branches * input_per_branch * hidden_dims
        # 输出层: hidden_dims * output_dim
        output_params = hidden_dims * output_dim
        # 时间常数: hidden_dims * (1 + num_branches)
        tau_params = hidden_dims * (1 + num_branches)
        total_params = branch_params + output_params + tau_params
        param_counts.append(total_params)

    # 计算参数效率 (准确率/参数数量)
    param_efficiency = [acc / params * 1000 for acc, params in zip(mean_accs, param_counts)]

    fig.add_trace(go.Scatter(
        x=branch_nums,
        y=param_efficiency,
        mode='lines+markers',
        name='Parameter Efficiency',
        line=dict(color='#4BC0C0', width=3),
        marker=dict(size=10)
    ), row=2, col=1)

    # Panel D: 计算复杂度
    # 理论复杂度 (与分支数量成正比)
    theoretical_complexity = branch_nums
    # 实际性能提升
    actual_improvement = [acc / baseline_acc for acc in mean_accs]

    fig.add_trace(go.Scatter(
        x=branch_nums,
        y=theoretical_complexity,
        mode='lines+markers',
        name='Theoretical Complexity',
        line=dict(color='#FF9F40', width=3, dash='dash'),
        marker=dict(size=8)
    ), row=2, col=2)

    fig.add_trace(go.Scatter(
        x=branch_nums,
        y=actual_improvement,
        mode='lines+markers',
        name='Performance Ratio',
        line=dict(color='#9966FF', width=3),
        marker=dict(size=10)
    ), row=2, col=2)

    # 更新布局
    fig.update_layout(
        title={
            'text': "Branch Number Comparison: Systematic Analysis of DH-SNN Multi-branch Architecture",
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 18, 'color': '#2E86AB', 'family': 'Arial Black'}
        },
        height=800,
        width=1200,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.15,
            xanchor="center",
            x=0.5
        ),
        font=dict(size=12, family='Arial')
    )

    # 更新坐标轴
    fig.update_xaxes(title_text="Number of Branches", row=1, col=1)
    fig.update_yaxes(title_text="Test Accuracy (%)", row=1, col=1)

    fig.update_xaxes(title_text="Number of Branches", row=1, col=2)
    fig.update_yaxes(title_text="Accuracy Improvement (%)", row=1, col=2)

    fig.update_xaxes(title_text="Number of Branches", row=2, col=1)
    fig.update_yaxes(title_text="Efficiency (Acc/1K Params)", row=2, col=1)

    fig.update_xaxes(title_text="Number of Branches", row=2, col=2)
    fig.update_yaxes(title_text="Relative Ratio", row=2, col=2)

    # 保存图表
    html_path = "results/branch_comparison_analysis.html"
    fig.write_html(html_path)
    print(f"✅ HTML图表已保存: {html_path}")

    try:
        png_path = "results/branch_comparison_analysis.png"
        fig.write_image(png_path, width=1200, height=800, scale=2)
        print(f"✅ PNG图表已保存: {png_path}")
    except Exception as e:
        print(f"⚠️ PNG保存失败: {e}")

    return fig

def main():
    """主函数"""
    try:
        # 运行实验
        results = run_branch_comparison_experiment()

        # 创建可视化
        print("\n🎨 创建可视化图表...")
        create_visualization(results)

        print(f"\n🏁 分支数量对比实验成功完成!")

        return results

    except Exception as e:
        print(f"\n❌ 实验失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()
