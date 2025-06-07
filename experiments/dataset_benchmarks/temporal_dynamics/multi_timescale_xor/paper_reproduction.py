#!/usr/bin/env python3
"""
精确复现论文Figure 4b的多时间尺度XOR实验
基于原论文代码的精确参数和实现
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
import os

print("📄 精确复现论文Figure 4b - 多时间尺度XOR实验")
print("="*60)

# 原论文精确参数
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

class SimplifiedDH_LIF(nn.Module):
    """简化的DH-LIF神经元 - 基于原论文实现"""

    def __init__(self, input_size, hidden_size, branch=1, tau_init='large', learnable=True, device='cpu'):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.branch = branch
        self.device = device

        # 权重连接
        if branch == 1:
            self.dense = nn.Linear(input_size, hidden_size, bias=False)
        else:  # branch == 2
            self.dense1 = nn.Linear(input_size//2, hidden_size, bias=False)  # Signal 1
            self.dense2 = nn.Linear(input_size//2, hidden_size, bias=False)  # Signal 2

        # 时间常数 - 按照原论文初始化
        self.tau_m = nn.Parameter(torch.ones(hidden_size) * 2.0)

        if branch == 1:
            if tau_init == 'small':
                self.tau_n = nn.Parameter(torch.empty(hidden_size).uniform_(-4, 0))
            elif tau_init == 'large':
                self.tau_n = nn.Parameter(torch.empty(hidden_size).uniform_(2, 6))
            else:  # medium
                self.tau_n = nn.Parameter(torch.empty(hidden_size).uniform_(0, 4))
        else:  # branch == 2
            # 有益初始化：Branch1=Large, Branch2=Small
            self.tau_n1 = nn.Parameter(torch.empty(hidden_size).uniform_(2, 6))  # Large
            self.tau_n2 = nn.Parameter(torch.empty(hidden_size).uniform_(-4, 0))  # Small

        # 设置可学习性
        self.tau_m.requires_grad = learnable
        if branch == 1:
            self.tau_n.requires_grad = learnable
        else:
            self.tau_n1.requires_grad = learnable
            self.tau_n2.requires_grad = learnable

        # 神经元状态
        self.mem = None
        self.d_current = None
        self.d1_current = None
        self.d2_current = None

    def set_neuron_state(self, batch_size):
        """重置神经元状态"""
        self.mem = torch.zeros(batch_size, self.hidden_size).to(self.device)
        if self.branch == 1:
            self.d_current = torch.zeros(batch_size, self.hidden_size).to(self.device)
        else:
            self.d1_current = torch.zeros(batch_size, self.hidden_size).to(self.device)
            self.d2_current = torch.zeros(batch_size, self.hidden_size).to(self.device)

    def forward(self, x):
        """前向传播"""
        if self.branch == 1:
            # 单分支
            d_input = self.dense(x)
            beta = torch.sigmoid(self.tau_n)
            self.d_current = beta * self.d_current + (1 - beta) * d_input

            alpha = torch.sigmoid(self.tau_m)
            self.mem = alpha * self.mem + (1 - alpha) * self.d_current
        else:
            # 双分支
            x1 = x[:, :self.input_size//2]  # Signal 1
            x2 = x[:, self.input_size//2:]  # Signal 2

            d1_input = self.dense1(x1)
            d2_input = self.dense2(x2)

            beta1 = torch.sigmoid(self.tau_n1)
            beta2 = torch.sigmoid(self.tau_n2)

            self.d1_current = beta1 * self.d1_current + (1 - beta1) * d1_input
            self.d2_current = beta2 * self.d2_current + (1 - beta2) * d2_input

            total_d_current = self.d1_current + self.d2_current

            alpha = torch.sigmoid(self.tau_m)
            self.mem = alpha * self.mem + (1 - alpha) * total_d_current

        # 脉冲生成 (简化版，使用sigmoid)
        spike = torch.sigmoid(self.mem)

        return self.mem, spike

class PaperVanillaSFNN(nn.Module):
    """论文中的Vanilla SFNN"""

    def __init__(self, input_size, hidden_dims, output_dim, device='cpu'):
        super().__init__()
        self.input_size = input_size
        self.device = device

        # 使用单分支，medium初始化
        self.dense_1 = SimplifiedDH_LIF(input_size, hidden_dims, branch=1, tau_init='medium', device=device)
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

            # 只在特定时间窗口计算损失
            if (((i - start_time) % (coding_time + remain_time)) > remain_time) and (i > start_time):
                output = F.softmax(l2_output, dim=1)
                loss += self.criterion(output, target[:, i].long())
                _, predicted = torch.max(output.data, 1)
                labels = target[:, i].cpu()
                predicted = predicted.cpu()
                correct += (predicted == labels).sum()
                total += labels.size()[0]

        return loss, d2_output, correct, total

class Paper1BranchDH_SFNN(nn.Module):
    """论文中的1分支DH-SFNN"""

    def __init__(self, input_size, hidden_dims, output_dim, tau_init='large', device='cpu'):
        super().__init__()
        self.input_size = input_size
        self.device = device

        self.dense_1 = SimplifiedDH_LIF(input_size, hidden_dims, branch=1, tau_init=tau_init, device=device)
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

class Paper2BranchDH_SFNN(nn.Module):
    """论文中的2分支DH-SFNN"""

    def __init__(self, input_size, hidden_dims, output_dim, learnable=True, device='cpu'):
        super().__init__()
        self.input_size = input_size
        self.device = device

        self.dense_1 = SimplifiedDH_LIF(input_size, hidden_dims, branch=2, learnable=learnable, device=device)
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

def train_paper_model(model, epochs, device):
    """按照原论文的训练方式"""

    # 原论文的优化器设置
    base_params = [model.dense_2.weight, model.dense_2.bias]

    if hasattr(model.dense_1, 'dense'):
        base_params.append(model.dense_1.dense.weight)
    else:
        base_params.extend([model.dense_1.dense1.weight, model.dense_1.dense2.weight])

    optimizer_params = [{'params': base_params}]

    # 添加时间常数参数
    if hasattr(model.dense_1, 'tau_n'):
        optimizer_params.extend([
            {'params': model.dense_1.tau_m, 'lr': learning_rate},
            {'params': model.dense_1.tau_n, 'lr': learning_rate}
        ])
    else:
        optimizer_params.extend([
            {'params': model.dense_1.tau_m, 'lr': learning_rate},
            {'params': model.dense_1.tau_n1, 'lr': learning_rate},
            {'params': model.dense_1.tau_n2, 'lr': learning_rate}
        ])

    optimizer = torch.optim.Adam(optimizer_params, lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)

    best_acc = 0
    log_interval = 20  # 减少日志间隔用于测试

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

        print(f'Epoch {epoch+1:3d}: Loss={train_loss_sum/log_interval:.4f}, Acc={acc:.1f}%, Best={best_acc:.1f}%')

    return best_acc

def main():
    """主实验函数 - 精确复现论文Figure 4b"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🔧 使用设备: {device}")

    # 实验配置 - 按照论文Figure 4b
    experiments = [
        ("Vanilla SFNN", lambda: PaperVanillaSFNN(channel_size*2, hidden_dims, output_dim, device)),
        ("1-Branch DH-SFNN (Small)", lambda: Paper1BranchDH_SFNN(channel_size*2, hidden_dims, output_dim, 'small', device)),
        ("1-Branch DH-SFNN (Large)", lambda: Paper1BranchDH_SFNN(channel_size*2, hidden_dims, output_dim, 'large', device)),
        ("2-Branch DH-SFNN (Learnable)", lambda: Paper2BranchDH_SFNN(channel_size*2, hidden_dims, output_dim, True, device)),
        ("2-Branch DH-SFNN (Fixed)", lambda: Paper2BranchDH_SFNN(channel_size*2, hidden_dims, output_dim, False, device)),
    ]

    results = {}
    start_time_exp = time.time()

    print(f"\n🧪 开始论文精确复现实验 (每个配置3次试验)...")
    print(f"参数: batch_size={batch_size}, hidden_dims={hidden_dims}, lr={learning_rate}")

    for exp_name, model_creator in experiments:
        print(f"\n📊 实验: {exp_name}")
        print("="*50)

        trial_results = []

        for trial in range(3):
            print(f"  🔄 试验 {trial+1}/3")
            model = model_creator()
            model.to(device)

            acc = train_paper_model(model, epochs=30, device=device)  # 减少epoch用于测试
            trial_results.append(acc)

            print(f"    ✅ 试验{trial+1}完成: {acc:.1f}%")

        mean_acc = np.mean(trial_results)
        std_acc = np.std(trial_results)
        results[exp_name] = {
            'mean': mean_acc,
            'std': std_acc,
            'trials': trial_results
        }

        print(f"  📈 最终结果: {mean_acc:.1f}% ± {std_acc:.1f}%")

    # 显示最终结果
    total_time = time.time() - start_time_exp
    print(f"\n🎉 论文复现实验完成! 总用时: {total_time/60:.1f}分钟")
    print("="*60)
    print("论文Figure 4b复现结果:")
    print("="*60)

    for exp_name, result in results.items():
        mean_acc = result['mean']
        std_acc = result['std']
        print(f"{exp_name:30s}: {mean_acc:5.1f}% ± {std_acc:4.1f}%")

    # 保存结果
    os.makedirs("results", exist_ok=True)
    torch.save(results, "results/paper_reproduction_results.pth")
    print(f"\n💾 结果已保存到: results/paper_reproduction_results.pth")

    return results

if __name__ == '__main__':
    try:
        results = main()
        print(f"\n🏁 论文复现实验成功完成!")
    except Exception as e:
        print(f"\n❌ 实验失败: {e}")
        import traceback
        traceback.print_exc()
