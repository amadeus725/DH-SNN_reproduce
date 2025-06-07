#!/usr/bin/env python3
"""
严格按照原论文参数的多时间尺度XOR实验
基于 multitimescale_xor/multi_xor_snn.py 的精确复现
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import time
import os
from datetime import datetime

# 设置随机种子
torch.manual_seed(42)
np.random.seed(42)

# 原论文精确参数
time_steps = 100  # 总时间步数
channel = 2  # signal1和signal2
channel_rate = [0.2, 0.6]  # 高低发放率
noise_rate = 0.01
channel_size = 20  # 每个通道的神经元数
coding_time = 10  # 信号持续时间
remain_time = 5   # 间隔时间
start_time = 10   # 开始时间

batch_size = 500  # 批次大小
hidden_dims = 16  # 隐藏层大小
learning_rate = 1e-2  # 学习率

# 设备设置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🔧 使用设备: {device}")

# XOR标签
label = torch.zeros(len(channel_rate), len(channel_rate))
label[1][0] = 1
label[0][1] = 1

def get_batch():
    """生成多时间尺度脉冲XOR问题数据集 - 原论文实现"""
    # 构建第一个序列
    values = torch.rand(batch_size, time_steps, channel_size*2, requires_grad=False) <= noise_rate
    targets = torch.zeros(time_steps, batch_size, requires_grad=False).int()
    
    # 构建signal 1
    init_pattern = torch.randint(len(channel_rate), size=(batch_size,))
    # 生成脉冲
    prob_matrix = torch.ones(start_time, channel_size, batch_size) * torch.tensor(channel_rate)[init_pattern]
    add_patterns = torch.bernoulli(prob_matrix).permute(2, 0, 1).bool()
    values[:, :start_time, :channel_size] = values[:, :start_time, :channel_size] | add_patterns
    
    # 构建signal 2
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
    
    return values, targets.transpose(0, 1).contiguous()

# 原论文精确的DH-SFNN实现
class OriginalDH_SFNN(nn.Module):
    """严格按照原论文实现的DH-SFNN"""

    def __init__(self, input_dim, output_dim, num_branches=2, vth=1.0, dt=1):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_branches = num_branches
        self.vth = vth
        self.dt = dt

        # 计算padding
        self.pad = ((input_dim) // num_branches * num_branches + num_branches - input_dim) % num_branches

        # 线性层 - 原论文实现
        self.dense = nn.Linear(input_dim + self.pad, output_dim * num_branches, bias=False)

        # 时间常数参数
        self.tau_m = nn.Parameter(torch.empty(output_dim).uniform_(0, 4))
        self.tau_n = nn.Parameter(torch.empty(output_dim, num_branches).uniform_(2, 6))

        # 创建掩码
        self.create_mask()

        # 状态变量
        self.reset_states()

    def create_mask(self):
        """创建连接掩码 - 原论文实现"""
        input_size = self.input_dim + self.pad
        self.mask = torch.zeros(self.output_dim * self.num_branches, input_size).to(device)
        for i in range(self.output_dim):
            for j in range(self.num_branches):
                start_idx = j * input_size // self.num_branches
                end_idx = (j + 1) * input_size // self.num_branches
                self.mask[i * self.num_branches + j, start_idx:end_idx] = 1

    def apply_mask(self):
        """应用掩码"""
        self.dense.weight.data = self.dense.weight.data * self.mask

    def reset_states(self):
        """重置状态"""
        self.mem = None
        self.spike = None
        self.d_input = None
        self.v_th = None

    def set_neuron_state(self, batch_size):
        """设置神经元状态"""
        self.mem = torch.rand(batch_size, self.output_dim).to(device)
        self.spike = torch.rand(batch_size, self.output_dim).to(device)
        self.d_input = torch.zeros(batch_size, self.output_dim, self.num_branches).to(device)
        self.v_th = torch.ones(batch_size, self.output_dim).to(device) * self.vth

    def forward(self, input_spike):
        """前向传播 - 原论文实现"""
        # 时间常数
        beta = torch.sigmoid(self.tau_n)
        alpha = torch.sigmoid(self.tau_m)

        # 添加padding
        padding = torch.zeros(input_spike.size(0), self.pad).to(device)
        k_input = torch.cat((input_spike, padding), 1)

        # 更新树突电流 - 关键：不重置
        dense_output = self.dense(k_input).reshape(-1, self.output_dim, self.num_branches)
        self.d_input = beta * self.d_input + (1 - beta) * dense_output

        # 汇总树突电流
        l_input = self.d_input.sum(dim=2, keepdim=False)

        # 更新膜电位 - 软重置LIF
        R_m = 1.0
        self.mem = self.mem * alpha + (1 - alpha) * R_m * l_input - self.v_th * self.spike

        # 生成脉冲 - 使用代理梯度
        inputs_ = self.mem - self.v_th
        self.spike = self.surrogate_gradient(inputs_)

        return self.mem, self.spike

    def surrogate_gradient(self, x):
        """多高斯代理梯度函数 - 原论文实现"""
        return SurrogateGradient.apply(x)

# 代理梯度函数
class SurrogateGradient(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return input.gt(0).float()

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()

        # 多高斯代理梯度
        lens = 0.5
        scale = 6.0
        hight = 0.15
        gamma = 0.5

        def gaussian(x, mu=0., sigma=0.5):
            return torch.exp(-((x - mu) ** 2) / (2 * sigma ** 2)) / torch.sqrt(2 * torch.tensor(torch.pi)) / sigma

        temp = gaussian(input, mu=0., sigma=lens) * (1. + hight) \
             - gaussian(input, mu=lens, sigma=scale * lens) * hight \
             - gaussian(input, mu=-lens, sigma=scale * lens) * hight

        return grad_input * temp.float() * gamma

class OriginalPaperModel(nn.Module):
    """原论文模型结构"""

    def __init__(self, input_size, hidden_dims, output_dim):
        super().__init__()
        self.input_size = input_size
        self.dh_layer = OriginalDH_SFNN(input_size, hidden_dims, num_branches=2)
        self.output_layer = nn.Linear(hidden_dims, output_dim)
        self.criterion = nn.CrossEntropyLoss()

    def init(self, batch_size):
        """初始化神经元状态"""
        self.dh_layer.set_neuron_state(batch_size)

    def forward(self, input_data, target):
        """前向传播 - 原论文逻辑"""
        batch_size, seq_num, input_size = input_data.shape

        output_history = torch.zeros(batch_size, seq_num, 2)
        total_loss = 0
        total_correct = 0
        total_samples = 0

        for i in range(seq_num):
            input_x = input_data[:, i, :].float()  # 转换为float类型
            mem, spikes = self.dh_layer(input_x)
            output = self.output_layer(spikes)
            output_history[:, i, :] = output.cpu()

            # 只在特定时间步计算损失 (原论文逻辑)
            if (((i - start_time) % (coding_time + remain_time)) > remain_time) and (i > start_time):
                output_prob = F.softmax(output, dim=1)
                loss = self.criterion(output_prob, target[:, i].long())
                total_loss += loss

                _, predicted = torch.max(output_prob.data, 1)
                labels = target[:, i].cpu()
                predicted = predicted.cpu()
                total_correct += (predicted == labels).sum()
                total_samples += labels.size(0)

        return total_loss, output_history, total_correct, total_samples

def train_original_paper_model(epochs=150):
    """训练模型 - 原论文参数"""
    print("🚀 开始训练 - 原论文参数设置")
    print(f"📊 参数: time_steps={time_steps}, batch_size={batch_size}, hidden_dims={hidden_dims}")
    print(f"🔧 学习率: {learning_rate}, 分支数: 2")
    
    # 创建模型
    model = OriginalPaperModel(channel_size * 2, hidden_dims, 2)
    model.to(device)
    
    # 优化器 - 原论文设置
    base_params = [
        model.output_layer.weight,
        model.output_layer.bias,
        model.dh_layer.dense.weight,
    ]

    optimizer = torch.optim.Adam([
        {'params': base_params, 'lr': learning_rate},
        {'params': model.dh_layer.tau_m, 'lr': learning_rate},
        {'params': model.dh_layer.tau_n, 'lr': learning_rate},
    ], lr=learning_rate)
    
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)
    
    # 训练循环
    log_interval = 100
    best_loss = float('inf')
    
    training_history = {
        'epochs': [],
        'losses': [],
        'accuracies': [],
        'tau_n_branch1': [],
        'tau_n_branch2': []
    }
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        epoch_correct = 0
        epoch_total = 0
        
        for batch_idx in range(log_interval):
            model.init(batch_size)
            
            # 生成数据
            data, target = get_batch()
            data = data.detach().to(device)
            target = target.detach().to(device)
            
            optimizer.zero_grad()
            loss, output, correct, total = model(data, target)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 20)
            optimizer.step()

            # 应用掩码 - 原论文关键步骤
            model.dh_layer.apply_mask()
            
            epoch_loss += loss.item()
            epoch_correct += correct.item()
            epoch_total += total
            
        scheduler.step()
        
        # 记录训练历史
        avg_loss = epoch_loss / log_interval
        accuracy = epoch_correct / epoch_total if epoch_total > 0 else 0
        
        training_history['epochs'].append(epoch)
        training_history['losses'].append(avg_loss)
        training_history['accuracies'].append(accuracy)
        
        # 记录时间常数
        with torch.no_grad():
            tau_n_sigmoid = torch.sigmoid(model.dh_layer.tau_n)
            training_history['tau_n_branch1'].append(tau_n_sigmoid[:, 0].mean().item())
            training_history['tau_n_branch2'].append(tau_n_sigmoid[:, 1].mean().item())
        
        print(f'Epoch {epoch:3d}: Loss={avg_loss:.4f}, Acc={accuracy:.3f}, '
              f'τ_n1={training_history["tau_n_branch1"][-1]:.3f}, '
              f'τ_n2={training_history["tau_n_branch2"][-1]:.3f}')
        
        # 保存最佳模型
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), 'best_original_paper_model.pth')
    
    return model, training_history

if __name__ == "__main__":
    print("📄 原论文多时间尺度XOR实验")
    print("=" * 60)
    
    # 训练模型
    model, history = train_original_paper_model(epochs=150)
    
    print(f"\n✅ 训练完成!")
    print(f"🎯 最终准确率: {history['accuracies'][-1]:.3f}")
    print(f"📊 最终损失: {history['losses'][-1]:.4f}")
