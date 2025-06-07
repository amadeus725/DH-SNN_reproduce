#!/usr/bin/env python3
"""
基于SpikingJelly框架的多时间尺度XOR实验
严格按照原论文参数，使用SpikingJelly的标准组件
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from spikingjelly.activation_based import neuron, functional, surrogate, layer
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

class SpikingJellyDH_SNN(nn.Module):
    """基于SpikingJelly的DH-SNN实现 - 修复版本"""

    def __init__(self, input_size, hidden_size, output_size, num_branches=2):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_branches = num_branches

        # 使用更简单的分支连接方式
        # Branch 1: 连接到前半部分输入 (signal 1)
        # Branch 2: 连接到后半部分输入 (signal 2)
        self.branch1_layer = layer.Linear(input_size // 2, hidden_size, bias=False)
        self.branch2_layer = layer.Linear(input_size // 2, hidden_size, bias=False)

        # 可学习的时间常数参数
        self.tau_n = nn.Parameter(torch.empty(num_branches, hidden_size).uniform_(2, 6))
        self.tau_m = nn.Parameter(torch.empty(hidden_size).uniform_(0, 4))

        # 树突电流状态
        self.dendritic_current1 = None
        self.dendritic_current2 = None
        self.membrane_potential = None
        self.spike_output = None

    def reset_states(self, batch_size):
        """重置神经元状态"""
        self.dendritic_current1 = torch.zeros(batch_size, self.hidden_size).to(device)
        self.dendritic_current2 = torch.zeros(batch_size, self.hidden_size).to(device)
        self.membrane_potential = torch.rand(batch_size, self.hidden_size).to(device)
        self.spike_output = torch.zeros(batch_size, self.hidden_size).to(device)

    def surrogate_gradient(self, x):
        """简单的代理梯度函数"""
        return SurrogateGradient.apply(x)

    def forward(self, x):
        """前向传播 - 手动实现LIF动态"""
        batch_size = x.size(0)

        # 分割输入：前半部分给Branch1，后半部分给Branch2
        input1 = x[:, :self.input_size//2]  # Signal 1 (低频)
        input2 = x[:, self.input_size//2:]  # Signal 2 (高频)

        # 分支线性变换
        branch1_input = self.branch1_layer(input1)
        branch2_input = self.branch2_layer(input2)

        # 更新树突电流 - 使用不同的时间常数
        beta1 = torch.sigmoid(self.tau_n[0])  # Branch 1 时间常数
        beta2 = torch.sigmoid(self.tau_n[1])  # Branch 2 时间常数

        self.dendritic_current1 = beta1 * self.dendritic_current1 + (1 - beta1) * branch1_input
        self.dendritic_current2 = beta2 * self.dendritic_current2 + (1 - beta2) * branch2_input

        # 汇总树突电流
        total_current = self.dendritic_current1 + self.dendritic_current2

        # 更新膜电位 - LIF动态
        alpha = torch.sigmoid(self.tau_m)
        R_m = 1.0
        v_th = 1.0

        self.membrane_potential = (alpha * self.membrane_potential +
                                 (1 - alpha) * R_m * total_current -
                                 v_th * self.spike_output)

        # 生成脉冲
        inputs_ = self.membrane_potential - v_th
        self.spike_output = self.surrogate_gradient(inputs_)

        return self.spike_output

# 保持原来的代理梯度函数
class SurrogateGradient(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return input.gt(0).float()

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()

        # 简化的代理梯度
        lens = 0.5
        gamma = 0.5
        temp = torch.exp(-(input**2)/(2*lens**2))/torch.sqrt(2*torch.tensor(torch.pi))/lens
        return grad_input * temp.float() * gamma

class SpikingJellyModel(nn.Module):
    """基于SpikingJelly的完整模型 - 修复版本"""

    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.dh_snn = SpikingJellyDH_SNN(input_size, hidden_size, hidden_size, num_branches=2)
        self.output_layer = layer.Linear(hidden_size, output_size)
        self.criterion = nn.CrossEntropyLoss()

    def init_states(self, batch_size):
        """初始化神经元状态"""
        self.dh_snn.reset_states(batch_size)

    def forward(self, input_data, target):
        """前向传播"""
        batch_size, seq_len, input_size = input_data.shape

        # 初始化状态（不使用functional.reset_net）
        self.init_states(batch_size)

        output_history = []
        total_loss = 0
        total_correct = 0
        total_samples = 0

        for t in range(seq_len):
            input_t = input_data[:, t, :].float()

            # DH-SNN处理
            spike_output = self.dh_snn(input_t)

            # 输出层
            output = self.output_layer(spike_output)
            output_history.append(output.cpu())

            # 计算损失和准确率 (只在特定时间步)
            if (((t - start_time) % (coding_time + remain_time)) > remain_time) and (t > start_time):
                output_prob = F.softmax(output, dim=1)
                loss = self.criterion(output_prob, target[:, t].long())
                total_loss += loss

                _, predicted = torch.max(output_prob.data, 1)
                labels = target[:, t].cpu()
                predicted = predicted.cpu()
                total_correct += (predicted == labels).sum()
                total_samples += labels.size(0)

        output_history = torch.stack(output_history, dim=1)
        return total_loss, output_history, total_correct, total_samples

def train_spikingjelly_model(epochs=150):
    """训练SpikingJelly模型"""
    print("🚀 开始训练 - SpikingJelly实现")
    print(f"📊 参数: time_steps={time_steps}, batch_size={batch_size}, hidden_dims={hidden_dims}")
    print(f"🔧 学习率: {learning_rate}, 分支数: 2")
    
    # 创建模型
    model = SpikingJellyModel(channel_size * 2, hidden_dims, 2)
    model.to(device)
    
    # 优化器 - 简化版本
    base_params = [
        model.output_layer.weight,
        model.output_layer.bias,
        model.dh_snn.branch1_layer.weight,
        model.dh_snn.branch2_layer.weight,
    ]

    optimizer = torch.optim.Adam([
        {'params': base_params, 'lr': learning_rate},
        {'params': model.dh_snn.tau_n, 'lr': learning_rate},
        {'params': model.dh_snn.tau_m, 'lr': learning_rate},
    ])
    
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)
    
    # 训练历史
    training_history = {
        'epochs': [],
        'losses': [],
        'accuracies': [],
        'tau_n_branch1': [],
        'tau_n_branch2': [],
        'tau_m': []
    }
    
    log_interval = 100
    best_loss = float('inf')
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        epoch_correct = 0
        epoch_total = 0
        
        for batch_idx in range(log_interval):
            # 生成数据
            data, target = get_batch()
            data = data.to(device)
            target = target.to(device)
            
            optimizer.zero_grad()
            loss, output, correct, total = model(data, target)
            
            if loss > 0:  # 确保有有效损失
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 20)
                optimizer.step()
            
            epoch_loss += loss.item() if loss > 0 else 0
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
            tau_n_sigmoid = torch.sigmoid(model.dh_snn.tau_n)
            tau_m_sigmoid = torch.sigmoid(model.dh_snn.tau_m)
            
            training_history['tau_n_branch1'].append(tau_n_sigmoid[0].mean().item())
            training_history['tau_n_branch2'].append(tau_n_sigmoid[1].mean().item())
            training_history['tau_m'].append(tau_m_sigmoid.mean().item())
        
        print(f'Epoch {epoch:3d}: Loss={avg_loss:.4f}, Acc={accuracy:.3f}, '
              f'τ_n1={training_history["tau_n_branch1"][-1]:.3f}, '
              f'τ_n2={training_history["tau_n_branch2"][-1]:.3f}, '
              f'τ_m={training_history["tau_m"][-1]:.3f}')
        
        # 保存最佳模型
        if avg_loss < best_loss and avg_loss > 0:
            best_loss = avg_loss
            torch.save(model.state_dict(), 'best_spikingjelly_model.pth')
    
    return model, training_history

if __name__ == "__main__":
    print("📄 SpikingJelly多时间尺度XOR实验")
    print("=" * 60)
    
    # 训练模型
    model, history = train_spikingjelly_model(epochs=100)
    
    print(f"\n✅ 训练完成!")
    print(f"🎯 最终准确率: {history['accuracies'][-1]:.3f}")
    print(f"📊 最终损失: {history['losses'][-1]:.4f}")
    
    # 保存训练历史
    torch.save(history, 'spikingjelly_training_history.pth')
    print("💾 训练历史已保存")
