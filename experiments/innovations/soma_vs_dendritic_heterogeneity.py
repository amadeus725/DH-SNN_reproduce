#!/usr/bin/env python3
"""
胞体异质性vs树突异质性延迟异或对比实验
=============================================

基于SpikingJelly框架的SH-SNN vs DH-SNN vs 普通SNN三方对比实验
使用延迟异或任务验证不同异质性机制的时间处理能力

SH-SNN: 胞体异质性脉冲神经网络 (Soma Heterogeneity SNN)
DH-SNN: 树突异质性脉冲神经网络 (Dendritic Heterogeneity SNN)

作者: DH-SNN Reproduction Team
日期: 2025-06-14
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import time
import json
import math
from pathlib import Path

# 添加项目路径
current_dir = Path(__file__).parent.parent.parent
sys.path.append(str(current_dir))

# SpikingJelly导入
from spikingjelly.activation_based import neuron, functional, layer, surrogate

# 修复导入问题：直接定义需要的函数而不是导入
def setup_seed(seed=42):
    """设置随机种子以保证实验可重现性"""
    import random
    import numpy as np
    import torch
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

# 多高斯替代函数
class MultiGaussianSurrogate(torch.autograd.Function):
    """多高斯替代函数"""
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return input.gt(0).float()

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        lens = 0.5
        scale = 6.0
        height = 0.15
        gamma = 0.5

        def gaussian(x, mu=0., sigma=0.5):
            return torch.exp(-((x - mu) ** 2) / (2 * sigma ** 2)) / torch.sqrt(2 * torch.tensor(math.pi)) / sigma

        temp = gaussian(input, mu=0., sigma=lens) * (1. + height) \
             - gaussian(input, mu=lens, sigma=scale * lens) * height \
             - gaussian(input, mu=-lens, sigma=scale * lens) * height

        return grad_input * temp.float() * gamma

multi_gaussian_surrogate = MultiGaussianSurrogate.apply

# 数据生成函数
def generate_delayed_xor_data(batch_size, seq_length, delay, num_samples=1000):
    """
    生成延迟异或任务数据 - 完全复制原论文的单脉冲方式
    
    参数:
        batch_size: 批次大小
        seq_length: 序列长度
        delay: 延迟步数
        num_samples: 样本数量
        
    返回:
        data: 输入脉冲序列 [样本数, 时间步, 输入维度]
        labels: 目标标签 [样本数]
    """
    data = torch.zeros(num_samples, seq_length, 2)
    labels = torch.zeros(num_samples, dtype=torch.long)
    
    for i in range(num_samples):
        # 在序列开始处生成两个随机脉冲
        signal1_time = np.random.randint(5, 15)  # 第一个信号的时间
        signal2_time = signal1_time + delay      # 第二个信号延迟delay步
        
        # 确保第二个信号在序列范围内
        if signal2_time < seq_length - 10:            # 生成脉冲
            signal1_value = np.random.choice([0, 1])
            signal2_value = np.random.choice([0, 1])
            
            data[i, signal1_time, 0] = float(signal1_value)
            data[i, signal2_time, 1] = float(signal2_value)
            
            # 异或标签
            labels[i] = int(signal1_value ^ signal2_value)
        else:
            # 如果延迟太长，标签设为0
            signal1_value = np.random.choice([0, 1])
            data[i, signal1_time, 0] = float(signal1_value)
            labels[i] = 0
    
    return data, labels

def create_delayed_xor_datasets(delays):
    """创建不同延迟的异或数据集 - 按照原论文配置"""
    datasets = {}
    
    for delay in delays:
        print(f"📊 生成延迟{delay}步的异或数据...")
          # 优化的样本数量，平衡训练效果和速度
        train_data, train_labels = generate_delayed_xor_data(BATCH_SIZE, SEQ_LENGTH, delay, num_samples=1000)  # 减少训练样本
        test_data, test_labels = generate_delayed_xor_data(BATCH_SIZE, SEQ_LENGTH, delay, num_samples=200)   # 减少测试样本
        
        train_dataset = torch.utils.data.TensorDataset(train_data, train_labels)
        test_dataset = torch.utils.data.TensorDataset(test_data, test_labels)
        
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
        
        datasets[delay] = {
            'train_loader': train_loader,
            'test_loader': test_loader,
            'train_size': len(train_data),
            'test_size': len(test_data)
        }
        
        print(f"   训练样本: {len(train_data)}, 测试样本: {len(test_data)}")
    
    return datasets

# 普通SNN模型
class DelayedXOR_Vanilla_SNN(nn.Module):
    """普通SNN模型"""
    def __init__(self, input_size=2, hidden_size=32, output_size=1):
        super(DelayedXOR_Vanilla_SNN, self).__init__()
        
        self.fc1 = layer.Linear(input_size, hidden_size)
        self.lif1 = neuron.LIFNode(tau=2.0, v_threshold=1.0, surrogate_function=surrogate.ATan(), step_mode='s')
        
        self.fc2 = layer.Linear(hidden_size, output_size)
        self.lif2 = neuron.LIFNode(tau=2.0, v_threshold=1.0, surrogate_function=surrogate.ATan(), step_mode='s')
    
    def forward(self, x):
        batch_size, seq_len, input_dim = x.shape
        functional.reset_net(self)
        
        outputs = []
        for t in range(seq_len):
            x_t = x[:, t, :]
            h1 = self.fc1(x_t)
            s1 = self.lif1(h1)
            h2 = self.fc2(s1)
            s2 = self.lif2(h2)
            outputs.append(s2)
        
        # 延迟异或任务的决策时间点 - 统一使用后半段
        decision_start = seq_len//2
        integrated_output = torch.stack(outputs[decision_start:], dim=1).sum(dim=1)
        return integrated_output

print("🚀 胞体异质性 vs 树突异质性延迟异或对比实验")
print("="*60)

# 实验参数 - 更保守的优化配置
BATCH_SIZE = 32      # 原论文批次大小，更稳定
LEARNING_RATE = 1e-3 # 恢复原论文学习率
NUM_EPOCHS = 50      # 减少训练轮数
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 延迟异或任务参数 - 平衡版本  
SEQ_LENGTH = 300     # 适中的序列长度，确保长期记忆测试
INPUT_SIZE = 2       
HIDDEN_SIZE = 32     
OUTPUT_SIZE = 1      
DELAY_RANGE = [25, 100]  # 只测试两个关键延迟：短期和长期

# ==================== 胞体异质性模型 ====================

class DelayedXOR_SH_SNN(nn.Module):
    """
    胞体异质性SNN模型 (SH-SNN)
    在胞体层面引入异质性，不同神经元具有不同的时间常数
    """
    
    def __init__(self, input_size=INPUT_SIZE, hidden_size=HIDDEN_SIZE, output_size=OUTPUT_SIZE):
        super(DelayedXOR_SH_SNN, self).__init__()
        
        print(f"🧠 创建胞体异质性SH-SNN模型:")
        print(f"   输入维度: {input_size}")
        print(f"   隐藏维度: {hidden_size}")
        print(f"   输出维度: {output_size}")
        print(f"   异质性类型: 胞体异质性 (不同神经元不同时间常数)")
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        # 输入到隐藏层
        self.fc1 = layer.Linear(input_size, hidden_size)
        
        # 胞体异质性：每个神经元有独立的时间常数
        # 一半神经元用于短期记忆，一半用于长期记忆
        self.tau_m = nn.Parameter(torch.empty(hidden_size))
        # 初始化：前半部分神经元快速时间常数，后半部分慢速时间常数
        nn.init.uniform_(self.tau_m[:hidden_size//2], 0.0, 2.0)  # 快速神经元
        nn.init.uniform_(self.tau_m[hidden_size//2:], 3.0, 6.0)  # 慢速神经元
        
        # 输出层
        self.fc2 = layer.Linear(hidden_size, output_size)
        
        # 神经元状态
        self.membrane_potential = None
        self.spike_output = None
        
        print("✅ 胞体异质性SH-SNN模型创建完成")
    
    def reset_states(self, batch_size):
        """重置神经元状态"""
        self.membrane_potential = torch.zeros(batch_size, self.hidden_size).to(DEVICE)
        self.spike_output = torch.zeros(batch_size, self.hidden_size).to(DEVICE)
    
    def surrogate_gradient(self, x):
        """使用多高斯替代函数"""
        return multi_gaussian_surrogate(x)
    
    def forward(self, x):
        """前向传播"""
        batch_size, seq_len, input_dim = x.shape
        
        # 重置状态
        self.reset_states(batch_size)
        
        outputs = []
        for t in range(seq_len):
            x_t = x[:, t, :]  # [batch, input_size]
            
            # 输入到隐藏层
            input_current = self.fc1(x_t)
            
            # 胞体异质性：每个神经元独立的时间常数
            alpha = torch.sigmoid(self.tau_m)  # [hidden_size]
            v_th = 1.0
            
            # 膜电位更新（每个神经元独立的衰减）
            self.membrane_potential = (
                alpha * self.membrane_potential + 
                (1 - alpha) * input_current - 
                v_th * self.spike_output
            )
            
            # 脉冲生成
            spike_input = self.membrane_potential - v_th
            self.spike_output = self.surrogate_gradient(spike_input)
            
            outputs.append(self.spike_output)
          # 延迟异或任务的决策时间点 - 统一使用后半段
        decision_start = seq_len//2
        integrated_output = torch.stack(outputs[decision_start:], dim=1).sum(dim=1)
        
        # 输出层
        final_output = self.fc2(integrated_output)
        
        return final_output

# ==================== 改进的胞体异质性模型 ====================

class DelayedXOR_SH_SNN_Improved(nn.Module):
    """
    改进的胞体异质性SNN模型 (SH-SNN)
    
    设计理念：
    1. 多个专门化的神经元群体，每个群体处理不同时间尺度
    2. 胞体层面的信息整合，类似DH-SNN的分支整合
    3. 与DH-SNN在复杂度和参数量上保持可比性
    """
    
    def __init__(self, input_size=INPUT_SIZE, hidden_size=HIDDEN_SIZE, output_size=OUTPUT_SIZE, num_soma_groups=2):
        super(DelayedXOR_SH_SNN_Improved, self).__init__()
        
        print(f"🧠 创建改进的胞体异质性SH-SNN模型:")
        print(f"   输入维度: {input_size}")
        print(f"   隐藏维度: {hidden_size}")
        print(f"   输出维度: {output_size}")
        print(f"   胞体群体数: {num_soma_groups}")
        print(f"   异质性类型: 胞体群体异质性 (不同群体不同时间特性)")
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_soma_groups = num_soma_groups
        self.group_size = hidden_size // num_soma_groups
        
        # 为每个胞体群体创建独立的输入处理
        self.soma_group_layers = nn.ModuleList()
        for i in range(num_soma_groups):
            self.soma_group_layers.append(
                layer.Linear(input_size, self.group_size, bias=False)
            )
          # 每个胞体群体有独立的时间常数
        self.tau_m_groups = nn.Parameter(torch.empty(num_soma_groups, self.group_size))
        # 群体1用于长期记忆，群体2用于短期记忆 - 使用更合理的范围
        nn.init.uniform_(self.tau_m_groups[0], 1.0, 3.0)  # 群体1：中等速度（长期记忆）
        if num_soma_groups > 1:
            nn.init.uniform_(self.tau_m_groups[1], -1.0, 1.0)  # 群体2：快速（短期记忆）
        
        # 胞体整合层 - 类似DH-SNN的胞体整合功能
        self.soma_integration_tau = nn.Parameter(torch.empty(hidden_size).uniform_(0.0, 2.0))
        
        # 输出层
        self.output_layer = layer.Linear(hidden_size, output_size)
        
        # 神经元状态
        self.group_potentials = None  # 每个群体的膜电位
        self.group_spikes = None      # 每个群体的脉冲
        self.integrated_potential = None  # 胞体整合后的膜电位
        self.final_spike = None       # 最终输出脉冲
        
        print("✅ 改进的胞体异质性SH-SNN模型创建完成")
    
    def reset_states(self, batch_size):
        """重置神经元状态"""
        self.group_potentials = [
            torch.zeros(batch_size, self.group_size).to(DEVICE)
            for _ in range(self.num_soma_groups)
        ]
        self.group_spikes = [
            torch.zeros(batch_size, self.group_size).to(DEVICE)
            for _ in range(self.num_soma_groups)
        ]
        self.integrated_potential = torch.zeros(batch_size, self.hidden_size).to(DEVICE)
        self.final_spike = torch.zeros(batch_size, self.hidden_size).to(DEVICE)
    
    def surrogate_gradient(self, x):
        """使用多高斯替代函数"""
        return multi_gaussian_surrogate(x)
    
    def forward(self, x):
        """前向传播"""
        batch_size, seq_len, input_dim = x.shape
        
        # 重置状态
        self.reset_states(batch_size)
        
        outputs = []
        for t in range(seq_len):
            x_t = x[:, t, :]  # [batch, input_size]
            
            # 处理各个胞体群体
            group_outputs = []
            for i in range(self.num_soma_groups):
                # 群体输入处理
                group_input = self.soma_group_layers[i](x_t)
                
                # 群体特化的膜电位更新
                alpha_group = torch.sigmoid(self.tau_m_groups[i])
                v_th = 1.0
                
                self.group_potentials[i] = (
                    alpha_group * self.group_potentials[i] + 
                    (1 - alpha_group) * group_input - 
                    v_th * self.group_spikes[i]
                )
                
                # 群体脉冲生成
                spike_input = self.group_potentials[i] - v_th
                self.group_spikes[i] = self.surrogate_gradient(spike_input)
                
                group_outputs.append(self.group_spikes[i])
            
            # 胞体整合 - 类似DH-SNN的分支整合
            integrated_input = torch.cat(group_outputs, dim=1)  # [batch, hidden_size]
            
            # 胞体层面的整合动态
            alpha_soma = torch.sigmoid(self.soma_integration_tau)
            v_th = 1.0
            
            self.integrated_potential = (
                alpha_soma * self.integrated_potential + 
                (1 - alpha_soma) * integrated_input - 
                v_th * self.final_spike
            )
            
            # 最终脉冲生成
            final_spike_input = self.integrated_potential - v_th
            self.final_spike = self.surrogate_gradient(final_spike_input)
            
            outputs.append(self.final_spike)
        
        # 延迟异或任务需要在序列末尾做决策，使用最后1/4的时间步
        decision_start = max(seq_len - seq_len//4, seq_len//2)
        integrated_output = torch.stack(outputs[decision_start:], dim=1).sum(dim=1)
        
        # 输出层
        final_output = self.output_layer(integrated_output)
        
        return final_output

# 原始简单版本改名为Legacy
class DelayedXOR_SH_SNN_Legacy(nn.Module):
    """
    原始胞体异质性SNN模型 (保留用于对比)
    """
    
    def __init__(self, input_size=INPUT_SIZE, hidden_size=HIDDEN_SIZE, output_size=OUTPUT_SIZE):
        super(DelayedXOR_SH_SNN_Legacy, self).__init__()
        
        print(f"🧠 创建原始胞体异质性SH-SNN模型 (遗留):")
        print(f"   输入维度: {input_size}")
        print(f"   隐藏维度: {hidden_size}")
        print(f"   输出维度: {output_size}")
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
          # 普通SNN：按照原论文使用固定时间常数
        self.fc1 = layer.Linear(input_size, hidden_size)
        self.lif1 = neuron.LIFNode(tau=2.0, v_threshold=1.0, surrogate_function=surrogate.ATan(), step_mode='s')
        
        self.fc2 = layer.Linear(hidden_size, output_size)
        self.lif2 = neuron.LIFNode(tau=2.0, v_threshold=1.0, surrogate_function=surrogate.ATan(), step_mode='s')
    
    def forward(self, x):
        batch_size, seq_len, input_dim = x.shape
        functional.reset_net(self)
        
        outputs = []
        for t in range(seq_len):
            x_t = x[:, t, :]
            h1 = self.fc1(x_t)
            s1 = self.lif1(h1)
            h2 = self.fc2(s1)
            s2 = self.lif2(h2)
            outputs.append(s2)
          # 延迟异或任务需要在序列末尾做决策，使用最后1/4的时间步
        decision_start = max(seq_len - seq_len//4, seq_len//2)
        integrated_output = torch.stack(outputs[decision_start:], dim=1).sum(dim=1)
        return integrated_output

# ==================== 树突异质性模型 ====================

class DelayedXOR_DH_SNN(nn.Module):
    """
    树突异质性SNN模型 (DH-SNN)
    在树突层面引入异质性，不同分支具有不同的时间常数
    """
    
    def __init__(self, input_size=INPUT_SIZE, hidden_size=HIDDEN_SIZE, output_size=OUTPUT_SIZE, num_branches=2):
        super(DelayedXOR_DH_SNN, self).__init__()
        
        print(f"🌳 创建树突异质性DH-SNN模型:")
        print(f"   输入维度: {input_size}")
        print(f"   隐藏维度: {hidden_size}")
        print(f"   输出维度: {output_size}")
        print(f"   分支数量: {num_branches}")
        print(f"   异质性类型: 树突异质性 (不同分支不同时间常数)")
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_branches = num_branches
        
        # 分支线性层
        self.branch_layers = nn.ModuleList()
        for i in range(num_branches):
            self.branch_layers.append(
                layer.Linear(input_size, hidden_size // num_branches, bias=False)
            )        # 树突异质性：严格按照原论文初始化
        # tau_n: 树突时间常数，Large配置(2,6)
        self.tau_n = nn.Parameter(torch.empty(num_branches, hidden_size // num_branches).uniform_(2, 6))
        # tau_m: 膜电位时间常数，Medium配置(0,4)
        self.tau_m = nn.Parameter(torch.empty(hidden_size).uniform_(0, 4))
        
        # 输出层
        self.output_layer = layer.Linear(hidden_size, output_size)
        
        # 神经元状态
        self.dendritic_currents = None
        self.membrane_potential = None
        self.spike_output = None
        
        print("✅ 树突异质性DH-SNN模型创建完成")
    
    def reset_states(self, batch_size):
        """重置神经元状态"""
        self.dendritic_currents = [
            torch.zeros(batch_size, self.hidden_size // self.num_branches).to(DEVICE)
            for _ in range(self.num_branches)
        ]
        self.membrane_potential = torch.zeros(batch_size, self.hidden_size).to(DEVICE)
        self.spike_output = torch.zeros(batch_size, self.hidden_size).to(DEVICE)
    
    def surrogate_gradient(self, x):
        """使用多高斯替代函数"""
        return multi_gaussian_surrogate(x)
    
    def forward(self, x):
        """前向传播"""
        batch_size, seq_len, input_dim = x.shape
        
        # 重置状态
        self.reset_states(batch_size)
        
        outputs = []
        for t in range(seq_len):
            x_t = x[:, t, :]  # [batch, input_size]
            
            # 处理各个分支
            branch_outputs = []
            for i in range(self.num_branches):
                # 分支线性变换
                branch_input = self.branch_layers[i](x_t)
                
                # 树突时间常数更新
                beta = torch.sigmoid(self.tau_n[i])
                self.dendritic_currents[i] = (
                    beta * self.dendritic_currents[i] + 
                    (1 - beta) * branch_input
                )
                
                branch_outputs.append(self.dendritic_currents[i])
            
            # 合并分支输出
            total_current = torch.cat(branch_outputs, dim=1)  # [batch, hidden_size]
            
            # 膜电位更新
            alpha = torch.sigmoid(self.tau_m)
            v_th = 1.0
            
            self.membrane_potential = (
                alpha * self.membrane_potential + 
                (1 - alpha) * total_current - 
                v_th * self.spike_output
            )
              # 脉冲生成
            spike_input = self.membrane_potential - v_th
            self.spike_output = self.surrogate_gradient(spike_input)
            
            outputs.append(self.spike_output)
        
        # 延迟异或任务需要在序列末尾做决策，使用最后1/4的时间步
        decision_start = max(seq_len - seq_len//4, seq_len//2)
        integrated_output = torch.stack(outputs[decision_start:], dim=1).sum(dim=1)
        
        # 输出层
        final_output = self.output_layer(integrated_output)
        
        return final_output

# ==================== 训练和测试函数 ====================

def train_delayed_xor_model(model, train_loader, test_loader, model_name, num_epochs=NUM_EPOCHS):
    """训练延迟异或模型"""
    print(f"\n🚀 开始训练 {model_name}")
    print("-" * 50)
    
    model = model.to(DEVICE)
    
    # 优化器配置
    if isinstance(model, (DelayedXOR_DH_SNN, DelayedXOR_SH_SNN, DelayedXOR_SH_SNN_Improved)):
        # 异质性SNN使用分层学习率
        base_params = []
        tau_params = []
        
        for name, param in model.named_parameters():
            if 'tau_' in name:
                tau_params.append(param)
            else:
                base_params.append(param)
        
        optimizer = torch.optim.Adam([
            {'params': base_params, 'lr': LEARNING_RATE},
            {'params': tau_params, 'lr': LEARNING_RATE * 2},  # 时间常数用2倍学习率
        ])
    else:
        # 普通SNN使用标准优化器
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)
    criterion = nn.BCEWithLogitsLoss()
    
    best_test_acc = 0.0
    
    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for batch_data, batch_labels in train_loader:
            batch_data, batch_labels = batch_data.to(DEVICE), batch_labels.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(batch_data)
            
            # 调整输出和标签形状
            outputs = outputs.squeeze(-1)
            batch_labels = batch_labels.float()
            
            loss = criterion(outputs, batch_labels)
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            running_loss += loss.item()
            
            # 计算准确率
            predicted = (torch.sigmoid(outputs) > 0.5).float()
            total += batch_labels.size(0)
            correct += (predicted == batch_labels).sum().item()
        
        scheduler.step()
        
        train_acc = 100 * correct / total
        train_loss = running_loss / len(train_loader)
        
        # 测试阶段
        model.eval()
        test_correct = 0
        test_total = 0
        
        with torch.no_grad():
            for batch_data, batch_labels in test_loader:
                batch_data, batch_labels = batch_data.to(DEVICE), batch_labels.to(DEVICE)
                
                outputs = model(batch_data)
                outputs = outputs.squeeze(-1)
                batch_labels = batch_labels.float()
                
                predicted = (torch.sigmoid(outputs) > 0.5).float()
                test_total += batch_labels.size(0)
                test_correct += (predicted == batch_labels).sum().item()
        
        test_acc = 100 * test_correct / test_total
        
        if test_acc > best_test_acc:
            best_test_acc = test_acc
          # 打印进度 - 更频繁的反馈
        if epoch % 10 == 0 or epoch == num_epochs - 1:
            print(f'轮次 [{epoch+1}/{num_epochs}]: 训练准确率={train_acc:.1f}%, 测试准确率={test_acc:.1f}%, 最佳={best_test_acc:.1f}%')
    
    return {
        'best_test_acc': best_test_acc,
        'final_test_acc': test_acc
    }

# ==================== 主实验函数 ====================

def run_heterogeneity_comparison_experiment():
    """运行胞体异质性vs树突异质性对比实验"""
    
    print("=" * 80)
    print("🧠🌳 胞体异质性 vs 树突异质性延迟异或对比实验")
    print("=" * 80)
    
    # 设置随机种子
    setup_seed(42)
    
    print(f"🖥️  使用设备: {DEVICE}")
    
    try:
        # 创建数据集
        print("📊 创建延迟异或数据集...")
        datasets = create_delayed_xor_datasets(DELAY_RANGE)
        
        all_results = {}
        
        # 对每个延迟设置进行三方对比实验
        for delay in DELAY_RANGE:
            print(f"\n🔬 实验延迟={delay}步的异或任务")
            print("=" * 50)
            
            train_loader = datasets[delay]['train_loader']
            test_loader = datasets[delay]['test_loader']
            
            # 创建三种模型 - 使用改进的SH-SNN
            sh_snn_model = DelayedXOR_SH_SNN_Improved()  # 改进的胞体异质性
            dh_snn_model = DelayedXOR_DH_SNN()  # 树突异质性
            vanilla_snn_model = DelayedXOR_Vanilla_SNN()  # 普通SNN
            
            print(f"\n📊 模型参数统计:")
            print(f"   SH-SNN参数: {sum(p.numel() for p in sh_snn_model.parameters()):,}")
            print(f"   DH-SNN参数: {sum(p.numel() for p in dh_snn_model.parameters()):,}")
            print(f"   普通SNN参数: {sum(p.numel() for p in vanilla_snn_model.parameters()):,}")
            
            # 训练三种模型
            print(f"\n🚀 训练延迟{delay}步的三种模型...")
            
            # 训练改进的SH-SNN
            sh_results = train_delayed_xor_model(
                sh_snn_model, train_loader, test_loader, f"SH-SNN-Improved (延迟{delay})", NUM_EPOCHS
            )
            
            # 训练DH-SNN
            dh_results = train_delayed_xor_model(
                dh_snn_model, train_loader, test_loader, f"DH-SNN (延迟{delay})", NUM_EPOCHS
            )
            
            # 训练普通SNN
            vanilla_results = train_delayed_xor_model(
                vanilla_snn_model, train_loader, test_loader, f"普通SNN (延迟{delay})", NUM_EPOCHS
            )
            
            # 保存结果
            all_results[delay] = {
                'sh_snn': sh_results,
                'dh_snn': dh_results,
                'vanilla_snn': vanilla_results,
                'sh_vs_vanilla': sh_results['best_test_acc'] - vanilla_results['best_test_acc'],
                'dh_vs_vanilla': dh_results['best_test_acc'] - vanilla_results['best_test_acc'],
                'dh_vs_sh': dh_results['best_test_acc'] - sh_results['best_test_acc']
            }
            
            print(f"\n📈 延迟{delay}步结果:")
            print(f"   SH-SNN最佳准确率: {sh_results['best_test_acc']:.1f}%")
            print(f"   DH-SNN最佳准确率: {dh_results['best_test_acc']:.1f}%")
            print(f"   普通SNN最佳准确率: {vanilla_results['best_test_acc']:.1f}%")
            print(f"   SH-SNN vs 普通SNN: {all_results[delay]['sh_vs_vanilla']:+.1f}%")
            print(f"   DH-SNN vs 普通SNN: {all_results[delay]['dh_vs_vanilla']:+.1f}%")
            print(f"   DH-SNN vs SH-SNN: {all_results[delay]['dh_vs_sh']:+.1f}%")
        
        # 总结所有结果
        print("\n" + "=" * 80)
        print("🎯 胞体异质性 vs 树突异质性对比总结")
        print("=" * 80)
        
        print("延迟步数 | SH-SNN | DH-SNN | 普通SNN | SH提升 | DH提升 | DH>SH")
        print("-" * 70)
        
        total_sh_acc = 0
        total_dh_acc = 0
        total_vanilla_acc = 0
        
        for delay in DELAY_RANGE:
            sh_acc = all_results[delay]['sh_snn']['best_test_acc']
            dh_acc = all_results[delay]['dh_snn']['best_test_acc']
            vanilla_acc = all_results[delay]['vanilla_snn']['best_test_acc']
            sh_improvement = all_results[delay]['sh_vs_vanilla']
            dh_improvement = all_results[delay]['dh_vs_vanilla']
            dh_vs_sh = all_results[delay]['dh_vs_sh']
            
            print(f"{delay:8d} | {sh_acc:6.1f}% | {dh_acc:6.1f}% | {vanilla_acc:7.1f}% | {sh_improvement:+5.1f}% | {dh_improvement:+5.1f}% | {dh_vs_sh:+5.1f}%")
            
            total_sh_acc += sh_acc
            total_dh_acc += dh_acc
            total_vanilla_acc += vanilla_acc
        
        # 计算平均性能
        avg_sh_acc = total_sh_acc / len(DELAY_RANGE)
        avg_dh_acc = total_dh_acc / len(DELAY_RANGE)
        avg_vanilla_acc = total_vanilla_acc / len(DELAY_RANGE)
        avg_sh_improvement = avg_sh_acc - avg_vanilla_acc
        avg_dh_improvement = avg_dh_acc - avg_vanilla_acc
        avg_dh_vs_sh = avg_dh_acc - avg_sh_acc
        
        print("-" * 70)
        print(f"平均     | {avg_sh_acc:6.1f}% | {avg_dh_acc:6.1f}% | {avg_vanilla_acc:7.1f}% | {avg_sh_improvement:+5.1f}% | {avg_dh_improvement:+5.1f}% | {avg_dh_vs_sh:+5.1f}%")
        
        # 保存结果
        results_path = Path("results/heterogeneity_comparison_results.json")
        results_path.parent.mkdir(parents=True, exist_ok=True)
        
        final_results = {
            'experiment_info': {
                'name': '胞体异质性vs树突异质性对比实验',
                'framework': 'SpikingJelly',
                'date': time.strftime('%Y-%m-%d %H:%M:%S'),
                'delay_range': DELAY_RANGE,
                'seq_length': SEQ_LENGTH,
                'num_epochs': NUM_EPOCHS,
                'device': str(DEVICE)
            },
            'results_by_delay': all_results,
            'summary': {
                'avg_sh_snn_acc': avg_sh_acc,
                'avg_dh_snn_acc': avg_dh_acc,
                'avg_vanilla_snn_acc': avg_vanilla_acc,
                'avg_sh_improvement': avg_sh_improvement,
                'avg_dh_improvement': avg_dh_improvement,
                'avg_dh_vs_sh': avg_dh_vs_sh
            }
        }
        
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(final_results, f, indent=2, ensure_ascii=False)
        
        print(f"\n💾 结果已保存到: {results_path}")
        
        # 分析结果
        print(f"\n📈 实验结论:")
        print(f"🧠 胞体异质性(SH-SNN)平均提升: {avg_sh_improvement:+.1f}%")
        print(f"🌳 树突异质性(DH-SNN)平均提升: {avg_dh_improvement:+.1f}%")
        print(f"🥇 DH-SNN vs SH-SNN优势: {avg_dh_vs_sh:+.1f}%")
        
        if avg_dh_vs_sh > 2:
            print("🎉 树突异质性明显优于胞体异质性！")
        elif avg_dh_vs_sh > 0:
            print("✅ 树突异质性略优于胞体异质性")
        elif avg_dh_vs_sh > -2:
            print("🤝 两种异质性机制性能相当")
        else:
            print("💡 胞体异质性在某些条件下更优")
        
        return final_results
        
    except Exception as e:
        print(f"\n❌ 实验失败: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    results = run_heterogeneity_comparison_experiment()
    if results:
        print(f"\n🏁 胞体异质性vs树突异质性对比实验成功完成!")
    else:
        print(f"\n❌ 对比实验失败")