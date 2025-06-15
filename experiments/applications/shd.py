#!/usr/bin/env python3
"""
DH-SNN SHD（脉冲海德堡数字）实验
====================================

基于SpikingJelly框架的DH-SNN vs 普通SNN对比实验
使用SHD数据集进行脉冲数字识别任务

"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import tables
import time
import json
import math
from pathlib import Path

# 添加项目路径
current_dir = Path(__file__).parent.parent.parent
sys.path.append(str(current_dir))

# SpikingJelly导入
from spikingjelly.activation_based import neuron, functional, layer, surrogate

from dh_snn.utils import setup_seed

print("🚀 DH-SNN SHD脉冲数字识别实验")
print("="*60)

# 实验配置
CONFIG = {
    'learning_rate': 1e-2,
    'batch_size': 100,
    'epochs': 100,
    'hidden_size': 64,
    'v_threshold': 1.0,
    'dt': 1,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu'
}

# 时间因子配置 - 按照论文Table S3
TIMING_CONFIGS = {
    'Small': {'tau_m': (-4.0, 0.0), 'tau_n': (-4.0, 0.0)},   # β̂,α̂ ~ U(-4,0)
    'Medium': {'tau_m': (0.0, 4.0), 'tau_n': (0.0, 4.0)},    # β̂,α̂ ~ U(0,4)
    'Large': {'tau_m': (2.0, 6.0), 'tau_n': (2.0, 6.0)}      # β̂,α̂ ~ U(2,6)
}

# ==================== 多高斯替代函数 ====================

class MultiGaussianSurrogate(torch.autograd.Function):
    """
    多高斯替代函数
    完全按照原论文实现的MultiGaussian替代函数
    """

    @staticmethod
    def forward(ctx, input):
        """前向传播：阶跃函数"""
        ctx.save_for_backward(input)
        return input.gt(0).float()

    @staticmethod
    def backward(ctx, grad_output):
        """反向传播：多高斯近似梯度"""
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()

        # 原论文参数: lens=0.5, scale=6.0, height=0.15, gamma=0.5
        lens = 0.5
        scale = 6.0
        height = 0.15
        gamma = 0.5

        def gaussian(x, mu=0., sigma=0.5):
            """高斯函数"""
            return torch.exp(-((x - mu) ** 2) / (2 * sigma ** 2)) / torch.sqrt(2 * torch.tensor(math.pi)) / sigma

        # MultiGaussian - 完全按照原论文公式
        temp = gaussian(input, mu=0., sigma=lens) * (1. + height) \
             - gaussian(input, mu=lens, sigma=scale * lens) * height \
             - gaussian(input, mu=-lens, sigma=scale * lens) * height

        return grad_input * temp.float() * gamma

multi_gaussian_surrogate = MultiGaussianSurrogate.apply

# ==================== 神经元模型 ====================

class DH_LIFNode(nn.Module):
    """
    树突异质性LIF神经元
    等价于原论文的LIF神经元实现
    """

    def __init__(self, size, tau_m_range=(0.0, 4.0), v_threshold=1.0, device='cpu'):
        """
        初始化DH-LIF神经元
        
        参数:
            size: 神经元数量
            tau_m_range: 膜电位时间常数初始化范围
            v_threshold: 脉冲阈值
            device: 计算设备
        """
        super().__init__()
        self.size = size
        self.v_threshold = v_threshold
        self.device = device

        # 膜电位时间常数参数
        self.tau_m = nn.Parameter(torch.empty(size))
        nn.init.uniform_(self.tau_m, tau_m_range[0], tau_m_range[1])

        # 神经元状态缓存
        self.register_buffer('mem', torch.zeros(1, size))
        self.register_buffer('spike', torch.zeros(1, size))

    def set_neuron_state(self, batch_size):
        """
        重置神经元状态
        
        参数:
            batch_size: 批次大小
        """
        self.mem = torch.rand(batch_size, self.size).to(self.device)
        self.spike = torch.rand(batch_size, self.size).to(self.device)

    def forward(self, input_current):
        """
        前向传播 - 完全按照原论文mem_update_pra实现
        
        参数:
            input_current: 输入电流
            
        返回:
            mem: 膜电位
            spike: 输出脉冲
        """
        # 原论文: alpha = torch.sigmoid(tau_m)
        alpha = torch.sigmoid(self.tau_m)

        # 原论文: mem = mem * alpha + (1 - alpha) * R_m * inputs - v_th * spike
        # R_m = 1 在原论文中
        self.mem = self.mem * alpha + (1 - alpha) * input_current - self.v_threshold * self.spike

        # 原论文: inputs_ = mem - v_th
        inputs_ = self.mem - self.v_threshold

        # 原论文: spike = act_fun_adp(inputs_)
        self.spike = multi_gaussian_surrogate(inputs_)

        return self.mem, self.spike

class ReadoutIntegrator(nn.Module):
    """
    读出积分器
    等价于原论文的readout_integrator_test
    """

    def __init__(self, input_dim, output_dim, tau_m_range=(0.0, 4.0), device='cpu'):
        """
        初始化读出积分器
        
        参数:
            input_dim: 输入维度
            output_dim: 输出维度  
            tau_m_range: 时间常数初始化范围
            device: 计算设备
        """
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.device = device

        # 线性层
        self.dense = nn.Linear(input_dim, output_dim)

        # 时间常数
        self.tau_m = nn.Parameter(torch.empty(output_dim))
        nn.init.uniform_(self.tau_m, tau_m_range[0], tau_m_range[1])

        # 膜电位状态
        self.register_buffer('mem', torch.zeros(1, output_dim))

    def set_neuron_state(self, batch_size):
        """重置神经元状态"""
        self.mem = torch.rand(batch_size, self.output_dim).to(self.device)

    def forward(self, input_spike):
        """
        前向传播 - 完全按照原论文output_Neuron_pra实现
        
        参数:
            input_spike: 输入脉冲
            
        返回:
            mem: 膜电位（不产生脉冲）
        """
        # 突触输入
        d_input = self.dense(input_spike.float())

        # 原论文: alpha = torch.sigmoid(tau_m)
        alpha = torch.sigmoid(self.tau_m)

        # 原论文: mem = mem * alpha + (1-alpha) * inputs
        self.mem = self.mem * alpha + (1 - alpha) * d_input

        return self.mem

class DH_DendriticLayer(nn.Module):
    """
    树突异质性层
    等价于原论文的spike_dense_test_denri_wotanh_R
    """

    def __init__(self, input_dim, output_dim, tau_m_range=(0.0, 4.0), tau_n_range=(2.0, 6.0),
                 num_branches=4, v_threshold=1.0, device='cpu'):
        """
        初始化树突异质性层
        
        参数:
            input_dim: 输入维度
            output_dim: 输出维度
            tau_m_range: 膜电位时间常数范围
            tau_n_range: 树突时间常数范围
            num_branches: 树突分支数量
            v_threshold: 脉冲阈值
            device: 计算设备
        """
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_branches = num_branches
        self.v_threshold = v_threshold
        self.device = device

        # 连接层 - 按照原论文实现
        self.pad = ((input_dim) // num_branches * num_branches + num_branches - input_dim) % num_branches
        self.dense = nn.Linear(input_dim + self.pad, output_dim * num_branches)

        # 时间常数参数
        self.tau_m = nn.Parameter(torch.empty(output_dim))
        self.tau_n = nn.Parameter(torch.empty(output_dim, num_branches))

        # 初始化时间常数
        nn.init.uniform_(self.tau_m, tau_m_range[0], tau_m_range[1])
        nn.init.uniform_(self.tau_n, tau_n_range[0], tau_n_range[1])

        # 神经元状态
        self.register_buffer('mem', torch.zeros(1, output_dim))
        self.register_buffer('spike', torch.zeros(1, output_dim))
        self.register_buffer('d_input', torch.zeros(1, output_dim, num_branches))

        # 创建连接掩码
        self.create_mask()

    def create_mask(self):
        """
        创建连接掩码 - 完全按照原论文实现
        实现稀疏的树突连接模式
        """
        input_size = self.input_dim + self.pad
        self.mask = torch.zeros(self.output_dim * self.num_branches, input_size).to(self.device)
        
        for i in range(self.output_dim):
            for j in range(self.num_branches):
                start_idx = j * input_size // self.num_branches
                end_idx = (j + 1) * input_size // self.num_branches
                self.mask[i * self.num_branches + j, start_idx:end_idx] = 1

    def apply_mask(self):
        """应用连接掩码到权重"""
        self.dense.weight.data = self.dense.weight.data * self.mask

    def set_neuron_state(self, batch_size):
        """重置神经元状态"""
        self.mem = torch.rand(batch_size, self.output_dim).to(self.device)
        self.spike = torch.rand(batch_size, self.output_dim).to(self.device)
        self.d_input = torch.zeros(batch_size, self.output_dim, self.num_branches).to(self.device)

    def forward(self, input_spike):
        """
        前向传播 - 完全按照原论文实现
        
        参数:
            input_spike: 输入脉冲
            
        返回:
            mem: 膜电位
            spike: 输出脉冲
        """
        # 树突时间常数
        beta = torch.sigmoid(self.tau_n)

        # 输入填充
        padding = torch.zeros(input_spike.size(0), self.pad).to(self.device)
        k_input = torch.cat((input_spike.float(), padding), 1)

        # 更新树突电流
        dense_output = self.dense(k_input).reshape(-1, self.output_dim, self.num_branches)
        self.d_input = beta * self.d_input + (1 - beta) * dense_output

        # 总输入电流
        l_input = self.d_input.sum(dim=2, keepdim=False)

        # 膜电位更新 - 按照原论文mem_update_pra
        alpha = torch.sigmoid(self.tau_m)
        self.mem = self.mem * alpha + (1 - alpha) * l_input - self.v_threshold * self.spike

        # 脉冲生成
        inputs_ = self.mem - self.v_threshold
        self.spike = multi_gaussian_surrogate(inputs_)

        return self.mem, self.spike

# ==================== 网络模型 ====================

class VanillaSFNN(nn.Module):
    """
    普通脉冲前馈神经网络
    等价于原论文的spike_dense_test_origin
    """

    def __init__(self, config, tau_m_range=(0.0, 4.0)):
        """
        初始化普通SFNN
        
        参数:
            config: 实验配置
            tau_m_range: 膜电位时间常数范围
        """
        super().__init__()
        self.config = config
        self.device = config['device']

        print(f"🏗️  创建普通SFNN模型:")
        print(f"   隐藏层大小: {config['hidden_size']}")
        print(f"   tau_m范围: {tau_m_range}")

        # 线性层
        self.dense = nn.Linear(700, config['hidden_size'])

        # LIF神经元层
        self.lif_layer = DH_LIFNode(
            config['hidden_size'],
            tau_m_range,
            config['v_threshold'],
            self.device
        )

        # 读出层
        self.readout = ReadoutIntegrator(
            config['hidden_size'],
            20,  # SHD有20个类别
            tau_m_range,
            self.device
        )

        # 初始化权重
        torch.nn.init.xavier_normal_(self.readout.dense.weight)
        torch.nn.init.constant_(self.readout.dense.bias, 0)
        
        print("✅ 普通SFNN模型创建完成")

    def forward(self, input_data):
        """
        前向传播 - 完全按照原论文Dense_test_1layer实现
        
        参数:
            input_data: 输入脉冲序列，形状为[批次, 时间步, 特征]
            
        返回:
            output: 累积的softmax输出
        """
        batch_size, seq_length, input_dim = input_data.shape

        # 设置神经元状态
        self.lif_layer.set_neuron_state(batch_size)
        self.readout.set_neuron_state(batch_size)

        output = 0
        for i in range(seq_length):
            input_x = input_data[:, i, :].reshape(batch_size, input_dim)

            # 线性变换
            d_input = self.dense(input_x.float())

            # LIF层
            mem_layer1, spike_layer1 = self.lif_layer.forward(d_input)

            # 读出层
            mem_layer2 = self.readout.forward(spike_layer1)

            # 累积输出 - 按照原论文，跳过前10个时间步
            if i > 10:
                output += F.softmax(mem_layer2, dim=1)

        return output

class DH_SFNN(nn.Module):
    """
    树突异质性脉冲前馈神经网络
    等价于原论文的DH-SFNN实现
    """

    def __init__(self, config, tau_m_range=(0.0, 4.0), tau_n_range=(2.0, 6.0)):
        """
        初始化DH-SFNN
        
        参数:
            config: 实验配置
            tau_m_range: 膜电位时间常数范围
            tau_n_range: 树突时间常数范围
        """
        super().__init__()
        self.config = config
        self.device = config['device']

        print(f"🏗️  创建DH-SFNN模型:")
        print(f"   隐藏层大小: {config['hidden_size']}")
        print(f"   tau_m范围: {tau_m_range}")
        print(f"   tau_n范围: {tau_n_range}")

        # 树突异质性层
        self.dh_layer = DH_DendriticLayer(
            700,  # SHD输入维度
            config['hidden_size'],
            tau_m_range,
            tau_n_range,
            4,  # 树突分支数量
            config['v_threshold'],
            self.device
        )

        # 读出层
        self.readout = ReadoutIntegrator(
            config['hidden_size'],
            20,  # SHD有20个类别
            tau_m_range,
            self.device
        )

        # 初始化权重
        torch.nn.init.xavier_normal_(self.readout.dense.weight)
        torch.nn.init.constant_(self.readout.dense.bias, 0)
        
        print("✅ DH-SFNN模型创建完成")

    def forward(self, input_data):
        """
        前向传播
        
        参数:
            input_data: 输入脉冲序列，形状为[批次, 时间步, 特征]
            
        返回:
            output: 累积的softmax输出
        """
        batch_size, seq_length, input_dim = input_data.shape

        # 设置神经元状态
        self.dh_layer.set_neuron_state(batch_size)
        self.readout.set_neuron_state(batch_size)

        output = 0
        for i in range(seq_length):
            input_x = input_data[:, i, :].reshape(batch_size, input_dim)

            # 应用连接掩码
            self.dh_layer.apply_mask()

            # 树突异质性层
            mem_layer1, spike_layer1 = self.dh_layer.forward(input_x)

            # 读出层
            mem_layer2 = self.readout.forward(spike_layer1)

            # 累积输出 - 跳过前10个时间步
            if i > 10:
                output += F.softmax(mem_layer2, dim=1)

        return output

# ==================== 数据处理 ====================

def convert_to_spike_tensor(times, units, dt=1e-3, max_time=1.0):
    """
    将脉冲事件转换为密集张量
    
    参数:
        times: 脉冲时间数组
        units: 神经元单元数组
        dt: 时间步长
        max_time: 最大时间
        
    返回:
        tensor: 脉冲张量，形状为[时间步, 神经元数]
    """
    num_steps = int(max_time / dt)
    tensor = torch.zeros(num_steps, 700)
    
    # 将时间转换为时间步索引
    time_indices = (times / dt).astype(int)
    
    # 过滤有效的脉冲事件
    valid_mask = (time_indices < num_steps) & (units >= 1) & (units <= 700)
    
    if np.any(valid_mask):
        valid_times = time_indices[valid_mask]
        valid_units = units[valid_mask] - 1  # 转换为0索引
        
        # 设置脉冲
        tensor[valid_times, valid_units] = 1.0
    
    return tensor

def load_mock_data(num_train=2000, num_test=500):
    """
    创建模拟SHD数据用于测试
    
    参数:
        num_train: 训练样本数
        num_test: 测试样本数
        
    返回:
        train_data, train_labels, test_data, test_labels: 训练和测试数据
    """
    print(f"🎲 创建模拟SHD数据: 训练{num_train}, 测试{num_test}")
    
    # 生成训练数据
    train_data = torch.zeros(num_train, 1000, 700)
    train_labels = torch.randint(0, 20, (num_train,))
    
    # 生成测试数据
    test_data = torch.zeros(num_test, 1000, 700)
    test_labels = torch.randint(0, 20, (num_test,))
    
    # 为每个样本添加类别相关的脉冲模式
    for i in range(num_train):
        label = train_labels[i].item()
        # 为不同数字创建不同的脉冲模式
        num_spikes = 200 + label * 10  # 不同数字有不同的脉冲密度
        spike_times = torch.randint(0, 1000, (num_spikes,))
        spike_neurons = torch.randint(label * 35, (label + 1) * 35, (num_spikes,))  # 特定神经元区域
        
        for t, n in zip(spike_times, spike_neurons):
            if n < 700:
                train_data[i, t, n] = 1.0
    
    for i in range(num_test):
        label = test_labels[i].item()
        num_spikes = 200 + label * 10
        spike_times = torch.randint(0, 1000, (num_spikes,))
        spike_neurons = torch.randint(label * 35, (label + 1) * 35, (num_spikes,))
        
        for t, n in zip(spike_times, spike_neurons):
            if n < 700:
                test_data[i, t, n] = 1.0
    
    print(f"✅ 模拟数据生成完成")
    return train_data, train_labels, test_data, test_labels

# ==================== 训练函数 ====================

def train_shd_model(model, train_data, train_labels, test_data, test_labels, config, model_name):
    """
    训练SHD模型 - 完全按照原论文方式
    
    参数:
        model: 待训练的模型
        train_data: 训练数据
        train_labels: 训练标签
        test_data: 测试数据
        test_labels: 测试标签
        config: 实验配置
        model_name: 模型名称
        
    返回:
        best_acc: 最佳测试准确率
    """
    print(f"🏋️  训练{model_name}")

    device = config['device']
    model = model.to(device)

    # 创建数据加载器
    train_dataset = torch.utils.data.TensorDataset(train_data, train_labels)
    test_dataset = torch.utils.data.TensorDataset(test_data, test_labels)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=config['batch_size'], shuffle=True
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=config['batch_size'], shuffle=False
    )

    criterion = nn.CrossEntropyLoss()

    # 分组优化器 - 按照原论文，时间常数使用2倍学习率
    base_params = []
    tau_m_params = []
    tau_n_params = []

    for name, param in model.named_parameters():
        if 'tau_m' in name:
            tau_m_params.append(param)
        elif 'tau_n' in name:
            tau_n_params.append(param)
        else:
            base_params.append(param)

    optimizer = torch.optim.Adam([
        {'params': base_params, 'lr': config['learning_rate']},
        {'params': tau_m_params, 'lr': config['learning_rate'] * 2},
        {'params': tau_n_params, 'lr': config['learning_rate'] * 2},
    ], lr=config['learning_rate'])

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

    best_acc = 0.0

    for epoch in range(config['epochs']):
        # 训练阶段
        model.train()
        train_acc = 0
        sum_sample = 0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            # 应用掩码 (对DH-SFNN)
            if hasattr(model, 'dh_layer'):
                model.dh_layer.apply_mask()

            optimizer.zero_grad()
            predictions = model(images)
            loss = criterion(predictions, labels)
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()

            # 再次应用掩码
            if hasattr(model, 'dh_layer'):
                model.dh_layer.apply_mask()

            _, predicted = torch.max(predictions.data, 1)
            train_acc += (predicted.cpu() == labels.cpu()).sum().item()
            sum_sample += labels.size(0)

        scheduler.step()
        train_acc = train_acc / sum_sample * 100

        # 测试阶段
        model.eval()
        test_acc = 0
        test_sum_sample = 0

        with torch.no_grad():
            for images, labels in test_loader:
                if hasattr(model, 'dh_layer'):
                    model.dh_layer.apply_mask()

                images = images.to(device)
                labels = labels.to(device)

                predictions = model(images)
                _, predicted = torch.max(predictions.data, 1)

                test_acc += (predicted.cpu() == labels.cpu()).sum().item()
                test_sum_sample += labels.size(0)

        test_acc = test_acc / test_sum_sample * 100

        if test_acc > best_acc:
            best_acc = test_acc

        # 打印进度
        if epoch % 20 == 0 or epoch == config['epochs'] - 1:
            print(f"   轮次 {epoch+1:2d}: 训练准确率={train_acc:5.1f}%, 测试准确率={test_acc:5.1f}%, 最佳={best_acc:5.1f}%")

    return best_acc

# ==================== 主实验函数 ====================

def run_shd_experiment():
    """运行SHD实验"""
    
    print("=" * 80)
    print("🔢 DH-SNN SHD脉冲数字识别实验")
    print("=" * 80)
    
    # 设置随机种子
    setup_seed(42)
    
    print(f"🖥️  使用设备: {CONFIG['device']}")

    try:
        # 加载数据（这里使用模拟数据，实际项目中可以加载真实SHD数据）
        train_data, train_labels, test_data, test_labels = load_mock_data(2000, 500)

        results = {}

        # 测试不同的时间因子配置
        for timing_name, timing_config in TIMING_CONFIGS.items():
            print(f"\n📊 测试 {timing_name} 时间因子配置")
            print(f"   tau_m: {timing_config['tau_m']}, tau_n: {timing_config['tau_n']}")

            # 训练普通SFNN
            print(f"\n🔬 训练普通SFNN ({timing_name})")
            vanilla_model = VanillaSFNN(CONFIG, tau_m_range=timing_config['tau_m'])
            vanilla_acc = train_shd_model(
                vanilla_model, train_data, train_labels, test_data, test_labels,
                CONFIG, f"普通SFNN ({timing_name})"
            )

            # 训练DH-SFNN
            print(f"\n🔬 训练DH-SFNN ({timing_name})")
            dh_model = DH_SFNN(
                CONFIG,
                tau_m_range=timing_config['tau_m'],
                tau_n_range=timing_config['tau_n']
            )
            dh_acc = train_shd_model(
                dh_model, train_data, train_labels, test_data, test_labels,
                CONFIG, f"DH-SFNN ({timing_name})"
            )

            # 保存结果
            results[timing_name] = {
                'vanilla': vanilla_acc,
                'dh_snn': dh_acc,
                'improvement': dh_acc - vanilla_acc
            }

            print(f"\n📈 {timing_name} 配置结果:")
            print(f"   普通SFNN: {vanilla_acc:.1f}%")
            print(f"   DH-SFNN:  {dh_acc:.1f}%")
            print(f"   性能提升: {dh_acc - vanilla_acc:+.1f} 个百分点")

        # 总结所有结果
        print(f"\n🎉 SHD实验完成!")
        print("=" * 60)
        print("📊 三种时间因子配置性能对比:")
        print("=" * 60)
        
        for timing_name in TIMING_CONFIGS.keys():
            vanilla_acc = results[timing_name]['vanilla']
            dh_acc = results[timing_name]['dh_snn']
            improvement = results[timing_name]['improvement']
            print(f"{timing_name:6s}: 普通SFNN {vanilla_acc:5.1f}% → DH-SFNN {dh_acc:5.1f}% (提升{improvement:+5.1f}%)")

        # 保存结果
        results_path = Path("results/shd_experiment_results.json")
        results_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(results_path, 'w', encoding='utf-8') as f:
            json_results = {
                'experiment_info': {
                    'name': 'SHD脉冲数字识别实验',
                    'framework': 'SpikingJelly + DH-SNN',
                    'date': time.strftime('%Y-%m-%d %H:%M:%S'),
                    'device': str(CONFIG['device'])
                },
                'timing_configs': TIMING_CONFIGS,
                'results': results
            }
            json.dump(json_results, f, indent=2, ensure_ascii=False)
        
        print(f"\n💾 结果已保存到: {results_path}")

        # 与论文结果对比
        print(f"\n📈 与论文结果对比:")
        print(f"论文普通SNN:   ~74%")
        print(f"论文DH-SNN:    ~80%")
        
        best_config = max(results.keys(), key=lambda k: results[k]['dh_snn'])
        best_improvement = results[best_config]['improvement']
        
        if best_improvement > 5:
            print("🎉 DH-SNN显著优于普通SNN - 符合预期!")
        elif best_improvement > 0:
            print("✅ DH-SNN优于普通SNN")
        else:
            print("⚠️  结果需要进一步分析")

        return results

    except Exception as e:
        print(f"\n❌ 实验失败: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    results = run_shd_experiment()
    if results:
        print(f"\n🏁 SHD实验成功完成!")
    else:
        print(f"\n❌ SHD实验失败")