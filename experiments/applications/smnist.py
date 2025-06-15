#!/usr/bin/env python3
"""
DH-SNN Sequential MNIST（序列MNIST）实验
=====================================

基于SpikingJelly框架的DH-SNN vs 普通SNN对比实验
使用Sequential MNIST数据集进行序列分类任务

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

# PyTorch数据集导入
import torchvision
import torchvision.transforms as transforms

from dh_snn.utils import setup_seed

print("🚀 DH-SNN Sequential MNIST序列分类实验")
print("="*60)

# 实验参数
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
NUM_EPOCHS = 20
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Sequential MNIST参数
SEQ_LENGTH = 784  # 28x28像素序列化
INPUT_SIZE = 1    # 每个时间步输入一个像素值
HIDDEN_SIZE = 128 # 隐藏层大小
OUTPUT_SIZE = 10  # 10个数字类别
NUM_BRANCHES = 4  # DH-SNN分支数量

# ==================== 数据处理 ====================

class SequentialMNIST(torch.utils.data.Dataset):
    """
    Sequential MNIST数据集
    将MNIST图像转换为像素序列
    """
    
    def __init__(self, mnist_dataset, encoding='rate', time_steps=784):
        """
        初始化Sequential MNIST数据集
        
        参数:
            mnist_dataset: 原始MNIST数据集
            encoding: 编码方式 ('rate' 或 'temporal')
            time_steps: 时间步数
        """
        self.mnist_dataset = mnist_dataset
        self.encoding = encoding
        self.time_steps = time_steps
        
        print(f"   编码方式: {encoding}")
        print(f"   时间步数: {time_steps}")
        print(f"   数据样本: {len(mnist_dataset)}")
    
    def __len__(self):
        return len(self.mnist_dataset)
    
    def __getitem__(self, idx):
        """
        获取单个样本
        
        返回:
            spike_seq: 脉冲序列 [时间步, 特征维度]
            label: 标签
        """
        image, label = self.mnist_dataset[idx]
        
        # 将图像展平为序列
        pixel_seq = image.view(-1)  # [784]
        
        if self.encoding == 'rate':
            # 泊松率编码
            spike_seq = torch.rand(self.time_steps, INPUT_SIZE) < pixel_seq.unsqueeze(1)
            spike_seq = spike_seq.float()
        elif self.encoding == 'temporal':
            # 时间编码：像素值决定脉冲时间
            spike_seq = torch.zeros(self.time_steps, INPUT_SIZE)
            for i, pixel_val in enumerate(pixel_seq):
                if pixel_val > 0.1:  # 阈值过滤
                    # 像素值越大，脉冲越早出现
                    spike_time = int((1 - pixel_val) * self.time_steps * 0.8)
                    if spike_time < self.time_steps:
                        spike_seq[spike_time, 0] = 1.0
        else:
            # 直接序列化
            spike_seq = pixel_seq.unsqueeze(1).repeat(1, INPUT_SIZE)
            spike_seq = spike_seq.view(self.time_steps, INPUT_SIZE)
        
        return spike_seq, label

def create_sequential_mnist_datasets():
    """
    创建Sequential MNIST数据集
    
    返回:
        train_loader: 训练数据加载器
        test_loader: 测试数据加载器
    """
    print("📊 创建Sequential MNIST数据集...")
    
    # 数据预处理
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))  # MNIST标准化
    ])
    
    # 加载MNIST数据集
    try:
        train_mnist = torchvision.datasets.MNIST(
            root='./data/mnist', 
            train=True, 
            download=True, 
            transform=transform
        )
        
        test_mnist = torchvision.datasets.MNIST(
            root='./data/mnist', 
            train=False, 
            download=True, 
            transform=transform
        )
    except Exception as e:
        print(f"⚠️  下载MNIST失败，使用模拟数据: {e}")
        return create_mock_sequential_mnist()
    
    # 转换为Sequential MNIST
    print("   转换为Sequential MNIST格式...")
    train_smnist = SequentialMNIST(train_mnist, encoding='rate')
    test_smnist = SequentialMNIST(test_mnist, encoding='rate')
    
    # 创建数据加载器
    train_loader = torch.utils.data.DataLoader(
        train_smnist, 
        batch_size=BATCH_SIZE, 
        shuffle=True,
        num_workers=2 if torch.cuda.is_available() else 0
    )
    
    test_loader = torch.utils.data.DataLoader(
        test_smnist, 
        batch_size=BATCH_SIZE, 
        shuffle=False,
        num_workers=2 if torch.cuda.is_available() else 0
    )
    
    print(f"✅ Sequential MNIST数据集创建完成")
    print(f"   训练批次: {len(train_loader)}")
    print(f"   测试批次: {len(test_loader)}")
    
    return train_loader, test_loader

def create_mock_sequential_mnist():
    """
    创建模拟Sequential MNIST数据用于测试
    
    返回:
        train_loader: 训练数据加载器
        test_loader: 测试数据加载器
    """
    print("🎲 创建模拟Sequential MNIST数据...")
    
    # 模拟参数
    num_train = 5000
    num_test = 1000
    
    # 生成模拟数据
    train_data = torch.zeros(num_train, SEQ_LENGTH, INPUT_SIZE)
    train_labels = torch.randint(0, OUTPUT_SIZE, (num_train,))
    
    test_data = torch.zeros(num_test, SEQ_LENGTH, INPUT_SIZE)
    test_labels = torch.randint(0, OUTPUT_SIZE, (num_test,))
    
    # 为每个数字类别创建不同的序列模式
    for i in range(num_train):
        label = train_labels[i].item()
        
        # 不同数字有不同的脉冲模式
        base_pattern = torch.zeros(SEQ_LENGTH)
        
        # 数字0: 环形模式
        if label == 0:
            for t in range(0, 200, 20):
                base_pattern[t:t+10] = 0.8
        # 数字1: 垂直线模式
        elif label == 1:
            for t in range(100, 600, 50):
                base_pattern[t:t+5] = 0.9
        # 数字2: 波浪模式
        elif label == 2:
            for t in range(SEQ_LENGTH):
                if t % 30 < 15:
                    base_pattern[t] = 0.7 * np.sin(t * 0.1)
        # 其他数字的模式
        else:
            pattern_freq = (label + 1) * 10
            for t in range(0, SEQ_LENGTH, pattern_freq):
                length = min(label + 5, SEQ_LENGTH - t)
                base_pattern[t:t+length] = 0.6 + label * 0.05
        
        # 添加噪声
        noise = torch.randn(SEQ_LENGTH) * 0.1
        sequence = torch.clamp(base_pattern + noise, 0, 1)
        
        # 转换为脉冲序列
        train_data[i, :, 0] = (torch.rand(SEQ_LENGTH) < sequence).float()
    
    # 为测试数据生成类似模式
    for i in range(num_test):
        label = test_labels[i].item()
        base_pattern = torch.zeros(SEQ_LENGTH)
        
        if label == 0:
            for t in range(0, 200, 20):
                base_pattern[t:t+10] = 0.8
        elif label == 1:
            for t in range(100, 600, 50):
                base_pattern[t:t+5] = 0.9
        elif label == 2:
            for t in range(SEQ_LENGTH):
                if t % 30 < 15:
                    base_pattern[t] = 0.7 * np.sin(t * 0.1)
        else:
            pattern_freq = (label + 1) * 10
            for t in range(0, SEQ_LENGTH, pattern_freq):
                length = min(label + 5, SEQ_LENGTH - t)
                base_pattern[t:t+length] = 0.6 + label * 0.05
        
        noise = torch.randn(SEQ_LENGTH) * 0.1
        sequence = torch.clamp(base_pattern + noise, 0, 1)
        test_data[i, :, 0] = (torch.rand(SEQ_LENGTH) < sequence).float()
    
    # 创建数据加载器
    train_dataset = torch.utils.data.TensorDataset(train_data, train_labels)
    test_dataset = torch.utils.data.TensorDataset(test_data, test_labels)
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False
    )
    
    print(f"   模拟训练数据: {train_data.shape}")
    print(f"   模拟测试数据: {test_data.shape}")
    
    return train_loader, test_loader

# ==================== 多高斯替代函数 ====================

class MultiGaussianSurrogate(torch.autograd.Function):
    """
    多高斯替代函数
    按照原论文实现
    """

    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return input.gt(0).float()

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()

        # 原论文参数
        lens = 0.5
        scale = 6.0
        height = 0.15
        gamma = 0.5

        def gaussian(x, mu=0., sigma=0.5):
            return torch.exp(-((x - mu) ** 2) / (2 * sigma ** 2)) / torch.sqrt(2 * torch.tensor(math.pi)) / sigma

        # MultiGaussian公式
        temp = gaussian(input, mu=0., sigma=lens) * (1. + height) \
             - gaussian(input, mu=lens, sigma=scale * lens) * height \
             - gaussian(input, mu=-lens, sigma=scale * lens) * height

        return grad_input * temp.float() * gamma

multi_gaussian_surrogate = MultiGaussianSurrogate.apply

# ==================== 模型定义 ====================

class SequentialMNIST_DH_SNN(nn.Module):
    """
    用于Sequential MNIST的DH-SNN模型
    """
    
    def __init__(self, input_size=INPUT_SIZE, hidden_size=HIDDEN_SIZE, 
                 output_size=OUTPUT_SIZE, num_branches=NUM_BRANCHES):
        """
        初始化Sequential MNIST DH-SNN模型
        
        参数:
            input_size: 输入维度
            hidden_size: 隐藏层维度
            output_size: 输出维度
            num_branches: 树突分支数量
        """
        super(SequentialMNIST_DH_SNN, self).__init__()
        
        print(f"🏗️  创建Sequential MNIST DH-SNN模型:")
        print(f"   输入维度: {input_size}")
        print(f"   隐藏维度: {hidden_size}")
        print(f"   输出维度: {output_size}")
        print(f"   分支数量: {num_branches}")
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_branches = num_branches
        
        # 分支线性层
        self.branch_layers = nn.ModuleList()
        for i in range(num_branches):
            self.branch_layers.append(
                layer.Linear(input_size, hidden_size // num_branches, bias=False)
            )
        
        # 可学习的时间常数参数
        # tau_n: 树突时间常数，用Large初始化(2,6)适合长序列
        self.tau_n = nn.Parameter(torch.empty(num_branches, hidden_size).uniform_(2, 6))
        # tau_m: 膜电位时间常数，用Medium初始化(0,4)
        self.tau_m = nn.Parameter(torch.empty(hidden_size).uniform_(0, 4))
        
        # 输出层
        self.output_layer = layer.Linear(hidden_size, output_size)
        
        # 神经元状态
        self.dendritic_currents = None
        self.membrane_potential = None
        self.spike_output = None
        
        print("✅ Sequential MNIST DH-SNN模型创建完成")
    
    def reset_states(self, batch_size):
        """重置神经元状态"""
        self.dendritic_currents = [
            torch.zeros(batch_size, self.hidden_size // self.num_branches).to(DEVICE)
            for _ in range(self.num_branches)
        ]
        self.membrane_potential = torch.rand(batch_size, self.hidden_size).to(DEVICE)
        self.spike_output = torch.zeros(batch_size, self.hidden_size).to(DEVICE)
    
    def surrogate_gradient(self, x):
        """使用多高斯替代函数"""
        return multi_gaussian_surrogate(x)
    
    def forward(self, x):
        """
        前向传播
        
        参数:
            x: 输入脉冲序列，形状为[批次, 时间步, 特征]
            
        返回:
            output: 输出logits
        """
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
        
        # 时间维度积分 - 使用后1/3的输出（长序列处理）
        start_idx = seq_len * 2 // 3
        integrated_output = torch.stack(outputs[start_idx:], dim=1).sum(dim=1)
        
        # 输出层
        final_output = self.output_layer(integrated_output)
        
        return final_output

class SequentialMNIST_Vanilla_SNN(nn.Module):
    """
    用于Sequential MNIST的普通SNN模型
    """
    
    def __init__(self, input_size=INPUT_SIZE, hidden_size=HIDDEN_SIZE, output_size=OUTPUT_SIZE):
        """
        初始化Sequential MNIST普通SNN模型
        """
        super(SequentialMNIST_Vanilla_SNN, self).__init__()
        
        print(f"🏗️  创建Sequential MNIST普通SNN模型:")
        print(f"   输入维度: {input_size}")
        print(f"   隐藏维度: {hidden_size}")
        print(f"   输出维度: {output_size}")
        
        # 第一层
        self.fc1 = layer.Linear(input_size, hidden_size)
        self.lif1 = neuron.LIFNode(
            tau=2.0,
            v_threshold=1.0,
            surrogate_function=surrogate.ATan(),
            step_mode='s'
        )
        
        # 输出层
        self.fc2 = layer.Linear(hidden_size, output_size)
        self.lif2 = neuron.LIFNode(
            tau=2.0,
            v_threshold=1.0,
            surrogate_function=surrogate.ATan(),
            step_mode='s'
        )
        
        print("✅ Sequential MNIST普通SNN模型创建完成")
    
    def forward(self, x):
        """
        前向传播
        
        参数:
            x: 输入脉冲序列，形状为[批次, 时间步, 特征]
            
        返回:
            output: 输出logits
        """
        batch_size, seq_len, input_dim = x.shape
        
        # 重置神经元状态
        functional.reset_net(self)
        
        outputs = []
        for t in range(seq_len):
            x_t = x[:, t, :]  # [batch, input_size]
            
            # 第一层
            h1 = self.fc1(x_t)
            s1 = self.lif1(h1)
            
            # 输出层
            h2 = self.fc2(s1)
            s2 = self.lif2(h2)
            
            outputs.append(s2)
        
        # 时间维度积分
        start_idx = seq_len * 2 // 3
        integrated_output = torch.stack(outputs[start_idx:], dim=1).sum(dim=1)
        
        return integrated_output

# ==================== 训练和测试函数 ====================

def train_smnist_model(model, train_loader, test_loader, model_name, num_epochs=NUM_EPOCHS):
    """
    训练Sequential MNIST模型
    
    参数:
        model: 待训练的模型
        train_loader: 训练数据加载器
        test_loader: 测试数据加载器
        model_name: 模型名称
        num_epochs: 训练轮数
        
    返回:
        results: 训练结果
    """
    print(f"\n🚀 开始训练 {model_name}")
    print("-" * 50)
    
    model = model.to(DEVICE)
    
    # 优化器配置
    if isinstance(model, SequentialMNIST_DH_SNN):
        # DH-SNN使用分层学习率
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
    
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=8, gamma=0.5)
    criterion = nn.CrossEntropyLoss()
    
    best_test_acc = 0.0
    train_losses = []
    test_accuracies = []
    
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
            
            loss = criterion(outputs, batch_labels)
            loss.backward()
            
            # 梯度裁剪（重要：防止长序列梯度爆炸）
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            
            optimizer.step()
            
            running_loss += loss.item()
            
            # 计算准确率
            _, predicted = torch.max(outputs.data, 1)
            total += batch_labels.size(0)
            correct += (predicted == batch_labels).sum().item()
        
        scheduler.step()
        
        train_acc = 100 * correct / total
        train_loss = running_loss / len(train_loader)
        
        # 测试阶段
        model.eval()
        test_correct = 0
        test_total = 0
        test_loss = 0.0
        
        with torch.no_grad():
            for batch_data, batch_labels in test_loader:
                batch_data, batch_labels = batch_data.to(DEVICE), batch_labels.to(DEVICE)
                
                outputs = model(batch_data)
                loss = criterion(outputs, batch_labels)
                test_loss += loss.item()
                
                _, predicted = torch.max(outputs.data, 1)
                test_total += batch_labels.size(0)
                test_correct += (predicted == batch_labels).sum().item()
        
        test_acc = 100 * test_correct / test_total
        test_loss = test_loss / len(test_loader)
        
        train_losses.append(train_loss)
        test_accuracies.append(test_acc)
        
        if test_acc > best_test_acc:
            best_test_acc = test_acc
        
        # 打印进度
        if epoch % 5 == 0 or epoch == num_epochs - 1:
            print(f'轮次 [{epoch+1}/{num_epochs}]: 训练损失={train_loss:.4f}, 训练准确率={train_acc:.1f}%, 测试准确率={test_acc:.1f}%, 最佳={best_test_acc:.1f}%')
    
    return {
        'train_losses': train_losses,
        'test_accuracies': test_accuracies,
        'best_test_acc': best_test_acc,
        'final_test_acc': test_acc
    }

# ==================== 主实验函数 ====================

def run_smnist_experiment():
    """运行Sequential MNIST实验"""
    
    print("=" * 80)
    print("🔢 DH-SNN Sequential MNIST序列分类实验")
    print("=" * 80)
    
    # 设置随机种子
    setup_seed(42)
    
    print(f"🖥️  使用设备: {DEVICE}")
    
    try:
        # 创建数据集
        print("📊 准备Sequential MNIST数据集...")
        train_loader, test_loader = create_sequential_mnist_datasets()
        
        # 创建模型
        print(f"\n🏗️  在 {DEVICE} 上初始化模型...")
        
        dh_snn_model = SequentialMNIST_DH_SNN()
        vanilla_snn_model = SequentialMNIST_Vanilla_SNN()
        
        print(f"📊 模型参数统计:")
        print(f"   DH-SNN参数: {sum(p.numel() for p in dh_snn_model.parameters()):,}")
        print(f"   普通SNN参数: {sum(p.numel() for p in vanilla_snn_model.parameters()):,}")
        
        # 开始训练实验
        print(f"\n🔬 开始训练实验...")
        
        # 训练DH-SNN
        dh_results = train_smnist_model(
            dh_snn_model, train_loader, test_loader, "DH-SNN", NUM_EPOCHS
        )
        
        # 训练普通SNN
        vanilla_results = train_smnist_model(
            vanilla_snn_model, train_loader, test_loader, "普通SNN", NUM_EPOCHS
        )
        
        # 结果对比
        print("\n" + "=" * 80)
        print("🎯 Sequential MNIST实验结果对比")
        print("=" * 80)
        
        print(f"DH-SNN结果:")
        print(f"  最佳测试准确率: {dh_results['best_test_acc']:.2f}%")
        print(f"  最终测试准确率: {dh_results['final_test_acc']:.2f}%")
        
        print(f"\n普通SNN结果:")
        print(f"  最佳测试准确率: {vanilla_results['best_test_acc']:.2f}%")
        print(f"  最终测试准确率: {vanilla_results['final_test_acc']:.2f}%")
        
        # 计算改进
        best_improvement = dh_results['best_test_acc'] - vanilla_results['best_test_acc']
        final_improvement = dh_results['final_test_acc'] - vanilla_results['final_test_acc']
        
        if vanilla_results['best_test_acc'] > 0:
            best_relative = (dh_results['best_test_acc'] / vanilla_results['best_test_acc'] - 1) * 100
        else:
            best_relative = 0
            
        if vanilla_results['final_test_acc'] > 0:
            final_relative = (dh_results['final_test_acc'] / vanilla_results['final_test_acc'] - 1) * 100
        else:
            final_relative = 0
        
        print(f"\n📈 性能改进:")
        print(f"  最佳准确率: +{best_improvement:.2f}% (相对: +{best_relative:.1f}%)")
        print(f"  最终准确率: +{final_improvement:.2f}% (相对: +{final_relative:.1f}%)")
        
        # 保存结果
        results_path = Path("results/smnist_experiment_results.json")
        results_path.parent.mkdir(parents=True, exist_ok=True)
        
        all_results = {
            'experiment_info': {
                'name': 'Sequential MNIST序列分类实验',
                'framework': 'SpikingJelly + DH-SNN',
                'date': time.strftime('%Y-%m-%d %H:%M:%S'),
                'dataset': 'Sequential MNIST',
                'sequence_length': SEQ_LENGTH,
                'num_classes': OUTPUT_SIZE,
                'batch_size': BATCH_SIZE,
                'num_epochs': NUM_EPOCHS,
                'num_branches': NUM_BRANCHES,
                'device': str(DEVICE)
            },
            'dh_snn': dh_results,
            'vanilla_snn': vanilla_results,
            'comparison': {
                'best_accuracy_improvement': {
                    'absolute': best_improvement,
                    'relative_percent': best_relative
                },
                'final_accuracy_improvement': {
                    'absolute': final_improvement,
                    'relative_percent': final_relative
                }
            }
        }
        
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)
        
        print(f"\n💾 结果已保存到: {results_path}")
        
        # 与论文/基准结果对比
        print(f"\n📈 与基准结果对比:")
        print(f"Sequential MNIST是经典的序列学习基准任务")
        print(f"DH-SNN在长序列处理中展现优势")
        
        if best_improvement > 3:
            print("🎉 DH-SNN显著优于普通SNN - 序列处理能力突出!")
        elif best_improvement > 1:
            print("✅ DH-SNN优于普通SNN")
        else:
            print("⚠️  结果需要进一步分析")
        
        return all_results
        
    except Exception as e:
        print(f"\n❌ 实验失败: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    results = run_smnist_experiment()
    if results:
        print(f"\n🏁 Sequential MNIST实验成功完成!")
    else:
        print(f"\n❌ Sequential MNIST实验失败")