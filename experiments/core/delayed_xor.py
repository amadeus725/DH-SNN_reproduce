#!/usr/bin/env python3
"""
DH-SNN 延迟异或（Delayed XOR）实验
==================================

基于SpikingJelly框架的DH-SNN vs 普通SNN对比实验
使用延迟异或任务验证DH-SNN的时间处理能力

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

from dh_snn.utils import setup_seed

print("🚀 DH-SNN 延迟异或任务实验")
print("="*60)

# 实验参数
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
NUM_EPOCHS = 200
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 延迟异或任务参数
SEQ_LENGTH = 400  # 序列长度
INPUT_SIZE = 2    # 输入维度（两个信号）
HIDDEN_SIZE = 32  # 隐藏层大小
OUTPUT_SIZE = 1   # 输出维度（二分类）
DELAY_RANGE = [25, 50, 100, 200, 400]  # 不同的延迟设置

# ==================== 延迟异或数据生成 ====================

def generate_delayed_xor_data(batch_size, seq_length, delay, num_samples=1000):
    """
    生成延迟异或任务数据
    
    参数:
        batch_size: 批次大小
        seq_length: 序列长度
        delay: 延迟步数
        num_samples: 样本数量
        
    返回:
        data: 输入脉冲序列 [样本数, 时间步, 输入维度]
        labels: 目标标签 [样本数]
    """
    data = torch.zeros(num_samples, seq_length, INPUT_SIZE)
    labels = torch.zeros(num_samples, dtype=torch.long)
    
    for i in range(num_samples):
        # 在序列开始处生成两个随机脉冲
        signal1_time = np.random.randint(5, 15)  # 第一个信号的时间
        signal2_time = signal1_time + delay      # 第二个信号延迟delay步
        
        # 确保第二个信号在序列范围内
        if signal2_time < seq_length - 10:
            # 生成脉冲
            signal1_value = np.random.choice([0, 1])
            signal2_value = np.random.choice([0, 1])
            
            data[i, signal1_time, 0] = signal1_value
            data[i, signal2_time, 1] = signal2_value
            
            # 异或标签
            labels[i] = signal1_value ^ signal2_value
        else:
            # 如果延迟太长，标签设为0
            data[i, signal1_time, 0] = np.random.choice([0, 1])
            labels[i] = 0
    
    return data, labels

def create_delayed_xor_datasets(delays=DELAY_RANGE):
    """
    创建不同延迟的异或数据集
    
    参数:
        delays: 延迟列表
        
    返回:
        datasets: 包含不同延迟数据集的字典
    """
    datasets = {}
    
    for delay in delays:
        print(f"📊 生成延迟{delay}步的异或数据...")
        
        # 生成训练数据
        train_data, train_labels = generate_delayed_xor_data(
            BATCH_SIZE, SEQ_LENGTH, delay, num_samples=2000
        )
        
        # 生成测试数据
        test_data, test_labels = generate_delayed_xor_data(
            BATCH_SIZE, SEQ_LENGTH, delay, num_samples=500
        )
        
        # 创建数据加载器
        train_dataset = torch.utils.data.TensorDataset(train_data, train_labels)
        test_dataset = torch.utils.data.TensorDataset(test_data, test_labels)
        
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=BATCH_SIZE, shuffle=True
        )
        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=BATCH_SIZE, shuffle=False
        )
        
        datasets[delay] = {
            'train_loader': train_loader,
            'test_loader': test_loader,
            'train_size': len(train_data),
            'test_size': len(test_data)
        }
        
        print(f"   训练样本: {len(train_data)}, 测试样本: {len(test_data)}")
    
    return datasets

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

class DelayedXOR_DH_SNN(nn.Module):
    """
    用于延迟异或任务的DH-SNN模型
    """
    
    def __init__(self, input_size=INPUT_SIZE, hidden_size=HIDDEN_SIZE, output_size=OUTPUT_SIZE, num_branches=2):
        """
        初始化延迟异或DH-SNN模型
        
        参数:
            input_size: 输入维度
            hidden_size: 隐藏层维度
            output_size: 输出维度
            num_branches: 树突分支数量
        """
        super(DelayedXOR_DH_SNN, self).__init__()
        
        print(f"🏗️  创建延迟异或DH-SNN模型:")
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
        # tau_n: 树突时间常数，用Large初始化(2,6)
        self.tau_n = nn.Parameter(torch.empty(num_branches, hidden_size).uniform_(2, 6))
        # tau_m: 膜电位时间常数，用Medium初始化(0,4)
        self.tau_m = nn.Parameter(torch.empty(hidden_size).uniform_(0, 4))
        
        # 输出层
        self.output_layer = layer.Linear(hidden_size, output_size)
        
        # 神经元状态
        self.dendritic_currents = None
        self.membrane_potential = None
        self.spike_output = None
        
        print("✅ 延迟异或DH-SNN模型创建完成")
    
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
        
        # 时间维度积分 - 只使用后半段输出
        integrated_output = torch.stack(outputs[seq_len//2:], dim=1).sum(dim=1)
        
        # 输出层
        final_output = self.output_layer(integrated_output)
        
        return final_output

class DelayedXOR_Vanilla_SNN(nn.Module):
    """
    用于延迟异或任务的普通SNN模型
    """
    
    def __init__(self, input_size=INPUT_SIZE, hidden_size=HIDDEN_SIZE, output_size=OUTPUT_SIZE):
        """
        初始化延迟异或普通SNN模型
        """
        super(DelayedXOR_Vanilla_SNN, self).__init__()
        
        print(f"🏗️  创建延迟异或普通SNN模型:")
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
        
        print("✅ 延迟异或普通SNN模型创建完成")
    
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
        integrated_output = torch.stack(outputs[seq_len//2:], dim=1).sum(dim=1)
        
        return integrated_output

# ==================== 训练和测试函数 ====================

def train_delayed_xor_model(model, train_loader, test_loader, model_name, num_epochs=NUM_EPOCHS):
    """
    训练延迟异或模型
    
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
    if isinstance(model, DelayedXOR_DH_SNN):
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
    
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)
    criterion = nn.BCEWithLogitsLoss()  # 二分类任务
    
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
            
            # 调整输出和标签形状
            outputs = outputs.squeeze(-1)  # [batch] 
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
        test_loss = 0.0
        
        with torch.no_grad():
            for batch_data, batch_labels in test_loader:
                batch_data, batch_labels = batch_data.to(DEVICE), batch_labels.to(DEVICE)
                
                outputs = model(batch_data)
                outputs = outputs.squeeze(-1)
                batch_labels = batch_labels.float()
                
                loss = criterion(outputs, batch_labels)
                test_loss += loss.item()
                
                predicted = (torch.sigmoid(outputs) > 0.5).float()
                test_total += batch_labels.size(0)
                test_correct += (predicted == batch_labels).sum().item()
        
        test_acc = 100 * test_correct / test_total
        test_loss = test_loss / len(test_loader)
        
        train_losses.append(train_loss)
        test_accuracies.append(test_acc)
        
        if test_acc > best_test_acc:
            best_test_acc = test_acc
        
        # 打印进度
        if epoch % 50 == 0 or epoch == num_epochs - 1:
            print(f'轮次 [{epoch+1}/{num_epochs}]: 训练损失={train_loss:.4f}, 训练准确率={train_acc:.1f}%, 测试准确率={test_acc:.1f}%, 最佳={best_test_acc:.1f}%')
    
    return {
        'train_losses': train_losses,
        'test_accuracies': test_accuracies,
        'best_test_acc': best_test_acc,
        'final_test_acc': test_acc
    }

# ==================== 主实验函数 ====================

def run_delayed_xor_experiment():
    """运行延迟异或实验"""
    
    print("=" * 80)
    print("⏰ DH-SNN 延迟异或任务实验")
    print("=" * 80)
    
    # 设置随机种子
    setup_seed(42)
    
    print(f"🖥️  使用设备: {DEVICE}")
    
    try:
        # 创建不同延迟的数据集
        print("📊 创建延迟异或数据集...")
        datasets = create_delayed_xor_datasets(DELAY_RANGE)
        
        all_results = {}
        
        # 对每个延迟设置进行实验
        for delay in DELAY_RANGE:
            print(f"\n🔬 实验延迟={delay}步的异或任务")
            print("=" * 50)
            
            train_loader = datasets[delay]['train_loader']
            test_loader = datasets[delay]['test_loader']
            
            # 创建模型
            dh_snn_model = DelayedXOR_DH_SNN()
            vanilla_snn_model = DelayedXOR_Vanilla_SNN()
            
            print(f"📊 模型参数统计:")
            print(f"   DH-SNN参数: {sum(p.numel() for p in dh_snn_model.parameters()):,}")
            print(f"   普通SNN参数: {sum(p.numel() for p in vanilla_snn_model.parameters()):,}")
            
            # 训练模型
            print(f"\n🚀 训练延迟{delay}步的模型...")
            
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
                'dh_snn': dh_results,
                'vanilla_snn': vanilla_results,
                'improvement': dh_results['best_test_acc'] - vanilla_results['best_test_acc']
            }
            
            print(f"\n📈 延迟{delay}步结果:")
            print(f"   DH-SNN最佳准确率: {dh_results['best_test_acc']:.1f}%")
            print(f"   普通SNN最佳准确率: {vanilla_results['best_test_acc']:.1f}%")
            print(f"   性能提升: {all_results[delay]['improvement']:+.1f}%")
        
        # 总结所有结果
        print("\n" + "=" * 80)
        print("🎯 延迟异或实验总结")
        print("=" * 80)
        
        print("延迟步数 | DH-SNN | 普通SNN | 提升")
        print("-" * 40)
        
        total_dh_acc = 0
        total_vanilla_acc = 0
        
        for delay in DELAY_RANGE:
            dh_acc = all_results[delay]['dh_snn']['best_test_acc']
            vanilla_acc = all_results[delay]['vanilla_snn']['best_test_acc']
            improvement = all_results[delay]['improvement']
            
            print(f"{delay:8d} | {dh_acc:6.1f}% | {vanilla_acc:7.1f}% | {improvement:+5.1f}%")
            
            total_dh_acc += dh_acc
            total_vanilla_acc += vanilla_acc
        
        # 计算平均性能
        avg_dh_acc = total_dh_acc / len(DELAY_RANGE)
        avg_vanilla_acc = total_vanilla_acc / len(DELAY_RANGE)
        avg_improvement = avg_dh_acc - avg_vanilla_acc
        
        print("-" * 40)
        print(f"平均     | {avg_dh_acc:6.1f}% | {avg_vanilla_acc:7.1f}% | {avg_improvement:+5.1f}%")
        
        # 保存结果
        results_path = Path("results/delayed_xor_experiment_results.json")
        results_path.parent.mkdir(parents=True, exist_ok=True)
        
        final_results = {
            'experiment_info': {
                'name': '延迟异或任务实验',
                'framework': 'SpikingJelly + DH-SNN',
                'date': time.strftime('%Y-%m-%d %H:%M:%S'),
                'delay_range': DELAY_RANGE,
                'seq_length': SEQ_LENGTH,
                'num_epochs': NUM_EPOCHS,
                'device': str(DEVICE)
            },
            'results_by_delay': all_results,
            'summary': {
                'avg_dh_snn_acc': avg_dh_acc,
                'avg_vanilla_snn_acc': avg_vanilla_acc,
                'avg_improvement': avg_improvement
            }
        }
        
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(final_results, f, indent=2, ensure_ascii=False)
        
        print(f"\n💾 结果已保存到: {results_path}")

        if avg_improvement > 10:
            print("🎉 DH-SNN显著优于普通SNN - 符合预期!")
        elif avg_improvement > 5:
            print("✅ DH-SNN明显优于普通SNN")
        elif avg_improvement > 0:
            print("✅ DH-SNN优于普通SNN")
        else:
            print("⚠️  结果需要进一步分析")
        
        return final_results
        
    except Exception as e:
        print(f"\n❌ 实验失败: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    results = run_delayed_xor_experiment()
    if results:
        print(f"\n🏁 延迟异或实验成功完成!")
    else:
        print(f"\n❌ 延迟异或实验失败")