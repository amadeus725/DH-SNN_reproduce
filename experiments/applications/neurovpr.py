#!/usr/bin/env python3
"""
DH-SNN NeuroVPR（神经视觉位置识别）实验
=========================================

基于SpikingJelly框架的DH-SNN vs 普通SNN对比实验
使用NeuroVPR数据集进行视觉位置识别任务

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
from pathlib import Path

# 添加项目路径
current_dir = Path(__file__).parent.parent.parent
sys.path.append(str(current_dir))

# SpikingJelly导入
from spikingjelly.activation_based import neuron, functional, layer, surrogate

from dh_snn.utils import setup_seed

print("🚀 DH-SNN NeuroVPR神经视觉位置识别实验")
print("="*60)

# 实验参数
BATCH_SIZE = 16
N_CLASS = 100
LEARNING_RATE = 1e-3
NUM_EPOCHS = 25
NUM_ITER = 40
NUM_BRANCHES = 4
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# DVS序列参数
SEQ_LEN_DVS = 4
DVS_EXPAND = 3

# ==================== 模拟数据生成 ====================

def create_mock_neurovpr_data():
    """
    创建模拟NeuroVPR数据用于测试
    模拟DVS事件相机数据的时空特性
    
    返回:
        train_loader: 训练数据加载器
        test_loader: 测试数据加载器
    """
    print("🎲 创建模拟NeuroVPR数据...")
    
    # 数据参数
    num_train = 1000
    num_test = 200
    input_height = 32
    input_width = 43
    channels = 2  # DVS双通道（ON/OFF事件）
    sequence_length = SEQ_LEN_DVS * DVS_EXPAND  # 时间序列长度
    
    # 生成训练数据
    train_data = torch.zeros(num_train, sequence_length, channels, input_height, input_width)
    train_labels = torch.randint(0, N_CLASS, (num_train,))
    
    # 生成测试数据
    test_data = torch.zeros(num_test, sequence_length, channels, input_height, input_width)
    test_labels = torch.randint(0, N_CLASS, (num_test,))
    
    # 为每个样本添加位置相关的事件模式
    for i in range(num_train):
        label = train_labels[i].item()
        
        # 不同位置有不同的空间模式
        center_h = (label % 10) * 3 + 5  # 位置相关的垂直中心
        center_w = (label // 10) * 4 + 5  # 位置相关的水平中心
        
        # 为每个时间步生成事件
        for t in range(sequence_length):
            # 在时间维度上添加变化
            time_offset = t * 0.2
            
            # 生成ON事件（通道0）
            for _ in range(50 + label % 20):  # 不同位置有不同的事件密度
                h = int(np.clip(np.random.normal(center_h + time_offset, 3), 0, input_height-1))
                w = int(np.clip(np.random.normal(center_w, 3), 0, input_width-1))
                train_data[i, t, 0, h, w] = 1.0
            
            # 生成OFF事件（通道1）
            for _ in range(30 + label % 15):
                h = int(np.clip(np.random.normal(center_h - time_offset, 2), 0, input_height-1))
                w = int(np.clip(np.random.normal(center_w + 1, 2), 0, input_width-1))
                train_data[i, t, 1, h, w] = 1.0
    
    # 为测试数据生成类似模式
    for i in range(num_test):
        label = test_labels[i].item()
        center_h = (label % 10) * 3 + 5
        center_w = (label // 10) * 4 + 5
        
        for t in range(sequence_length):
            time_offset = t * 0.2
            
            # ON事件
            for _ in range(50 + label % 20):
                h = int(np.clip(np.random.normal(center_h + time_offset, 3), 0, input_height-1))
                w = int(np.clip(np.random.normal(center_w, 3), 0, input_width-1))
                test_data[i, t, 0, h, w] = 1.0
            
            # OFF事件
            for _ in range(30 + label % 15):
                h = int(np.clip(np.random.normal(center_h - time_offset, 2), 0, input_height-1))
                w = int(np.clip(np.random.normal(center_w + 1, 2), 0, input_width-1))
                test_data[i, t, 1, h, w] = 1.0
    
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
    print(f"   输入特征维度: {sequence_length * channels * input_height * input_width}")
    
    return train_loader, test_loader

# ==================== 精度计算函数 ====================

def accuracy(output, target, topk=(1,)):
    """
    计算Top-K准确率
    
    参数:
        output: 模型输出，形状为[batch_size, num_classes]
        target: 真实标签，形状为[batch_size]
        topk: K值元组
        
    返回:
        res: 各个K值对应的准确率
    """
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size).item())
        return res

# ==================== 模型定义 ====================

class NeuroVPR_DH_SNN(nn.Module):
    """
    用于NeuroVPR任务的DH-SNN模型
    处理DVS事件相机的时空数据
    """
    
    def __init__(self, input_dim, hidden_dims=[256, 256], output_dim=N_CLASS, num_branches=NUM_BRANCHES):
        """
        初始化NeuroVPR DH-SNN模型
        
        参数:
            input_dim: 输入特征维度
            hidden_dims: 隐藏层维度列表
            output_dim: 输出类别数
            num_branches: 树突分支数量
        """
        super(NeuroVPR_DH_SNN, self).__init__()
        
        print(f"🏗️  创建NeuroVPR DH-SNN模型:")
        print(f"   输入维度: {input_dim}")
        print(f"   隐藏层: {hidden_dims}")
        print(f"   输出维度: {output_dim}")
        print(f"   分支数量: {num_branches}")
        
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        self.num_branches = num_branches
        
        # 构建多层DH-SNN网络
        self.layers = nn.ModuleList()
        
        current_dim = input_dim
        for i, hidden_dim in enumerate(hidden_dims):
            # 分支线性层
            branch_layers = nn.ModuleList()
            for j in range(num_branches):
                branch_layers.append(layer.Linear(current_dim // num_branches, hidden_dim // num_branches, bias=False))
            self.layers.append(branch_layers)
            
            # 树突时间常数参数
            tau_n = nn.Parameter(torch.empty(num_branches, hidden_dim).uniform_(0, 4))
            self.register_parameter(f'tau_n_{i}', tau_n)
            
            # 膜电位时间常数参数
            tau_m = nn.Parameter(torch.empty(hidden_dim).uniform_(0, 4))
            self.register_parameter(f'tau_m_{i}', tau_m)
            
            current_dim = hidden_dim
        
        # 输出层
        self.output_layer = layer.Linear(current_dim, output_dim)
        
        # 神经元状态缓存
        self.reset_states()
        
        print("✅ NeuroVPR DH-SNN模型创建完成")
    
    def reset_states(self):
        """重置所有神经元状态"""
        self.dendritic_currents = []
        self.membrane_potentials = []
        self.spike_outputs = []
        
        for i, hidden_dim in enumerate(self.hidden_dims):
            self.dendritic_currents.append([None] * self.num_branches)
            self.membrane_potentials.append(None)
            self.spike_outputs.append(None)
    
    def set_batch_size(self, batch_size):
        """设置批次大小并初始化状态"""
        for i, hidden_dim in enumerate(self.hidden_dims):
            # 初始化树突电流
            for j in range(self.num_branches):
                self.dendritic_currents[i][j] = torch.zeros(batch_size, hidden_dim // self.num_branches).to(DEVICE)
            
            # 初始化膜电位和脉冲输出
            self.membrane_potentials[i] = torch.rand(batch_size, hidden_dim).to(DEVICE)
            self.spike_outputs[i] = torch.zeros(batch_size, hidden_dim).to(DEVICE)
    
    def surrogate_gradient(self, x):
        """代理梯度函数"""
        return (x > 0).float() + 0.5 * torch.tanh(2 * x) * (1 - (x > 0).float())
    
    def forward_layer(self, x, layer_idx):
        """单层前向传播"""
        batch_size = x.size(0)
        
        # 分割输入到各个分支
        input_splits = torch.chunk(x, self.num_branches, dim=1)
        
        # 处理各个分支
        branch_outputs = []
        for j in range(self.num_branches):
            # 分支线性变换
            branch_input = self.layers[layer_idx][j](input_splits[j])
            
            # 树突时间常数
            tau_n = getattr(self, f'tau_n_{layer_idx}')
            beta = torch.sigmoid(tau_n[j])
            
            # 更新树突电流
            self.dendritic_currents[layer_idx][j] = (
                beta * self.dendritic_currents[layer_idx][j] + 
                (1 - beta) * branch_input
            )
            
            branch_outputs.append(self.dendritic_currents[layer_idx][j])
        
        # 汇总分支输出
        total_current = torch.cat(branch_outputs, dim=1)
        
        # 膜电位更新
        tau_m = getattr(self, f'tau_m_{layer_idx}')
        alpha = torch.sigmoid(tau_m)
        
        self.membrane_potentials[layer_idx] = (
            alpha * self.membrane_potentials[layer_idx] + 
            (1 - alpha) * total_current - 
            self.spike_outputs[layer_idx]
        )
        
        # 脉冲生成
        spike_input = self.membrane_potentials[layer_idx] - 1.0
        self.spike_outputs[layer_idx] = self.surrogate_gradient(spike_input)
        
        return self.spike_outputs[layer_idx]
    
    def forward(self, dvs_input):
        """
        前向传播
        
        参数:
            dvs_input: DVS输入数据，形状为[batch, seq_len, channels, height, width]
            
        返回:
            output: 分类输出
        """
        batch_size, seq_len, channels, height, width = dvs_input.shape
        
        # 设置批次大小
        self.set_batch_size(batch_size)
        
        # 将DVS数据重塑为特征向量
        dvs_reshaped = dvs_input.view(batch_size, seq_len, -1)  # [batch, seq_len, features]
        
        outputs = []
        for t in range(seq_len):
            x = dvs_reshaped[:, t, :]  # [batch, features]
            
            # 逐层前向传播
            for layer_idx in range(len(self.hidden_dims)):
                x = self.forward_layer(x, layer_idx)
            
            # 输出层
            output = self.output_layer(x)
            outputs.append(output)
        
        # 时间维度聚合 - 使用最后几个时间步的平均
        final_output = torch.stack(outputs[-3:], dim=1).mean(dim=1)
        
        return final_output

class NeuroVPR_Vanilla_SNN(nn.Module):
    """
    用于NeuroVPR任务的普通SNN模型
    """
    
    def __init__(self, input_dim, hidden_dims=[256, 256], output_dim=N_CLASS):
        """
        初始化NeuroVPR 普通SNN模型
        
        参数:
            input_dim: 输入特征维度
            hidden_dims: 隐藏层维度列表
            output_dim: 输出类别数
        """
        super(NeuroVPR_Vanilla_SNN, self).__init__()
        
        print(f"🏗️  创建NeuroVPR 普通SNN模型:")
        print(f"   输入维度: {input_dim}")
        print(f"   隐藏层: {hidden_dims}")
        print(f"   输出维度: {output_dim}")
        
        # 构建普通SNN网络
        self.layers = nn.ModuleList()
        
        current_dim = input_dim
        for hidden_dim in hidden_dims:
            # 线性层
            self.layers.append(layer.Linear(current_dim, hidden_dim))
            # LIF神经元
            self.layers.append(neuron.LIFNode(
                tau=2.0,
                v_threshold=1.0,
                surrogate_function=surrogate.ATan(),
                step_mode='s'
            ))
            current_dim = hidden_dim
        
        # 输出层
        self.layers.append(layer.Linear(current_dim, output_dim))
        self.layers.append(neuron.LIFNode(
            tau=2.0,
            v_threshold=1.0,
            surrogate_function=surrogate.ATan(),
            step_mode='s'
        ))
        
        print("✅ NeuroVPR 普通SNN模型创建完成")
    
    def forward(self, dvs_input):
        """
        前向传播
        
        参数:
            dvs_input: DVS输入数据，形状为[batch, seq_len, channels, height, width]
            
        返回:
            output: 分类输出
        """
        batch_size, seq_len, channels, height, width = dvs_input.shape
        
        # 将DVS数据重塑为特征向量
        dvs_reshaped = dvs_input.view(batch_size, seq_len, -1)  # [batch, seq_len, features]
        
        outputs = []
        for t in range(seq_len):
            x = dvs_reshaped[:, t, :]  # [batch, features]
            
            # 重置神经元状态
            functional.reset_net(self)
            
            # 逐层前向传播
            for layer in self.layers:
                x = layer(x)
            
            outputs.append(x)
        
        # 时间维度聚合
        final_output = torch.stack(outputs[-3:], dim=1).mean(dim=1)
        
        return final_output

# ==================== 训练函数 ====================

def train_neurovpr_model(model, train_loader, test_loader, model_name, num_epochs=NUM_EPOCHS):
    """
    训练NeuroVPR模型
    
    参数:
        model: 待训练的模型
        train_loader: 训练数据加载器
        test_loader: 测试数据加载器
        model_name: 模型名称
        num_epochs: 训练轮数
        
    返回:
        results: 训练结果字典
    """
    print(f"\n🚀 开始训练 {model_name}")
    print("-" * 50)
    
    model = model.to(DEVICE)
    
    # 优化器和调度器
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    criterion = nn.CrossEntropyLoss()
    
    # 训练指标
    train_losses = []
    test_accuracies = []
    best_test_acc1 = 0.0
    best_test_acc5 = 0.0
    
    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        running_loss = 0.0
        train_acc1 = 0.0
        train_acc5 = 0.0
        train_batches = 0
        
        for batch_idx, (dvs_data, target) in enumerate(train_loader):
            if batch_idx >= NUM_ITER:  # 限制每轮迭代次数
                break
                
            dvs_data, target = dvs_data.to(DEVICE), target.to(DEVICE)
            
            # 重置模型状态
            if hasattr(model, 'reset_states'):
                model.reset_states()
            
            optimizer.zero_grad()
            outputs = model(dvs_data)
            loss = criterion(outputs, target)
            
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            running_loss += loss.item()
            
            # 计算训练准确率
            acc1, acc5 = accuracy(outputs.cpu(), target.cpu(), topk=(1, 5))
            train_acc1 += acc1
            train_acc5 += acc5
            train_batches += 1
        
        lr_scheduler.step()
        
        # 测试阶段
        model.eval()
        test_loss = 0.0
        test_acc1 = 0.0
        test_acc5 = 0.0
        test_batches = 0
        
        with torch.no_grad():
            for dvs_data, target in test_loader:
                dvs_data, target = dvs_data.to(DEVICE), target.to(DEVICE)
                
                if hasattr(model, 'reset_states'):
                    model.reset_states()
                
                outputs = model(dvs_data)
                loss = criterion(outputs, target)
                test_loss += loss.item()
                
                acc1, acc5 = accuracy(outputs.cpu(), target.cpu(), topk=(1, 5))
                test_acc1 += acc1
                test_acc5 += acc5
                test_batches += 1
        
        # 计算平均指标
        avg_train_loss = running_loss / train_batches
        avg_train_acc1 = train_acc1 / train_batches
        avg_train_acc5 = train_acc5 / train_batches
        avg_test_loss = test_loss / test_batches
        avg_test_acc1 = test_acc1 / test_batches
        avg_test_acc5 = test_acc5 / test_batches
        
        train_losses.append(avg_train_loss)
        test_accuracies.append(avg_test_acc1)
        
        # 更新最佳准确率
        if avg_test_acc1 > best_test_acc1:
            best_test_acc1 = avg_test_acc1
        if avg_test_acc5 > best_test_acc5:
            best_test_acc5 = avg_test_acc5
        
        # 打印进度
        if epoch % 5 == 0 or epoch == num_epochs - 1:
            print(f'轮次 [{epoch+1}/{num_epochs}]:')
            print(f'  训练损失: {avg_train_loss:.4f}, 训练Acc1: {avg_train_acc1:.2f}%')
            print(f'  测试损失: {avg_test_loss:.4f}, 测试Acc1: {avg_test_acc1:.2f}%, 测试Acc5: {avg_test_acc5:.2f}%')
            print(f'  最佳测试Acc1: {best_test_acc1:.2f}%, 最佳测试Acc5: {best_test_acc5:.2f}%')
    
    return {
        'train_losses': train_losses,
        'test_accuracies': test_accuracies,
        'best_test_acc1': best_test_acc1,
        'best_test_acc5': best_test_acc5,
        'final_test_acc1': avg_test_acc1,
        'final_test_acc5': avg_test_acc5
    }

# ==================== 主实验函数 ====================

def run_neurovpr_experiment():
    """运行NeuroVPR实验"""
    
    print("=" * 80)
    print("👁️  DH-SNN NeuroVPR神经视觉位置识别实验")
    print("=" * 80)
    
    # 设置随机种子
    setup_seed(42)
    
    print(f"🖥️  使用设备: {DEVICE}")
    
    try:
        # 创建模拟数据
        train_loader, test_loader = create_mock_neurovpr_data()
        
        # 计算输入特征维度
        sample_data, _ = next(iter(train_loader))
        input_features = sample_data.view(sample_data.shape[0], -1).shape[1] // SEQ_LEN_DVS // DVS_EXPAND
        print(f"   输入特征维度: {input_features}")
        
        # 创建模型
        print(f"\n🏗️  在 {DEVICE} 上初始化模型...")
        
        # DH-SNN模型
        dh_snn_model = NeuroVPR_DH_SNN(
            input_dim=input_features,
            hidden_dims=[256, 256],
            output_dim=N_CLASS,
            num_branches=NUM_BRANCHES
        )
        
        # 普通SNN模型
        vanilla_snn_model = NeuroVPR_Vanilla_SNN(
            input_dim=input_features,
            hidden_dims=[256, 256],
            output_dim=N_CLASS
        )
        
        print(f"📊 模型参数统计:")
        print(f"   DH-SNN参数: {sum(p.numel() for p in dh_snn_model.parameters()):,}")
        print(f"   普通SNN参数: {sum(p.numel() for p in vanilla_snn_model.parameters()):,}")
        
        # 开始训练实验
        print(f"\n🔬 开始训练实验...")
        
        # 训练DH-SNN
        dh_results = train_neurovpr_model(dh_snn_model, train_loader, test_loader, "DH-SNN")
        
        # 训练普通SNN
        vanilla_results = train_neurovpr_model(vanilla_snn_model, train_loader, test_loader, "普通SNN")
        
        # 结果对比
        print("\n" + "=" * 80)
        print("🎯 最终结果对比")
        print("=" * 80)
        
        print(f"DH-SNN结果:")
        print(f"  最佳测试准确率: {dh_results['best_test_acc1']:.2f}%")
        print(f"  最终测试准确率: {dh_results['final_test_acc1']:.2f}%")
        print(f"  最佳Top-5准确率: {dh_results['best_test_acc5']:.2f}%")
        
        print(f"\n普通SNN结果:")
        print(f"  最佳测试准确率: {vanilla_results['best_test_acc1']:.2f}%")
        print(f"  最终测试准确率: {vanilla_results['final_test_acc1']:.2f}%")
        print(f"  最佳Top-5准确率: {vanilla_results['best_test_acc5']:.2f}%")
        
        # 计算改进
        best_improvement = dh_results['best_test_acc1'] - vanilla_results['best_test_acc1']
        final_improvement = dh_results['final_test_acc1'] - vanilla_results['final_test_acc1']
        
        if vanilla_results['best_test_acc1'] > 0:
            best_relative = (dh_results['best_test_acc1'] / vanilla_results['best_test_acc1'] - 1) * 100
        else:
            best_relative = 0
            
        if vanilla_results['final_test_acc1'] > 0:
            final_relative = (dh_results['final_test_acc1'] / vanilla_results['final_test_acc1'] - 1) * 100
        else:
            final_relative = 0
        
        print(f"\n📈 性能改进:")
        print(f"  最佳准确率: +{best_improvement:.2f}% (相对: +{best_relative:.1f}%)")
        print(f"  最终准确率: +{final_improvement:.2f}% (相对: +{final_relative:.1f}%)")
        
        # 保存结果
        results_path = Path("results/neurovpr_experiment_results.json")
        results_path.parent.mkdir(parents=True, exist_ok=True)
        
        all_results = {
            'experiment_info': {
                'name': 'NeuroVPR神经视觉位置识别实验',
                'framework': 'SpikingJelly + DH-SNN',
                'date': time.strftime('%Y-%m-%d %H:%M:%S'),
                'dataset': 'NeuroVPR (模拟数据)',
                'num_classes': N_CLASS,
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
        
        # 与论文结果对比
        print(f"\n📈 与论文结果对比:")
        print(f"论文NeuroVPR任务中DH-SNN展现出了优越性")
        
        if best_improvement > 2:
            print("🎉 DH-SNN显著优于普通SNN - 符合预期!")
        elif best_improvement > 0:
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
    results = run_neurovpr_experiment()
    if results:
        print(f"\n🏁 NeuroVPR实验成功完成!")
    else:
        print(f"\n❌ NeuroVPR实验失败")
