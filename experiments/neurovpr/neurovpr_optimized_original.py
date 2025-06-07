#!/usr/bin/env python3
"""
NeuroVPR优化实验：基于原始代码应用性能优化方案
===================================================

应用识别的优化方案：
1. 差异化学习率配置
2. 短时间常数初始化
3. 减少模型复杂度
4. 时间步融合策略
5. 梯度裁剪

基于原始的 neurovpr_spikingjelly_fixed.py，应用性能gap分析中的优化方案。

Authors: DH-SNN Reproduction Study
Date: 2025
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import time
import json
from pathlib import Path
from collections import Counter
import traceback

# 添加项目路径
sys.path.append('/root/DH-SNN_reproduce')
sys.path.append('/root/DH-SNN_reproduce/src')

# SpikingJelly imports
from spikingjelly.activation_based import neuron, functional, surrogate, layer, base

# 导入我们的核心DH-SNN实现
from src.core.neurons import DH_LIFNode, ReadoutNeuron, ParametricLIFNode
from src.core.layers import DendriticDenseLayer, ReadoutIntegrator
from src.core.models import DH_SNN, create_dh_snn
from src.core.surrogate import MultiGaussianSurrogate

import torchvision

# ==================== 原始数据加载器 ====================

# 导入原始的数据加载类
from neurovpr_spikingjelly_fixed import FixedNeuroVPRDataset, setup_seed, accuracy

# ==================== 优化的模型定义 ====================

class OptimizedDH_SNN(nn.Module):
    """优化的DH-SNN模型，应用性能gap分析的改进"""
    
    def __init__(self, input_dim, hidden_dims, output_dim, num_branches=2, step_mode='s'):
        super().__init__()
        
        print(f"🏗️  创建优化DH-SNN模型:")
        print(f"   - 输入维度: {input_dim}")
        print(f"   - 隐藏维度: {hidden_dims}")
        print(f"   - 输出维度: {output_dim}")
        print(f"   - 分支数: {num_branches} (减少复杂度)")
        
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        self.num_branches = num_branches
        self.step_mode = step_mode
        self.T = 3  # DVS数据的时间步数
        
        # 优化的DH-SNN配置
        dh_snn_config = {
            'input_dim': input_dim,
            'hidden_dims': hidden_dims,
            'output_dim': output_dim,
            'num_branches': num_branches,  # 减少分支数
            'v_threshold': 0.3,  # 适中阈值
            'tau_m_init': (0.1, 1.0),  # 短时间常数，适合短序列
            'tau_n_init': (0.1, 1.0),  # 短时间常数，适合短序列
            'tau_initializer': 'uniform',
            'sparsity': 1.0/num_branches,
            'mask_share': 1,
            'bias': True,
            'surrogate_function': MultiGaussianSurrogate(),
            'reset_mode': 'soft',
            'step_mode': step_mode
        }
        
        self.dh_snn = create_dh_snn(dh_snn_config)
        
        print("✅ 优化DH-SNN模型创建完成")
    
    def forward(self, inp):
        """优化的前向传播，包含时间步融合"""
        # 输入格式: ([aps, gps, dvs], labels) -> 取DVS数据
        dvs_inp = inp[2]  # [batch, 3, 2, 32, 43]
        batch_size = dvs_inp.shape[0]
        
        # 重塑时间维度：[batch, 3, 2, 32, 43] -> [batch, 3, 2752]
        dvs_reshaped = dvs_inp.view(batch_size, self.T, -1)  # [batch, 3, 2752]
        dvs_input = dvs_reshaped.transpose(0, 1)  # [3, batch, 2752]
        
        # 处理所有时间步并累积
        outputs = []
        for t in range(self.T):
            output = self.dh_snn(dvs_input[t])  # 输入: [batch, 2752]
            outputs.append(output)
        
        # 时间步融合：加权平均，给后期时间步更高权重
        weights = torch.softmax(torch.tensor([0.5, 0.7, 1.0]), dim=0).to(outputs[0].device)
        final_output = sum(w * out for w, out in zip(weights, outputs))
        
        return final_output

class OptimizedVanilla_SNN(nn.Module):
    """优化的Vanilla SNN，用于对比"""
    
    def __init__(self, input_dim, hidden_dims, output_dim, step_mode='s'):
        super().__init__()
        
        self.layers = nn.ModuleList()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.step_mode = step_mode
        self.T = 3
        
        # 构建网络
        current_dim = input_dim
        for hidden_dim in hidden_dims:
            self.layers.append(layer.Linear(current_dim, hidden_dim))
            self.layers.append(neuron.LIFNode(
                tau=2.0,  # 适中时间常数
                v_threshold=0.3,  # 适中阈值
                surrogate_function=surrogate.ATan(),
                step_mode=step_mode
            ))
            current_dim = hidden_dim
        
        # 输出层
        self.layers.append(layer.Linear(current_dim, output_dim))
    
    def forward(self, inp):
        """前向传播"""
        # 输入格式同DH-SNN
        dvs_inp = inp[2]  # [batch, 3, 2, 32, 43]
        batch_size = dvs_inp.shape[0]
        
        dvs_reshaped = dvs_inp.view(batch_size, self.T, -1)
        dvs_input = dvs_reshaped.transpose(0, 1)  # [3, batch, 2752]
        
        outputs = []
        for t in range(self.T):
            x = dvs_input[t]
            for layer_module in self.layers:
                x = layer_module(x)
            outputs.append(x)
        
        # 同样的时间步融合策略
        weights = torch.softmax(torch.tensor([0.5, 0.7, 1.0]), dim=0).to(outputs[0].device)
        final_output = sum(w * out for w, out in zip(weights, outputs))
        
        return final_output

# ==================== 优化的训练函数 ====================

def create_optimized_optimizer(model, base_lr=1e-3):
    """创建差异化学习率优化器"""
    
    # 分离不同类型的参数
    base_params = []
    tau_params = []
    
    for name, param in model.named_parameters():
        if 'tau' in name.lower():
            tau_params.append(param)
            print(f"时间常数参数: {name}")
        else:
            base_params.append(param)
    
    # 创建差异化学习率的优化器
    if len(tau_params) > 0:
        optimizer = torch.optim.Adam([
            {'params': base_params, 'lr': base_lr},
            {'params': tau_params, 'lr': base_lr * 0.5}  # 时间常数使用更小学习率
        ])
        print(f"✅ 差异化学习率优化器:")
        print(f"   - 基础参数: lr = {base_lr}")
        print(f"   - 时间常数: lr = {base_lr * 0.5}")
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=base_lr)
        print(f"✅ 标准优化器: lr = {base_lr}")
    
    return optimizer

def train_optimized_model(model, train_loader, test_loader, model_name, device, num_epochs=30):
    """优化的训练过程"""
    
    print(f"\n🚀 开始训练优化的{model_name}模型")
    
    model = model.to(device)
    
    # 使用差异化学习率优化器
    optimizer = create_optimized_optimizer(model, base_lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    
    best_test_acc1 = 0.0
    best_test_acc5 = 0.0
    train_losses = []
    test_accuracies = []
    
    for epoch in range(num_epochs):
        start_time = time.time()
        
        # 训练阶段
        model.train()
        running_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            targets = targets.to(device)
            inputs = [inp.to(device) for inp in inputs]
            
            optimizer.zero_grad()
            functional.reset_net(model)
            
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            
            # 梯度裁剪防止梯度爆炸
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += targets.size(0)
            train_correct += predicted.eq(targets).sum().item()
            
            if batch_idx % 20 == 0:
                print(f"   Batch {batch_idx}: Loss = {loss.item():.4f}")
        
        # 测试阶段
        model.eval()
        test_correct = 0
        test_total = 0
        test_loss = 0.0
        test_correct_5 = 0
        
        with torch.no_grad():
            for inputs, targets in test_loader:
                targets = targets.to(device)
                inputs = [inp.to(device) for inp in inputs]
                
                functional.reset_net(model)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                
                test_loss += loss.item()
                
                # Top-1准确率
                _, predicted = outputs.max(1)
                test_total += targets.size(0)
                test_correct += predicted.eq(targets).sum().item()
                
                # Top-5准确率
                acc1, acc5 = accuracy(outputs, targets, topk=(1, 5))
                test_correct_5 += acc5 * targets.size(0) / 100.0
        
        train_acc = 100.0 * train_correct / train_total
        test_acc1 = 100.0 * test_correct / test_total
        test_acc5 = 100.0 * test_correct_5 / test_total
        avg_train_loss = running_loss / len(train_loader)
        
        train_losses.append(avg_train_loss)
        test_accuracies.append(test_acc1)
        
        if test_acc1 > best_test_acc1:
            best_test_acc1 = test_acc1
        if test_acc5 > best_test_acc5:
            best_test_acc5 = test_acc5
        
        scheduler.step()
        
        epoch_time = time.time() - start_time
        print(f"轮次 [{epoch+1}/{num_epochs}] ({epoch_time:.1f}s):")
        print(f"  训练损失: {avg_train_loss:.4f}, 训练准确率: {train_acc:.2f}%")
        print(f"  测试准确率: {test_acc1:.2f}% (最佳: {best_test_acc1:.2f}%)")
        print(f"  Top-5准确率: {test_acc5:.2f}% (最佳: {best_test_acc5:.2f}%)")
        print(f"  学习率: {optimizer.param_groups[0]['lr']:.6f}")
        
        # 提前停止
        if epoch > 10 and test_acc1 < 5.0:  # 如果性能太差，提前停止
            print("  ⚠️  性能太差，提前停止训练")
            break
    
    return {
        'best_test_acc1': best_test_acc1,
        'best_test_acc5': best_test_acc5,
        'final_test_acc1': test_acc1,
        'final_test_acc5': test_acc5,
        'train_losses': train_losses,
        'test_accuracies': test_accuracies
    }

# ==================== 主实验函数 ====================

def main():
    """主实验函数"""
    
    print("="*80)
    print("NeuroVPR优化实验 - 基于原始代码应用性能优化")
    print("="*80)
    
    # 设置随机种子
    setup_seed(42)
    
    # 配置GPU
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"✅ 使用GPU: {torch.cuda.get_device_name(0)}")
        torch.cuda.empty_cache()
        torch.backends.cudnn.benchmark = True
    else:
        device = torch.device('cpu')
        print("❌ 使用CPU")
    
    # 实验参数（优化后）
    BATCH_SIZE = 32
    N_CLASS = 25  # 减少类别数量
    NUM_EPOCHS = 25  # 减少训练轮数
    NUM_BRANCHES = 2  # 减少分支数量，降低复杂度
    POSITION_GRANULARITY = 5.0
    DATA_PATH = '/root/autodl-tmp/neurovpr/datasets/'
    EXP_NAMES = ['floor3_v9', 'room_v5']
    RESULTS_PATH = '/root/DH-SNN_reproduce/results'
    
    print(f"\n📋 实验配置:")
    print(f"   数据集: {EXP_NAMES}")
    print(f"   批次大小: {BATCH_SIZE}")
    print(f"   类别数: {N_CLASS}")
    print(f"   训练轮数: {NUM_EPOCHS}")
    print(f"   DH-SNN分支数: {NUM_BRANCHES}")
    print(f"   位置粒度: {POSITION_GRANULARITY}m")
    
    # 加载数据
    print(f"\n📁 加载NeuroVPR数据...")
    
    train_dataset = FixedNeuroVPRDataset(
        data_path=DATA_PATH,
        exp_names=EXP_NAMES,
        batch_size=BATCH_SIZE,
        is_shuffle=True,
        nclass=N_CLASS,
        split_type='train',
        train_ratio=0.7,
        position_granularity=POSITION_GRANULARITY
    )
    
    test_dataset = FixedNeuroVPRDataset(
        data_path=DATA_PATH,
        exp_names=EXP_NAMES,
        batch_size=BATCH_SIZE,
        is_shuffle=False,
        nclass=N_CLASS,
        split_type='test',
        train_ratio=0.7,
        position_granularity=POSITION_GRANULARITY
    )
    
    print(f"✅ 数据加载完成:")
    print(f"   训练批次: {len(train_dataset)}")
    print(f"   测试批次: {len(test_dataset)}")
    
    # 计算输入特征维度
    input_features = 2 * 32 * 43  # 极性 * 高度 * 宽度 (下采样后) = 2752
    print(f"   输入特征维度: {input_features}")
    
    # 创建模型
    print(f"\n🏗️  创建模型...")
    
    # 优化的DH-SNN模型
    dh_snn_model = OptimizedDH_SNN(
        input_dim=input_features,
        hidden_dims=[512],
        output_dim=N_CLASS,
        num_branches=NUM_BRANCHES,
        step_mode='s'
    )
    
    # 优化的Vanilla SNN模型
    vanilla_snn_model = OptimizedVanilla_SNN(
        input_dim=input_features,
        hidden_dims=[512],
        output_dim=N_CLASS,
        step_mode='s'
    )
    
    print(f"✅ 模型创建完成:")
    print(f"   DH-SNN参数数量: {sum(p.numel() for p in dh_snn_model.parameters()):,}")
    print(f"   Vanilla SNN参数数量: {sum(p.numel() for p in vanilla_snn_model.parameters()):,}")
    
    # 开始训练实验
    print(f"\n🚀 开始训练实验...")
    
    # 训练优化的DH-SNN
    print(f"\n{'='*60}")
    print("1. 训练优化的DH-SNN")
    print("="*60)
    
    dh_results = train_optimized_model(
        dh_snn_model, train_dataset, test_dataset, 
        "DH-SNN", device, num_epochs=NUM_EPOCHS
    )
    
    # 训练优化的Vanilla SNN
    print(f"\n{'='*60}")
    print("2. 训练优化的Vanilla SNN")
    print("="*60)
    
    vanilla_results = train_optimized_model(
        vanilla_snn_model, train_dataset, test_dataset,
        "Vanilla SNN", device, num_epochs=NUM_EPOCHS
    )
    
    # 结果对比
    print("\n" + "="*80)
    print("优化实验结果对比")
    print("="*80)
    
    print(f"\n📊 性能对比:")
    print(f"优化DH-SNN:")
    print(f"  - 最佳测试准确率: {dh_results['best_test_acc1']:.2f}%")
    print(f"  - 最终测试准确率: {dh_results['final_test_acc1']:.2f}%")
    print(f"  - 最佳Top-5准确率: {dh_results['best_test_acc5']:.2f}%")
    
    print(f"\n优化Vanilla SNN:")
    print(f"  - 最佳测试准确率: {vanilla_results['best_test_acc1']:.2f}%")
    print(f"  - 最终测试准确率: {vanilla_results['final_test_acc1']:.2f}%")
    print(f"  - 最佳Top-5准确率: {vanilla_results['best_test_acc5']:.2f}%")
    
    # 计算改进
    best_improvement = dh_results['best_test_acc1'] - vanilla_results['best_test_acc1']
    final_improvement = dh_results['final_test_acc1'] - vanilla_results['final_test_acc1']
    
    print(f"\n🎯 性能分析:")
    print(f"DH-SNN vs Vanilla SNN:")
    print(f"  - 最佳准确率差异: {best_improvement:+.2f}%")
    print(f"  - 最终准确率差异: {final_improvement:+.2f}%")
    
    if best_improvement > 0:
        print("✅ DH-SNN优于Vanilla SNN - 符合预期!")
    elif best_improvement > -5:
        print("⚠️  DH-SNN与Vanilla SNN性能相近")
    else:
        print("❌ DH-SNN仍然表现不佳 - 需要进一步优化")
    
    # 保存结果
    results = {
        'experiment_info': {
            'type': 'Optimized NeuroVPR Experiment',
            'framework': 'SpikingJelly',
            'dataset': f'NeuroVPR ({EXP_NAMES})',
            'optimization_applied': [
                'Differentiated learning rates',
                'Short time constants',
                'Reduced model complexity',
                'Temporal step fusion',
                'Gradient clipping'
            ],
            'num_classes': N_CLASS,
            'batch_size': BATCH_SIZE,
            'num_epochs': NUM_EPOCHS,
            'num_branches': NUM_BRANCHES,
            'position_granularity': POSITION_GRANULARITY
        },
        'optimized_dh_snn': dh_results,
        'optimized_vanilla_snn': vanilla_results,
        'improvement': {
            'best_accuracy': best_improvement,
            'final_accuracy': final_improvement
        }
    }
    
    results_file = Path(RESULTS_PATH) / 'neurovpr_optimized_results.json'
    os.makedirs(RESULTS_PATH, exist_ok=True)
    
    # 转换tensor为可序列化格式
    for model_name in ['optimized_dh_snn', 'optimized_vanilla_snn']:
        if 'train_losses' in results[model_name]:
            results[model_name]['train_losses'] = [float(x) for x in results[model_name]['train_losses']]
        if 'test_accuracies' in results[model_name]:
            results[model_name]['test_accuracies'] = [float(x) for x in results[model_name]['test_accuracies']]
    
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n💾 结果已保存到: {results_file}")
    
    return results

if __name__ == "__main__":
    main()
