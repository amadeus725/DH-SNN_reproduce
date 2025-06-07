#!/usr/bin/env python3
"""
NeuroVPR SpikingJelly实验：基于SpikingJelly框架的DH-SNN vs Vanilla SNN对比
================================================================

使用SpikingJelly框架重新实现NeuroVPR实验，对比DH-SNN与Vanilla SNN在
真实DAVIS 346事件相机数据上的视觉位置识别性能。

数据集: https://zenodo.org/records/7827108
论文: Zheng et al. "Dendritic heterogeneity spiking neural networks"

Authors: DH-SNN Reproduction Study
Date: 2025
Framework: SpikingJelly
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

# 添加项目路径
sys.path.append('/root/DH-SNN_reproduce')
sys.path.append('/root/DH-SNN_reproduce/src')
sys.path.append('/root/DH-SNN_reproduce/experiments/neurovpr')

# SpikingJelly imports
from spikingjelly.activation_based import neuron, functional, surrogate, layer, base

# 导入我们的核心DH-SNN实现
from src.core.neurons import DH_LIFNode, ReadoutNeuron, ParametricLIFNode
from src.core.layers import DendriticDenseLayer, ReadoutIntegrator
from src.core.models import DH_SNN, create_dh_snn
from src.core.surrogate import MultiGaussianSurrogate

# 导入数据处理工具（修复后的版本）
from tool_function_bak import Data, setup_seed, accuracy

import torchvision

# 配置GPU使用
if torch.cuda.is_available():
    device = torch.device('cuda')
    print(f"✅ GPU可用: {torch.cuda.get_device_name(0)}")
    print(f"使用设备: {device}")
    print(f"CUDA版本: {torch.version.cuda}")
    print(f"GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    # GPU优化设置
    torch.cuda.empty_cache()
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
else:
    print("❌ GPU不可用，使用CPU")
    device = torch.device('cpu')

# 实验参数（针对NeuroVPR数据集优化）
BATCH_SIZE = 16          # 降低批次大小以适应内存限制
N_CLASS = 100           # NeuroVPR分类数量
LEARNING_RATE = 1e-3    # 学习率
NUM_EPOCHS = 25         # 训练轮数
NUM_ITER = 40           # 每轮迭代次数
NUM_BRANCHES = 4        # DH-SNN分支数量

# 数据路径
DATA_PATH = '/root/autodl-tmp/neurovpr/datasets/'
RESULTS_PATH = '/root/DH-SNN_reproduce/results'

# 数据集配置
TRAIN_EXP_IDX = ['room_v5']     # 训练数据集
TEST_EXP_IDX = ['floor3_v9']    # 测试数据集

# 序列长度参数
SEQ_LEN_APS = 3
SEQ_LEN_DVS = 4
SEQ_LEN_GPS = 3
DVS_EXPAND = 3

def check_spikingjelly_installation():
    """检查SpikingJelly安装和配置"""
    try:
        from spikingjelly import __version__ as sj_version
        print(f"✅ SpikingJelly版本: {sj_version}")
        
        # 测试基本功能
        test_neuron = neuron.LIFNode()
        print("✅ SpikingJelly基本功能正常")
        
        # 测试我们的DH-SNN组件
        test_dh_neuron = DH_LIFNode()
        print("✅ DH-SNN神经元组件正常")
        
        return True
    except Exception as e:
        print(f"❌ SpikingJelly检查失败: {e}")
        return False

def check_data_availability():
    """检查NeuroVPR数据集可用性"""
    print("检查NeuroVPR数据集可用性...")
    
    dataset_path = Path(DATA_PATH)
    if not dataset_path.exists():
        print(f"❌ 数据集路径不存在: {dataset_path}")
        return False
    
    found_datasets = []
    all_datasets = TRAIN_EXP_IDX + TEST_EXP_IDX
    
    for dataset_name in all_datasets:
        dataset_path_full = Path(DATA_PATH) / dataset_name
        if dataset_path_full.exists():
            found_datasets.append(dataset_name)
            print(f"✅ 找到数据集 {dataset_name}")
            
            # 检查关键文件
            dvs_7ms = dataset_path_full / "dvs_7ms_3seq"
            position_file = dataset_path_full / "position.txt"
            
            if dvs_7ms.exists():
                dvs_count = len(list(dvs_7ms.glob("*")))
                print(f"   - DVS序列: {dvs_count} 文件")
            else:
                print(f"   - ⚠️  DVS序列目录缺失")
                
            if position_file.exists():
                print(f"   - 位置文件: ✅")
            else:
                print(f"   - ⚠️  位置文件缺失")
        else:
            print(f"❌ 数据集 {dataset_name} 未找到")
    
    if len(found_datasets) >= 2:
        print(f"\n✅ 数据集检查通过: 找到 {len(found_datasets)} 个数据集")
        return True
    else:
        print(f"\n❌ 数据集检查失败: 仅找到 {len(found_datasets)} 个数据集")
        return False

def create_data_loaders():
    """创建数据加载器"""
    print("创建数据加载器...")
    
    # 标准化变换
    normalize = torchvision.transforms.Normalize(
        mean=[0.3537, 0.3537, 0.3537],
        std=[0.3466, 0.3466, 0.3466]
    )
    
    try:
        train_loader = Data(
            data_path=DATA_PATH, 
            batch_size=BATCH_SIZE, 
            exp_idx=TRAIN_EXP_IDX, 
            is_shuffle=True,
            normalize=normalize, 
            nclass=N_CLASS,
            seq_len_aps=SEQ_LEN_APS, 
            seq_len_dvs=SEQ_LEN_DVS, 
            seq_len_gps=SEQ_LEN_GPS,
            dvs_expand=DVS_EXPAND
        )
        
        test_loader = Data(
            data_path=DATA_PATH, 
            batch_size=BATCH_SIZE, 
            exp_idx=TEST_EXP_IDX, 
            is_shuffle=False,
            normalize=normalize, 
            nclass=N_CLASS,
            seq_len_aps=SEQ_LEN_APS, 
            seq_len_dvs=SEQ_LEN_DVS, 
            seq_len_gps=SEQ_LEN_GPS,
            dvs_expand=DVS_EXPAND
        )
        
        print(f"✅ 数据加载器创建成功")
        print(f"   - 训练批次: {len(train_loader)}")
        print(f"   - 测试批次: {len(test_loader)}")
        
        return train_loader, test_loader
        
    except Exception as e:
        print(f"❌ 创建数据加载器时出错: {e}")
        return None, None

class NeuroVPR_DH_SNN(nn.Module):
    """基于SpikingJelly的NeuroVPR DH-SNN模型"""
    
    def __init__(self, input_dim=12, hidden_dims=[256, 256], output_dim=N_CLASS, 
                 num_branches=NUM_BRANCHES, step_mode='s'):
        super(NeuroVPR_DH_SNN, self).__init__()
        
        self.step_mode = step_mode
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # 使用我们的DH-SNN核心架构
        self.dh_snn = create_dh_snn(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            output_dim=output_dim,
            num_branches=num_branches,
            v_threshold=1.0,
            tau_m_init=(0, 4),
            tau_n_init=(0, 4),
            tau_initializer='uniform',
            sparsity=1.0/num_branches,  # 按分支数设置稀疏度
            mask_share=1,
            bias=True,
            surrogate_function=MultiGaussianSurrogate(),
            reset_mode='soft',
            step_mode=step_mode
        )
        
        # 时间步初始化
        self.T = SEQ_LEN_DVS * 3  # 适应3序列DVS数据
        
    def forward(self, inp):
        """前向传播"""
        # 获取DVS输入（索引2）
        dvs_inp = inp[2]  # shape: [batch, seq_len*3, 2, 32, 43]
        
        # 调整输入维度以适应网络
        batch_size = dvs_inp.shape[0]
        
        # 将DVS数据重塑为合适的输入格式
        # [batch, seq_len*3, 2, 32, 43] -> [T, batch, input_features]
        dvs_reshaped = dvs_inp.view(batch_size, self.T, -1)  # [batch, T, features]
        
        # 转置为时间步优先：[T, batch, features]
        dvs_input = dvs_reshaped.transpose(0, 1)
        
        # 通过DH-SNN网络
        if self.step_mode == 's':
            # 单步模式：逐时间步处理
            outputs = []
            for t in range(self.T):
                output = self.dh_snn(dvs_input[t])
                outputs.append(output)
            
            # 取最后一个时间步的输出
            final_output = outputs[-1]
        else:
            # 多步模式
            final_output = self.dh_snn(dvs_input)
            if isinstance(final_output, list):
                final_output = final_output[-1]
        
        return final_output

class NeuroVPR_Vanilla_SNN(nn.Module):
    """基于SpikingJelly的NeuroVPR Vanilla SNN模型"""
    
    def __init__(self, input_dim=12, hidden_dims=[256, 256], output_dim=N_CLASS, step_mode='s'):
        super(NeuroVPR_Vanilla_SNN, self).__init__()
        
        self.step_mode = step_mode
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # 构建Vanilla SNN网络
        self.layers = nn.ModuleList()
        
        # 输入层
        current_dim = input_dim
        for hidden_dim in hidden_dims:
            # 线性层
            self.layers.append(layer.Linear(current_dim, hidden_dim))
            # LIF神经元
            self.layers.append(neuron.LIFNode(
                tau=2.0, 
                v_threshold=1.0,
                surrogate_function=surrogate.ATan(),
                step_mode=step_mode
            ))
            current_dim = hidden_dim
        
        # 输出层
        self.layers.append(layer.Linear(current_dim, output_dim))
        self.layers.append(neuron.LIFNode(
            tau=2.0,
            v_threshold=1.0, 
            surrogate_function=surrogate.ATan(),
            step_mode=step_mode
        ))
        
        # 时间步初始化
        self.T = SEQ_LEN_DVS * 3
        
    def forward(self, inp):
        """前向传播"""
        # 获取DVS输入（索引2）
        dvs_inp = inp[2]
        
        # 调整输入维度
        batch_size = dvs_inp.shape[0]
        dvs_reshaped = dvs_inp.view(batch_size, self.T, -1)
        dvs_input = dvs_reshaped.transpose(0, 1)  # [T, batch, features]
        
        if self.step_mode == 's':
            # 单步模式
            outputs = []
            for t in range(self.T):
                x = dvs_input[t]
                for layer in self.layers:
                    x = layer(x)
                outputs.append(x)
            
            final_output = outputs[-1]
        else:
            # 多步模式
            x = dvs_input
            for layer in self.layers:
                x = layer(x)
            final_output = x[-1] if isinstance(x, list) else x
        
        return final_output

def train_model(model, train_loader, test_loader, model_name, num_epochs=NUM_EPOCHS):
    """训练模型并返回结果"""
    print(f"\n{'='*60}")
    print(f"训练 {model_name} 模型 (SpikingJelly框架)")
    print(f"{'='*60}")
    
    # 优化器和调度器
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    lr_schedule = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    criterion = nn.CrossEntropyLoss()
    
    # 训练指标
    train_losses = []
    test_accuracies = []
    best_test_acc1 = 0.0
    best_test_acc5 = 0.0
    
    train_iters = iter(train_loader)
    
    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        running_loss = 0.0
        train_acc1 = 0.0
        train_acc5 = 0.0
        
        for iter_idx in range(NUM_ITER):
            try:
                inputs, target = next(train_iters)
            except StopIteration:
                train_iters = iter(train_loader)
                inputs, target = next(train_iters)
            
            # 重置神经元状态
            functional.reset_net(model)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, target.to(device))
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.cpu().item()
            
            # 计算训练准确率
            acc1, acc5, _ = accuracy(outputs.cpu(), target, topk=(1, 5, 10))
            train_acc1 += acc1 / len(outputs)
            train_acc5 += acc5 / len(outputs)
        
        lr_schedule.step()
        
        # 测试阶段
        model.eval()
        test_loss = 0.0
        test_acc1 = 0.0
        test_acc5 = 0.0
        test_batches = 0
        
        with torch.no_grad():
            for batch_idx, (inputs, target) in enumerate(test_loader):
                functional.reset_net(model)
                outputs = model(inputs)
                loss = criterion(outputs.cpu(), target)
                test_loss += loss.item()
                
                acc1, acc5, _ = accuracy(outputs.cpu(), target, topk=(1, 5, 10))
                test_acc1 += acc1 / len(outputs)
                test_acc5 += acc5 / len(outputs)
                test_batches += 1
        
        # 平均指标
        avg_train_loss = running_loss / NUM_ITER
        avg_train_acc1 = train_acc1 / NUM_ITER
        avg_train_acc5 = train_acc5 / NUM_ITER
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
            print(f'轮次 [{epoch+1}/{num_epochs}]')
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

def main():
    """主实验函数"""
    print("="*80)
    print("NeuroVPR SpikingJelly实验: DH-SNN vs Vanilla SNN")
    print("="*80)
    
    # 环境检查
    setup_seed(42)
    os.makedirs(RESULTS_PATH, exist_ok=True)
    
    # 检查SpikingJelly
    if not check_spikingjelly_installation():
        print("❌ SpikingJelly环境检查失败！")
        return None
    
    # 检查数据集
    if not check_data_availability():
        print("\n❌ NeuroVPR数据集未找到！")
        print("请从以下地址下载数据集: https://zenodo.org/records/7827108")
        print(f"并将其放置在: {DATA_PATH}")
        return None
    
    # 创建数据加载器
    train_loader, test_loader = create_data_loaders()
    if train_loader is None or test_loader is None:
        print("❌ 创建数据加载器失败。请检查数据集结构。")
        return None
    
    # 计算输入维度（基于DVS数据）
    sample_data = next(iter(train_loader))
    dvs_sample = sample_data[0][2]  # 获取DVS样本
    input_features = dvs_sample.view(dvs_sample.shape[0], -1).shape[1] // (SEQ_LEN_DVS * 3)
    print(f"计算得到的输入特征维度: {input_features}")
    
    # 初始化模型
    print(f"\n在 {device} 上初始化模型...")
    
    # DH-SNN模型
    dh_snn_model = NeuroVPR_DH_SNN(
        input_dim=input_features,
        hidden_dims=[256, 256],
        output_dim=N_CLASS,
        num_branches=NUM_BRANCHES,
        step_mode='s'
    ).to(device)
    
    # Vanilla SNN模型
    vanilla_snn_model = NeuroVPR_Vanilla_SNN(
        input_dim=input_features,
        hidden_dims=[256, 256], 
        output_dim=N_CLASS,
        step_mode='s'
    ).to(device)
    
    print(f"DH-SNN参数数量: {sum(p.numel() for p in dh_snn_model.parameters()):,}")
    print(f"Vanilla SNN参数数量: {sum(p.numel() for p in vanilla_snn_model.parameters()):,}")
    
    # 验证模型在GPU上
    if torch.cuda.is_available():
        print(f"✅ DH-SNN模型设备: {next(dh_snn_model.parameters()).device}")
        print(f"✅ Vanilla SNN模型设备: {next(vanilla_snn_model.parameters()).device}")
    
    # 开始训练实验
    print(f"\n开始训练实验...")
    
    # 训练DH-SNN
    dh_results = train_model(dh_snn_model, train_loader, test_loader, "DH-SNN", num_epochs=NUM_EPOCHS)
    
    # 训练Vanilla SNN
    vanilla_results = train_model(vanilla_snn_model, train_loader, test_loader, "Vanilla SNN", num_epochs=NUM_EPOCHS)
    
    # 结果对比
    print("\n" + "="*80)
    print("最终结果对比 (SpikingJelly框架)")
    print("="*80)
    
    print(f"DH-SNN结果:")
    print(f"  最佳测试准确率: {dh_results['best_test_acc1']:.2f}%")
    print(f"  最终测试准确率: {dh_results['final_test_acc1']:.2f}%")
    print(f"  最佳Top-5准确率: {dh_results['best_test_acc5']:.2f}%")
    
    print(f"\nVanilla SNN结果:")
    print(f"  最佳测试准确率: {vanilla_results['best_test_acc1']:.2f}%")
    print(f"  最终测试准确率: {vanilla_results['final_test_acc1']:.2f}%")
    print(f"  最佳Top-5准确率: {vanilla_results['best_test_acc5']:.2f}%")
    
    # 计算改进
    best_improvement = dh_results['best_test_acc1'] - vanilla_results['best_test_acc1']
    final_improvement = dh_results['final_test_acc1'] - vanilla_results['final_test_acc1']
    best_relative = (dh_results['best_test_acc1'] / vanilla_results['best_test_acc1'] - 1) * 100
    final_relative = (dh_results['final_test_acc1'] / vanilla_results['final_test_acc1'] - 1) * 100
    
    print(f"\n性能改进:")
    print(f"  最佳准确率: +{best_improvement:.2f}% (相对: +{best_relative:.1f}%)")
    print(f"  最终准确率: +{final_improvement:.2f}% (相对: +{final_relative:.1f}%)")
    
    # 保存详细结果
    all_results = {
        'experiment_info': {
            'framework': 'SpikingJelly',
            'dataset': 'NeuroVPR (Real Data)',
            'dataset_source': 'https://zenodo.org/records/7827108',
            'train_experiments': TRAIN_EXP_IDX,
            'test_experiments': TEST_EXP_IDX,
            'num_classes': N_CLASS,
            'batch_size': BATCH_SIZE,
            'num_epochs': NUM_EPOCHS,
            'num_iter_per_epoch': NUM_ITER,
            'num_branches': NUM_BRANCHES,
            'device': str(device),
            'gpu_name': torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'
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
    
    # 保存结果
    results_file = os.path.join(RESULTS_PATH, 'neurovpr_spikingjelly_results.json')
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    # 保存模型
    torch.save(dh_snn_model.state_dict(), os.path.join(RESULTS_PATH, 'neurovpr_dh_snn_spikingjelly.pth'))
    torch.save(vanilla_snn_model.state_dict(), os.path.join(RESULTS_PATH, 'neurovpr_vanilla_snn_spikingjelly.pth'))
    
    print(f"\n✅ 结果保存至: {results_file}")
    print(f"✅ 模型保存至: {RESULTS_PATH}")
    
    return all_results

if __name__ == "__main__":
    results = main()
