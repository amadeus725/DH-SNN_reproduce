#!/usr/bin/env python3
"""
NeuroVPR SpikingJelly实验（修复版）：基于SpikingJelly框架的DH-SNN vs Vanilla SNN对比
================================================================

修复问题：
1. 位置量化不一致
2. 数据归一化和预处理
3. 类别分布不均匀
4. 学习率和模型参数优化

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

# ==================== 本地工具函数 ====================

def setup_seed(seed):
    """设置随机种子"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def accuracy(output, target, topk=(1,)):
    """计算Top-K准确率"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        
        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size).item())
        return res

class FixedNeuroVPRDataset:
    """修复版NeuroVPR数据集加载器"""
    
    def __init__(self, data_path, exp_names, batch_size=16, is_shuffle=True, nclass=50, 
                 split_type='train', train_ratio=0.7, position_granularity=5.0):
        self.data_path = Path(data_path)
        self.exp_names = exp_names
        self.batch_size = batch_size
        self.is_shuffle = is_shuffle
        self.nclass = nclass
        self.split_type = split_type
        self.train_ratio = train_ratio
        self.position_granularity = position_granularity  # 位置量化粒度（米）
        
        # 加载数据
        self.data_samples = []
        self.labels = []
        
        for exp_name in exp_names:
            exp_path = self.data_path / exp_name
            if not exp_path.exists():
                print(f"警告: 数据集 {exp_name} 不存在于 {exp_path}")
                continue
                
            self._load_experiment_data(exp_path)
        
        print(f"原始数据加载完成: {len(self.data_samples)} 个样本")
        
        # 分析类别分布
        self._analyze_class_distribution()
        
        # 执行训练/测试分割
        self._perform_train_test_split()
        
        print(f"数据分割完成({split_type}): {len(self.data_samples)} 个样本")
        
        # 创建批次索引
        self.indices = list(range(len(self.data_samples)))
        if self.is_shuffle:
            random.shuffle(self.indices)
        
        self.batch_indices = []
        for i in range(0, len(self.indices), batch_size):
            batch = self.indices[i:i+batch_size]
            if len(batch) == batch_size:  # 只保留完整批次
                self.batch_indices.append(batch)
    
    def _analyze_class_distribution(self):
        """分析类别分布"""
        label_counts = Counter(self.labels)
        print(f"类别分析:")
        print(f"  总类别数: {len(label_counts)}")
        print(f"  样本数范围: {min(label_counts.values())} - {max(label_counts.values())}")
        print(f"  平均每类样本数: {np.mean(list(label_counts.values())):.1f}")
        
        # 显示前10个类别
        top_classes = label_counts.most_common(10)
        print(f"  前10个类别样本数: {[count for _, count in top_classes]}")
        
        # 过滤样本数过少的类别
        min_samples_per_class = 10
        valid_classes = set([label for label, count in label_counts.items() if count >= min_samples_per_class])
        
        if len(valid_classes) < len(label_counts):
            print(f"  过滤掉样本数少于{min_samples_per_class}的类别: {len(label_counts) - len(valid_classes)}个")
            
            # 重新过滤数据
            filtered_samples = []
            filtered_labels = []
            for sample, label in zip(self.data_samples, self.labels):
                if label in valid_classes:
                    filtered_samples.append(sample)
                    filtered_labels.append(label)
            
            self.data_samples = filtered_samples
            self.labels = filtered_labels
            
            print(f"  过滤后保留: {len(self.data_samples)} 个样本, {len(valid_classes)} 个类别")
    
    def _perform_train_test_split(self):
        """执行训练/测试数据分割 - 按类别分层"""
        if len(self.data_samples) == 0:
            return
        
        # 设置随机种子确保可重现性
        random.seed(42)
        np.random.seed(42)
        
        # 按类别分组
        class_samples = {}
        for i, label in enumerate(self.labels):
            if label not in class_samples:
                class_samples[label] = []
            class_samples[label].append(i)
        
        # 对每个类别进行分层采样
        train_indices = []
        test_indices = []
        
        for label, indices in class_samples.items():
            random.shuffle(indices)
            train_size = int(len(indices) * self.train_ratio)
            
            train_indices.extend(indices[:train_size])
            test_indices.extend(indices[train_size:])
        
        # 根据分割类型选择索引
        if self.split_type == 'train':
            selected_indices = train_indices
        else:  # test
            selected_indices = test_indices
        
        # 根据选中的索引重新构建数据
        selected_samples = [self.data_samples[i] for i in selected_indices]
        selected_labels = [self.labels[i] for i in selected_indices]
        
        self.data_samples = selected_samples
        self.labels = selected_labels
        
        print(f"执行分层{self.split_type}分割: 总样本{len(self.data_samples)}")
    
    def _load_experiment_data(self, exp_path):
        """加载单个实验的数据"""
        dvs_path = exp_path / "dvs_7ms_3seq"
        position_file = exp_path / "position.txt"
        
        if not dvs_path.exists():
            print(f"警告: DVS数据路径不存在: {dvs_path}")
            return
        
        if not position_file.exists():
            print(f"警告: 位置文件不存在: {position_file}")
            return
        
        # 读取所有DVS文件时间戳
        dvs_files = sorted(list(dvs_path.glob("*.npy")))
        dvs_timestamps = []
        dvs_file_map = {}
        
        for dvs_file in dvs_files:
            try:
                filename_base = dvs_file.stem
                timestamp = float(filename_base)
                dvs_timestamps.append(timestamp)
                dvs_file_map[timestamp] = dvs_file
            except (ValueError, IndexError):
                continue
        
        print(f"找到 {len(dvs_timestamps)} 个有效DVS文件")
        
        # 读取位置标签
        position_data = []
        with open(position_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 4:
                    try:
                        timestamp = float(parts[0])
                        x, y = float(parts[1]), float(parts[2])
                        position_data.append((timestamp, x, y))
                    except (ValueError, IndexError):
                        continue
        
        print(f"加载 {len(position_data)} 个位置标签")
        
        # 获取时间范围重叠区域
        if not position_data or not dvs_timestamps:
            print("警告: 位置数据或DVS数据为空")
            return
            
        pos_times = [p[0] for p in position_data]
        pos_min, pos_max = min(pos_times), max(pos_times)
        dvs_min, dvs_max = min(dvs_timestamps), max(dvs_timestamps)
        
        # 计算重叠时间范围
        overlap_start = max(pos_min, dvs_min)
        overlap_end = min(pos_max, dvs_max)
        
        print(f"时间重叠范围: {overlap_start:.2f} 到 {overlap_end:.2f} ({overlap_end-overlap_start:.2f}秒)")
        
        # 过滤到重叠范围的数据
        dvs_filtered = [(t, dvs_file_map[t]) for t in dvs_timestamps if overlap_start <= t <= overlap_end]
        pos_filtered = [(t, x, y) for t, x, y in position_data if overlap_start <= t <= overlap_end]
        
        print(f"过滤后: {len(dvs_filtered)} DVS文件, {len(pos_filtered)} 位置标签")
        
        # 创建位置到类别的映射 - 使用固定的粒度
        position_list = []
        position_to_class = {}
        
        for _, x, y in pos_filtered:
            # 使用固定的位置量化粒度
            position_key = (round(x / self.position_granularity) * self.position_granularity, 
                          round(y / self.position_granularity) * self.position_granularity)
            if position_key not in position_to_class:
                class_id = len(position_list)
                position_to_class[position_key] = class_id
                position_list.append(position_key)
        
        print(f"创建 {len(position_list)} 个位置类别（粒度={self.position_granularity}m）")
        
        # 匹配DVS文件和位置标签
        loaded_count = 0
        tolerance = 5.0  # 减少时间戳匹配容差
        
        for dvs_timestamp, dvs_file in dvs_filtered:
            # 寻找最接近的位置时间戳
            best_match = None
            min_diff = float('inf')
            
            for pos_timestamp, x, y in pos_filtered:
                diff = abs(dvs_timestamp - pos_timestamp)
                if diff < min_diff:
                    min_diff = diff
                    if diff <= tolerance:
                        best_match = (x, y, min_diff)
            
            if best_match is not None:
                x, y, time_diff = best_match
                # 使用相同的量化粒度
                position_key = (round(x / self.position_granularity) * self.position_granularity, 
                              round(y / self.position_granularity) * self.position_granularity)
                
                if position_key in position_to_class:
                    try:
                        # 加载DVS数据
                        dvs_data = torch.load(dvs_file, map_location='cpu')
                        
                        # 确保数据是tensor格式
                        if not isinstance(dvs_data, torch.Tensor):
                            if isinstance(dvs_data, np.ndarray):
                                dvs_data = torch.from_numpy(dvs_data)
                            else:
                                continue
                        
                        # 检查数据形状 [2, 3, 260, 346]
                        if len(dvs_data.shape) == 4 and dvs_data.shape[0] == 2 and dvs_data.shape[1] == 3:
                            # 转换为 [3, 2, 260, 346] 时间优先
                            dvs_data = dvs_data.permute(1, 0, 2, 3)  # [3, 2, 260, 346]
                            
                            # 下采样并归一化
                            dvs_data_flat = dvs_data.contiguous().view(-1, 260, 346)
                            dvs_data = F.avg_pool2d(dvs_data_flat, kernel_size=8, stride=8)
                            dvs_data = dvs_data.view(3, 2, dvs_data.shape[-2], dvs_data.shape[-1])
                            
                            # 数据归一化和增强
                            dvs_data = dvs_data.float()
                            # 增强稀疏事件数据的对比度
                            dvs_data = torch.clamp(dvs_data * 2.0, 0, 1)  # 增强对比度
                            
                            self.data_samples.append({
                                'dvs': dvs_data,
                                'file_path': str(dvs_file)
                            })
                            self.labels.append(position_to_class[position_key])
                            loaded_count += 1
                            
                            # 限制样本数量
                            if loaded_count >= MAX_SAMPLES_PER_DATASET:
                                break
                                    
                    except Exception as e:
                        continue
        
        print(f"成功加载 {loaded_count} 个样本")
    
    def __len__(self):
        return len(self.batch_indices)
    
    def __iter__(self):
        if self.is_shuffle:
            random.shuffle(self.batch_indices)
        
        for batch_idx in self.batch_indices:
            batch_dvs = []
            batch_labels = []
            
            for idx in batch_idx:
                dvs_data = self.data_samples[idx]['dvs']
                label = self.labels[idx]
                
                batch_dvs.append(dvs_data)
                batch_labels.append(label)
            
            # 转换为张量
            dvs_tensor = torch.stack(batch_dvs)  # [batch, 3, 2, 32, 43]
            labels_tensor = torch.tensor(batch_labels, dtype=torch.long)
            
            # 返回格式: ([aps, gps, dvs], labels) 与原始格式兼容
            dummy_aps = torch.zeros(len(batch_dvs), 3, 64)
            dummy_gps = torch.zeros(len(batch_dvs), 3, 3)
            
            yield ([dummy_aps, dummy_gps, dvs_tensor], labels_tensor)

# ==================== 配置参数 ====================

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

# 实验参数（优化后）
BATCH_SIZE = 32  # 增加批次大小
N_CLASS = 50     # 减少类别数量，提高每类样本数
LEARNING_RATE = 2e-3  # 增加学习率
NUM_EPOCHS = 30       # 减少训练轮数，快速验证
NUM_ITER = None       # 自适应迭代数
NUM_BRANCHES = 4
MAX_SAMPLES_PER_DATASET = 3000  # 减少样本数量以提高训练速度
POSITION_GRANULARITY = 5.0      # 5米位置粒度

# 数据路径
DATA_PATH = '/root/autodl-tmp/neurovpr/datasets/'
RESULTS_PATH = '/root/DH-SNN_reproduce/results'

# 数据集配置
DATASET_NAME = 'floor3_v9'
TRAIN_TEST_SPLIT = 0.7

# 实验配置
TRAIN_EXP_IDX = [DATASET_NAME]
TEST_EXP_IDX = [DATASET_NAME]

# ==================== 数据检查和加载 ====================

def check_data_availability():
    """检查NeuroVPR数据集可用性"""
    print("检查NeuroVPR数据集可用性...")
    
    dataset_path = Path(DATA_PATH)
    if not dataset_path.exists():
        print(f"❌ 数据集路径不存在: {dataset_path}")
        return False
    
    dataset_path_full = dataset_path / DATASET_NAME
    if dataset_path_full.exists():
        print(f"✅ 找到数据集 {DATASET_NAME}")
        
        dvs_7ms = dataset_path_full / "dvs_7ms_3seq"
        position_file = dataset_path_full / "position.txt"
        
        if dvs_7ms.exists():
            dvs_count = len(list(dvs_7ms.glob("*.npy")))
            print(f"   - DVS序列: {dvs_count} 文件")
        else:
            print(f"   - ⚠️  DVS序列目录缺失")
            return False
            
        if position_file.exists():
            print(f"   - 位置文件: ✅")
        else:
            print(f"   - ⚠️  位置文件缺失")
            return False
            
        print(f"\n✅ 数据集检查通过")
        return True
    else:
        print(f"❌ 数据集 {DATASET_NAME} 未找到")
        return False

def create_data_loaders():
    """创建数据加载器"""
    print("创建修复版数据加载器...")
    
    try:
        # 训练集
        train_loader = FixedNeuroVPRDataset(
            data_path=DATA_PATH,
            exp_names=TRAIN_EXP_IDX,
            batch_size=BATCH_SIZE,
            is_shuffle=True,
            nclass=N_CLASS,
            split_type='train',
            train_ratio=TRAIN_TEST_SPLIT,
            position_granularity=POSITION_GRANULARITY
        )
        
        # 测试集
        test_loader = FixedNeuroVPRDataset(
            data_path=DATA_PATH,
            exp_names=TEST_EXP_IDX,
            batch_size=BATCH_SIZE,
            is_shuffle=False,
            nclass=N_CLASS,
            split_type='test',
            train_ratio=TRAIN_TEST_SPLIT,
            position_granularity=POSITION_GRANULARITY
        )
        
        print(f"✅ 数据加载器创建成功")
        print(f"   - 训练批次: {len(train_loader)} (约{len(train_loader)*BATCH_SIZE}样本)")
        print(f"   - 测试批次: {len(test_loader)} (约{len(test_loader)*BATCH_SIZE}样本)")
        
        return train_loader, test_loader
        
    except Exception as e:
        print(f"❌ 创建数据加载器时出错: {e}")
        traceback.print_exc()
        return None, None

# ==================== 简化模型定义 ====================

class SimpleNeuroVPR_DH_SNN(nn.Module):
    """简化的NeuroVPR DH-SNN模型"""
    
    def __init__(self, input_dim=2752, hidden_dims=[512], output_dim=N_CLASS, 
                 num_branches=NUM_BRANCHES, step_mode='s'):
        super(SimpleNeuroVPR_DH_SNN, self).__init__()
        
        self.step_mode = step_mode
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.T = 3
        
        # 简化的DH-SNN配置
        dh_snn_config = {
            'input_dim': input_dim,
            'hidden_dims': hidden_dims,
            'output_dim': output_dim,
            'num_branches': num_branches,
            'v_threshold': 1.0,
            'tau_m_init': (1, 3),
            'tau_n_init': (1, 3),
            'tau_initializer': 'uniform',
            'sparsity': 1.0/num_branches,
            'mask_share': 1,
            'bias': True,
            'surrogate_function': MultiGaussianSurrogate(),
            'reset_mode': 'soft',
            'step_mode': step_mode
        }
        
        self.dh_snn = create_dh_snn(dh_snn_config)
        
    def forward(self, inp):
        dvs_inp = inp[2]  # [batch, 3, 2, 32, 43]
        
        batch_size = dvs_inp.shape[0]
        dvs_reshaped = dvs_inp.view(batch_size, self.T, -1)
        dvs_input = dvs_reshaped.transpose(0, 1)  # [3, batch, features]
        
        if self.step_mode == 's':
            outputs = []
            for t in range(self.T):
                output = self.dh_snn(dvs_input[t])
                outputs.append(output)
            final_output = outputs[-1]
        else:
            final_output = self.dh_snn(dvs_input)
            if isinstance(final_output, list):
                final_output = final_output[-1]
        
        return final_output

class SimpleNeuroVPR_Vanilla_SNN(nn.Module):
    """简化的NeuroVPR Vanilla SNN模型"""
    
    def __init__(self, input_dim=2752, hidden_dims=[512], output_dim=N_CLASS, step_mode='s'):
        super(SimpleNeuroVPR_Vanilla_SNN, self).__init__()
        
        self.step_mode = step_mode
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.T = 3
        
        # 构建简化网络
        self.layers = nn.ModuleList()
        
        current_dim = input_dim
        for hidden_dim in hidden_dims:
            self.layers.append(layer.Linear(current_dim, hidden_dim))
            self.layers.append(neuron.LIFNode(
                tau=2.0, 
                v_threshold=1.0,
                surrogate_function=surrogate.ATan(),
                step_mode=step_mode
            ))
            current_dim = hidden_dim
        
        # 输出层
        self.layers.append(layer.Linear(current_dim, output_dim))
        
    def forward(self, inp):
        dvs_inp = inp[2]  # [batch, 3, 2, 32, 43]
        
        batch_size = dvs_inp.shape[0]
        dvs_reshaped = dvs_inp.view(batch_size, self.T, -1)
        dvs_input = dvs_reshaped.transpose(0, 1)  # [3, batch, features]
        
        if self.step_mode == 's':
            outputs = []
            for t in range(self.T):
                x = dvs_input[t]
                for layer in self.layers:
                    x = layer(x)
                outputs.append(x)
            final_output = outputs[-1]
        else:
            x = dvs_input
            for layer in self.layers:
                x = layer(x)
            final_output = x[-1] if isinstance(x, list) else x
        
        return final_output

# ==================== 训练函数 ====================

def train_model(model, train_loader, test_loader, model_name, num_epochs=NUM_EPOCHS):
    """训练模型并返回结果"""
    print(f"\n{'='*60}")
    print(f"训练 {model_name} 模型 (修复版)")
    print(f"{'='*60}")
    
    # 优化器和调度器
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    lr_schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
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
        
        for inputs, target in train_loader:
            # 移动到设备
            target = target.to(device)
            inputs = [inp.to(device) for inp in inputs]
            
            # 重置神经元状态
            functional.reset_net(model)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, target)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # 梯度剪裁
            optimizer.step()
            
            running_loss += loss.item()
            
            # 计算训练准确率
            acc1, acc5 = accuracy(outputs.cpu(), target.cpu(), topk=(1, 5))
            train_acc1 += acc1
            train_acc5 += acc5
            train_batches += 1
        
        lr_schedule.step()
        
        # 测试阶段
        model.eval()
        test_loss = 0.0
        test_acc1 = 0.0
        test_acc5 = 0.0
        test_batches = 0
        
        with torch.no_grad():
            for inputs, target in test_loader:
                target = target.to(device)
                inputs = [inp.to(device) for inp in inputs]
                
                functional.reset_net(model)
                outputs = model(inputs)
                loss = criterion(outputs, target)
                test_loss += loss.item()
                
                acc1, acc5 = accuracy(outputs.cpu(), target.cpu(), topk=(1, 5))
                test_acc1 += acc1
                test_acc5 += acc5
                test_batches += 1
        
        # 平均指标
        if train_batches == 0:
            print("⚠️  警告: 没有训练数据")
            continue
            
        avg_train_loss = running_loss / train_batches
        avg_train_acc1 = train_acc1 / train_batches
        avg_train_acc5 = train_acc5 / train_batches
        avg_test_loss = test_loss / test_batches if test_batches > 0 else 0
        avg_test_acc1 = test_acc1 / test_batches if test_batches > 0 else 0
        avg_test_acc5 = test_acc5 / test_batches if test_batches > 0 else 0
        
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
            print(f'  学习率: {optimizer.param_groups[0]["lr"]:.6f}')
    
    return {
        'train_losses': train_losses,
        'test_accuracies': test_accuracies,
        'best_test_acc1': best_test_acc1,
        'best_test_acc5': best_test_acc5,
        'final_test_acc1': avg_test_acc1,
        'final_test_acc5': avg_test_acc5
    }

# ==================== 主函数 ====================

def main():
    """主实验函数"""
    print("="*80)
    print("NeuroVPR SpikingJelly实验: DH-SNN vs Vanilla SNN (修复版)")
    print("="*80)
    
    # 环境检查
    setup_seed(42)
    os.makedirs(RESULTS_PATH, exist_ok=True)
    
    # 检查SpikingJelly
    try:
        test_neuron = neuron.LIFNode()
        print("✅ SpikingJelly基本功能正常")
        
        test_dh_neuron = DH_LIFNode()
        print("✅ DH-SNN神经元组件正常")
        
        print("✅ SpikingJelly环境检查通过")
    except Exception as e:
        print(f"❌ SpikingJelly检查失败: {e}")
        return None
    
    # 检查数据集
    if not check_data_availability():
        return None
    
    # 创建数据加载器
    train_loader, test_loader = create_data_loaders()
    if train_loader is None or test_loader is None:
        return None
    
    if len(train_loader) == 0 or len(test_loader) == 0:
        print("❌ 没有成功加载任何数据样本")
        return None
    
    # 计算输入维度
    input_features = 2 * 32 * 43
    print(f"计算得到的输入特征维度: {input_features}")
    
    # 初始化模型
    print(f"\n在 {device} 上初始化简化模型...")
    
    # DH-SNN模型
    dh_snn_model = SimpleNeuroVPR_DH_SNN(
        input_dim=input_features,
        hidden_dims=[512],  # 简化网络结构
        output_dim=N_CLASS,
        num_branches=NUM_BRANCHES,
        step_mode='s'
    ).to(device)
    
    # Vanilla SNN模型
    vanilla_snn_model = SimpleNeuroVPR_Vanilla_SNN(
        input_dim=input_features,
        hidden_dims=[512],
        output_dim=N_CLASS,
        step_mode='s'
    ).to(device)
    
    print(f"DH-SNN参数数量: {sum(p.numel() for p in dh_snn_model.parameters()):,}")
    print(f"Vanilla SNN参数数量: {sum(p.numel() for p in vanilla_snn_model.parameters()):,}")
    
    # 开始训练实验
    print(f"\n开始训练实验...")
    
    # 训练DH-SNN
    dh_results = train_model(dh_snn_model, train_loader, test_loader, "DH-SNN", num_epochs=NUM_EPOCHS)
    
    # 训练Vanilla SNN
    vanilla_results = train_model(vanilla_snn_model, train_loader, test_loader, "Vanilla SNN", num_epochs=NUM_EPOCHS)
    
    # 结果对比
    print("\n" + "="*80)
    print("最终结果对比 (修复版)")
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
    
    print(f"\n性能改进:")
    print(f"  最佳准确率: +{best_improvement:.2f}%")
    print(f"  最终准确率: +{final_improvement:.2f}%")
    
    # 保存结果
    results = {
        'experiment_info': {
            'framework': 'SpikingJelly',
            'dataset': f'NeuroVPR ({DATASET_NAME})',
            'version': 'Fixed Version',
            'num_classes': N_CLASS,
            'batch_size': BATCH_SIZE,
            'num_epochs': NUM_EPOCHS,
            'position_granularity': POSITION_GRANULARITY
        },
        'dh_snn': dh_results,
        'vanilla_snn': vanilla_results,
        'improvement': {
            'best_accuracy': best_improvement,
            'final_accuracy': final_improvement
        }
    }
    
    results_file = Path(RESULTS_PATH) / 'neurovpr_fixed_results.json'
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n结果已保存到: {results_file}")
    
    return results

if __name__ == "__main__":
    main()
