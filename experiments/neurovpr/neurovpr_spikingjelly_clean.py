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

class SimpleNeuroVPRDataset:
    """简化的NeuroVPR数据集加载器 - 支持单数据集内部划分"""
    
    def __init__(self, data_path, exp_names, batch_size=16, is_shuffle=True, nclass=100, 
                 split_type='train', train_ratio=0.7):
        self.data_path = Path(data_path)
        self.exp_names = exp_names
        self.batch_size = batch_size
        self.is_shuffle = is_shuffle
        self.nclass = nclass
        self.split_type = split_type
        self.train_ratio = train_ratio
        
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
    
    def _perform_train_test_split(self):
        """执行训练/测试数据分割"""
        if len(self.data_samples) == 0:
            return
        
        # 设置随机种子确保可重现性
        random.seed(42)
        np.random.seed(42)
        
        # 创建索引列表并打乱
        total_samples = len(self.data_samples)
        indices = list(range(total_samples))
        random.shuffle(indices)
        
        # 计算分割点
        train_size = int(total_samples * self.train_ratio)
        
        if self.split_type == 'train':
            # 使用前train_ratio%的数据作为训练集
            selected_indices = indices[:train_size]
        else:  # test
            # 使用后(1-train_ratio)%的数据作为测试集
            selected_indices = indices[train_size:]
        
        # 根据选中的索引重新构建数据
        selected_samples = [self.data_samples[i] for i in selected_indices]
        selected_labels = [self.labels[i] for i in selected_indices]
        
        self.data_samples = selected_samples
        self.labels = selected_labels
        
        print(f"执行{self.split_type}分割: 总样本{total_samples}, 训练样本{train_size}, "
              f"当前分割({self.split_type})样本{len(self.data_samples)}")
    
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
        dvs_files = sorted(list(dvs_path.glob("*.npy")))  # 移除限制，加载所有文件
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
        
        # 创建位置到类别的映射
        position_list = []
        position_to_class = {}
        
        for _, x, y in pos_filtered:
            # 位置量化 - 使用更细致的量化以创建更多类别
            position_key = (round(x / 2) * 2, round(y / 2) * 2)  # 2米粒度，创建更多类别
            if position_key not in position_to_class:
                class_id = len(position_list) % self.nclass
                position_to_class[position_key] = class_id
                position_list.append(position_key)
        
        print(f"创建 {len(position_list)} 个位置类别")
        
        # 匹配DVS文件和位置标签
        loaded_count = 0
        tolerance = 10.0  # 时间戳匹配容差（秒）- 大幅增加以适应数据特性
        
        # 调试信息：检查时间戳分布
        if len(dvs_filtered) > 0 and len(pos_filtered) > 0:
            dvs_sample_times = sorted([t for t, _ in dvs_filtered[:10]])
            pos_sample_times = sorted([t for t, _, _ in pos_filtered[:10]])
            print(f"调试 - DVS样本时间戳: {[f'{t:.3f}' for t in dvs_sample_times]}")
            print(f"调试 - 位置样本时间戳: {[f'{t:.3f}' for t in pos_sample_times]}")
        
        # 只在过滤后的DVS文件中进行匹配，提高效率
        matched_count = 0
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
                matched_count += 1
                if matched_count <= 5:  # 前5个匹配的调试信息
                    print(f"调试 - 匹配 {matched_count}: DVS={dvs_timestamp:.3f}, 位置={best_match[0]:.1f},{best_match[1]:.1f}, 时差={best_match[2]:.3f}s")
                
                # 找到匹配的位置
                x, y, time_diff = best_match
                # 使用与类别创建时相同的量化粒度
                position_key = (round(x / 2) * 2, round(y / 2) * 2)  # 2米粒度，与上面保持一致
                
                if position_key in position_to_class:
                    try:
                        # 加载DVS数据 - 这些是PyTorch保存的文件
                        dvs_data = torch.load(dvs_file, map_location='cpu')
                        
                        if loaded_count < 3:  # 调试前3个文件
                            print(f"调试 - 文件 {loaded_count+1}: {dvs_file.name}, 数据类型: {type(dvs_data)}, 数据形状: {dvs_data.shape if hasattr(dvs_data, 'shape') else 'N/A'}")
                        
                        # 确保数据是tensor格式
                        if not isinstance(dvs_data, torch.Tensor):
                            if isinstance(dvs_data, np.ndarray):
                                dvs_data = torch.from_numpy(dvs_data)
                            else:
                                if loaded_count < 3:
                                    print(f"调试 - 跳过文件（未知数据类型）: {type(dvs_data)}")
                                continue
                        
                        # 检查数据形状 - 实际格式是 [2, 3, 260, 346]
                        if len(dvs_data.shape) == 4 and dvs_data.shape[0] == 2 and dvs_data.shape[1] == 3:
                            # 数据格式: [2_polarity, 3_timesteps, 260_height, 346_width]
                            # 转换为 [3_timesteps, 2_polarity, 260_height, 346_width] 以符合时间优先的格式
                            dvs_data = dvs_data.permute(1, 0, 2, 3)  # [3, 2, 260, 346]
                            
                            # 下采样空间分辨率以减少计算量 (260, 346) -> (32, 43)
                            # 使用平均池化进行下采样
                            import torch.nn.functional as F
                            dvs_data_flat = dvs_data.contiguous().reshape(-1, 260, 346)  # 先确保连续性
                            dvs_data = F.avg_pool2d(dvs_data_flat, kernel_size=8, stride=8)  # 大约8倍下采样
                            dvs_data = dvs_data.reshape(3, 2, dvs_data.shape[-2], dvs_data.shape[-1])  # 恢复形状
                            
                            self.data_samples.append({
                                'dvs': dvs_data.float(),
                                'file_path': str(dvs_file)
                            })
                            self.labels.append(position_to_class[position_key])
                            loaded_count += 1
                            
                            if loaded_count < 3:  # 调试信息
                                print(f"调试 - 成功加载样本 {loaded_count}, 标签: {position_to_class[position_key]}, 最终形状: {dvs_data.shape}")
                            
                            # 限制样本数量以避免内存问题
                            if loaded_count >= MAX_SAMPLES_PER_DATASET:  # 每个数据集最多5000样本
                                break
                        else:
                            if loaded_count < 3:  # 调试信息
                                print(f"调试 - 跳过文件（形状不匹配）: {dvs_data.shape}, 期望形状: [2, 3, height, width]")
                                    
                    except Exception as e:
                        # 跳过有问题的文件
                        if loaded_count < 3:  # 调试信息
                            print(f"调试 - 文件加载失败: {e}")
                        continue
                else:
                    if matched_count < 3:  # 调试信息
                        print(f"调试 - 位置键不在映射中: {position_key}, 可用键: {list(position_to_class.keys())[:5]}")
        
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
            dummy_aps = torch.zeros(len(batch_dvs), 3, 64)  # 虚拟APS数据
            dummy_gps = torch.zeros(len(batch_dvs), 3, 3)   # 虚拟GPS数据
            
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
BATCH_SIZE = 16
N_CLASS = 100
LEARNING_RATE = 1e-3
NUM_EPOCHS = 50          # 增加训练轮数
NUM_ITER = None          # 自适应迭代数（基于实际数据量）
NUM_BRANCHES = 4
MAX_SAMPLES_PER_DATASET = 5000  # 大幅增加样本数量

# 数据路径
DATA_PATH = '/root/autodl-tmp/neurovpr/datasets/'
RESULTS_PATH = '/root/DH-SNN_reproduce/results'

# 数据集配置 - 只使用走廊数据，避免域偏移
DATASET_NAME = 'floor3_v9'  # 统一使用走廊数据
TRAIN_TEST_SPLIT = 0.7      # 70%训练，30%测试

# 实验配置 - 使用单一数据集内部划分
TRAIN_EXP_IDX = [DATASET_NAME]  # 训练时使用完整数据集的70%
TEST_EXP_IDX = [DATASET_NAME]   # 测试时使用完整数据集的30%

# ==================== 数据检查和加载 ====================

def check_data_availability():
    """检查NeuroVPR数据集可用性"""
    print("检查NeuroVPR数据集可用性...")
    
    dataset_path = Path(DATA_PATH)
    if not dataset_path.exists():
        print(f"❌ 数据集路径不存在: {dataset_path}")
        return False
    
    # 检查统一使用的数据集
    dataset_path_full = dataset_path / DATASET_NAME
    if dataset_path_full.exists():
        print(f"✅ 找到数据集 {DATASET_NAME}")
        
        # 检查关键文件
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
            
        print(f"\n✅ 数据集检查通过: 将使用 {DATASET_NAME} 进行训练/测试划分")
        return True
    else:
        print(f"❌ 数据集 {DATASET_NAME} 未找到")
        return False

def create_data_loaders():
    """创建数据加载器 - 使用单数据集内部分割"""
    print("创建数据加载器(单数据集内部分割)...")
    
    try:
        # 训练集：使用floor3_v9数据集的70%
        train_loader = SimpleNeuroVPRDataset(
            data_path=DATA_PATH,
            exp_names=TRAIN_EXP_IDX,  # [DATASET_NAME]
            batch_size=BATCH_SIZE,
            is_shuffle=True,
            nclass=N_CLASS,
            split_type='train',
            train_ratio=TRAIN_TEST_SPLIT
        )
        
        # 测试集：使用floor3_v9数据集的30%
        test_loader = SimpleNeuroVPRDataset(
            data_path=DATA_PATH,
            exp_names=TEST_EXP_IDX,  # [DATASET_NAME]
            batch_size=BATCH_SIZE,
            is_shuffle=False,
            nclass=N_CLASS,
            split_type='test',
            train_ratio=TRAIN_TEST_SPLIT
        )
        
        print(f"✅ 数据加载器创建成功 (单环境内分割)")
        print(f"   - 训练批次: {len(train_loader)} (约{len(train_loader)*BATCH_SIZE}样本)")
        print(f"   - 测试批次: {len(test_loader)} (约{len(test_loader)*BATCH_SIZE}样本)")
        print(f"   - 数据集: {DATASET_NAME}")
        print(f"   - 分割比例: {TRAIN_TEST_SPLIT:.0%}训练 / {1-TRAIN_TEST_SPLIT:.0%}测试")
        
        return train_loader, test_loader
        
    except Exception as e:
        print(f"❌ 创建数据加载器时出错: {e}")
        import traceback
        traceback.print_exc()
        return None, None

# ==================== 模型定义 ====================

class NeuroVPR_DH_SNN(nn.Module):
    """基于SpikingJelly的NeuroVPR DH-SNN模型"""
    
    def __init__(self, input_dim=2752, hidden_dims=[256, 256], output_dim=N_CLASS, 
                 num_branches=NUM_BRANCHES, step_mode='s'):
        super(NeuroVPR_DH_SNN, self).__init__()
        
        self.step_mode = step_mode
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.T = 3  # DVS序列长度
        
        # 创建DH-SNN配置
        dh_snn_config = {
            'input_dim': input_dim,
            'hidden_dims': hidden_dims,
            'output_dim': output_dim,
            'num_branches': num_branches,
            'v_threshold': 1.0,
            'tau_m_init': (0, 4),
            'tau_n_init': (0, 4),
            'tau_initializer': 'uniform',
            'sparsity': 1.0/num_branches,
            'mask_share': 1,
            'bias': True,
            'surrogate_function': MultiGaussianSurrogate(),
            'reset_mode': 'soft',
            'step_mode': step_mode
        }
        
        # 使用我们的DH-SNN核心架构
        self.dh_snn = create_dh_snn(dh_snn_config)
        
    def forward(self, inp):
        """前向传播"""
        # 获取DVS输入（索引2）
        dvs_inp = inp[2]  # shape: [batch, 3, 2, 32, 43]
        
        batch_size = dvs_inp.shape[0]
        
        # 重塑DVS数据: [batch, 3, 2, 32, 43] -> [3, batch, 2*32*43]
        dvs_reshaped = dvs_inp.reshape(batch_size, self.T, -1)  # [batch, 3, features]
        dvs_input = dvs_reshaped.transpose(0, 1)  # [3, batch, features]
        
        # 通过DH-SNN网络
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

class NeuroVPR_Vanilla_SNN(nn.Module):
    """基于SpikingJelly的NeuroVPR Vanilla SNN模型"""
    
    def __init__(self, input_dim=2752, hidden_dims=[256, 256], output_dim=N_CLASS, step_mode='s'):
        super(NeuroVPR_Vanilla_SNN, self).__init__()
        
        self.step_mode = step_mode
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.T = 3
        
        # 构建Vanilla SNN网络
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
        self.layers.append(neuron.LIFNode(
            tau=2.0,
            v_threshold=1.0, 
            surrogate_function=surrogate.ATan(),
            step_mode=step_mode
        ))
        
    def forward(self, inp):
        """前向传播"""
        dvs_inp = inp[2]  # [batch, 3, 2, 32, 43]
        
        batch_size = dvs_inp.shape[0]
        dvs_reshaped = dvs_inp.reshape(batch_size, self.T, -1)
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
        
        # 自适应迭代数：使用完整的数据集
        max_iter = len(train_loader) if NUM_ITER is None else min(NUM_ITER, len(train_loader))
        for iter_idx in range(max_iter):
            try:
                inputs, target = next(train_iters)
            except StopIteration:
                train_iters = iter(train_loader)
                inputs, target = next(train_iters)
            
            # 移动到设备
            target = target.to(device)
            inputs = [inp.to(device) for inp in inputs]
            
            # 重置神经元状态
            functional.reset_net(model)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, target)
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.cpu().item()
            
            # 计算训练准确率
            acc1, acc5 = accuracy(outputs.cpu(), target.cpu(), topk=(1, 5))
            train_acc1 += acc1
            train_acc5 += acc5
        
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
        num_train_iter = max_iter  # 使用前面计算的max_iter
        if num_train_iter == 0:
            print("⚠️  警告: 没有训练数据，跳过此轮次")
            continue
            
        avg_train_loss = running_loss / num_train_iter
        avg_train_acc1 = train_acc1 / num_train_iter
        avg_train_acc5 = train_acc5 / num_train_iter
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
    print("NeuroVPR SpikingJelly实验: DH-SNN vs Vanilla SNN (Clean Version)")
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
        print("\n❌ NeuroVPR数据集未找到！")
        print("请从以下地址下载数据集: https://zenodo.org/records/7827108")
        print(f"并将其放置在: {DATA_PATH}")
        return None
    
    # 创建数据加载器
    train_loader, test_loader = create_data_loaders()
    if train_loader is None or test_loader is None:
        print("❌ 创建数据加载器失败。请检查数据集结构。")
        return None
    
    if len(train_loader) == 0 or len(test_loader) == 0:
        print("❌ 没有成功加载任何数据样本。请检查数据格式和时间戳匹配逻辑。")
        print(f"训练批次: {len(train_loader)}, 测试批次: {len(test_loader)}")
        return None
    
    # 计算输入维度（基于DVS数据形状）
    # DVS数据形状: [batch, 3, 2, 32, 43] -> 展平后每个时间步的特征数: 2*32*43 = 2752
    input_features = 2 * 32 * 43
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
    
    # 避免除零错误
    if vanilla_results['best_test_acc1'] > 0:
        best_relative = (dh_results['best_test_acc1'] / vanilla_results['best_test_acc1'] - 1) * 100
    else:
        best_relative = 0
        
    if vanilla_results['final_test_acc1'] > 0:
        final_relative = (dh_results['final_test_acc1'] / vanilla_results['final_test_acc1'] - 1) * 100
    else:
        final_relative = 0
    
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
            'num_iter_per_epoch': max_iter,
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
    results_file = os.path.join(RESULTS_PATH, 'neurovpr_spikingjelly_clean_results.json')
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    # 保存模型
    torch.save(dh_snn_model.state_dict(), os.path.join(RESULTS_PATH, 'neurovpr_dh_snn_spikingjelly_clean.pth'))
    torch.save(vanilla_snn_model.state_dict(), os.path.join(RESULTS_PATH, 'neurovpr_vanilla_snn_spikingjelly_clean.pth'))
    
    print(f"\n✅ 结果保存至: {results_file}")
    print(f"✅ 模型保存至: {RESULTS_PATH}")
    
    return all_results

if __name__ == "__main__":
    results = main()
