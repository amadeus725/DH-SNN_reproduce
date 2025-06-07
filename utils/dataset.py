#!/usr/bin/env python3
"""
SHD数据集加载器 - 基于SpikingJelly框架
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, List, Optional
import glob


class SHDDataset(Dataset):
    """SHD数据集类"""
    
    def __init__(self, data_dir: str, dt_ms: float = 1.0, max_length: Optional[int] = None):
        """
        初始化SHD数据集
        
        Args:
            data_dir: 数据目录路径
            dt_ms: 时间分辨率 (毫秒)
            max_length: 最大序列长度，如果为None则使用实际长度
        """
        self.data_dir = data_dir
        self.dt_ms = dt_ms
        self.max_length = max_length
        
        # 查找所有数据文件
        pattern = os.path.join(data_dir, f'*_dt_{dt_ms}ms', '*.npy')
        self.file_paths = sorted(glob.glob(pattern))
        
        if len(self.file_paths) == 0:
            raise ValueError(f"No data files found in {data_dir} for dt={dt_ms}ms")
        
        print(f"Found {len(self.file_paths)} samples for dt={dt_ms}ms")
        
        # 提取标签
        self.labels = []
        for file_path in self.file_paths:
            filename = os.path.basename(file_path)
            # 从文件名提取标签: sample_xxxxx_label_y.npy
            label = int(filename.split('_label_')[1].split('.')[0])
            self.labels.append(label)
        
        self.labels = np.array(self.labels)
        self.num_classes = len(np.unique(self.labels))
        
        print(f"Dataset loaded: {len(self)} samples, {self.num_classes} classes")
    
    def __len__(self) -> int:
        return len(self.file_paths)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        获取单个样本
        
        Returns:
            spike_data: [T, 700] 脉冲数据
            label: 标签
        """
        # 加载脉冲数据
        spike_data = np.load(self.file_paths[idx])  # [T, 700]
        label = self.labels[idx]
        
        # 处理序列长度
        if self.max_length is not None and spike_data.shape[0] > self.max_length:
            spike_data = spike_data[:self.max_length]
        
        # 转换为tensor
        spike_data = torch.from_numpy(spike_data).float()
        label = torch.tensor(label, dtype=torch.long)
        
        return spike_data, label


def collate_fn(batch: List[Tuple[torch.Tensor, torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    自定义批处理函数，处理不同长度的序列
    """
    spike_data_list, labels = zip(*batch)
    
    # 找到最大序列长度
    max_length = max(data.shape[0] for data in spike_data_list)
    batch_size = len(spike_data_list)
    input_dim = spike_data_list[0].shape[1]  # 700
    
    # 创建填充后的批次
    padded_data = torch.zeros(batch_size, max_length, input_dim)
    
    for i, data in enumerate(spike_data_list):
        seq_len = data.shape[0]
        padded_data[i, :seq_len] = data
    
    labels = torch.stack(labels)
    
    return padded_data, labels


def create_shd_dataloaders(data_dir: str, dt_ms: float = 1.0, batch_size: int = 100,
                          max_length: Optional[int] = None, num_workers: int = 4) -> Tuple[DataLoader, DataLoader]:
    """
    创建SHD数据加载器
    
    Args:
        data_dir: 数据目录
        dt_ms: 时间分辨率 (毫秒)
        batch_size: 批次大小
        max_length: 最大序列长度
        num_workers: 工作进程数
    
    Returns:
        train_loader, test_loader
    """
    # 训练集和测试集目录
    train_dir = os.path.join(data_dir, 'processed_data')
    test_dir = os.path.join(data_dir, 'processed_data')
    
    # 创建数据集
    train_dataset = SHDDataset(train_dir, dt_ms=dt_ms, max_length=max_length)
    test_dataset = SHDDataset(test_dir, dt_ms=dt_ms, max_length=max_length)
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, test_loader


class SHDDatasetFixed(Dataset):
    """固定长度的SHD数据集类 - 用于更高效的训练"""
    
    def __init__(self, data_dir: str, dt_ms: float = 1.0, fixed_length: int = 1000):
        """
        初始化固定长度的SHD数据集
        
        Args:
            data_dir: 数据目录路径
            dt_ms: 时间分辨率 (毫秒)
            fixed_length: 固定序列长度
        """
        self.data_dir = data_dir
        self.dt_ms = dt_ms
        self.fixed_length = fixed_length
        
        # 查找所有数据文件
        pattern = os.path.join(data_dir, f'*_dt_{dt_ms}ms', '*.npy')
        self.file_paths = sorted(glob.glob(pattern))
        
        if len(self.file_paths) == 0:
            raise ValueError(f"No data files found in {data_dir} for dt={dt_ms}ms")
        
        # 提取标签
        self.labels = []
        for file_path in self.file_paths:
            filename = os.path.basename(file_path)
            label = int(filename.split('_label_')[1].split('.')[0])
            self.labels.append(label)
        
        self.labels = np.array(self.labels)
        self.num_classes = len(np.unique(self.labels))
        
        print(f"Fixed-length dataset loaded: {len(self)} samples, {self.num_classes} classes, length={fixed_length}")
    
    def __len__(self) -> int:
        return len(self.file_paths)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        获取单个样本
        
        Returns:
            spike_data: [fixed_length, 700] 脉冲数据
            label: 标签
        """
        # 加载脉冲数据
        spike_data = np.load(self.file_paths[idx])  # [T, 700]
        label = self.labels[idx]
        
        # 处理序列长度
        original_length = spike_data.shape[0]
        
        if original_length >= self.fixed_length:
            # 截断
            spike_data = spike_data[:self.fixed_length]
        else:
            # 填充
            padding = np.zeros((self.fixed_length - original_length, 700), dtype=np.float32)
            spike_data = np.concatenate([spike_data, padding], axis=0)
        
        # 转换为tensor
        spike_data = torch.from_numpy(spike_data).float()
        label = torch.tensor(label, dtype=torch.long)
        
        return spike_data, label


def create_shd_dataloaders_fixed(data_dir: str, dt_ms: float = 1.0, batch_size: int = 100,
                                fixed_length: int = 1000, num_workers: int = 4) -> Tuple[DataLoader, DataLoader]:
    """
    创建固定长度的SHD数据加载器
    
    Args:
        data_dir: 数据目录
        dt_ms: 时间分辨率 (毫秒)
        batch_size: 批次大小
        fixed_length: 固定序列长度
        num_workers: 工作进程数
    
    Returns:
        train_loader, test_loader
    """
    # 训练集和测试集目录
    train_dir = os.path.join(data_dir, 'processed_data')
    test_dir = os.path.join(data_dir, 'processed_data')
    
    # 创建数据集
    train_dataset = SHDDatasetFixed(train_dir, dt_ms=dt_ms, fixed_length=fixed_length)
    test_dataset = SHDDatasetFixed(test_dir, dt_ms=dt_ms, fixed_length=fixed_length)
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, test_loader


if __name__ == "__main__":
    # 测试数据加载器
    data_dir = "."
    dt_ms = 1.0
    
    try:
        train_loader, test_loader = create_shd_dataloaders_fixed(
            data_dir, dt_ms=dt_ms, batch_size=4, fixed_length=1000, num_workers=0
        )
        
        # 测试一个批次
        for batch_idx, (data, labels) in enumerate(train_loader):
            print(f"Batch {batch_idx}: data shape = {data.shape}, labels shape = {labels.shape}")
            print(f"Labels: {labels}")
            if batch_idx >= 2:
                break
                
    except Exception as e:
        print(f"Error: {e}")
        print("Please run preprocessing.py first to generate processed data.")
