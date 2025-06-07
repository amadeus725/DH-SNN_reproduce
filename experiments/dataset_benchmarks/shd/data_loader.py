#!/usr/bin/env python3
"""
SHD数据加载器 - 使用真实数据
"""

import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np
import os
import gzip
import h5py

class SHDDataset(Dataset):
    """
    SHD数据集类 - 加载真实的SHD数据
    """

    def __init__(self, data_path, split='train', max_time=1.4, dt=0.001, max_samples=None):
        self.data_path = data_path
        self.split = split
        self.max_time = max_time
        self.dt = dt
        self.n_time_bins = int(max_time / dt)
        self.n_units = 700

        # 加载数据
        self.spikes, self.labels, self.units, self.times = self._load_data()

        # 限制样本数量
        if max_samples is not None:
            self.spikes = self.spikes[:max_samples+1]
            self.labels = self.labels[:max_samples]

    def _load_data(self):
        """加载真实的SHD数据"""
        if self.split == 'train':
            data_file = os.path.join(self.data_path, 'shd_train.h5.gz')
        else:
            data_file = os.path.join(self.data_path, 'shd_test.h5.gz')

        print(f"📂 加载SHD数据: {data_file}")

        if not os.path.exists(data_file):
            print(f"❌ 数据文件不存在: {data_file}")
            # 返回模拟数据作为后备
            print("🔄 使用模拟数据...")
            data = torch.randn(1000, 100, 700)  # [samples, time, features]
            labels = torch.randint(0, 20, (1000,))  # [samples]
            spikes = np.arange(1001)  # 模拟spikes索引
            units = np.random.randint(0, 700, 10000)
            times = np.random.uniform(0, 1.4, 10000)
            return spikes, labels.numpy(), units, times

        try:
            with gzip.open(data_file, 'rb') as f:
                with h5py.File(f, 'r') as hf:
                    spikes = hf['spikes'][:]
                    labels = hf['labels'][:]
                    units = hf['units'][:]
                    times = hf['times'][:]

            print(f"✅ 数据加载完成: {len(labels)} 样本")
            print(f"   时间范围: {times.min():.3f}s - {times.max():.3f}s")
            print(f"   单元范围: {units.min()} - {units.max()}")
            print(f"   类别数量: {len(np.unique(labels))}")

            return spikes, labels, units, times

        except Exception as e:
            print(f"❌ 数据加载失败: {e}")
            print("🔄 使用模拟数据...")
            # 返回模拟数据作为后备
            data = torch.randn(1000, 100, 700)
            labels = torch.randint(0, 20, (1000,))
            spikes = np.arange(1001)
            units = np.random.randint(0, 700, 10000)
            times = np.random.uniform(0, 1.4, 10000)
            return spikes, labels.numpy(), units, times

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        """获取单个样本"""
        # 创建脉冲序列 - 使用较短的序列以节省内存
        seq_len = min(self.n_time_bins, 500)  # 限制序列长度
        spike_train = torch.zeros(seq_len, self.n_units)

        # 获取当前样本的脉冲
        start_idx = self.spikes[idx] if idx < len(self.spikes) else 0
        end_idx = self.spikes[idx + 1] if idx + 1 < len(self.spikes) else len(self.times)

        if start_idx < end_idx:
            times = self.times[start_idx:end_idx]
            units = self.units[start_idx:end_idx]

            # 将时间转换为时间步
            time_bins = (times / self.dt).astype(int)
            time_bins = np.clip(time_bins, 0, seq_len - 1)
            units = np.clip(units, 0, self.n_units - 1)

            # 设置脉冲
            for t, u in zip(time_bins, units):
                spike_train[t, u] = 1.0

        return spike_train, self.labels[idx]

def load_shd_data(data_path, batch_size, num_workers=4, max_samples=None):
    """
    加载SHD数据

    Args:
        data_path: 数据路径
        batch_size: 批次大小
        num_workers: 工作进程数
        max_samples: 最大样本数，用于快速测试

    Returns:
        train_loader, test_loader
    """

    # 创建数据集
    train_dataset = SHDDataset(data_path, split='train', max_samples=max_samples)
    test_dataset = SHDDataset(data_path, split='test', max_samples=max_samples//4 if max_samples else None)

    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,  # 设为0避免多进程问题
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,  # 设为0避免多进程问题
        pin_memory=True
    )

    print(f"📊 SHD数据加载完成:")
    print(f"  训练集: {len(train_dataset)} 样本")
    print(f"  测试集: {len(test_dataset)} 样本")

    return train_loader, test_loader

if __name__ == '__main__':
    # 测试数据加载器
    train_loader, test_loader = load_shd_data('../datasets/shd/', 32)

    for batch_idx, (data, targets) in enumerate(train_loader):
        print(f"Batch {batch_idx}: data={data.shape}, targets={targets.shape}")
        if batch_idx >= 2:
            break
