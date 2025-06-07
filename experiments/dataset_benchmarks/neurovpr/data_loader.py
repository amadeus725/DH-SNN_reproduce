#!/usr/bin/env python3
"""
NEUROVPR数据加载器
"""

import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np
import os

class NEUROVPRDataset(Dataset):
    """
    NEUROVPR数据集类
    """
    
    def __init__(self, data_path, split='train', transform=None):
        self.data_path = data_path
        self.split = split
        self.transform = transform
        
        # 加载数据
        self.data, self.labels = self._load_data()
    
    def _load_data(self):
        """加载数据"""
        # TODO: 实现具体的数据加载逻辑
        # 这里需要根据具体数据集格式实现
        
        if self.split == 'train':
            data_file = os.path.join(self.data_path, f'neurovpr_train.h5.gz')
        else:
            data_file = os.path.join(self.data_path, f'neurovpr_test.h5.gz')
        
        # 示例代码 - 需要根据实际数据格式修改
        data = torch.randn(1000, 100, 700)  # [samples, time, features]
        labels = torch.randint(0, 20, (1000,))  # [samples]
        
        return data, labels
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        data = self.data[idx]
        label = self.labels[idx]
        
        if self.transform:
            data = self.transform(data)
        
        return data, label

def load_neurovpr_data(data_path, batch_size, num_workers=4):
    """
    加载NEUROVPR数据
    
    Args:
        data_path: 数据路径
        batch_size: 批次大小
        num_workers: 工作进程数
    
    Returns:
        train_loader, test_loader
    """
    
    # 创建数据集
    train_dataset = NEUROVPRDataset(data_path, split='train')
    test_dataset = NEUROVPRDataset(data_path, split='test')
    
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
    
    print(f"📊 NEUROVPR数据加载完成:")
    print(f"  训练集: {len(train_dataset)} 样本")
    print(f"  测试集: {len(test_dataset)} 样本")
    
    return train_loader, test_loader

if __name__ == '__main__':
    # 测试数据加载器
    train_loader, test_loader = load_neurovpr_data('../datasets/neurovpr/', 32)
    
    for batch_idx, (data, targets) in enumerate(train_loader):
        print(f"Batch {batch_idx}: data={data.shape}, targets={targets.shape}")
        if batch_idx >= 2:
            break
