#!/usr/bin/env python3
"""
Sequential MNIST数据加载器 - SpikingJelly版本
基于原论文实现，将MNIST转换为序列数据
"""

import os
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import numpy as np

class SequentialMNISTDataset(Dataset):
    """
    Sequential MNIST数据集类
    将28x28的MNIST图像转换为784步的序列数据
    """

    def __init__(self, root='./mnist_data', train=True, permute=False, seed=0):
        """
        Args:
            root: 数据根目录
            train: 是否为训练集
            permute: 是否使用置换序列 (True for PS-MNIST, False for S-MNIST)
            seed: 随机种子
        """
        self.train = train
        self.permute = permute

        # 下载并加载MNIST数据
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

        self.mnist_dataset = torchvision.datasets.MNIST(
            root=root,
            train=train,
            download=True,
            transform=transform
        )

        # 创建置换索引 (用于PS-MNIST)
        if permute:
            torch.manual_seed(seed)
            self.perm = torch.randperm(784)
        else:
            self.perm = None

    def __len__(self):
        return len(self.mnist_dataset)

    def __getitem__(self, idx):
        """
        返回序列化的MNIST数据

        Returns:
            data: [784, 1] - 序列数据，每个时间步一个像素
            label: int - 类别标签
        """
        image, label = self.mnist_dataset[idx]

        # 将28x28图像展平为784序列
        sequence = image.view(784, 1)  # [seq_len, input_dim]

        # 如果需要置换 (PS-MNIST)
        if self.perm is not None:
            sequence = sequence[self.perm, :]

        return sequence, label

def load_sequential_mnist_data(batch_size=128, num_workers=4, permute=False, seed=0):
    """
    加载Sequential MNIST数据

    Args:
        batch_size: 批次大小
        num_workers: 工作进程数
        permute: 是否使用置换序列 (True for PS-MNIST, False for S-MNIST)
        seed: 随机种子

    Returns:
        train_loader, test_loader
    """

    # 创建数据集
    train_dataset = SequentialMNISTDataset(train=True, permute=permute, seed=seed)
    test_dataset = SequentialMNISTDataset(train=False, permute=permute, seed=seed)

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

    task_name = "PS-MNIST" if permute else "S-MNIST"
    print(f"📊 {task_name} 数据加载完成:")
    print(f"  训练集: {len(train_dataset)} 样本")
    print(f"  测试集: {len(test_dataset)} 样本")
    print(f"  序列长度: 784")
    print(f"  输入维度: 1")
    print(f"  类别数: 10")

    return train_loader, test_loader

if __name__ == '__main__':
    # 测试数据加载器
    train_loader, test_loader = load_sequential_mnist_data('../datasets/sequential_mnist/', 32)
    
    for batch_idx, (data, targets) in enumerate(train_loader):
        print(f"Batch {batch_idx}: data={data.shape}, targets={targets.shape}")
        if batch_idx >= 2:
            break
