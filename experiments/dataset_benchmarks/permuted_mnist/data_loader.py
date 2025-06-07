#!/usr/bin/env python3
"""
PERMUTED_MNIST数据加载器
"""

import torch
from torch.utils.data import DataLoader, Dataset
import torchvision
import torchvision.transforms as transforms
import numpy as np
import os

class PermutedMNISTDataset(Dataset):
    """
    Permuted MNIST数据集类
    将MNIST数据转换为序列形式，并应用像素置换
    """

    def __init__(self, data_path, split='train', permutation_seed=42, transform=None):
        self.data_path = data_path
        self.split = split
        self.transform = transform
        self.permutation_seed = permutation_seed

        # 创建置换索引
        np.random.seed(permutation_seed)
        self.permutation = np.random.permutation(784)

        # 加载数据
        self.data, self.labels = self._load_data()

    def _load_data(self):
        """加载MNIST数据并转换为序列形式"""

        # 下载MNIST数据
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

        if self.split == 'train':
            dataset = torchvision.datasets.MNIST(
                root=self.data_path,
                train=True,
                download=True,
                transform=transform
            )
        else:
            dataset = torchvision.datasets.MNIST(
                root=self.data_path,
                train=False,
                download=True,
                transform=transform
            )

        # 转换为序列数据
        data_list = []
        labels_list = []

        for img, label in dataset:
            # 展平图像 [28, 28] -> [784]
            img_flat = img.view(-1)

            # 应用置换
            img_permuted = img_flat[self.permutation]

            # 转换为序列 [784] -> [784, 1]
            img_seq = img_permuted.unsqueeze(-1)

            data_list.append(img_seq)
            labels_list.append(label)

        data = torch.stack(data_list)
        labels = torch.tensor(labels_list)

        return data, labels
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        data = self.data[idx]
        label = self.labels[idx]
        
        if self.transform:
            data = self.transform(data)
        
        return data, label

def load_permuted_mnist_data(data_path, batch_size, num_workers=4, permutation_seed=42):
    """
    加载Permuted MNIST数据

    Args:
        data_path: 数据路径
        batch_size: 批次大小
        num_workers: 工作进程数
        permutation_seed: 置换随机种子

    Returns:
        train_loader, test_loader
    """

    # 创建数据集
    train_dataset = PermutedMNISTDataset(data_path, split='train', permutation_seed=permutation_seed)
    test_dataset = PermutedMNISTDataset(data_path, split='test', permutation_seed=permutation_seed)

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

    print(f"📊 Permuted MNIST数据加载完成:")
    print(f"  训练集: {len(train_dataset)} 样本")
    print(f"  测试集: {len(test_dataset)} 样本")
    print(f"  序列长度: 784")
    print(f"  置换种子: {permutation_seed}")

    return train_loader, test_loader

if __name__ == '__main__':
    # 测试数据加载器
    train_loader, test_loader = load_permuted_mnist_data('../datasets/permuted_mnist/', 32)
    
    for batch_idx, (data, targets) in enumerate(train_loader):
        print(f"Batch {batch_idx}: data={data.shape}, targets={targets.shape}")
        if batch_idx >= 2:
            break
