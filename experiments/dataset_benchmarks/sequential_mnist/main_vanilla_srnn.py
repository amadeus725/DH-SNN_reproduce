#!/usr/bin/env python3
"""
Sequential MNIST Vanilla SRNN实验 - SpikingJelly版本
用作DH-SRNN的基线对比
"""

import os
import sys
import time
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import numpy as np
from tqdm import tqdm

# 添加项目根目录到路径
sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))

from data_loader import load_sequential_mnist_data
from models import SequentialMNISTModel

# 简单的日志设置函数
def setup_logger(name, log_file):
    import logging
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # 创建文件处理器
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.INFO)

    # 创建格式器
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)

    # 添加处理器到logger
    logger.addHandler(fh)

    return logger

def train_epoch(model, train_loader, criterion, optimizer, device, epoch):
    """训练一个epoch"""
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    pbar = tqdm(train_loader, desc=f'Epoch {epoch+1} [Train]')
    for batch_idx, (data, target) in enumerate(pbar):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()

        # 前向传播
        output = model(data)
        loss = criterion(output, target)

        # 反向传播
        loss.backward()
        optimizer.step()

        # 计算准确率
        pred = output.argmax(dim=1)
        correct = pred.eq(target).sum().item()

        total_loss += loss.item()
        total_correct += correct
        total_samples += target.size(0)

        # 更新进度条
        pbar.set_postfix({
            'Loss': f'{loss.item():.4f}',
            'Acc': f'{100.*correct/target.size(0):.2f}%'
        })

    avg_loss = total_loss / len(train_loader)
    avg_acc = 100. * total_correct / total_samples

    return avg_loss, avg_acc

def test_epoch(model, test_loader, criterion, device):
    """测试一个epoch"""
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        pbar = tqdm(test_loader, desc='Testing')
        for data, target in pbar:
            data, target = data.to(device), target.to(device)

            # 前向传播
            output = model(data)
            loss = criterion(output, target)

            # 计算准确率
            pred = output.argmax(dim=1)
            correct = pred.eq(target).sum().item()

            total_loss += loss.item()
            total_correct += correct
            total_samples += target.size(0)

            # 更新进度条
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{100.*correct/target.size(0):.2f}%'
            })

    avg_loss = total_loss / len(test_loader)
    avg_acc = 100. * total_correct / total_samples

    return avg_loss, avg_acc

def main():
    parser = argparse.ArgumentParser(description='Sequential MNIST Vanilla SRNN实验')
    parser.add_argument('--permute', action='store_true',
                       help='使用置换序列 (PS-MNIST)')
    parser.add_argument('--batch_size', type=int, default=128,
                       help='批次大小')
    parser.add_argument('--epochs', type=int, default=150,
                       help='训练轮数')
    parser.add_argument('--lr', type=float, default=1e-2,
                       help='学习率')
    parser.add_argument('--lr_decay', type=float, default=0.1,
                       help='学习率衰减因子')
    parser.add_argument('--lr_step', type=int, default=50,
                       help='学习率衰减步长')
    parser.add_argument('--device', type=str, default='cuda',
                       help='设备')
    parser.add_argument('--seed', type=int, default=42,
                       help='随机种子')
    parser.add_argument('--save_dir', type=str, default='./results',
                       help='结果保存目录')

    args = parser.parse_args()

    # 设置随机种子
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    # 设置设备
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"🔧 使用设备: {device}")

    # 创建保存目录
    os.makedirs(args.save_dir, exist_ok=True)

    # 设置日志
    task_name = "PS-MNIST" if args.permute else "S-MNIST"
    logger = setup_logger(f'{task_name}_vanilla_srnn',
                         os.path.join(args.save_dir, f'{task_name}_vanilla_srnn.log'))

    # 加载数据
    print("📊 加载数据...")
    train_loader, test_loader = load_sequential_mnist_data(
        batch_size=args.batch_size,
        permute=args.permute,
        seed=args.seed
    )

    # 创建模型
    print(f"🧠 创建模型: Vanilla SRNN")
    model = SequentialMNISTModel(
        model_type='vanilla_srnn',
        tau_m_init=(0, 4)
    )

    model = model.to(device)

    # 打印模型信息
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"📈 模型参数: {total_params:,} (可训练: {trainable_params:,})")

    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = StepLR(optimizer, step_size=args.lr_step, gamma=args.lr_decay)

    # 训练循环
    print("🚀 开始训练...")
    best_acc = 0.0
    train_losses, train_accs = [], []
    test_losses, test_accs = [], []

    start_time = time.time()

    for epoch in range(args.epochs):
        # 训练
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device, epoch)

        # 测试
        test_loss, test_acc = test_epoch(model, test_loader, criterion, device)

        # 学习率调度
        scheduler.step()

        # 记录结果
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        test_losses.append(test_loss)
        test_accs.append(test_acc)

        # 保存最佳模型
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(),
                      os.path.join(args.save_dir, f'{task_name}_vanilla_srnn_best.pth'))

        # 日志记录
        logger.info(f'Epoch {epoch+1}/{args.epochs}: '
                   f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, '
                   f'Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%, '
                   f'LR: {scheduler.get_last_lr()[0]:.6f}')

        print(f"Epoch {epoch+1}/{args.epochs}: "
              f"Train Acc: {train_acc:.2f}%, Test Acc: {test_acc:.2f}% "
              f"(Best: {best_acc:.2f}%)")

    total_time = time.time() - start_time

    # 保存训练结果
    results = {
        'args': vars(args),
        'train_losses': train_losses,
        'train_accs': train_accs,
        'test_losses': test_losses,
        'test_accs': test_accs,
        'best_acc': best_acc,
        'total_time': total_time
    }

    torch.save(results, os.path.join(args.save_dir, f'{task_name}_vanilla_srnn_results.pth'))

    # 最终结果
    print(f"\n🎯 训练完成!")
    print(f"📊 最佳测试准确率: {best_acc:.2f}%")
    print(f"⏱️  总训练时间: {total_time/3600:.2f} 小时")

    logger.info(f'Training completed. Best accuracy: {best_acc:.2f}%, Time: {total_time/3600:.2f}h')

if __name__ == '__main__':
    main()
