#!/usr/bin/env python3
"""
Sequential MNIST DH-SRNN实验 - TBPTT版本
基于原论文实现，使用Truncated Backpropagation Through Time
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
import logging

# 添加项目根目录到路径
sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))

from data_loader import load_sequential_mnist_data
from models import SequentialMNISTModel

def setup_logger(log_file):
    """设置日志记录器"""
    logger = logging.getLogger('S-MNIST_dh_srnn_tbptt')
    logger.setLevel(logging.INFO)
    
    # 文件处理器
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.INFO)
    
    # 格式化器
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    
    # 添加处理器到logger
    logger.addHandler(fh)
    
    return logger

def train_epoch_tbptt(model, train_loader, criterion, optimizer, device, epoch, tbptt_steps=300):
    """使用TBPTT训练一个epoch - 与原论文完全一致"""
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    pbar = tqdm(train_loader, desc=f'Epoch {epoch+1} [Train TBPTT]')
    for batch_idx, (data, target) in enumerate(pbar):
        data, target = data.to(device), target.to(device)
        batch_size, seq_len, input_dim = data.shape

        # 初始化模型状态
        model.reset_states(batch_size)

        # 应用稀疏连接掩码 (与原论文第138行一致)
        model.apply_mask()

        # 累积输出 (与原论文第88行一致: output += spike_layer3)
        accumulated_output = torch.zeros(batch_size, 10, device=device)

        # 逐时间步处理，每tbptt_steps步进行一次反向传播
        for t in range(seq_len):
            x_t = data[:, t, :]  # [batch_size, input_dim]

            # 单时间步前向传播
            step_output = model.forward_single_step(x_t)

            # 累积输出 (与原论文一致)
            accumulated_output += step_output

            # TBPTT: 与原论文一致，只在第0步和最后一步进行反向传播
            if t % tbptt_steps == 0 and t > 0:
                optimizer.zero_grad()
                loss = criterion(accumulated_output, target)
                loss.backward(retain_graph=True)
                optimizer.step()

                # 应用稀疏连接掩码 (与原论文第159行一致)
                model.apply_mask()

                # 分离计算图 (与原论文第98行一致)
                accumulated_output = accumulated_output.detach()
                model.detach_states()

        # 最后进行一次反向传播
        optimizer.zero_grad()
        loss = criterion(accumulated_output, target)
        loss.backward()
        optimizer.step()

        # 应用稀疏连接掩码
        model.apply_mask()

        # 计算准确率
        pred = accumulated_output.argmax(dim=1)
        correct = pred.eq(target).sum().item()

        total_loss += loss.item()
        total_correct += correct
        total_samples += batch_size

        # 更新进度条
        pbar.set_postfix({
            'Loss': f'{loss.item():.4f}',
            'Acc': f'{100.0 * correct / batch_size:.2f}%'
        })

    avg_loss = total_loss / len(train_loader)
    avg_acc = 100.0 * total_correct / total_samples

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
                'Acc': f'{100.0 * correct / target.size(0):.2f}%'
            })
    
    avg_loss = total_loss / len(test_loader)
    avg_acc = 100.0 * total_correct / total_samples
    
    return avg_loss, avg_acc

def main():
    parser = argparse.ArgumentParser(description='Sequential MNIST DH-SRNN TBPTT实验')
    parser.add_argument('--model_type', type=str, default='dh_srnn',
                       choices=['dh_srnn', 'vanilla_srnn'],
                       help='模型类型')
    parser.add_argument('--permute', action='store_true',
                       help='使用置换序列 (PS-MNIST)')
    parser.add_argument('--num_branches', type=int, default=4,
                       help='树突分支数量')
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
    parser.add_argument('--tbptt_steps', type=int, default=300,
                       help='TBPTT步长')
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
    
    # 设备设置
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"📱 使用设备: {device}")
    
    # 创建保存目录
    os.makedirs(args.save_dir, exist_ok=True)
    
    # 设置日志
    task_name = 'PS-MNIST' if args.permute else 'S-MNIST'
    log_file = os.path.join(args.save_dir, f'{task_name}_{args.model_type}_tbptt.log')
    logger = setup_logger(log_file)
    
    print(f"🧠 Sequential MNIST TBPTT实验")
    print(f"📊 任务: {task_name}")
    print(f"🏗️ 模型: {args.model_type}")
    print(f"🔄 TBPTT步长: {args.tbptt_steps}")
    
    # 加载数据
    print("📊 加载数据...")
    train_loader, test_loader = load_sequential_mnist_data(
        batch_size=args.batch_size,
        permute=args.permute,
        seed=args.seed
    )
    
    # 创建模型
    print(f"🧠 创建模型: {args.model_type}")
    if args.model_type == 'dh_srnn':
        # 论文中的Medium初始化: U(0,4)
        model = SequentialMNISTModel(
            model_type='dh_srnn',
            num_branches=args.num_branches,
            tau_m_init=(0, 4),
            tau_n_init=(0, 4)
        )
    else:
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
    print("🚀 开始TBPTT训练...")
    best_acc = 0.0
    train_losses, train_accs = [], []
    test_losses, test_accs = [], []
    
    start_time = time.time()
    
    for epoch in range(args.epochs):
        # 训练
        train_loss, train_acc = train_epoch_tbptt(
            model, train_loader, criterion, optimizer, device, epoch, args.tbptt_steps
        )
        
        # 测试
        test_loss, test_acc = test_epoch(model, test_loader, criterion, device)
        
        # 学习率调度
        scheduler.step()
        
        # 记录最佳准确率
        if test_acc > best_acc:
            best_acc = test_acc
            # 保存最佳模型
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'best_acc': best_acc,
                'args': args
            }, os.path.join(args.save_dir, f'{task_name}_{args.model_type}_tbptt_best.pth'))
        
        # 记录训练历史
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        test_losses.append(test_loss)
        test_accs.append(test_acc)
        
        # 日志记录
        logger.info(f'Epoch {epoch+1}/{args.epochs}: '
                   f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, '
                   f'Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%, '
                   f'LR: {scheduler.get_last_lr()[0]:.6f}')
        
        print(f"Epoch {epoch+1}/{args.epochs}: "
              f"Train Acc: {train_acc:.2f}%, Test Acc: {test_acc:.2f}% "
              f"(Best: {best_acc:.2f}%)")
    
    total_time = time.time() - start_time
    
    # 保存最终结果
    results = {
        'model_type': args.model_type,
        'task': task_name,
        'best_acc': best_acc,
        'train_losses': train_losses,
        'train_accs': train_accs,
        'test_losses': test_losses,
        'test_accs': test_accs,
        'total_time': total_time,
        'args': vars(args)
    }
    
    result_file = os.path.join(args.save_dir, f'{task_name}_{args.model_type}_tbptt_results.pth')
    torch.save(results, result_file)
    
    print(f"\n🎉 TBPTT训练完成!")
    print(f"📈 最佳准确率: {best_acc:.2f}%")
    print(f"⏱️ 总训练时间: {total_time/3600:.2f} 小时")
    print(f"💾 结果已保存: {result_file}")

if __name__ == '__main__':
    main()
