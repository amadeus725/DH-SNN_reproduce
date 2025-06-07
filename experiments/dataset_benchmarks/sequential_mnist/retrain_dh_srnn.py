#!/usr/bin/env python3
"""
重新训练修复后的DH-SRNN
"""

import os
import sys
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import numpy as np
from tqdm import tqdm

# 添加项目路径
sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))

from data_loader import load_sequential_mnist_data
from models import SequentialMNISTModel

def train_epoch(model, train_loader, optimizer, criterion, device, epoch):
    """训练一个epoch"""
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    
    pbar = tqdm(train_loader, desc=f'Epoch {epoch}')
    for batch_idx, (data, target) in enumerate(pbar):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        # 计算准确率
        pred = output.argmax(dim=1)
        correct = pred.eq(target).sum().item()
        
        total_loss += loss.item()
        total_correct += correct
        total_samples += target.size(0)
        
        # 更新进度条
        if batch_idx % 100 == 0:
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{100.*correct/target.size(0):.1f}%'
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
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            
            pred = output.argmax(dim=1)
            correct = pred.eq(target).sum().item()
            
            total_loss += loss.item()
            total_correct += correct
            total_samples += target.size(0)
    
    avg_loss = total_loss / len(test_loader)
    avg_acc = 100. * total_correct / total_samples
    
    return avg_loss, avg_acc

def main():
    """主训练函数"""
    print("🔧 重新训练修复后的DH-SRNN")
    print("=" * 50)
    
    # 设备配置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🔧 使用设备: {device}")
    
    # 数据加载
    print("\n📊 加载S-MNIST数据...")
    train_loader, test_loader = load_sequential_mnist_data(
        batch_size=64,
        permute=False,  # S-MNIST
        seed=42
    )
    
    print(f"   训练集: {len(train_loader.dataset)} 样本")
    print(f"   测试集: {len(test_loader.dataset)} 样本")
    
    # 创建模型
    print("\n🧠 创建DH-SRNN模型...")
    model = SequentialMNISTModel(
        model_type='dh_srnn',
        num_branches=4,
        tau_m_init=(0, 4),
        tau_n_init=(0, 4)
    ).to(device)
    
    print(f"📈 模型参数数量: {sum(p.numel() for p in model.parameters()):,}")
    
    # 训练配置
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)  # 降低学习率
    scheduler = StepLR(optimizer, step_size=25, gamma=0.5)
    
    # 训练参数
    num_epochs = 50  # 减少epoch数进行快速验证
    best_acc = 0.0
    
    # 记录训练过程
    train_losses = []
    train_accs = []
    test_losses = []
    test_accs = []
    
    print(f"\n🚀 开始训练 ({num_epochs} epochs)...")
    start_time = time.time()
    
    for epoch in range(1, num_epochs + 1):
        # 训练
        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, criterion, device, epoch
        )
        
        # 测试
        test_loss, test_acc = test_epoch(
            model, test_loader, criterion, device
        )
        
        # 更新学习率
        scheduler.step()
        
        # 记录结果
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        test_losses.append(test_loss)
        test_accs.append(test_acc)
        
        # 保存最佳模型
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save({
                'model_state_dict': model.state_dict(),
                'epoch': epoch,
                'best_acc': best_acc,
                'optimizer_state_dict': optimizer.state_dict(),
            }, 'results/S-MNIST_dh_srnn_fixed_best.pth')
        
        # 输出结果
        print(f"Epoch {epoch:2d}: "
              f"Train Loss={train_loss:.4f}, Train Acc={train_acc:.2f}%, "
              f"Test Loss={test_loss:.4f}, Test Acc={test_acc:.2f}%, "
              f"Best={best_acc:.2f}%")
        
        # 早停检查
        if test_acc > 70.0:  # 如果达到70%就认为修复成功
            print(f"✅ 达到70%准确率，修复成功!")
            break
    
    end_time = time.time()
    training_time = (end_time - start_time) / 3600
    
    # 保存完整结果
    results = {
        'model_type': 'dh_srnn_fixed',
        'dataset': 'S-MNIST',
        'best_test_accuracy': best_acc,
        'final_test_accuracy': test_accs[-1],
        'epochs': len(test_accs),
        'training_time_hours': training_time,
        'train_losses': train_losses,
        'train_accuracies': train_accs,
        'test_losses': test_losses,
        'test_accuracies': test_accs,
        'config': {
            'batch_size': 64,
            'learning_rate': 1e-3,
            'num_branches': 4,
            'tau_m_init': (0, 4),
            'tau_n_init': (0, 4)
        }
    }
    
    torch.save(results, 'results/S-MNIST_dh_srnn_fixed_results.pth')
    
    # 输出总结
    print(f"\n📋 训练总结:")
    print("=" * 30)
    print(f"最佳测试准确率: {best_acc:.2f}%")
    print(f"最终测试准确率: {test_accs[-1]:.2f}%")
    print(f"训练时间: {training_time:.2f} 小时")
    print(f"训练轮数: {len(test_accs)} epochs")
    
    # 判断修复是否成功
    if best_acc > 70.0:
        print(f"\n🎉 DH-SRNN修复成功!")
        print(f"   原始结果: 10.1%")
        print(f"   修复后结果: {best_acc:.2f}%")
        print(f"   改进幅度: +{best_acc - 10.1:.1f}个百分点")
        return True
    elif best_acc > 30.0:
        print(f"\n⚠️  DH-SRNN部分修复")
        print(f"   有明显改进但仍需优化")
        return False
    else:
        print(f"\n❌ DH-SRNN修复失败")
        print(f"   仍然存在问题")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
