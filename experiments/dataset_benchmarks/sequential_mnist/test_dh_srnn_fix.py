#!/usr/bin/env python3
"""
测试DH-SRNN修复后的效果
快速验证状态重置和梯度流动是否正常
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm

# 添加项目路径
sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))

from data_loader import load_sequential_mnist_data
from models import SequentialMNISTModel

def quick_test_training(model, train_loader, device, epochs=3):
    """快速测试训练几个epoch"""
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-2)
    
    print(f"🧪 快速训练测试 ({epochs} epochs):")
    
    for epoch in range(epochs):
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}')
        for batch_idx, (data, target) in enumerate(pbar):
            if batch_idx >= 10:  # 只训练前10个批次
                break
                
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            # 计算准确率
            pred = output.argmax(dim=1)
            correct = pred.eq(target).sum().item()
            
            total_loss += loss.item()
            total_correct += correct
            total_samples += target.size(0)
            
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{100.*correct/target.size(0):.1f}%'
            })
        
        avg_loss = total_loss / min(10, len(train_loader))
        avg_acc = 100. * total_correct / total_samples
        
        print(f"   Epoch {epoch+1}: Loss={avg_loss:.4f}, Acc={avg_acc:.2f}%")
        
        # 检查是否有学习
        if epoch == 0:
            initial_acc = avg_acc
        elif epoch == epochs - 1:
            final_acc = avg_acc
            improvement = final_acc - initial_acc
            print(f"   📈 准确率改进: {improvement:+.2f}%")
            
            if improvement > 5.0:
                print("   ✅ 模型正在学习!")
                return True
            elif improvement > 1.0:
                print("   ⚠️  模型学习缓慢")
                return True
            else:
                print("   ❌ 模型没有明显学习")
                return False
    
    return False

def test_state_reset(model, device):
    """测试状态重置是否正常"""
    print(f"\n🔄 测试状态重置:")
    
    model.eval()
    batch_size = 4
    seq_len = 784
    
    # 创建测试数据
    x1 = torch.randn(batch_size, seq_len, 1, device=device)
    x2 = torch.randn(batch_size, seq_len, 1, device=device)
    
    with torch.no_grad():
        # 第一次前向传播
        output1 = model(x1)
        
        # 第二次前向传播（应该重置状态）
        output2 = model(x1)  # 使用相同输入
        
        # 检查输出是否一致
        diff = torch.abs(output1 - output2).max().item()
        print(f"   相同输入的输出差异: {diff:.6f}")
        
        if diff < 1e-5:
            print("   ✅ 状态重置正常")
            return True
        else:
            print("   ❌ 状态重置异常")
            return False

def test_gradient_flow(model, device):
    """测试梯度流动是否正常"""
    print(f"\n🌊 测试梯度流动:")
    
    model.train()
    criterion = nn.CrossEntropyLoss()
    
    # 创建测试数据
    batch_size = 4
    seq_len = 784
    x = torch.randn(batch_size, seq_len, 1, device=device)
    target = torch.randint(0, 10, (batch_size,), device=device)
    
    # 前向传播
    output = model(x)
    loss = criterion(output, target)
    
    # 反向传播
    loss.backward()
    
    # 检查梯度
    has_grad = False
    grad_norms = []
    
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            grad_norms.append(grad_norm)
            has_grad = True
            print(f"   {name}: grad_norm={grad_norm:.6f}")
        else:
            print(f"   {name}: 无梯度")
    
    if has_grad and len(grad_norms) > 0:
        avg_grad_norm = np.mean(grad_norms)
        print(f"   平均梯度范数: {avg_grad_norm:.6f}")
        
        if avg_grad_norm > 1e-6:
            print("   ✅ 梯度流动正常")
            return True
        else:
            print("   ⚠️  梯度过小")
            return False
    else:
        print("   ❌ 没有梯度")
        return False

def main():
    """主函数"""
    print("🔧 DH-SRNN修复验证测试")
    print("=" * 50)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🔧 使用设备: {device}")
    
    # 加载数据
    print("\n📊 加载测试数据...")
    train_loader, test_loader = load_sequential_mnist_data(
        batch_size=32,
        permute=False,
        seed=42
    )
    
    # 创建修复后的DH-SRNN模型
    print("\n🧠 创建DH-SRNN模型...")
    model = SequentialMNISTModel(
        model_type='dh_srnn',
        num_branches=4,
        tau_m_init=(0, 4),
        tau_n_init=(0, 4)
    ).to(device)
    
    print(f"📈 模型参数数量: {sum(p.numel() for p in model.parameters()):,}")
    
    # 测试状态重置
    reset_ok = test_state_reset(model, device)
    
    # 测试梯度流动
    grad_ok = test_gradient_flow(model, device)
    
    # 快速训练测试
    if reset_ok and grad_ok:
        print(f"\n🚀 开始快速训练测试...")
        learning_ok = quick_test_training(model, train_loader, device, epochs=3)
    else:
        learning_ok = False
    
    # 总结
    print(f"\n📋 测试总结:")
    print("=" * 30)
    print(f"状态重置: {'✅ 正常' if reset_ok else '❌ 异常'}")
    print(f"梯度流动: {'✅ 正常' if grad_ok else '❌ 异常'}")
    print(f"学习能力: {'✅ 正常' if learning_ok else '❌ 异常'}")
    
    if reset_ok and grad_ok and learning_ok:
        print(f"\n🎉 DH-SRNN修复成功!")
        print(f"💡 建议重新运行完整训练")
    else:
        print(f"\n⚠️  仍有问题需要解决")
        
        if not reset_ok:
            print("   - 检查状态重置逻辑")
        if not grad_ok:
            print("   - 检查梯度计算")
        if not learning_ok:
            print("   - 检查模型架构或超参数")

if __name__ == "__main__":
    main()
