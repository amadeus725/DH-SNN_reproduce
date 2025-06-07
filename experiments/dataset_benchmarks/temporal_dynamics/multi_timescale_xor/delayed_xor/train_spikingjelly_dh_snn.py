#!/usr/bin/env python3
"""
SpikingJelly版本的DH-SNN训练示例 - 延迟XOR任务
基于最小测试版本，确保稳定运行
"""

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import sys
sys.path.append('.')

# 使用最小测试版本的组件
from test_spikingjelly_minimal import (
    MultiGaussianSurrogate, 
    CustomParametricLIFNode,
    DendriticDenseLayer,
    ReadoutIntegrator,
    SimpleDH_SNN
)


def generate_delayed_xor_data(batch_size, seq_length, delay_steps, num_samples=1000):
    """生成延迟XOR数据"""
    X = torch.zeros(num_samples, seq_length, 2)
    y = torch.zeros(num_samples, dtype=torch.long)
    
    for i in range(num_samples):
        # 在前两个时间步设置输入
        X[i, 0, 0] = torch.randint(0, 2, (1,)).float()
        X[i, 1, 1] = torch.randint(0, 2, (1,)).float()
        
        # 计算XOR结果
        xor_result = int(X[i, 0, 0]) ^ int(X[i, 1, 1])
        y[i] = xor_result
    
    return X, y


def train_dh_snn_delayed_xor():
    """训练DH-SNN进行延迟XOR任务"""
    print("=== SpikingJelly DH-SNN 延迟XOR训练 ===\n")
    
    # 设备设置
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"使用设备: {device}")
    
    # 超参数
    input_dim = 2
    hidden_dim = 100
    output_dim = 2
    num_branches = 4
    seq_length = 20
    delay_steps = 10
    batch_size = 32
    num_epochs = 10
    learning_rate = 0.001
    
    # 生成数据
    print("生成训练数据...")
    X_train, y_train = generate_delayed_xor_data(batch_size, seq_length, delay_steps, 1000)
    X_test, y_test = generate_delayed_xor_data(batch_size, seq_length, delay_steps, 200)
    
    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    print(f"训练数据: {X_train.shape}, 测试数据: {X_test.shape}")
    
    # 创建网络
    print("创建DH-SNN网络...")
    net = SimpleDH_SNN(
        input_dim=input_dim,
        hidden_dims=[hidden_dim],
        output_dim=output_dim,
        num_branches=num_branches
    ).to(device)
    
    total_params = sum(p.numel() for p in net.parameters())
    trainable_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print(f"网络参数: 总计 {total_params}, 可训练 {trainable_params}")
    
    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
    
    # 训练循环
    print("\n开始训练...")
    for epoch in range(num_epochs):
        net.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            
            # 序列处理
            outputs = []
            net.reset_states()  # 重置网络状态
            
            for t in range(seq_length):
                output = net(data[:, t, :])
                outputs.append(output)
            
            # 使用最后的输出
            final_output = outputs[-1]
            loss = criterion(final_output, target)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            # 计算准确率
            pred = final_output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)
            
            if batch_idx % 10 == 0:
                print(f'Epoch {epoch+1}/{num_epochs}, Batch {batch_idx}, '
                      f'Loss: {loss.item():.4f}, Acc: {100.*correct/total:.2f}%')
        
        avg_loss = total_loss / len(train_loader)
        accuracy = 100. * correct / total
        print(f'Epoch {epoch+1} 完成 - 平均损失: {avg_loss:.4f}, 训练准确率: {accuracy:.2f}%')
        
        # 测试
        if epoch % 2 == 0:
            net.eval()
            test_correct = 0
            test_total = 0
            
            with torch.no_grad():
                for data, target in test_loader:
                    data, target = data.to(device), target.to(device)
                    
                    outputs = []
                    net.reset_states()
                    
                    for t in range(seq_length):
                        output = net(data[:, t, :])
                        outputs.append(output)
                    
                    final_output = outputs[-1]
                    pred = final_output.argmax(dim=1)
                    test_correct += pred.eq(target).sum().item()
                    test_total += target.size(0)
            
            test_accuracy = 100. * test_correct / test_total
            print(f'测试准确率: {test_accuracy:.2f}%\n')
    
    print("=== 训练完成 ===")


if __name__ == '__main__':
    try:
        train_dh_snn_delayed_xor()
    except Exception as e:
        print(f"训练过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
