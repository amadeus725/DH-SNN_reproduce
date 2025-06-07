#!/usr/bin/env python3
"""
调试实用DH-SRNN版本
"""

import os
import sys
import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm

# 添加项目路径
sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))

from data_loader import load_sequential_mnist_data

class SimpleDHSRNNCell(nn.Module):
    """
    简化的DH-SRNN单元 - 专注于调试
    """
    
    def __init__(self, input_size, hidden_size, num_branches=2):
        super(SimpleDHSRNNCell, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_branches = num_branches
        
        # 简单的分支结构
        total_input_size = input_size + hidden_size
        
        # 每个分支一个线性层
        self.branch_layers = nn.ModuleList([
            nn.Linear(total_input_size, hidden_size) 
            for _ in range(num_branches)
        ])
        
        # 固定的时间常数
        self.register_buffer('branch_alphas', torch.tensor([0.3, 0.7]))
        
        # 输出层
        self.output_layer = nn.Linear(hidden_size * num_branches, hidden_size)
        
    def forward(self, input_t, hidden_spike=None, branch_states=None):
        batch_size = input_t.size(0)
        device = input_t.device
        
        # 初始化
        if hidden_spike is None:
            hidden_spike = torch.zeros(batch_size, self.hidden_size, device=device)
        
        if branch_states is None:
            branch_states = torch.zeros(batch_size, self.hidden_size, self.num_branches, device=device)
        
        # 拼接输入
        combined_input = torch.cat([input_t, hidden_spike], dim=1)
        
        # 处理每个分支
        branch_outputs = []
        new_states = []
        
        for i in range(self.num_branches):
            # 分支计算
            branch_output = self.branch_layers[i](combined_input)
            
            # 状态更新
            alpha = self.branch_alphas[i]
            new_state = alpha * branch_states[:, :, i] + (1 - alpha) * branch_output
            
            branch_outputs.append(new_state)
            new_states.append(new_state)
        
        # 合并分支
        combined = torch.cat(branch_outputs, dim=1)
        output = self.output_layer(combined)
        
        # 简单的激活
        spike_output = torch.relu(output)
        
        # 更新状态
        new_branch_states = torch.stack(new_states, dim=2)
        
        return spike_output, new_branch_states

class SimpleSequentialMNISTModel(nn.Module):
    """
    简化的Sequential MNIST模型
    """
    
    def __init__(self, num_branches=2, sequence_subsample=4):
        super(SimpleSequentialMNISTModel, self).__init__()
        
        self.sequence_subsample = sequence_subsample  # 子采样减少序列长度
        
        # 简化的网络
        self.rnn1 = SimpleDHSRNNCell(1, 32, num_branches)
        self.rnn2 = SimpleDHSRNNCell(32, 64, num_branches)
        self.classifier = nn.Linear(64, 10)
        
    def forward(self, x):
        batch_size, seq_len, input_dim = x.shape
        device = x.device
        
        # 子采样序列以减少计算量
        if self.sequence_subsample > 1:
            x = x[:, ::self.sequence_subsample, :]
            seq_len = x.shape[1]
        
        # 初始化状态
        h1_spike = None
        h2_spike = None
        h1_states = None
        h2_states = None
        
        # 逐步处理 (使用截断减少内存)
        chunk_size = 50  # 每次处理50个时间步
        final_output = None
        
        for start_idx in range(0, seq_len, chunk_size):
            end_idx = min(start_idx + chunk_size, seq_len)
            
            # 分离梯度以截断BPTT
            if h1_spike is not None:
                h1_spike = h1_spike.detach()
                h1_states = h1_states.detach()
            if h2_spike is not None:
                h2_spike = h2_spike.detach()
                h2_states = h2_states.detach()
            
            # 处理当前块
            for t in range(start_idx, end_idx):
                x_t = x[:, t, :]
                
                h1_spike, h1_states = self.rnn1(x_t, h1_spike, h1_states)
                h2_spike, h2_states = self.rnn2(h1_spike, h2_spike, h2_states)
            
            # 保存最后的输出
            final_output = h2_spike
        
        # 分类
        output = self.classifier(final_output)
        
        return output

def test_model_basic():
    """测试模型基本功能"""
    print("🧪 测试模型基本功能...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"设备: {device}")
    
    # 创建模型
    model = SimpleSequentialMNISTModel(num_branches=2, sequence_subsample=4).to(device)
    print(f"模型参数: {sum(p.numel() for p in model.parameters()):,}")
    
    # 测试前向传播
    batch_size = 8
    seq_len = 784
    x = torch.randn(batch_size, seq_len, 1, device=device)
    
    print(f"输入形状: {x.shape}")
    
    try:
        start_time = time.time()
        output = model(x)
        forward_time = time.time() - start_time
        
        print(f"✅ 前向传播成功")
        print(f"输出形状: {output.shape}")
        print(f"前向时间: {forward_time:.2f}秒")
        print(f"输出范围: [{output.min().item():.3f}, {output.max().item():.3f}]")
        
        return True
        
    except Exception as e:
        print(f"❌ 前向传播失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_training_step():
    """测试训练步骤"""
    print("\n🏋️ 测试训练步骤...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 创建模型
    model = SimpleSequentialMNISTModel(num_branches=2, sequence_subsample=4).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    # 创建测试数据
    batch_size = 16
    seq_len = 784
    x = torch.randn(batch_size, seq_len, 1, device=device)
    target = torch.randint(0, 10, (batch_size,), device=device)
    
    try:
        # 训练步骤
        model.train()
        optimizer.zero_grad()
        
        start_time = time.time()
        output = model(x)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        training_time = time.time() - start_time
        
        # 计算准确率
        pred = output.argmax(dim=1)
        accuracy = pred.eq(target).float().mean().item()
        
        print(f"✅ 训练步骤成功")
        print(f"损失: {loss.item():.4f}")
        print(f"准确率: {accuracy*100:.1f}%")
        print(f"训练时间: {training_time:.2f}秒")
        
        return True
        
    except Exception as e:
        print(f"❌ 训练步骤失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def quick_training_test():
    """快速训练测试"""
    print("\n🚀 快速训练测试...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 加载少量数据
    print("加载数据...")
    train_loader, test_loader = load_sequential_mnist_data(
        batch_size=32,
        permute=False,
        seed=42
    )
    
    # 创建模型
    model = SimpleSequentialMNISTModel(num_branches=2, sequence_subsample=8).to(device)  # 更大的子采样
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=2e-3)
    
    print(f"模型参数: {sum(p.numel() for p in model.parameters()):,}")
    
    # 训练几个批次
    model.train()
    num_batches = 10
    
    print(f"训练 {num_batches} 个批次...")
    start_time = time.time()
    
    for batch_idx, (data, target) in enumerate(train_loader):
        if batch_idx >= num_batches:
            break
            
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        if batch_idx % 3 == 0:
            pred = output.argmax(dim=1)
            accuracy = pred.eq(target).float().mean().item()
            print(f"  批次 {batch_idx}: Loss={loss.item():.4f}, Acc={accuracy*100:.1f}%")
    
    training_time = time.time() - start_time
    time_per_batch = training_time / num_batches
    
    print(f"✅ 快速训练完成")
    print(f"总时间: {training_time:.2f}秒")
    print(f"平均时间: {time_per_batch:.2f}秒/批次")
    
    # 预估完整训练时间
    total_batches = len(train_loader)
    estimated_epoch_time = time_per_batch * total_batches / 60  # 分钟
    
    print(f"📊 预估一个epoch: {estimated_epoch_time:.1f}分钟")
    
    if estimated_epoch_time < 30:  # 30分钟以内认为合理
        print(f"✅ 训练时间合理，可以开始完整训练")
        return True
    else:
        print(f"⚠️  训练时间仍然较长，需要进一步优化")
        return False

def main():
    """主调试函数"""
    print("🔧 调试实用DH-SRNN版本")
    print("=" * 50)
    
    # 基本功能测试
    basic_ok = test_model_basic()
    if not basic_ok:
        print("❌ 基本功能测试失败，停止调试")
        return False
    
    # 训练步骤测试
    training_ok = test_training_step()
    if not training_ok:
        print("❌ 训练步骤测试失败，停止调试")
        return False
    
    # 快速训练测试
    quick_ok = quick_training_test()
    
    print(f"\n📋 调试总结:")
    print("=" * 30)
    print(f"基本功能: {'✅ 正常' if basic_ok else '❌ 失败'}")
    print(f"训练步骤: {'✅ 正常' if training_ok else '❌ 失败'}")
    print(f"训练速度: {'✅ 合理' if quick_ok else '⚠️  较慢'}")
    
    if basic_ok and training_ok:
        print(f"\n🎉 调试成功！模型可以正常工作")
        if quick_ok:
            print(f"💡 建议开始完整训练")
        else:
            print(f"💡 建议进一步优化后再训练")
        return True
    else:
        print(f"\n❌ 调试失败，需要修复问题")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
