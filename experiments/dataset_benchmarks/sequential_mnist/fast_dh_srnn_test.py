#!/usr/bin/env python3
"""
快速DH-SRNN性能测试和优化
"""

import os
import sys
import time
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm

# 添加项目路径
sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))

from data_loader import load_sequential_mnist_data

class FastDHSRNNCell(nn.Module):
    """
    优化的DH-SRNN单元 - 专注于性能
    """
    
    def __init__(self, input_size, hidden_size, num_branches=4):
        super(FastDHSRNNCell, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_branches = num_branches
        
        # 简化的分支连接 - 使用单个大矩阵
        total_input_size = input_size + hidden_size
        self.branch_fc = nn.Linear(total_input_size, hidden_size * num_branches)
        
        # 分支时间常数 (固定值，避免复杂计算)
        self.register_buffer('branch_alphas', torch.tensor([0.1, 0.3, 0.5, 0.7]))
        
        # 输出层
        self.output_fc = nn.Linear(hidden_size * num_branches, hidden_size)
        
        # 激活函数 (简化的脉冲函数)
        self.spike_fn = torch.nn.Threshold(1.0, 0.0)
        
    def forward(self, input_t, hidden_spike=None, branch_states=None):
        """优化的前向传播"""
        batch_size = input_t.size(0)
        device = input_t.device
        
        # 初始化
        if hidden_spike is None:
            hidden_spike = torch.zeros(batch_size, self.hidden_size, device=device)
        
        if branch_states is None:
            branch_states = torch.zeros(batch_size, self.hidden_size, self.num_branches, device=device)
        
        # 拼接输入
        combined_input = torch.cat([input_t, hidden_spike], dim=1)
        
        # 一次性计算所有分支
        branch_outputs = self.branch_fc(combined_input)  # [batch, hidden*branches]
        branch_outputs = branch_outputs.view(batch_size, self.hidden_size, self.num_branches)
        
        # 更新分支状态 (向量化操作)
        alphas = self.branch_alphas.view(1, 1, -1)  # [1, 1, branches]
        new_branch_states = alphas * branch_states + (1 - alphas) * branch_outputs
        
        # 合并分支输出
        combined_output = new_branch_states.view(batch_size, -1)  # [batch, hidden*branches]
        output = self.output_fc(combined_output)
        
        # 简化的脉冲生成
        spike_output = self.spike_fn(output)
        
        return spike_output, new_branch_states

class FastSequentialMNISTModel(nn.Module):
    """
    优化的Sequential MNIST模型
    """
    
    def __init__(self, num_branches=4):
        super(FastSequentialMNISTModel, self).__init__()
        
        # 简化的网络结构
        self.rnn1 = FastDHSRNNCell(1, 64, num_branches)
        self.rnn2 = FastDHSRNNCell(64, 128, num_branches)  # 减少隐藏层大小
        self.output_layer = nn.Linear(128, 10)
        
    def forward(self, x):
        """优化的前向传播"""
        batch_size, seq_len, input_dim = x.shape
        device = x.device
        
        # 初始化状态
        h1_spike = None
        h2_spike = None
        h1_states = None
        h2_states = None
        
        # 累积输出 (避免在每个时间步计算输出)
        output_accumulator = torch.zeros(batch_size, 10, device=device)
        
        # 逐时间步处理 (考虑使用TBPTT减少内存)
        for t in range(seq_len):
            x_t = x[:, t, :]
            
            # 第一层
            h1_spike, h1_states = self.rnn1(x_t, h1_spike, h1_states)
            
            # 第二层
            h2_spike, h2_states = self.rnn2(h1_spike, h2_spike, h2_states)
            
            # 累积输出 (只在最后几个时间步)
            if t >= seq_len - 50:  # 只使用最后50个时间步
                output_accumulator += self.output_layer(h2_spike)
        
        return output_accumulator / 50  # 平均

def benchmark_models():
    """对比不同模型的性能"""
    print("⚡ DH-SRNN性能基准测试")
    print("=" * 50)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🔧 使用设备: {device}")
    
    # 加载小批量数据进行测试
    print("\n📊 加载测试数据...")
    train_loader, _ = load_sequential_mnist_data(
        batch_size=32,  # 减小批次大小
        permute=False,
        seed=42
    )
    
    # 只取前几个批次进行测试
    test_batches = []
    for i, (data, target) in enumerate(train_loader):
        if i >= 5:  # 只测试5个批次
            break
        test_batches.append((data.to(device), target.to(device)))
    
    print(f"   测试批次数: {len(test_batches)}")
    print(f"   每批次大小: {test_batches[0][0].shape[0]}")
    print(f"   序列长度: {test_batches[0][0].shape[1]}")
    
    # 测试原始DH-SRNN
    print(f"\n🧪 测试原始DH-SRNN...")
    try:
        from models import SequentialMNISTModel
        
        original_model = SequentialMNISTModel(
            model_type='dh_srnn',
            num_branches=4
        ).to(device)
        
        start_time = time.time()
        for data, target in test_batches:
            with torch.no_grad():
                output = original_model(data)
        original_time = time.time() - start_time
        
        print(f"   ✅ 原始DH-SRNN: {original_time:.2f}秒 ({original_time/len(test_batches):.2f}秒/批次)")
        
    except Exception as e:
        print(f"   ❌ 原始DH-SRNN测试失败: {e}")
        original_time = float('inf')
    
    # 测试优化DH-SRNN
    print(f"\n🚀 测试优化DH-SRNN...")
    try:
        fast_model = FastSequentialMNISTModel(num_branches=4).to(device)
        
        start_time = time.time()
        for data, target in test_batches:
            with torch.no_grad():
                output = fast_model(data)
        fast_time = time.time() - start_time
        
        print(f"   ✅ 优化DH-SRNN: {fast_time:.2f}秒 ({fast_time/len(test_batches):.2f}秒/批次)")
        
        # 计算加速比
        if original_time != float('inf'):
            speedup = original_time / fast_time
            print(f"   📈 加速比: {speedup:.1f}x")
        
    except Exception as e:
        print(f"   ❌ 优化DH-SRNN测试失败: {e}")
        fast_time = float('inf')
    
    # 测试Vanilla SRNN作为基准
    print(f"\n📊 测试Vanilla SRNN (基准)...")
    try:
        from models import SequentialMNISTModel
        
        vanilla_model = SequentialMNISTModel(
            model_type='vanilla_srnn'
        ).to(device)
        
        start_time = time.time()
        for data, target in test_batches:
            with torch.no_grad():
                output = vanilla_model(data)
        vanilla_time = time.time() - start_time
        
        print(f"   ✅ Vanilla SRNN: {vanilla_time:.2f}秒 ({vanilla_time/len(test_batches):.2f}秒/批次)")
        
    except Exception as e:
        print(f"   ❌ Vanilla SRNN测试失败: {e}")
        vanilla_time = float('inf')
    
    # 总结
    print(f"\n📋 性能总结:")
    print("=" * 30)
    if vanilla_time != float('inf'):
        print(f"Vanilla SRNN:  {vanilla_time/len(test_batches):.2f}秒/批次 (基准)")
    if original_time != float('inf'):
        print(f"原始DH-SRNN:   {original_time/len(test_batches):.2f}秒/批次")
    if fast_time != float('inf'):
        print(f"优化DH-SRNN:   {fast_time/len(test_batches):.2f}秒/批次")
    
    # 建议
    print(f"\n💡 优化建议:")
    if original_time > 10 * vanilla_time:
        print("1. 原始DH-SRNN性能严重不足，建议使用优化版本")
    if fast_time < original_time / 2:
        print("2. 优化版本显著提升性能")
    print("3. 考虑使用TBPTT减少内存使用")
    print("4. 考虑减少序列长度或使用子采样")
    print("5. 考虑使用更简单的DH-SRNN实现")

def quick_training_test():
    """快速训练测试"""
    print(f"\n🏃 快速训练测试 (优化版本)...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 小规模数据
    train_loader, test_loader = load_sequential_mnist_data(
        batch_size=64,
        permute=False,
        seed=42
    )
    
    # 优化模型
    model = FastSequentialMNISTModel(num_branches=4).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    print(f"📈 模型参数: {sum(p.numel() for p in model.parameters()):,}")
    
    # 训练一个epoch
    model.train()
    start_time = time.time()
    total_batches = 0
    
    for batch_idx, (data, target) in enumerate(train_loader):
        if batch_idx >= 50:  # 只训练50个批次
            break
            
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        total_batches += 1
        
        if batch_idx % 10 == 0:
            print(f"   批次 {batch_idx}: Loss={loss.item():.4f}")
    
    training_time = time.time() - start_time
    time_per_batch = training_time / total_batches
    
    print(f"   ✅ 50批次训练完成: {training_time:.2f}秒")
    print(f"   ⚡ 平均速度: {time_per_batch:.2f}秒/批次")
    
    # 预估完整训练时间
    total_batches_per_epoch = len(train_loader)
    estimated_epoch_time = time_per_batch * total_batches_per_epoch / 60  # 分钟
    estimated_50_epochs = estimated_epoch_time * 50 / 60  # 小时
    
    print(f"   📊 预估一个epoch: {estimated_epoch_time:.1f}分钟")
    print(f"   📊 预估50个epochs: {estimated_50_epochs:.1f}小时")
    
    if estimated_50_epochs < 12:
        print(f"   ✅ 训练时间合理")
        return True
    else:
        print(f"   ⚠️  训练时间仍然过长")
        return False

def main():
    """主函数"""
    print("🔍 DH-SRNN性能诊断和优化")
    print("=" * 60)
    
    # 性能基准测试
    benchmark_models()
    
    # 快速训练测试
    success = quick_training_test()
    
    return success

if __name__ == "__main__":
    main()
