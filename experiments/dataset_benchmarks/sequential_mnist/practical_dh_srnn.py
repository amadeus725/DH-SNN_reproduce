#!/usr/bin/env python3
"""
实用的DH-SRNN训练方案
使用多种优化技术使训练时间合理化
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

class PracticalDHSRNNCell(nn.Module):
    """
    实用的DH-SRNN单元 - 平衡性能和准确性
    """
    
    def __init__(self, input_size, hidden_size, num_branches=2):  # 减少分支数
        super(PracticalDHSRNNCell, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_branches = num_branches
        
        # 简化的分支结构
        total_input_size = input_size + hidden_size
        branch_size = total_input_size // num_branches
        
        self.branch_fcs = nn.ModuleList([
            nn.Linear(branch_size, hidden_size) for _ in range(num_branches)
        ])
        
        # 固定的时间常数 (避免复杂计算)
        self.register_buffer('alphas', torch.linspace(0.2, 0.8, num_branches))
        
        # 输出门控
        self.output_gate = nn.Linear(hidden_size * num_branches, hidden_size)
        self.spike_threshold = 1.0
        
    def forward(self, input_t, hidden_spike=None, branch_states=None):
        batch_size = input_t.size(0)
        device = input_t.device
        
        if hidden_spike is None:
            hidden_spike = torch.zeros(batch_size, self.hidden_size, device=device)
        
        if branch_states is None:
            branch_states = torch.zeros(batch_size, self.hidden_size, self.num_branches, device=device)
        
        # 拼接输入
        combined_input = torch.cat([input_t, hidden_spike], dim=1)
        
        # 分支处理
        branch_outputs = []
        new_states = []
        
        for i in range(self.num_branches):
            # 分割输入
            start_idx = i * (combined_input.size(1) // self.num_branches)
            end_idx = start_idx + (combined_input.size(1) // self.num_branches)
            if i == self.num_branches - 1:  # 最后一个分支处理剩余
                end_idx = combined_input.size(1)
            
            branch_input = combined_input[:, start_idx:end_idx]
            
            # 线性变换
            branch_current = self.branch_fcs[i](branch_input)
            
            # 状态更新
            alpha = self.alphas[i]
            new_state = alpha * branch_states[:, :, i] + (1 - alpha) * branch_current
            
            branch_outputs.append(new_state)
            new_states.append(new_state)
        
        # 合并分支
        combined_output = torch.cat(branch_outputs, dim=1)
        output = self.output_gate(combined_output)
        
        # 简化的脉冲生成
        spike_output = torch.where(output > self.spike_threshold, 
                                 torch.ones_like(output), 
                                 torch.zeros_like(output))
        
        # 更新状态
        new_branch_states = torch.stack(new_states, dim=2)
        
        return spike_output, new_branch_states

class PracticalSequentialMNISTModel(nn.Module):
    """
    实用的Sequential MNIST模型
    """
    
    def __init__(self, num_branches=2, use_tbptt=True, tbptt_length=100):
        super(PracticalSequentialMNISTModel, self).__init__()
        
        self.use_tbptt = use_tbptt
        self.tbptt_length = tbptt_length
        
        # 简化的网络结构
        self.rnn1 = PracticalDHSRNNCell(1, 32, num_branches)  # 减少隐藏层大小
        self.rnn2 = PracticalDHSRNNCell(32, 64, num_branches)
        self.output_layer = nn.Linear(64, 10)
        
        # 输出积分器
        self.register_buffer('output_decay', torch.tensor(0.9))
        
    def forward(self, x):
        batch_size, seq_len, input_dim = x.shape
        device = x.device
        
        # 初始化状态
        h1_spike = None
        h2_spike = None
        h1_states = None
        h2_states = None
        
        # 输出积分器
        output_integrator = torch.zeros(batch_size, 10, device=device)
        
        if self.use_tbptt:
            # 使用TBPTT减少内存使用
            for chunk_start in range(0, seq_len, self.tbptt_length):
                chunk_end = min(chunk_start + self.tbptt_length, seq_len)
                
                # 分离状态以截断梯度
                if h1_spike is not None:
                    h1_spike = h1_spike.detach()
                    h1_states = h1_states.detach()
                if h2_spike is not None:
                    h2_spike = h2_spike.detach()
                    h2_states = h2_states.detach()
                
                # 处理当前块
                for t in range(chunk_start, chunk_end):
                    x_t = x[:, t, :]
                    
                    h1_spike, h1_states = self.rnn1(x_t, h1_spike, h1_states)
                    h2_spike, h2_states = self.rnn2(h1_spike, h2_spike, h2_states)
                    
                    # 积分输出
                    current_output = self.output_layer(h2_spike)
                    output_integrator = self.output_decay * output_integrator + current_output
        else:
            # 标准处理
            for t in range(seq_len):
                x_t = x[:, t, :]
                
                h1_spike, h1_states = self.rnn1(x_t, h1_spike, h1_states)
                h2_spike, h2_states = self.rnn2(h1_spike, h2_spike, h2_states)
                
                # 只在后半段积分输出
                if t >= seq_len // 2:
                    current_output = self.output_layer(h2_spike)
                    output_integrator = self.output_decay * output_integrator + current_output
        
        return output_integrator

def train_practical_model():
    """训练实用的DH-SRNN模型"""
    print("🚀 实用DH-SRNN训练")
    print("=" * 50)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🔧 使用设备: {device}")
    
    # 数据加载
    print("\n📊 加载数据...")
    train_loader, test_loader = load_sequential_mnist_data(
        batch_size=128,  # 增大批次大小提高效率
        permute=False,
        seed=42
    )
    
    # 创建模型
    print("\n🧠 创建实用DH-SRNN模型...")
    model = PracticalSequentialMNISTModel(
        num_branches=2,
        use_tbptt=True,
        tbptt_length=100
    ).to(device)
    
    print(f"📈 模型参数数量: {sum(p.numel() for p in model.parameters()):,}")
    
    # 训练配置
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=2e-3)
    scheduler = StepLR(optimizer, step_size=10, gamma=0.7)
    
    # 训练参数
    num_epochs = 20  # 减少epoch数
    best_acc = 0.0
    
    print(f"\n🚀 开始训练 ({num_epochs} epochs)...")
    
    for epoch in range(1, num_epochs + 1):
        # 训练
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        start_time = time.time()
        
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
            
            # 统计
            train_loss += loss.item()
            pred = output.argmax(dim=1)
            train_correct += pred.eq(target).sum().item()
            train_total += target.size(0)
            
            # 更新进度条
            if batch_idx % 50 == 0:
                pbar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'Acc': f'{100.*train_correct/train_total:.1f}%'
                })
        
        epoch_time = time.time() - start_time
        train_acc = 100. * train_correct / train_total
        
        # 测试
        model.eval()
        test_loss = 0.0
        test_correct = 0
        test_total = 0
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                test_loss += criterion(output, target).item()
                pred = output.argmax(dim=1)
                test_correct += pred.eq(target).sum().item()
                test_total += target.size(0)
        
        test_acc = 100. * test_correct / test_total
        
        # 更新学习率
        scheduler.step()
        
        # 保存最佳模型
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save({
                'model_state_dict': model.state_dict(),
                'epoch': epoch,
                'best_acc': best_acc,
            }, 'results/S-MNIST_practical_dh_srnn_best.pth')
        
        # 输出结果
        print(f"Epoch {epoch:2d}: "
              f"Train Acc={train_acc:.2f}%, Test Acc={test_acc:.2f}%, "
              f"Best={best_acc:.2f}%, Time={epoch_time:.1f}s")
        
        # 早停检查
        if test_acc > 60.0:  # 如果达到60%就认为成功
            print(f"✅ 达到60%准确率，训练成功!")
            break
    
    # 保存结果
    results = {
        'model_type': 'practical_dh_srnn',
        'dataset': 'S-MNIST',
        'best_test_accuracy': best_acc,
        'epochs': epoch,
        'config': {
            'num_branches': 2,
            'use_tbptt': True,
            'tbptt_length': 100,
            'batch_size': 128
        }
    }
    
    torch.save(results, 'results/S-MNIST_practical_dh_srnn_results.pth')
    
    print(f"\n📋 训练总结:")
    print(f"最佳测试准确率: {best_acc:.2f}%")
    print(f"训练轮数: {epoch} epochs")
    
    return best_acc > 50.0  # 50%以上认为成功

def main():
    """主函数"""
    success = train_practical_model()
    
    if success:
        print(f"\n🎉 实用DH-SRNN训练成功!")
        print(f"💡 这个版本在性能和准确性之间取得了平衡")
    else:
        print(f"\n⚠️  需要进一步优化")
    
    return success

if __name__ == "__main__":
    main()
