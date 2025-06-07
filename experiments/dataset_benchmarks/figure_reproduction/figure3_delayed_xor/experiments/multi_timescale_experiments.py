#!/usr/bin/env python3
"""
简化的多时间尺度XOR测试
验证基础框架是否工作
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

# 添加路径
# sys.path.append removed during restructure

print("🚀 简化多时间尺度XOR测试")
print("="*50)

class SimpleMultiTimescaleXORGenerator:
    """简化的多时间尺度XOR数据生成器"""
    
    def __init__(self, device='cpu'):
        self.device = device
        self.total_time = 200
        self.input_size = 20
        
    def generate_sample(self):
        """生成单个样本"""
        # 创建输入数据
        input_data = torch.zeros(self.total_time, self.input_size)
        target_data = torch.zeros(self.total_time, 1)
        
        # Signal 1: 低频信号 (时间步 20-40)
        signal1_type = np.random.choice([0, 1])  # 0=低, 1=高
        if signal1_type == 1:
            input_data[20:40, :10] = torch.rand(20, 10) * 0.5  # 高发放率
        else:
            input_data[20:40, :10] = torch.rand(20, 10) * 0.1  # 低发放率
        
        # Signal 2序列: 高频信号
        signal2_times = [60, 100, 140]  # 三个Signal 2
        xor_results = []
        
        for i, start_time in enumerate(signal2_times):
            signal2_type = np.random.choice([0, 1])
            if signal2_type == 1:
                input_data[start_time:start_time+15, 10:] = torch.rand(15, 10) * 0.5
            else:
                input_data[start_time:start_time+15, 10:] = torch.rand(15, 10) * 0.1
            
            # XOR结果
            xor_result = signal1_type ^ signal2_type
            xor_results.append(xor_result)
            
            # 设置目标输出
            target_data[start_time+15:start_time+25, 0] = xor_result
        
        return input_data.to(self.device), target_data.to(self.device)
    
    def generate_dataset(self, num_samples):
        """生成数据集"""
        inputs = []
        targets = []
        
        for _ in range(num_samples):
            input_data, target_data = self.generate_sample()
            inputs.append(input_data)
            targets.append(target_data)
        
        return torch.stack(inputs), torch.stack(targets)

class SimpleTwoBranchModel(nn.Module):
    """简化的双分支模型"""
    
    def __init__(self, input_size=20, hidden_size=32, output_size=1):
        super().__init__()
        
        # 分支1: 处理Signal 1 (长期记忆)
        self.branch1_dense = nn.Linear(input_size//2, hidden_size)
        self.branch1_tau = nn.Parameter(torch.ones(hidden_size) * 3.0)  # 大时间常数
        
        # 分支2: 处理Signal 2 (快速响应)  
        self.branch2_dense = nn.Linear(input_size//2, hidden_size)
        self.branch2_tau = nn.Parameter(torch.ones(hidden_size) * 0.5)  # 小时间常数
        
        # 输出层
        self.output = nn.Linear(hidden_size, output_size)
        
        # 神经元状态
        self.register_buffer('mem', torch.zeros(1, hidden_size))
        self.register_buffer('d1_current', torch.zeros(1, hidden_size))
        self.register_buffer('d2_current', torch.zeros(1, hidden_size))
        
    def forward(self, x):
        batch_size, seq_len, input_size = x.shape
        
        # 重置状态
        self.mem = torch.zeros(batch_size, self.mem.size(1)).to(x.device)
        self.d1_current = torch.zeros(batch_size, self.d1_current.size(1)).to(x.device)
        self.d2_current = torch.zeros(batch_size, self.d2_current.size(1)).to(x.device)
        
        outputs = []
        
        for t in range(seq_len):
            # 分离输入到两个分支
            branch1_input = x[:, t, :input_size//2]  # Signal 1
            branch2_input = x[:, t, input_size//2:]  # Signal 2
            
            # 分支1处理 (长期记忆)
            d1_in = self.branch1_dense(branch1_input)
            alpha1 = torch.sigmoid(self.branch1_tau)
            self.d1_current = alpha1 * self.d1_current + (1 - alpha1) * d1_in
            
            # 分支2处理 (快速响应)
            d2_in = self.branch2_dense(branch2_input)
            alpha2 = torch.sigmoid(self.branch2_tau)
            self.d2_current = alpha2 * self.d2_current + (1 - alpha2) * d2_in
            
            # 整合两个分支
            total_input = self.d1_current + self.d2_current
            
            # 膜电位更新
            alpha_m = 0.8  # 固定膜时间常数
            self.mem = alpha_m * self.mem + (1 - alpha_m) * total_input
            
            # 输出
            output = torch.sigmoid(self.output(self.mem))
            outputs.append(output)
        
        return torch.stack(outputs, dim=1)

class SimpleVanillaModel(nn.Module):
    """简化的Vanilla模型"""
    
    def __init__(self, input_size=20, hidden_size=32, output_size=1):
        super().__init__()
        
        self.dense = nn.Linear(input_size, hidden_size)
        self.output = nn.Linear(hidden_size, output_size)
        self.tau = nn.Parameter(torch.ones(hidden_size) * 2.0)
        
        self.register_buffer('mem', torch.zeros(1, hidden_size))
        
    def forward(self, x):
        batch_size, seq_len, input_size = x.shape
        
        self.mem = torch.zeros(batch_size, self.mem.size(1)).to(x.device)
        outputs = []
        
        for t in range(seq_len):
            input_current = self.dense(x[:, t, :])
            alpha = torch.sigmoid(self.tau)
            self.mem = alpha * self.mem + (1 - alpha) * input_current
            output = torch.sigmoid(self.output(self.mem))
            outputs.append(output)
        
        return torch.stack(outputs, dim=1)

def quick_test():
    """快速测试"""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"🔧 使用设备: {device}")
    
    # 1. 测试数据生成
    print("\n📊 测试数据生成...")
    generator = SimpleMultiTimescaleXORGenerator(device)
    train_data, train_targets = generator.generate_dataset(100)
    test_data, test_targets = generator.generate_dataset(20)
    
    print(f"  ✅ 训练数据: {train_data.shape}")
    print(f"  ✅ 测试数据: {test_data.shape}")
    
    # 2. 测试模型
    print("\n🏗️ 测试模型...")
    vanilla_model = SimpleVanillaModel().to(device)
    two_branch_model = SimpleTwoBranchModel().to(device)
    
    print(f"  ✅ Vanilla模型参数: {sum(p.numel() for p in vanilla_model.parameters())}")
    print(f"  ✅ 双分支模型参数: {sum(p.numel() for p in two_branch_model.parameters())}")
    
    # 3. 测试前向传播
    print("\n🔄 测试前向传播...")
    with torch.no_grad():
        vanilla_out = vanilla_model(test_data[:5])
        two_branch_out = two_branch_model(test_data[:5])
        
        print(f"  ✅ Vanilla输出: {vanilla_out.shape}")
        print(f"  ✅ 双分支输出: {two_branch_out.shape}")
    
    # 4. 简单训练测试
    print("\n🏋️ 简单训练测试...")
    
    def train_model(model, model_name, epochs=10):
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        criterion = nn.MSELoss()
        
        model.train()
        for epoch in range(epochs):
            total_loss = 0
            for i in range(0, len(train_data), 10):
                batch_data = train_data[i:i+10]
                batch_targets = train_targets[i:i+10]
                
                optimizer.zero_grad()
                outputs = model(batch_data)
                loss = criterion(outputs, batch_targets)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            if epoch % 5 == 0:
                print(f"    {model_name} Epoch {epoch}: Loss = {total_loss/10:.4f}")
        
        return total_loss/10
    
    vanilla_loss = train_model(vanilla_model, "Vanilla", epochs=10)
    two_branch_loss = train_model(two_branch_model, "双分支", epochs=10)
    
    # 5. 时间常数分析
    print("\n🔍 时间常数分析...")
    with torch.no_grad():
        branch1_tau = torch.sigmoid(two_branch_model.branch1_tau).mean()
        branch2_tau = torch.sigmoid(two_branch_model.branch2_tau).mean()
        vanilla_tau = torch.sigmoid(vanilla_model.tau).mean()
        
        print(f"  Vanilla时间常数: {vanilla_tau:.3f}")
        print(f"  分支1时间常数: {branch1_tau:.3f} (长期记忆)")
        print(f"  分支2时间常数: {branch2_tau:.3f} (快速响应)")
        
        if branch1_tau > branch2_tau:
            print("  ✅ 时间常数分化正确: 分支1 > 分支2")
        else:
            print("  ⚠️ 时间常数分化可能有问题")
    
    print("\n🎉 简化测试完成!")
    print(f"📊 最终损失对比: Vanilla={vanilla_loss:.4f}, 双分支={two_branch_loss:.4f}")
    
    return True

if __name__ == '__main__':
    try:
        success = quick_test()
        print(f"\n🏁 测试结果: {'✅ 成功' if success else '❌ 失败'}")
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
