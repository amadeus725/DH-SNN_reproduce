#!/usr/bin/env python3
"""
改进的多时间尺度XOR实验
使用更具挑战性的数据生成器和更精确的模型实现
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from typing import Dict, List, Tuple
import time

# 导入挑战性数据生成器
from challenging_data_generator import AdaptiveDifficultyGenerator

print("🕰️ 改进的多时间尺度XOR实验 - Figure 4b")
print("="*60)

class ImprovedVanillaSFNN(nn.Module):
    """改进的Vanilla SFNN模型"""
    
    def __init__(self, input_size=40, hidden_size=64, output_size=1):
        super().__init__()
        
        self.dense = nn.Linear(input_size, hidden_size)
        self.output = nn.Linear(hidden_size, output_size)
        
        # 固定的中等时间常数
        self.tau = nn.Parameter(torch.ones(hidden_size) * 2.0)
        
        self.register_buffer('mem', torch.zeros(1, hidden_size))
        
    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        
        self.mem = torch.zeros(batch_size, self.mem.size(1)).to(x.device)
        outputs = []
        
        for t in range(seq_len):
            input_current = self.dense(x[:, t, :])
            alpha = torch.sigmoid(self.tau)
            self.mem = alpha * self.mem + (1 - alpha) * input_current
            output = torch.sigmoid(self.output(self.mem))
            outputs.append(output)
        
        return torch.stack(outputs, dim=1)

class ImprovedOneBranchDH_SFNN(nn.Module):
    """改进的单分支DH-SFNN模型"""
    
    def __init__(self, input_size=40, hidden_size=64, output_size=1, tau_init='medium'):
        super().__init__()
        
        self.dense = nn.Linear(input_size, hidden_size)
        self.output = nn.Linear(hidden_size, output_size)
        
        # 时间常数初始化
        self.tau_m = nn.Parameter(torch.zeros(hidden_size))
        self.tau_n = nn.Parameter(torch.zeros(hidden_size))
        
        # 膜电位时间常数 (Medium)
        nn.init.uniform_(self.tau_m, 0.0, 4.0)
        
        # 树突时间常数根据初始化类型
        if tau_init == 'small':
            nn.init.uniform_(self.tau_n, -4.0, 0.0)
        elif tau_init == 'large':
            nn.init.uniform_(self.tau_n, 2.0, 6.0)
        else:  # medium
            nn.init.uniform_(self.tau_n, 0.0, 4.0)
        
        self.register_buffer('mem', torch.zeros(1, hidden_size))
        self.register_buffer('d_current', torch.zeros(1, hidden_size))
        
    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        
        self.mem = torch.zeros(batch_size, self.mem.size(1)).to(x.device)
        self.d_current = torch.zeros(batch_size, self.d_current.size(1)).to(x.device)
        outputs = []
        
        for t in range(seq_len):
            # 树突电流更新
            d_input = self.dense(x[:, t, :])
            beta = torch.sigmoid(self.tau_n)
            self.d_current = beta * self.d_current + (1 - beta) * d_input
            
            # 膜电位更新
            alpha = torch.sigmoid(self.tau_m)
            self.mem = alpha * self.mem + (1 - alpha) * self.d_current
            
            output = torch.sigmoid(self.output(self.mem))
            outputs.append(output)
        
        return torch.stack(outputs, dim=1)

class ImprovedTwoBranchDH_SFNN(nn.Module):
    """改进的双分支DH-SFNN模型"""
    
    def __init__(self, input_size=40, hidden_size=64, output_size=1, beneficial_init=True, learnable=True):
        super().__init__()
        
        # 分支连接
        self.branch1_dense = nn.Linear(input_size//2, hidden_size)  # Signal 1
        self.branch2_dense = nn.Linear(input_size//2, hidden_size)  # Signal 2
        self.output = nn.Linear(hidden_size, output_size)
        
        # 时间常数
        self.tau_m = nn.Parameter(torch.zeros(hidden_size))
        self.tau_n_branch1 = nn.Parameter(torch.zeros(hidden_size))
        self.tau_n_branch2 = nn.Parameter(torch.zeros(hidden_size))
        
        # 膜电位时间常数 (Medium)
        nn.init.uniform_(self.tau_m, 0.0, 4.0)
        
        # 有益初始化
        if beneficial_init:
            nn.init.uniform_(self.tau_n_branch1, 2.0, 6.0)  # Large for long-term
            nn.init.uniform_(self.tau_n_branch2, -4.0, 0.0)  # Small for short-term
        else:
            nn.init.uniform_(self.tau_n_branch1, 0.0, 4.0)  # Medium
            nn.init.uniform_(self.tau_n_branch2, 0.0, 4.0)  # Medium
        
        # 设置可学习性
        self.tau_m.requires_grad = learnable
        self.tau_n_branch1.requires_grad = learnable
        self.tau_n_branch2.requires_grad = learnable
        
        self.register_buffer('mem', torch.zeros(1, hidden_size))
        self.register_buffer('d1_current', torch.zeros(1, hidden_size))
        self.register_buffer('d2_current', torch.zeros(1, hidden_size))
        
    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        
        self.mem = torch.zeros(batch_size, self.mem.size(1)).to(x.device)
        self.d1_current = torch.zeros(batch_size, self.d1_current.size(1)).to(x.device)
        self.d2_current = torch.zeros(batch_size, self.d2_current.size(1)).to(x.device)
        outputs = []
        
        for t in range(seq_len):
            # 分支输入
            branch1_input = x[:, t, :20]  # Signal 1
            branch2_input = x[:, t, 20:]  # Signal 2
            
            # 分支1: 长期记忆
            d1_in = self.branch1_dense(branch1_input)
            beta1 = torch.sigmoid(self.tau_n_branch1)
            self.d1_current = beta1 * self.d1_current + (1 - beta1) * d1_in
            
            # 分支2: 快速响应
            d2_in = self.branch2_dense(branch2_input)
            beta2 = torch.sigmoid(self.tau_n_branch2)
            self.d2_current = beta2 * self.d2_current + (1 - beta2) * d2_in
            
            # 整合
            total_input = self.d1_current + self.d2_current
            
            # 膜电位更新
            alpha = torch.sigmoid(self.tau_m)
            self.mem = alpha * self.mem + (1 - alpha) * total_input
            
            output = torch.sigmoid(self.output(self.mem))
            outputs.append(output)
        
        return torch.stack(outputs, dim=1)
    
    def get_time_constants(self):
        """获取时间常数"""
        return {
            'tau_m': torch.sigmoid(self.tau_m).detach().cpu(),
            'tau_n_branch1': torch.sigmoid(self.tau_n_branch1).detach().cpu(),
            'tau_n_branch2': torch.sigmoid(self.tau_n_branch2).detach().cpu()
        }

def improved_train_model(model, train_data, train_targets, test_data, test_targets, model_name, epochs=100):
    """改进的训练函数"""
    print(f"🏋️ 训练 {model_name}")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    
    # 优化器设置
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)
    criterion = nn.BCELoss()
    
    best_acc = 0.0
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        num_batches = 0
        
        # 训练
        batch_size = 16
        for i in range(0, len(train_data), batch_size):
            batch_data = train_data[i:i+batch_size].to(device)
            batch_targets = train_targets[i:i+batch_size].to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_data)
            
            # 只在有目标的时间步计算损失
            mask = (batch_targets.sum(dim=-1) > 0).unsqueeze(-1)
            if mask.sum() > 0:
                masked_outputs = outputs[mask]
                masked_targets = batch_targets[mask]
                loss = criterion(masked_outputs, masked_targets)
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
        
        scheduler.step()
        
        # 测试
        if epoch % 10 == 0 or epoch == epochs - 1:
            model.eval()
            with torch.no_grad():
                test_outputs = model(test_data.to(device))
                test_targets_device = test_targets.to(device)
                
                # 评估准确率
                mask = (test_targets_device.sum(dim=-1) > 0).unsqueeze(-1)
                if mask.sum() > 0:
                    masked_outputs = test_outputs[mask]
                    masked_targets = test_targets_device[mask]
                    
                    test_pred = (masked_outputs > 0.5).float()
                    acc = (test_pred == masked_targets).float().mean().item() * 100
                else:
                    acc = 0.0
                
                if acc > best_acc:
                    best_acc = acc
                
                avg_loss = total_loss / max(num_batches, 1)
                print(f"  Epoch {epoch+1:3d}: Loss={avg_loss:.4f}, Acc={acc:.1f}%, Best={best_acc:.1f}%")
    
    return best_acc

def main():
    """主实验函数"""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"🔧 使用设备: {device}")
    
    # 使用挑战性数据生成器
    print("\n📊 生成挑战性多时间尺度XOR数据...")
    generator = AdaptiveDifficultyGenerator(device=device, difficulty_level='medium')
    train_data, train_targets = generator.generate_dataset(300)
    test_data, test_targets = generator.generate_dataset(100)
    
    print(f"✅ 数据生成完成: 训练{train_data.shape}, 测试{test_data.shape}")
    print(f"   训练数据发放率: {train_data.mean():.4f}")
    print(f"   目标覆盖率: {(train_targets > 0).float().mean():.4f}")
    
    # 实验配置
    experiments = [
        ("Vanilla SFNN", lambda: ImprovedVanillaSFNN()),
        ("1-Branch DH-SFNN (Small)", lambda: ImprovedOneBranchDH_SFNN(tau_init='small')),
        ("1-Branch DH-SFNN (Large)", lambda: ImprovedOneBranchDH_SFNN(tau_init='large')),
        ("2-Branch DH-SFNN (Beneficial)", lambda: ImprovedTwoBranchDH_SFNN(beneficial_init=True, learnable=True)),
        ("2-Branch DH-SFNN (Fixed)", lambda: ImprovedTwoBranchDH_SFNN(beneficial_init=True, learnable=False)),
        ("2-Branch DH-SFNN (Random)", lambda: ImprovedTwoBranchDH_SFNN(beneficial_init=False, learnable=True)),
    ]
    
    results = {}
    
    print(f"\n🧪 开始改进实验 (每个配置3次试验)...")
    
    for exp_name, model_creator in experiments:
        print(f"\n📊 实验: {exp_name}")
        print("="*50)
        
        trial_results = []
        
        for trial in range(3):
            print(f"  🔄 试验 {trial+1}/3")
            model = model_creator()
            
            acc = improved_train_model(model, train_data, train_targets, test_data, test_targets, 
                                     f"{exp_name}_trial_{trial+1}", epochs=60)
            trial_results.append(acc)
            
            print(f"    ✅ 试验{trial+1}完成: {acc:.1f}%")
        
        mean_acc = np.mean(trial_results)
        std_acc = np.std(trial_results)
        results[exp_name] = {
            'mean': mean_acc, 
            'std': std_acc, 
            'trials': trial_results
        }
        
        print(f"  📈 最终结果: {mean_acc:.1f}% ± {std_acc:.1f}%")
    
    # 显示最终结果
    print(f"\n🎉 改进实验完成! Figure 4b结果:")
    print("="*60)
    for exp_name, result in results.items():
        mean_acc = result['mean']
        std_acc = result['std']
        print(f"{exp_name:30s}: {mean_acc:5.1f}% ± {std_acc:4.1f}%")
    
    # 保存结果
    os.makedirs("results", exist_ok=True)
    torch.save(results, "results/improved_figure4b_results.pth")
    print(f"\n💾 结果已保存到: results/improved_figure4b_results.pth")
    
    return results

if __name__ == '__main__':
    try:
        results = main()
        print(f"\n🏁 改进实验成功完成!")
    except Exception as e:
        print(f"\n❌ 实验失败: {e}")
        import traceback
        traceback.print_exc()
