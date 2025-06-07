#!/usr/bin/env python3
"""
执行最终的挑战性多时间尺度XOR实验
使用UltraChallengingXORGenerator进行完整的Figure 4b实验
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import os
import time
from final_challenging_experiment import UltraChallengingXORGenerator

print("🚀 执行最终挑战性多时间尺度XOR实验")
print("="*60)

class FinalVanillaSFNN(nn.Module):
    """最终版Vanilla SFNN"""
    
    def __init__(self, input_size=40, hidden_size=64, output_size=1):
        super().__init__()
        self.dense = nn.Linear(input_size, hidden_size)
        self.output = nn.Linear(hidden_size, output_size)
        self.tau = nn.Parameter(torch.ones(hidden_size) * 2.0)  # Medium初始化
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

class FinalOneBranchDH_SFNN(nn.Module):
    """最终版单分支DH-SFNN"""
    
    def __init__(self, input_size=40, hidden_size=64, output_size=1, tau_init='medium'):
        super().__init__()
        self.dense = nn.Linear(input_size, hidden_size)
        self.output = nn.Linear(hidden_size, output_size)
        
        self.tau_m = nn.Parameter(torch.zeros(hidden_size))
        self.tau_n = nn.Parameter(torch.zeros(hidden_size))
        
        nn.init.uniform_(self.tau_m, 0.0, 4.0)  # Medium
        
        if tau_init == 'small':
            nn.init.uniform_(self.tau_n, -4.0, 0.0)
        elif tau_init == 'large':
            nn.init.uniform_(self.tau_n, 2.0, 6.0)
        else:
            nn.init.uniform_(self.tau_n, 0.0, 4.0)
        
        self.register_buffer('mem', torch.zeros(1, hidden_size))
        self.register_buffer('d_current', torch.zeros(1, hidden_size))
        
    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        self.mem = torch.zeros(batch_size, self.mem.size(1)).to(x.device)
        self.d_current = torch.zeros(batch_size, self.d_current.size(1)).to(x.device)
        outputs = []
        
        for t in range(seq_len):
            d_input = self.dense(x[:, t, :])
            beta = torch.sigmoid(self.tau_n)
            self.d_current = beta * self.d_current + (1 - beta) * d_input
            
            alpha = torch.sigmoid(self.tau_m)
            self.mem = alpha * self.mem + (1 - alpha) * self.d_current
            
            output = torch.sigmoid(self.output(self.mem))
            outputs.append(output)
        
        return torch.stack(outputs, dim=1)

class FinalTwoBranchDH_SFNN(nn.Module):
    """最终版双分支DH-SFNN"""
    
    def __init__(self, input_size=40, hidden_size=64, output_size=1, beneficial_init=True, learnable=True):
        super().__init__()
        
        self.branch1_dense = nn.Linear(input_size//2, hidden_size)
        self.branch2_dense = nn.Linear(input_size//2, hidden_size)
        self.output = nn.Linear(hidden_size, output_size)
        
        self.tau_m = nn.Parameter(torch.zeros(hidden_size))
        self.tau_n_branch1 = nn.Parameter(torch.zeros(hidden_size))
        self.tau_n_branch2 = nn.Parameter(torch.zeros(hidden_size))
        
        nn.init.uniform_(self.tau_m, 0.0, 4.0)
        
        if beneficial_init:
            nn.init.uniform_(self.tau_n_branch1, 2.0, 6.0)  # Large for long-term
            nn.init.uniform_(self.tau_n_branch2, -4.0, 0.0)  # Small for short-term
        else:
            nn.init.uniform_(self.tau_n_branch1, 0.0, 4.0)
            nn.init.uniform_(self.tau_n_branch2, 0.0, 4.0)
        
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
            branch1_input = x[:, t, :20]  # Signal 1
            branch2_input = x[:, t, 20:]  # Signal 2
            
            d1_in = self.branch1_dense(branch1_input)
            beta1 = torch.sigmoid(self.tau_n_branch1)
            self.d1_current = beta1 * self.d1_current + (1 - beta1) * d1_in
            
            d2_in = self.branch2_dense(branch2_input)
            beta2 = torch.sigmoid(self.tau_n_branch2)
            self.d2_current = beta2 * self.d2_current + (1 - beta2) * d2_in
            
            total_input = self.d1_current + self.d2_current
            
            alpha = torch.sigmoid(self.tau_m)
            self.mem = alpha * self.mem + (1 - alpha) * total_input
            
            output = torch.sigmoid(self.output(self.mem))
            outputs.append(output)
        
        return torch.stack(outputs, dim=1)
    
    def get_time_constants(self):
        return {
            'tau_m': torch.sigmoid(self.tau_m).detach().cpu(),
            'tau_n_branch1': torch.sigmoid(self.tau_n_branch1).detach().cpu(),
            'tau_n_branch2': torch.sigmoid(self.tau_n_branch2).detach().cpu()
        }

def train_model_final(model, train_data, train_targets, test_data, test_targets, model_name, epochs=80):
    """最终训练函数"""
    print(f"🏋️ 训练 {model_name}")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=25, gamma=0.5)
    criterion = nn.CrossEntropyLoss()
    
    best_acc = 0.0
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        num_batches = 0
        
        batch_size = 8  # 小批次
        for i in range(0, len(train_data), batch_size):
            batch_data = train_data[i:i+batch_size].to(device)
            batch_targets = train_targets[i:i+batch_size].to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_data)
            
            # 计算损失 - 只在有标签的时间步
            loss_total = 0
            loss_count = 0
            
            for b in range(batch_data.size(0)):
                for t in range(batch_data.size(1)):
                    if batch_targets[b, t] >= 0:  # 有效标签
                        output_t = outputs[b, t, 0]  # [1] -> scalar
                        target_t = batch_targets[b, t].float()
                        
                        # 转换为二分类概率
                        output_prob = torch.stack([1-output_t, output_t])
                        target_class = batch_targets[b, t].long()
                        
                        loss_total += criterion(output_prob.unsqueeze(0), target_class.unsqueeze(0))
                        loss_count += 1
            
            if loss_count > 0:
                loss = loss_total / loss_count
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
                
                correct = 0
                total = 0
                
                for b in range(test_data.size(0)):
                    for t in range(test_data.size(1)):
                        if test_targets[b, t] >= 0:
                            pred = (test_outputs[b, t, 0] > 0.5).float()
                            target = test_targets[b, t].float()
                            
                            if pred == target:
                                correct += 1
                            total += 1
                
                acc = correct / total * 100 if total > 0 else 0
                
                if acc > best_acc:
                    best_acc = acc
                
                avg_loss = total_loss / max(num_batches, 1)
                print(f"  Epoch {epoch+1:3d}: Loss={avg_loss:.4f}, Acc={acc:.1f}%, Best={best_acc:.1f}%")
    
    return best_acc

def main():
    """执行最终实验"""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"🔧 使用设备: {device}")
    
    # 生成挑战性数据
    print("\n📊 生成超级挑战性多时间尺度XOR数据...")
    generator = UltraChallengingXORGenerator(device=device)
    
    train_data, train_targets = generator.generate_dataset(400)
    test_data, test_targets = generator.generate_dataset(100)
    
    print(f"✅ 数据生成完成: 训练{train_data.shape}, 测试{test_data.shape}")
    print(f"   训练数据发放率: {train_data.mean():.4f}")
    print(f"   目标覆盖率: {(train_targets >= 0).float().mean():.4f}")
    
    # 实验配置
    experiments = [
        ("Vanilla SFNN", lambda: FinalVanillaSFNN()),
        ("1-Branch DH-SFNN (Small)", lambda: FinalOneBranchDH_SFNN(tau_init='small')),
        ("1-Branch DH-SFNN (Large)", lambda: FinalOneBranchDH_SFNN(tau_init='large')),
        ("2-Branch DH-SFNN (Beneficial)", lambda: FinalTwoBranchDH_SFNN(beneficial_init=True, learnable=True)),
        ("2-Branch DH-SFNN (Fixed)", lambda: FinalTwoBranchDH_SFNN(beneficial_init=True, learnable=False)),
        ("2-Branch DH-SFNN (Random)", lambda: FinalTwoBranchDH_SFNN(beneficial_init=False, learnable=True)),
    ]
    
    results = {}
    start_time = time.time()
    
    print(f"\n🧪 开始最终实验 (每个配置3次试验)...")
    
    for exp_name, model_creator in experiments:
        print(f"\n📊 实验: {exp_name}")
        print("="*50)
        
        trial_results = []
        
        for trial in range(3):
            print(f"  🔄 试验 {trial+1}/3")
            model = model_creator()
            
            acc = train_model_final(model, train_data, train_targets, test_data, test_targets, 
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
    total_time = time.time() - start_time
    print(f"\n🎉 最终实验完成! 总用时: {total_time/60:.1f}分钟")
    print("="*60)
    print("Figure 4b 多时间尺度XOR实验结果:")
    print("="*60)
    
    for exp_name, result in results.items():
        mean_acc = result['mean']
        std_acc = result['std']
        print(f"{exp_name:30s}: {mean_acc:5.1f}% ± {std_acc:4.1f}%")
    
    # 保存结果
    os.makedirs("results", exist_ok=True)
    torch.save(results, "results/final_experiment_results.pth")
    print(f"\n💾 结果已保存到: results/final_experiment_results.pth")
    
    return results

if __name__ == '__main__':
    try:
        results = main()
        print(f"\n🏁 最终实验成功完成!")
    except Exception as e:
        print(f"\n❌ 实验失败: {e}")
        import traceback
        traceback.print_exc()
