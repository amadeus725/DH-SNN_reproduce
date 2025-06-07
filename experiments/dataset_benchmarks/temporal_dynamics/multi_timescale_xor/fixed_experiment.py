#!/usr/bin/env python3
"""
修复版多时间尺度XOR实验
解决训练问题，确保模型能够正常学习
"""

import torch
import torch.nn as nn
import numpy as np
import time
import os

print("🔧 修复版多时间尺度XOR实验")
print("="*50)

class FixedXORGenerator:
    """修复的XOR数据生成器"""
    
    def __init__(self, device='cpu'):
        self.device = device
        self.time_steps = 60
        self.input_size = 40
        
    def generate_sample(self):
        """生成单个样本"""
        input_data = torch.zeros(self.time_steps, self.input_size)
        target_data = torch.full((self.time_steps, 1), -1.0)  # 使用float类型
        
        # Signal 1: 时间步 10-20 (更明显的差异)
        signal1_type = np.random.choice([0, 1])
        if signal1_type == 1:
            input_data[10:20, :20] = torch.rand(10, 20) * 0.3 + 0.6  # 0.6-0.9
        else:
            input_data[10:20, :20] = torch.rand(10, 20) * 0.2 + 0.1  # 0.1-0.3
        
        # Signal 2序列: 多个Signal 2
        signal2_starts = [30, 40, 50]
        
        for i, start_time in enumerate(signal2_starts):
            if start_time + 8 >= self.time_steps:
                break
                
            signal2_type = np.random.choice([0, 1])
            if signal2_type == 1:
                input_data[start_time:start_time+8, 20:] = torch.rand(8, 20) * 0.3 + 0.6
            else:
                input_data[start_time:start_time+8, 20:] = torch.rand(8, 20) * 0.2 + 0.1
            
            # XOR结果
            xor_result = signal1_type ^ signal2_type
            response_start = start_time + 8
            response_end = min(response_start + 5, self.time_steps)
            target_data[response_start:response_end, 0] = float(xor_result)
        
        return input_data.to(self.device), target_data.to(self.device)
    
    def generate_dataset(self, num_samples):
        inputs, targets = [], []
        for _ in range(num_samples):
            inp, tgt = self.generate_sample()
            inputs.append(inp)
            targets.append(tgt)
        return torch.stack(inputs), torch.stack(targets)

class FixedVanillaSFNN(nn.Module):
    """修复的Vanilla SFNN"""
    
    def __init__(self, input_size=40, hidden_size=64, output_size=1):
        super().__init__()
        self.dense = nn.Linear(input_size, hidden_size)
        self.output = nn.Linear(hidden_size, output_size)
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

class FixedTwoBranchDH_SFNN(nn.Module):
    """修复的双分支DH-SFNN"""
    
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

def fixed_train_model(model, train_data, train_targets, test_data, test_targets, model_name, epochs=60):
    """修复的训练函数"""
    print(f"🏋️ 训练 {model_name}")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    
    # 简化的训练配置
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)  # 适中的学习率
    criterion = nn.BCELoss()  # 直接使用BCE损失
    
    best_acc = 0.0
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        num_batches = 0
        
        batch_size = 16
        for i in range(0, len(train_data), batch_size):
            batch_data = train_data[i:i+batch_size].to(device)
            batch_targets = train_targets[i:i+batch_size].to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_data)
            
            # 简化的损失计算
            valid_mask = (batch_targets >= 0)
            if valid_mask.sum() > 0:
                valid_outputs = outputs[valid_mask]
                valid_targets = batch_targets[valid_mask]
                
                loss = criterion(valid_outputs, valid_targets)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
        
        # 测试
        if epoch % 10 == 0 or epoch == epochs - 1:
            model.eval()
            with torch.no_grad():
                test_outputs = model(test_data.to(device))
                test_targets_device = test_targets.to(device)
                
                valid_mask = (test_targets_device >= 0)
                if valid_mask.sum() > 0:
                    valid_outputs = test_outputs[valid_mask]
                    valid_targets = test_targets_device[valid_mask]
                    
                    pred_binary = (valid_outputs > 0.5).float()
                    acc = (pred_binary == valid_targets).float().mean().item() * 100
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
    
    # 生成修复的数据
    print("\n📊 生成修复版多时间尺度XOR数据...")
    generator = FixedXORGenerator(device=device)
    train_data, train_targets = generator.generate_dataset(400)
    test_data, test_targets = generator.generate_dataset(100)
    
    print(f"✅ 数据生成完成: 训练{train_data.shape}, 测试{test_data.shape}")
    print(f"   训练数据发放率: {train_data.mean():.4f}")
    print(f"   目标覆盖率: {(train_targets >= 0).float().mean():.4f}")
    
    # 分析数据质量
    valid_labels = train_targets[train_targets >= 0]
    if len(valid_labels) > 0:
        class_0 = (valid_labels == 0).sum().item()
        class_1 = (valid_labels == 1).sum().item()
        print(f"   类别分布: 0={class_0}, 1={class_1} (平衡度: {min(class_0, class_1)/max(class_0, class_1):.3f})")
    
    # 实验配置
    experiments = [
        ("Vanilla SFNN", lambda: FixedVanillaSFNN()),
        ("2-Branch DH-SFNN (Beneficial)", lambda: FixedTwoBranchDH_SFNN(beneficial_init=True, learnable=True)),
        ("2-Branch DH-SFNN (Random)", lambda: FixedTwoBranchDH_SFNN(beneficial_init=False, learnable=True)),
    ]
    
    results = {}
    start_time = time.time()
    
    print(f"\n🧪 开始修复实验 (每个配置3次试验)...")
    
    for exp_name, model_creator in experiments:
        print(f"\n📊 实验: {exp_name}")
        print("="*50)
        
        trial_results = []
        
        for trial in range(3):
            print(f"  🔄 试验 {trial+1}/3")
            model = model_creator()
            
            acc = fixed_train_model(model, train_data, train_targets, test_data, test_targets, 
                                  f"{exp_name}_trial_{trial+1}", epochs=50)
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
    print(f"\n🎉 修复实验完成! 总用时: {total_time/60:.1f}分钟")
    print("="*60)
    print("修复版 Figure 4b 多时间尺度XOR实验结果:")
    print("="*60)
    
    for exp_name, result in results.items():
        mean_acc = result['mean']
        std_acc = result['std']
        print(f"{exp_name:30s}: {mean_acc:5.1f}% ± {std_acc:4.1f}%")
    
    # 分析结果
    print(f"\n📊 结果分析:")
    vanilla_acc = results.get("Vanilla SFNN", {}).get('mean', 0)
    beneficial_acc = results.get("2-Branch DH-SFNN (Beneficial)", {}).get('mean', 0)
    random_acc = results.get("2-Branch DH-SFNN (Random)", {}).get('mean', 0)
    
    if beneficial_acc > vanilla_acc + 5:
        print(f"✅ 有益初始化DH-SFNN比Vanilla SFNN好 {beneficial_acc - vanilla_acc:.1f}%")
    else:
        print(f"⚠️ 有益初始化DH-SFNN提升不明显")
    
    if beneficial_acc > random_acc + 3:
        print(f"✅ 有益初始化比随机初始化好 {beneficial_acc - random_acc:.1f}%")
    else:
        print(f"⚠️ 有益初始化优势不明显")
    
    # 保存结果
    os.makedirs("results", exist_ok=True)
    torch.save(results, "results/fixed_experiment_results.pth")
    print(f"\n💾 结果已保存到: results/fixed_experiment_results.pth")
    
    return results

if __name__ == '__main__':
    try:
        results = main()
        print(f"\n🏁 修复实验成功完成!")
    except Exception as e:
        print(f"\n❌ 实验失败: {e}")
        import traceback
        traceback.print_exc()
