#!/usr/bin/env python3
"""
多时间尺度实验主脚本
实现Figure 4b的分支数量对比实验
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

# 添加路径
# sys.path.append removed during restructure
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'utils'))

print("🕰️ 多时间尺度XOR实验 - Figure 4b")
print("="*60)

class MultiTimescaleXORGenerator:
    """多时间尺度XOR数据生成器 - 按照论文Figure 4a精确实现"""

    def __init__(self, device='cpu'):
        self.device = device
        self.total_time = 600
        self.input_size = 40  # 20 for Signal 1, 20 for Signal 2

        # 论文参数设置
        self.signal1_duration = 100  # Signal 1持续时间
        self.signal2_duration = 30   # 每个Signal 2持续时间
        self.signal2_interval = 80   # Signal 2之间的间隔
        self.response_window = 20    # 响应窗口

        # 发放率设置 - 增加差异以提高任务难度
        self.low_rate = 0.05   # 低发放率
        self.high_rate = 0.25  # 高发放率
        self.noise_rate = 0.01 # 背景噪声

    def generate_sample(self):
        """生成单个多时间尺度XOR样本"""
        input_data = torch.zeros(self.total_time, self.input_size)
        target_data = torch.zeros(self.total_time, 1)

        # 添加背景噪声
        noise_mask = torch.rand(self.total_time, self.input_size) < self.noise_rate
        input_data[noise_mask] = 1.0

        # Signal 1: 低频长期信号 (时间步 50-150)
        signal1_start = 50
        signal1_end = signal1_start + self.signal1_duration
        signal1_type = np.random.choice([0, 1])  # 0=低发放率, 1=高发放率

        # 生成Signal 1的脉冲模式
        if signal1_type == 1:
            # 高发放率模式
            signal1_mask = torch.rand(self.signal1_duration, 20) < self.high_rate
        else:
            # 低发放率模式
            signal1_mask = torch.rand(self.signal1_duration, 20) < self.low_rate

        input_data[signal1_start:signal1_end, :20] = signal1_mask.float()

        # Signal 2序列: 高频短期信号
        signal2_starts = [200, 280, 360, 440, 520]  # 5个Signal 2，间隔80时间步

        for i, start_time in enumerate(signal2_starts):
            if start_time + self.signal2_duration >= self.total_time:
                break

            signal2_type = np.random.choice([0, 1])

            # 生成Signal 2的脉冲模式
            if signal2_type == 1:
                signal2_mask = torch.rand(self.signal2_duration, 20) < self.high_rate
            else:
                signal2_mask = torch.rand(self.signal2_duration, 20) < self.low_rate

            input_data[start_time:start_time+self.signal2_duration, 20:] = signal2_mask.float()

            # XOR结果 - 在Signal 2结束后的响应窗口内
            xor_result = signal1_type ^ signal2_type
            response_start = start_time + self.signal2_duration
            response_end = min(response_start + self.response_window, self.total_time)

            target_data[response_start:response_end, 0] = float(xor_result)

        return input_data.to(self.device), target_data.to(self.device)

    def generate_dataset(self, num_samples):
        """生成数据集"""
        inputs, targets = [], []

        for _ in range(num_samples):
            input_data, target_data = self.generate_sample()
            inputs.append(input_data)
            targets.append(target_data)

        return torch.stack(inputs), torch.stack(targets)

class VanillaSFNN(nn.Module):
    """Vanilla SFNN模型"""

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

class OneBranchDH_SFNN(nn.Module):
    """单分支DH-SFNN模型"""

    def __init__(self, input_size=40, hidden_size=64, output_size=1, tau_init='medium'):
        super().__init__()

        self.dense = nn.Linear(input_size, hidden_size)
        self.output = nn.Linear(hidden_size, output_size)

        # 时间常数
        self.tau_m = nn.Parameter(torch.ones(hidden_size) * 2.0)
        self.tau_n = nn.Parameter(torch.ones(hidden_size) * 2.0)

        # 根据初始化类型设置时间常数
        if tau_init == 'small':
            nn.init.uniform_(self.tau_n, -2.0, 0.0)
        elif tau_init == 'large':
            nn.init.uniform_(self.tau_n, 2.0, 4.0)
        else:  # medium
            nn.init.uniform_(self.tau_n, 0.0, 2.0)

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

class TwoBranchDH_SFNN(nn.Module):
    """双分支DH-SFNN模型 - 按照论文Figure 4精确实现"""

    def __init__(self, input_size=40, hidden_size=64, output_size=1, beneficial_init=True, learnable=True):
        super().__init__()

        # 分支连接 - Branch 1处理Signal 1, Branch 2处理Signal 2
        self.branch1_dense = nn.Linear(input_size//2, hidden_size)  # Signal 1 (低频长期)
        self.branch2_dense = nn.Linear(input_size//2, hidden_size)  # Signal 2 (高频短期)
        self.output = nn.Linear(hidden_size, output_size)

        # 膜电位时间常数 (Medium初始化: U(0,4))
        self.tau_m = nn.Parameter(torch.zeros(hidden_size))
        nn.init.uniform_(self.tau_m, 0.0, 4.0)

        # 树突时间常数
        self.tau_n_branch1 = nn.Parameter(torch.zeros(hidden_size))
        self.tau_n_branch2 = nn.Parameter(torch.zeros(hidden_size))

        # 有益初始化 vs 随机初始化
        if beneficial_init:
            # Branch 1: Large初始化 U(2,6) - 用于长期记忆Signal 1
            nn.init.uniform_(self.tau_n_branch1, 2.0, 6.0)
            # Branch 2: Small初始化 U(-4,0) - 用于快速响应Signal 2
            nn.init.uniform_(self.tau_n_branch2, -4.0, 0.0)
        else:
            # 随机初始化 - Medium U(0,4)
            nn.init.uniform_(self.tau_n_branch1, 0.0, 4.0)
            nn.init.uniform_(self.tau_n_branch2, 0.0, 4.0)

        # 设置是否可学习
        self.tau_m.requires_grad = learnable
        self.tau_n_branch1.requires_grad = learnable
        self.tau_n_branch2.requires_grad = learnable

        # 神经元状态
        self.register_buffer('mem', torch.zeros(1, hidden_size))
        self.register_buffer('d1_current', torch.zeros(1, hidden_size))
        self.register_buffer('d2_current', torch.zeros(1, hidden_size))

        # 记录用于分析
        self.branch1_activities = []
        self.branch2_activities = []

    def forward(self, x):
        batch_size, seq_len, _ = x.shape

        # 重置神经元状态
        self.mem = torch.zeros(batch_size, self.mem.size(1)).to(x.device)
        self.d1_current = torch.zeros(batch_size, self.d1_current.size(1)).to(x.device)
        self.d2_current = torch.zeros(batch_size, self.d2_current.size(1)).to(x.device)

        outputs = []
        self.branch1_activities = []
        self.branch2_activities = []

        for t in range(seq_len):
            # 分离输入到两个分支
            branch1_input = x[:, t, :20]  # Signal 1 (低频长期)
            branch2_input = x[:, t, 20:]  # Signal 2 (高频短期)

            # 分支1: 长期记忆分支 (大时间常数)
            d1_in = self.branch1_dense(branch1_input)
            beta1 = torch.sigmoid(self.tau_n_branch1)  # 大时间常数 -> 接近1 -> 慢衰减
            self.d1_current = beta1 * self.d1_current + (1 - beta1) * d1_in

            # 分支2: 快速响应分支 (小时间常数)
            d2_in = self.branch2_dense(branch2_input)
            beta2 = torch.sigmoid(self.tau_n_branch2)  # 小时间常数 -> 接近0 -> 快衰减
            self.d2_current = beta2 * self.d2_current + (1 - beta2) * d2_in

            # 记录分支活动用于分析
            self.branch1_activities.append(self.d1_current.clone().detach())
            self.branch2_activities.append(self.d2_current.clone().detach())

            # 整合两个分支的树突电流
            total_dendritic_current = self.d1_current + self.d2_current

            # 膜电位更新 (Medium时间常数)
            alpha = torch.sigmoid(self.tau_m)
            self.mem = alpha * self.mem + (1 - alpha) * total_dendritic_current

            # 输出 (使用sigmoid而不是脉冲，简化训练)
            output = torch.sigmoid(self.output(self.mem))
            outputs.append(output)

        return torch.stack(outputs, dim=1)

    def get_time_constants(self):
        """获取当前时间常数用于分析"""
        return {
            'tau_m': torch.sigmoid(self.tau_m).detach().cpu(),
            'tau_n_branch1': torch.sigmoid(self.tau_n_branch1).detach().cpu(),
            'tau_n_branch2': torch.sigmoid(self.tau_n_branch2).detach().cpu()
        }

def train_model(model, train_data, train_targets, test_data, test_targets, model_name, epochs=50):
    """训练模型 - 改进的训练过程"""
    print(f"🏋️ 训练 {model_name}")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)

    # 使用更小的学习率和权重衰减
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4, weight_decay=1e-5)
    criterion = nn.BCELoss()  # 使用BCE损失更适合二分类

    best_acc = 0.0
    train_losses = []

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        num_batches = 0

        # 训练 - 使用更小的批次
        batch_size = 8
        for i in range(0, len(train_data), batch_size):
            batch_data = train_data[i:i+batch_size].to(device)
            batch_targets = train_targets[i:i+batch_size].to(device)

            optimizer.zero_grad()
            outputs = model(batch_data)

            # 只在有目标的时间步计算损失 (响应窗口)
            mask = (batch_targets.sum(dim=-1) > 0).unsqueeze(-1)  # [batch, time, 1]
            if mask.sum() > 0:
                masked_outputs = outputs[mask]
                masked_targets = batch_targets[mask]
                loss = criterion(masked_outputs, masked_targets)

                loss.backward()
                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                total_loss += loss.item()
                num_batches += 1

        # 测试
        if epoch % 5 == 0 or epoch == epochs - 1:
            model.eval()
            with torch.no_grad():
                test_outputs = model(test_data.to(device))
                test_targets_device = test_targets.to(device)

                # 只在响应窗口评估准确率
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
                train_losses.append(avg_loss)
                print(f"  Epoch {epoch+1:2d}: Loss={avg_loss:.4f}, Acc={acc:.1f}%, Best={best_acc:.1f}%")

    return best_acc

def main():
    """主实验函数"""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"🔧 使用设备: {device}")

    # 生成数据
    print("\n📊 生成多时间尺度XOR数据...")
    generator = MultiTimescaleXORGenerator(device)
    train_data, train_targets = generator.generate_dataset(200)
    test_data, test_targets = generator.generate_dataset(50)

    print(f"✅ 数据生成完成: 训练{train_data.shape}, 测试{test_data.shape}")

    # 实验配置 - 按照论文Figure 4b
    experiments = [
        ("Vanilla SFNN", lambda: VanillaSFNN()),
        ("1-Branch DH-SFNN (Small)", lambda: OneBranchDH_SFNN(tau_init='small')),
        ("1-Branch DH-SFNN (Large)", lambda: OneBranchDH_SFNN(tau_init='large')),
        ("2-Branch DH-SFNN (Beneficial)", lambda: TwoBranchDH_SFNN(beneficial_init=True, learnable=True)),
        ("2-Branch DH-SFNN (Fixed)", lambda: TwoBranchDH_SFNN(beneficial_init=True, learnable=False)),
        ("2-Branch DH-SFNN (Random)", lambda: TwoBranchDH_SFNN(beneficial_init=False, learnable=True)),
    ]

    results = {}

    print(f"\n🧪 开始实验 (每个配置5次试验)...")

    for exp_name, model_creator in experiments:
        print(f"\n📊 实验: {exp_name}")
        print("="*50)

        trial_results = []
        time_constants_history = []

        for trial in range(5):  # 增加试验次数
            print(f"  🔄 试验 {trial+1}/5")
            model = model_creator()

            # 记录初始时间常数
            if hasattr(model, 'get_time_constants'):
                initial_tau = model.get_time_constants()
                print(f"    初始时间常数: Branch1={initial_tau['tau_n_branch1'].mean():.3f}, Branch2={initial_tau['tau_n_branch2'].mean():.3f}")

            acc = train_model(model, train_data, train_targets, test_data, test_targets,
                            f"{exp_name}_trial_{trial+1}", epochs=80)  # 增加训练轮数
            trial_results.append(acc)

            # 记录最终时间常数
            if hasattr(model, 'get_time_constants'):
                final_tau = model.get_time_constants()
                time_constants_history.append(final_tau)
                print(f"    最终时间常数: Branch1={final_tau['tau_n_branch1'].mean():.3f}, Branch2={final_tau['tau_n_branch2'].mean():.3f}")

        mean_acc = np.mean(trial_results)
        std_acc = np.std(trial_results)
        results[exp_name] = {
            'mean': mean_acc,
            'std': std_acc,
            'trials': trial_results,
            'time_constants': time_constants_history
        }

        print(f"  📈 最终结果: {mean_acc:.1f}% ± {std_acc:.1f}%")
        print(f"  📊 试验结果: {trial_results}")

    # 显示最终结果
    print(f"\n🎉 实验完成! Figure 4b结果:")
    print("="*60)
    for exp_name, result in results.items():
        mean_acc = result['mean']
        std_acc = result['std']
        print(f"{exp_name:30s}: {mean_acc:5.1f}% ± {std_acc:4.1f}%")

    # 保存结果
    os.makedirs("results", exist_ok=True)
    torch.save(results, "results/figure4b_results.pth")
    print(f"\n💾 结果已保存到: results/figure4b_results.pth")

    return results

if __name__ == '__main__':
    try:
        results = main()
        print(f"\n🏁 实验成功完成!")
    except Exception as e:
        print(f"\n❌ 实验失败: {e}")
        import traceback
        traceback.print_exc()
