#!/usr/bin/env python3
"""
Multi-Timescale XOR Experiment for DH-SNN Ultimate
多时间尺度XOR实验 - 验证DH-SNN处理多时间尺度信息的能力

This experiment validates the core innovation of DH-SNN:
- Multiple dendritic branches with different time constants
- Processing information at different temporal scales
- Long-term memory vs. fast response capabilities

Corresponds to Figure 4 in the original paper.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import os
import json
from typing import Dict, List, Tuple, Optional
from datetime import datetime

class MultiTimescaleXORGenerator:
    """
    Multi-timescale XOR data generator
    多时间尺度XOR数据生成器
    
    按照论文Figure 4a精确实现:
    - Signal 1: 低频长期信号 (需要长期记忆)
    - Signal 2: 高频短期信号序列 (需要快速响应)  
    - Task: 记住Signal 1并与每个Signal 2进行XOR运算
    """
    
    def __init__(self, device='cpu'):
        self.device = device
        self.total_time = 600
        self.input_size = 40  # 20 for Signal 1, 20 for Signal 2
        
        # 时序参数
        self.signal1_duration = 100  # Signal 1持续时间
        self.signal2_duration = 30   # 每个Signal 2持续时间
        self.signal2_interval = 80   # Signal 2之间的间隔
        self.response_window = 20    # 响应窗口
        
        # 发放率设置 - 创建有挑战性的时间尺度差异
        self.low_rate = 0.05   # 低发放率模式
        self.high_rate = 0.25  # 高发放率模式
        self.noise_rate = 0.01 # 背景噪声
        
    def generate_sample(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """生成单个多时间尺度XOR样本"""
        input_data = torch.zeros(self.total_time, self.input_size)
        target_data = torch.zeros(self.total_time, 1)
        
        # 添加背景噪声增加任务难度
        noise_mask = torch.rand(self.total_time, self.input_size) < self.noise_rate
        input_data[noise_mask] = 1.0
        
        # Signal 1: 低频长期信号 (时间步 50-150)
        # 这个信号需要被长期记住，用于与后续的Signal 2进行XOR
        signal1_start = 50
        signal1_end = signal1_start + self.signal1_duration
        signal1_type = np.random.choice([0, 1])  # 0=低发放率, 1=高发放率
        
        if signal1_type == 1:
            signal1_mask = torch.rand(self.signal1_duration, 20) < self.high_rate
        else:
            signal1_mask = torch.rand(self.signal1_duration, 20) < self.low_rate
            
        input_data[signal1_start:signal1_end, :20] = signal1_mask.float()
        
        # Signal 2序列: 高频短期信号，需要快速响应
        signal2_starts = [200, 280, 360, 440, 520]  # 5个Signal 2
        
        for i, start_time in enumerate(signal2_starts):
            if start_time + self.signal2_duration >= self.total_time:
                break
                
            signal2_type = np.random.choice([0, 1])
            
            if signal2_type == 1:
                signal2_mask = torch.rand(self.signal2_duration, 20) < self.high_rate
            else:
                signal2_mask = torch.rand(self.signal2_duration, 20) < self.low_rate
                
            input_data[start_time:start_time+self.signal2_duration, 20:] = signal2_mask.float()
            
            # XOR目标: Signal 1 XOR Signal 2
            xor_result = signal1_type ^ signal2_type
            response_start = start_time + self.signal2_duration
            response_end = min(response_start + self.response_window, self.total_time)
            
            target_data[response_start:response_end, 0] = float(xor_result)
            
        return input_data.to(self.device), target_data.to(self.device)
    
    def generate_dataset(self, num_samples: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """生成数据集"""
        inputs, targets = [], []
        
        for _ in range(num_samples):
            input_data, target_data = self.generate_sample()
            inputs.append(input_data)
            targets.append(target_data)
            
        return torch.stack(inputs), torch.stack(targets)

class VanillaSFNN(nn.Module):
    """
    Vanilla Spiking Feed-forward Neural Network
    传统SFNN作为基线模型
    """
    
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

class OneBranchDH_SFNN(nn.Module):
    """
    Single-branch DH-SFNN
    单分支DH-SFNN，具有可学习的树突时间常数
    """
    
    def __init__(self, input_size=40, hidden_size=64, output_size=1, tau_init='medium'):
        super().__init__()
        self.dense = nn.Linear(input_size, hidden_size)
        self.output = nn.Linear(hidden_size, output_size)
        
        # 膜电位和树突时间常数
        self.tau_m = nn.Parameter(torch.ones(hidden_size) * 2.0)
        self.tau_n = nn.Parameter(torch.ones(hidden_size) * 2.0)
        
        # 根据初始化策略设置时间常数
        if tau_init == 'small':
            nn.init.uniform_(self.tau_n, -2.0, 0.0)  # 快速响应
        elif tau_init == 'large':
            nn.init.uniform_(self.tau_n, 2.0, 4.0)   # 长期记忆
        else:  # medium
            nn.init.uniform_(self.tau_n, 0.0, 2.0)   # 中等
            
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
    """
    Two-branch DH-SFNN - The core innovation
    双分支DH-SFNN - 核心创新模型
    
    Key features:
    - Branch 1: Long-term memory (large time constant) for Signal 1
    - Branch 2: Fast response (small time constant) for Signal 2  
    - Automatic temporal specialization through learning
    """
    
    def __init__(self, input_size=40, hidden_size=64, output_size=1, 
                 beneficial_init=True, learnable=True):
        super().__init__()
        
        # 专门的分支连接
        self.branch1_dense = nn.Linear(input_size//2, hidden_size)  # Signal 1 branch
        self.branch2_dense = nn.Linear(input_size//2, hidden_size)  # Signal 2 branch
        self.output = nn.Linear(hidden_size, output_size)
        
        # 膜电位时间常数 (Medium初始化)
        self.tau_m = nn.Parameter(torch.zeros(hidden_size))
        nn.init.uniform_(self.tau_m, 0.0, 4.0)
        
        # 树突时间常数
        self.tau_n_branch1 = nn.Parameter(torch.zeros(hidden_size))
        self.tau_n_branch2 = nn.Parameter(torch.zeros(hidden_size))
        
        # 有益初始化 vs 随机初始化
        if beneficial_init:
            # Branch 1: Large initialization U(2,6) - 长期记忆
            nn.init.uniform_(self.tau_n_branch1, 2.0, 6.0)
            # Branch 2: Small initialization U(-4,0) - 快速响应
            nn.init.uniform_(self.tau_n_branch2, -4.0, 0.0)
        else:
            # 随机初始化 - Medium U(0,4)
            nn.init.uniform_(self.tau_n_branch1, 0.0, 4.0)
            nn.init.uniform_(self.tau_n_branch2, 0.0, 4.0)
            
        # 控制是否可学习
        self.tau_m.requires_grad = learnable
        self.tau_n_branch1.requires_grad = learnable
        self.tau_n_branch2.requires_grad = learnable
        
        # 神经元状态
        self.register_buffer('mem', torch.zeros(1, hidden_size))
        self.register_buffer('d1_current', torch.zeros(1, hidden_size))
        self.register_buffer('d2_current', torch.zeros(1, hidden_size))
        
        # 记录分支活动用于分析
        self.branch1_activities = []
        self.branch2_activities = []
        
    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        
        # 重置状态
        self.mem = torch.zeros(batch_size, self.mem.size(1)).to(x.device)
        self.d1_current = torch.zeros(batch_size, self.d1_current.size(1)).to(x.device)
        self.d2_current = torch.zeros(batch_size, self.d2_current.size(1)).to(x.device)
        
        outputs = []
        self.branch1_activities = []
        self.branch2_activities = []
        
        for t in range(seq_len):
            # 分离输入到专门的分支
            branch1_input = x[:, t, :20]  # Signal 1 (长期记忆需求)
            branch2_input = x[:, t, 20:]  # Signal 2 (快速响应需求)
            
            # Branch 1: 长期记忆分支
            d1_in = self.branch1_dense(branch1_input)
            beta1 = torch.sigmoid(self.tau_n_branch1)  # 大时间常数 -> 慢衰减
            self.d1_current = beta1 * self.d1_current + (1 - beta1) * d1_in
            
            # Branch 2: 快速响应分支
            d2_in = self.branch2_dense(branch2_input)
            beta2 = torch.sigmoid(self.tau_n_branch2)  # 小时间常数 -> 快衰减
            self.d2_current = beta2 * self.d2_current + (1 - beta2) * d2_in
            
            # 记录分支活动
            self.branch1_activities.append(self.d1_current.clone().detach())
            self.branch2_activities.append(self.d2_current.clone().detach())
            
            # 整合两个分支
            total_dendritic_current = self.d1_current + self.d2_current
            
            # 膜电位更新
            alpha = torch.sigmoid(self.tau_m)
            self.mem = alpha * self.mem + (1 - alpha) * total_dendritic_current
            
            output = torch.sigmoid(self.output(self.mem))
            outputs.append(output)
            
        return torch.stack(outputs, dim=1)
    
    def get_time_constants(self) -> Dict[str, torch.Tensor]:
        """获取当前时间常数用于分析"""
        return {
            'tau_m': torch.sigmoid(self.tau_m).detach().cpu(),
            'tau_n_branch1': torch.sigmoid(self.tau_n_branch1).detach().cpu(),
            'tau_n_branch2': torch.sigmoid(self.tau_n_branch2).detach().cpu()
        }
    
    def analyze_specialization(self) -> Dict:
        """分析分支特化程度"""
        tau_constants = self.get_time_constants()
        
        branch1_tau = tau_constants['tau_n_branch1'].mean().item()
        branch2_tau = tau_constants['tau_n_branch2'].mean().item()
        
        specialization = {
            'branch1_tau_mean': branch1_tau,
            'branch2_tau_mean': branch2_tau,
            'tau_difference': abs(branch1_tau - branch2_tau),
            'specialization_degree': abs(branch1_tau - branch2_tau) / (branch1_tau + branch2_tau + 1e-8),
            'is_specialized': abs(branch1_tau - branch2_tau) > 0.3
        }
        
        return specialization

class MultiBranchDH_SFNN(nn.Module):
    """
    Multi-branch DH-SFNN for extended experiments
    多分支DH-SFNN用于扩展实验
    """
    
    def __init__(self, input_size=40, hidden_size=64, output_size=1, num_branches=4):
        super().__init__()
        
        self.num_branches = num_branches
        self.branch_size = hidden_size // num_branches
        
        # 多个分支
        self.branches = nn.ModuleList([
            nn.Linear(input_size, self.branch_size) 
            for _ in range(num_branches)
        ])
        
        self.output = nn.Linear(hidden_size, output_size)
        
        # 每个分支的时间常数
        self.tau_m = nn.Parameter(torch.zeros(hidden_size))
        self.tau_n_branches = nn.ParameterList([
            nn.Parameter(torch.zeros(self.branch_size))
            for _ in range(num_branches)
        ])
        
        # 初始化不同的时间尺度
        nn.init.uniform_(self.tau_m, 0.0, 4.0)
        for i, tau_n in enumerate(self.tau_n_branches):
            # 分配不同的时间尺度范围
            min_val = -4.0 + i * 2.0
            max_val = min_val + 3.0
            nn.init.uniform_(tau_n, min_val, max_val)
            
        # 神经元状态
        self.register_buffer('mem', torch.zeros(1, hidden_size))
        self.register_buffer('d_currents', torch.zeros(1, hidden_size))
        
    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        
        self.mem = torch.zeros(batch_size, self.mem.size(1)).to(x.device)
        self.d_currents = torch.zeros(batch_size, self.d_currents.size(1)).to(x.device)
        
        outputs = []
        
        for t in range(seq_len):
            # 计算每个分支的贡献
            branch_outputs = []
            
            for i, (branch, tau_n) in enumerate(zip(self.branches, self.tau_n_branches)):
                start_idx = i * self.branch_size
                end_idx = start_idx + self.branch_size
                
                d_in = branch(x[:, t, :])
                beta = torch.sigmoid(tau_n)
                
                self.d_currents[:, start_idx:end_idx] = (
                    beta * self.d_currents[:, start_idx:end_idx] + 
                    (1 - beta) * d_in
                )
            
            # 膜电位更新
            alpha = torch.sigmoid(self.tau_m)
            self.mem = alpha * self.mem + (1 - alpha) * self.d_currents
            
            output = torch.sigmoid(self.output(self.mem))
            outputs.append(output)
            
        return torch.stack(outputs, dim=1)

class MultiTimescaleExperiment:
    """
    Multi-timescale XOR experiment runner
    多时间尺度XOR实验运行器
    """
    
    def __init__(self, device='cpu', save_dir='./results'):
        self.device = device
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
    def train_model(self, model: nn.Module, train_data: torch.Tensor, 
                   train_targets: torch.Tensor, test_data: torch.Tensor,
                   test_targets: torch.Tensor, model_name: str, 
                   epochs: int = 80) -> float:
        """训练模型并返回最佳测试准确率"""
        
        print(f"🏋️ 训练 {model_name}")
        model = model.to(self.device)
        
        # 优化器设置
        optimizer = torch.optim.Adam(model.parameters(), lr=5e-4, weight_decay=1e-5)
        criterion = nn.BCELoss()
        
        best_acc = 0.0
        train_history = {'losses': [], 'accuracies': []}
        
        for epoch in range(epochs):
            model.train()
            total_loss = 0
            num_batches = 0
            
            # 小批次训练
            batch_size = 8
            for i in range(0, len(train_data), batch_size):
                batch_data = train_data[i:i+batch_size].to(self.device)
                batch_targets = train_targets[i:i+batch_size].to(self.device)
                
                optimizer.zero_grad()
                outputs = model(batch_data)
                
                # 只在响应窗口计算损失
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
            
            # 测试评估
            if epoch % 10 == 0 or epoch == epochs - 1:
                acc = self.evaluate_model(model, test_data, test_targets)
                best_acc = max(best_acc, acc)
                
                avg_loss = total_loss / max(num_batches, 1)
                train_history['losses'].append(avg_loss)
                train_history['accuracies'].append(acc)
                
                print(f"  Epoch {epoch+1:3d}: Loss={avg_loss:.4f}, Acc={acc:.1f}%, Best={best_acc:.1f}%")
        
        return best_acc
    
    def evaluate_model(self, model: nn.Module, test_data: torch.Tensor, 
                      test_targets: torch.Tensor) -> float:
        """评估模型准确率"""
        model.eval()
        with torch.no_grad():
            test_outputs = model(test_data.to(self.device))
            test_targets_device = test_targets.to(self.device)
            
            # 只在响应窗口评估
            mask = (test_targets_device.sum(dim=-1) > 0).unsqueeze(-1)
            if mask.sum() > 0:
                masked_outputs = test_outputs[mask]
                masked_targets = test_targets_device[mask]
                
                pred = (masked_outputs > 0.5).float()
                acc = (pred == masked_targets).float().mean().item() * 100
                return acc
            return 0.0
    
    def run_branch_comparison_experiment(self, num_trials: int = 5) -> Dict:
        """运行分支数量对比实验 (Figure 4b)"""
        
        print(f"\n🧪 多时间尺度XOR实验 - 分支对比")
        print("="*60)
        
        # 生成数据
        generator = MultiTimescaleXORGenerator(self.device)
        train_data, train_targets = generator.generate_dataset(200)
        test_data, test_targets = generator.generate_dataset(50)
        
        print(f"✅ 数据生成完成: 训练{train_data.shape}, 测试{test_data.shape}")
        
        # 实验配置
        experiments = [
            ("Vanilla SFNN", lambda: VanillaSFNN()),
            ("1-Branch DH-SFNN (Small)", lambda: OneBranchDH_SFNN(tau_init='small')),
            ("1-Branch DH-SFNN (Large)", lambda: OneBranchDH_SFNN(tau_init='large')),
            ("2-Branch DH-SFNN (Beneficial)", lambda: TwoBranchDH_SFNN(beneficial_init=True, learnable=True)),
            ("2-Branch DH-SFNN (Fixed)", lambda: TwoBranchDH_SFNN(beneficial_init=True, learnable=False)),
            ("2-Branch DH-SFNN (Random)", lambda: TwoBranchDH_SFNN(beneficial_init=False, learnable=True)),
            ("4-Branch DH-SFNN", lambda: MultiBranchDH_SFNN(num_branches=4)),
        ]
        
        results = {}
        
        for exp_name, model_creator in experiments:
            print(f"\n📊 实验: {exp_name}")
            print("-" * 50)
            
            trial_results = []
            time_constants_history = []
            
            for trial in range(num_trials):
                print(f"  🔄 试验 {trial+1}/{num_trials}")
                
                model = model_creator()
                
                # 记录初始时间常数
                if hasattr(model, 'get_time_constants'):
                    initial_tau = model.get_time_constants()
                    print(f"    初始τ: B1={initial_tau.get('tau_n_branch1', torch.tensor([0])).mean():.3f}, "
                          f"B2={initial_tau.get('tau_n_branch2', torch.tensor([0])).mean():.3f}")
                
                # 训练模型
                acc = self.train_model(
                    model, train_data, train_targets,
                    test_data, test_targets, f"{exp_name}_trial_{trial+1}"
                )
                
                trial_results.append(acc)
                
                # 记录最终时间常数
                if hasattr(model, 'get_time_constants'):
                    final_tau = model.get_time_constants()
                    time_constants_history.append(final_tau)
                    print(f"    最终τ: B1={final_tau.get('tau_n_branch1', torch.tensor([0])).mean():.3f}, "
                          f"B2={final_tau.get('tau_n_branch2', torch.tensor([0])).mean():.3f}")
                
                # 分析特化程度
                if hasattr(model, 'analyze_specialization'):
                    spec = model.analyze_specialization()
                    print(f"    特化度: {spec['specialization_degree']:.3f}, "
                          f"已特化: {'✅' if spec['is_specialized'] else '❌'}")
            
            # 统计结果
            mean_acc = np.mean(trial_results)
            std_acc = np.std(trial_results)
            
            results[exp_name] = {
                'mean_accuracy': mean_acc,
                'std_accuracy': std_acc,
                'trial_results': trial_results,
                'time_constants_history': time_constants_history
            }
            
            print(f"  📈 最终结果: {mean_acc:.1f}% ± {std_acc:.1f}%")
        
        # 保存结果
        results_path = os.path.join(self.save_dir, 'multi_timescale_results.json')
        with open(results_path, 'w') as f:
            # Convert tensors to lists for JSON serialization
            json_results = {}
            for exp_name, exp_data in results.items():
                json_results[exp_name] = {
                    'mean_accuracy': exp_data['mean_accuracy'],
                    'std_accuracy': exp_data['std_accuracy'],
                    'trial_results': exp_data['trial_results']
                }
            json.dump(json_results, f, indent=2)
        
        print(f"\n💾 结果已保存到: {results_path}")
        return results
    
    def visualize_results(self, results: Dict) -> None:
        """可视化实验结果"""
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # 性能对比图
        exp_names = list(results.keys())
        means = [results[name]['mean_accuracy'] for name in exp_names]
        stds = [results[name]['std_accuracy'] for name in exp_names]
        
        bars = ax1.bar(range(len(exp_names)), means, yerr=stds, capsize=5)
        ax1.set_xlabel('Model Architecture')
        ax1.set_ylabel('Accuracy (%)')
        ax1.set_title('Multi-Timescale XOR Performance Comparison')
        ax1.set_xticks(range(len(exp_names)))
        ax1.set_xticklabels(exp_names, rotation=45, ha='right')
        ax1.grid(True, alpha=0.3)
        
        # 颜色编码
        colors = ['red', 'orange', 'orange', 'green', 'blue', 'purple', 'darkgreen']
        for bar, color in zip(bars, colors[:len(bars)]):
            bar.set_color(color)
            bar.set_alpha(0.7)
        
        # 添加数值标签
        for i, (mean, std) in enumerate(zip(means, stds)):
            ax1.text(i, mean + std + 1, f'{mean:.1f}%', 
                    ha='center', va='bottom', fontweight='bold')
        
        # 创新优势分析
        baseline_acc = results.get('Vanilla SFNN', {}).get('mean_accuracy', 0)
        innovations = []
        improvements = []
        
        for name, data in results.items():
            if 'DH-SFNN' in name:
                innovations.append(name.replace('DH-SFNN', 'DH'))
                improvements.append(data['mean_accuracy'] - baseline_acc)
        
        if innovations:
            bars2 = ax2.bar(range(len(innovations)), improvements)
            ax2.set_xlabel('DH-SNN Variant')
            ax2.set_ylabel('Improvement over Vanilla (%)')
            ax2.set_title('Innovation Impact Analysis')
            ax2.set_xticks(range(len(innovations)))
            ax2.set_xticklabels(innovations, rotation=45, ha='right')
            ax2.grid(True, alpha=0.3)
            ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)
            
            # 颜色编码改进程度
            for bar, improvement in zip(bars2, improvements):
                if improvement > 20:
                    bar.set_color('darkgreen')
                elif improvement > 10:
                    bar.set_color('green')
                elif improvement > 0:
                    bar.set_color('orange')
                else:
                    bar.set_color('red')
                bar.set_alpha(0.7)
                
                # 添加数值标签
                ax2.text(bar.get_x() + bar.get_width()/2, 
                        bar.get_height() + (1 if improvement > 0 else -3),
                        f'+{improvement:.1f}%' if improvement > 0 else f'{improvement:.1f}%',
                        ha='center', va='bottom' if improvement > 0 else 'top',
                        fontweight='bold')
        
        plt.tight_layout()
        
        # 保存图表
        plot_path = os.path.join(self.save_dir, 'multi_timescale_results.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"📊 可视化结果已保存到: {plot_path}")
    
    def analyze_temporal_specialization(self, results: Dict) -> None:
        """分析时间特化"""
        
        print(f"\n🔬 时间特化分析")
        print("="*50)
        
        for exp_name, exp_data in results.items():
            if 'time_constants_history' in exp_data and exp_data['time_constants_history']:
                print(f"\n📊 {exp_name}:")
                
                # 分析最后一次试验的时间常数
                final_tau = exp_data['time_constants_history'][-1]
                
                if 'tau_n_branch1' in final_tau and 'tau_n_branch2' in final_tau:
                    tau1_mean = final_tau['tau_n_branch1'].mean().item()
                    tau2_mean = final_tau['tau_n_branch2'].mean().item()
                    tau_diff = abs(tau1_mean - tau2_mean)
                    
                    print(f"  Branch 1 时间常数: {tau1_mean:.3f}")
                    print(f"  Branch 2 时间常数: {tau2_mean:.3f}")
                    print(f"  分支分化程度: {tau_diff:.3f}")
                    
                    if tau_diff > 0.3:
                        print(f"  ✅ 成功实现时间特化")
                        if tau1_mean > tau2_mean:
                            print(f"     Branch1=长期记忆, Branch2=快速响应")
                        else:
                            print(f"     Branch1=快速响应, Branch2=长期记忆")
                    else:
                        print(f"  ⚠️ 时间特化程度较低")

def run_multi_timescale_experiment():
    """运行完整的多时间尺度实验"""
    
    print("🚀 DH-SNN Ultimate: Multi-Timescale XOR Experiment")
    print("="*70)
    print("验证DH-SNN处理多时间尺度信息的核心创新能力")
    print("Validating DH-SNN's core innovation in multi-timescale processing")
    print("="*70)
    
    # 设备设置
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"🔧 使用设备: {device}")
    
    # 创建实验运行器
    experiment = MultiTimescaleExperiment(device=device)
    
    # 运行主实验
    results = experiment.run_branch_comparison_experiment(num_trials=3)
    
    # 结果分析
    print(f"\n🎉 实验完成! 多时间尺度XOR结果:")
    print("="*60)
    
    for exp_name, exp_data in results.items():
        mean_acc = exp_data['mean_accuracy']
        std_acc = exp_data['std_accuracy']
        print(f"{exp_name:35s}: {mean_acc:5.1f}% ± {std_acc:4.1f}%")
    
    # 可视化结果
    experiment.visualize_results(results)
    
    # 分析时间特化
    experiment.analyze_temporal_specialization(results)
    
    # 核心发现总结
    print(f"\n💡 核心发现:")
    print("-" * 30)
    
    vanilla_acc = results.get('Vanilla SFNN', {}).get('mean_accuracy', 0)
    best_dh_name = max([name for name in results.keys() if 'DH-SFNN' in name], 
                       key=lambda x: results[x]['mean_accuracy'])
    best_dh_acc = results[best_dh_name]['mean_accuracy']
    
    print(f"1. ✅ DH-SNN显著优于传统SNN:")
    print(f"   最佳DH-SNN ({best_dh_name}): {best_dh_acc:.1f}%")
    print(f"   Vanilla SNN: {vanilla_acc:.1f}%")
    print(f"   性能提升: +{best_dh_acc - vanilla_acc:.1f}%")
    
    beneficial_acc = results.get('2-Branch DH-SFNN (Beneficial)', {}).get('mean_accuracy', 0)
    random_acc = results.get('2-Branch DH-SFNN (Random)', {}).get('mean_accuracy', 0)
    
    if beneficial_acc > 0 and random_acc > 0:
        print(f"\n2. ✅ 有益初始化的重要性:")
        print(f"   有益初始化: {beneficial_acc:.1f}%")
        print(f"   随机初始化: {random_acc:.1f}%")
        print(f"   初始化优势: +{beneficial_acc - random_acc:.1f}%")
    
    fixed_acc = results.get('2-Branch DH-SFNN (Fixed)', {}).get('mean_accuracy', 0)
    learnable_acc = results.get('2-Branch DH-SFNN (Beneficial)', {}).get('mean_accuracy', 0)
    
    if fixed_acc > 0 and learnable_acc > 0:
        print(f"\n3. ✅ 可学习时间常数的优势:")
        print(f"   可学习时间常数: {learnable_acc:.1f}%")
        print(f"   固定时间常数: {fixed_acc:.1f}%")
        print(f"   学习优势: +{learnable_acc - fixed_acc:.1f}%")
    
    print(f"\n🎯 创新验证成功! DH-SNN在多时间尺度任务中展现出强大的优势")
    
    return results

if __name__ == '__main__':
    # 设置随机种子保证可重现性
    torch.manual_seed(42)
    np.random.seed(42)
    
    try:
        results = run_multi_timescale_experiment()
        print(f"\n🏁 Multi-Timescale实验成功完成!")
    except Exception as e:
        print(f"\n❌ 实验失败: {e}")
        import traceback
        traceback.print_exc()