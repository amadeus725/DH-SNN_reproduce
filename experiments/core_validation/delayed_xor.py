#!/usr/bin/env python3
"""
多时间尺度XOR实验
实现Figure 4b的分支数量对比实验
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import sys
from typing import Dict, List, Tuple, Optional
import time

# 导入组件
# sys.path.append removed during restructure

from .data_generator import MultiTimescaleXORGenerator
from .models import TwoBranchDH_SFNN, MultiBranchDH_SFNN
from dh_snn.core.models import VanillaSFNN

class MultiTimescaleXORExperiment:
    """
    多时间尺度XOR实验类
    实现Figure 4b中的各种配置对比
    """
    
    def __init__(self, 
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
                 save_dir: str = 'outputs/multi_timescale_xor'):
        """
        初始化实验
        
        Args:
            device: 计算设备
            save_dir: 保存目录
        """
        self.device = device
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
        # 实验配置
        self.config = {
            'input_size': 100,
            'hidden_size': 64,
            'output_size': 1,
            'learning_rate': 1e-3,
            'batch_size': 32,
            'epochs': 200,
            'v_threshold': 1.0,
            'num_trials': 10,  # 重复实验次数
        }
        
        # 数据生成器
        self.data_generator = MultiTimescaleXORGenerator(
            dt=1.0,
            total_time=1000,
            signal1_duration=100,
            signal2_duration=50,
            signal2_interval=100,
            num_signal2=5,
            input_size=self.config['input_size'],
            device=device
        )
        
        print(f"🚀 多时间尺度XOR实验初始化完成")
        print(f"📱 使用设备: {device}")
        print(f"💾 保存目录: {save_dir}")
    
    def create_model(self, 
                    model_type: str,
                    num_branches: int = 2,
                    timing_config: str = 'beneficial',
                    learnable_timing: bool = True) -> nn.Module:
        """
        创建模型
        
        Args:
            model_type: 模型类型 ('vanilla', 'two_branch', 'multi_branch')
            num_branches: 分支数量
            timing_config: 时间常数配置 ('beneficial', 'small', 'large', 'medium')
            learnable_timing: 时间常数是否可学习
            
        Returns:
            模型实例
        """
        # 时间常数配置
        timing_ranges = {
            'small': (-4.0, 0.0),
            'medium': (0.0, 4.0),
            'large': (2.0, 6.0),
            'beneficial_branch1': (2.0, 6.0),  # 大时间常数
            'beneficial_branch2': (-4.0, 0.0)  # 小时间常数
        }
        
        if model_type == 'vanilla':
            # Vanilla SFNN
            model = VanillaSFNN(
                input_size=self.config['input_size'],
                hidden_size=self.config['hidden_size'],
                output_size=self.config['output_size'],
                tau_m_range=timing_ranges['medium'],
                v_threshold=self.config['v_threshold'],
                device=self.device
            )
            
        elif model_type == 'two_branch':
            # 双分支DH-SFNN
            if timing_config == 'beneficial':
                tau_n_branch1_range = timing_ranges['beneficial_branch1']
                tau_n_branch2_range = timing_ranges['beneficial_branch2']
                beneficial_init = True
            else:
                tau_n_branch1_range = timing_ranges[timing_config]
                tau_n_branch2_range = timing_ranges[timing_config]
                beneficial_init = False
            
            model = TwoBranchDH_SFNN(
                input_size=self.config['input_size'],
                hidden_size=self.config['hidden_size'],
                output_size=self.config['output_size'],
                tau_m_range=timing_ranges['medium'],
                tau_n_branch1_range=tau_n_branch1_range,
                tau_n_branch2_range=tau_n_branch2_range,
                v_threshold=self.config['v_threshold'],
                beneficial_init=beneficial_init,
                learnable_timing=learnable_timing,
                device=self.device
            )
            
        elif model_type == 'multi_branch':
            # 多分支DH-SFNN
            model = MultiBranchDH_SFNN(
                input_size=self.config['input_size'],
                hidden_size=self.config['hidden_size'],
                output_size=self.config['output_size'],
                num_branches=num_branches,
                tau_m_range=timing_ranges['medium'],
                tau_n_range=timing_ranges[timing_config],
                v_threshold=self.config['v_threshold'],
                learnable_timing=learnable_timing,
                device=self.device
            )
            
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        return model.to(self.device)
    
    def train_model(self, 
                   model: nn.Module,
                   train_data: torch.Tensor,
                   train_targets: torch.Tensor,
                   test_data: torch.Tensor,
                   test_targets: torch.Tensor,
                   model_name: str) -> Dict:
        """
        训练模型
        
        Args:
            model: 模型
            train_data: 训练数据
            train_targets: 训练目标
            test_data: 测试数据
            test_targets: 测试目标
            model_name: 模型名称
            
        Returns:
            训练结果字典
        """
        print(f"🏋️ 训练 {model_name}")
        
        # 优化器和损失函数
        optimizer = torch.optim.Adam(model.parameters(), lr=self.config['learning_rate'])
        criterion = nn.MSELoss()
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)
        
        # 训练历史
        train_losses = []
        train_accs = []
        test_accs = []
        best_test_acc = 0.0
        
        num_batches = len(train_data) // self.config['batch_size']
        
        for epoch in range(self.config['epochs']):
            model.train()
            epoch_loss = 0.0
            epoch_acc = 0.0
            
            # 随机打乱数据
            indices = torch.randperm(len(train_data))
            
            for batch_idx in range(num_batches):
                start_idx = batch_idx * self.config['batch_size']
                end_idx = start_idx + self.config['batch_size']
                batch_indices = indices[start_idx:end_idx]
                
                batch_data = train_data[batch_indices].to(self.device)
                batch_targets = train_targets[batch_indices].to(self.device)
                
                optimizer.zero_grad()
                
                # 前向传播
                if isinstance(model, TwoBranchDH_SFNN):
                    # 双分支模型需要分离的输入
                    _, _, branch1_data, branch2_data = self.data_generator.generate_dataset(
                        self.config['batch_size'], split_by_branch=True
                    )
                    batch_branch1 = branch1_data[batch_indices].to(self.device)
                    batch_branch2 = branch2_data[batch_indices].to(self.device)
                    outputs, _ = model(batch_data, batch_branch1, batch_branch2)
                else:
                    outputs, _ = model(batch_data)
                
                # 计算损失（只在输出时间点）
                loss = criterion(outputs, batch_targets)
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                
                # 计算准确率（基于XOR逻辑）
                with torch.no_grad():
                    pred_binary = (outputs > 0.5).float()
                    target_binary = (batch_targets > 0.5).float()
                    acc = (pred_binary == target_binary).float().mean()
                    epoch_acc += acc.item()
            
            scheduler.step()
            
            # 平均损失和准确率
            avg_loss = epoch_loss / num_batches
            avg_acc = epoch_acc / num_batches * 100
            
            # 测试
            test_acc = self.evaluate_model(model, test_data, test_targets)
            
            train_losses.append(avg_loss)
            train_accs.append(avg_acc)
            test_accs.append(test_acc)
            
            if test_acc > best_test_acc:
                best_test_acc = test_acc
            
            if epoch % 20 == 0 or epoch == self.config['epochs'] - 1:
                print(f"  Epoch {epoch+1:3d}: Loss={avg_loss:.4f}, Train={avg_acc:.1f}%, Test={test_acc:.1f}%, Best={best_test_acc:.1f}%")
        
        return {
            'train_losses': train_losses,
            'train_accs': train_accs,
            'test_accs': test_accs,
            'best_test_acc': best_test_acc,
            'final_test_acc': test_accs[-1]
        }
    
    def evaluate_model(self, 
                      model: nn.Module,
                      test_data: torch.Tensor,
                      test_targets: torch.Tensor) -> float:
        """
        评估模型
        
        Args:
            model: 模型
            test_data: 测试数据
            test_targets: 测试目标
            
        Returns:
            测试准确率
        """
        model.eval()
        total_acc = 0.0
        num_batches = len(test_data) // self.config['batch_size']
        
        with torch.no_grad():
            for batch_idx in range(num_batches):
                start_idx = batch_idx * self.config['batch_size']
                end_idx = start_idx + self.config['batch_size']
                
                batch_data = test_data[start_idx:end_idx].to(self.device)
                batch_targets = test_targets[start_idx:end_idx].to(self.device)
                
                # 前向传播
                if isinstance(model, TwoBranchDH_SFNN):
                    # 双分支模型需要分离的输入
                    _, _, branch1_data, branch2_data = self.data_generator.generate_dataset(
                        self.config['batch_size'], split_by_branch=True
                    )
                    outputs, _ = model(batch_data, branch1_data, branch2_data)
                else:
                    outputs, _ = model(batch_data)
                
                # 计算准确率
                pred_binary = (outputs > 0.5).float()
                target_binary = (batch_targets > 0.5).float()
                acc = (pred_binary == target_binary).float().mean()
                total_acc += acc.item()
        
        return total_acc / num_batches * 100
    
    def run_branch_comparison_experiment(self) -> Dict:
        """
        运行分支数量对比实验 (Figure 4b)
        
        Returns:
            实验结果字典
        """
        print("\n🧪 运行分支数量对比实验 (Figure 4b)")
        
        # 生成数据集
        print("📊 生成多时间尺度XOR数据集...")
        train_input, train_target, train_branch1, train_branch2 = self.data_generator.generate_dataset(
            num_samples=1000, split_by_branch=True
        )
        test_input, test_target, test_branch1, test_branch2 = self.data_generator.generate_dataset(
            num_samples=200, split_by_branch=True
        )
        
        # 实验配置
        experiments = [
            ('Vanilla SFNN', 'vanilla', 0, 'medium', True),
            ('1-Branch DH-SFNN (Small)', 'multi_branch', 1, 'small', True),
            ('1-Branch DH-SFNN (Large)', 'multi_branch', 1, 'large', True),
            ('2-Branch DH-SFNN (Beneficial)', 'two_branch', 2, 'beneficial', True),
            ('2-Branch DH-SFNN (Fixed)', 'two_branch', 2, 'beneficial', False),
            ('4-Branch DH-SFNN', 'multi_branch', 4, 'large', True),
        ]
        
        results = {}
        
        for exp_name, model_type, num_branches, timing_config, learnable in experiments:
            print(f"\n🔬 实验: {exp_name}")
            
            trial_results = []
            
            for trial in range(self.config['num_trials']):
                print(f"  试验 {trial+1}/{self.config['num_trials']}")
                
                # 创建模型
                model = self.create_model(
                    model_type=model_type,
                    num_branches=num_branches,
                    timing_config=timing_config,
                    learnable_timing=learnable
                )
                
                # 训练模型
                result = self.train_model(
                    model, train_input, train_target,
                    test_input, test_target, f"{exp_name}_trial_{trial+1}"
                )
                
                trial_results.append(result['best_test_acc'])
            
            # 统计结果
            mean_acc = np.mean(trial_results)
            std_acc = np.std(trial_results)
            
            results[exp_name] = {
                'mean_accuracy': mean_acc,
                'std_accuracy': std_acc,
                'all_trials': trial_results,
                'model_type': model_type,
                'num_branches': num_branches,
                'timing_config': timing_config,
                'learnable_timing': learnable
            }
            
            print(f"  📈 结果: {mean_acc:.1f}% ± {std_acc:.1f}%")
        
        # 保存结果
        torch.save(results, os.path.join(self.save_dir, 'branch_comparison_results.pth'))
        
        return results

# 测试代码
if __name__ == '__main__':
    # 创建实验
    experiment = MultiTimescaleXORExperiment()
    
    # 运行分支对比实验
    results = experiment.run_branch_comparison_experiment()
    
    # 打印总结
    print("\n🎉 实验完成！结果总结:")
    for exp_name, result in results.items():
        mean_acc = result['mean_accuracy']
        std_acc = result['std_accuracy']
        print(f"  {exp_name}: {mean_acc:.1f}% ± {std_acc:.1f}%")
