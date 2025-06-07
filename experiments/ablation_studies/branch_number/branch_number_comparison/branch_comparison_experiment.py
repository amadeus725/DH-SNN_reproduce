#!/usr/bin/env python3
"""
分支数量对比实验
基于原论文设计，系统性地比较不同分支数量对DH-SNN性能的影响
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.subplots as sp
from plotly.subplots import make_subplots
import os
import json
import time
from datetime import datetime
import argparse

# 添加项目根目录到路径
import sys
sys.path.append('/root/DH-SNN_reproduce')

# 导入SpikingJelly组件
from spikingjelly.activation_based import neuron, layer, functional, surrogate

class SimpleDH_SFNN(nn.Module):
    """简化的DH-SFNN实现，用于分支数量对比实验"""

    def __init__(self, input_dim=2, hidden_dims=[16], output_dim=2, num_branches=2,
                 v_threshold=1.0, tau_m_init=(0.0, 4.0), tau_n_init=(0.0, 4.0)):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        self.num_branches = num_branches

        # 构建网络层
        self.layers = nn.ModuleList()

        # 输入层到第一个隐藏层
        prev_dim = input_dim
        for i, hidden_dim in enumerate(hidden_dims):
            if num_branches > 1:
                # 多分支DH-SNN层
                self.layers.append(DH_Layer(prev_dim, hidden_dim, num_branches, v_threshold))
            else:
                # 单分支传统SNN层
                self.layers.append(layer.Linear(prev_dim, hidden_dim))
                self.layers.append(neuron.LIFNode(v_threshold=v_threshold, surrogate_function=surrogate.ATan()))
            prev_dim = hidden_dim

        # 输出层
        self.output_layer = layer.Linear(prev_dim, output_dim)

    def forward(self, x):
        # x: [batch, time, features]
        batch_size, time_steps, _ = x.shape

        # 重置网络状态
        functional.reset_net(self)

        outputs = []
        for t in range(time_steps):
            h = x[:, t]

            # 通过隐藏层
            for layer_module in self.layers:
                h = layer_module(h)

            # 输出层
            output = self.output_layer(h)
            outputs.append(output)

        # 返回最后时间步的输出
        return outputs[-1]

class DH_Layer(nn.Module):
    """简化的DH层实现"""

    def __init__(self, input_dim, output_dim, num_branches, v_threshold=1.0):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_branches = num_branches

        # 每个分支的线性层
        self.branch_layers = nn.ModuleList([
            layer.Linear(input_dim, output_dim) for _ in range(num_branches)
        ])

        # 每个分支的时间常数
        self.branch_taus = nn.ParameterList([
            nn.Parameter(torch.tensor(0.5 + 0.1 * i)) for i in range(num_branches)
        ])

        # LIF神经元
        self.lif = neuron.LIFNode(v_threshold=v_threshold, surrogate_function=surrogate.ATan())

        # 分支状态
        self.register_buffer('branch_states', torch.zeros(1, output_dim, num_branches))

    def forward(self, x):
        batch_size = x.size(0)

        # 确保状态张量大小正确
        if self.branch_states.size(0) != batch_size:
            self.branch_states = torch.zeros(batch_size, self.output_dim, self.num_branches,
                                           device=x.device, dtype=x.dtype)

        # 计算每个分支的输出
        branch_outputs = []
        for i, (branch_layer, tau) in enumerate(zip(self.branch_layers, self.branch_taus)):
            branch_input = branch_layer(x)

            # 更新分支状态
            alpha = torch.sigmoid(tau)
            self.branch_states[:, :, i] = (alpha * self.branch_states[:, :, i] +
                                         (1 - alpha) * branch_input)
            branch_outputs.append(self.branch_states[:, :, i])

        # 合并分支输出
        combined_input = torch.stack(branch_outputs, dim=-1).sum(dim=-1)

        # 通过LIF神经元
        return self.lif(combined_input)

class BranchComparisonExperiment:
    """分支数量对比实验类"""
    
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.results = {}
        
        # 创建结果保存目录
        self.save_dir = config.get('save_dir', 'experiments/branch_number_comparison/results')
        os.makedirs(self.save_dir, exist_ok=True)
        
        print(f"🎯 分支数量对比实验初始化")
        print(f"📱 设备: {self.device}")
        print(f"💾 结果保存目录: {self.save_dir}")

    def create_multi_timescale_xor_dataloader(self):
        """创建多时间尺度XOR任务的数据加载器"""

        def generate_multi_timescale_xor_data(num_samples=1000, time_steps=100):
            """生成多时间尺度XOR数据"""

            data = []
            labels = []

            for _ in range(num_samples):
                # 创建输入序列
                sequence = torch.zeros(time_steps, 2)  # 2个输入通道

                # Signal 1: 低频信号 (时间步10-20)
                signal1_start = np.random.randint(5, 15)
                signal1_end = signal1_start + 10
                signal1_value = np.random.choice([0, 1])

                if signal1_value == 1:
                    # 低发放率 (0.2)
                    for t in range(signal1_start, min(signal1_end, time_steps)):
                        if np.random.random() < 0.2:
                            sequence[t, 0] = 1.0

                # Signal 2: 高频信号 (时间步70-90)
                signal2_start = np.random.randint(65, 75)
                signal2_end = signal2_start + 20
                signal2_value = np.random.choice([0, 1])

                if signal2_value == 1:
                    # 高发放率 (0.6)
                    for t in range(signal2_start, min(signal2_end, time_steps)):
                        if np.random.random() < 0.6:
                            sequence[t, 1] = 1.0

                # XOR标签
                label = signal1_value ^ signal2_value

                data.append(sequence)
                labels.append(label)

            return torch.stack(data), torch.tensor(labels, dtype=torch.long)

        # 生成训练和测试数据
        train_data, train_labels = generate_multi_timescale_xor_data(
            num_samples=self.config.get('train_samples', 5000)
        )
        test_data, test_labels = generate_multi_timescale_xor_data(
            num_samples=self.config.get('test_samples', 1000)
        )

        # 创建数据集
        train_dataset = torch.utils.data.TensorDataset(train_data, train_labels)
        test_dataset = torch.utils.data.TensorDataset(test_data, test_labels)

        # 创建数据加载器
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self.config['batch_size'],
            shuffle=True,
            num_workers=0  # 避免多进程问题
        )

        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=self.config['batch_size'],
            shuffle=False,
            num_workers=0
        )

        return train_loader, test_loader

    def train_model_simple(self, model, train_loader, test_loader, optimizer, scheduler, num_epochs):
        """简化的模型训练函数"""

        criterion = nn.CrossEntropyLoss()
        best_accuracy = 0.0
        train_history = {'train_loss': [], 'train_acc': [], 'test_acc': []}

        for epoch in range(num_epochs):
            # 训练阶段
            model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0

            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(self.device), target.to(self.device)

                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                _, predicted = torch.max(output.data, 1)
                train_total += target.size(0)
                train_correct += (predicted == target).sum().item()

            # 测试阶段
            test_accuracy = self.evaluate_model_simple(model, test_loader)

            # 记录历史
            train_accuracy = 100.0 * train_correct / train_total
            train_history['train_loss'].append(train_loss / len(train_loader))
            train_history['train_acc'].append(train_accuracy)
            train_history['test_acc'].append(test_accuracy)

            # 更新最佳准确率
            if test_accuracy > best_accuracy:
                best_accuracy = test_accuracy

            # 学习率调度
            scheduler.step()

            if epoch % 10 == 0:
                print(f"Epoch {epoch}: Train Acc: {train_accuracy:.2f}%, Test Acc: {test_accuracy:.2f}%")

        return train_history, best_accuracy

    def evaluate_model_simple(self, model, test_loader):
        """简化的模型评估函数"""

        model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = model(data)
                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()

        accuracy = 100.0 * correct / total
        return accuracy
    
    def create_model(self, num_branches):
        """创建指定分支数量的DH-SNN模型"""

        model = SimpleDH_SFNN(
            input_dim=self.config['input_dim'],
            hidden_dims=self.config['hidden_dims'],
            output_dim=self.config['output_dim'],
            num_branches=num_branches,
            v_threshold=self.config['v_threshold'],
            tau_m_init=self.config['tau_m_init'],
            tau_n_init=self.config['tau_n_init']
        )

        return model.to(self.device)
    
    def run_single_experiment(self, num_branches, trial=0):
        """运行单个分支数量的实验"""
        
        print(f"\n🔬 开始实验: {num_branches}分支 (试验 {trial+1})")
        
        # 创建模型
        model = self.create_model(num_branches)
        
        # 计算参数量
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"📊 模型参数: 总计 {total_params:,}, 可训练 {trainable_params:,}")
        
        # 创建数据加载器 - 使用简化的多时间尺度XOR任务
        train_loader, test_loader = self.create_multi_timescale_xor_dataloader()
        
        # 创建优化器
        optimizer = optim.Adam(
            model.parameters(),
            lr=self.config['learning_rate'],
            weight_decay=self.config.get('weight_decay', 1e-4)
        )
        
        # 创建学习率调度器
        scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=self.config.get('lr_step_size', 50),
            gamma=self.config.get('lr_gamma', 0.5)
        )
        
        # 训练模型
        start_time = time.time()

        train_history, best_accuracy = self.train_model_simple(
            model=model,
            train_loader=train_loader,
            test_loader=test_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            num_epochs=self.config['num_epochs']
        )

        training_time = time.time() - start_time

        # 最终评估
        test_accuracy = self.evaluate_model_simple(model, test_loader)

        # 简化的效率指标
        efficiency_metrics = {
            'inference_time': training_time / self.config['num_epochs'],
            'memory_usage': total_params * 4 / (1024**2),  # MB (假设float32)
            'flops_estimate': total_params * 2  # 简化估计
        }
        
        results = {
            'num_branches': num_branches,
            'trial': trial,
            'test_accuracy': test_accuracy,
            'training_time': training_time,
            'total_params': total_params,
            'trainable_params': trainable_params,
            'train_history': train_history,
            'efficiency_metrics': efficiency_metrics,
            'timestamp': datetime.now().isoformat()
        }
        
        print(f"✅ 完成: {num_branches}分支, 准确率: {test_accuracy:.2f}%, 训练时间: {training_time:.1f}s")
        
        return results
    
    def run_all_experiments(self):
        """运行所有分支数量的对比实验"""
        
        branch_numbers = self.config['branch_numbers']
        num_trials = self.config.get('num_trials', 3)
        
        print(f"🚀 开始分支数量对比实验")
        print(f"📋 分支数量: {branch_numbers}")
        print(f"🔄 每个配置试验次数: {num_trials}")
        
        all_results = []
        
        for num_branches in branch_numbers:
            branch_results = []
            
            for trial in range(num_trials):
                try:
                    result = self.run_single_experiment(num_branches, trial)
                    branch_results.append(result)
                    
                    # 保存单次实验结果
                    result_file = os.path.join(
                        self.save_dir, 
                        f'{num_branches}branch_trial{trial}_results.json'
                    )
                    with open(result_file, 'w') as f:
                        json.dump(result, f, indent=2, default=str)
                    
                except Exception as e:
                    print(f"❌ 实验失败: {num_branches}分支 试验{trial}, 错误: {e}")
                    continue
            
            # 计算该分支数量的统计结果
            if branch_results:
                accuracies = [r['test_accuracy'] for r in branch_results]
                training_times = [r['training_time'] for r in branch_results]
                
                summary = {
                    'num_branches': num_branches,
                    'mean_accuracy': np.mean(accuracies),
                    'std_accuracy': np.std(accuracies),
                    'mean_training_time': np.mean(training_times),
                    'std_training_time': np.std(training_times),
                    'total_params': branch_results[0]['total_params'],
                    'trials': branch_results
                }
                
                all_results.append(summary)
                
                print(f"📊 {num_branches}分支汇总: {summary['mean_accuracy']:.2f}% ± {summary['std_accuracy']:.2f}%")
        
        self.results = all_results
        
        # 保存完整结果
        final_results_file = os.path.join(self.save_dir, 'branch_comparison_final_results.json')
        with open(final_results_file, 'w') as f:
            json.dump(all_results, f, indent=2, default=str)
        
        print(f"💾 完整结果已保存: {final_results_file}")
        
        return all_results
    
    def create_visualization(self):
        """创建可视化图表"""
        
        if not self.results:
            print("❌ 没有结果数据，无法创建可视化")
            return
        
        # 创建综合分析图表
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                'a) Performance vs Branch Number',
                'b) Parameter Efficiency Analysis',
                'c) Training Time Comparison',
                'd) Computational Overhead Analysis'
            ],
            specs=[[{"secondary_y": False}, {"secondary_y": True}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # 提取数据
        branch_nums = [r['num_branches'] for r in self.results]
        mean_accs = [r['mean_accuracy'] for r in self.results]
        std_accs = [r['std_accuracy'] for r in self.results]
        total_params = [r['total_params'] for r in self.results]
        mean_times = [r['mean_training_time'] for r in self.results]
        
        # Panel A: 性能对比
        fig.add_trace(go.Scatter(
            x=branch_nums,
            y=mean_accs,
            error_y=dict(type='data', array=std_accs, visible=True),
            mode='lines+markers',
            name='Test Accuracy',
            line=dict(color='#2E86AB', width=3),
            marker=dict(size=10)
        ), row=1, col=1)
        
        # Panel B: 参数效率分析
        baseline_params = total_params[0] if total_params else 1
        param_ratios = [p / baseline_params for p in total_params]
        baseline_acc = mean_accs[0] if mean_accs else 0
        acc_improvements = [acc - baseline_acc for acc in mean_accs]
        
        fig.add_trace(go.Scatter(
            x=branch_nums,
            y=param_ratios,
            mode='lines+markers',
            name='Parameter Ratio',
            line=dict(color='#FF6B6B', width=3),
            marker=dict(size=8)
        ), row=1, col=2)
        
        fig.add_trace(go.Scatter(
            x=branch_nums,
            y=acc_improvements,
            mode='lines+markers',
            name='Accuracy Improvement',
            line=dict(color='#4BC0C0', width=3),
            marker=dict(size=8),
            yaxis='y2'
        ), row=1, col=2, secondary_y=True)
        
        # Panel C: 训练时间对比
        baseline_time = mean_times[0] if mean_times else 1
        time_ratios = [t / baseline_time for t in mean_times]
        
        fig.add_trace(go.Bar(
            x=branch_nums,
            y=time_ratios,
            name='Training Time Ratio',
            marker_color='#9966FF',
            text=[f'{ratio:.1f}x' for ratio in time_ratios],
            textposition='outside'
        ), row=2, col=1)
        
        # Panel D: 计算开销分析
        # 计算理论计算复杂度比率
        complexity_ratios = [num_branches for num_branches in branch_nums]
        
        fig.add_trace(go.Scatter(
            x=branch_nums,
            y=complexity_ratios,
            mode='lines+markers',
            name='Theoretical Complexity',
            line=dict(color='#FF9F40', width=3, dash='dash'),
            marker=dict(size=8)
        ), row=2, col=2)
        
        fig.add_trace(go.Scatter(
            x=branch_nums,
            y=time_ratios,
            mode='lines+markers',
            name='Actual Training Time',
            line=dict(color='#36A2EB', width=3),
            marker=dict(size=10)
        ), row=2, col=2)
        
        # 更新布局
        fig.update_layout(
            title={
                'text': "Branch Number Comparison Experiment: Systematic Analysis of DH-SNN Architecture",
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 18, 'color': '#2E86AB', 'family': 'Arial Black'}
            },
            height=800,
            width=1200,
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=-0.15,
                xanchor="center",
                x=0.5
            ),
            font=dict(size=12, family='Arial')
        )
        
        # 更新坐标轴
        fig.update_xaxes(title_text="Number of Branches", row=1, col=1)
        fig.update_yaxes(title_text="Test Accuracy (%)", row=1, col=1)
        
        fig.update_xaxes(title_text="Number of Branches", row=1, col=2)
        fig.update_yaxes(title_text="Parameter Ratio", row=1, col=2)
        fig.update_yaxes(title_text="Accuracy Improvement (%)", row=1, col=2, secondary_y=True)
        
        fig.update_xaxes(title_text="Number of Branches", row=2, col=1)
        fig.update_yaxes(title_text="Training Time Ratio", row=2, col=1)
        
        fig.update_xaxes(title_text="Number of Branches", row=2, col=2)
        fig.update_yaxes(title_text="Computational Overhead", row=2, col=2)
        
        # 保存图表
        html_path = os.path.join(self.save_dir, 'branch_comparison_analysis.html')
        fig.write_html(html_path)
        
        try:
            png_path = os.path.join(self.save_dir, 'branch_comparison_analysis.png')
            fig.write_image(png_path, width=1200, height=800, scale=2)
            print(f"✅ PNG图表已保存: {png_path}")
        except Exception as e:
            print(f"⚠️ PNG保存失败: {e}")
        
        print(f"✅ HTML图表已保存: {html_path}")
        
        return fig

def main():
    """主函数"""
    
    # 实验配置
    config = {
        'branch_numbers': [1, 2, 4, 8],  # 要测试的分支数量
        'num_trials': 3,  # 每个配置的试验次数
        
        # 模型参数 (多时间尺度XOR任务)
        'input_dim': 2,  # 2个输入通道
        'hidden_dims': [16],  # 隐藏层维度
        'output_dim': 2,  # XOR二分类
        'v_threshold': 1.0,
        'tau_m_init': (0.0, 4.0),
        'tau_n_init': (0.0, 4.0),
        
        # 训练参数
        'batch_size': 100,
        'learning_rate': 1e-2,
        'num_epochs': 50,  # 减少训练轮次以加快实验
        'weight_decay': 1e-4,
        'lr_step_size': 25,
        'lr_gamma': 0.5,
        'patience': 10,

        # 数据参数
        'train_samples': 2000,  # 减少样本数以加快实验
        'test_samples': 500,
        
        # 其他设置
        'num_workers': 4,
        'save_dir': 'experiments/branch_number_comparison/results'
    }
    
    # 创建实验对象
    experiment = BranchComparisonExperiment(config)
    
    # 运行实验
    results = experiment.run_all_experiments()
    
    # 创建可视化
    experiment.create_visualization()
    
    # 打印最终结果摘要
    print("\n" + "="*60)
    print("🎯 分支数量对比实验完成!")
    print("="*60)
    
    for result in results:
        print(f"📊 {result['num_branches']}分支: "
              f"{result['mean_accuracy']:.2f}% ± {result['std_accuracy']:.2f}% "
              f"(参数: {result['total_params']:,})")
    
    print("="*60)

if __name__ == "__main__":
    main()
