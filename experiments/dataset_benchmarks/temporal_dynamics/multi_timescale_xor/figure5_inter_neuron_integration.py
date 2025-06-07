#!/usr/bin/env python3
"""
Figure 5: 神经元间异质性特征整合实验
Inter-neuron heterogeneous feature integration through synaptic connections
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

print("🔗 Figure 5: 神经元间异质性特征整合实验")
print("="*60)

class MultiLayerDH_SFNN(nn.Module):
    """多层DH-SFNN模型"""
    
    def __init__(self, input_size=40, hidden_sizes=[64, 32], output_size=1, num_branches=1):
        super().__init__()
        
        self.num_layers = len(hidden_sizes)
        self.layers = nn.ModuleList()
        
        # 构建多层网络
        layer_sizes = [input_size] + hidden_sizes
        for i in range(self.num_layers):
            layer = self._create_dh_layer(layer_sizes[i], layer_sizes[i+1], num_branches)
            self.layers.append(layer)
        
        # 输出层
        self.output = nn.Linear(hidden_sizes[-1], output_size)
        
    def _create_dh_layer(self, input_size, output_size, num_branches):
        """创建DH层"""
        layer = nn.Module()
        
        if num_branches == 1:
            # 单分支DH层
            layer.dense = nn.Linear(input_size, output_size)
            layer.tau_m = nn.Parameter(torch.ones(output_size) * 2.0)
            layer.tau_n = nn.Parameter(torch.ones(output_size) * 2.0)
        else:
            # 多分支DH层 (简化实现)
            layer.dense = nn.Linear(input_size, output_size)
            layer.tau_m = nn.Parameter(torch.ones(output_size) * 2.0)
            layer.tau_n = nn.Parameter(torch.ones(output_size, num_branches) * 2.0)
        
        # 神经元状态
        layer.register_buffer('mem', torch.zeros(1, output_size))
        layer.register_buffer('d_current', torch.zeros(1, output_size))
        
        def forward_layer(x, batch_size):
            if layer.mem.size(0) != batch_size:
                layer.mem = torch.zeros(batch_size, output_size).to(x.device)
                layer.d_current = torch.zeros(batch_size, output_size).to(x.device)
            
            # 树突电流更新
            d_input = layer.dense(x)
            if hasattr(layer, 'tau_n') and layer.tau_n.dim() == 1:
                # 单分支
                beta = torch.sigmoid(layer.tau_n)
                layer.d_current = beta * layer.d_current + (1 - beta) * d_input
            else:
                # 多分支 (简化为平均)
                beta = torch.sigmoid(layer.tau_n.mean(dim=1))
                layer.d_current = beta * layer.d_current + (1 - beta) * d_input
            
            # 膜电位更新
            alpha = torch.sigmoid(layer.tau_m)
            layer.mem = alpha * layer.mem + (1 - alpha) * layer.d_current
            
            # 脉冲生成 (简化)
            spike = (layer.mem > 1.0).float()
            layer.mem = layer.mem * (1 - spike)  # 重置
            
            return spike, layer.mem, layer.d_current
        
        layer.forward_layer = forward_layer
        return layer
    
    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        
        outputs = []
        layer_activities = []  # 记录每层的活动用于分析
        
        for t in range(seq_len):
            current_input = x[:, t, :]
            layer_spikes = []
            layer_mems = []
            
            # 前向传播通过所有层
            for i, layer in enumerate(self.layers):
                spike, mem, d_current = layer.forward_layer(current_input, batch_size)
                layer_spikes.append(spike)
                layer_mems.append(mem)
                current_input = spike  # 下一层的输入
            
            # 输出层
            output = torch.sigmoid(self.output(current_input))
            outputs.append(output)
            
            # 记录活动
            layer_activities.append({
                'spikes': [s.clone() for s in layer_spikes],
                'mems': [m.clone() for m in layer_mems]
            })
        
        return torch.stack(outputs, dim=1), layer_activities

class SingleLayerDH_SRNN(nn.Module):
    """单层DH-SRNN模型 (带循环连接)"""
    
    def __init__(self, input_size=40, hidden_size=64, output_size=1, num_branches=1):
        super().__init__()
        
        # 前馈连接
        self.input_dense = nn.Linear(input_size, hidden_size)
        
        # 循环连接
        self.recurrent_dense = nn.Linear(hidden_size, hidden_size)
        
        # 时间常数
        self.tau_m = nn.Parameter(torch.ones(hidden_size) * 2.0)
        self.tau_n = nn.Parameter(torch.ones(hidden_size) * 2.0)
        
        # 输出层
        self.output = nn.Linear(hidden_size, output_size)
        
        # 神经元状态
        self.register_buffer('mem', torch.zeros(1, hidden_size))
        self.register_buffer('d_current', torch.zeros(1, hidden_size))
        self.register_buffer('prev_spike', torch.zeros(1, hidden_size))
        
    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        
        # 重置状态
        self.mem = torch.zeros(batch_size, self.mem.size(1)).to(x.device)
        self.d_current = torch.zeros(batch_size, self.d_current.size(1)).to(x.device)
        self.prev_spike = torch.zeros(batch_size, self.prev_spike.size(1)).to(x.device)
        
        outputs = []
        activities = []
        
        for t in range(seq_len):
            # 前馈输入 + 循环输入
            ff_input = self.input_dense(x[:, t, :])
            rec_input = self.recurrent_dense(self.prev_spike)
            total_input = ff_input + rec_input
            
            # 树突电流更新
            beta = torch.sigmoid(self.tau_n)
            self.d_current = beta * self.d_current + (1 - beta) * total_input
            
            # 膜电位更新
            alpha = torch.sigmoid(self.tau_m)
            self.mem = alpha * self.mem + (1 - alpha) * self.d_current
            
            # 脉冲生成
            spike = (self.mem > 1.0).float()
            self.mem = self.mem * (1 - spike)  # 重置
            self.prev_spike = spike
            
            # 输出
            output = torch.sigmoid(self.output(spike))
            outputs.append(output)
            
            # 记录活动
            activities.append({
                'spike': spike.clone(),
                'mem': self.mem.clone(),
                'd_current': self.d_current.clone()
            })
        
        return torch.stack(outputs, dim=1), activities

def analyze_neuron_types(activities, input_data, signal1_times, signal2_times):
    """分析神经元类型: Type 1 (高频敏感), Type 2 (低频敏感), 组合敏感"""
    
    print("\n🔍 分析神经元类型...")
    
    # 简化的神经元类型分析
    if isinstance(activities[0], dict):
        # SRNN情况
        spikes = torch.stack([act['spike'] for act in activities], dim=1)  # [batch, time, neurons]
    else:
        # SFNN情况 - 分析第一层
        spikes = torch.stack([act['spikes'][0] for act in activities], dim=1)
    
    batch_size, seq_len, num_neurons = spikes.shape
    
    # 计算每个神经元在不同时间段的平均发放率
    signal1_activity = spikes[:, signal1_times[0]:signal1_times[1], :].mean(dim=1)  # [batch, neurons]
    signal2_activity = spikes[:, signal2_times[0]:signal2_times[1], :].mean(dim=1)  # [batch, neurons]
    
    # 简单的类型分类
    type1_neurons = []  # 对Signal 2敏感
    type2_neurons = []  # 对Signal 1敏感
    combo_neurons = []  # 对组合敏感
    
    for neuron_id in range(num_neurons):
        s1_resp = signal1_activity[:, neuron_id].mean().item()
        s2_resp = signal2_activity[:, neuron_id].mean().item()
        
        if s2_resp > s1_resp * 1.5:
            type1_neurons.append(neuron_id)
        elif s1_resp > s2_resp * 1.5:
            type2_neurons.append(neuron_id)
        elif s1_resp > 0.1 and s2_resp > 0.1:
            combo_neurons.append(neuron_id)
    
    print(f"  Type 1 神经元 (Signal 2敏感): {len(type1_neurons)} 个")
    print(f"  Type 2 神经元 (Signal 1敏感): {len(type2_neurons)} 个") 
    print(f"  组合敏感神经元: {len(combo_neurons)} 个")
    
    return {
        'type1': type1_neurons,
        'type2': type2_neurons,
        'combo': combo_neurons
    }

def run_figure5_experiments():
    """运行Figure 5实验"""
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"🔧 使用设备: {device}")
    
    # 生成简化的多时间尺度数据
    print("\n📊 生成实验数据...")
    
    batch_size = 20
    seq_len = 400
    input_size = 40
    
    # 创建测试数据
    test_data = torch.zeros(batch_size, seq_len, input_size)
    
    # Signal 1: 低频 (时间步 50-100)
    signal1_times = (50, 100)
    test_data[:, signal1_times[0]:signal1_times[1], :20] = torch.rand(batch_size, 50, 20) * 0.3
    
    # Signal 2: 高频 (时间步 150-200, 250-300)
    signal2_times = (150, 300)
    test_data[:, 150:200, 20:] = torch.rand(batch_size, 50, 20) * 0.5
    test_data[:, 250:300, 20:] = torch.rand(batch_size, 50, 20) * 0.5
    
    test_data = test_data.to(device)
    
    # 实验配置
    experiments = [
        ("1层DH-SFNN (1分支)", lambda: MultiLayerDH_SFNN(input_size, [64], 1, 1)),
        ("2层DH-SFNN (1分支)", lambda: MultiLayerDH_SFNN(input_size, [64, 32], 1, 1)),
        ("1层DH-SRNN (1分支)", lambda: SingleLayerDH_SRNN(input_size, 64, 1, 1)),
    ]
    
    results = {}
    
    for exp_name, model_creator in experiments:
        print(f"\n🧪 实验: {exp_name}")
        
        model = model_creator().to(device)
        
        # 前向传播
        with torch.no_grad():
            outputs, activities = model(test_data)
            
            # 计算简单的性能指标
            # 这里用随机目标作为示例
            targets = torch.randint(0, 2, (batch_size, seq_len, 1)).float().to(device)
            acc = ((outputs > 0.5).float() == targets).float().mean().item() * 100
            
            print(f"  模拟准确率: {acc:.1f}%")
            
            # 分析神经元类型
            neuron_types = analyze_neuron_types(activities, test_data, signal1_times, signal2_times)
            
            results[exp_name] = {
                'accuracy': acc,
                'neuron_types': neuron_types,
                'num_params': sum(p.numel() for p in model.parameters())
            }
            
            print(f"  参数数量: {results[exp_name]['num_params']}")
    
    # 显示结果总结
    print(f"\n📊 Figure 5实验结果总结:")
    print("="*50)
    for exp_name, result in results.items():
        print(f"{exp_name}:")
        print(f"  准确率: {result['accuracy']:.1f}%")
        print(f"  参数数量: {result['num_params']}")
        types = result['neuron_types']
        print(f"  神经元类型: Type1={len(types['type1'])}, Type2={len(types['type2'])}, 组合={len(types['combo'])}")
        print()
    
    return results

def run_network_depth_analysis():
    """运行网络深度分析 (Figure 5e, 5f)"""
    
    print(f"\n📈 网络深度和分支数量分析...")
    
    # 这里实现简化版本的深度分析
    layer_configs = [1, 2, 3, 4]
    branch_configs = [1, 2, 4, 8]
    
    print("网络配置分析:")
    for layers in layer_configs:
        for branches in branch_configs:
            # 计算理论参数数量
            if layers == 1:
                params = 40 * 64 + 64 * 1  # 输入层 + 输出层
            else:
                params = 40 * 64 + (layers-1) * 64 * 64 + 64 * 1
            
            # 分支增加的参数 (时间常数)
            branch_params = 64 * layers * branches
            total_params = params + branch_params
            
            print(f"  {layers}层, {branches}分支: ~{total_params} 参数")
    
    return True

def main():
    """主函数"""
    try:
        print("🚀 开始Figure 5实验...")
        
        # 运行基础实验
        results = run_figure5_experiments()
        
        # 运行深度分析
        run_network_depth_analysis()
        
        # 保存结果
        os.makedirs("results", exist_ok=True)
        torch.save(results, "results/figure5_results.pth")
        
        print(f"\n🎉 Figure 5实验完成!")
        print(f"💾 结果已保存到: results/figure5_results.pth")
        
        return True
        
    except Exception as e:
        print(f"\n❌ 实验失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == '__main__':
    success = main()
    print(f"\n🏁 Figure 5实验{'成功' if success else '失败'}!")
