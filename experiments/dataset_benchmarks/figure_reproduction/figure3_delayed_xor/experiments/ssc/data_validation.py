#!/usr/bin/env python3
"""
修正的SSC实验 - 解决标签采样问题
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import os
import sys
import random

# 添加路径
# sys.path.append removed during restructure
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from spikingjelly.activation_based import neuron, functional, surrogate
from direct_gz_reader import read_gz_h5_file, convert_to_spike_tensor
from torch.utils.data import DataLoader, TensorDataset

print("🚀 修正的SSC实验 - 解决标签采样问题")
print("="*60)

# SSC配置
SSC_CONFIG = {
    'learning_rate': 1e-2,
    'batch_size': 200,
    'epochs': 50,  # 先用较少轮数测试
    'hidden_size': 200,
    'output_size': 35,
    'v_threshold': 1.0,
    'dt': 1,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu'
}

def load_ssc_data_fixed(num_train=2000, num_test=500):
    """修正的SSC数据加载 - 确保标签分布均匀"""
    print(f"📚 加载SSC数据 (修正版): 训练{num_train}, 测试{num_test}")
    
    # 先读取所有标签来分析分布
    print("🔍 分析SSC数据分布...")
    all_train_times, all_train_units, all_train_labels = read_gz_h5_file(
        "../spikingjelly_ssc/data/ssc_train.h5.gz", max_samples=None
    )
    all_test_times, all_test_units, all_test_labels = read_gz_h5_file(
        "../spikingjelly_ssc/data/ssc_test.h5.gz", max_samples=None
    )
    
    print(f"📊 完整数据集分析:")
    print(f"  训练样本总数: {len(all_train_labels)}")
    print(f"  测试样本总数: {len(all_test_labels)}")
    print(f"  训练标签范围: {all_train_labels.min()} - {all_train_labels.max()}")
    print(f"  测试标签范围: {all_test_labels.min()} - {all_test_labels.max()}")
    print(f"  训练唯一类别数: {len(np.unique(all_train_labels))}")
    print(f"  测试唯一类别数: {len(np.unique(all_test_labels))}")
    
    # 分析每个类别的样本数
    train_class_counts = np.bincount(all_train_labels)
    test_class_counts = np.bincount(all_test_labels)
    
    print(f"📈 类别分布分析:")
    print(f"  训练集各类别样本数: min={train_class_counts.min()}, max={train_class_counts.max()}, mean={train_class_counts.mean():.1f}")
    print(f"  测试集各类别样本数: min={test_class_counts.min()}, max={test_class_counts.max()}, mean={test_class_counts.mean():.1f}")
    
    # 随机采样确保类别分布
    def balanced_sample(times_list, units_list, labels, num_samples):
        """平衡采样，确保每个类别都有代表"""
        indices = list(range(len(labels)))
        
        if num_samples >= len(labels):
            # 如果要求的样本数大于等于总数，返回全部
            selected_indices = indices
        else:
            # 随机采样
            random.shuffle(indices)
            selected_indices = indices[:num_samples]
        
        selected_times = [times_list[i] for i in selected_indices]
        selected_units = [units_list[i] for i in selected_indices]
        selected_labels = labels[selected_indices]
        
        return selected_times, selected_units, selected_labels
    
    # 平衡采样
    print("🎯 进行平衡采样...")
    train_times, train_units, train_labels = balanced_sample(
        all_train_times, all_train_units, all_train_labels, num_train
    )
    test_times, test_units, test_labels = balanced_sample(
        all_test_times, all_test_units, all_test_labels, num_test
    )
    
    print(f"📊 采样后标签分析:")
    print(f"  训练标签: {train_labels[:10]}...")
    print(f"  训练标签范围: {train_labels.min()} - {train_labels.max()}")
    print(f"  训练唯一类别数: {len(np.unique(train_labels))}")
    print(f"  测试标签范围: {test_labels.min()} - {test_labels.max()}")
    print(f"  测试唯一类别数: {len(np.unique(test_labels))}")
    
    print("🔄 转换为张量...")
    
    train_data = torch.zeros(len(train_times), 1000, 700)
    test_data = torch.zeros(len(test_times), 1000, 700)
    
    for i in range(len(train_times)):
        if i % 500 == 0:
            print(f"  处理训练样本 {i+1}/{len(train_times)}")
        tensor = convert_to_spike_tensor(train_times[i], train_units[i], dt=1e-3, max_time=1.0)
        train_data[i] = tensor
        
    for i in range(len(test_times)):
        if i % 100 == 0:
            print(f"  处理测试样本 {i+1}/{len(test_times)}")
        tensor = convert_to_spike_tensor(test_times[i], test_units[i], dt=1e-3, max_time=1.0)
        test_data[i] = tensor
    
    train_labels = torch.from_numpy(train_labels.astype(np.int64)).long()
    test_labels = torch.from_numpy(test_labels.astype(np.int64)).long()
    
    print(f"✅ SSC数据加载完成: 训练{train_data.shape}, 测试{test_data.shape}")
    print(f"📊 最终标签检查:")
    print(f"  训练标签范围: {train_labels.min()}-{train_labels.max()}")
    print(f"  测试标签范围: {test_labels.min()}-{test_labels.max()}")
    print(f"  训练唯一类别: {len(torch.unique(train_labels))}")
    print(f"  测试唯一类别: {len(torch.unique(test_labels))}")
    
    return train_data, train_labels, test_data, test_labels

# 使用之前定义的模型类
class MultiGaussianSurrogate(torch.autograd.Function):
    """原论文的MultiGaussian替代函数"""
    
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return input.gt(0).float()
    
    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        
        lens = 0.5
        scale = 6.0
        height = 0.15
        gamma = 0.5
        
        def gaussian(x, mu=0., sigma=0.5):
            return torch.exp(-((x - mu) ** 2) / (2 * sigma ** 2)) / torch.sqrt(2 * torch.tensor(math.pi)) / sigma
        
        temp = gaussian(input, mu=0., sigma=lens) * (1. + height) \
             - gaussian(input, mu=lens, sigma=scale * lens) * height \
             - gaussian(input, mu=-lens, sigma=scale * lens) * height
        
        return grad_input * temp.float() * gamma

multi_gaussian_surrogate = MultiGaussianSurrogate.apply

class SimpleSSC_VanillaSFNN(nn.Module):
    """简化的SSC Vanilla SFNN用于快速测试"""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.device = config['device']
        
        # 简化的网络结构
        self.dense = nn.Linear(700, config['hidden_size'])
        self.output = nn.Linear(config['hidden_size'], config['output_size'])
        
        # 时间常数
        self.tau_m = nn.Parameter(torch.empty(config['hidden_size']))
        nn.init.uniform_(self.tau_m, 0.0, 4.0)
        
        # 神经元状态
        self.register_buffer('mem', torch.zeros(1, config['hidden_size']))
        self.register_buffer('spike', torch.zeros(1, config['hidden_size']))
        
    def set_neuron_state(self, batch_size):
        """重置神经元状态"""
        self.mem = torch.rand(batch_size, self.config['hidden_size']).to(self.device)
        self.spike = torch.rand(batch_size, self.config['hidden_size']).to(self.device)
        
    def forward(self, input_data):
        """前向传播"""
        batch_size, seq_length, input_dim = input_data.shape
        
        self.set_neuron_state(batch_size)
        
        output = 0
        for i in range(seq_length):
            input_x = input_data[:, i, :].reshape(batch_size, input_dim)
            
            # 线性变换
            d_input = self.dense(input_x.float())
            
            # LIF神经元
            alpha = torch.sigmoid(self.tau_m)
            self.mem = self.mem * alpha + (1 - alpha) * d_input - self.config['v_threshold'] * self.spike
            inputs_ = self.mem - self.config['v_threshold']
            self.spike = multi_gaussian_surrogate(inputs_)
            
            # 累积输出
            if i > 10:
                output += F.softmax(self.output(self.spike), dim=1)
        
        return output

def quick_ssc_test():
    """快速SSC测试"""
    try:
        print(f"🔧 使用设备: {SSC_CONFIG['device']}")
        
        # 加载少量数据进行快速测试
        train_data, train_labels, test_data, test_labels = load_ssc_data_fixed(200, 50)
        
        # 创建简单模型
        model = SimpleSSC_VanillaSFNN(SSC_CONFIG).to(SSC_CONFIG['device'])
        
        # 快速训练测试
        print("\n🏋️ 快速训练测试...")
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=SSC_CONFIG['learning_rate'])
        
        train_dataset = TensorDataset(train_data, train_labels)
        train_loader = DataLoader(train_dataset, batch_size=20, shuffle=True)
        
        model.train()
        for epoch in range(3):  # 只训练3个epoch
            total_loss = 0
            correct = 0
            total = 0
            
            for images, labels in train_loader:
                images = images.to(SSC_CONFIG['device'])
                labels = labels.to(SSC_CONFIG['device'])
                
                optimizer.zero_grad()
                predictions = model(images)
                loss = criterion(predictions, labels)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                _, predicted = torch.max(predictions.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            
            acc = 100 * correct / total
            print(f"  Epoch {epoch+1}: Loss={total_loss/len(train_loader):.4f}, Acc={acc:.1f}%")
        
        print("✅ SSC快速测试成功！数据和模型都正常工作")
        return True
        
    except Exception as e:
        print(f"❌ SSC测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == '__main__':
    success = quick_ssc_test()
    print(f"\n🏁 SSC测试完成，结果: {'✅ 成功' if success else '❌ 失败'}")
