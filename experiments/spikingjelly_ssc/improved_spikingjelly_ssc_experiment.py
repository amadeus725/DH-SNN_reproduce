#!/usr/bin/env python3
"""
改进的SpikingJelly SSC实验
基于成功的多时间尺度XOR实现的训练策略
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import Dataset, DataLoader
import tables
import time
import os
from pathlib import Path

# SpikingJelly imports
from spikingjelly.activation_based import neuron, functional, layer, surrogate

print("🎯 改进的SpikingJelly SSC实验")
print("="*60)

# 改进的实验参数 - 基于成功的多时间尺度XOR实验
torch.manual_seed(42)  # 使用与成功实验相同的种子
batch_size = 100
learning_rate = 1e-2  # 保持原论文的学习率
num_epochs = 100  # 增加训练轮数
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 数据盘路径
TEMP_DIR = "/root/autodl-tmp/ssc_data"

def binary_image_readout_fast(times, units, dt=1e-3):
    """优化的binary_image_readout函数"""
    N = int(1/dt)
    img = np.zeros((N, 700), dtype=np.float32)
    
    # 向量化处理
    time_bins = (times / dt).astype(int)
    valid_mask = (time_bins < N) & (units > 0) & (units <= 700)
    
    if np.any(valid_mask):
        valid_times = time_bins[valid_mask]
        valid_units = units[valid_mask]
        img[valid_times, 700 - valid_units] = 1
    
    return img

# 改进的数据集类
class ImprovedSpikingJellySSCDataset(Dataset):
    def __init__(self, h5_file_path, max_samples=8000):
        self.h5_file_path = h5_file_path
        
        print(f"预加载SSC数据: {h5_file_path}")
        
        with tables.open_file(h5_file_path, mode='r') as f:
            total_samples = len(f.root.labels)
            self.indices = list(range(min(max_samples, total_samples)))
            
            print(f"  总样本数: {total_samples}, 使用: {len(self.indices)}")
            
            # 预加载数据
            self.data = []
            self.labels = []
            
            for i, idx in enumerate(self.indices):
                if i % 2000 == 0:
                    print(f"  预加载 {i+1}/{len(self.indices)}")
                
                times = f.root.spikes.times[idx]
                units = f.root.spikes.units[idx]
                label = f.root.labels[idx]
                
                # 转换为密集表示
                img = binary_image_readout_fast(times, units, dt=1e-3)
                
                self.data.append(img)
                self.labels.append(label)
            
            print(f"  预加载完成: {len(self.data)} 个样本")
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        # SpikingJelly期望的格式: [T, N] -> [T, 1, N]
        data = torch.FloatTensor(self.data[idx]).unsqueeze(1)  # [1000, 1, 700]
        label = torch.LongTensor([self.labels[idx]]).squeeze()
        return data, label

# 改进的Vanilla SNN - 基于成功的实现
class ImprovedVanillaSNN(nn.Module):
    def __init__(self, input_size=700, hidden_size=200, output_size=35):
        super(ImprovedVanillaSNN, self).__init__()
        
        # 使用LIF神经元 - 固定时间常数，避免参数冲突
        self.fc1 = layer.Linear(input_size, hidden_size)
        self.lif1 = neuron.LIFNode(
            tau=2.0,
            surrogate_function=surrogate.ATan(),
            detach_reset=True
        )
        
        self.fc2 = layer.Linear(hidden_size, output_size)
        self.lif2 = neuron.LIFNode(
            tau=2.0,
            surrogate_function=surrogate.ATan(),
            detach_reset=True
        )
        
    def forward(self, x):
        # x shape: [T, N, input_size]
        T, N = x.shape[0], x.shape[1]
        
        # 重置神经元状态
        functional.reset_net(self)
        
        outputs = []
        for t in range(T):
            x_t = x[t]  # [N, input_size]
            
            # 第一层
            h1 = self.fc1(x_t)
            s1 = self.lif1(h1)
            
            # 第二层
            h2 = self.fc2(s1)
            s2 = self.lif2(h2)
            
            outputs.append(s2)
        
        # 对时间维度求和（积分）
        output = torch.stack(outputs, dim=0).sum(0)  # [N, output_size]
        
        return F.log_softmax(output, dim=1)

# 改进的DH-SNN - 基于成功的多时间尺度XOR实现
class ImprovedDHSNN(nn.Module):
    def __init__(self, input_size=700, hidden_size=200, output_size=35, num_branches=2):
        super(ImprovedDHSNN, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_branches = num_branches
        
        # 分支线性层 - 每个分支处理输入的一部分
        self.branch1_layer = layer.Linear(input_size // 2, hidden_size, bias=False)
        self.branch2_layer = layer.Linear(input_size // 2, hidden_size, bias=False)
        
        # 可学习的时间常数参数 - 基于成功的实现
        self.tau_n = nn.Parameter(torch.empty(num_branches, hidden_size).uniform_(2, 6))
        self.tau_m = nn.Parameter(torch.empty(hidden_size).uniform_(0, 4))
        
        # 输出层
        self.output_layer = layer.Linear(hidden_size, output_size)
        
        # 状态变量
        self.dendritic_current1 = None
        self.dendritic_current2 = None
        self.membrane_potential = None
        self.spike_output = None
        
    def reset_states(self, batch_size):
        """重置神经元状态"""
        self.dendritic_current1 = torch.zeros(batch_size, self.hidden_size).to(device)
        self.dendritic_current2 = torch.zeros(batch_size, self.hidden_size).to(device)
        self.membrane_potential = torch.rand(batch_size, self.hidden_size).to(device)
        self.spike_output = torch.zeros(batch_size, self.hidden_size).to(device)
    
    def surrogate_gradient(self, x):
        """代理梯度函数"""
        return SurrogateGradient.apply(x)
    
    def forward(self, x):
        # x shape: [T, N, input_size]
        T, N = x.shape[0], x.shape[1]
        
        # 初始化状态
        self.reset_states(N)
        
        outputs = []
        for t in range(T):
            x_t = x[t]  # [N, input_size]
            
            # 分割输入：前半部分给Branch1，后半部分给Branch2
            input1 = x_t[:, :self.input_size//2]
            input2 = x_t[:, self.input_size//2:]
            
            # 分支线性变换
            branch1_input = self.branch1_layer(input1)
            branch2_input = self.branch2_layer(input2)
            
            # 更新树突电流 - 使用不同的时间常数
            beta1 = torch.sigmoid(self.tau_n[0])  # Branch 1 时间常数
            beta2 = torch.sigmoid(self.tau_n[1])  # Branch 2 时间常数
            
            self.dendritic_current1 = beta1 * self.dendritic_current1 + (1 - beta1) * branch1_input
            self.dendritic_current2 = beta2 * self.dendritic_current2 + (1 - beta2) * branch2_input
            
            # 汇总树突电流
            total_current = self.dendritic_current1 + self.dendritic_current2
            
            # 更新膜电位 - LIF动态
            alpha = torch.sigmoid(self.tau_m)
            R_m = 1.0
            v_th = 1.0
            
            self.membrane_potential = (alpha * self.membrane_potential + 
                                     (1 - alpha) * R_m * total_current - 
                                     v_th * self.spike_output)
            
            # 生成脉冲
            inputs_ = self.membrane_potential - v_th
            self.spike_output = self.surrogate_gradient(inputs_)
            
            outputs.append(self.spike_output)
        
        # 对时间维度求和（积分）
        output = torch.stack(outputs, dim=0).sum(0)  # [N, hidden_size]
        
        # 输出层
        final_output = self.output_layer(output)
        
        return F.log_softmax(final_output, dim=1)

# 代理梯度函数
class SurrogateGradient(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return input.gt(0).float()
    
    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        
        # 简化的代理梯度
        lens = 0.5
        gamma = 0.5
        temp = torch.exp(-(input**2)/(2*lens**2))/torch.sqrt(2*torch.tensor(torch.pi))/lens
        return grad_input * temp.float() * gamma

def prepare_improved_data():
    """准备改进的数据"""
    print("📊 准备改进的SpikingJelly格式数据...")
    
    # 原始数据路径
    data_path = Path("/root/DH-SNN_reproduce/datasets/raw/ssc/data")
    
    # 确保数据盘目录存在
    os.makedirs(TEMP_DIR, exist_ok=True)
    
    # 解压H5文件到数据盘
    train_h5_temp = Path(TEMP_DIR) / "ssc_train.h5"
    test_h5_temp = Path(TEMP_DIR) / "ssc_test.h5"
    
    if not train_h5_temp.exists():
        print("解压训练数据...")
        os.system(f"gunzip -c {data_path}/ssc_train.h5.gz > {train_h5_temp}")
    
    if not test_h5_temp.exists():
        print("解压测试数据...")
        os.system(f"gunzip -c {data_path}/ssc_test.h5.gz > {test_h5_temp}")
    
    # 创建数据集 - 使用更多数据
    print("创建训练数据集...")
    train_dataset = ImprovedSpikingJellySSCDataset(str(train_h5_temp), max_samples=15000)
    print("创建测试数据集...")
    test_dataset = ImprovedSpikingJellySSCDataset(str(test_h5_temp), max_samples=5000)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                             num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, 
                            num_workers=4, pin_memory=True)
    
    return train_loader, test_loader

def test_model(model, test_loader):
    """测试模型"""
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    
    return correct / total

def train_improved_model(model, train_loader, test_loader, epochs, model_name):
    """改进的训练函数 - 基于成功的多时间尺度XOR实现"""
    
    criterion = nn.CrossEntropyLoss()
    model.to(device)
    
    # 改进的优化器设置 - 基于成功的实现
    if isinstance(model, ImprovedDHSNN):
        # DH-SNN使用分层学习率
        base_params = [
            model.output_layer.weight,
            model.output_layer.bias,
            model.branch1_layer.weight,
            model.branch2_layer.weight,
        ]
        
        optimizer = torch.optim.Adam([
            {'params': base_params, 'lr': learning_rate},
            {'params': model.tau_n, 'lr': learning_rate},
            {'params': model.tau_m, 'lr': learning_rate},
        ])
    else:
        # Vanilla SNN使用标准优化器
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # 改进的学习率调度 - 延迟降低学习率
    scheduler = StepLR(optimizer, step_size=50, gamma=0.1)
    
    best_acc = 0
    
    print(f"🚀 开始训练 {model_name}")
    print(f"参数数量: {sum(p.numel() for p in model.parameters())}")
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        correct = 0
        total = 0
        
        epoch_start = time.time()
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            
            # 添加梯度裁剪 - 基于成功的实现
            torch.nn.utils.clip_grad_norm_(model.parameters(), 20)
            
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
        
        scheduler.step()
        
        train_acc = correct / total
        test_acc = test_model(model, test_loader)
        epoch_time = time.time() - epoch_start
        
        if test_acc > best_acc and train_acc > 0.5:
            best_acc = test_acc
        
        print(f'Epoch: {epoch:3d}, Train Loss: {train_loss/len(train_loader):.4f}, '
              f'Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}, '
              f'Best: {best_acc:.4f}, Time: {epoch_time:.1f}s')
        
        # 改进的早停条件
        if model_name == "Improved Vanilla SNN" and best_acc > 0.70:
            print(f"✅ Vanilla SNN达到70%以上，提前停止训练")
            break
        elif model_name == "Improved DH-SNN" and best_acc > 0.80:
            print(f"✅ DH-SNN达到80%以上，提前停止训练")
            break
    
    return best_acc

def main():
    """主函数"""
    
    print(f"🔧 使用设备: {device}")
    print(f"🔧 数据盘路径: {TEMP_DIR}")
    print(f"🔧 改进的SpikingJelly版本")
    
    try:
        # 准备数据
        train_loader, test_loader = prepare_improved_data()
        
        print(f"\n📊 数据集统计:")
        print(f"训练批次数: {len(train_loader)}")
        print(f"测试批次数: {len(test_loader)}")
        
        # 实验配置
        experiments = [
            ("Improved Vanilla SNN", ImprovedVanillaSNN),
            ("Improved DH-SNN", ImprovedDHSNN),
        ]
        
        results = {}
        start_time = time.time()
        
        for exp_name, model_class in experiments:
            print(f"\n🔬 实验: {exp_name}")
            print("="*50)
            
            model = model_class()
            best_acc = train_improved_model(model, train_loader, test_loader, num_epochs, exp_name)
            results[exp_name] = best_acc * 100
            
            print(f"✅ {exp_name} 最佳准确率: {best_acc*100:.1f}%")
        
        # 结果总结
        total_time = time.time() - start_time
        print(f"\n🎉 改进的SpikingJelly SSC实验完成! 用时: {total_time/60:.1f}分钟")
        print("="*60)
        print("改进的SpikingJelly SSC实验结果:")
        print("="*60)
        
        vanilla_acc = results.get("Improved Vanilla SNN", 0)
        dh_acc = results.get("Improved DH-SNN", 0)
        improvement = dh_acc - vanilla_acc
        
        print(f"Improved Vanilla SNN:   {vanilla_acc:.1f}%")
        print(f"Improved DH-SNN:        {dh_acc:.1f}%")
        print(f"性能提升:               {improvement:+.1f} 个百分点")
        
        # 与论文结果对比
        print(f"\n📊 与论文结果对比:")
        print(f"论文Vanilla SNN:   ~70%")
        print(f"论文DH-SNN:        ~80%")
        
        if improvement > 5:
            print("🎉 改进的DH-SNN显著优于Vanilla SNN!")
        elif improvement > 0:
            print("✅ 改进的DH-SNN优于Vanilla SNN")
        else:
            print("⚠️ 结果需要进一步分析")
        
        return results
        
    except Exception as e:
        print(f"\n❌ 实验失败: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == '__main__':
    results = main()
    if results:
        print(f"\n🏁 改进的SpikingJelly SSC实验成功完成!")
    else:
        print(f"\n❌ 改进的SpikingJelly SSC实验失败")
