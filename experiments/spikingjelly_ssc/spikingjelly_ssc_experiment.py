#!/usr/bin/env python3
"""
基于SpikingJelly框架的SSC实验
参考成功的SHD实验实现
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

print("🎯 基于SpikingJelly框架的SSC实验")
print("="*60)

# 实验参数
torch.manual_seed(0)
batch_size = 100
learning_rate = 1e-2
num_epochs = 50
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

# SpikingJelly版本的数据集类
class SpikingJellySSCDataset(Dataset):
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

# SpikingJelly版本的Vanilla SNN
class VanillaSNN(nn.Module):
    def __init__(self, input_size=700, hidden_size=200, output_size=35, tau_m_init=(0.0, 4.0)):
        super(VanillaSNN, self).__init__()
        
        # 使用LIF神经元
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
        
        # 初始化时间常数
        self._init_tau_parameters(tau_m_init)
        
    def _init_tau_parameters(self, tau_m_init):
        """初始化时间常数参数"""
        low_m, high_m = tau_m_init
        
        # 将tau参数设为可学习
        self.lif1.tau = nn.Parameter(torch.empty(1).uniform_(low_m, high_m))
        self.lif2.tau = nn.Parameter(torch.empty(1).uniform_(low_m, high_m))
        
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

# SpikingJelly版本的DH-SNN
class DHSNN(nn.Module):
    def __init__(self, input_size=700, hidden_size=200, output_size=35, 
                 tau_m_init=(0.0, 4.0), tau_n_init=(2.0, 6.0), num_branches=4):
        super(DHSNN, self).__init__()
        
        self.num_branches = num_branches
        self.hidden_size = hidden_size
        
        # 多分支连接
        self.branch_fcs = nn.ModuleList([
            layer.Linear(input_size // num_branches, hidden_size)
            for _ in range(num_branches)
        ])
        
        # 分支时间常数（树突时间常数）
        self.branch_taus = nn.ParameterList([
            nn.Parameter(torch.empty(hidden_size).uniform_(*tau_n_init))
            for _ in range(num_branches)
        ])
        
        # 主神经元
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
        
        # 初始化膜时间常数
        self._init_tau_parameters(tau_m_init)
        
        # 分支状态
        self.register_buffer('branch_states', torch.zeros(1, hidden_size, num_branches))
        
    def _init_tau_parameters(self, tau_m_init):
        """初始化膜时间常数参数"""
        low_m, high_m = tau_m_init
        
        self.lif1.tau = nn.Parameter(torch.empty(1).uniform_(low_m, high_m))
        self.lif2.tau = nn.Parameter(torch.empty(1).uniform_(low_m, high_m))
        
    def forward(self, x):
        # x shape: [T, N, input_size]
        T, N = x.shape[0], x.shape[1]
        
        # 重置神经元状态
        functional.reset_net(self)
        
        # 重置分支状态
        self.branch_states = torch.zeros(N, self.hidden_size, self.num_branches).to(x.device)
        
        outputs = []
        for t in range(T):
            x_t = x[t]  # [N, input_size]
            
            # 分支处理
            branch_outputs = []
            for i, (branch_fc, branch_tau) in enumerate(zip(self.branch_fcs, self.branch_taus)):
                # 输入分割
                start_idx = i * (x_t.shape[1] // self.num_branches)
                end_idx = (i + 1) * (x_t.shape[1] // self.num_branches)
                x_branch = x_t[:, start_idx:end_idx]
                
                # 分支前向传播
                branch_input = branch_fc(x_branch)
                
                # 分支时间常数更新
                alpha = torch.sigmoid(branch_tau)
                self.branch_states[:, :, i] = alpha * self.branch_states[:, :, i] + (1 - alpha) * branch_input
                
                branch_outputs.append(self.branch_states[:, :, i])
            
            # 合并分支输出
            combined_input = torch.stack(branch_outputs, dim=2).sum(dim=2)  # [N, hidden_size]
            
            # 主神经元处理
            s1 = self.lif1(combined_input)
            
            # 输出层
            h2 = self.fc2(s1)
            s2 = self.lif2(h2)
            
            outputs.append(s2)
        
        # 对时间维度求和（积分）
        output = torch.stack(outputs, dim=0).sum(0)  # [N, output_size]
        
        return F.log_softmax(output, dim=1)

def prepare_spikingjelly_data():
    """准备SpikingJelly格式的数据"""
    print("📊 准备SpikingJelly格式数据...")
    
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
    
    # 创建数据集
    print("创建训练数据集...")
    train_dataset = SpikingJellySSCDataset(str(train_h5_temp), max_samples=8000)
    print("创建测试数据集...")
    test_dataset = SpikingJellySSCDataset(str(test_h5_temp), max_samples=3000)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
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

def train_model(model, train_loader, test_loader, epochs, model_name):
    """训练模型"""
    
    criterion = nn.CrossEntropyLoss()
    model.to(device)
    
    # 优化器设置
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = StepLR(optimizer, step_size=25, gamma=0.1)
    
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
        
        # 早停条件
        if model_name == "Vanilla SNN" and best_acc > 0.65:
            print(f"✅ Vanilla SNN达到65%以上，提前停止训练")
            break
        elif model_name == "DH-SNN" and best_acc > 0.75:
            print(f"✅ DH-SNN达到75%以上，提前停止训练")
            break
    
    return best_acc

def main():
    """主函数"""
    
    print(f"🔧 使用设备: {device}")
    print(f"🔧 数据盘路径: {TEMP_DIR}")
    print(f"🔧 SpikingJelly版本")
    
    try:
        # 准备数据
        train_loader, test_loader = prepare_spikingjelly_data()
        
        print(f"\n📊 数据集统计:")
        print(f"训练批次数: {len(train_loader)}")
        print(f"测试批次数: {len(test_loader)}")
        
        # 实验配置
        experiments = [
            ("Vanilla SNN", VanillaSNN),
            ("DH-SNN", DHSNN),
        ]
        
        results = {}
        start_time = time.time()
        
        for exp_name, model_class in experiments:
            print(f"\n🔬 实验: {exp_name}")
            print("="*50)
            
            model = model_class()
            best_acc = train_model(model, train_loader, test_loader, num_epochs, exp_name)
            results[exp_name] = best_acc * 100
            
            print(f"✅ {exp_name} 最佳准确率: {best_acc*100:.1f}%")
        
        # 结果总结
        total_time = time.time() - start_time
        print(f"\n🎉 SpikingJelly SSC实验完成! 用时: {total_time/60:.1f}分钟")
        print("="*60)
        print("SpikingJelly SSC实验结果:")
        print("="*60)
        
        vanilla_acc = results.get("Vanilla SNN", 0)
        dh_acc = results.get("DH-SNN", 0)
        improvement = dh_acc - vanilla_acc
        
        print(f"Vanilla SNN:   {vanilla_acc:.1f}%")
        print(f"DH-SNN:        {dh_acc:.1f}%")
        print(f"性能提升:      {improvement:+.1f} 个百分点")
        
        # 与论文结果对比
        print(f"\n📊 与论文结果对比:")
        print(f"论文Vanilla SNN:   ~70%")
        print(f"论文DH-SNN:        ~80%")
        
        if improvement > 5:
            print("🎉 DH-SNN显著优于Vanilla SNN!")
        elif improvement > 0:
            print("✅ DH-SNN优于Vanilla SNN")
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
        print(f"\n🏁 SpikingJelly SSC实验成功完成!")
    else:
        print(f"\n❌ SpikingJelly SSC实验失败")
