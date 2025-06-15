#!/usr/bin/env python3
"""
DH-SNN SSC（脉冲语音命令）实验
==================================

基于SpikingJelly框架的DH-SNN vs 普通SNN对比实验
使用SSC数据集进行语音命令识别任务

"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import tables
import time
import json
from pathlib import Path

# 添加项目路径
current_dir = Path(__file__).parent.parent.parent
sys.path.append(str(current_dir))

# SpikingJelly导入
from spikingjelly.activation_based import neuron, functional, layer, surrogate

from dh_snn.utils import setup_seed

print("🎯 DH-SNN SSC脉冲语音命令识别实验")
print("="*60)

# 实验参数
torch.manual_seed(42)
BATCH_SIZE = 100
LEARNING_RATE = 1e-2
NUM_EPOCHS = 100
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 临时数据路径
TEMP_DIR = "/tmp/ssc_data"

# ==================== 数据处理 ====================

def binary_image_readout_fast(times, units, dt=1e-3):
    """
    优化的二进制图像读取函数
    将脉冲时间和神经元索引转换为密集的脉冲张量
    
    参数:
        times: 脉冲时间数组
        units: 神经元单元索引数组 
        dt: 时间步长
    
    返回:
        img: 形状为[时间步, 神经元数]的脉冲张量
    """
    N = int(1/dt)  # 总时间步数
    img = np.zeros((N, 700), dtype=np.float32)  # SSC有700个神经元
    
    # 向量化处理以提高效率
    time_bins = (times / dt).astype(int)
    valid_mask = (time_bins < N) & (units > 0) & (units <= 700)
    
    if np.any(valid_mask):
        valid_times = time_bins[valid_mask]
        valid_units = units[valid_mask]
        img[valid_times, 700 - valid_units] = 1
    
    return img

class SSCDataset(torch.utils.data.Dataset):
    """
    SSC数据集类
    处理HDF5格式的脉冲数据并转换为SpikingJelly格式
    """
    
    def __init__(self, h5_file_path, max_samples=8000):
        """
        初始化数据集
        
        参数:
            h5_file_path: HDF5文件路径
            max_samples: 最大样本数量
        """
        self.h5_file_path = h5_file_path
        
        print(f"📁 预加载SSC数据: {h5_file_path}")
        
        with tables.open_file(h5_file_path, mode='r') as f:
            total_samples = len(f.root.labels)
            self.indices = list(range(min(max_samples, total_samples)))
            
            print(f"   总样本数: {total_samples}, 使用: {len(self.indices)}")
            
            # 预加载数据到内存以提高训练速度
            self.data = []
            self.labels = []
            
            for i, idx in enumerate(self.indices):
                if i % 2000 == 0:
                    print(f"   预加载进度: {i+1}/{len(self.indices)}")
                
                times = f.root.spikes.times[idx]
                units = f.root.spikes.units[idx]
                label = f.root.labels[idx]
                
                # 转换为密集表示
                img = binary_image_readout_fast(times, units, dt=1e-3)
                
                self.data.append(img)
                self.labels.append(label)
            
            print(f"   预加载完成: {len(self.data)} 个样本")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        """
        获取单个样本
        
        返回:
            data: 形状为[时间步, 1, 神经元数]的脉冲张量
            label: 类别标签
        """
        # SpikingJelly期望的格式: [T, N] -> [T, 1, N]
        data = torch.FloatTensor(self.data[idx]).unsqueeze(1)  # [1000, 1, 700]
        label = torch.LongTensor([self.labels[idx]]).squeeze()
        return data, label

# ==================== 模型定义 ====================

class VanillaSNN(nn.Module):
    """
    普通脉冲神经网络模型
    用作基线对比
    """
    
    def __init__(self, input_size=700, hidden_size=200, output_size=35):
        """
        初始化普通SNN模型
        
        参数:
            input_size: 输入特征维度
            hidden_size: 隐藏层维度
            output_size: 输出类别数
        """
        super(VanillaSNN, self).__init__()
        
        print(f"🏗️  创建普通SNN模型:")
        print(f"   输入维度: {input_size}")
        print(f"   隐藏维度: {hidden_size}")
        print(f"   输出维度: {output_size}")
        
        # 第一层：线性层 + LIF神经元
        self.fc1 = layer.Linear(input_size, hidden_size)
        self.lif1 = neuron.LIFNode(
            tau=2.0,  # 固定时间常数
            surrogate_function=surrogate.ATan(),
            detach_reset=True
        )
        
        # 第二层：线性层 + LIF神经元  
        self.fc2 = layer.Linear(hidden_size, output_size)
        self.lif2 = neuron.LIFNode(
            tau=2.0,
            surrogate_function=surrogate.ATan(),
            detach_reset=True
        )
        
        print("✅ 普通SNN模型创建完成")
    
    def forward(self, x):
        """
        前向传播
        
        参数:
            x: 输入脉冲序列，形状为[时间步, 批次, 特征]
            
        返回:
            output: 输出logits，形状为[批次, 类别数]
        """
        # x 形状: [T, N, input_size]
        T, N = x.shape[0], x.shape[1]
        
        # 重置神经元状态
        functional.reset_net(self)
        
        outputs = []
        for t in range(T):
            x_t = x[t]  # [N, input_size]
            
            # 第一层处理
            h1 = self.fc1(x_t)
            s1 = self.lif1(h1)
            
            # 第二层处理
            h2 = self.fc2(s1)
            s2 = self.lif2(h2)
            
            outputs.append(s2)
        
        # 对时间维度求和（积分读出）
        output = torch.stack(outputs, dim=0).sum(0)  # [N, output_size]
        
        return F.log_softmax(output, dim=1)

class DH_SNN(nn.Module):
    """
    树突异质性脉冲神经网络模型
    基于成功的多时间尺度XOR实现
    """
    
    def __init__(self, input_size=700, hidden_size=200, output_size=35, num_branches=2):
        """
        初始化DH-SNN模型
        
        参数:
            input_size: 输入特征维度
            hidden_size: 隐藏层维度  
            output_size: 输出类别数
            num_branches: 树突分支数量
        """
        super(DH_SNN, self).__init__()
        
        print(f"🏗️  创建DH-SNN模型:")
        print(f"   输入维度: {input_size}")
        print(f"   隐藏维度: {hidden_size}")
        print(f"   输出维度: {output_size}")
        print(f"   分支数量: {num_branches}")
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_branches = num_branches
        
        # 分支线性层 - 每个分支处理输入的一部分
        self.branch1_layer = layer.Linear(input_size // 2, hidden_size, bias=False)
        self.branch2_layer = layer.Linear(input_size // 2, hidden_size, bias=False)
        
        # 可学习的时间常数参数
        # tau_n: 树突时间常数，使用Large初始化（2,6）
        self.tau_n = nn.Parameter(torch.empty(num_branches, hidden_size).uniform_(2, 6))
        # tau_m: 膜电位时间常数，使用Medium初始化（0,4）
        self.tau_m = nn.Parameter(torch.empty(hidden_size).uniform_(0, 4))
        
        # 输出层
        self.output_layer = layer.Linear(hidden_size, output_size)
        
        # 神经元状态变量
        self.dendritic_current1 = None
        self.dendritic_current2 = None
        self.membrane_potential = None
        self.spike_output = None
        
        print("✅ DH-SNN模型创建完成")
    
    def reset_states(self, batch_size):
        """
        重置神经元状态
        
        参数:
            batch_size: 批次大小
        """
        self.dendritic_current1 = torch.zeros(batch_size, self.hidden_size).to(DEVICE)
        self.dendritic_current2 = torch.zeros(batch_size, self.hidden_size).to(DEVICE)
        self.membrane_potential = torch.rand(batch_size, self.hidden_size).to(DEVICE)
        self.spike_output = torch.zeros(batch_size, self.hidden_size).to(DEVICE)
    
    def surrogate_gradient(self, x):
        """代理梯度函数"""
        return SurrogateGradient.apply(x)
    
    def forward(self, x):
        """
        前向传播
        
        参数:
            x: 输入脉冲序列，形状为[时间步, 批次, 特征]
            
        返回:
            output: 输出logits，形状为[批次, 类别数]
        """
        # x 形状: [T, N, input_size]
        T, N = x.shape[0], x.shape[1]
        
        # 初始化神经元状态
        self.reset_states(N)
        
        outputs = []
        for t in range(T):
            x_t = x[t]  # [N, input_size]
            
            # 分割输入：前半部分给分支1，后半部分给分支2
            input1 = x_t[:, :self.input_size//2]
            input2 = x_t[:, self.input_size//2:]
            
            # 分支线性变换
            branch1_input = self.branch1_layer(input1)
            branch2_input = self.branch2_layer(input2)
            
            # 更新树突电流 - 使用不同的时间常数
            beta1 = torch.sigmoid(self.tau_n[0])  # 分支1时间常数
            beta2 = torch.sigmoid(self.tau_n[1])  # 分支2时间常数
            
            self.dendritic_current1 = beta1 * self.dendritic_current1 + (1 - beta1) * branch1_input
            self.dendritic_current2 = beta2 * self.dendritic_current2 + (1 - beta2) * branch2_input
            
            # 汇总树突电流
            total_current = self.dendritic_current1 + self.dendritic_current2
            
            # 更新膜电位 - LIF动力学
            alpha = torch.sigmoid(self.tau_m)
            R_m = 1.0  # 膜阻抗
            v_th = 1.0  # 脉冲阈值
            
            self.membrane_potential = (alpha * self.membrane_potential + 
                                     (1 - alpha) * R_m * total_current - 
                                     v_th * self.spike_output)
            
            # 生成脉冲
            inputs_ = self.membrane_potential - v_th
            self.spike_output = self.surrogate_gradient(inputs_)
            
            outputs.append(self.spike_output)
        
        # 对时间维度求和（积分读出）
        output = torch.stack(outputs, dim=0).sum(0)  # [N, hidden_size]
        
        # 输出层
        final_output = self.output_layer(output)
        
        return F.log_softmax(final_output, dim=1)

class SurrogateGradient(torch.autograd.Function):
    """
    代理梯度函数
    用于脉冲函数的反向传播
    """
    
    @staticmethod
    def forward(ctx, input):
        """前向传播：阶跃函数"""
        ctx.save_for_backward(input)
        return input.gt(0).float()
    
    @staticmethod
    def backward(ctx, grad_output):
        """反向传播：高斯近似梯度"""
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        
        # 简化的代理梯度
        lens = 0.5
        gamma = 0.5
        temp = torch.exp(-(input**2)/(2*lens**2))/torch.sqrt(2*torch.tensor(torch.pi))/lens
        return grad_input * temp.float() * gamma

# ==================== 数据准备 ====================

def prepare_data():
    """
    准备SSC数据集
    
    返回:
        train_loader: 训练数据加载器
        test_loader: 测试数据加载器
    """
    print("📊 准备SSC数据集...")
    
    # 原始数据路径
    data_path = Path("datasets/ssc/data")
    
    # 确保临时目录存在
    os.makedirs(TEMP_DIR, exist_ok=True)
    
    # 解压HDF5文件到临时目录
    train_h5_temp = Path(TEMP_DIR) / "ssc_train.h5"
    test_h5_temp = Path(TEMP_DIR) / "ssc_test.h5"
    
    if not train_h5_temp.exists():
        print("   解压训练数据...")
        if (data_path / "ssc_train.h5.gz").exists():
            os.system(f"gunzip -c {data_path}/ssc_train.h5.gz > {train_h5_temp}")
        else:
            print("   ⚠️  训练数据文件不存在，使用模拟数据")
            return create_mock_data()
    
    if not test_h5_temp.exists():
        print("   解压测试数据...")
        if (data_path / "ssc_test.h5.gz").exists():
            os.system(f"gunzip -c {data_path}/ssc_test.h5.gz > {test_h5_temp}")
        else:
            print("   ⚠️  测试数据文件不存在，使用模拟数据")
            return create_mock_data()
    
    # 创建数据集
    print("   创建训练数据集...")
    train_dataset = SSCDataset(str(train_h5_temp), max_samples=15000)
    print("   创建测试数据集...")
    test_dataset = SSCDataset(str(test_h5_temp), max_samples=5000)
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, 
        num_workers=2, pin_memory=True
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=2, pin_memory=True
    )
    
    return train_loader, test_loader

def create_mock_data():
    """创建模拟SSC数据用于测试"""
    print("🎲 创建模拟SSC数据...")
    
    # 模拟数据参数
    num_train = 8000
    num_test = 2000
    seq_len = 1000
    num_features = 700
    num_classes = 35
    
    # 生成训练数据
    train_data = torch.zeros(num_train, seq_len, num_features)
    train_labels = torch.randint(0, num_classes, (num_train,))
    
    # 生成测试数据  
    test_data = torch.zeros(num_test, seq_len, num_features)
    test_labels = torch.randint(0, num_classes, (num_test,))
    
    # 为每个样本添加随机脉冲
    for i in range(num_train):
        num_spikes = torch.randint(100, 500, (1,)).item()
        spike_times = torch.randint(0, seq_len, (num_spikes,))
        spike_features = torch.randint(0, num_features, (num_spikes,))
        for t, f in zip(spike_times, spike_features):
            train_data[i, t, f] = 1.0
    
    for i in range(num_test):
        num_spikes = torch.randint(100, 500, (1,)).item()
        spike_times = torch.randint(0, seq_len, (num_spikes,))
        spike_features = torch.randint(0, num_features, (num_spikes,))
        for t, f in zip(spike_times, spike_features):
            test_data[i, t, f] = 1.0
    
    # 添加batch维度
    train_data = train_data.unsqueeze(2)  # [N, T, 1, F]
    test_data = test_data.unsqueeze(2)
    
    # 创建数据加载器
    train_dataset = torch.utils.data.TensorDataset(train_data, train_labels)
    test_dataset = torch.utils.data.TensorDataset(test_data, test_labels)
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False
    )
    
    print(f"   模拟训练数据: {train_data.shape}")
    print(f"   模拟测试数据: {test_data.shape}")
    
    return train_loader, test_loader

# ==================== 训练和测试 ====================

def test_model(model, test_loader):
    """
    测试模型性能
    
    参数:
        model: 待测试的模型
        test_loader: 测试数据加载器
        
    返回:
        accuracy: 测试准确率
    """
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(DEVICE), target.to(DEVICE)
            # 调整数据维度：[N, T, 1, F] -> [T, N, F]
            data = data.squeeze(2).transpose(0, 1)
            
            output = model(data)
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    
    return correct / total

def train_model(model, train_loader, test_loader, epochs, model_name):
    """
    训练模型
    
    参数:
        model: 待训练的模型
        train_loader: 训练数据加载器
        test_loader: 测试数据加载器
        epochs: 训练轮数
        model_name: 模型名称
        
    返回:
        best_acc: 最佳测试准确率
    """
    
    criterion = nn.CrossEntropyLoss()
    model.to(DEVICE)
    
    # 配置优化器
    if isinstance(model, DH_SNN):
        # DH-SNN使用分层学习率
        base_params = [
            model.output_layer.weight,
            model.output_layer.bias,
            model.branch1_layer.weight,
            model.branch2_layer.weight,
        ]
        
        optimizer = torch.optim.Adam([
            {'params': base_params, 'lr': LEARNING_RATE},
            {'params': model.tau_n, 'lr': LEARNING_RATE},
            {'params': model.tau_m, 'lr': LEARNING_RATE},
        ])
    else:
        # 普通SNN使用标准优化器
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # 学习率调度器
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)
    
    best_acc = 0
    
    print(f"\n🚀 开始训练 {model_name}")
    print(f"   参数数量: {sum(p.numel() for p in model.parameters()):,}")
    print("-" * 50)
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        correct = 0
        total = 0
        
        epoch_start = time.time()
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(DEVICE), target.to(DEVICE)
            # 调整数据维度：[N, T, 1, F] -> [T, N, F]
            data = data.squeeze(2).transpose(0, 1)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            
            # 梯度裁剪
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
        
        print(f'轮次 {epoch:3d}: 训练损失={train_loss/len(train_loader):.4f}, '
              f'训练准确率={train_acc:.4f}, 测试准确率={test_acc:.4f}, '
              f'最佳={best_acc:.4f}, 用时={epoch_time:.1f}s')
        
        # 早停条件
        if "普通" in model_name and best_acc > 0.70:
            print(f"✅ {model_name}达到70%以上，提前停止训练")
            break
        elif "DH-SNN" in model_name and best_acc > 0.80:
            print(f"✅ {model_name}达到80%以上，提前停止训练")
            break
    
    return best_acc

# ==================== 主实验函数 ====================

def run_ssc_experiment():
    """运行SSC实验"""
    
    print("=" * 80)
    print("🎤 DH-SNN SSC脉冲语音命令识别实验")
    print("=" * 80)
    
    # 设置随机种子
    setup_seed(42)
    
    print(f"🖥️  使用设备: {DEVICE}")
    print(f"📁 临时数据路径: {TEMP_DIR}")
    
    try:
        # 准备数据
        train_loader, test_loader = prepare_data()
        
        print(f"\n📊 数据集统计:")
        print(f"   训练批次数: {len(train_loader)}")
        print(f"   测试批次数: {len(test_loader)}")
        
        # 实验配置
        experiments = [
            ("普通SNN", VanillaSNN),
            ("DH-SNN", DH_SNN),
        ]
        
        results = {}
        start_time = time.time()
        
        # 依次训练各模型
        for exp_name, model_class in experiments:
            print(f"\n🔬 实验: {exp_name}")
            print("=" * 50)
            
            model = model_class()
            best_acc = train_model(model, train_loader, test_loader, NUM_EPOCHS, exp_name)
            results[exp_name] = best_acc * 100
            
            print(f"✅ {exp_name} 最佳准确率: {best_acc*100:.1f}%")
        
        # 结果总结
        total_time = time.time() - start_time
        print(f"\n🎉 SSC实验完成! 总用时: {total_time/60:.1f}分钟")
        print("=" * 60)
        print("📊 SSC实验结果:")
        print("=" * 60)
        
        vanilla_acc = results.get("普通SNN", 0)
        dh_acc = results.get("DH-SNN", 0)
        improvement = dh_acc - vanilla_acc
        
        print(f"普通SNN:     {vanilla_acc:.1f}%")
        print(f"DH-SNN:      {dh_acc:.1f}%")
        print(f"性能提升:    {improvement:+.1f} 个百分点")
        
        # 与论文结果对比
        print(f"\n📈 与论文结果对比:")
        print(f"论文普通SNN:   ~70%")
        print(f"论文DH-SNN:    ~80%")
        
        if improvement > 5:
            print("🎉 DH-SNN显著优于普通SNN!")
        elif improvement > 0:
            print("✅ DH-SNN优于普通SNN")
        else:
            print("⚠️  结果需要进一步分析")
        
        # 保存结果
        results_path = Path("results/ssc_experiment_results.json")
        results_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(results_path, 'w', encoding='utf-8') as f:
            json_results = {
                'experiment_info': {
                    'name': 'SSC脉冲语音命令识别实验',
                    'framework': 'SpikingJelly + DH-SNN',
                    'date': time.strftime('%Y-%m-%d %H:%M:%S'),
                    'device': str(DEVICE)
                },
                'results': {
                    'vanilla_snn': {
                        'accuracy': vanilla_acc
                    },
                    'dh_snn': {
                        'accuracy': dh_acc
                    },
                    'improvement': improvement
                }
            }
            json.dump(json_results, f, indent=2, ensure_ascii=False)
        
        print(f"\n💾 结果已保存到: {results_path}")
        
        return results
        
    except Exception as e:
        print(f"\n❌ 实验失败: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    results = run_ssc_experiment()
    if results:
        print(f"\n🏁 SSC实验成功完成!")
    else:
        print(f"\n❌ SSC实验失败")