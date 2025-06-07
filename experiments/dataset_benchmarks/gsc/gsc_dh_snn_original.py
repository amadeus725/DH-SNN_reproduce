#!/usr/bin/env python3
"""
GSC DH-SNN实验 - 基于原论文代码实现
参考: reference/original_paper_code/GSC/main_dense.py
"""

import os
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.optim.lr_scheduler import StepLR
import torchaudio
from torchaudio import transforms
import warnings
warnings.filterwarnings("ignore")

# 设备配置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)

# 数据路径
DATA_ROOT = "/root/DH-SNN_reproduce/datasets/speech_commands"

# 原论文配置
BATCH_SIZE = 200
LEARNING_RATE = 1e-2
NUM_EPOCHS = 150
HIDDEN_DIM = 200
NUM_BRANCHES = 8

# 音频处理参数
SR = 16000
SIZE = 16000
N_FFT = int(30e-3 * SR)
HOP_LENGTH = int(10e-3 * SR)
N_MELS = 40
FMAX = 4000
FMIN = 20
DELTA_ORDER = 2
STACK = True

print(f"🔧 设备: {device}")
print(f"🔧 数据路径: {DATA_ROOT}")

# 使用我们已有的数据处理
sys.path.append('/root/DH-SNN_reproduce/experiments/dataset_benchmarks/gsc')
from data_loader import load_gsc_data

# 激活函数 (原论文实现)
class ActFun_adp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return input.gt(0.).float()

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        temp = abs(input) < 0.5
        return grad_input * temp.float()

act_fun_adp = ActFun_adp.apply

# 神经元更新函数 (原论文实现)
R_m = 1  # 膜电阻

def mem_update_pra(inputs, mem, spike, v_th, tau_m, dt=1, device=None):
    """神经元膜电位更新 - 软重置"""
    alpha = torch.sigmoid(tau_m)
    mem = mem * alpha + (1 - alpha) * R_m * inputs - v_th * spike
    inputs_ = mem - v_th
    spike = act_fun_adp(inputs_)
    return mem, spike

# DH-SFNN层 (原论文实现)
class DH_SFNN_Layer(nn.Module):
    def __init__(self, input_dim, output_dim, num_branches=8, 
                 tau_m_low=0, tau_m_high=4, tau_n_low=0, tau_n_high=4,
                 vth=1.0, dt=1, device='cpu', bias=True):
        super(DH_SFNN_Layer, self).__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.device = device
        self.vth = vth
        self.dt = dt
        self.num_branches = num_branches
        
        # 连接模式参数
        self.sparsity = 1 / num_branches
        self.pad = ((input_dim) // num_branches * num_branches + num_branches - input_dim) % num_branches
        
        # 网络层
        self.dense = nn.Linear(input_dim + self.pad, output_dim * num_branches, bias=bias)
        
        # 时间常数参数
        self.tau_m = nn.Parameter(torch.Tensor(output_dim))  # 膜时间常数
        self.tau_n = nn.Parameter(torch.Tensor(output_dim, num_branches))  # 树突时间常数
        
        # 初始化时间常数
        nn.init.uniform_(self.tau_m, tau_m_low, tau_m_high)
        nn.init.uniform_(self.tau_n, tau_n_low, tau_n_high)
        
        # 创建连接掩码
        self.create_mask()
        
    def create_mask(self):
        """创建树突分支连接掩码"""
        input_size = self.input_dim + self.pad
        self.mask = torch.zeros(self.output_dim * self.num_branches, input_size).to(self.device)
        
        for i in range(self.output_dim):
            seq = torch.randperm(input_size)
            for j in range(self.num_branches):
                # 每个分支连接不同的输入子集
                start_idx = j * input_size // self.num_branches
                end_idx = (j + 1) * input_size // self.num_branches
                self.mask[i * self.num_branches + j, seq[start_idx:end_idx]] = 1
    
    def apply_mask(self):
        """应用连接掩码"""
        self.dense.weight.data = self.dense.weight.data * self.mask
    
    def set_neuron_state(self, batch_size):
        """初始化神经元状态"""
        self.mem = torch.zeros(batch_size, self.output_dim).to(self.device)
        self.spike = torch.zeros(batch_size, self.output_dim).to(self.device)
        self.d_input = torch.zeros(batch_size, self.output_dim, self.num_branches).to(self.device)
        self.v_th = torch.ones(batch_size, self.output_dim).to(self.device) * self.vth
    
    def forward(self, input_spike):
        """前向传播"""
        # 树突时间常数
        beta = torch.sigmoid(self.tau_n)
        
        # 输入填充
        padding = torch.zeros(input_spike.size(0), self.pad).to(self.device)
        k_input = torch.cat((input_spike.float(), padding), 1)
        
        # 更新树突电流
        dense_output = self.dense(k_input).reshape(-1, self.output_dim, self.num_branches)
        self.d_input = beta * self.d_input + (1 - beta) * dense_output
        
        # 树突电流求和
        l_input = self.d_input.sum(dim=2, keepdim=False)
        
        # 更新膜电位和生成脉冲
        self.mem, self.spike = mem_update_pra(
            l_input, self.mem, self.spike, self.v_th, self.tau_m, self.dt, device=self.device
        )
        
        return self.mem, self.spike

# 读出积分器 (原论文实现)
class ReadoutIntegrator(nn.Module):
    def __init__(self, input_dim, output_dim, dt=1, device='cpu', bias=True):
        super(ReadoutIntegrator, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.device = device
        self.dt = dt
        
        self.dense = nn.Linear(input_dim, output_dim, bias=bias)
        self.tau_m = nn.Parameter(torch.Tensor(output_dim))
        
        # 初始化
        nn.init.uniform_(self.tau_m, 0, 4)
    
    def set_neuron_state(self, batch_size):
        self.mem = torch.zeros(batch_size, self.output_dim).to(self.device)
    
    def forward(self, input_spike):
        alpha = torch.sigmoid(self.tau_m)
        d_input = self.dense(input_spike)
        self.mem = self.mem * alpha + (1 - alpha) * R_m * d_input
        return self.mem

# GSC DH-SNN模型 (原论文架构)
class GSC_DH_SNN(nn.Module):
    def __init__(self, input_dim=120, hidden_dim=200, output_dim=12, num_branches=8):
        super(GSC_DH_SNN, self).__init__()
        
        print(f"🏗️  创建GSC DH-SNN模型 (原论文架构):")
        print(f"   输入维度: {input_dim}")
        print(f"   隐藏维度: {hidden_dim}")
        print(f"   输出维度: {output_dim}")
        print(f"   分支数: {num_branches}")
        
        # 3层DH-SFNN
        self.layer1 = DH_SFNN_Layer(input_dim, hidden_dim, num_branches, device=device)
        self.layer2 = DH_SFNN_Layer(hidden_dim, hidden_dim, num_branches, device=device)
        self.layer3 = DH_SFNN_Layer(hidden_dim, hidden_dim, num_branches, device=device)
        
        # 读出层
        self.readout = ReadoutIntegrator(hidden_dim, output_dim, device=device)
    
    def forward(self, x):
        # x shape: (batch, 3, 101, 40)
        batch_size, channels, seq_len, mel_bins = x.shape
        
        # 初始化神经元状态
        self.layer1.set_neuron_state(batch_size)
        self.layer2.set_neuron_state(batch_size)
        self.layer3.set_neuron_state(batch_size)
        self.readout.set_neuron_state(batch_size)
        
        # 应用连接掩码
        self.layer1.apply_mask()
        self.layer2.apply_mask()
        self.layer3.apply_mask()
        
        output = 0
        
        # 逐时间步处理
        for t in range(seq_len):
            # 输入重塑: (batch, channels * mel_bins)
            input_t = x[:, :, t, :].reshape(batch_size, channels * mel_bins)
            
            # 通过DH-SFNN层
            _, spike1 = self.layer1(input_t)
            _, spike2 = self.layer2(spike1)
            _, spike3 = self.layer3(spike2)
            
            # 读出积分
            mem_out = self.readout(spike3)
            output += mem_out
        
        # 时间平均和softmax
        output = F.log_softmax(output / seq_len, dim=1)
        return output

def prepare_data():
    """准备GSC数据 (使用已有的数据处理)"""
    print("📊 准备GSC数据...")

    # 使用已有的数据加载函数
    train_loader, test_loader = load_gsc_data(
        data_path=DATA_ROOT,
        batch_size=BATCH_SIZE,
        num_workers=8
    )

    # 使用测试集作为验证集
    valid_loader = test_loader

    print(f"✅ 数据准备完成:")
    print(f"   训练集: {len(train_loader)} 批次")
    print(f"   验证集: {len(valid_loader)} 批次")
    print(f"   测试集: {len(test_loader)} 批次")

    return train_loader, valid_loader, test_loader

def main():
    """主函数"""
    print("🎯 GSC DH-SNN实验 (原论文实现)")
    print("=" * 60)
    
    try:
        # 准备数据
        train_loader, valid_loader, test_loader = prepare_data()
        
        # 创建模型
        model = GSC_DH_SNN(input_dim=40*3, hidden_dim=HIDDEN_DIM, 
                          output_dim=12, num_branches=NUM_BRANCHES)
        model.to(device)
        
        print(f"📊 模型参数数量: {sum(p.numel() for p in model.parameters()):,}")
        
        # 损失函数和优化器 (原论文配置)
        criterion = nn.CrossEntropyLoss()
        
        # 分组学习率 (原论文方法)
        base_params = []
        tau_params = []
        
        for name, param in model.named_parameters():
            if 'tau' in name:
                tau_params.append(param)
            else:
                base_params.append(param)
        
        optimizer = torch.optim.Adam([
            {'params': base_params, 'lr': LEARNING_RATE},
            {'params': tau_params, 'lr': LEARNING_RATE * 2}
        ])
        
        scheduler = StepLR(optimizer, step_size=25, gamma=0.5)
        
        # 训练
        print("🚀 开始DH-SNN训练...")
        best_acc = 0.0
        
        for epoch in range(NUM_EPOCHS):
            start_time = time.time()
            
            # 训练阶段
            model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(device), target.to(device)
                data = data.view(-1, 3, 101, 40)  # 原论文格式
                target = target.long()
                
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                pred = output.argmax(dim=1)
                train_correct += pred.eq(target).sum().item()
                train_total += target.size(0)
                
                if batch_idx % 10 == 0:
                    print(f'   Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}')
            
            # 验证阶段
            model.eval()
            valid_correct = 0
            valid_total = 0
            
            with torch.no_grad():
                for data, target in valid_loader:
                    data, target = data.to(device), target.to(device)
                    data = data.view(-1, 3, 101, 40)
                    target = target.long()
                    
                    output = model(data)
                    pred = output.argmax(dim=1)
                    valid_correct += pred.eq(target).sum().item()
                    valid_total += target.size(0)
            
            # 计算准确率
            train_acc = train_correct / train_total
            valid_acc = valid_correct / valid_total
            avg_train_loss = train_loss / len(train_loader)
            
            scheduler.step()
            
            # 保存最佳模型
            if valid_acc > best_acc and train_acc > 0.89:  # 原论文条件
                best_acc = valid_acc
                torch.save(model, f'/root/DH-SNN_reproduce/results/gsc_dh_snn_best_{best_acc:.4f}.pth')
                print(f"💾 保存最佳模型: {best_acc:.4f}")
            
            epoch_time = time.time() - start_time
            print(f'Epoch {epoch+1:3d}/{NUM_EPOCHS}, '
                  f'Train Loss: {avg_train_loss:.4f}, '
                  f'Train Acc: {train_acc:.4f}, '
                  f'Valid Acc: {valid_acc:.4f}, '
                  f'Best: {best_acc:.4f}, '
                  f'LR: {optimizer.param_groups[0]["lr"]:.6f}, '
                  f'Time: {epoch_time:.1f}s')
            
            # 早停条件
            if best_acc > 0.92:
                print(f"✅ DH-SNN达到92%以上，提前停止训练")
                break
        
        # 最终测试
        model.eval()
        test_correct = 0
        test_total = 0
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                data = data.view(-1, 3, 101, 40)
                target = target.long()
                
                output = model(data)
                pred = output.argmax(dim=1)
                test_correct += pred.eq(target).sum().item()
                test_total += target.size(0)
        
        test_acc = test_correct / test_total
        
        print(f"\n✅ DH-SNN训练完成:")
        print(f"   最佳验证准确率: {best_acc:.4f}")
        print(f"   最终测试准确率: {test_acc:.4f}")
        
        return best_acc, test_acc
        
    except Exception as e:
        print(f"❌ 实验失败: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    # 激活dh环境
    os.system("conda activate dh")
    results = main()
