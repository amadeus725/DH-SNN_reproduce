#!/usr/bin/env python3
"""
简单的DH-SRNN训练 - 避免复杂的数据加载
"""

import os
import sys
import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm

# 添加项目路径
sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))

class SimpleDHSRNNCell(nn.Module):
    """简化的DH-SRNN单元"""
    
    def __init__(self, input_size, hidden_size, num_branches=2):
        super(SimpleDHSRNNCell, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_branches = num_branches
        
        # 简单的分支结构
        total_input_size = input_size + hidden_size
        self.branch_fc = nn.Linear(total_input_size, hidden_size * num_branches)
        self.output_fc = nn.Linear(hidden_size * num_branches, hidden_size)
        
        # 固定时间常数
        self.register_buffer('alphas', torch.tensor([0.3, 0.7]))
        
    def forward(self, input_t, hidden_state=None, branch_states=None):
        batch_size = input_t.size(0)
        device = input_t.device
        
        if hidden_state is None:
            hidden_state = torch.zeros(batch_size, self.hidden_size, device=device)
        
        if branch_states is None:
            branch_states = torch.zeros(batch_size, self.hidden_size, self.num_branches, device=device)
        
        # 拼接输入
        combined_input = torch.cat([input_t, hidden_state], dim=1)
        
        # 分支计算
        branch_outputs = self.branch_fc(combined_input)
        branch_outputs = branch_outputs.view(batch_size, self.hidden_size, self.num_branches)
        
        # 状态更新
        alphas = self.alphas.view(1, 1, -1)
        new_branch_states = alphas * branch_states + (1 - alphas) * branch_outputs
        
        # 输出
        combined = new_branch_states.view(batch_size, -1)
        output = self.output_fc(combined)
        output = torch.tanh(output)
        
        return output, new_branch_states

class SimpleDHSRNNModel(nn.Module):
    """简化的DH-SRNN模型"""
    
    def __init__(self, num_branches=2, subsample=8):
        super(SimpleDHSRNNModel, self).__init__()
        
        self.subsample = subsample
        self.rnn1 = SimpleDHSRNNCell(1, 32, num_branches)
        self.rnn2 = SimpleDHSRNNCell(32, 64, num_branches)
        self.classifier = nn.Linear(64, 10)
        
    def forward(self, x):
        batch_size, seq_len, input_dim = x.shape
        device = x.device
        
        # 子采样
        x = x[:, ::self.subsample, :]
        seq_len = x.shape[1]
        
        # 初始化
        h1_state = None
        h2_state = None
        h1_branches = None
        h2_branches = None
        
        # 处理序列
        for t in range(seq_len):
            x_t = x[:, t, :]
            h1_state, h1_branches = self.rnn1(x_t, h1_state, h1_branches)
            h2_state, h2_branches = self.rnn2(h1_state, h2_state, h2_branches)
            
            # 每20步分离一次梯度
            if t % 20 == 0 and t > 0:
                h1_state = h1_state.detach()
                h1_branches = h1_branches.detach()
                h2_state = h2_state.detach()
                h2_branches = h2_branches.detach()
        
        # 分类
        output = self.classifier(h2_state)
        return output

def load_simple_data():
    """简单的数据加载"""
    print("📊 加载MNIST数据...")
    
    try:
        from torchvision import datasets, transforms
        from torch.utils.data import DataLoader
        
        # 简单的变换
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        
        # 加载数据
        train_dataset = datasets.MNIST('./mnist_data', train=True, download=True, transform=transform)
        test_dataset = datasets.MNIST('./mnist_data', train=False, transform=transform)
        
        # 创建DataLoader (不使用多进程)
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=0)
        test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=0)
        
        print(f"   训练集: {len(train_dataset)} 样本")
        print(f"   测试集: {len(test_dataset)} 样本")
        
        return train_loader, test_loader
        
    except Exception as e:
        print(f"❌ 数据加载失败: {e}")
        return None, None

def convert_to_sequential(data):
    """将MNIST图像转换为序列"""
    # data: [batch, 1, 28, 28]
    batch_size = data.size(0)
    # 展平为序列: [batch, 784, 1]
    sequential_data = data.view(batch_size, 784, 1)
    return sequential_data

def train_simple_model():
    """训练简单的DH-SRNN"""
    print("🚀 简单DH-SRNN训练")
    print("=" * 50)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🔧 使用设备: {device}")
    
    # 加载数据
    train_loader, test_loader = load_simple_data()
    if train_loader is None:
        print("❌ 数据加载失败，退出")
        return False
    
    # 创建模型
    print("\n🧠 创建模型...")
    model = SimpleDHSRNNModel(num_branches=2, subsample=8).to(device)
    print(f"📈 模型参数: {sum(p.numel() for p in model.parameters()):,}")
    
    # 训练配置
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=2e-3)
    
    num_epochs = 10
    best_acc = 0.0
    
    print(f"\n🚀 开始训练 ({num_epochs} epochs)...")
    
    for epoch in range(1, num_epochs + 1):
        # 训练
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        epoch_start = time.time()
        
        print(f"\nEpoch {epoch}/{num_epochs}:")
        pbar = tqdm(train_loader, desc='Training')
        
        for batch_idx, (data, target) in enumerate(pbar):
            data, target = data.to(device), target.to(device)
            
            # 转换为序列
            sequential_data = convert_to_sequential(data)
            
            optimizer.zero_grad()
            output = model(sequential_data)
            loss = criterion(output, target)
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            # 统计
            train_loss += loss.item()
            pred = output.argmax(dim=1)
            train_correct += pred.eq(target).sum().item()
            train_total += target.size(0)
            
            # 更新进度条
            if batch_idx % 50 == 0:
                current_acc = 100. * train_correct / train_total
                pbar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'Acc': f'{current_acc:.1f}%'
                })
        
        epoch_time = time.time() - epoch_start
        train_acc = 100. * train_correct / train_total
        
        # 测试
        model.eval()
        test_correct = 0
        test_total = 0
        
        print("Testing...")
        with torch.no_grad():
            for data, target in tqdm(test_loader, desc='Testing'):
                data, target = data.to(device), target.to(device)
                sequential_data = convert_to_sequential(data)
                output = model(sequential_data)
                pred = output.argmax(dim=1)
                test_correct += pred.eq(target).sum().item()
                test_total += target.size(0)
        
        test_acc = 100. * test_correct / test_total
        
        # 保存最佳模型
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save({
                'model_state_dict': model.state_dict(),
                'epoch': epoch,
                'best_acc': best_acc,
            }, 'results/S-MNIST_simple_dh_srnn_best.pth')
        
        # 输出结果
        print(f"Epoch {epoch}: Train Acc={train_acc:.2f}%, Test Acc={test_acc:.2f}%, "
              f"Best={best_acc:.2f}%, Time={epoch_time:.1f}s")
        
        # 早停
        if test_acc > 70.0:
            print(f"✅ 达到70%准确率!")
            break
    
    # 保存结果
    results = {
        'model_type': 'simple_dh_srnn',
        'dataset': 'S-MNIST',
        'best_test_accuracy': best_acc,
        'epochs': epoch,
        'config': {
            'num_branches': 2,
            'subsample': 8,
            'model_params': sum(p.numel() for p in model.parameters())
        }
    }
    
    torch.save(results, 'results/S-MNIST_simple_dh_srnn_results.pth')
    
    print(f"\n📋 训练总结:")
    print(f"最佳准确率: {best_acc:.2f}%")
    print(f"训练轮数: {epoch}")
    
    # 与原始结果对比
    print(f"\n📊 对比:")
    print(f"原始DH-SRNN: 10.1% (失败)")
    print(f"简单DH-SRNN: {best_acc:.2f}% ({'成功' if best_acc > 50 else '需改进'})")
    
    return best_acc > 50.0

def main():
    """主函数"""
    print("开始简单DH-SRNN训练测试...")
    success = train_simple_model()
    
    if success:
        print(f"\n🎉 DH-SRNN训练成功!")
    else:
        print(f"\n⚠️  需要进一步调试")
    
    return success

if __name__ == "__main__":
    main()
