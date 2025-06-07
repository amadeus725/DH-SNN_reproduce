#!/usr/bin/env python3
"""
SSC实验 - 基于已验证的SHD代码
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import h5py
import gzip
import os
import sys
from pathlib import Path

# 使用已验证的core模块
sys.path.append("experiments/figure_reproduction/figure3_delayed_xor")
from core.models import VanillaSFNN, DH_SFNN

def load_ssc_data(data_path, num_train=2000, num_test=500):
    """加载SSC数据"""
    
    print(f"📚 加载SSC数据: 训练{num_train}, 测试{num_test}")
    
    train_file = os.path.join(data_path, "ssc_train.h5.gz")
    test_file = os.path.join(data_path, "ssc_test.h5.gz")
    
    def read_h5_gz(file_path, max_samples):
        print(f"📖 读取: {file_path}")
        with gzip.open(file_path, 'rb') as gz_file:
            with h5py.File(gz_file, 'r') as h5_file:
                spikes = h5_file['spikes'][:max_samples]
                labels = h5_file['labels'][:max_samples]
                print(f"  📊 读取样本: {len(spikes)}")
                return spikes, labels
    
    # 读取数据
    train_spikes, train_labels = read_h5_gz(train_file, num_train)
    test_spikes, test_labels = read_h5_gz(test_file, num_test)
    
    # 转换为张量
    print("🔄 转换为张量...")
    train_data = torch.FloatTensor(train_spikes)
    train_targets = torch.LongTensor(train_labels)
    test_data = torch.FloatTensor(test_spikes)
    test_targets = torch.LongTensor(test_labels)
    
    print(f"✅ 数据加载完成: 训练{train_data.shape}, 测试{test_data.shape}")
    return train_data, train_targets, test_data, test_targets

def train_model(model, train_data, train_targets, test_data, test_targets, model_name, device):
    """训练模型"""
    
    print(f"🏋️ 训练 {model_name}")
    
    model = model.to(device)
    train_data = train_data.to(device)
    train_targets = train_targets.to(device)
    test_data = test_data.to(device)
    test_targets = test_targets.to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=1e-2)
    criterion = nn.CrossEntropyLoss()
    
    best_acc = 0.0
    batch_size = 100
    
    for epoch in range(50):  # 减少epoch数进行快速测试
        # 训练
        model.train()
        total_loss = 0.0
        correct_train = 0
        total_train = 0
        
        for i in range(0, len(train_data), batch_size):
            batch_data = train_data[i:i+batch_size]
            batch_targets = train_targets[i:i+batch_size]
            
            optimizer.zero_grad()
            outputs = model(batch_data)
            loss = criterion(outputs, batch_targets)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_train += batch_targets.size(0)
            correct_train += (predicted == batch_targets).sum().item()
        
        # 测试
        model.eval()
        correct_test = 0
        total_test = 0
        
        with torch.no_grad():
            for i in range(0, len(test_data), batch_size):
                batch_data = test_data[i:i+batch_size]
                batch_targets = test_targets[i:i+batch_size]
                
                outputs = model(batch_data)
                _, predicted = torch.max(outputs.data, 1)
                total_test += batch_targets.size(0)
                correct_test += (predicted == batch_targets).sum().item()
        
        train_acc = 100.0 * correct_train / total_train
        test_acc = 100.0 * correct_test / total_test
        
        if test_acc > best_acc:
            best_acc = test_acc
        
        if epoch % 10 == 0 or epoch == 49:
            print(f"  Epoch {epoch+1:2d}: 训练 {train_acc:5.1f}%, 测试 {test_acc:5.1f}%, 最佳 {best_acc:5.1f}%")
    
    return best_acc

def main():
    """主函数"""
    
    print("🚀 SSC (Spiking Speech Commands) 实验")
    print("="*60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"🔧 使用设备: {device}")
    
    try:
        # 加载数据
        data_path = "../datasets/ssc/data"
        train_data, train_targets, test_data, test_targets = load_ssc_data(data_path)
        
        # 网络配置 (SSC配置)
        input_size = 700
        hidden_size = 200
        output_size = 35
        
        results = {}
        
        # 实验1: Vanilla SFNN
        print(f"\n📊 实验1: Vanilla SFNN")
        vanilla_model = VanillaSFNN(
            input_size=input_size,
            hidden_size=hidden_size,
            output_size=output_size,
            v_threshold=1.0,
            device=device
        )
        
        vanilla_acc = train_model(
            vanilla_model, train_data, train_targets, test_data, test_targets,
            "Vanilla SFNN", device
        )
        results['vanilla'] = vanilla_acc
        
        # 实验2: DH-SFNN
        print(f"\n📊 实验2: DH-SFNN")
        dh_model = DH_SFNN(
            input_size=input_size,
            hidden_size=hidden_size,
            output_size=output_size,
            num_branches=8,
            v_threshold=1.0,
            device=device
        )
        
        dh_acc = train_model(
            dh_model, train_data, train_targets, test_data, test_targets,
            "DH-SFNN", device
        )
        results['dh_snn'] = dh_acc
        
        # 显示结果
        print(f"\n🎉 SSC实验完成!")
        print("="*50)
        
        improvement = dh_acc - vanilla_acc
        
        print(f"📊 结果对比:")
        print(f"  Vanilla SFNN: {vanilla_acc:5.1f}%")
        print(f"  DH-SFNN:      {dh_acc:5.1f}%")
        print(f"  性能提升:     +{improvement:4.1f} 个百分点")
        
        # 保存结果
        os.makedirs("outputs/results", exist_ok=True)
        torch.save(results, "outputs/results/ssc_simple_results.pth")
        print(f"\n💾 结果已保存")
        
        return results
        
    except Exception as e:
        print(f"\n❌ SSC实验失败: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == '__main__':
    main()
