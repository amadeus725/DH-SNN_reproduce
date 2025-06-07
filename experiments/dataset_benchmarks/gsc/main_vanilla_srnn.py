#!/usr/bin/env python3
"""
GSC - VANILLA_SRNN 实验
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import sys
from pathlib import Path

# 添加路径
sys.path.append(str(Path(__file__).parent.parent.parent))

from dh_snn.core.models import *
from dh_snn.core.neurons import *
from dh_snn.core.utils import *
from config import *
from data_loader import load_gsc_data

def create_model(config):
    """创建模型"""
    
    if 'vanilla_srnn' == 'vanilla_sfnn':
        model = VanillaSFNN(
            input_size=config['input_size'],
            hidden_size=config['hidden_size'], 
            output_size=config['output_size'],
            v_threshold=config['v_threshold'],
            device=config['device']
        )
    elif 'vanilla_srnn' == 'dh_sfnn':
        model = DH_SFNN(
            input_size=config['input_size'],
            hidden_size=config['hidden_size'],
            output_size=config['output_size'],
            num_branches=DH_CONFIG['num_branches'],
            v_threshold=config['v_threshold'],
            device=config['device']
        )
    elif 'vanilla_srnn' == 'vanilla_srnn':
        model = VanillaSRNN(
            input_size=config['input_size'],
            hidden_size=config['hidden_size'],
            output_size=config['output_size'],
            v_threshold=config['v_threshold'],
            device=config['device']
        )
    elif 'vanilla_srnn' == 'dh_srnn':
        model = DH_SRNN(
            input_size=config['input_size'],
            hidden_size=config['hidden_size'],
            output_size=config['output_size'],
            num_branches=DH_CONFIG['num_branches'],
            v_threshold=config['v_threshold'],
            device=config['device']
        )
    else:
        raise ValueError(f"Unknown model type: vanilla_srnn")
    
    return model

def train_model(model, train_loader, test_loader, config):
    """训练模型"""
    
    device = config['device']
    model = model.to(device)
    
    # 优化器
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
    criterion = nn.CrossEntropyLoss()
    
    best_acc = 0.0
    
    for epoch in range(config['epochs']):
        # 训练
        model.train()
        train_loss = 0.0
        train_acc = 0.0
        
        for batch_idx, (data, targets) in enumerate(train_loader):
            data, targets = data.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            pred = outputs.argmax(dim=1)
            train_acc += (pred == targets).float().mean().item()
        
        # 测试
        model.eval()
        test_acc = 0.0
        with torch.no_grad():
            for data, targets in test_loader:
                data, targets = data.to(device), targets.to(device)
                outputs = model(data)
                pred = outputs.argmax(dim=1)
                test_acc += (pred == targets).float().mean().item()
        
        train_loss /= len(train_loader)
        train_acc /= len(train_loader)
        test_acc /= len(test_loader)
        
        if test_acc > best_acc:
            best_acc = test_acc
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch:3d}: Loss={train_loss:.4f}, Train={train_acc:.3f}, Test={test_acc:.3f}, Best={best_acc:.3f}")
    
    return best_acc

def main():
    """主函数"""
    
    print(f"🚀 GSC - VANILLA_SRNN 实验")
    print("="*60)
    
    # 合并配置
    config = {**BASE_CONFIG, **NETWORK_CONFIG, **TRAINING_CONFIG}
    
    print(f"📱 设备: {config['device']}")
    print(f"🏗️ 模型: VANILLA_SRNN")
    
    # 加载数据
    train_loader, test_loader = load_gsc_data(
        DATA_CONFIG['data_path'],
        config['batch_size'],
        config['num_workers']
    )
    
    # 创建模型
    model = create_model(config)
    print(f"📊 参数数量: {sum(p.numel() for p in model.parameters())}")
    
    # 训练
    best_acc = train_model(model, train_loader, test_loader, config)
    
    print(f"\n🎉 训练完成!")
    print(f"📈 最佳准确率: {best_acc:.3f}")
    
    # 保存结果
    os.makedirs(EXPERIMENT_CONFIG['output_dir'], exist_ok=True)
    results = {
        'model_type': 'vanilla_srnn',
        'best_accuracy': best_acc,
        'config': config
    }
    
    torch.save(results, f"{EXPERIMENT_CONFIG['output_dir']}/vanilla_srnn_results.pth")
    
    return best_acc

if __name__ == '__main__':
    main()
