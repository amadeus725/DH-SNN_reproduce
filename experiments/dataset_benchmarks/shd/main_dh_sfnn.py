#!/usr/bin/env python3
"""
SHD - DH_SFNN 实验
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import sys
from pathlib import Path

# 添加路径
current_file = Path(__file__).resolve()
project_root = current_file.parent.parent.parent.parent
sys.path.insert(0, str(project_root))

# 验证路径
print(f"📁 当前文件: {current_file}")
print(f"📁 项目根目录: {project_root}")
print(f"📁 dh_snn路径: {project_root / 'dh_snn'}")
print(f"📁 dh_snn存在: {(project_root / 'dh_snn').exists()}")

# 导入配置
sys.path.append(str(project_root / 'experiments' / 'configs' / 'datasets'))
from shd_config import *

from dh_snn.core.models import *
from dh_snn.core.neurons import *
from dh_snn.core.utils import *
from data_loader import load_shd_data

def create_model(config):
    """创建模型"""

    # 直接创建DH_SFNN模型
    model = DH_SFNN(
        input_dim=config['input_size'],
        hidden_dims=[config['hidden_size']],
        output_dim=config['output_size'],
        num_branches=DH_CONFIG['num_branches'],
        v_threshold=config['v_threshold']
    )

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

            # 重置模型状态
            model.reset_states()

            optimizer.zero_grad()

            # 处理时间序列数据: (batch, time, features) -> (time, batch, features)
            data = data.transpose(0, 1)

            # 逐时间步处理
            outputs = []
            for t in range(data.size(0)):
                output = model(data[t])
                outputs.append(output)

            # 取最后一个时间步的输出
            final_output = outputs[-1]

            loss = criterion(final_output, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            pred = final_output.argmax(dim=1)
            train_acc += (pred == targets).float().mean().item()

        # 测试
        model.eval()
        test_acc = 0.0
        with torch.no_grad():
            for data, targets in test_loader:
                data, targets = data.to(device), targets.to(device)

                # 重置模型状态
                model.reset_states()

                # 处理时间序列数据: (batch, time, features) -> (time, batch, features)
                data = data.transpose(0, 1)

                # 逐时间步处理
                outputs = []
                for t in range(data.size(0)):
                    output = model(data[t])
                    outputs.append(output)

                # 取最后一个时间步的输出
                final_output = outputs[-1]

                pred = final_output.argmax(dim=1)
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

    print(f"🚀 SHD - DH_SFNN 实验")
    print("="*60)

    # 合并配置
    config = {**BASE_CONFIG, **NETWORK_CONFIG, **TRAINING_CONFIG}

    print(f"📱 设备: {config['device']}")
    print(f"🏗️ 模型: DH_SFNN")

    # 加载数据
    train_loader, test_loader = load_shd_data(
        DATA_CONFIG['data_path'],
        config['batch_size'],
        config['num_workers'],
        max_samples=DATA_CONFIG['train_samples']
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
        'model_type': 'dh_sfnn',
        'best_accuracy': best_acc,
        'config': config
    }

    torch.save(results, f"{EXPERIMENT_CONFIG['output_dir']}/dh_sfnn_results.pth")

    return best_acc

if __name__ == '__main__':
    main()
