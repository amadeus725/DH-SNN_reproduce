#!/usr/bin/env python3
"""
SSC (Spiking Speech Commands) 实验
基于已验证的SHD实验代码
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
import time

# 添加路径
current_dir = Path(__file__).parent
project_root = current_dir.parent
sys.path.append(str(current_dir))
sys.path.append(str(project_root))

# 尝试导入core模块
try:
    from core.models import VanillaSFNN, DH_SFNN
    from core.neurons import ParametricLIF
    from core.surrogate import MultiGaussian
except ImportError:
    # 如果失败，尝试从spikingjelly_implementation导入
    sys.path.append(str(project_root / "spikingjelly_implementation"))
    from core.models import VanillaSFNN, DH_SFNN
    from core.neurons import ParametricLIF
    from core.surrogate import MultiGaussian

def load_ssc_data(data_path, num_train=None, num_test=None):
    """加载SSC数据集"""

    print(f"📚 加载SSC数据...")

    # 数据文件路径
    train_file = os.path.join(data_path, "ssc_train.h5.gz")
    test_file = os.path.join(data_path, "ssc_test.h5.gz")

    if not os.path.exists(train_file):
        raise FileNotFoundError(f"训练文件不存在: {train_file}")
    if not os.path.exists(test_file):
        raise FileNotFoundError(f"测试文件不存在: {test_file}")

    def read_compressed_h5(file_path, max_samples=None):
        """读取压缩的h5文件"""
        print(f"📖 直接读取压缩文件: {file_path}")

        with gzip.open(file_path, 'rb') as gz_file:
            with h5py.File(gz_file, 'r') as h5_file:
                # 获取数据
                spikes = h5_file['spikes'][:]
                labels = h5_file['labels'][:]

                print(f"  📊 总样本数: {len(spikes)}, 读取: {max_samples if max_samples else len(spikes)}")

                if max_samples:
                    spikes = spikes[:max_samples]
                    labels = labels[:max_samples]

                return spikes, labels

    # 读取训练数据
    train_spikes, train_labels = read_compressed_h5(train_file, num_train)

    # 读取测试数据
    test_spikes, test_labels = read_compressed_h5(test_file, num_test)

    print(f"🔄 转换为张量...")

    # 转换为张量
    def convert_to_tensor(spikes, labels, split_name):
        print(f"  处理{split_name}样本 1/{len(spikes)}")

        spike_tensors = []
        for i, spike_data in enumerate(spikes):
            if (i + 1) % 500 == 0 or i == 0:
                print(f"  处理{split_name}样本 {i+1}/{len(spikes)}")

            # 转换为张量
            spike_tensor = torch.FloatTensor(spike_data)
            spike_tensors.append(spike_tensor)

        # 堆叠所有样本
        all_spikes = torch.stack(spike_tensors)
        all_labels = torch.LongTensor(labels)

        return all_spikes, all_labels

    train_data, train_targets = convert_to_tensor(train_spikes, train_labels, "训练")
    test_data, test_targets = convert_to_tensor(test_spikes, test_labels, "测试")

    print(f"✅ SSC数据加载完成: 训练{train_data.shape}, 测试{test_data.shape}")

    return train_data, train_targets, test_data, test_targets

def create_ssc_model(model_type, input_size=700, hidden_size=200, output_size=35, device='cuda'):
    """创建SSC模型"""

    if model_type == 'vanilla':
        model = VanillaSFNN(
            input_size=input_size,
            hidden_size=hidden_size,
            output_size=output_size,
            v_threshold=1.0,
            device=device
        )
    elif model_type == 'dh_snn':
        model = DH_SFNN(
            input_size=input_size,
            hidden_size=hidden_size,
            output_size=output_size,
            num_branches=8,  # SSC使用8个分支
            v_threshold=1.0,
            device=device
        )
    else:
        raise ValueError(f"未知模型类型: {model_type}")

    return model

def train_ssc_model(model, train_data, train_targets, test_data, test_targets,
                   model_name, config, device='cuda'):
    """训练SSC模型"""

    print(f"🏋️ 训练 {model_name}")

    model = model.to(device)
    train_data = train_data.to(device)
    train_targets = train_targets.to(device)
    test_data = test_data.to(device)
    test_targets = test_targets.to(device)

    # 优化器和损失函数
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
    criterion = nn.CrossEntropyLoss()

    # 学习率调度器
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

    best_acc = 0.0
    training_history = []

    for epoch in range(config['epochs']):
        # 训练阶段
        model.train()
        total_loss = 0.0
        correct_train = 0
        total_train = 0

        # 分批训练
        batch_size = config['batch_size']
        num_batches = len(train_data) // batch_size

        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = start_idx + batch_size

            batch_data = train_data[start_idx:end_idx]
            batch_targets = train_targets[start_idx:end_idx]

            optimizer.zero_grad()
            outputs = model(batch_data)
            loss = criterion(outputs, batch_targets)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            # 计算训练准确率
            _, predicted = torch.max(outputs.data, 1)
            total_train += batch_targets.size(0)
            correct_train += (predicted == batch_targets).sum().item()

        # 测试阶段
        model.eval()
        correct_test = 0
        total_test = 0

        with torch.no_grad():
            # 分批测试
            test_batch_size = config['batch_size']
            test_num_batches = len(test_data) // test_batch_size

            for batch_idx in range(test_num_batches):
                start_idx = batch_idx * test_batch_size
                end_idx = start_idx + test_batch_size

                batch_data = test_data[start_idx:end_idx]
                batch_targets = test_targets[start_idx:end_idx]

                outputs = model(batch_data)
                _, predicted = torch.max(outputs.data, 1)
                total_test += batch_targets.size(0)
                correct_test += (predicted == batch_targets).sum().item()

        # 计算准确率
        train_acc = 100.0 * correct_train / total_train
        test_acc = 100.0 * correct_test / total_test
        avg_loss = total_loss / num_batches

        # 更新最佳准确率
        if test_acc > best_acc:
            best_acc = test_acc

        # 记录历史
        training_history.append({
            'epoch': epoch + 1,
            'train_acc': train_acc,
            'test_acc': test_acc,
            'loss': avg_loss,
            'best_acc': best_acc
        })

        # 学习率调度
        scheduler.step()

        # 打印进度
        if epoch % 10 == 0 or epoch == config['epochs'] - 1:
            print(f"  Epoch {epoch+1:3d}: 训练 {train_acc:5.1f}%, 测试 {test_acc:5.1f}%, 最佳 {best_acc:5.1f}%")

    return best_acc, training_history

def run_ssc_experiment():
    """运行SSC实验"""

    print("🚀 SSC (Spiking Speech Commands) 实验")
    print("="*60)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"🔧 使用设备: {device}")

    # 实验配置
    config = {
        'learning_rate': 1e-2,
        'batch_size': 200,
        'epochs': 100,
        'input_size': 700,
        'hidden_size': 200,
        'output_size': 35
    }

    # 数据路径
    data_path = "../datasets/ssc/data"

    try:
        # 加载数据 (使用较小的子集进行快速测试)
        train_data, train_targets, test_data, test_targets = load_ssc_data(
            data_path,
            num_train=2000,  # 训练样本数
            num_test=500     # 测试样本数
        )

        # 实验配置
        experiments = [
            ('Vanilla SFNN', 'vanilla'),
            ('DH-SFNN', 'dh_snn')
        ]

        results = {}

        for model_name, model_type in experiments:
            print(f"\n📊 开始实验: {model_name}")

            # 创建模型
            model = create_ssc_model(
                model_type=model_type,
                input_size=config['input_size'],
                hidden_size=config['hidden_size'],
                output_size=config['output_size'],
                device=device
            )

            print(f"🏗️ 模型参数数量: {sum(p.numel() for p in model.parameters())}")

            # 训练模型
            best_acc, history = train_ssc_model(
                model, train_data, train_targets, test_data, test_targets,
                model_name, config, device
            )

            results[model_type] = {
                'model_name': model_name,
                'best_accuracy': best_acc,
                'final_accuracy': history[-1]['test_acc'],
                'training_history': history,
                'config': config
            }

            print(f"📈 {model_name} 最佳准确率: {best_acc:.1f}%")

        # 显示结果对比
        print(f"\n🎉 SSC实验完成!")
        print("="*50)

        vanilla_acc = results['vanilla']['best_accuracy']
        dh_acc = results['dh_snn']['best_accuracy']
        improvement = dh_acc - vanilla_acc

        print(f"📊 结果对比:")
        print(f"  Vanilla SFNN: {vanilla_acc:5.1f}%")
        print(f"  DH-SFNN:      {dh_acc:5.1f}%")
        print(f"  性能提升:     +{improvement:4.1f} 个百分点")

        # 保存结果
        os.makedirs("outputs/results", exist_ok=True)
        result_file = "outputs/results/ssc_experiment_results.pth"

        torch.save(results, result_file)
        print(f"\n💾 结果已保存到: {result_file}")

        return results

    except Exception as e:
        print(f"\n❌ SSC实验失败: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    """主函数"""

    try:
        results = run_ssc_experiment()

        if results:
            print(f"\n🏁 SSC实验成功完成!")
        else:
            print(f"\n❌ SSC实验失败")

    except KeyboardInterrupt:
        print(f"\n⏹️ 实验被用户中断")
    except Exception as e:
        print(f"\n❌ 实验异常: {e}")

if __name__ == '__main__':
    main()
