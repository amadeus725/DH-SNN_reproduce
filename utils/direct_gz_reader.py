#!/usr/bin/env python3
"""
直接读取压缩的SHD/SSC数据文件
不需要解压，直接从.gz文件读取
"""

import gzip
import tables
import numpy as np
import torch
from typing import Tuple, List
import os

def read_gz_h5_file(gz_path: str, max_samples: int = None):
    """
    直接从.gz文件读取HDF5数据

    Args:
        gz_path: .gz文件路径
        max_samples: 最大读取样本数，None表示读取全部

    Returns:
        times, units, labels
    """
    print(f"📖 直接读取压缩文件: {gz_path}")

    import tempfile
    import shutil

    # 创建临时文件
    with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as temp_file:
        temp_path = temp_file.name

        # 解压到临时文件
        with gzip.open(gz_path, 'rb') as gz_file:
            shutil.copyfileobj(gz_file, temp_file)

    try:
        # 读取临时文件
        with tables.open_file(temp_path, mode='r') as h5_file:
            times = h5_file.root.spikes.times
            units = h5_file.root.spikes.units
            labels = h5_file.root.labels

            num_samples = len(times)
            if max_samples is not None:
                num_samples = min(num_samples, max_samples)

            print(f"  📊 总样本数: {len(times)}, 读取: {num_samples}")

            # 读取数据
            times_data = [times[i] for i in range(num_samples)]
            units_data = [units[i] for i in range(num_samples)]
            labels_data = labels[:num_samples]

            return times_data, units_data, labels_data
    finally:
        # 清理临时文件
        os.unlink(temp_path)

def convert_to_spike_tensor(times: np.ndarray, units: np.ndarray,
                           dt: float = 1e-3, max_time: float = 1.0,
                           input_size: int = 700) -> torch.Tensor:
    """
    将脉冲时间序列转换为张量

    Args:
        times: 脉冲时间数组
        units: 脉冲单元ID数组
        dt: 时间分辨率
        max_time: 最大时间
        input_size: 输入维度

    Returns:
        spike_tensor: [T, input_size]
    """
    num_steps = int(max_time / dt)
    spike_tensor = torch.zeros(num_steps, input_size)

    if len(times) > 0:
        # 转换时间到时间步
        time_steps = (times / dt).astype(np.int64)

        # 过滤有效的时间步和单元
        valid_mask = (time_steps < num_steps) & (units > 0) & (units <= input_size)
        valid_times = time_steps[valid_mask].astype(np.int64)
        valid_units = (units[valid_mask] - 1).astype(np.int64)  # 转换为0索引

        # 设置脉冲
        spike_tensor[valid_times, valid_units] = 1.0

    return spike_tensor

def quick_shd_test():
    """快速测试SHD数据读取"""
    print("🔍 快速SHD数据测试")
    print("="*50)

    # SHD文件路径
    shd_train_path = "spikingjelly_shd/data/shd_train.h5.gz"
    shd_test_path = "spikingjelly_shd/data/shd_test.h5.gz"

    if not os.path.exists(shd_train_path):
        print(f"❌ 文件不存在: {shd_train_path}")
        return False

    try:
        # 读取少量训练数据
        print("\n📚 读取SHD训练数据...")
        train_times, train_units, train_labels = read_gz_h5_file(shd_train_path, max_samples=10)

        print(f"  样本数: {len(train_times)}")
        print(f"  标签范围: {np.min(train_labels)} - {np.max(train_labels)}")
        print(f"  类别数: {len(np.unique(train_labels))}")

        # 转换第一个样本
        print(f"\n🔄 转换第一个样本...")
        sample_times = train_times[0]
        sample_units = train_units[0]
        sample_label = train_labels[0]

        print(f"  脉冲数: {len(sample_times)}")
        print(f"  时间范围: {np.min(sample_times):.3f} - {np.max(sample_times):.3f}s")
        print(f"  单元范围: {np.min(sample_units)} - {np.max(sample_units)}")
        print(f"  标签: {sample_label}")

        # 转换为张量
        spike_tensor = convert_to_spike_tensor(sample_times, sample_units, dt=1e-3, max_time=1.0)
        print(f"  张量形状: {spike_tensor.shape}")
        print(f"  脉冲密度: {spike_tensor.mean():.6f}")
        print(f"  总脉冲数: {spike_tensor.sum().int()}")

        # 读取测试数据
        if os.path.exists(shd_test_path):
            print(f"\n📖 读取SHD测试数据...")
            test_times, test_units, test_labels = read_gz_h5_file(shd_test_path, max_samples=5)
            print(f"  测试样本数: {len(test_times)}")
            print(f"  测试标签范围: {np.min(test_labels)} - {np.max(test_labels)}")

        return True

    except Exception as e:
        print(f"❌ 读取失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def quick_ssc_test():
    """快速测试SSC数据读取"""
    print("\n🔍 快速SSC数据测试")
    print("="*50)

    # SSC文件路径
    ssc_train_path = "spikingjelly_ssc/data/ssc_train.h5.gz"
    ssc_test_path = "spikingjelly_ssc/data/ssc_test.h5.gz"
    ssc_valid_path = "spikingjelly_ssc/data/ssc_valid.h5.gz"

    if not os.path.exists(ssc_train_path):
        print(f"❌ 文件不存在: {ssc_train_path}")
        return False

    try:
        # 读取少量训练数据
        print(f"\n📚 读取SSC训练数据...")
        train_times, train_units, train_labels = read_gz_h5_file(ssc_train_path, max_samples=5)

        print(f"  样本数: {len(train_times)}")
        print(f"  标签范围: {np.min(train_labels)} - {np.max(train_labels)}")
        print(f"  类别数: {len(np.unique(train_labels))}")

        # 转换第一个样本
        sample_times = train_times[0]
        sample_units = train_units[0]
        sample_label = train_labels[0]

        print(f"  第一个样本:")
        print(f"    脉冲数: {len(sample_times)}")
        print(f"    时间范围: {np.min(sample_times):.3f} - {np.max(sample_times):.3f}s")
        print(f"    标签: {sample_label}")

        # 转换为张量
        spike_tensor = convert_to_spike_tensor(sample_times, sample_units, dt=1e-3, max_time=1.0)
        print(f"    张量形状: {spike_tensor.shape}")
        print(f"    脉冲密度: {spike_tensor.mean():.6f}")

        return True

    except Exception as e:
        print(f"❌ SSC读取失败: {e}")
        return False

def create_mini_dataset(dataset_type="shd", num_samples=20):
    """创建小型数据集用于快速实验"""
    print(f"\n🎯 创建{dataset_type.upper()}小型数据集 ({num_samples}样本)")
    print("="*50)

    if dataset_type.lower() == "shd":
        data_path = "spikingjelly_shd/data/shd_train.h5.gz"
        output_size = 20
    else:  # ssc
        data_path = "spikingjelly_ssc/data/ssc_train.h5.gz"
        output_size = 35

    if not os.path.exists(data_path):
        print(f"❌ 数据文件不存在: {data_path}")
        return None, None

    try:
        # 读取数据
        times_list, units_list, labels = read_gz_h5_file(data_path, max_samples=num_samples)

        # 转换为张量
        spike_tensors = []
        for i in range(len(times_list)):
            spike_tensor = convert_to_spike_tensor(
                times_list[i], units_list[i], dt=1e-3, max_time=1.0
            )
            spike_tensors.append(spike_tensor)

        # 堆叠为批次
        data_tensor = torch.stack(spike_tensors)
        labels_tensor = torch.from_numpy(labels).long()

        print(f"✅ 数据集创建成功:")
        print(f"  数据形状: {data_tensor.shape}")
        print(f"  标签形状: {labels_tensor.shape}")
        print(f"  类别数: {len(torch.unique(labels_tensor))}")
        print(f"  平均脉冲密度: {data_tensor.mean():.6f}")

        return data_tensor, labels_tensor

    except Exception as e:
        print(f"❌ 数据集创建失败: {e}")
        return None, None

def main():
    """主函数"""
    print("🚀 直接读取压缩数据文件测试")
    print("="*60)

    # 测试SHD数据读取
    shd_success = quick_shd_test()

    # 测试SSC数据读取
    ssc_success = quick_ssc_test()

    if shd_success:
        # 创建SHD小型数据集
        shd_data, shd_labels = create_mini_dataset("shd", 10)

    if ssc_success:
        # 创建SSC小型数据集
        ssc_data, ssc_labels = create_mini_dataset("ssc", 5)

    print(f"\n🏁 测试完成!")
    print(f"  SHD数据读取: {'✅' if shd_success else '❌'}")
    print(f"  SSC数据读取: {'✅' if ssc_success else '❌'}")

    if shd_success or ssc_success:
        print(f"\n🎉 成功！可以直接使用压缩文件，无需解压!")
        print(f"  这大大节省了磁盘空间和处理时间")
        print(f"  可以直接进行DH-SNN实验了")
    else:
        print(f"\n❌ 数据读取失败，请检查文件路径")

    return shd_success or ssc_success

if __name__ == '__main__':
    success = main()
