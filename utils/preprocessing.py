#!/usr/bin/env python3
"""
SHD数据集预处理 - 基于SpikingJelly框架
支持不同的时间分辨率(dt)以复现Figure 4g实验
"""

import os
import gzip
import tables
import numpy as np
import torch
from typing import Tuple, List
import argparse


def extract_gz_file(gz_path: str, output_path: str):
    """解压.gz文件"""
    print(f"Extracting {gz_path} to {output_path}")
    with gzip.open(gz_path, 'rb') as f_in:
        with open(output_path, 'wb') as f_out:
            f_out.write(f_in.read())
    print(f"Extraction completed: {output_path}")


def binary_spike_encoding(times: np.ndarray, units: np.ndarray, 
                         dt: float = 1e-3, max_time: float = None) -> np.ndarray:
    """
    将脉冲时间序列编码为二进制矩阵
    
    Args:
        times: 脉冲时间数组 (秒)
        units: 脉冲单元ID数组 (1-700)
        dt: 时间分辨率 (秒)
        max_time: 最大时间长度，如果为None则使用实际最大时间
    
    Returns:
        spike_matrix: [T, 700] 二进制脉冲矩阵
    """
    if max_time is None:
        max_time = np.max(times)
    
    # 计算时间步数
    num_steps = int(np.ceil(max_time / dt))
    
    # 初始化脉冲矩阵
    spike_matrix = np.zeros((num_steps, 700), dtype=np.float32)
    
    # 将连续时间转换为离散时间步
    time_indices = np.floor(times / dt).astype(int)
    
    # 过滤有效的时间索引和单元ID
    valid_mask = (time_indices < num_steps) & (units > 0) & (units <= 700)
    valid_time_indices = time_indices[valid_mask]
    valid_units = units[valid_mask] - 1  # 转换为0-699索引
    
    # 设置脉冲
    spike_matrix[valid_time_indices, valid_units] = 1.0
    
    return spike_matrix


def add_poisson_noise(spike_matrix: np.ndarray, noise_rate: float = 0.01) -> np.ndarray:
    """添加泊松噪声"""
    noise = np.random.poisson(noise_rate, spike_matrix.shape).astype(np.float32)
    return np.clip(spike_matrix + noise, 0, 1)


def process_shd_dataset(h5_file: str, output_dir: str, dt: float = 1e-3, 
                       max_time: float = 1.0, add_noise: bool = False) -> None:
    """
    处理SHD数据集
    
    Args:
        h5_file: HDF5文件路径
        output_dir: 输出目录
        dt: 时间分辨率
        max_time: 固定的最大时间长度
        add_noise: 是否添加噪声
    """
    print(f"Processing {h5_file} with dt={dt}ms")
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 打开HDF5文件
    with tables.open_file(h5_file, mode='r') as fileh:
        units = fileh.root.spikes.units
        times = fileh.root.spikes.times
        labels = fileh.root.labels
        
        num_samples = len(times)
        print(f"Number of samples: {num_samples}")
        
        # 处理每个样本
        for i in range(num_samples):
            # 获取当前样本的脉冲数据
            sample_times = times[i]
            sample_units = units[i]
            sample_label = labels[i]
            
            # 编码为二进制矩阵
            spike_matrix = binary_spike_encoding(
                sample_times, sample_units, dt=dt, max_time=max_time
            )
            
            # 添加噪声（可选）
            if add_noise:
                spike_matrix = add_poisson_noise(spike_matrix)
            
            # 保存处理后的数据
            output_file = os.path.join(output_dir, f'sample_{i:05d}_label_{sample_label}.npy')
            np.save(output_file, spike_matrix)
            
            if (i + 1) % 1000 == 0:
                print(f"Processed {i + 1}/{num_samples} samples")
    
    print(f"Dataset processing completed. Output saved to: {output_dir}")


def prepare_shd_data(data_dir: str = 'data', dt_list: List[float] = None) -> None:
    """
    准备SHD数据集，支持多种时间分辨率
    
    Args:
        data_dir: 数据目录
        dt_list: 时间分辨率列表 (毫秒)
    """
    if dt_list is None:
        # 论文中使用的时间分辨率
        dt_list = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]  # 毫秒
    
    # 文件路径
    train_gz = os.path.join(data_dir, 'shd_train.h5.gz')
    test_gz = os.path.join(data_dir, 'shd_test.h5.gz')
    
    train_h5 = os.path.join(data_dir, 'shd_train.h5')
    test_h5 = os.path.join(data_dir, 'shd_test.h5')
    
    # 解压文件
    if not os.path.exists(train_h5):
        extract_gz_file(train_gz, train_h5)
    if not os.path.exists(test_h5):
        extract_gz_file(test_gz, test_h5)
    
    # 处理不同时间分辨率的数据
    for dt_ms in dt_list:
        dt_s = dt_ms * 1e-3  # 转换为秒
        
        print(f"\n=== Processing dt = {dt_ms}ms ===")
        
        # 创建输出目录
        train_output_dir = f'processed_data/train_dt_{dt_ms}ms'
        test_output_dir = f'processed_data/test_dt_{dt_ms}ms'
        
        # 处理训练集
        process_shd_dataset(
            train_h5, train_output_dir, dt=dt_s, max_time=1.0, add_noise=False
        )
        
        # 处理测试集
        process_shd_dataset(
            test_h5, test_output_dir, dt=dt_s, max_time=1.0, add_noise=False
        )


def get_dataset_statistics(data_dir: str) -> None:
    """获取数据集统计信息"""
    train_h5 = os.path.join(data_dir, 'shd_train.h5')
    test_h5 = os.path.join(data_dir, 'shd_test.h5')
    
    for split, h5_file in [('Train', train_h5), ('Test', test_h5)]:
        if not os.path.exists(h5_file):
            print(f"File not found: {h5_file}")
            continue
            
        with tables.open_file(h5_file, mode='r') as fileh:
            times = fileh.root.spikes.times
            labels = fileh.root.labels
            
            num_samples = len(times)
            num_classes = len(np.unique(labels[:]))
            
            # 计算序列长度统计
            seq_lengths = [len(t) for t in times]
            max_time = [np.max(t) if len(t) > 0 else 0 for t in times]
            
            print(f"\n{split} Set Statistics:")
            print(f"  Number of samples: {num_samples}")
            print(f"  Number of classes: {num_classes}")
            print(f"  Sequence length - Min: {np.min(seq_lengths)}, Max: {np.max(seq_lengths)}, Mean: {np.mean(seq_lengths):.1f}")
            print(f"  Max time - Min: {np.min(max_time):.3f}s, Max: {np.max(max_time):.3f}s, Mean: {np.mean(max_time):.3f}s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='SHD Dataset Preprocessing')
    parser.add_argument('--data_dir', type=str, default='data', help='Data directory')
    parser.add_argument('--dt_list', type=float, nargs='+', default=[1.0], 
                       help='Time resolution list in milliseconds')
    parser.add_argument('--stats_only', action='store_true', help='Only show dataset statistics')
    
    args = parser.parse_args()
    
    if args.stats_only:
        get_dataset_statistics(args.data_dir)
    else:
        prepare_shd_data(args.data_dir, args.dt_list)
        get_dataset_statistics(args.data_dir)
