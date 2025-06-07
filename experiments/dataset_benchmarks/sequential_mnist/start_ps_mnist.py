#!/usr/bin/env python3
"""
启动PS-MNIST (Permuted Sequential MNIST) 实验
在S-MNIST实验完成后自动启动
"""

import os
import time
import subprocess
import torch
from pathlib import Path

def check_s_mnist_completion():
    """检查S-MNIST实验是否完成"""
    results_dir = Path('./results')
    
    # 检查需要的结果文件
    required_files = [
        'S-MNIST_vanilla_srnn_results.pth',
        'S-MNIST_dh_srnn_results.pth'
    ]
    
    completed = []
    for filename in required_files:
        filepath = results_dir / filename
        if filepath.exists():
            try:
                data = torch.load(filepath, map_location='cpu')
                epochs_completed = len(data.get('train_losses', []))
                if epochs_completed >= 150:  # 完整训练完成
                    completed.append(filename)
                    print(f"✅ {filename}: {epochs_completed}/150 epochs completed")
                else:
                    print(f"⏳ {filename}: {epochs_completed}/150 epochs completed")
            except Exception as e:
                print(f"❌ Error loading {filename}: {e}")
        else:
            print(f"⏳ Waiting for {filename}")
    
    return len(completed) == len(required_files)

def start_ps_mnist_experiments():
    """启动PS-MNIST实验"""
    print("🚀 Starting PS-MNIST experiments...")
    
    # 启动Vanilla SRNN PS-MNIST
    print("📊 Starting PS-MNIST Vanilla SRNN...")
    vanilla_cmd = [
        'python', 'main_vanilla_srnn.py',
        '--permute',  # 启用置换
        '--batch_size', '128',
        '--epochs', '150',
        '--lr', '1e-2',
        '--lr_decay', '0.1',
        '--lr_step', '50',
        '--device', 'cuda',
        '--seed', '42',
        '--save_dir', './results'
    ]
    
    vanilla_process = subprocess.Popen(vanilla_cmd)
    print(f"✅ PS-MNIST Vanilla SRNN started (PID: {vanilla_process.pid})")
    
    # 等待一段时间再启动DH-SRNN，避免GPU内存冲突
    time.sleep(60)
    
    # 启动DH-SRNN PS-MNIST
    print("📊 Starting PS-MNIST DH-SRNN...")
    dh_cmd = [
        'python', 'main_dh_srnn.py',
        '--model_type', 'dh_srnn',
        '--num_branches', '4',
        '--permute',  # 启用置换
        '--batch_size', '128',
        '--epochs', '150',
        '--lr', '1e-2',
        '--lr_decay', '0.1',
        '--lr_step', '50',
        '--device', 'cuda',
        '--seed', '42',
        '--save_dir', './results'
    ]
    
    dh_process = subprocess.Popen(dh_cmd)
    print(f"✅ PS-MNIST DH-SRNN started (PID: {dh_process.pid})")
    
    return vanilla_process, dh_process

def monitor_and_start():
    """监控S-MNIST完成情况并启动PS-MNIST"""
    print("🔍 Monitoring S-MNIST experiments...")
    print("Will start PS-MNIST experiments when S-MNIST is completed.")
    
    check_interval = 300  # 5分钟检查一次
    
    while True:
        print(f"\n⏰ {time.strftime('%Y-%m-%d %H:%M:%S')} - Checking S-MNIST status...")
        
        if check_s_mnist_completion():
            print("\n🎉 S-MNIST experiments completed!")
            print("🚀 Starting PS-MNIST experiments...")
            
            vanilla_process, dh_process = start_ps_mnist_experiments()
            
            print(f"\n📋 PS-MNIST Experiments Status:")
            print(f"   Vanilla SRNN PID: {vanilla_process.pid}")
            print(f"   DH-SRNN PID: {dh_process.pid}")
            print(f"\n💡 You can monitor progress with:")
            print(f"   python monitor_training.py --interval 300")
            
            break
        else:
            print(f"⏳ S-MNIST experiments still running. Checking again in {check_interval} seconds...")
            time.sleep(check_interval)

def force_start_ps_mnist():
    """强制启动PS-MNIST实验（不等待S-MNIST完成）"""
    print("⚠️  Force starting PS-MNIST experiments...")
    vanilla_process, dh_process = start_ps_mnist_experiments()
    
    print(f"\n📋 PS-MNIST Experiments Status:")
    print(f"   Vanilla SRNN PID: {vanilla_process.pid}")
    print(f"   DH-SRNN PID: {dh_process.pid}")

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Start PS-MNIST experiments')
    parser.add_argument('--force', action='store_true',
                       help='Force start PS-MNIST without waiting for S-MNIST completion')
    parser.add_argument('--check-only', action='store_true',
                       help='Only check S-MNIST status, do not start PS-MNIST')
    
    args = parser.parse_args()
    
    if args.check_only:
        print("🔍 Checking S-MNIST completion status...")
        if check_s_mnist_completion():
            print("✅ S-MNIST experiments are completed!")
        else:
            print("⏳ S-MNIST experiments are still running.")
    elif args.force:
        force_start_ps_mnist()
    else:
        monitor_and_start()
