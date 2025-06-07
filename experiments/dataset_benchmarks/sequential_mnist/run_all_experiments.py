#!/usr/bin/env python3
"""
Sequential MNIST完整实验套件
按照原论文参数运行所有实验：S-MNIST和PS-MNIST，Vanilla SRNN和DH-SRNN
"""

import os
import sys
import time
import argparse
import subprocess
import torch
from pathlib import Path

def run_experiment(script_name, args_list, experiment_name):
    """运行单个实验"""
    print(f"\n🚀 Starting {experiment_name}...")
    print(f"📝 Command: python {script_name} {' '.join(args_list)}")
    
    cmd = ['python', script_name] + args_list
    
    try:
        # 启动进程
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1
        )
        
        print(f"✅ {experiment_name} started (PID: {process.pid})")
        
        # 实时输出日志
        for line in process.stdout:
            print(f"[{experiment_name}] {line.rstrip()}")
        
        # 等待进程完成
        return_code = process.wait()
        
        if return_code == 0:
            print(f"✅ {experiment_name} completed successfully!")
        else:
            print(f"❌ {experiment_name} failed with return code {return_code}")
        
        return return_code == 0
        
    except Exception as e:
        print(f"❌ Error running {experiment_name}: {e}")
        return False

def check_experiment_completion(task_name, model_name):
    """检查实验是否已完成"""
    results_dir = Path('./results')
    filename = f'{task_name}_{model_name}_results.pth'
    filepath = results_dir / filename
    
    if filepath.exists():
        try:
            data = torch.load(filepath, map_location='cpu')
            epochs_completed = len(data.get('train_losses', []))
            if epochs_completed >= 150:
                print(f"✅ {filename} already completed ({epochs_completed}/150 epochs)")
                return True
            else:
                print(f"⏳ {filename} partially completed ({epochs_completed}/150 epochs)")
                return False
        except Exception as e:
            print(f"❌ Error loading {filename}: {e}")
            return False
    else:
        print(f"⏳ {filename} not found")
        return False

def run_all_experiments(skip_completed=True):
    """运行所有实验"""
    
    # 实验配置
    experiments = [
        {
            'name': 'S-MNIST Vanilla SRNN',
            'script': 'main_vanilla_srnn.py',
            'args': [
                '--batch_size', '128',
                '--epochs', '150',
                '--lr', '1e-2',
                '--lr_decay', '0.1',
                '--lr_step', '50',
                '--device', 'cuda',
                '--seed', '42',
                '--save_dir', './results'
            ],
            'task': 'S-MNIST',
            'model': 'vanilla_srnn'
        },
        {
            'name': 'S-MNIST DH-SRNN',
            'script': 'main_dh_srnn.py',
            'args': [
                '--model_type', 'dh_srnn',
                '--num_branches', '4',
                '--batch_size', '128',
                '--epochs', '150',
                '--lr', '1e-2',
                '--lr_decay', '0.1',
                '--lr_step', '50',
                '--device', 'cuda',
                '--seed', '42',
                '--save_dir', './results'
            ],
            'task': 'S-MNIST',
            'model': 'dh_srnn'
        },
        {
            'name': 'PS-MNIST Vanilla SRNN',
            'script': 'main_vanilla_srnn.py',
            'args': [
                '--permute',
                '--batch_size', '128',
                '--epochs', '150',
                '--lr', '1e-2',
                '--lr_decay', '0.1',
                '--lr_step', '50',
                '--device', 'cuda',
                '--seed', '42',
                '--save_dir', './results'
            ],
            'task': 'PS-MNIST',
            'model': 'vanilla_srnn'
        },
        {
            'name': 'PS-MNIST DH-SRNN',
            'script': 'main_dh_srnn.py',
            'args': [
                '--model_type', 'dh_srnn',
                '--num_branches', '4',
                '--permute',
                '--batch_size', '128',
                '--epochs', '150',
                '--lr', '1e-2',
                '--lr_decay', '0.1',
                '--lr_step', '50',
                '--device', 'cuda',
                '--seed', '42',
                '--save_dir', './results'
            ],
            'task': 'PS-MNIST',
            'model': 'dh_srnn'
        }
    ]
    
    print("🧪 Sequential MNIST Complete Experiment Suite")
    print("=" * 60)
    print(f"📊 Total experiments: {len(experiments)}")
    print(f"⏱️  Estimated total time: ~{len(experiments) * 12} hours")
    print("=" * 60)
    
    # 创建结果目录
    os.makedirs('./results', exist_ok=True)
    
    completed_count = 0
    failed_count = 0
    
    for i, exp in enumerate(experiments, 1):
        print(f"\n📋 Experiment {i}/{len(experiments)}: {exp['name']}")
        
        # 检查是否已完成
        if skip_completed and check_experiment_completion(exp['task'], exp['model']):
            print(f"⏭️  Skipping {exp['name']} (already completed)")
            completed_count += 1
            continue
        
        # 运行实验
        success = run_experiment(exp['script'], exp['args'], exp['name'])
        
        if success:
            completed_count += 1
            print(f"✅ {exp['name']} completed successfully!")
        else:
            failed_count += 1
            print(f"❌ {exp['name']} failed!")
            
            # 询问是否继续
            response = input(f"Continue with remaining experiments? (y/n): ")
            if response.lower() != 'y':
                print("🛑 Experiment suite stopped by user.")
                break
    
    # 最终报告
    print(f"\n🎯 Experiment Suite Summary:")
    print(f"✅ Completed: {completed_count}/{len(experiments)}")
    print(f"❌ Failed: {failed_count}/{len(experiments)}")
    
    if completed_count == len(experiments):
        print("🎉 All experiments completed successfully!")
        
        # 生成最终结果
        print("\n📊 Generating final results...")
        try:
            subprocess.run(['python', 'visualize_results.py'], check=True)
            print("✅ Results visualization generated!")
        except Exception as e:
            print(f"⚠️  Failed to generate visualization: {e}")
    
    return completed_count, failed_count

def main():
    parser = argparse.ArgumentParser(description='Run Sequential MNIST experiments')
    parser.add_argument('--no-skip', action='store_true',
                       help='Do not skip completed experiments')
    parser.add_argument('--experiment', type=str, choices=['s-mnist', 'ps-mnist'],
                       help='Run only specific experiment type')
    
    args = parser.parse_args()
    
    skip_completed = not args.no_skip
    
    print("🧪 Sequential MNIST Experiment Suite")
    print("Based on paper: 'Temporal dendritic heterogeneity incorporated with spiking neural networks'")
    print(f"⚙️  Skip completed: {skip_completed}")
    
    if args.experiment:
        print(f"🎯 Running only: {args.experiment.upper()}")
    
    # 运行实验
    completed, failed = run_all_experiments(skip_completed)
    
    # 退出码
    sys.exit(0 if failed == 0 else 1)

if __name__ == '__main__':
    main()
