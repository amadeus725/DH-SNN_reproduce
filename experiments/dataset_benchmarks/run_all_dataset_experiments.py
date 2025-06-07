#!/usr/bin/env python3
"""
运行所有数据集基准实验的统一脚本
"""

import os
import sys
import subprocess
import time
import argparse
from pathlib import Path

# 实验配置
EXPERIMENTS = {
    'sequential_mnist': {
        'path': 'sequential_mnist',
        'scripts': ['main_vanilla_srnn.py', 'main_dh_srnn.py'],
        'description': 'Sequential MNIST - 序列MNIST数字识别',
        'estimated_time': '2-3小时',
        'priority': 1
    },
    'permuted_mnist': {
        'path': 'permuted_mnist',
        'scripts': ['main_vanilla_srnn.py', 'main_dh_srnn.py'],
        'description': 'Permuted MNIST - 置换MNIST数字识别',
        'estimated_time': '2-3小时',
        'priority': 1
    },
    'gsc': {
        'path': 'gsc',
        'scripts': ['main_vanilla_sfnn.py', 'main_dh_sfnn.py', 'main_vanilla_srnn.py', 'main_dh_srnn.py'],
        'description': 'Google Speech Commands - 语音命令识别',
        'estimated_time': '4-6小时',
        'priority': 2
    },
    'shd': {
        'path': 'shd',
        'scripts': ['main_vanilla_sfnn.py', 'main_dh_sfnn.py', 'main_vanilla_srnn.py', 'main_dh_srnn.py'],
        'description': 'Spiking Heidelberg Digits - 脉冲数字识别',
        'estimated_time': '3-4小时',
        'priority': 2
    },
    'ssc': {
        'path': 'ssc',
        'scripts': ['main_vanilla_sfnn.py', 'main_dh_sfnn.py', 'main_vanilla_srnn.py', 'main_dh_srnn.py'],
        'description': 'Spiking Speech Commands - 脉冲语音命令',
        'estimated_time': '4-6小时',
        'priority': 2
    },
    'timit': {
        'path': 'timit',
        'scripts': ['main_dh_sfnn.py', 'main_dh_srnn.py'],
        'description': 'TIMIT - 语音识别',
        'estimated_time': '6-8小时',
        'priority': 3
    },
    'deap': {
        'path': 'deap',
        'scripts': ['main_vanilla_sfnn.py', 'main_dh_sfnn.py', 'main_vanilla_srnn.py', 'main_dh_srnn.py'],
        'description': 'DEAP - EEG情感识别',
        'estimated_time': '5-7小时',
        'priority': 3
    },
    'neurovpr': {
        'path': 'neurovpr',
        'scripts': ['main_dh_sfnn.py'],
        'description': 'NeuroVPR - 视觉位置识别',
        'estimated_time': '8-10小时',
        'priority': 3
    }
}

def print_banner():
    """打印横幅"""
    print("="*80)
    print("🚀 DH-SNN 数据集基准实验运行器")
    print("="*80)
    print()

def list_experiments():
    """列出所有可用实验"""
    print("📋 可用实验列表:")
    print()
    
    for priority in [1, 2, 3]:
        print(f"🎯 优先级 {priority}:")
        for name, config in EXPERIMENTS.items():
            if config['priority'] == priority:
                print(f"  • {name:15} - {config['description']}")
                print(f"    {'':17} 预计时间: {config['estimated_time']}")
                print(f"    {'':17} 脚本: {', '.join(config['scripts'])}")
        print()

def run_experiment(experiment_name, script_name, dry_run=False):
    """运行单个实验"""
    
    if experiment_name not in EXPERIMENTS:
        print(f"❌ 未知实验: {experiment_name}")
        return False
    
    config = EXPERIMENTS[experiment_name]
    experiment_path = Path(config['path'])
    script_path = experiment_path / script_name
    
    if not script_path.exists():
        print(f"❌ 脚本不存在: {script_path}")
        return False
    
    print(f"🏃 运行实验: {experiment_name} - {script_name}")
    print(f"📁 路径: {experiment_path}")
    
    if dry_run:
        print("🔍 [DRY RUN] 模拟运行，不执行实际命令")
        return True
    
    try:
        # 切换到实验目录
        original_cwd = os.getcwd()
        os.chdir(experiment_path)
        
        # 运行实验
        start_time = time.time()
        result = subprocess.run([sys.executable, script_name], 
                              capture_output=False, 
                              text=True)
        end_time = time.time()
        
        # 返回原目录
        os.chdir(original_cwd)
        
        duration = end_time - start_time
        if result.returncode == 0:
            print(f"✅ 实验完成: {experiment_name} - {script_name}")
            print(f"⏱️  用时: {duration:.1f}秒")
            return True
        else:
            print(f"❌ 实验失败: {experiment_name} - {script_name}")
            print(f"💥 返回码: {result.returncode}")
            return False
            
    except Exception as e:
        print(f"💥 运行错误: {e}")
        os.chdir(original_cwd)
        return False

def run_experiments_by_priority(priority, dry_run=False):
    """按优先级运行实验"""
    
    print(f"🎯 运行优先级 {priority} 的所有实验")
    print()
    
    success_count = 0
    total_count = 0
    
    for name, config in EXPERIMENTS.items():
        if config['priority'] == priority:
            print(f"📦 开始实验组: {name}")
            for script in config['scripts']:
                total_count += 1
                if run_experiment(name, script, dry_run):
                    success_count += 1
                print()
    
    print(f"📊 优先级 {priority} 完成统计:")
    print(f"   成功: {success_count}/{total_count}")
    print()

def main():
    parser = argparse.ArgumentParser(description='DH-SNN数据集基准实验运行器')
    parser.add_argument('--list', action='store_true', help='列出所有可用实验')
    parser.add_argument('--experiment', type=str, help='运行指定实验')
    parser.add_argument('--script', type=str, help='运行指定脚本')
    parser.add_argument('--priority', type=int, choices=[1, 2, 3], help='运行指定优先级的所有实验')
    parser.add_argument('--all', action='store_true', help='运行所有实验')
    parser.add_argument('--dry-run', action='store_true', help='模拟运行，不执行实际命令')
    
    args = parser.parse_args()
    
    print_banner()
    
    if args.list:
        list_experiments()
        return
    
    if args.experiment and args.script:
        run_experiment(args.experiment, args.script, args.dry_run)
        return
    
    if args.priority:
        run_experiments_by_priority(args.priority, args.dry_run)
        return
    
    if args.all:
        for priority in [1, 2, 3]:
            run_experiments_by_priority(priority, args.dry_run)
        return
    
    # 默认显示帮助
    list_experiments()
    print("💡 使用示例:")
    print("  python run_all_dataset_experiments.py --list")
    print("  python run_all_dataset_experiments.py --priority 1")
    print("  python run_all_dataset_experiments.py --experiment sequential_mnist --script main_dh_srnn.py")
    print("  python run_all_dataset_experiments.py --all --dry-run")

if __name__ == '__main__':
    main()
