#!/usr/bin/env python3
"""
Sequential MNIST完整实验运行脚本
包括S-MNIST和PS-MNIST，DH-SRNN和Vanilla SRNN的对比
"""

import os
import sys
import subprocess
import time
from pathlib import Path

def run_experiment(script_name, args_list, experiment_name):
    """运行单个实验"""
    print(f"\n🚀 开始实验: {experiment_name}")
    print("="*60)
    
    # 构建命令
    cmd = [sys.executable, script_name] + args_list
    print(f"执行命令: {' '.join(cmd)}")
    
    start_time = time.time()
    
    try:
        # 运行实验
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=os.getcwd())
        
        if result.returncode == 0:
            print(f"✅ {experiment_name} 完成")
            print("标准输出:")
            print(result.stdout[-1000:])  # 显示最后1000个字符
        else:
            print(f"❌ {experiment_name} 失败")
            print("错误输出:")
            print(result.stderr)
            return False
            
    except Exception as e:
        print(f"❌ {experiment_name} 执行异常: {e}")
        return False
    
    elapsed_time = time.time() - start_time
    print(f"⏱️ 实验耗时: {elapsed_time/3600:.2f} 小时")
    
    return True

def main():
    """主函数"""
    print("🧪 Sequential MNIST 完整实验套件")
    print("="*60)
    print("将运行以下实验:")
    print("1. S-MNIST + Vanilla SRNN (基线)")
    print("2. S-MNIST + DH-SRNN")
    print("3. PS-MNIST + Vanilla SRNN (基线)")
    print("4. PS-MNIST + DH-SRNN")
    print("5. 结果可视化")
    print("="*60)
    
    # 确保在正确的目录
    script_dir = Path(__file__).parent
    os.chdir(script_dir)
    
    # 创建结果目录
    results_dir = Path('./results')
    results_dir.mkdir(exist_ok=True)
    
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
            ]
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
            ]
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
            ]
        },
        {
            'name': 'PS-MNIST DH-SRNN',
            'script': 'main_dh_srnn.py',
            'args': [
                '--model_type', 'dh_srnn',
                '--permute',
                '--num_branches', '4',
                '--batch_size', '128',
                '--epochs', '150',
                '--lr', '1e-2',
                '--lr_decay', '0.1',
                '--lr_step', '50',
                '--device', 'cuda',
                '--seed', '42',
                '--save_dir', './results'
            ]
        }
    ]
    
    # 运行所有实验
    successful_experiments = 0
    total_start_time = time.time()
    
    for i, exp in enumerate(experiments, 1):
        print(f"\n📊 进度: {i}/{len(experiments)}")
        
        if run_experiment(exp['script'], exp['args'], exp['name']):
            successful_experiments += 1
        else:
            print(f"⚠️ {exp['name']} 失败，继续下一个实验...")
    
    # 运行可视化
    if successful_experiments > 0:
        print(f"\n🎨 生成结果可视化...")
        if run_experiment('visualize_results.py', [], '结果可视化'):
            print(f"✅ 可视化完成")
        else:
            print(f"⚠️ 可视化失败")
    
    total_time = time.time() - total_start_time
    
    # 总结
    print(f"\n🎯 实验总结")
    print("="*60)
    print(f"总实验数: {len(experiments)}")
    print(f"成功实验: {successful_experiments}")
    print(f"失败实验: {len(experiments) - successful_experiments}")
    print(f"总耗时: {total_time/3600:.2f} 小时")
    
    if successful_experiments == len(experiments):
        print(f"🎉 所有实验成功完成!")
        print(f"📁 结果保存在: {results_dir.absolute()}")
        
        # 显示结果文件
        result_files = list(results_dir.glob('*.pth'))
        if result_files:
            print(f"\n📄 生成的结果文件:")
            for file in result_files:
                print(f"  - {file.name}")
        
        # 显示图表文件
        plot_files = list(results_dir.glob('*.png')) + list(results_dir.glob('*.html'))
        if plot_files:
            print(f"\n🎨 生成的图表文件:")
            for file in plot_files:
                print(f"  - {file.name}")
    else:
        print(f"⚠️ 部分实验失败，请检查错误信息")
    
    return successful_experiments == len(experiments)

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
