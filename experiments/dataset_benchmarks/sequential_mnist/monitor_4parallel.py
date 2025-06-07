#!/usr/bin/env python3
"""
监控4个并行实验的进度
"""

import os
import time
import re
from pathlib import Path
from datetime import datetime
import argparse

def parse_log_file(log_path):
    """解析日志文件获取训练进度"""
    
    if not log_path.exists():
        return None
    
    try:
        with open(log_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 提取关键信息
        info = {
            'status': 'unknown',
            'current_epoch': 0,
            'total_epochs': 0,
            'best_accuracy': 0.0,
            'current_accuracy': 0.0,
            'last_update': None,
            'model_type': 'unknown',
            'dataset': 'unknown'
        }
        
        # 解析模型类型和数据集
        if 'DH-SRNN' in str(log_path):
            info['model_type'] = 'DH-SRNN'
        elif 'Vanilla-SRNN' in str(log_path):
            info['model_type'] = 'Vanilla-SRNN'
            
        if 'S-MNIST' in str(log_path):
            info['dataset'] = 'S-MNIST'
        elif 'PS-MNIST' in str(log_path):
            info['dataset'] = 'PS-MNIST'
        
        # 查找epoch信息
        epoch_pattern = r'Epoch\s+(\d+)/(\d+)'
        epoch_matches = re.findall(epoch_pattern, content)
        if epoch_matches:
            current_epoch, total_epochs = epoch_matches[-1]
            info['current_epoch'] = int(current_epoch)
            info['total_epochs'] = int(total_epochs)
        
        # 查找准确率信息
        acc_pattern = r'Test Acc:\s*([\d.]+)'
        acc_matches = re.findall(acc_pattern, content)
        if acc_matches:
            info['current_accuracy'] = float(acc_matches[-1])
        
        # 查找最佳准确率
        best_pattern = r'Best:\s*([\d.]+)'
        best_matches = re.findall(best_pattern, content)
        if best_matches:
            info['best_accuracy'] = float(best_matches[-1])
        
        # 检查状态
        if '训练完成' in content or 'Training completed' in content:
            info['status'] = 'completed'
        elif info['current_epoch'] > 0:
            info['status'] = 'running'
        else:
            info['status'] = 'starting'
        
        # 获取文件修改时间
        info['last_update'] = datetime.fromtimestamp(os.path.getmtime(log_path))
        
        return info
        
    except Exception as e:
        print(f"解析日志文件错误 {log_path}: {e}")
        return None

def get_experiment_progress():
    """获取所有实验的进度"""
    
    results_dir = Path('results_parallel_4')
    experiments = {}
    
    if not results_dir.exists():
        return experiments
    
    # 扫描所有实验目录
    for exp_dir in results_dir.iterdir():
        if exp_dir.is_dir():
            log_file = exp_dir / f"{exp_dir.name}.log"
            if log_file.exists():
                info = parse_log_file(log_file)
                if info:
                    experiments[exp_dir.name] = info
    
    return experiments

def print_progress_summary(experiments):
    """打印进度摘要"""
    
    print("="*80)
    print("📊 4个并行实验监控")
    print("="*80)
    print(f"🕐 更新时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    if not experiments:
        print("❌ 未找到实验数据")
        return
    
    # 按数据集分组
    s_mnist_exps = {k: v for k, v in experiments.items() if 'S-MNIST' in k}
    ps_mnist_exps = {k: v for k, v in experiments.items() if 'PS-MNIST' in k}
    
    def print_experiment_group(title, exps):
        if not exps:
            return
            
        print(f"📦 {title}")
        for exp_name, info in exps.items():
            status_icon = {
                'completed': '✅',
                'running': '🏃',
                'starting': '🔄',
                'unknown': '❓'
            }.get(info['status'], '❓')
            
            progress = 0
            if info['total_epochs'] > 0:
                progress = (info['current_epoch'] / info['total_epochs']) * 100
            
            print(f"   {status_icon} {info['model_type']:12} | "
                  f"Epoch {info['current_epoch']:3d}/{info['total_epochs']:3d} "
                  f"({progress:5.1f}%) | "
                  f"Acc: {info['current_accuracy']:.3f} | "
                  f"Best: {info['best_accuracy']:.3f}")
            
            if info['last_update']:
                time_diff = datetime.now() - info['last_update']
                if time_diff.total_seconds() < 60:
                    time_str = f"{int(time_diff.total_seconds())}秒前"
                elif time_diff.total_seconds() < 3600:
                    time_str = f"{int(time_diff.total_seconds()/60)}分钟前"
                else:
                    time_str = f"{int(time_diff.total_seconds()/3600)}小时前"
                print(f"      {'':15} 最后更新: {time_str}")
        print()
    
    print_experiment_group("Sequential MNIST", s_mnist_exps)
    print_experiment_group("Permuted Sequential MNIST", ps_mnist_exps)
    
    # 总体统计
    total_exps = len(experiments)
    completed_exps = sum(1 for exp in experiments.values() if exp['status'] == 'completed')
    running_exps = sum(1 for exp in experiments.values() if exp['status'] == 'running')
    
    print("📈 总体进度:")
    print(f"   总实验数: {total_exps}")
    print(f"   已完成:   {completed_exps} ({completed_exps/total_exps*100:.1f}%)")
    print(f"   运行中:   {running_exps} ({running_exps/total_exps*100:.1f}%)")
    print(f"   待开始:   {total_exps-completed_exps-running_exps}")
    print()

def print_detailed_progress(experiments):
    """打印详细进度"""
    
    print("="*80)
    print("📋 详细实验进度")
    print("="*80)
    print()
    
    for exp_name, info in experiments.items():
        print(f"🔬 {exp_name}")
        print(f"   数据集: {info['dataset']}")
        print(f"   模型:   {info['model_type']}")
        print(f"   状态:   {info['status']}")
        print(f"   进度:   {info['current_epoch']}/{info['total_epochs']} epochs")
        print(f"   当前准确率: {info['current_accuracy']:.3f}")
        print(f"   最佳准确率: {info['best_accuracy']:.3f}")
        if info['last_update']:
            print(f"   最后更新: {info['last_update'].strftime('%Y-%m-%d %H:%M:%S')}")
        print()

def monitor_continuously(interval=30):
    """持续监控"""
    
    print("🔄 开始持续监控4个并行实验...")
    print(f"⏱️  刷新间隔: {interval}秒")
    print("💡 按 Ctrl+C 退出")
    print()
    
    try:
        while True:
            # 清屏 (在支持的终端中)
            os.system('clear' if os.name == 'posix' else 'cls')
            
            experiments = get_experiment_progress()
            print_progress_summary(experiments)
            
            print(f"⏳ 下次更新: {interval}秒后...")
            time.sleep(interval)
            
    except KeyboardInterrupt:
        print("\n👋 监控已停止")

def show_log_tail(exp_name, lines=20):
    """显示指定实验的最新日志"""
    
    log_path = Path(f'results_parallel_4/{exp_name}/{exp_name}.log')
    
    if not log_path.exists():
        print(f"❌ 日志文件不存在: {log_path}")
        return
    
    print(f"📄 {exp_name} 最新日志 (最后{lines}行):")
    print("="*60)
    
    try:
        with open(log_path, 'r', encoding='utf-8') as f:
            all_lines = f.readlines()
            recent_lines = all_lines[-lines:] if len(all_lines) > lines else all_lines
            
            for line in recent_lines:
                print(line.rstrip())
                
    except Exception as e:
        print(f"❌ 读取日志文件错误: {e}")

def main():
    parser = argparse.ArgumentParser(description='监控4个并行实验')
    parser.add_argument('--monitor', action='store_true', help='持续监控模式')
    parser.add_argument('--interval', type=int, default=30, help='监控刷新间隔(秒)')
    parser.add_argument('--detailed', action='store_true', help='显示详细信息')
    parser.add_argument('--log', type=str, help='显示指定实验的日志')
    parser.add_argument('--lines', type=int, default=20, help='显示日志行数')
    
    args = parser.parse_args()
    
    if args.log:
        show_log_tail(args.log, args.lines)
    elif args.monitor:
        monitor_continuously(args.interval)
    else:
        experiments = get_experiment_progress()
        if args.detailed:
            print_detailed_progress(experiments)
        else:
            print_progress_summary(experiments)

if __name__ == '__main__':
    main()
