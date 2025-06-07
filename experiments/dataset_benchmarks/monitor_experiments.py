#!/usr/bin/env python3
"""
实验状态监控脚本
"""

import os
import sys
import time
import json
import glob
from pathlib import Path
from datetime import datetime
import argparse

def check_experiment_status(experiment_path):
    """检查实验状态"""
    
    results_dir = experiment_path / 'results'
    if not results_dir.exists():
        return 'not_started', None, None
    
    # 查找结果文件
    result_files = list(results_dir.glob('*.pth')) + list(results_dir.glob('*.json'))
    
    if not result_files:
        return 'running', None, None
    
    # 获取最新的结果文件
    latest_file = max(result_files, key=os.path.getmtime)
    mod_time = datetime.fromtimestamp(os.path.getmtime(latest_file))
    
    # 尝试读取结果
    try:
        if latest_file.suffix == '.json':
            with open(latest_file, 'r') as f:
                results = json.load(f)
        else:
            # PyTorch文件，简单检查存在性
            results = {'file': str(latest_file)}
        
        return 'completed', results, mod_time
    except:
        return 'error', None, mod_time

def get_experiment_info():
    """获取实验信息"""
    
    experiments = {}
    base_path = Path('.')
    
    # 扫描所有实验目录
    for exp_dir in base_path.iterdir():
        if exp_dir.is_dir() and exp_dir.name not in ['.', '..', '__pycache__']:
            
            # 查找Python脚本
            scripts = list(exp_dir.glob('main_*.py'))
            if scripts:
                experiments[exp_dir.name] = {
                    'path': exp_dir,
                    'scripts': [s.name for s in scripts],
                    'status': {},
                    'results': {}
                }
                
                # 检查每个脚本的状态
                for script in scripts:
                    script_name = script.stem
                    status, results, mod_time = check_experiment_status(exp_dir)
                    experiments[exp_dir.name]['status'][script_name] = status
                    experiments[exp_dir.name]['results'][script_name] = {
                        'data': results,
                        'modified': mod_time
                    }
    
    return experiments

def print_status_summary(experiments):
    """打印状态摘要"""
    
    print("="*80)
    print("📊 DH-SNN 实验状态监控")
    print("="*80)
    print(f"🕐 更新时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # 统计信息
    total_experiments = 0
    completed_experiments = 0
    running_experiments = 0
    error_experiments = 0
    not_started_experiments = 0
    
    for exp_name, exp_info in experiments.items():
        print(f"📦 {exp_name}")
        print(f"   📁 路径: {exp_info['path']}")
        
        for script_name in exp_info['scripts']:
            status = exp_info['status'].get(script_name.replace('.py', ''), 'unknown')
            result_info = exp_info['results'].get(script_name.replace('.py', ''), {})
            
            total_experiments += 1
            
            if status == 'completed':
                completed_experiments += 1
                icon = "✅"
                mod_time = result_info.get('modified')
                time_str = mod_time.strftime('%m-%d %H:%M') if mod_time else 'Unknown'
                print(f"   {icon} {script_name:25} - 已完成 ({time_str})")
                
                # 尝试显示准确率
                results = result_info.get('data', {})
                if isinstance(results, dict) and 'best_accuracy' in results:
                    acc = results['best_accuracy']
                    print(f"      {'':27} 准确率: {acc:.3f}")
                    
            elif status == 'running':
                running_experiments += 1
                icon = "🏃"
                print(f"   {icon} {script_name:25} - 运行中...")
                
            elif status == 'error':
                error_experiments += 1
                icon = "❌"
                print(f"   {icon} {script_name:25} - 错误")
                
            else:
                not_started_experiments += 1
                icon = "⏸️"
                print(f"   {icon} {script_name:25} - 未开始")
        
        print()
    
    # 总体统计
    print("📈 总体统计:")
    print(f"   总实验数: {total_experiments}")
    print(f"   已完成:   {completed_experiments} ({completed_experiments/total_experiments*100:.1f}%)")
    print(f"   运行中:   {running_experiments} ({running_experiments/total_experiments*100:.1f}%)")
    print(f"   错误:     {error_experiments} ({error_experiments/total_experiments*100:.1f}%)")
    print(f"   未开始:   {not_started_experiments} ({not_started_experiments/total_experiments*100:.1f}%)")
    print()

def monitor_continuously(interval=60):
    """持续监控"""
    
    print("🔄 开始持续监控模式...")
    print(f"⏱️  刷新间隔: {interval}秒")
    print("💡 按 Ctrl+C 退出")
    print()
    
    try:
        while True:
            # 清屏 (在支持的终端中)
            os.system('clear' if os.name == 'posix' else 'cls')
            
            experiments = get_experiment_info()
            print_status_summary(experiments)
            
            print(f"⏳ 下次更新: {interval}秒后...")
            time.sleep(interval)
            
    except KeyboardInterrupt:
        print("\n👋 监控已停止")

def export_results(output_file='experiment_results.json'):
    """导出结果到JSON文件"""
    
    experiments = get_experiment_info()
    
    # 准备导出数据
    export_data = {
        'timestamp': datetime.now().isoformat(),
        'experiments': {}
    }
    
    for exp_name, exp_info in experiments.items():
        export_data['experiments'][exp_name] = {
            'scripts': exp_info['scripts'],
            'status': exp_info['status'],
            'results': {}
        }
        
        # 提取关键结果信息
        for script_name, result_info in exp_info['results'].items():
            if result_info['data'] and isinstance(result_info['data'], dict):
                export_data['experiments'][exp_name]['results'][script_name] = {
                    'best_accuracy': result_info['data'].get('best_accuracy'),
                    'modified': result_info['modified'].isoformat() if result_info['modified'] else None
                }
    
    # 保存到文件
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(export_data, f, indent=2, ensure_ascii=False)
    
    print(f"📄 结果已导出到: {output_file}")

def main():
    parser = argparse.ArgumentParser(description='DH-SNN实验状态监控')
    parser.add_argument('--monitor', action='store_true', help='持续监控模式')
    parser.add_argument('--interval', type=int, default=60, help='监控刷新间隔(秒)')
    parser.add_argument('--export', type=str, help='导出结果到JSON文件')
    
    args = parser.parse_args()
    
    if args.monitor:
        monitor_continuously(args.interval)
    elif args.export:
        export_results(args.export)
    else:
        experiments = get_experiment_info()
        print_status_summary(experiments)

if __name__ == '__main__':
    main()
