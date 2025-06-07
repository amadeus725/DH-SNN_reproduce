#!/usr/bin/env python3
"""
查看DH-SNN实验结果脚本
"""

import torch
import json
import os
from pathlib import Path
import numpy as np

def check_shd_results():
    """检查SHD实验结果"""
    
    print("🔍 检查SHD SpikingJelly实验结果...")
    print("="*50)
    
    results_dir = Path("spikingjelly_delayed_xor/outputs/results")
    
    if not results_dir.exists():
        print("❌ 结果目录不存在")
        return
    
    # 检查已保存的结果文件
    result_files = list(results_dir.glob("*.pth"))
    
    if not result_files:
        print("📝 暂无保存的结果文件")
        return
    
    print(f"📊 找到 {len(result_files)} 个结果文件:")
    
    for result_file in sorted(result_files):
        print(f"\n📄 {result_file.name}")
        
        try:
            # 加载结果
            results = torch.load(result_file, map_location='cpu')
            
            if isinstance(results, dict):
                # 显示结果内容
                for key, value in results.items():
                    if isinstance(value, (int, float)):
                        print(f"  {key}: {value}")
                    elif isinstance(value, dict) and 'accuracy' in str(value):
                        print(f"  {key}: {value}")
                    elif key == 'config':
                        print(f"  配置: {type(value).__name__}")
            else:
                print(f"  类型: {type(results).__name__}")
                
        except Exception as e:
            print(f"  ❌ 加载失败: {e}")

def check_multi_timescale_results():
    """检查多时间尺度实验结果"""
    
    print("\n🕰️ 检查多时间尺度实验结果...")
    print("="*50)
    
    results_dir = Path("multi_timescale_experiments/results")
    
    if not results_dir.exists():
        print("❌ 结果目录不存在")
        return
    
    # 检查Figure 4b结果
    figure4b_file = results_dir / "figure4b_results.pth"
    if figure4b_file.exists():
        print("📊 Figure 4b结果:")
        try:
            results = torch.load(figure4b_file, map_location='cpu')
            
            for model_name, result in results.items():
                if isinstance(result, dict):
                    mean_acc = result.get('mean', 0)
                    std_acc = result.get('std', 0)
                    print(f"  {model_name:30s}: {mean_acc:5.1f}% ± {std_acc:4.1f}%")
                else:
                    print(f"  {model_name}: {result}")
                    
        except Exception as e:
            print(f"  ❌ 加载失败: {e}")
    else:
        print("📝 Figure 4b结果尚未完成")
    
    # 检查Figure 6结果
    figure6_file = results_dir / "figure6_analysis_results.pth"
    if figure6_file.exists():
        print("\n📊 Figure 6分析结果:")
        try:
            results = torch.load(figure6_file, map_location='cpu')
            
            # 参数分析
            if 'parameter_analysis' in results:
                param_analysis = results['parameter_analysis']
                print("  参数分析:")
                
                vanilla = param_analysis.get('vanilla', {})
                print(f"    Vanilla SNN: {vanilla.get('params', 0)} 参数")
                
                dh_analyses = param_analysis.get('dh_snn', [])
                for analysis in dh_analyses:
                    branches = analysis.get('type', '').split('(')[1].split(' ')[0] if '(' in analysis.get('type', '') else 'N/A'
                    params = analysis.get('params', 0)
                    print(f"    DH-SNN ({branches}分支): {params} 参数")
            
            # 鲁棒性分析
            if 'robustness_analysis' in results:
                print("  鲁棒性分析:")
                robustness = results['robustness_analysis']
                for model_name, model_results in robustness.items():
                    clean_acc = model_results.get(0.0, 0)
                    noisy_acc = model_results.get(0.3, 0)
                    degradation = clean_acc - noisy_acc
                    print(f"    {model_name}: 清洁{clean_acc:.1f}% → 噪声{noisy_acc:.1f}% (降解{degradation:.1f}%)")
                    
        except Exception as e:
            print(f"  ❌ 加载失败: {e}")
    else:
        print("📝 Figure 6分析结果尚未完成")

def check_models():
    """检查保存的模型"""
    
    print("\n🧠 检查保存的模型...")
    print("="*50)
    
    # 检查SHD模型
    models_dir = Path("spikingjelly_delayed_xor/outputs/models")
    if models_dir.exists():
        model_files = list(models_dir.glob("*.pth"))
        if model_files:
            print(f"📊 SHD模型: {len(model_files)} 个")
            for model_file in sorted(model_files):
                try:
                    model_size = model_file.stat().st_size / (1024*1024)  # MB
                    print(f"  {model_file.name}: {model_size:.1f} MB")
                except:
                    print(f"  {model_file.name}: 无法获取大小")
        else:
            print("📝 暂无SHD模型")
    
    # 检查多时间尺度模型
    multi_models_dir = Path("multi_timescale_experiments/models")
    if multi_models_dir.exists():
        model_files = list(multi_models_dir.glob("*.pth"))
        if model_files:
            print(f"📊 多时间尺度模型: {len(model_files)} 个")
            for model_file in sorted(model_files):
                try:
                    model_size = model_file.stat().st_size / (1024*1024)  # MB
                    print(f"  {model_file.name}: {model_size:.1f} MB")
                except:
                    print(f"  {model_file.name}: 无法获取大小")
        else:
            print("📝 暂无多时间尺度模型")

def check_logs():
    """检查训练日志"""
    
    print("\n📋 检查训练日志...")
    print("="*50)
    
    # 检查SHD日志
    logs_dir = Path("spikingjelly_delayed_xor/outputs/logs")
    if logs_dir.exists():
        log_files = list(logs_dir.glob("*.txt")) + list(logs_dir.glob("*.json"))
        if log_files:
            print(f"📊 SHD日志: {len(log_files)} 个")
            for log_file in sorted(log_files):
                try:
                    log_size = log_file.stat().st_size / 1024  # KB
                    print(f"  {log_file.name}: {log_size:.1f} KB")
                except:
                    print(f"  {log_file.name}: 无法获取大小")
        else:
            print("📝 暂无SHD日志")

def check_disk_usage():
    """检查磁盘使用情况"""
    
    print("\n💾 检查磁盘使用情况...")
    print("="*50)
    
    directories = [
        "spikingjelly_delayed_xor/outputs",
        "multi_timescale_experiments/results", 
        "datasets",
        "results"
    ]
    
    total_size = 0
    
    for dir_path in directories:
        path = Path(dir_path)
        if path.exists():
            size = sum(f.stat().st_size for f in path.rglob('*') if f.is_file())
            size_mb = size / (1024*1024)
            total_size += size_mb
            print(f"  {dir_path:35s}: {size_mb:8.1f} MB")
        else:
            print(f"  {dir_path:35s}: 不存在")
    
    print(f"  {'总计':35s}: {total_size:8.1f} MB")

def show_current_experiments():
    """显示当前运行的实验"""
    
    print("\n🔄 当前运行的实验...")
    print("="*50)
    
    import subprocess
    
    try:
        # 检查Python进程
        result = subprocess.run(['ps', 'aux'], capture_output=True, text=True)
        lines = result.stdout.split('\n')
        
        python_processes = []
        for line in lines:
            if 'python' in line and ('experiment' in line or 'main' in line):
                python_processes.append(line)
        
        if python_processes:
            print("🚀 发现运行中的实验:")
            for i, process in enumerate(python_processes, 1):
                parts = process.split()
                if len(parts) >= 11:
                    pid = parts[1]
                    cpu = parts[2]
                    mem = parts[3]
                    command = ' '.join(parts[10:])
                    print(f"  {i}. PID:{pid} CPU:{cpu}% MEM:{mem}% - {command}")
        else:
            print("📝 暂无运行中的实验")
            
    except Exception as e:
        print(f"❌ 无法检查进程: {e}")

def main():
    """主函数"""
    
    print("🔍 DH-SNN实验结果检查器")
    print("="*60)
    
    # 检查各种结果
    check_shd_results()
    check_multi_timescale_results()
    check_models()
    check_logs()
    check_disk_usage()
    show_current_experiments()
    
    print("\n✅ 检查完成!")
    print("\n💡 提示:")
    print("  - 使用 'tail -f spikingjelly_delayed_xor/outputs/logs/*.txt' 查看实时日志")
    print("  - 使用 'nvidia-smi' 查看GPU使用情况")
    print("  - 结果文件可以用 torch.load() 加载查看详细内容")

if __name__ == '__main__':
    main()
