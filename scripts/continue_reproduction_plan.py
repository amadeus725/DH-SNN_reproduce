#!/usr/bin/env python3
"""
DH-SNN论文复现继续计划
基于当前进展制定的下一步实验计划
"""

import os
import sys
import subprocess
from pathlib import Path

def print_status():
    """打印当前状态"""
    print("🎯 DH-SNN论文复现继续计划")
    print("="*60)
    
    print("\n✅ 已完成的重要成果:")
    print("1. SHD数据集实验成功 - DH-SNN显著优于Vanilla SNN")
    print("   - Medium配置: 69.8% → 79.8% (+10.0%)")
    print("2. 多时间尺度XOR实验 - 97.8%准确率超越论文")
    print("3. SpikingJelly框架完整实现")
    
    print("\n🚀 下一步优先任务:")
    print("1. 完成SSC数据集实验")
    print("2. 复现Figure 4完整可视化")
    print("3. 运行其他数据集基准测试")
    print("4. 生成论文级别的结果报告")

def run_ssc_experiment():
    """运行SSC数据集实验"""
    print("\n🔬 启动SSC数据集实验...")
    
    ssc_script = "experiments/legacy_spikingjelly/original_experiments/ssc/main_dh_sfnn.py"
    if os.path.exists(ssc_script):
        print(f"运行: {ssc_script}")
        return subprocess.Popen([sys.executable, ssc_script])
    else:
        print(f"❌ 脚本不存在: {ssc_script}")
        return None

def run_figure4_visualization():
    """运行Figure 4完整可视化"""
    print("\n📊 生成Figure 4完整可视化...")
    
    fig4_script = "experiments/legacy_spikingjelly/original_experiments/temporal_dynamics/multi_timescale_xor/complete_figure4_plotly.py"
    if os.path.exists(fig4_script):
        print(f"运行: {fig4_script}")
        return subprocess.Popen([sys.executable, fig4_script])
    else:
        print(f"❌ 脚本不存在: {fig4_script}")
        return None

def run_gsc_experiment():
    """运行GSC数据集实验"""
    print("\n🎵 启动GSC数据集实验...")
    
    gsc_script = "experiments/legacy_spikingjelly/original_experiments/gsc/main_dh_sfnn.py"
    if os.path.exists(gsc_script):
        print(f"运行: {gsc_script}")
        return subprocess.Popen([sys.executable, gsc_script])
    else:
        print(f"❌ 脚本不存在: {gsc_script}")
        return None

def check_datasets():
    """检查数据集状态"""
    print("\n📁 检查数据集状态...")
    
    datasets = {
        "SHD": "datasets/raw/shd",
        "SSC": "datasets/raw/ssc", 
        "GSC": "datasets/raw/gsc"
    }
    
    for name, path in datasets.items():
        if os.path.exists(path):
            files = os.listdir(path)
            print(f"✅ {name}: {len(files)} 文件")
        else:
            print(f"❌ {name}: 路径不存在 {path}")

def main():
    """主函数"""
    print_status()
    check_datasets()
    
    print("\n🚀 开始执行实验...")
    
    # 并行运行多个实验
    processes = []
    
    # 1. SSC实验
    ssc_proc = run_ssc_experiment()
    if ssc_proc:
        processes.append(("SSC实验", ssc_proc))
    
    # 2. Figure 4可视化
    fig4_proc = run_figure4_visualization()
    if fig4_proc:
        processes.append(("Figure 4可视化", fig4_proc))
    
    # 3. GSC实验
    gsc_proc = run_gsc_experiment()
    if gsc_proc:
        processes.append(("GSC实验", gsc_proc))
    
    if processes:
        print(f"\n⏳ 正在运行 {len(processes)} 个并行任务...")
        print("您可以继续其他工作，实验将在后台运行。")
        
        for name, proc in processes:
            print(f"  - {name}: PID {proc.pid}")
    else:
        print("\n❌ 没有找到可运行的实验脚本")

if __name__ == "__main__":
    main()
