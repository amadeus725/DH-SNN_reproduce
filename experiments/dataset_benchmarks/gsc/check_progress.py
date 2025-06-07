#!/usr/bin/env python3
"""
快速检查GSC训练进度
"""

import os
import glob
from datetime import datetime

def check_training_progress():
    """检查训练进度"""
    print("🔍 GSC训练进度检查")
    print("=" * 50)
    
    # 查找最新的训练日志
    log_pattern = "/root/DH-SNN_reproduce/results/gsc_training_log_*.txt"
    log_files = glob.glob(log_pattern)
    
    if not log_files:
        print("❌ 未找到训练日志文件")
        return
    
    latest_log = max(log_files, key=os.path.getctime)
    print(f"📝 日志文件: {os.path.basename(latest_log)}")
    
    # 读取日志内容
    try:
        with open(latest_log, 'r') as f:
            lines = f.readlines()
        
        print(f"📊 日志行数: {len(lines)}")
        
        # 查找最新的epoch信息
        current_model = None
        latest_epoch_info = None
        
        for line in reversed(lines):
            line = line.strip()
            
            if "🔬 实验:" in line:
                current_model = line.split("🔬 实验:")[1].strip()
                break
            
            if "Epoch" in line and "Train Loss:" in line:
                latest_epoch_info = line
                break
        
        if current_model:
            print(f"🔬 当前训练模型: {current_model}")
        
        if latest_epoch_info:
            print(f"📈 最新进度: {latest_epoch_info}")
        else:
            print("⏳ 训练正在进行中，暂无epoch结果")
        
        # 显示最后几行日志
        print(f"\n📋 最新日志 (最后5行):")
        for line in lines[-5:]:
            print(f"   {line.rstrip()}")
        
        # 检查文件修改时间
        mod_time = datetime.fromtimestamp(os.path.getmtime(latest_log))
        print(f"\n⏰ 日志最后更新: {mod_time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        # 检查是否有保存的模型
        results_dir = "/root/DH-SNN_reproduce/results"
        best_models = glob.glob(os.path.join(results_dir, "gsc_*_best.pth"))
        progress_models = glob.glob(os.path.join(results_dir, "gsc_*_epoch_*.pth"))
        
        if best_models or progress_models:
            print(f"\n💾 已保存的模型:")
            for model in best_models:
                print(f"   ✅ {os.path.basename(model)}")
            for model in progress_models:
                print(f"   📊 {os.path.basename(model)}")
        
    except Exception as e:
        print(f"❌ 读取日志文件时出错: {e}")

def check_system_status():
    """检查系统状态"""
    print(f"\n🖥️  系统状态:")
    
    # 检查GPU使用情况
    try:
        import torch
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            gpu_allocated = torch.cuda.memory_allocated() / 1024**3
            gpu_cached = torch.cuda.memory_reserved() / 1024**3
            print(f"   🎮 GPU: {torch.cuda.get_device_name()}")
            print(f"   💾 显存: {gpu_allocated:.1f}GB / {gpu_memory:.1f}GB (已分配)")
            print(f"   📦 缓存: {gpu_cached:.1f}GB")
        else:
            print("   ❌ CUDA不可用")
    except ImportError:
        print("   ⚠️  无法检查GPU状态")
    
    # 检查磁盘空间
    import shutil
    stat = shutil.disk_usage("/root/DH-SNN_reproduce")
    free_gb = stat.free / (1024**3)
    total_gb = stat.total / (1024**3)
    used_gb = (stat.total - stat.free) / (1024**3)
    print(f"   💽 磁盘: {used_gb:.1f}GB / {total_gb:.1f}GB (剩余 {free_gb:.1f}GB)")

def main():
    """主函数"""
    check_training_progress()
    check_system_status()
    
    print(f"\n⏰ 检查时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("💡 提示: 可以定期运行此脚本检查训练进度")

if __name__ == "__main__":
    main()
