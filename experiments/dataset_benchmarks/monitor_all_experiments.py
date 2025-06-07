#!/usr/bin/env python3
"""
实验监控脚本 - 监控所有正在运行的实验
"""

import os
import sys
import time
import json
import subprocess
from datetime import datetime
from pathlib import Path

class ExperimentMonitor:
    def __init__(self):
        self.base_path = Path(__file__).parent
        self.monitoring = True
        
    def check_process_status(self):
        """检查进程状态"""
        try:
            result = subprocess.run(
                ["ps", "aux"], capture_output=True, text=True
            )
            return result.stdout
        except Exception as e:
            print(f"❌ 检查进程失败: {e}")
            return ""
    
    def check_gsc_training(self):
        """检查GSC训练状态"""
        log_file = "/root/DH-SNN_reproduce/results/gsc_training_log_20250606_082559.txt"
        
        if not os.path.exists(log_file):
            return "❌ GSC日志文件不存在"
        
        try:
            # 读取最后几行
            with open(log_file, 'r') as f:
                lines = f.readlines()
            
            if len(lines) == 0:
                return "⚠️  GSC日志为空"
            
            # 获取最后几行
            last_lines = lines[-5:]
            last_content = ''.join(last_lines).strip()
            
            # 检查是否还在训练
            if "Epoch" in last_content and "Batch" in last_content:
                # 提取当前进度
                for line in reversed(lines):
                    if "Epoch" in line and "/" in line:
                        return f"🔄 GSC训练中: {line.strip()}"
                return "🔄 GSC训练中"
            elif "实验完成" in last_content or "训练完成" in last_content:
                return "✅ GSC训练已完成"
            elif "失败" in last_content or "错误" in last_content:
                return "❌ GSC训练失败"
            else:
                return f"⚠️  GSC状态未知: {last_content[-100:]}"
                
        except Exception as e:
            return f"❌ 读取GSC日志失败: {e}"
    
    def check_sequential_mnist(self):
        """检查Sequential MNIST训练状态"""
        processes = self.check_process_status()
        
        if "simple_dh_srnn_training.py" in processes:
            return "🔄 Sequential MNIST DH-SRNN训练中"
        elif "optimized_dh_srnn_training.py" in processes:
            return "🔄 Sequential MNIST优化版训练中"
        elif "debug_practical_dh_srnn.py" in processes:
            return "🔄 Sequential MNIST调试版运行中"
        else:
            # 检查结果文件
            results_dir = self.base_path / "sequential_mnist" / "results"
            if results_dir.exists():
                result_files = list(results_dir.glob("*dh_srnn*.pth"))
                if result_files:
                    latest_file = max(result_files, key=os.path.getctime)
                    return f"✅ Sequential MNIST有结果: {latest_file.name}"
            
            return "⚠️  Sequential MNIST状态未知"
    
    def check_other_experiments(self):
        """检查其他实验状态"""
        processes = self.check_process_status()
        
        running_experiments = []
        
        # 检查各种实验脚本
        experiment_scripts = [
            ("SHD", "shd"),
            ("SSC", "ssc"),
            ("TIMIT", "timit"),
            ("DEAP", "deap"),
            ("NeuroVPR", "neurovpr"),
            ("Permuted MNIST", "permuted_mnist")
        ]
        
        for exp_name, exp_path in experiment_scripts:
            if exp_path in processes or f"main_" in processes:
                running_experiments.append(f"🔄 {exp_name}")
        
        return running_experiments
    
    def check_gpu_usage(self):
        """检查GPU使用情况"""
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total", 
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True
            )
            
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')
                gpu_info = []
                for i, line in enumerate(lines):
                    parts = line.split(', ')
                    if len(parts) >= 3:
                        util, mem_used, mem_total = parts[0], parts[1], parts[2]
                        gpu_info.append(f"GPU{i}: {util}% 使用率, {mem_used}/{mem_total}MB 内存")
                return gpu_info
            else:
                return ["❌ 无法获取GPU信息"]
                
        except Exception as e:
            return [f"❌ GPU检查失败: {e}"]
    
    def check_disk_space(self):
        """检查磁盘空间"""
        try:
            result = subprocess.run(
                ["df", "-h", "/root"], capture_output=True, text=True
            )
            
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')
                if len(lines) >= 2:
                    header = lines[0]
                    data = lines[1]
                    return f"💾 磁盘空间: {data}"
            
            return "❌ 无法获取磁盘信息"
            
        except Exception as e:
            return f"❌ 磁盘检查失败: {e}"
    
    def check_experiment_results(self):
        """检查实验结果"""
        results_dir = Path("/root/DH-SNN_reproduce/results")
        
        if not results_dir.exists():
            return "❌ 结果目录不存在"
        
        result_files = list(results_dir.glob("*.pth")) + list(results_dir.glob("*.json"))
        
        if not result_files:
            return "⚠️  暂无结果文件"
        
        # 按修改时间排序
        result_files.sort(key=os.path.getctime, reverse=True)
        
        recent_results = []
        for f in result_files[:5]:  # 最近5个文件
            mtime = datetime.fromtimestamp(os.path.getctime(f))
            recent_results.append(f"📄 {f.name} ({mtime.strftime('%H:%M:%S')})")
        
        return recent_results
    
    def print_status(self):
        """打印当前状态"""
        os.system('clear')  # 清屏
        
        print("🔍 DH-SNN实验监控面板")
        print("=" * 80)
        print(f"⏰ 当前时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print()
        
        # GSC训练状态
        print("📊 主要实验状态:")
        print("-" * 40)
        gsc_status = self.check_gsc_training()
        print(f"GSC: {gsc_status}")
        
        seq_mnist_status = self.check_sequential_mnist()
        print(f"Sequential MNIST: {seq_mnist_status}")
        
        # 其他实验
        other_experiments = self.check_other_experiments()
        if other_experiments:
            print("\n其他运行中的实验:")
            for exp in other_experiments:
                print(f"  {exp}")
        
        # 系统资源
        print("\n💻 系统资源:")
        print("-" * 40)
        
        gpu_info = self.check_gpu_usage()
        for info in gpu_info:
            print(f"  {info}")
        
        disk_info = self.check_disk_space()
        print(f"  {disk_info}")
        
        # 最近结果
        print("\n📁 最近结果:")
        print("-" * 40)
        recent_results = self.check_experiment_results()
        if isinstance(recent_results, list):
            for result in recent_results:
                print(f"  {result}")
        else:
            print(f"  {recent_results}")
        
        print("\n" + "=" * 80)
        print("按 Ctrl+C 退出监控")
    
    def monitor_loop(self, interval=30):
        """监控循环"""
        try:
            while self.monitoring:
                self.print_status()
                time.sleep(interval)
        except KeyboardInterrupt:
            print("\n\n👋 监控已停止")
            self.monitoring = False
    
    def quick_status(self):
        """快速状态检查"""
        print("🔍 快速状态检查")
        print("=" * 50)
        
        # 检查主要实验
        gsc_status = self.check_gsc_training()
        print(f"GSC: {gsc_status}")
        
        seq_mnist_status = self.check_sequential_mnist()
        print(f"Sequential MNIST: {seq_mnist_status}")
        
        # 检查GPU
        gpu_info = self.check_gpu_usage()
        print(f"GPU: {gpu_info[0] if gpu_info else '未知'}")
        
        # 检查进程数
        processes = self.check_process_status()
        python_processes = len([line for line in processes.split('\n') if 'python' in line])
        print(f"Python进程数: {python_processes}")

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='实验监控脚本')
    parser.add_argument('--interval', type=int, default=30,
                       help='监控间隔(秒)')
    parser.add_argument('--quick', action='store_true',
                       help='快速状态检查')
    
    args = parser.parse_args()
    
    monitor = ExperimentMonitor()
    
    if args.quick:
        monitor.quick_status()
    else:
        print("🚀 开始实验监控...")
        monitor.monitor_loop(args.interval)

if __name__ == "__main__":
    main()
