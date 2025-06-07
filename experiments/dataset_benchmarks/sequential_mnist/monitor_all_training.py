#!/usr/bin/env python3
"""
全局训练监控脚本
实时监控所有并行训练的进度和状态
"""

import os
import sys
import time
import torch
import json
from pathlib import Path
from datetime import datetime

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

try:
    import GPUtil
    GPUTIL_AVAILABLE = True
except ImportError:
    GPUTIL_AVAILABLE = False

class GlobalTrainingMonitor:
    def __init__(self, check_interval=300):
        self.check_interval = check_interval
        self.start_time = datetime.now()
        
        # 定义所有实验
        self.experiments = {
            'S-MNIST_dh_srnn': {
                'path': './results_fixed/S-MNIST_dh_srnn_tbptt_results.pth',
                'name': 'S-MNIST + DH-SRNN',
                'total_epochs': 150
            },
            'S-MNIST_vanilla_srnn': {
                'path': './results_fixed/S-MNIST_vanilla_srnn_tbptt_results.pth',
                'name': 'S-MNIST + Vanilla SRNN',
                'total_epochs': 150
            },
            'PS-MNIST_dh_srnn': {
                'path': './results_fixed/PS-MNIST_dh_srnn_tbptt_results.pth',
                'name': 'PS-MNIST + DH-SRNN',
                'total_epochs': 150
            },
            'PS-MNIST_vanilla_srnn': {
                'path': './results_fixed/PS-MNIST_vanilla_srnn_tbptt_results.pth',
                'name': 'PS-MNIST + Vanilla SRNN',
                'total_epochs': 150
            }
        }
        
        # SHD和SSC实验
        self.other_experiments = {
            'SHD_dh_sfnn': {
                'path': '../shd/results/SHD_DH_SFNN_results.pth',
                'name': 'SHD + DH-SFNN',
                'total_epochs': 100
            },
            'SSC_dh_sfnn': {
                'path': '../ssc/results/SSC_DH_SFNN_results.pth',
                'name': 'SSC + DH-SFNN',
                'total_epochs': 100
            }
        }
        
        self.all_experiments = {**self.experiments, **self.other_experiments}
        
    def get_system_info(self):
        """获取系统资源信息"""
        info = {}

        # CPU和内存信息
        if PSUTIL_AVAILABLE:
            try:
                cpu_percent = psutil.cpu_percent(interval=1)
                memory = psutil.virtual_memory()
                info.update({
                    'cpu_percent': cpu_percent,
                    'memory_used_gb': memory.used / (1024**3),
                    'memory_total_gb': memory.total / (1024**3),
                    'memory_percent': memory.percent
                })
            except Exception as e:
                info['cpu_memory_error'] = str(e)

        # GPU信息
        if GPUTIL_AVAILABLE:
            try:
                gpus = GPUtil.getGPUs()
                gpu_info = []
                for gpu in gpus:
                    gpu_info.append({
                        'id': gpu.id,
                        'name': gpu.name,
                        'load': gpu.load * 100,
                        'memory_used': gpu.memoryUsed,
                        'memory_total': gpu.memoryTotal,
                        'memory_percent': (gpu.memoryUsed / gpu.memoryTotal) * 100,
                        'temperature': gpu.temperature
                    })
                info['gpus'] = gpu_info
            except Exception as e:
                info['gpu_error'] = str(e)
        else:
            # 尝试使用nvidia-smi获取基本GPU信息
            try:
                import subprocess
                result = subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu', '--format=csv,noheader,nounits'],
                                      capture_output=True, text=True, timeout=5)
                if result.returncode == 0:
                    lines = result.stdout.strip().split('\n')
                    gpu_info = []
                    for i, line in enumerate(lines):
                        parts = line.split(', ')
                        if len(parts) == 4:
                            gpu_info.append({
                                'id': i,
                                'name': f'GPU {i}',
                                'load': float(parts[0]),
                                'memory_used': int(parts[1]),
                                'memory_total': int(parts[2]),
                                'memory_percent': (int(parts[1]) / int(parts[2])) * 100,
                                'temperature': int(parts[3])
                            })
                    info['gpus'] = gpu_info
            except Exception as e:
                info['nvidia_smi_error'] = str(e)

        return info
    
    def load_experiment_status(self, exp_key, exp_info):
        """加载单个实验的状态"""
        filepath = Path(exp_info['path'])
        
        if not filepath.exists():
            return {
                'status': 'not_started',
                'epochs_completed': 0,
                'progress_percent': 0,
                'best_acc': 0,
                'current_train_acc': 0,
                'current_test_acc': 0,
                'training_time': 0,
                'last_updated': None
            }
        
        try:
            data = torch.load(filepath, map_location='cpu')
            
            epochs_completed = len(data.get('train_losses', []))
            total_epochs = exp_info['total_epochs']
            progress_percent = (epochs_completed / total_epochs) * 100
            
            # 判断状态
            if epochs_completed == 0:
                status = 'starting'
            elif epochs_completed >= total_epochs:
                status = 'completed'
            else:
                status = 'running'
            
            return {
                'status': status,
                'epochs_completed': epochs_completed,
                'total_epochs': total_epochs,
                'progress_percent': progress_percent,
                'best_acc': data.get('best_acc', 0),
                'current_train_acc': data.get('train_accs', [0])[-1] if data.get('train_accs') else 0,
                'current_test_acc': data.get('test_accs', [0])[-1] if data.get('test_accs') else 0,
                'training_time': data.get('total_time', 0) / 3600,  # 转换为小时
                'last_updated': datetime.now().isoformat()
            }
            
        except Exception as e:
            return {
                'status': 'error',
                'error': str(e),
                'epochs_completed': 0,
                'progress_percent': 0,
                'best_acc': 0,
                'current_train_acc': 0,
                'current_test_acc': 0,
                'training_time': 0,
                'last_updated': None
            }
    
    def generate_status_report(self):
        """生成状态报告"""
        print("=" * 80)
        print(f"🧠 DH-SNN 全局训练监控报告")
        print(f"⏰ 生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"🕐 运行时长: {datetime.now() - self.start_time}")
        print("=" * 80)
        
        # 系统资源信息
        system_info = self.get_system_info()
        print(f"\n💻 系统资源:")

        if 'cpu_percent' in system_info:
            print(f"   CPU使用率: {system_info['cpu_percent']:.1f}%")
            print(f"   内存使用: {system_info['memory_used_gb']:.1f}GB / {system_info['memory_total_gb']:.1f}GB ({system_info['memory_percent']:.1f}%)")
        else:
            print("   CPU/内存信息: 不可用")

        if 'gpus' in system_info and system_info['gpus']:
            for gpu in system_info['gpus']:
                print(f"   GPU {gpu['id']} ({gpu['name']}): {gpu['load']:.1f}% | "
                      f"内存 {gpu['memory_used']}MB/{gpu['memory_total']}MB ({gpu['memory_percent']:.1f}%) | "
                      f"温度 {gpu['temperature']}°C")
        else:
            print("   GPU信息: 不可用")
        
        # 实验状态
        print(f"\n📊 实验进度:")
        print("-" * 80)
        
        total_experiments = len(self.all_experiments)
        completed_count = 0
        running_count = 0
        not_started_count = 0
        error_count = 0
        
        for exp_key, exp_info in self.all_experiments.items():
            status = self.load_experiment_status(exp_key, exp_info)

            # 状态图标
            if status['status'] == 'completed':
                icon = "✅"
                completed_count += 1
            elif status['status'] == 'running':
                icon = "🔄"
                running_count += 1
            elif status['status'] == 'starting':
                icon = "⏳"
                running_count += 1
            elif status['status'] == 'not_started':
                icon = "⭕"
                not_started_count += 1
            else:
                icon = "❌"
                error_count += 1

            # 进度条
            progress = status['progress_percent']
            bar_length = 20
            filled_length = int(bar_length * progress / 100)
            bar = "█" * filled_length + "░" * (bar_length - filled_length)

            total_epochs = status.get('total_epochs', exp_info['total_epochs'])

            print(f"{icon} {exp_info['name']:<25} │ {bar} │ "
                  f"{status['epochs_completed']:3d}/{total_epochs:3d} │ "
                  f"{progress:5.1f}% │ "
                  f"最佳: {status['best_acc']:5.1f}% │ "
                  f"当前: {status['current_test_acc']:5.1f}%")
        
        # 总体统计
        print("-" * 80)
        print(f"📈 总体进度: {completed_count}/{total_experiments} 完成 | "
              f"{running_count} 运行中 | "
              f"{not_started_count} 未开始 | "
              f"{error_count} 错误")
        
        overall_progress = (completed_count / total_experiments) * 100
        print(f"🎯 整体完成度: {overall_progress:.1f}%")
        
        # 预计完成时间
        if running_count > 0:
            print(f"⏱️ 预计剩余时间: 根据当前进度估算...")
        
        print("=" * 80)
        
        return {
            'timestamp': datetime.now().isoformat(),
            'system_info': system_info,
            'experiments': {exp_key: self.load_experiment_status(exp_key, exp_info) 
                          for exp_key, exp_info in self.all_experiments.items()},
            'summary': {
                'total': total_experiments,
                'completed': completed_count,
                'running': running_count,
                'not_started': not_started_count,
                'error': error_count,
                'overall_progress': overall_progress
            }
        }
    
    def save_status_json(self, status_data):
        """保存状态到JSON文件"""
        status_file = Path('./results_fixed/training_status.json')
        status_file.parent.mkdir(exist_ok=True)
        
        with open(status_file, 'w') as f:
            json.dump(status_data, f, indent=2)
    
    def run_monitoring(self):
        """运行监控循环"""
        print(f"🔍 开始全局训练监控...")
        print(f"⏱️ 检查间隔: {self.check_interval} 秒")
        print("按 Ctrl+C 停止监控\n")
        
        try:
            while True:
                # 生成状态报告
                status_data = self.generate_status_report()
                
                # 保存状态
                self.save_status_json(status_data)
                
                # 检查是否所有实验都完成
                if status_data['summary']['completed'] == status_data['summary']['total']:
                    print("\n🎉 所有实验已完成！")
                    break
                
                # 等待下次检查
                print(f"\n⏳ 等待 {self.check_interval} 秒后下次检查...")
                time.sleep(self.check_interval)
                
        except KeyboardInterrupt:
            print("\n🛑 监控已停止")
            print("可以稍后重新运行监控")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='DH-SNN 全局训练监控')
    parser.add_argument('--interval', type=int, default=300,
                       help='检查间隔(秒), 默认300秒(5分钟)')
    parser.add_argument('--once', action='store_true',
                       help='只检查一次，不进行持续监控')
    
    args = parser.parse_args()
    
    monitor = GlobalTrainingMonitor(check_interval=args.interval)
    
    if args.once:
        print("📊 生成一次性状态报告...")
        status_data = monitor.generate_status_report()
        monitor.save_status_json(status_data)
    else:
        monitor.run_monitoring()

if __name__ == '__main__':
    main()
