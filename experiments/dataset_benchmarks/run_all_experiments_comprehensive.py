#!/usr/bin/env python3
"""
全面实验执行脚本 - 完成所有DH-SNN实验
"""

import os
import sys
import time
import json
import subprocess
import threading
from datetime import datetime
from pathlib import Path

# 实验配置
EXPERIMENTS = {
    # 优先级1: 核心实验
    'priority_1': [
        {
            'name': 'GSC',
            'path': 'gsc',
            'scripts': ['gsc_spikingjelly_experiment.py'],
            'description': 'Google Speech Commands - 语音识别',
            'expected_time': '2-3小时',
            'status': 'running'  # 当前正在运行
        },
        {
            'name': 'Permuted_MNIST',
            'path': 'permuted_mnist',
            'scripts': ['main_dh_srnn.py', 'main_vanilla_srnn.py'],
            'description': 'Permuted MNIST - 序列学习',
            'expected_time': '1-2小时',
            'status': 'ready'
        },
        {
            'name': 'SHD',
            'path': 'shd',
            'scripts': ['main_dh_srnn.py', 'main_dh_sfnn.py', 'main_vanilla_srnn.py', 'main_vanilla_sfnn.py'],
            'description': 'Spiking Heidelberg Digits - 语音识别',
            'expected_time': '2-3小时',
            'status': 'ready'
        }
    ],
    
    # 优先级2: 重要实验
    'priority_2': [
        {
            'name': 'SSC',
            'path': 'ssc',
            'scripts': ['main_dh_srnn.py', 'main_dh_sfnn.py', 'main_vanilla_srnn.py', 'main_vanilla_sfnn.py'],
            'description': 'Spiking Speech Commands - 语音识别',
            'expected_time': '2-3小时',
            'status': 'ready'
        },
        {
            'name': 'Sequential_MNIST_DH_Fix',
            'path': 'sequential_mnist',
            'scripts': ['simple_dh_srnn_training.py'],
            'description': 'Sequential MNIST DH-SRNN修复版本',
            'expected_time': '1小时',
            'status': 'running'  # 当前正在运行
        },
        {
            'name': 'TIMIT',
            'path': 'timit',
            'scripts': ['main_dh_srnn.py', 'main_dh_sfnn.py'],
            'description': 'TIMIT - 语音识别',
            'expected_time': '3-4小时',
            'status': 'ready'
        }
    ],
    
    # 优先级3: 扩展实验
    'priority_3': [
        {
            'name': 'DEAP',
            'path': 'deap',
            'scripts': ['main_dh_srnn.py', 'main_dh_sfnn.py', 'main_vanilla_srnn.py', 'main_vanilla_sfnn.py'],
            'description': 'DEAP - 情感识别',
            'expected_time': '2-3小时',
            'status': 'ready'
        },
        {
            'name': 'NeuroVPR',
            'path': 'neurovpr',
            'scripts': ['main_dh_sfnn.py'],
            'description': 'NeuroVPR - 视觉定位',
            'expected_time': '2-3小时',
            'status': 'ready'
        }
    ],
    
    # 优先级4: 时间动态分析
    'priority_4': [
        {
            'name': 'Multi_Timescale_XOR',
            'path': 'temporal_dynamics/multi_timescale_xor',
            'scripts': ['main_experiment.py'],
            'description': 'Multi-timescale XOR - 时间动态分析',
            'expected_time': '30分钟',
            'status': 'needs_completion'
        },
        {
            'name': 'Delayed_XOR',
            'path': 'figure_reproduction/figure3_delayed_xor',
            'scripts': ['main_experiment.py'],
            'description': 'Delayed XOR - 时间动态分析',
            'expected_time': '30分钟',
            'status': 'needs_completion'
        }
    ]
}

class ExperimentRunner:
    def __init__(self):
        self.base_path = Path(__file__).parent
        self.results = {}
        self.running_processes = {}
        self.start_time = datetime.now()
        
    def log(self, message):
        """记录日志"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        print(f"[{timestamp}] {message}")
        
        # 同时写入日志文件
        log_file = self.base_path / "comprehensive_experiment_log.txt"
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(f"[{timestamp}] {message}\n")
    
    def run_experiment(self, experiment, script):
        """运行单个实验"""
        exp_path = self.base_path / experiment['path']
        script_path = exp_path / script
        
        if not script_path.exists():
            self.log(f"❌ 脚本不存在: {script_path}")
            return False
        
        self.log(f"🚀 开始实验: {experiment['name']} - {script}")
        
        try:
            # 切换到实验目录
            os.chdir(exp_path)
            
            # 运行实验
            process = subprocess.Popen(
                [sys.executable, script],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
                universal_newlines=True
            )
            
            # 记录进程
            exp_key = f"{experiment['name']}_{script}"
            self.running_processes[exp_key] = process
            
            # 等待完成
            stdout, stderr = process.communicate()
            
            if process.returncode == 0:
                self.log(f"✅ 实验完成: {experiment['name']} - {script}")
                self.results[exp_key] = {
                    'status': 'completed',
                    'stdout': stdout,
                    'stderr': stderr,
                    'return_code': process.returncode
                }
                return True
            else:
                self.log(f"❌ 实验失败: {experiment['name']} - {script}")
                self.log(f"错误信息: {stderr}")
                self.results[exp_key] = {
                    'status': 'failed',
                    'stdout': stdout,
                    'stderr': stderr,
                    'return_code': process.returncode
                }
                return False
                
        except Exception as e:
            self.log(f"❌ 实验异常: {experiment['name']} - {script}: {e}")
            self.results[exp_key] = {
                'status': 'error',
                'error': str(e)
            }
            return False
        finally:
            # 清理进程记录
            if exp_key in self.running_processes:
                del self.running_processes[exp_key]
    
    def run_experiments_parallel(self, experiments, max_parallel=2):
        """并行运行实验"""
        import concurrent.futures
        
        tasks = []
        for exp in experiments:
            if exp['status'] in ['ready', 'needs_completion']:
                for script in exp['scripts']:
                    tasks.append((exp, script))
        
        self.log(f"📋 准备并行运行 {len(tasks)} 个实验任务 (最大并行数: {max_parallel})")
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_parallel) as executor:
            futures = {
                executor.submit(self.run_experiment, exp, script): (exp['name'], script)
                for exp, script in tasks
            }
            
            for future in concurrent.futures.as_completed(futures):
                exp_name, script = futures[future]
                try:
                    success = future.result()
                    if success:
                        self.log(f"🎉 {exp_name} - {script} 完成")
                    else:
                        self.log(f"💥 {exp_name} - {script} 失败")
                except Exception as e:
                    self.log(f"💥 {exp_name} - {script} 异常: {e}")
    
    def run_experiments_sequential(self, experiments):
        """顺序运行实验"""
        for exp in experiments:
            if exp['status'] in ['ready', 'needs_completion']:
                self.log(f"📂 开始实验组: {exp['name']}")
                for script in exp['scripts']:
                    success = self.run_experiment(exp, script)
                    if not success:
                        self.log(f"⚠️  {exp['name']} - {script} 失败，继续下一个")
                self.log(f"📂 完成实验组: {exp['name']}")
    
    def check_running_experiments(self):
        """检查当前运行的实验"""
        self.log("🔍 检查当前运行的实验...")
        
        # 检查GSC训练
        gsc_log = "/root/DH-SNN_reproduce/results/gsc_training_log_20250606_082559.txt"
        if os.path.exists(gsc_log):
            self.log("🔄 GSC训练正在进行中")
        
        # 检查Sequential MNIST训练
        seq_mnist_processes = subprocess.run(
            ["ps", "aux"], capture_output=True, text=True
        ).stdout
        
        if "simple_dh_srnn_training.py" in seq_mnist_processes:
            self.log("🔄 Sequential MNIST DH-SRNN训练正在进行中")
    
    def save_progress(self):
        """保存进度"""
        progress_file = self.base_path / "comprehensive_experiment_progress.json"
        
        progress_data = {
            'start_time': self.start_time.isoformat(),
            'current_time': datetime.now().isoformat(),
            'results': self.results,
            'experiments': EXPERIMENTS
        }
        
        with open(progress_file, 'w', encoding='utf-8') as f:
            json.dump(progress_data, f, indent=2, ensure_ascii=False)
        
        self.log(f"💾 进度已保存: {progress_file}")
    
    def run_all(self, priority=None, parallel=True, max_parallel=2):
        """运行所有实验"""
        self.log("🚀 开始全面实验执行")
        self.log("=" * 60)
        
        # 检查当前运行的实验
        self.check_running_experiments()
        
        # 确定要运行的实验
        if priority:
            experiments_to_run = EXPERIMENTS.get(f'priority_{priority}', [])
            self.log(f"🎯 运行优先级 {priority} 实验")
        else:
            experiments_to_run = []
            for priority_key in ['priority_1', 'priority_2', 'priority_3', 'priority_4']:
                experiments_to_run.extend(EXPERIMENTS[priority_key])
            self.log("🎯 运行所有实验")
        
        # 过滤掉正在运行的实验
        experiments_to_run = [
            exp for exp in experiments_to_run 
            if exp['status'] not in ['running']
        ]
        
        if not experiments_to_run:
            self.log("ℹ️  没有需要运行的新实验")
            return
        
        self.log(f"📋 计划运行 {len(experiments_to_run)} 个实验组")
        
        # 运行实验
        if parallel:
            self.run_experiments_parallel(experiments_to_run, max_parallel)
        else:
            self.run_experiments_sequential(experiments_to_run)
        
        # 保存进度
        self.save_progress()
        
        # 输出总结
        self.print_summary()
    
    def print_summary(self):
        """打印总结"""
        total_time = datetime.now() - self.start_time
        
        self.log("\n🎉 实验执行总结")
        self.log("=" * 60)
        self.log(f"总执行时间: {total_time}")
        
        completed = sum(1 for r in self.results.values() if r['status'] == 'completed')
        failed = sum(1 for r in self.results.values() if r['status'] == 'failed')
        error = sum(1 for r in self.results.values() if r['status'] == 'error')
        
        self.log(f"完成: {completed}")
        self.log(f"失败: {failed}")
        self.log(f"错误: {error}")
        self.log(f"总计: {len(self.results)}")

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='全面实验执行脚本')
    parser.add_argument('--priority', type=int, choices=[1, 2, 3, 4], 
                       help='运行指定优先级的实验')
    parser.add_argument('--parallel', action='store_true', default=True,
                       help='并行运行实验')
    parser.add_argument('--sequential', action='store_true',
                       help='顺序运行实验')
    parser.add_argument('--max-parallel', type=int, default=2,
                       help='最大并行数')
    
    args = parser.parse_args()
    
    runner = ExperimentRunner()
    
    parallel = args.parallel and not args.sequential
    
    runner.run_all(
        priority=args.priority,
        parallel=parallel,
        max_parallel=args.max_parallel
    )

if __name__ == "__main__":
    main()
