#!/usr/bin/env python3
"""
最终实验启动器 - 确保所有DH-SNN实验都在运行
"""

import os
import sys
import time
import subprocess
from datetime import datetime

class FinalExperimentLauncher:
    def __init__(self):
        self.base_path = "/root/DH-SNN_reproduce/experiments/dataset_benchmarks"
        self.experiments = {
            # 核心实验
            'SHD_DH_SRNN': ('shd', 'main_dh_srnn.py'),
            'SHD_DH_SFNN': ('shd', 'main_dh_sfnn.py'),
            'SHD_Vanilla_SRNN': ('shd', 'main_vanilla_srnn.py'),
            'SHD_Vanilla_SFNN': ('shd', 'main_vanilla_sfnn.py'),
            
            'SSC_DH_SRNN': ('ssc', 'main_dh_srnn.py'),
            'SSC_DH_SFNN': ('ssc', 'main_dh_sfnn.py'),
            'SSC_Vanilla_SRNN': ('ssc', 'main_vanilla_srnn.py'),
            'SSC_Vanilla_SFNN': ('ssc', 'main_vanilla_sfnn.py'),
            
            'Permuted_MNIST_DH': ('permuted_mnist', 'main_dh_srnn.py'),
            'Permuted_MNIST_Vanilla': ('permuted_mnist', 'main_vanilla_srnn.py'),
            
            'TIMIT_DH_SRNN': ('timit', 'main_dh_srnn.py'),
            'TIMIT_DH_SFNN': ('timit', 'main_dh_sfnn.py'),
            'TIMIT_Vanilla_SRNN': ('timit', 'main_vanilla_srnn.py'),
            
            'DEAP_DH_SRNN': ('deap', 'main_dh_srnn.py'),
            'DEAP_DH_SFNN': ('deap', 'main_dh_sfnn.py'),
            'DEAP_Vanilla_SRNN': ('deap', 'main_vanilla_srnn.py'),
            'DEAP_Vanilla_SFNN': ('deap', 'main_vanilla_sfnn.py'),
            
            'NeuroVPR_DH_SFNN': ('neurovpr', 'main_dh_sfnn.py'),
        }
    
    def check_screen_sessions(self):
        """检查screen会话"""
        try:
            result = subprocess.run(['screen', '-ls'], capture_output=True, text=True)
            return result.stdout
        except Exception as e:
            print(f"❌ 检查screen失败: {e}")
            return ""
    
    def check_python_processes(self):
        """检查Python进程"""
        try:
            result = subprocess.run(['ps', 'aux'], capture_output=True, text=True)
            lines = result.stdout.split('\n')
            
            python_procs = []
            for line in lines:
                if 'python' in line and ('main_' in line or 'gsc_' in line or 'simple_' in line):
                    python_procs.append(line.strip())
            
            return python_procs
        except Exception as e:
            print(f"❌ 检查进程失败: {e}")
            return []
    
    def launch_experiment(self, exp_name, exp_dir, script):
        """启动单个实验"""
        full_path = os.path.join(self.base_path, exp_dir)
        script_path = os.path.join(full_path, script)
        
        if not os.path.exists(script_path):
            print(f"⚠️  脚本不存在: {script_path}")
            return False
        
        try:
            # 使用screen启动
            cmd = f"cd {full_path} && screen -dmS '{exp_name}' python {script}"
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            
            if result.returncode == 0:
                print(f"✅ 启动成功: {exp_name}")
                return True
            else:
                print(f"❌ 启动失败: {exp_name} - {result.stderr}")
                return False
                
        except Exception as e:
            print(f"❌ 启动异常: {exp_name} - {e}")
            return False
    
    def get_running_experiments(self):
        """获取正在运行的实验"""
        screen_output = self.check_screen_sessions()
        python_procs = self.check_python_processes()
        
        running = set()
        
        # 从screen会话中提取
        for line in screen_output.split('\n'):
            if '.' in line and '(' in line:
                parts = line.strip().split('.')
                if len(parts) >= 2:
                    session_name = parts[1].split()[0]
                    running.add(session_name)
        
        # 从进程中提取
        for proc in python_procs:
            if 'main_dh_srnn.py' in proc:
                if 'shd' in proc:
                    running.add('SHD_DH_SRNN')
                elif 'ssc' in proc:
                    running.add('SSC_DH_SRNN')
                elif 'permuted_mnist' in proc:
                    running.add('Permuted_MNIST_DH')
                elif 'timit' in proc:
                    running.add('TIMIT_DH_SRNN')
                elif 'deap' in proc:
                    running.add('DEAP_DH_SRNN')
            elif 'main_vanilla_srnn.py' in proc:
                if 'shd' in proc:
                    running.add('SHD_Vanilla_SRNN')
                elif 'ssc' in proc:
                    running.add('SSC_Vanilla_SRNN')
                elif 'permuted_mnist' in proc:
                    running.add('Permuted_MNIST_Vanilla')
                elif 'timit' in proc:
                    running.add('TIMIT_Vanilla_SRNN')
                elif 'deap' in proc:
                    running.add('DEAP_Vanilla_SRNN')
            elif 'gsc_spikingjelly_experiment.py' in proc:
                running.add('GSC_Training')
            elif 'simple_dh_srnn_training.py' in proc:
                running.add('Sequential_MNIST_DH_Fix')
        
        return running
    
    def launch_missing_experiments(self):
        """启动缺失的实验"""
        print("🔍 检查实验状态...")
        
        running = self.get_running_experiments()
        print(f"📊 当前运行中: {len(running)} 个实验")
        for exp in sorted(running):
            print(f"  🔄 {exp}")
        
        print(f"\n🚀 启动缺失的实验...")
        launched = 0
        
        for exp_name, (exp_dir, script) in self.experiments.items():
            if exp_name not in running:
                print(f"\n启动: {exp_name}")
                success = self.launch_experiment(exp_name, exp_dir, script)
                if success:
                    launched += 1
                    time.sleep(2)  # 避免同时启动太多
            else:
                print(f"✅ 已运行: {exp_name}")
        
        print(f"\n📈 新启动了 {launched} 个实验")
        return launched
    
    def print_final_status(self):
        """打印最终状态"""
        print("\n" + "="*60)
        print("🎯 DH-SNN全面实验执行状态")
        print("="*60)
        
        # 检查screen会话
        screen_output = self.check_screen_sessions()
        screen_count = len([line for line in screen_output.split('\n') if '.' in line and '(' in line])
        
        # 检查Python进程
        python_procs = self.check_python_processes()
        
        print(f"📱 Screen会话: {screen_count} 个")
        print(f"🐍 Python进程: {len(python_procs)} 个")
        
        # 显示运行中的实验
        running = self.get_running_experiments()
        print(f"\n🔄 运行中的实验 ({len(running)} 个):")
        for exp in sorted(running):
            print(f"  • {exp}")
        
        # 显示缺失的实验
        missing = set(self.experiments.keys()) - running
        if missing:
            print(f"\n⚠️  未运行的实验 ({len(missing)} 个):")
            for exp in sorted(missing):
                print(f"  • {exp}")
        else:
            print(f"\n✅ 所有实验都在运行中!")
        
        # 预计完成时间
        total_experiments = len(self.experiments) + 2  # +GSC +Sequential MNIST
        running_experiments = len(running)
        completion_rate = running_experiments / total_experiments * 100
        
        print(f"\n📊 完成度: {completion_rate:.1f}% ({running_experiments}/{total_experiments})")
        print(f"⏰ 预计总时间: 24-36小时")
        print(f"🎯 预计完成: {datetime.now().strftime('%Y-%m-%d')} 晚上")
        
        print(f"\n🔧 监控命令:")
        print(f"  screen -ls                    # 查看所有会话")
        print(f"  screen -r <session_name>      # 连接到特定会话")
        print(f"  python monitor_all_experiments.py  # 运行监控脚本")
        
        return len(running), len(missing)

def main():
    """主函数"""
    print("🚀 DH-SNN最终实验启动器")
    print("="*60)
    
    launcher = FinalExperimentLauncher()
    
    # 启动缺失的实验
    launched = launcher.launch_missing_experiments()
    
    # 等待一下让实验启动
    if launched > 0:
        print(f"\n⏳ 等待 {launched} 个实验启动...")
        time.sleep(10)
    
    # 打印最终状态
    running_count, missing_count = launcher.print_final_status()
    
    if missing_count == 0:
        print(f"\n🎉 所有实验已启动! 总计 {running_count} 个实验正在运行")
    else:
        print(f"\n⚠️  还有 {missing_count} 个实验需要手动检查")
    
    print(f"\n✅ 启动器执行完成!")

if __name__ == "__main__":
    main()
