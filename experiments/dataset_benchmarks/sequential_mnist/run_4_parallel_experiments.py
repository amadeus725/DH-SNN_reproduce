#!/usr/bin/env python3
"""
同时训练4个模型：
1. S-MNIST + DH-SRNN
2. S-MNIST + Vanilla SRNN  
3. PS-MNIST + DH-SRNN
4. PS-MNIST + Vanilla SRNN

优化DataLoader设置：num_workers=4, pin_memory=True
"""

import os
import sys
import time
import subprocess
import threading
from pathlib import Path
import json
from datetime import datetime

def run_single_experiment(exp_name, script_name, args=None):
    """运行单个实验"""
    print(f"🚀 启动实验: {exp_name}")
    
    # 创建结果目录
    result_dir = Path(f"results_parallel_4/{exp_name}")
    result_dir.mkdir(parents=True, exist_ok=True)
    
    # 设置日志文件
    log_file = result_dir / f"{exp_name}.log"
    
    try:
        # 构建命令
        cmd = ['python', script_name]
        if args:
            cmd.extend(args)
        
        print(f"  命令: {' '.join(cmd)}")
        
        # 运行实验
        with open(log_file, 'w') as f:
            f.write(f"开始时间: {datetime.now()}\n")
            f.write(f"命令: {' '.join(cmd)}\n")
            f.write("=" * 50 + "\n")
            f.flush()
            
            process = subprocess.Popen(
                cmd,
                stdout=f,
                stderr=subprocess.STDOUT,
                cwd=Path(__file__).parent,
                bufsize=1,
                universal_newlines=True
            )
            
            # 等待完成
            return_code = process.wait()
            
            f.write(f"\n" + "=" * 50 + "\n")
            f.write(f"结束时间: {datetime.now()}\n")
            f.write(f"返回码: {return_code}\n")
        
        if return_code == 0:
            print(f"✅ 实验完成: {exp_name}")
            return True
        else:
            print(f"❌ 实验失败: {exp_name} (返回码: {return_code})")
            return False
            
    except Exception as e:
        print(f"❌ 实验异常: {exp_name} - {e}")
        with open(log_file, 'a') as f:
            f.write(f"\n异常: {e}\n")
        return False

def monitor_progress():
    """监控所有实验的进度"""
    experiments = [
        "S-MNIST_DH-SRNN",
        "S-MNIST_Vanilla-SRNN", 
        "PS-MNIST_DH-SRNN",
        "PS-MNIST_Vanilla-SRNN"
    ]
    
    while True:
        time.sleep(60)  # 每分钟检查一次
        
        print(f"\n📊 实验进度报告 ({datetime.now().strftime('%H:%M:%S')}):")
        print("=" * 80)
        
        for exp_name in experiments:
            log_file = Path(f"results_parallel_4/{exp_name}/{exp_name}.log")
            if log_file.exists():
                try:
                    with open(log_file, 'r') as f:
                        lines = f.readlines()
                        
                    # 查找最新的epoch信息
                    latest_info = "启动中..."
                    for line in reversed(lines[-50:]):  # 检查最后50行
                        if "Epoch" in line and ("Acc" in line or "Loss" in line):
                            latest_info = line.strip()
                            break
                        elif "训练完成" in line or "Training completed" in line:
                            latest_info = "✅ 训练完成"
                            break
                        elif "错误" in line or "Error" in line:
                            latest_info = "❌ 出现错误"
                            break
                    
                    print(f"  {exp_name:25s}: {latest_info}")
                    
                except Exception as e:
                    print(f"  {exp_name:25s}: 日志读取失败 - {e}")
            else:
                print(f"  {exp_name:25s}: 未启动")
        
        print("=" * 80)

def main():
    """主函数"""
    print("🚀 启动4个并行Sequential MNIST实验")
    print("=" * 60)
    
    # 检查必要的脚本文件
    required_scripts = [
        'main_dh_srnn.py',
        'main_vanilla_srnn.py'
    ]
    
    missing_scripts = []
    for script in required_scripts:
        if not Path(script).exists():
            missing_scripts.append(script)
    
    if missing_scripts:
        print(f"❌ 缺少脚本文件: {missing_scripts}")
        return
    
    print("📋 实验配置:")
    print("  1. S-MNIST + DH-SRNN: 标准序列 + 多分支树突")
    print("  2. S-MNIST + Vanilla SRNN: 标准序列 + 普通循环网络")
    print("  3. PS-MNIST + DH-SRNN: 置换序列 + 多分支树突")
    print("  4. PS-MNIST + Vanilla SRNN: 置换序列 + 普通循环网络")
    
    print(f"\n🔧 优化设置:")
    print(f"  - DataLoader num_workers: 4")
    print(f"  - DataLoader pin_memory: True")
    print(f"  - 并行实验数: 4")
    
    # 启动监控线程
    monitor_thread = threading.Thread(target=monitor_progress, daemon=True)
    monitor_thread.start()
    
    # 定义实验配置
    experiments = [
        {
            'name': 'S-MNIST_DH-SRNN',
            'script': 'main_dh_srnn.py',
            'args': None  # 默认S-MNIST
        },
        {
            'name': 'S-MNIST_Vanilla-SRNN',
            'script': 'main_vanilla_srnn.py',
            'args': None  # 默认S-MNIST
        },
        {
            'name': 'PS-MNIST_DH-SRNN',
            'script': 'main_dh_srnn.py',
            'args': ['--permute']  # 启用置换
        },
        {
            'name': 'PS-MNIST_Vanilla-SRNN',
            'script': 'main_vanilla_srnn.py',
            'args': ['--permute']  # 启用置换
        }
    ]
    
    # 创建线程运行实验
    threads = []
    results = {}
    
    def run_experiment_thread(exp_config):
        """线程包装函数"""
        result = run_single_experiment(
            exp_config['name'], 
            exp_config['script'], 
            exp_config['args']
        )
        results[exp_config['name']] = result
    
    # 启动所有实验
    start_time = time.time()
    
    for i, exp in enumerate(experiments):
        thread = threading.Thread(target=run_experiment_thread, args=(exp,))
        thread.start()
        threads.append(thread)
        
        # 错开启动时间，避免资源冲突
        if i < len(experiments) - 1:
            time.sleep(10)
    
    print(f"\n🏃 所有4个实验已启动，等待完成...")
    print(f"📊 监控进度每分钟更新一次")
    
    # 等待所有实验完成
    for i, thread in enumerate(threads):
        print(f"⏳ 等待实验 {i+1}/4 完成...")
        thread.join()
    
    end_time = time.time()
    total_time = end_time - start_time
    
    # 生成最终报告
    print(f"\n🎉 所有实验完成!")
    print(f"⏱️  总耗时: {total_time/3600:.2f} 小时")
    print("=" * 60)
    
    success_count = 0
    for exp_name, success in results.items():
        status = "✅ 成功" if success else "❌ 失败"
        print(f"  {exp_name:25s}: {status}")
        if success:
            success_count += 1
    
    print(f"\n📊 成功率: {success_count}/{len(experiments)} ({success_count/len(experiments)*100:.1f}%)")
    
    # 保存实验报告
    report = {
        'timestamp': datetime.now().isoformat(),
        'total_time_hours': total_time / 3600,
        'experiments': experiments,
        'results': results,
        'success_rate': success_count / len(experiments),
        'dataloader_config': {
            'num_workers': 4,
            'pin_memory': True
        }
    }
    
    report_file = Path('results_parallel_4/experiment_report.json')
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"📄 实验报告已保存: {report_file}")
    
    # 显示结果文件位置
    print(f"\n📁 结果文件位置:")
    for exp in experiments:
        result_dir = Path(f"results_parallel_4/{exp['name']}")
        if result_dir.exists():
            files = list(result_dir.glob('*'))
            print(f"  {exp['name']:25s}: {len(files)} 个文件")

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print(f"\n⚠️  用户中断实验")
    except Exception as e:
        print(f"\n❌ 实验异常: {e}")
        import traceback
        traceback.print_exc()
