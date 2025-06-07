#!/usr/bin/env python3
"""
并行运行4组Sequential MNIST实验
优化DataLoader设置，同时运行多个实验以提高效率
"""

import os
import sys
import time
import subprocess
import threading
from pathlib import Path
import json
from datetime import datetime

def run_experiment(experiment_config):
    """运行单个实验"""
    exp_name = experiment_config['name']
    script_path = experiment_config['script']
    output_dir = experiment_config['output_dir']
    
    print(f"🚀 启动实验: {exp_name}")
    
    # 创建输出目录
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # 设置日志文件
    log_file = Path(output_dir) / f"{exp_name}.log"
    
    try:
        # 运行实验
        with open(log_file, 'w') as f:
            process = subprocess.Popen(
                ['python', script_path],
                stdout=f,
                stderr=subprocess.STDOUT,
                cwd=Path(__file__).parent,
                env=dict(os.environ, CUDA_VISIBLE_DEVICES=str(experiment_config.get('gpu_id', 0)))
            )
            
            # 等待完成
            return_code = process.wait()
            
            if return_code == 0:
                print(f"✅ 实验完成: {exp_name}")
                return True
            else:
                print(f"❌ 实验失败: {exp_name} (返回码: {return_code})")
                return False
                
    except Exception as e:
        print(f"❌ 实验异常: {exp_name} - {e}")
        return False

def monitor_experiments(experiments):
    """监控实验进度"""
    while True:
        time.sleep(30)  # 每30秒检查一次
        
        status_report = []
        for exp in experiments:
            log_file = Path(exp['output_dir']) / f"{exp['name']}.log"
            if log_file.exists():
                try:
                    with open(log_file, 'r') as f:
                        lines = f.readlines()
                        if lines:
                            last_line = lines[-1].strip()
                            status_report.append(f"{exp['name']}: {last_line}")
                        else:
                            status_report.append(f"{exp['name']}: 启动中...")
                except:
                    status_report.append(f"{exp['name']}: 日志读取失败")
            else:
                status_report.append(f"{exp['name']}: 未启动")
        
        print(f"\n📊 实验进度报告 ({datetime.now().strftime('%H:%M:%S')}):")
        print("=" * 80)
        for status in status_report:
            print(f"  {status}")
        print("=" * 80)

def main():
    """主函数"""
    print("🚀 启动并行Sequential MNIST实验")
    print("=" * 60)
    
    # 实验配置
    experiments = [
        {
            'name': 'S-MNIST_DH-SRNN_Exp1',
            'script': 'main_dh_srnn.py',
            'output_dir': 'results_parallel/exp1_dh_srnn',
            'gpu_id': 0,
            'description': 'DH-SRNN on S-MNIST (标准配置)'
        },
        {
            'name': 'S-MNIST_Vanilla-SRNN_Exp2',
            'script': 'main_vanilla_srnn.py', 
            'output_dir': 'results_parallel/exp2_vanilla_srnn',
            'gpu_id': 0,
            'description': 'Vanilla SRNN on S-MNIST (对比基线)'
        },
        {
            'name': 'PS-MNIST_DH-SRNN_Exp3',
            'script': 'start_ps_mnist.py',
            'output_dir': 'results_parallel/exp3_ps_mnist_dh',
            'gpu_id': 0,
            'description': 'DH-SRNN on PS-MNIST (置换序列)'
        },
        {
            'name': 'S-MNIST_DH-SRNN-TBPTT_Exp4',
            'script': 'main_dh_srnn_tbptt.py',
            'output_dir': 'results_parallel/exp4_dh_srnn_tbptt',
            'gpu_id': 0,
            'description': 'DH-SRNN with TBPTT (截断反向传播)'
        }
    ]
    
    print("📋 实验配置:")
    for i, exp in enumerate(experiments, 1):
        print(f"  {i}. {exp['name']}: {exp['description']}")
    
    # 检查脚本是否存在
    missing_scripts = []
    for exp in experiments:
        script_path = Path(exp['script'])
        if not script_path.exists():
            missing_scripts.append(exp['script'])
    
    if missing_scripts:
        print(f"❌ 缺少脚本文件: {missing_scripts}")
        return
    
    print(f"\n🔧 DataLoader优化设置:")
    print(f"  - num_workers: 4")
    print(f"  - pin_memory: True")
    print(f"  - 并行实验数: {len(experiments)}")
    
    # 创建线程运行实验
    threads = []
    results = {}
    
    def run_experiment_thread(exp_config):
        """线程包装函数"""
        result = run_experiment(exp_config)
        results[exp_config['name']] = result
    
    # 启动监控线程
    monitor_thread = threading.Thread(target=monitor_experiments, args=(experiments,), daemon=True)
    monitor_thread.start()
    
    # 启动实验线程
    start_time = time.time()
    
    for exp in experiments:
        thread = threading.Thread(target=run_experiment_thread, args=(exp,))
        thread.start()
        threads.append(thread)
        time.sleep(5)  # 错开启动时间，避免资源冲突
    
    print(f"\n🏃 所有实验已启动，等待完成...")
    
    # 等待所有实验完成
    for thread in threads:
        thread.join()
    
    end_time = time.time()
    total_time = end_time - start_time
    
    # 生成结果报告
    print(f"\n🎉 所有实验完成!")
    print(f"⏱️  总耗时: {total_time/3600:.2f} 小时")
    print("=" * 60)
    
    success_count = 0
    for exp_name, success in results.items():
        status = "✅ 成功" if success else "❌ 失败"
        print(f"  {exp_name}: {status}")
        if success:
            success_count += 1
    
    print(f"\n📊 成功率: {success_count}/{len(experiments)} ({success_count/len(experiments)*100:.1f}%)")
    
    # 保存实验报告
    report = {
        'timestamp': datetime.now().isoformat(),
        'total_time_hours': total_time / 3600,
        'experiments': experiments,
        'results': results,
        'success_rate': success_count / len(experiments)
    }
    
    report_file = Path('results_parallel/experiment_report.json')
    report_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"📄 实验报告已保存: {report_file}")
    
    # 生成结果汇总
    print(f"\n📈 结果文件位置:")
    for exp in experiments:
        result_dir = Path(exp['output_dir'])
        if result_dir.exists():
            files = list(result_dir.glob('*'))
            print(f"  {exp['name']}: {len(files)} 个文件在 {result_dir}")

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print(f"\n⚠️  用户中断实验")
    except Exception as e:
        print(f"\n❌ 实验异常: {e}")
        import traceback
        traceback.print_exc()
