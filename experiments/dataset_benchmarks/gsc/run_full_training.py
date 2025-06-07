#!/usr/bin/env python3
"""
GSC完整训练脚本
运行完整的150个epoch训练，包括Vanilla SNN和DH-SNN的对比
"""

import os
import sys
import time
import subprocess
import json
from datetime import datetime

def run_training():
    """运行完整训练"""
    print("🚀 开始GSC完整训练")
    print("=" * 60)
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"预计训练时间: ~20-30小时")
    print("=" * 60)
    
    # 确保结果目录存在
    results_dir = "/root/DH-SNN_reproduce/results"
    os.makedirs(results_dir, exist_ok=True)
    
    # 创建训练日志文件
    log_file = os.path.join(results_dir, f"gsc_training_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
    
    try:
        # 运行训练脚本
        cmd = [
            "python", 
            "/root/DH-SNN_reproduce/experiments/dataset_benchmarks/gsc/gsc_spikingjelly_experiment.py"
        ]
        
        print(f"📝 训练日志将保存到: {log_file}")
        print("🔄 开始训练...")
        
        # 启动训练进程，同时输出到控制台和文件
        with open(log_file, 'w') as f:
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                bufsize=1
            )
            
            # 实时输出并保存日志
            for line in process.stdout:
                print(line.rstrip())
                f.write(line)
                f.flush()
            
            # 等待进程完成
            return_code = process.wait()
            
            if return_code == 0:
                print("\n🎉 训练成功完成!")
                return True
            else:
                print(f"\n❌ 训练失败，返回码: {return_code}")
                return False
                
    except KeyboardInterrupt:
        print("\n⚠️  训练被用户中断")
        return False
    except Exception as e:
        print(f"\n❌ 训练过程中出现错误: {e}")
        return False

def check_results():
    """检查训练结果"""
    results_dir = "/root/DH-SNN_reproduce/results"
    
    print("\n📊 检查训练结果...")
    
    # 查找结果文件
    vanilla_results = os.path.join(results_dir, "gsc_vanilla_snn_results.json")
    dh_results = os.path.join(results_dir, "gsc_dh-snn_results.json")
    
    results_summary = {}
    
    if os.path.exists(vanilla_results):
        with open(vanilla_results, 'r') as f:
            vanilla_data = json.load(f)
            results_summary['Vanilla SNN'] = {
                'best_valid_acc': vanilla_data['best_valid_acc'],
                'final_test_acc': vanilla_data['final_test_acc'],
                'epochs_trained': vanilla_data['total_epochs_trained']
            }
            print(f"✅ Vanilla SNN: 验证准确率 {vanilla_data['best_valid_acc']:.1f}%, 测试准确率 {vanilla_data['final_test_acc']:.1f}%")
    
    if os.path.exists(dh_results):
        with open(dh_results, 'r') as f:
            dh_data = json.load(f)
            results_summary['DH-SNN'] = {
                'best_valid_acc': dh_data['best_valid_acc'],
                'final_test_acc': dh_data['final_test_acc'],
                'epochs_trained': dh_data['total_epochs_trained']
            }
            print(f"✅ DH-SNN: 验证准确率 {dh_data['best_valid_acc']:.1f}%, 测试准确率 {dh_data['final_test_acc']:.1f}%")
    
    # 计算改进幅度
    if len(results_summary) == 2:
        vanilla_acc = results_summary['Vanilla SNN']['final_test_acc']
        dh_acc = results_summary['DH-SNN']['final_test_acc']
        improvement = dh_acc - vanilla_acc
        print(f"\n🚀 DH-SNN相对改进: +{improvement:.1f}个百分点")
    
    return results_summary

def main():
    """主函数"""
    print("🎯 GSC完整训练启动器")
    print("=" * 60)
    
    # 检查环境
    print("🔍 检查环境...")
    
    # 检查CUDA
    import torch
    if torch.cuda.is_available():
        print(f"✅ CUDA可用: {torch.cuda.get_device_name()}")
        print(f"   显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
    else:
        print("⚠️  CUDA不可用，将使用CPU训练（速度较慢）")
    
    # 检查磁盘空间
    import shutil
    stat = shutil.disk_usage("/root/DH-SNN_reproduce")
    free_gb = stat.free / (1024**3)
    print(f"💾 可用磁盘空间: {free_gb:.1f}GB")
    
    if free_gb < 2:
        print("⚠️  磁盘空间不足，建议至少2GB")
        response = input("是否继续? (y/N): ")
        if response.lower() != 'y':
            return False
    
    # 询问用户确认
    print(f"\n📋 训练配置:")
    print(f"   模型: Vanilla SNN + DH-SNN")
    print(f"   数据集: GSC (Google Speech Commands)")
    print(f"   训练轮数: 150 epochs")
    print(f"   批次大小: 200")
    print(f"   预计时间: 20-30小时")
    
    response = input("\n是否开始完整训练? (y/N): ")
    if response.lower() != 'y':
        print("❌ 训练取消")
        return False
    
    # 开始训练
    start_time = time.time()
    success = run_training()
    end_time = time.time()
    
    # 输出总结
    total_time = end_time - start_time
    print(f"\n⏱️  总训练时间: {total_time/3600:.1f}小时")
    
    if success:
        # 检查结果
        results = check_results()
        
        # 保存总结
        summary_file = "/root/DH-SNN_reproduce/results/gsc_training_summary.json"
        summary = {
            'training_completed': True,
            'total_time_hours': total_time / 3600,
            'completion_time': datetime.now().isoformat(),
            'results': results
        }
        
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"💾 训练总结保存到: {summary_file}")
        print("🎉 完整训练成功完成!")
        return True
    else:
        print("❌ 训练未能完成")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
