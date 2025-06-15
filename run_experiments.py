#!/usr/bin/env python3
"""
DH-SNN 实验运行器
=================

运行dh-snn-ultra-minimal项目中的各种实验
包括核心实验、应用实验和创新实验

"""

import os
import sys
import argparse
import time
from pathlib import Path

# 添加项目路径
current_dir = Path(__file__).parent
sys.path.append(str(current_dir))

print("🚀 DH-SNN 实验运行器")
print("=" * 60)

def run_delayed_xor_experiment():
    """运行延迟异或实验"""
    print("启动延迟异或实验...")
    try:
        from experiments.core.delayed_xor import run_delayed_xor_experiment
        return run_delayed_xor_experiment()
    except ImportError as e:
        print(f"❌ 导入延迟异或实验模块失败: {e}")
        return None

def run_multi_timescale_experiment():
    """运行多时间尺度XOR实验"""
    print("启动多时间尺度XOR实验...")
    try:
        from experiments.core_validation.multi_timescale import run_multi_timescale_experiment
        return run_multi_timescale_experiment()
    except ImportError as e:
        print(f"❌ 导入多时间尺度实验模块失败: {e}")
        return None

def run_innovation_experiments():
    """运行所有创新实验"""
    print("启动创新实验集合...")
    try:
        from experiments.innovations import main as run_innovations
        # 模拟运行创新实验
        sys.argv = ['innovations.py', 'all']
        return run_innovations()
    except ImportError as e:
        print(f"❌ 导入创新实验模块失败: {e}")
        return None

def run_ssc_experiment():
    """运行SSC实验"""
    print("启动SSC语音命令识别实验...")
    try:
        from experiments.applications.ssc import run_ssc_experiment
        return run_ssc_experiment()
    except ImportError as e:
        print(f"❌ 导入SSC实验模块失败: {e}")
        return None

def run_shd_experiment():
    """运行SHD实验"""
    print("启动SHD数字识别实验...")
    try:
        from experiments.applications.shd import run_shd_experiment
        return run_shd_experiment()
    except ImportError as e:
        print(f"❌ 导入SHD实验模块失败: {e}")
        return None

def run_neurovpr_experiment():
    """运行NeuroVPR实验"""
    print("启动NeuroVPR视觉位置识别实验...")
    try:
        from experiments.applications.neurovpr import run_neurovpr_experiment
        return run_neurovpr_experiment()
    except ImportError as e:
        print(f"❌ 导入NeuroVPR实验模块失败: {e}")
        return None

def run_smnist_experiment():
    """运行Sequential MNIST实验"""
    print("启动Sequential MNIST序列分类实验...")
    try:
        from experiments.applications.smnist import run_smnist_experiment
        return run_smnist_experiment()
    except ImportError as e:
        print(f"❌ 导入Sequential MNIST实验模块失败: {e}")
        return None

def run_core_experiments():
    """运行核心验证实验"""
    print("🔬 运行核心验证实验")
    print("=" * 60)
    
    core_experiments = [
        ("延迟异或", run_delayed_xor_experiment),
        ("多时间尺度XOR", run_multi_timescale_experiment),
    ]
    
    results = {}
    start_time = time.time()
    
    for exp_name, exp_func in core_experiments:
        print(f"\n{'='*20} {exp_name}实验 {'='*20}")
        exp_start = time.time()
        
        try:
            result = exp_func()
            if result is not None:
                results[exp_name] = result
                print(f"✅ {exp_name}实验完成")
            else:
                print(f"❌ {exp_name}实验失败")
                results[exp_name] = None
        except Exception as e:
            print(f"❌ {exp_name}实验异常: {e}")
            results[exp_name] = None
        
        exp_time = time.time() - exp_start
        print(f"⏱️  {exp_name}实验用时: {exp_time/60:.1f}分钟")
    
    total_time = time.time() - start_time
    print(f"\n📊 核心实验总用时: {total_time/60:.1f}分钟")
    
    return results

def run_all_experiments():
    """运行所有实验"""
    print("🔬 运行所有DH-SNN实验")
    print("=" * 60)
    
    # 核心验证实验
    experiments = [
        ("延迟异或", run_delayed_xor_experiment),
        ("多时间尺度XOR", run_multi_timescale_experiment),
        ("Sequential MNIST", run_smnist_experiment),
        ("SSC语音命令", run_ssc_experiment),
        ("SHD数字识别", run_shd_experiment),
        ("NeuroVPR位置识别", run_neurovpr_experiment),
    ]
    
    results = {}
    start_time = time.time()
    
    for exp_name, exp_func in experiments:
        print(f"\n{'='*20} {exp_name}实验 {'='*20}")
        exp_start = time.time()
        
        try:
            result = exp_func()
            if result is not None:
                results[exp_name] = result
                print(f"✅ {exp_name}实验完成")
            else:
                print(f"❌ {exp_name}实验失败")
                results[exp_name] = None
        except Exception as e:
            print(f"❌ {exp_name}实验异常: {e}")
            results[exp_name] = None
        
        exp_time = time.time() - exp_start
        print(f"⏱️  {exp_name}实验用时: {exp_time/60:.1f}分钟")
    
    # 运行创新实验
    print(f"\n{'='*20} 创新实验集合 {'='*20}")
    innovation_start = time.time()
    try:
        innovation_results = run_innovation_experiments()
        if innovation_results is not None:
            results["创新实验"] = innovation_results
            print(f"✅ 创新实验完成")
        else:
            print(f"❌ 创新实验失败")
            results["创新实验"] = None
    except Exception as e:
        print(f"❌ 创新实验异常: {e}")
        results["创新实验"] = None
    
    innovation_time = time.time() - innovation_start
    print(f"⏱️  创新实验用时: {innovation_time/60:.1f}分钟")
    
    total_time = time.time() - start_time
    
    # 总结所有实验结果
    print("\n" + "=" * 80)
    print("🎯 所有实验总结")
    print("=" * 80)
    
    successful_experiments = 0
    total_experiments = len(results)
    
    for exp_name, result in results.items():
        if result is not None:
            print(f"✅ {exp_name}实验: 成功")
            successful_experiments += 1
        else:
            print(f"❌ {exp_name}实验: 失败")
    
    print(f"\n📊 实验统计:")
    print(f"   成功实验: {successful_experiments}/{total_experiments}")
    print(f"   总用时: {total_time/60:.1f}分钟")
    print(f"   平均每个实验: {total_time/total_experiments/60:.1f}分钟")
    
    if successful_experiments == total_experiments:
        print("\n🎉 所有实验都成功完成!")
    elif successful_experiments > 0:
        print(f"\n✅ {successful_experiments}个实验成功完成")
    else:
        print("\n❌ 所有实验都失败了")
    
    return results

def list_experiments():
    """列出所有可用的实验"""
    print("📋 可用实验列表:")
    print("-" * 40)
    print("🔬 核心验证实验:")
    print("   delayed_xor      - 延迟异或任务实验")
    print("   multi_timescale  - 多时间尺度XOR实验 (新增)")
    print("   core_all         - 运行所有核心实验")
    print("")
    print("🧪 创新实验:")
    print("   innovations      - 运行创新实验集合")
    print("")
    print("📱 应用实验:")
    print("   ssc              - SSC语音命令识别实验")
    print("   shd              - SHD数字识别实验")
    print("   neurovpr         - NeuroVPR视觉位置识别实验")
    print("   smnist           - Sequential MNIST序列分类实验")
    print("")
    print("🎯 批量运行:")
    print("   all              - 运行所有实验")
    print("")
    print("💡 关键创新实验:")
    print("   multi_timescale是DH-SNN的核心创新实验，")
    print("   验证模型处理多时间尺度信息的能力。")
    print("   这是区别于delayed_xor的真正多时间尺度任务。")
    print("")
    print("使用方法: python run_experiments.py <实验名称>")
    print("例如: python run_experiments.py multi_timescale")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="DH-SNN实验运行器",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
可用实验:
  🔬 核心验证:
    delayed_xor      - 延迟异或任务实验
    multi_timescale  - 多时间尺度XOR实验 (核心创新)
    core_all         - 运行所有核心实验
  
  🧪 创新扩展:
    innovations      - 运行创新实验集合
  
  📱 应用实验:
    ssc              - SSC语音命令识别实验
    shd              - SHD数字识别实验  
    neurovpr         - NeuroVPR视觉位置识别实验
    smnist           - Sequential MNIST序列分类实验
  
  🎯 批量运行:
    all              - 运行所有实验
    list             - 列出所有可用实验

重要说明:
  multi_timescale是DH-SNN的核心创新实验，验证处理多时间尺度信息的能力。
  这个实验展示了DH-SNN相比传统SNN的关键优势。

示例:
  python run_experiments.py multi_timescale   # 运行多时间尺度实验
  python run_experiments.py core_all          # 运行所有核心实验
  python run_experiments.py innovations       # 运行创新实验
  python run_experiments.py all               # 运行所有实验
        """
    )
    
    parser.add_argument(
        'experiment',
        nargs='?',
        default='list',
        help='要运行的实验名称 (默认: list)'
    )
    
    parser.add_argument(
        '--device',
        choices=['cpu', 'cuda', 'auto'],
        default='auto',
        help='指定计算设备 (默认: auto)'
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='随机种子 (默认: 42)'
    )
    
    args = parser.parse_args()
    
    # 设置设备
    if args.device != 'auto':
        os.environ['CUDA_VISIBLE_DEVICES'] = '0' if args.device == 'cuda' else ''
    
    # 设置随机种子
    os.environ['PYTHONHASHSEED'] = str(args.seed)
    
    print(f"🔧 配置:")
    print(f"   计算设备: {args.device}")
    print(f"   随机种子: {args.seed}")
    print()
    
    # 根据参数运行相应实验
    if args.experiment == 'list':
        list_experiments()
    elif args.experiment == 'delayed_xor':
        run_delayed_xor_experiment()
    elif args.experiment == 'multi_timescale':
        run_multi_timescale_experiment()
    elif args.experiment == 'core_all':
        run_core_experiments()
    elif args.experiment == 'innovations':
        run_innovation_experiments()
    elif args.experiment == 'ssc':
        run_ssc_experiment()
    elif args.experiment == 'shd':
        run_shd_experiment()
    elif args.experiment == 'neurovpr':
        run_neurovpr_experiment()
    elif args.experiment == 'smnist':
        run_smnist_experiment()
    elif args.experiment == 'all':
        run_all_experiments()
    else:
        print(f"❌ 未知实验: {args.experiment}")
        print("使用 'python run_experiments.py list' 查看可用实验")
        sys.exit(1)

if __name__ == "__main__":
    main()