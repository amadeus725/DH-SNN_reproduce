#!/usr/bin/env python3
"""
运行分支数量对比实验的简化脚本
"""

import sys
import os
sys.path.append('/root/DH-SNN_reproduce')

from branch_comparison_experiment import BranchComparisonExperiment

def main():
    """运行分支数量对比实验"""
    
    print("🚀 启动分支数量对比实验")
    
    # 简化的实验配置
    config = {
        'branch_numbers': [1, 2, 4],  # 减少分支数量以加快实验
        'num_trials': 2,  # 减少试验次数
        
        # 模型参数
        'input_dim': 2,
        'hidden_dims': [16],
        'output_dim': 2,
        'v_threshold': 1.0,
        'tau_m_init': (0.0, 4.0),
        'tau_n_init': (0.0, 4.0),
        
        # 训练参数
        'batch_size': 100,
        'learning_rate': 1e-2,
        'num_epochs': 30,  # 进一步减少轮次
        'weight_decay': 1e-4,
        'lr_step_size': 15,
        'lr_gamma': 0.5,
        
        # 数据参数
        'train_samples': 1000,
        'test_samples': 200,
        
        # 其他设置
        'save_dir': 'experiments/branch_number_comparison/results'
    }
    
    try:
        # 创建实验对象
        experiment = BranchComparisonExperiment(config)
        
        # 运行实验
        print("📊 开始运行实验...")
        results = experiment.run_all_experiments()
        
        # 创建可视化
        print("🎨 创建可视化...")
        experiment.create_visualization()
        
        # 打印结果摘要
        print("\n" + "="*50)
        print("🎯 实验完成!")
        print("="*50)
        
        for result in results:
            print(f"📊 {result['num_branches']}分支: "
                  f"{result['mean_accuracy']:.2f}% ± {result['std_accuracy']:.2f}%")
        
        print("="*50)
        print("✅ 所有结果已保存到:", config['save_dir'])
        
    except Exception as e:
        print(f"❌ 实验失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
