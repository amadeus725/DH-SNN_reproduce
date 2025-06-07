#!/usr/bin/env python3
"""
分析现有实验结果，确定补充实验的优先级
"""

import os
import glob
import torch
import json
import pandas as pd
from pathlib import Path

def find_all_results():
    """查找所有实验结果文件"""
    base_dir = "/root/DH-SNN_reproduce"
    
    # 查找所有可能的结果文件
    result_patterns = [
        "**/*results*.pth",
        "**/*results*.json", 
        "**/*results*.pkl",
        "**/best_*.pth",
        "**/final_*.pth"
    ]
    
    all_results = []
    for pattern in result_patterns:
        files = glob.glob(os.path.join(base_dir, pattern), recursive=True)
        all_results.extend(files)
    
    return all_results

def analyze_result_file(file_path):
    """分析单个结果文件"""
    try:
        if file_path.endswith('.pth'):
            data = torch.load(file_path, map_location='cpu')
        elif file_path.endswith('.json'):
            with open(file_path, 'r') as f:
                data = json.load(f)
        else:
            return None
        
        # 提取关键信息
        info = {
            'file_path': file_path,
            'experiment_type': extract_experiment_type(file_path),
            'model_type': data.get('model_type', 'unknown'),
            'accuracy': extract_accuracy(data),
            'dataset': extract_dataset(file_path),
            'status': 'completed' if 'best_accuracy' in data or 'accuracy' in data else 'incomplete'
        }
        
        return info
        
    except Exception as e:
        print(f"Error analyzing {file_path}: {e}")
        return None

def extract_experiment_type(file_path):
    """从文件路径提取实验类型"""
    path_parts = file_path.split('/')
    
    if 'figure3' in file_path or 'delayed_xor' in file_path:
        return 'Delayed XOR'
    elif 'figure4' in file_path or 'multitimescale' in file_path:
        return 'Multi-timescale XOR'
    elif 'shd' in file_path.lower():
        return 'SHD'
    elif 'ssc' in file_path.lower():
        return 'SSC'
    elif 'sequential_mnist' in file_path:
        return 'Sequential MNIST'
    elif 'permuted_mnist' in file_path:
        return 'Permuted MNIST'
    elif 'gsc' in file_path.lower():
        return 'GSC'
    else:
        return 'Unknown'

def extract_dataset(file_path):
    """从文件路径提取数据集名称"""
    if 'xor' in file_path.lower():
        return 'Synthetic XOR'
    elif 'shd' in file_path.lower():
        return 'SHD'
    elif 'ssc' in file_path.lower():
        return 'SSC'
    elif 'mnist' in file_path.lower():
        return 'MNIST'
    elif 'gsc' in file_path.lower():
        return 'GSC'
    else:
        return 'Unknown'

def extract_accuracy(data):
    """提取准确率信息"""
    if isinstance(data, dict):
        # 尝试不同的键名
        for key in ['best_accuracy', 'accuracy', 'test_accuracy', 'final_accuracy']:
            if key in data:
                acc = data[key]
                if isinstance(acc, (int, float)):
                    return float(acc)
                elif isinstance(acc, str) and '%' in acc:
                    return float(acc.replace('%', '')) / 100
        
        # 如果是训练历史，取最后一个值
        if 'train_history' in data:
            history = data['train_history']
            if isinstance(history, list) and len(history) > 0:
                last_epoch = history[-1]
                if isinstance(last_epoch, dict) and 'test_accuracy' in last_epoch:
                    return float(last_epoch['test_accuracy'])
    
    return None

def generate_experiment_summary():
    """生成实验总结"""
    print("🔍 分析现有实验结果...")
    
    # 查找所有结果文件
    result_files = find_all_results()
    print(f"📁 找到 {len(result_files)} 个结果文件")
    
    # 分析每个文件
    experiments = []
    for file_path in result_files:
        info = analyze_result_file(file_path)
        if info:
            experiments.append(info)
    
    # 按实验类型分组
    experiment_groups = {}
    for exp in experiments:
        exp_type = exp['experiment_type']
        if exp_type not in experiment_groups:
            experiment_groups[exp_type] = []
        experiment_groups[exp_type].append(exp)
    
    print(f"\n📊 实验结果总结:")
    print("=" * 60)
    
    completed_experiments = []
    for exp_type, exps in experiment_groups.items():
        print(f"\n🧪 {exp_type}:")
        
        # 按模型类型分组
        model_results = {}
        for exp in exps:
            model_type = exp['model_type']
            if model_type not in model_results:
                model_results[model_type] = []
            model_results[model_type].append(exp)
        
        for model_type, results in model_results.items():
            accuracies = [r['accuracy'] for r in results if r['accuracy'] is not None]
            if accuracies:
                avg_acc = sum(accuracies) / len(accuracies)
                print(f"  • {model_type}: {avg_acc:.3f} ({len(accuracies)} runs)")
                
                if exp_type not in [r['experiment'] for r in completed_experiments]:
                    completed_experiments.append({
                        'experiment': exp_type,
                        'dataset': results[0]['dataset'],
                        'models': list(model_results.keys()),
                        'best_accuracy': max(accuracies),
                        'status': 'completed'
                    })
            else:
                print(f"  • {model_type}: No valid results")
    
    return completed_experiments

def identify_missing_experiments():
    """识别缺失的实验"""
    
    # 原论文中的关键实验
    paper_experiments = [
        {'name': 'Delayed XOR', 'dataset': 'Synthetic', 'priority': 1, 'paper_result': '75.4%'},
        {'name': 'Multi-timescale XOR', 'dataset': 'Synthetic', 'priority': 1, 'paper_result': '96.2%'},
        {'name': 'SHD', 'dataset': 'SHD', 'priority': 1, 'paper_result': '91.34%'},
        {'name': 'SSC', 'dataset': 'SSC', 'priority': 1, 'paper_result': '79.64%'},
        {'name': 'Sequential MNIST', 'dataset': 'MNIST', 'priority': 2, 'paper_result': '98.7%'},
        {'name': 'Permuted MNIST', 'dataset': 'MNIST', 'priority': 2, 'paper_result': '95.3%'},
        {'name': 'GSC', 'dataset': 'GSC', 'priority': 2, 'paper_result': '95.1%'},
        {'name': 'TIMIT', 'dataset': 'TIMIT', 'priority': 3, 'paper_result': '78.9%'},
        {'name': 'DEAP', 'dataset': 'DEAP', 'priority': 3, 'paper_result': '89.2%'},
        {'name': 'NeuroVPR', 'dataset': 'NeuroVPR', 'priority': 3, 'paper_result': '92.4%'}
    ]
    
    # 获取已完成的实验
    completed = generate_experiment_summary()
    completed_names = [exp['experiment'] for exp in completed]
    
    print(f"\n🎯 缺失实验分析:")
    print("=" * 60)
    
    missing_experiments = []
    for paper_exp in paper_experiments:
        if paper_exp['name'] not in completed_names:
            missing_experiments.append(paper_exp)
            status = "❌ 缺失"
        else:
            status = "✅ 已完成"
        
        print(f"{status} {paper_exp['name']} ({paper_exp['dataset']}) - 论文结果: {paper_exp['paper_result']}")
    
    return missing_experiments

def generate_priority_recommendations():
    """生成优先级推荐"""
    missing = identify_missing_experiments()
    
    print(f"\n🚀 补充实验推荐:")
    print("=" * 60)
    
    # 按优先级分组
    priority_groups = {1: [], 2: [], 3: []}
    for exp in missing:
        priority_groups[exp['priority']].append(exp)
    
    for priority in [1, 2, 3]:
        if priority_groups[priority]:
            print(f"\n🎯 优先级 {priority}:")
            for exp in priority_groups[priority]:
                print(f"  • {exp['name']} ({exp['dataset']}) - 目标: {exp['paper_result']}")
    
    # 推荐立即执行的实验
    immediate_recommendations = []
    for exp in missing:
        if exp['priority'] <= 2 and exp['dataset'] in ['MNIST', 'Synthetic']:
            immediate_recommendations.append(exp)
    
    if immediate_recommendations:
        print(f"\n⚡ 立即推荐执行:")
        for exp in immediate_recommendations:
            print(f"  🔥 {exp['name']} - 数据易获取，快速验证")
    
    return missing

def main():
    """主函数"""
    print("🔬 DH-SNN实验结果分析与补充计划")
    print("=" * 80)
    
    # 分析现有结果
    completed = generate_experiment_summary()
    
    # 识别缺失实验
    missing = generate_priority_recommendations()
    
    # 生成总结报告
    print(f"\n📋 总结:")
    print(f"  ✅ 已完成实验: {len(completed)}")
    print(f"  ❌ 缺失实验: {len(missing)}")
    print(f"  📈 完成度: {len(completed)/(len(completed)+len(missing))*100:.1f}%")
    
    # 保存详细报告
    report = {
        'completed_experiments': completed,
        'missing_experiments': missing,
        'completion_rate': len(completed)/(len(completed)+len(missing))
    }
    
    with open('/root/DH-SNN_reproduce/experiment_analysis_report.json', 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\n💾 详细报告已保存到: experiment_analysis_report.json")

if __name__ == "__main__":
    main()
