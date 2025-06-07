#!/usr/bin/env python3
"""
DH-SRNN训练问题诊断脚本
分析为什么DH-SRNN只能达到10.1%的准确率
"""

import os
import sys
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

# 添加项目路径
sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))

from data_loader import load_sequential_mnist_data
from models import SequentialMNISTModel, DHSRNNCell, VanillaSRNNCell

def analyze_model_weights(model, model_name):
    """分析模型权重分布"""
    print(f"\n🔍 分析 {model_name} 权重分布:")
    print("=" * 50)
    
    for name, param in model.named_parameters():
        if param.requires_grad:
            weight_mean = param.data.mean().item()
            weight_std = param.data.std().item()
            weight_min = param.data.min().item()
            weight_max = param.data.max().item()
            
            print(f"📊 {name}:")
            print(f"   形状: {param.shape}")
            print(f"   均值: {weight_mean:.6f}")
            print(f"   标准差: {weight_std:.6f}")
            print(f"   范围: [{weight_min:.6f}, {weight_max:.6f}]")
            
            # 检查是否有异常值
            if abs(weight_mean) > 10 or weight_std > 10:
                print(f"   ⚠️  权重可能异常!")
            
            # 检查梯度
            if param.grad is not None:
                grad_mean = param.grad.mean().item()
                grad_std = param.grad.std().item()
                print(f"   梯度均值: {grad_mean:.6f}")
                print(f"   梯度标准差: {grad_std:.6f}")
                
                if abs(grad_mean) > 1 or grad_std > 1:
                    print(f"   ⚠️  梯度可能异常!")

def test_forward_pass(model, data_loader, device, model_name):
    """测试前向传播"""
    print(f"\n🧪 测试 {model_name} 前向传播:")
    print("=" * 50)
    
    model.eval()
    
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(data_loader):
            if batch_idx >= 3:  # 只测试前3个批次
                break
                
            data, target = data.to(device), target.to(device)
            print(f"\n批次 {batch_idx + 1}:")
            print(f"   输入形状: {data.shape}")
            print(f"   输入范围: [{data.min().item():.3f}, {data.max().item():.3f}]")
            
            # 前向传播
            try:
                output = model(data)
                print(f"   输出形状: {output.shape}")
                print(f"   输出范围: [{output.min().item():.3f}, {output.max().item():.3f}]")
                print(f"   输出均值: {output.mean().item():.6f}")
                print(f"   输出标准差: {output.std().item():.6f}")
                
                # 检查输出分布
                pred = output.argmax(dim=1)
                unique_preds, counts = torch.unique(pred, return_counts=True)
                print(f"   预测分布: {dict(zip(unique_preds.cpu().numpy(), counts.cpu().numpy()))}")
                
                # 计算准确率
                correct = pred.eq(target).sum().item()
                accuracy = 100. * correct / target.size(0)
                print(f"   批次准确率: {accuracy:.2f}%")
                
                # 检查是否所有预测都相同
                if len(unique_preds) == 1:
                    print(f"   ⚠️  所有预测都是类别 {unique_preds[0].item()}!")
                
            except Exception as e:
                print(f"   ❌ 前向传播失败: {e}")
                return False
    
    return True

def compare_models():
    """对比DH-SRNN和Vanilla SRNN"""
    print("🔬 DH-SRNN vs Vanilla SRNN 对比分析")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🔧 使用设备: {device}")
    
    # 加载数据
    print("\n📊 加载测试数据...")
    train_loader, test_loader = load_sequential_mnist_data(
        batch_size=32,
        permute=False,
        seed=42
    )
    
    # 创建模型
    print("\n🧠 创建模型...")
    
    # DH-SRNN
    dh_model = SequentialMNISTModel(
        model_type='dh_srnn',
        num_branches=4,
        tau_m_init=(0, 4),
        tau_n_init=(0, 4)
    ).to(device)
    
    # Vanilla SRNN
    vanilla_model = SequentialMNISTModel(
        model_type='vanilla_srnn',
        tau_m_init=(0, 4)
    ).to(device)
    
    # 分析模型结构
    print(f"\n📈 DH-SRNN参数数量: {sum(p.numel() for p in dh_model.parameters()):,}")
    print(f"📈 Vanilla SRNN参数数量: {sum(p.numel() for p in vanilla_model.parameters()):,}")
    
    # 分析权重分布
    analyze_model_weights(dh_model, "DH-SRNN")
    analyze_model_weights(vanilla_model, "Vanilla SRNN")
    
    # 测试前向传播
    dh_success = test_forward_pass(dh_model, test_loader, device, "DH-SRNN")
    vanilla_success = test_forward_pass(vanilla_model, test_loader, device, "Vanilla SRNN")
    
    return dh_success, vanilla_success

def analyze_dh_srnn_cell():
    """详细分析DH-SRNN单元"""
    print("\n🔬 DH-SRNN单元详细分析:")
    print("=" * 50)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 创建DH-SRNN单元
    cell = DHSRNNCell(
        input_size=1,
        hidden_size=64,
        num_branches=4,
        tau_m_init=(0, 4),
        tau_n_init=(0, 4)
    ).to(device)
    
    print(f"📊 分支输入大小: {cell.branch_input_sizes}")
    print(f"📊 总输入大小: {sum(cell.branch_input_sizes)}")
    print(f"📊 期望输入大小: {1 + 64}")  # input_size + hidden_size
    
    # 测试单步前向传播
    batch_size = 8
    input_t = torch.randn(batch_size, 1, device=device)
    
    print(f"\n🧪 测试单步前向传播:")
    print(f"   输入形状: {input_t.shape}")
    
    try:
        output = cell(input_t)
        print(f"   输出形状: {output.shape}")
        print(f"   输出范围: [{output.min().item():.3f}, {output.max().item():.3f}]")
        print(f"   输出均值: {output.mean().item():.6f}")
        print(f"   脉冲率: {output.mean().item():.3f}")
        
        # 检查分支状态
        if cell.branch_states is not None:
            print(f"   分支状态形状: {cell.branch_states.shape}")
            print(f"   分支状态范围: [{cell.branch_states.min().item():.3f}, {cell.branch_states.max().item():.3f}]")
        
    except Exception as e:
        print(f"   ❌ 前向传播失败: {e}")
        import traceback
        traceback.print_exc()

def check_training_results():
    """检查已有的训练结果"""
    print("\n📋 检查已有训练结果:")
    print("=" * 50)
    
    result_files = [
        'results/S-MNIST_dh_srnn_results.pth',
        'results/S-MNIST_vanilla_srnn_results.pth'
    ]
    
    for file_path in result_files:
        if os.path.exists(file_path):
            try:
                results = torch.load(file_path, map_location='cpu')
                model_name = os.path.basename(file_path).replace('.pth', '')
                
                print(f"\n📊 {model_name}:")
                print(f"   最佳准确率: {results.get('best_acc', 'N/A'):.2f}%")
                print(f"   最终准确率: {results['test_accs'][-1]:.2f}%")
                print(f"   训练轮数: {len(results['test_accs'])}")
                print(f"   训练时间: {results.get('total_time', 0)/3600:.2f} 小时")
                
                # 分析训练曲线
                test_accs = results['test_accs']
                if len(test_accs) > 1:
                    print(f"   初始准确率: {test_accs[0]:.2f}%")
                    print(f"   最大准确率: {max(test_accs):.2f}%")
                    print(f"   最终准确率: {test_accs[-1]:.2f}%")
                    
                    # 检查是否有学习
                    if max(test_accs) - test_accs[0] < 1.0:
                        print(f"   ⚠️  模型几乎没有学习!")
                
            except Exception as e:
                print(f"❌ 加载 {file_path} 失败: {e}")
        else:
            print(f"⚠️  文件不存在: {file_path}")

def main():
    """主函数"""
    print("🚨 DH-SRNN训练问题诊断")
    print("=" * 60)
    
    # 检查已有结果
    check_training_results()
    
    # 分析DH-SRNN单元
    analyze_dh_srnn_cell()
    
    # 对比模型
    dh_success, vanilla_success = compare_models()
    
    print(f"\n📋 诊断总结:")
    print("=" * 30)
    print(f"DH-SRNN前向传播: {'✅ 成功' if dh_success else '❌ 失败'}")
    print(f"Vanilla SRNN前向传播: {'✅ 成功' if vanilla_success else '❌ 失败'}")
    
    if not dh_success:
        print("\n🔧 建议修复措施:")
        print("1. 检查DH-SRNN单元的分支连接逻辑")
        print("2. 验证分支状态更新机制")
        print("3. 检查梯度流动是否正常")
        print("4. 调整初始化参数")
    
    print(f"\n💡 可能的问题原因:")
    print("1. 分支连接逻辑错误")
    print("2. 时间常数初始化不当")
    print("3. 梯度消失或爆炸")
    print("4. 学习率设置问题")
    print("5. 模型架构与原论文不一致")

if __name__ == "__main__":
    main()
