#!/usr/bin/env python3
"""
GSC训练监控脚本
实时监控训练进度和性能
"""

import os
import time
import json
import glob
from datetime import datetime
import matplotlib.pyplot as plt

def get_latest_log_file():
    """获取最新的训练日志文件"""
    log_pattern = "/root/DH-SNN_reproduce/results/gsc_training_log_*.txt"
    log_files = glob.glob(log_pattern)
    if log_files:
        return max(log_files, key=os.path.getctime)
    return None

def parse_training_progress(log_file):
    """解析训练进度"""
    if not os.path.exists(log_file):
        return None
    
    progress = {
        'current_model': None,
        'current_epoch': 0,
        'total_epochs': 150,
        'vanilla_snn': {'epochs': [], 'train_acc': [], 'valid_acc': [], 'train_loss': []},
        'dh_snn': {'epochs': [], 'train_acc': [], 'valid_acc': [], 'train_loss': []},
        'last_update': None
    }
    
    try:
        with open(log_file, 'r') as f:
            lines = f.readlines()
        
        current_model = None
        
        for line in lines:
            line = line.strip()
            
            # 检测当前训练的模型
            if "🔬 实验: Vanilla SNN" in line:
                current_model = 'vanilla_snn'
                progress['current_model'] = 'Vanilla SNN'
            elif "🔬 实验: DH-SNN" in line:
                current_model = 'dh_snn'
                progress['current_model'] = 'DH-SNN'
            
            # 解析epoch结果
            if "Epoch" in line and "Train Loss:" in line and current_model:
                try:
                    parts = line.split(',')
                    epoch_part = parts[0].split('/')[0].split()[-1]
                    epoch = int(epoch_part)
                    
                    train_loss = float([p for p in parts if 'Train Loss:' in p][0].split(':')[1].strip())
                    train_acc = float([p for p in parts if 'Train Acc:' in p][0].split(':')[1].strip())
                    valid_acc = float([p for p in parts if 'Valid Acc:' in p][0].split(':')[1].strip())
                    
                    progress[current_model]['epochs'].append(epoch)
                    progress[current_model]['train_loss'].append(train_loss)
                    progress[current_model]['train_acc'].append(train_acc)
                    progress[current_model]['valid_acc'].append(valid_acc)
                    
                    progress['current_epoch'] = epoch
                    
                except (ValueError, IndexError):
                    continue
        
        progress['last_update'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        return progress
        
    except Exception as e:
        print(f"解析日志文件时出错: {e}")
        return None

def display_progress(progress):
    """显示训练进度"""
    if not progress:
        print("❌ 无法获取训练进度")
        return
    
    print("\n" + "="*60)
    print(f"🔍 GSC训练监控 - {progress['last_update']}")
    print("="*60)
    
    if progress['current_model']:
        print(f"🔬 当前训练模型: {progress['current_model']}")
        print(f"📊 当前进度: Epoch {progress['current_epoch']}/{progress['total_epochs']}")
        
        # 计算进度百分比
        if progress['current_model'] == 'Vanilla SNN':
            total_progress = progress['current_epoch'] / progress['total_epochs'] * 50  # Vanilla SNN占50%
        else:
            vanilla_epochs = len(progress['vanilla_snn']['epochs'])
            dh_epochs = progress['current_epoch']
            total_progress = 50 + (dh_epochs / progress['total_epochs'] * 50)  # DH-SNN占50%
        
        print(f"🚀 总体进度: {total_progress:.1f}%")
    
    # 显示Vanilla SNN结果
    if progress['vanilla_snn']['epochs']:
        vanilla_data = progress['vanilla_snn']
        latest_epoch = vanilla_data['epochs'][-1]
        latest_train_acc = vanilla_data['train_acc'][-1]
        latest_valid_acc = vanilla_data['valid_acc'][-1]
        latest_train_loss = vanilla_data['train_loss'][-1]
        best_valid_acc = max(vanilla_data['valid_acc'])
        
        print(f"\n📈 Vanilla SNN (已完成 {len(vanilla_data['epochs'])} epochs):")
        print(f"   最新结果 (Epoch {latest_epoch}):")
        print(f"     训练损失: {latest_train_loss:.4f}")
        print(f"     训练准确率: {latest_train_acc:.1%}")
        print(f"     验证准确率: {latest_valid_acc:.1%}")
        print(f"   最佳验证准确率: {best_valid_acc:.1%}")
    
    # 显示DH-SNN结果
    if progress['dh_snn']['epochs']:
        dh_data = progress['dh_snn']
        latest_epoch = dh_data['epochs'][-1]
        latest_train_acc = dh_data['train_acc'][-1]
        latest_valid_acc = dh_data['valid_acc'][-1]
        latest_train_loss = dh_data['train_loss'][-1]
        best_valid_acc = max(dh_data['valid_acc'])
        
        print(f"\n📈 DH-SNN (已完成 {len(dh_data['epochs'])} epochs):")
        print(f"   最新结果 (Epoch {latest_epoch}):")
        print(f"     训练损失: {latest_train_loss:.4f}")
        print(f"     训练准确率: {latest_train_acc:.1%}")
        print(f"     验证准确率: {latest_valid_acc:.1%}")
        print(f"   最佳验证准确率: {best_valid_acc:.1%}")
    
    # 对比结果
    if progress['vanilla_snn']['epochs'] and progress['dh_snn']['epochs']:
        vanilla_best = max(progress['vanilla_snn']['valid_acc'])
        dh_best = max(progress['dh_snn']['valid_acc'])
        improvement = dh_best - vanilla_best
        print(f"\n🚀 性能对比:")
        print(f"   Vanilla SNN最佳: {vanilla_best:.1%}")
        print(f"   DH-SNN最佳: {dh_best:.1%}")
        print(f"   DH-SNN改进: +{improvement:.1%}")

def check_saved_models():
    """检查已保存的模型"""
    results_dir = "/root/DH-SNN_reproduce/results"
    
    print(f"\n💾 已保存的模型:")
    
    # 检查最佳模型
    best_models = glob.glob(os.path.join(results_dir, "*_best.pth"))
    for model_path in best_models:
        model_name = os.path.basename(model_path).replace('_best.pth', '')
        file_size = os.path.getsize(model_path) / (1024*1024)  # MB
        mod_time = datetime.fromtimestamp(os.path.getmtime(model_path))
        print(f"   ✅ {model_name}: {file_size:.1f}MB (更新: {mod_time.strftime('%H:%M:%S')})")
    
    # 检查进度模型
    progress_models = glob.glob(os.path.join(results_dir, "*_epoch_*.pth"))
    if progress_models:
        latest_progress = max(progress_models, key=os.path.getctime)
        epoch_num = latest_progress.split('_epoch_')[1].split('.')[0]
        print(f"   📊 最新进度保存: Epoch {epoch_num}")

def main():
    """主监控循环"""
    print("🔍 GSC训练监控器启动")
    print("按 Ctrl+C 退出监控")
    
    try:
        while True:
            # 清屏
            os.system('clear' if os.name == 'posix' else 'cls')
            
            # 获取最新日志文件
            log_file = get_latest_log_file()
            
            if log_file:
                # 解析并显示进度
                progress = parse_training_progress(log_file)
                display_progress(progress)
                
                # 检查保存的模型
                check_saved_models()
                
                print(f"\n📝 日志文件: {os.path.basename(log_file)}")
            else:
                print("❌ 未找到训练日志文件")
            
            print(f"\n⏰ 下次更新: 30秒后 (当前时间: {datetime.now().strftime('%H:%M:%S')})")
            print("按 Ctrl+C 退出监控")
            
            # 等待30秒
            time.sleep(30)
            
    except KeyboardInterrupt:
        print("\n👋 监控器已退出")

if __name__ == "__main__":
    main()
