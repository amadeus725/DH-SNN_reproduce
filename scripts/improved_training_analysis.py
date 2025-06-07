#!/usr/bin/env python3
"""
改进的训练分析 - 检查训练是否真正有效
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from final_challenging_experiment import UltraChallengingXORGenerator
from run_final_experiment import FinalVanillaSFNN, FinalTwoBranchDH_SFNN

def analyze_training_effectiveness():
    """分析训练有效性"""
    print("🔍 分析训练有效性...")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 生成数据
    generator = UltraChallengingXORGenerator(device=device)
    train_data, train_targets = generator.generate_dataset(100)
    test_data, test_targets = generator.generate_dataset(50)
    
    print(f"数据形状: 训练{train_data.shape}, 测试{test_data.shape}")
    
    # 分析数据特征
    print(f"\n📊 数据分析:")
    print(f"训练数据发放率: {train_data.mean():.4f}")
    print(f"测试数据发放率: {test_data.mean():.4f}")
    
    # 计算有效标签的比例
    train_valid_labels = (train_targets >= 0).sum().item()
    test_valid_labels = (test_targets >= 0).sum().item()
    
    print(f"训练集有效标签: {train_valid_labels}/{train_targets.numel()} ({train_valid_labels/train_targets.numel()*100:.1f}%)")
    print(f"测试集有效标签: {test_valid_labels}/{test_targets.numel()} ({test_valid_labels/test_targets.numel()*100:.1f}%)")
    
    # 分析标签分布
    train_labels = train_targets[train_targets >= 0]
    test_labels = test_targets[test_targets >= 0]
    
    if len(train_labels) > 0:
        train_class_0 = (train_labels == 0).sum().item()
        train_class_1 = (train_labels == 1).sum().item()
        print(f"训练集类别分布: 0={train_class_0}, 1={train_class_1} (平衡度: {min(train_class_0, train_class_1)/max(train_class_0, train_class_1):.3f})")
    
    if len(test_labels) > 0:
        test_class_0 = (test_labels == 0).sum().item()
        test_class_1 = (test_labels == 1).sum().item()
        print(f"测试集类别分布: 0={test_class_0}, 1={test_class_1} (平衡度: {min(test_class_0, test_class_1)/max(test_class_0, test_class_1):.3f})")
    
    return train_data, train_targets, test_data, test_targets

def test_random_baseline():
    """测试随机基线性能"""
    print(f"\n🎲 测试随机基线性能...")
    
    generator = UltraChallengingXORGenerator()
    test_data, test_targets = generator.generate_dataset(100)
    
    # 随机预测
    correct_random = 0
    total_predictions = 0
    
    for b in range(test_data.size(0)):
        for t in range(test_data.size(1)):
            if test_targets[b, t] >= 0:
                random_pred = np.random.randint(0, 2)
                actual = test_targets[b, t].item()
                if random_pred == actual:
                    correct_random += 1
                total_predictions += 1
    
    random_acc = correct_random / total_predictions * 100 if total_predictions > 0 else 0
    print(f"随机基线准确率: {random_acc:.1f}%")
    
    # 多数类基线
    test_labels = test_targets[test_targets >= 0]
    if len(test_labels) > 0:
        majority_class = 1 if (test_labels == 1).sum() > (test_labels == 0).sum() else 0
        majority_acc = (test_labels == majority_class).float().mean().item() * 100
        print(f"多数类基线准确率: {majority_acc:.1f}%")
    
    return random_acc

def improved_training_experiment():
    """改进的训练实验"""
    print(f"\n🚀 改进的训练实验...")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 生成更多数据
    generator = UltraChallengingXORGenerator(device=device)
    train_data, train_targets = generator.generate_dataset(500)  # 更多训练数据
    test_data, test_targets = generator.generate_dataset(100)
    
    # 测试Vanilla SFNN
    print(f"\n🧪 测试改进的Vanilla SFNN训练...")
    model = FinalVanillaSFNN().to(device)
    
    # 改进的训练配置
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4, weight_decay=1e-4)  # 更小学习率
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
    criterion = nn.CrossEntropyLoss()
    
    train_losses = []
    train_accs = []
    test_accs = []
    
    best_test_acc = 0
    patience_counter = 0
    max_patience = 20
    
    for epoch in range(100):
        model.train()
        total_loss = 0
        train_correct = 0
        train_total = 0
        num_batches = 0
        
        # 训练
        batch_size = 16
        for i in range(0, len(train_data), batch_size):
            batch_data = train_data[i:i+batch_size].to(device)
            batch_targets = train_targets[i:i+batch_size].to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_data)
            
            # 计算损失和准确率
            loss_total = 0
            loss_count = 0
            
            for b in range(batch_data.size(0)):
                for t in range(batch_data.size(1)):
                    if batch_targets[b, t] >= 0:
                        output_t = outputs[b, t, 0]
                        target_t = batch_targets[b, t].float()
                        
                        # 二分类损失
                        output_prob = torch.stack([1-output_t, output_t])
                        target_class = batch_targets[b, t].long()
                        
                        loss_total += criterion(output_prob.unsqueeze(0), target_class.unsqueeze(0))
                        loss_count += 1
                        
                        # 训练准确率
                        pred = (output_t > 0.5).float()
                        if pred == target_t:
                            train_correct += 1
                        train_total += 1
            
            if loss_count > 0:
                loss = loss_total / loss_count
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)  # 更小的梯度裁剪
                optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
        
        # 测试
        model.eval()
        with torch.no_grad():
            test_outputs = model(test_data.to(device))
            test_correct = 0
            test_total = 0
            
            for b in range(test_data.size(0)):
                for t in range(test_data.size(1)):
                    if test_targets[b, t] >= 0:
                        pred = (test_outputs[b, t, 0] > 0.5).float()
                        target = test_targets[b, t].float()
                        
                        if pred == target:
                            test_correct += 1
                        test_total += 1
        
        # 计算指标
        avg_loss = total_loss / max(num_batches, 1)
        train_acc = train_correct / train_total * 100 if train_total > 0 else 0
        test_acc = test_correct / test_total * 100 if test_total > 0 else 0
        
        train_losses.append(avg_loss)
        train_accs.append(train_acc)
        test_accs.append(test_acc)
        
        # 学习率调度
        scheduler.step(avg_loss)
        
        # 早停
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            patience_counter = 0
        else:
            patience_counter += 1
        
        if epoch % 10 == 0 or epoch < 10:
            print(f"Epoch {epoch+1:3d}: Loss={avg_loss:.4f}, Train_Acc={train_acc:.1f}%, Test_Acc={test_acc:.1f}%, Best={best_test_acc:.1f}%")
        
        # 早停检查
        if patience_counter >= max_patience:
            print(f"早停在epoch {epoch+1}, 最佳测试准确率: {best_test_acc:.1f}%")
            break
    
    return {
        'best_test_acc': best_test_acc,
        'train_losses': train_losses,
        'train_accs': train_accs,
        'test_accs': test_accs
    }

def plot_training_curves(results):
    """绘制训练曲线"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    epochs = range(1, len(results['train_losses']) + 1)
    
    # 损失曲线
    axes[0].plot(epochs, results['train_losses'], 'b-', label='Training Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training Loss')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    
    # 准确率曲线
    axes[1].plot(epochs, results['train_accs'], 'b-', label='Training Accuracy')
    axes[1].plot(epochs, results['test_accs'], 'r-', label='Test Accuracy')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].set_title('Training vs Test Accuracy')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    
    # 过拟合分析
    overfitting = np.array(results['train_accs']) - np.array(results['test_accs'])
    axes[2].plot(epochs, overfitting, 'g-', label='Train - Test Acc')
    axes[2].axhline(y=0, color='k', linestyle='--', alpha=0.5)
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('Accuracy Difference (%)')
    axes[2].set_title('Overfitting Analysis')
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()
    
    plt.tight_layout()
    plt.savefig('results/training_analysis.png', dpi=300, bbox_inches='tight')
    print("✅ 训练分析图已保存到 results/training_analysis.png")
    
    return fig

def main():
    """主函数"""
    print("🔍 训练有效性分析")
    print("="*50)
    
    # 分析数据
    train_data, train_targets, test_data, test_targets = analyze_training_effectiveness()
    
    # 测试随机基线
    random_acc = test_random_baseline()
    
    # 改进的训练实验
    results = improved_training_experiment()
    
    # 绘制训练曲线
    plot_training_curves(results)
    
    print(f"\n📊 训练有效性总结:")
    print(f"随机基线: {random_acc:.1f}%")
    print(f"改进训练最佳结果: {results['best_test_acc']:.1f}%")
    print(f"相对于随机基线提升: +{results['best_test_acc'] - random_acc:.1f}%")
    
    if results['best_test_acc'] - random_acc > 10:
        print("✅ 训练有效，模型确实在学习")
    elif results['best_test_acc'] - random_acc > 5:
        print("⚠️ 训练效果一般，可能需要调整")
    else:
        print("❌ 训练效果差，模型可能没有真正学习")

if __name__ == '__main__':
    main()
