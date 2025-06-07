#!/usr/bin/env python3
"""
下载MNIST数据集到datasets目录
"""

import torch
import torchvision
import torchvision.transforms as transforms
from pathlib import Path
import os

def download_mnist():
    """下载MNIST数据集"""
    
    print("📥 开始下载MNIST数据集...")
    print("="*50)
    
    # 创建数据集目录
    datasets_dir = Path("../datasets")
    datasets_dir.mkdir(exist_ok=True)
    
    mnist_dir = datasets_dir / "mnist"
    mnist_dir.mkdir(exist_ok=True)
    
    print(f"📁 数据保存路径: {mnist_dir.absolute()}")
    
    try:
        # 下载训练集
        print("\n📊 下载MNIST训练集...")
        train_dataset = torchvision.datasets.MNIST(
            root=str(mnist_dir),
            train=True,
            download=True,
            transform=transforms.ToTensor()
        )
        
        print(f"✅ 训练集下载完成: {len(train_dataset)} 样本")
        
        # 下载测试集
        print("\n📊 下载MNIST测试集...")
        test_dataset = torchvision.datasets.MNIST(
            root=str(mnist_dir),
            train=False,
            download=True,
            transform=transforms.ToTensor()
        )
        
        print(f"✅ 测试集下载完成: {len(test_dataset)} 样本")
        
        # 验证数据
        print(f"\n🔍 验证数据...")
        
        # 获取一个样本
        sample_data, sample_label = train_dataset[0]
        print(f"  样本形状: {sample_data.shape}")
        print(f"  样本标签: {sample_label}")
        print(f"  数据类型: {sample_data.dtype}")
        print(f"  数值范围: [{sample_data.min():.3f}, {sample_data.max():.3f}]")
        
        # 检查类别分布
        print(f"\n📊 训练集类别分布:")
        train_labels = [train_dataset[i][1] for i in range(min(1000, len(train_dataset)))]
        for digit in range(10):
            count = train_labels.count(digit)
            print(f"  数字 {digit}: {count} 样本")
        
        # 检查文件大小
        print(f"\n💾 检查下载的文件:")
        for file_path in mnist_dir.rglob("*"):
            if file_path.is_file():
                size_mb = file_path.stat().st_size / (1024 * 1024)
                print(f"  {file_path.name}: {size_mb:.1f} MB")
        
        print(f"\n🎉 MNIST数据集下载完成!")
        print(f"📍 数据位置: {mnist_dir.absolute()}")
        
        return True
        
    except Exception as e:
        print(f"\n❌ MNIST下载失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_sequential_mnist():
    """测试Sequential MNIST数据加载"""
    
    print(f"\n🧪 测试Sequential MNIST数据加载...")
    
    mnist_dir = Path("../datasets/mnist")
    
    if not mnist_dir.exists():
        print("❌ MNIST数据集不存在，请先下载")
        return False
    
    try:
        # 加载数据集
        train_dataset = torchvision.datasets.MNIST(
            root=str(mnist_dir),
            train=True,
            transform=transforms.ToTensor()
        )
        
        # 获取一个样本并转换为序列
        sample_data, sample_label = train_dataset[0]
        
        print(f"  原始图像形状: {sample_data.shape}")  # [1, 28, 28]
        
        # 转换为序列 (Sequential MNIST)
        # 方法1: 按行展开
        sequential_data = sample_data.view(1, 28, 28).permute(1, 0, 2).squeeze()  # [28, 28]
        print(f"  序列形状 (按行): {sequential_data.shape}")  # [28, 28] - 28个时间步，每步28个特征
        
        # 方法2: 完全展开
        flat_sequential = sample_data.view(-1)  # [784]
        sequential_steps = flat_sequential.view(784, 1)  # [784, 1] - 784个时间步，每步1个特征
        print(f"  序列形状 (完全展开): {sequential_steps.shape}")
        
        # 显示序列的前几个时间步
        print(f"  前5个时间步的数据:")
        for i in range(5):
            print(f"    时间步 {i}: {sequential_data[i][:5].tolist()}")  # 显示前5个特征
        
        print(f"✅ Sequential MNIST测试成功")
        return True
        
    except Exception as e:
        print(f"❌ Sequential MNIST测试失败: {e}")
        return False

def create_mnist_readme():
    """创建MNIST使用说明"""
    
    readme_content = """# 📊 MNIST数据集

## 📋 数据集信息
- **名称**: MNIST手写数字识别数据集
- **训练样本**: 60,000张图像
- **测试样本**: 10,000张图像
- **图像尺寸**: 28×28像素，灰度图
- **类别数**: 10 (数字0-9)

## 🔄 Sequential MNIST
Sequential MNIST将28×28的图像按行或像素顺序展开为序列：

### 方法1: 按行序列化
- **序列长度**: 28个时间步
- **每步特征**: 28个像素值
- **用途**: 测试RNN处理图像序列的能力

### 方法2: 按像素序列化  
- **序列长度**: 784个时间步
- **每步特征**: 1个像素值
- **用途**: 测试长序列记忆能力

## 🔀 Permuted MNIST
在Sequential MNIST基础上，对像素顺序进行固定的随机置换：
- **目的**: 破坏空间结构，测试纯序列建模能力
- **难度**: 比Sequential MNIST更困难

## 📁 文件结构
```
mnist/
├── MNIST/
│   └── raw/
│       ├── train-images-idx3-ubyte
│       ├── train-labels-idx1-ubyte
│       ├── t10k-images-idx3-ubyte
│       └── t10k-labels-idx1-ubyte
└── README.md (本文件)
```

## 🚀 使用示例

### 加载标准MNIST
```python
import torchvision
import torchvision.transforms as transforms

dataset = torchvision.datasets.MNIST(
    root='../datasets/mnist',
    train=True,
    transform=transforms.ToTensor()
)
```

### 转换为Sequential MNIST
```python
# 按行序列化
def to_sequential_rows(image):
    # image: [1, 28, 28]
    return image.view(1, 28, 28).permute(1, 0, 2).squeeze()  # [28, 28]

# 按像素序列化
def to_sequential_pixels(image):
    # image: [1, 28, 28] 
    return image.view(-1, 1)  # [784, 1]
```

### 创建Permuted MNIST
```python
import torch

# 创建固定的置换索引
perm_indices = torch.randperm(784)

def permute_pixels(image):
    flat_image = image.view(-1)  # [784]
    return flat_image[perm_indices].view(-1, 1)  # [784, 1]
```

## 📈 DH-SNN实验配置

### Sequential MNIST
- **网络**: 1层DH-SRNN
- **隐藏单元**: 256
- **分支数**: 1-8
- **预期性能**: Vanilla ~95%, DH-SNN ~98.87%

### Permuted MNIST  
- **网络**: 1层DH-SRNN
- **隐藏单元**: 256
- **分支数**: 1-8
- **预期性能**: Vanilla ~92%, DH-SNN ~94.52%
"""
    
    mnist_dir = Path("../datasets/mnist")
    readme_file = mnist_dir / "README.md"
    
    with open(readme_file, 'w', encoding='utf-8') as f:
        f.write(readme_content)
    
    print(f"📝 创建MNIST使用说明: {readme_file}")

def main():
    """主函数"""
    
    print("🚀 MNIST数据集下载器")
    print("="*60)
    
    # 1. 下载MNIST
    success = download_mnist()
    
    if success:
        # 2. 测试Sequential MNIST
        test_sequential_mnist()
        
        # 3. 创建使用说明
        create_mnist_readme()
        
        print(f"\n✅ 所有任务完成!")
        print(f"💡 下一步:")
        print(f"  - 可以开始Sequential MNIST实验")
        print(f"  - 可以开始Permuted MNIST实验")
        print(f"  - 数据已准备就绪，位于 ../datasets/mnist/")
    else:
        print(f"\n❌ 下载失败，请检查网络连接")

if __name__ == '__main__':
    main()
