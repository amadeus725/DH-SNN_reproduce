# Sequential MNIST 实验

基于原论文配置的Sequential MNIST (S-MNIST) 和 Permuted Sequential MNIST (PS-MNIST) 实验，使用SpikingJelly框架实现。

## 📋 实验概述

### 任务描述
- **S-MNIST**: 将28×28的MNIST图像按行展开为784步的序列，测试模型的长期记忆能力
- **PS-MNIST**: 在S-MNIST基础上随机置换像素顺序，增加记忆和分类难度

### 网络架构
根据论文Table S2配置：
- **网络结构**: `1-r64-r256-10` (输入维度1，DH-SRNN 64神经元，DH-SRNN 256神经元，输出10类)
- **序列长度**: 784 (28×28像素序列化)
- **输入格式**: 实值输入(非脉冲)，第一层作为编码层
- **输出解码**: 脉冲计数解码

### 训练配置
- **学习率**: 1e-2，每50个epoch衰减0.1
- **批次大小**: 128
- **训练轮数**: 150
- **优化器**: Adam
- **损失函数**: CrossEntropy

## 🚀 快速开始

### 1. 运行完整实验套件
```bash
cd experiments/dataset_benchmarks/sequential_mnist/
python run_experiments.py
```

这将自动运行以下4个实验：
1. S-MNIST + Vanilla SRNN (基线)
2. S-MNIST + DH-SRNN
3. PS-MNIST + Vanilla SRNN (基线)
4. PS-MNIST + DH-SRNN

### 2. 单独运行实验

#### DH-SRNN实验
```bash
# S-MNIST
python main_dh_srnn.py --model_type dh_srnn --num_branches 4

# PS-MNIST
python main_dh_srnn.py --model_type dh_srnn --num_branches 4 --permute
```

#### Vanilla SRNN基线
```bash
# S-MNIST
python main_vanilla_srnn.py

# PS-MNIST
python main_vanilla_srnn.py --permute
```

### 3. 结果可视化
```bash
python visualize_results.py
```

## 📊 预期结果

根据论文报告，预期性能：
- **S-MNIST**: DH-SRNN应显著优于Vanilla SRNN
- **PS-MNIST**: DH-SRNN应显著优于Vanilla SRNN，且PS-MNIST整体难度高于S-MNIST

## 📚 参考

- 原论文代码: `original_paper_code/sequential_mnist/`
- SpikingJelly文档: https://github.com/fangwei123456/spikingjelly
