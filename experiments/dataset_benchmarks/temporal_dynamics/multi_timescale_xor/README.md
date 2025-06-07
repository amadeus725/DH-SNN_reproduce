# 🕰️ 多时间尺度实验

## 📋 概述

本目录包含DH-SNN论文中多时间尺度相关的所有实验，主要对应论文的Figure 4。

## 🏗️ 目录结构

```
multi_timescale_experiments/
├── delayed_xor/                    # 延迟XOR实验 (Figure 3)
│   ├── config_spikingjelly.py     # SpikingJelly配置
│   ├── sequence_length_experiment_fixed.py
│   ├── test_spikingjelly_minimal.py
│   └── train_spikingjelly_dh_snn.py
├── multitimescale_xor/            # 多时间尺度XOR实验 (Figure 4)
│   ├── SNN_layers/                # 原论文神经元层
│   └── multi_xor_snn.py          # 主实验脚本
├── spikingjelly_multi_timescale/  # SpikingJelly多时间尺度实现
│   ├── data_generator.py         # 数据生成器
│   ├── models.py                 # 模型定义
│   ├── experiment.py             # 实验脚本
│   └── visualization.py          # 可视化工具
└── README.md                      # 本文件
```

## 🎯 实验内容

### 1. 延迟XOR实验 (Figure 3)
- **目标**: 验证DH-SNN的长期记忆能力
- **任务**: 记住第一个输入信号，与延迟后的第二个信号进行XOR运算
- **关键发现**: DH-SNN在长延迟任务中显著优于Vanilla SNN

### 2. 多时间尺度XOR实验 (Figure 4a-4e)
- **目标**: 验证DH-SNN处理多时间尺度信息的能力
- **任务**: 
  - Signal 1: 低频信号（长期记忆）
  - Signal 2: 高频信号序列（快速响应）
  - 目标: 记住Signal 1并与每个Signal 2进行XOR运算

### 3. 分支数量对比实验 (Figure 4b)
- **对比模型**:
  - Vanilla SFNN
  - 1-Branch DH-SFNN (Small/Large初始化)
  - 2-Branch DH-SFNN (有益初始化/固定时间常数)
  - 4-Branch DH-SFNN
- **关键发现**: 分支数量增加提升性能，有益初始化很重要

### 4. 时间因子分析 (Figure 4c)
- **分析内容**: 训练前后时间因子分布变化
- **关键发现**: 
  - Branch 1学习大时间常数（长期记忆）
  - Branch 2学习小时间常数（快速响应）

### 5. 神经元活动可视化 (Figure 4d-4e)
- **可视化内容**:
  - 输出脉冲模式
  - 树突电流演化
  - 不同分支的功能分化
- **关键发现**: 不同分支学习处理不同时间尺度的信息

## 🚀 快速开始

### 1. 运行延迟XOR实验
```bash
cd delayed_xor
python train_spikingjelly_dh_snn.py
```

### 2. 运行多时间尺度XOR实验
```bash
cd spikingjelly_multi_timescale
python experiment.py
```

### 3. 生成可视化
```bash
cd spikingjelly_multi_timescale
python visualization.py
```

## 📊 实验配置

### 网络架构
- **输入维度**: 100
- **隐藏层大小**: 64
- **输出维度**: 1
- **分支数量**: 1, 2, 4

### 时间因子初始化
- **Small**: τ ~ U(-4, 0)
- **Medium**: τ ~ U(0, 4)  
- **Large**: τ ~ U(2, 6)
- **有益初始化**: Branch1=Large, Branch2=Small

### 训练参数
- **学习率**: 1e-3
- **批次大小**: 32
- **训练轮数**: 200
- **重复次数**: 10次

## 📈 预期结果

### 性能对比 (准确率)
- **Vanilla SFNN**: ~60%
- **1-Branch DH-SFNN**: ~65%
- **2-Branch DH-SFNN (有益初始化)**: ~85%
- **2-Branch DH-SFNN (固定时间常数)**: ~75%
- **4-Branch DH-SFNN**: ~90%

### 关键发现
1. ✅ **分支数量重要**: 更多分支 → 更好性能
2. ✅ **有益初始化关键**: 正确的时间常数初始化显著提升性能
3. ✅ **时间常数可学习**: 训练过程中自动分化为不同时间尺度
4. ✅ **功能分化**: 不同分支学习处理不同频率的信号

## 🔬 技术细节

### 多时间尺度数据生成
```python
# Signal 1: 低频信号 (时间步 20-120)
signal1_type = random.choice(['low', 'high'])
input_data[20:120, :50] = generate_pattern(signal1_type, low_freq=True)

# Signal 2序列: 高频信号 (时间步 150, 250, 350, ...)
for i, start_time in enumerate([150, 250, 350, 450, 550]):
    signal2_type = random.choice(['low', 'high'])
    input_data[start_time:start_time+30, 50:] = generate_pattern(signal2_type, high_freq=True)
    
    # XOR目标
    xor_result = signal1_type ^ signal2_type
    target_data[start_time+30:start_time+50] = xor_result
```

### 双分支DH-LIF神经元
```python
# 分支1: 长期记忆 (大时间常数)
beta1 = sigmoid(tau_n_branch1)  # tau_n_branch1 ~ U(2, 6)
d1_current = beta1 * d1_current + (1-beta1) * branch1_input

# 分支2: 快速响应 (小时间常数)  
beta2 = sigmoid(tau_n_branch2)  # tau_n_branch2 ~ U(-4, 0)
d2_current = beta2 * d2_current + (1-beta2) * branch2_input

# 整合
total_input = d1_current + d2_current
```

## 📚 参考文献

1. 原论文: "Temporal dendritic heterogeneity incorporated with spiking neural networks for learning multi-timescale dynamics"
2. SpikingJelly框架: https://github.com/fangwei123456/spikingjelly
3. 相关工作: 多时间尺度神经计算理论

## 🎯 下一步计划

1. **完成Figure 4b实验**: 分支数量对比的完整实验
2. **实现Figure 4c**: 时间因子分布分析和可视化
3. **实现Figure 4d-4e**: 神经元活动和树突电流可视化
4. **性能优化**: 提升训练速度和内存效率
5. **扩展实验**: 更多分支数量、不同网络架构的测试
