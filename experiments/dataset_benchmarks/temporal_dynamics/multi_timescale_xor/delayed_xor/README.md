# DH-SNN 延迟异或任务 - SpikingJelly 实现

## 📁 目录内容

### 核心实现
- **`test_spikingjelly_minimal.py`** - 基于 SpikingJelly 框架的 DH-SNN 核心实现
- **`train_spikingjelly_dh_snn.py`** - 延迟异或任务的训练示例
- **`config_spikingjelly.py`** - 配置和工具函数

### 关键实验
- **`sequence_length_experiment_fixed.py`** - 发现序列长度控制重要性的关键实验

### 结果与文档
- **`figure3b_final_reproduction_report.md`** - 综合最终报告
- **`PROJECT_COMPLETION_SUMMARY.md`** - 项目总结和成就
- **`figure3b_reproduction_data.csv`** - 实验结果数据
- **`figure3b_reproduction_data.json`** - 机器可读结果
- **`figure3b_plotly.html`** - 图3b复现的交互式可视化

### 参考资料
- **`41467_2023_44614_MOESM1_ESM.pdf`** - 原始 DH-SNN 论文补充材料

## 🎯 快速开始

```bash
# 运行核心 DH-SNN 实现
python test_spikingjelly_minimal.py

# 在延迟异或任务上训练 DH-SNN
python train_spikingjelly_dh_snn.py

# 运行序列长度控制实验
python sequence_length_experiment_fixed.py
```

## 📊 关键结果

**成功复现图3b：**
- DH-SFNN 平均准确率：75.4%
- 普通 SFNN 平均准确率：55.0%
- **性能优势：+20.4个百分点（+37.1%相对提升）**
- 在所有延迟条件下（25-400步）均表现出一致的优越性

## 🔬 关键技术洞察

1. **树突电流保持**：核心创新在于树突电流在脉冲后不重置，保持长期记忆
2. **序列长度控制**：固定有效处理时间对于不同延迟条件下的公平比较至关重要
3. **SpikingJelly 集成**：成功将 DH-SNN 概念移植到现代脉冲神经网络框架

---
*这个实现证实了原论文关于树突异质性增强脉冲神经网络时间处理能力的声明。*
