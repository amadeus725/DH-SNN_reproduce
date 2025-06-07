# 🧠 DH-SNN: 树突异质性脉冲神经网络

## 📋 项目概述

本项目是对论文 "Temporal dendritic heterogeneity incorporated with spiking neural networks for learning multi-timescale dynamics" 的完整复现实现。

## 🚀 快速开始

### 安装依赖
```bash
pip install -e .
```

### 运行核心实验
```bash
# 延迟XOR实验
python experiments/core_reproduction/delayed_xor/main.py

# 多时间尺度XOR实验  
python experiments/core_reproduction/multitimescale_xor/main.py

# SHD数据集实验
python experiments/dataset_benchmarks/shd/main.py
```

## 📁 项目结构

- `dh_snn/` - 核心DH-SNN库
- `experiments/` - 实验代码
- `reports/` - 学术报告和演示
- `results/` - 实验结果
- `docs/` - 项目文档

## 📊 主要结果

| 实验 | Vanilla SNN | DH-SNN | 提升幅度 |
|------|-------------|---------|----------|
| 延迟XOR | 55.0% | 75.4% | +20.4% |
| 多时间尺度XOR | 50.2% | 96.2% | +46.0% |
| SHD数据集 | 54.5% | 79.8% | +25.3% |

## 🤝 贡献

欢迎提交Issue和Pull Request！

## 📄 许可证

MIT License
