# 🚀 快速开始指南

## 环境准备

1. 安装Python 3.8+
2. 安装PyTorch
3. 安装SpikingJelly

## 运行第一个实验

```bash
# 克隆项目
git clone <repository-url>
cd DH-SNN_reproduce

# 安装依赖
pip install -e .

# 运行延迟XOR实验
python experiments/core_reproduction/delayed_xor/main.py
```

## 查看结果

实验结果保存在 `results/` 目录下，包括：
- 训练日志
- 模型权重
- 可视化图表

## 下一步

- 查看 `docs/EXPERIMENTS.md` 了解更多实验
- 阅读 `reports/reproduction_report/` 查看详细报告
- 参考 `docs/API_REFERENCE.md` 了解API使用
