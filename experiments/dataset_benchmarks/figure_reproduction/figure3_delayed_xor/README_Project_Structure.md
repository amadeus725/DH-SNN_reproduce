# 🏗️ DH-SNN SpikingJelly复现项目结构

## 📁 项目组织架构

```
DH-SNN_reproduce/
├── spikingjelly_delayed_xor/           # 主项目目录
│   ├── core/                           # 核心组件
│   │   ├── __init__.py
│   │   ├── models.py                   # 模型定义
│   │   ├── neurons.py                  # 神经元组件
│   │   ├── surrogate.py               # 替代函数
│   │   └── utils.py                   # 工具函数
│   ├── experiments/                    # 实验脚本
│   │   ├── delayed_xor/               # 延迟XOR实验
│   │   ├── multi_timescale_xor/       # 多时间尺度XOR实验
│   │   ├── shd_experiments/           # SHD数据集实验
│   │   ├── ssc_experiments/           # SSC数据集实验
│   │   └── figure_reproduction/       # 论文图表复现
│   ├── configs/                       # 配置文件
│   │   ├── __init__.py
│   │   ├── base_config.py            # 基础配置
│   │   ├── model_configs.py          # 模型配置
│   │   └── experiment_configs.py     # 实验配置
│   ├── data/                          # 数据处理
│   │   ├── __init__.py
│   │   ├── loaders.py                # 数据加载器
│   │   ├── generators.py             # 数据生成器
│   │   └── preprocessors.py          # 数据预处理
│   ├── training/                      # 训练框架
│   │   ├── __init__.py
│   │   ├── trainer.py                # 训练器
│   │   ├── evaluator.py              # 评估器
│   │   └── schedulers.py             # 调度器
│   ├── visualization/                 # 可视化
│   │   ├── __init__.py
│   │   ├── plotters.py               # 绘图工具
│   │   ├── analyzers.py              # 分析工具
│   │   └── figure_generators.py      # 图表生成
│   ├── outputs/                       # 输出目录
│   │   ├── results/                  # 实验结果
│   │   ├── figures/                  # 生成图表
│   │   ├── models/                   # 保存的模型
│   │   └── logs/                     # 日志文件
│   └── tests/                         # 测试文件
│       ├── test_models.py
│       ├── test_neurons.py
│       └── test_experiments.py
├── spikingjelly_shd/                  # SHD数据
├── spikingjelly_ssc/                  # SSC数据
└── SHD/                               # 原论文代码
```

## 🎯 核心组件设计

### 1. 模型层次结构
```
BaseModel
├── VanillaSFNN
├── DH_SFNN
│   ├── SingleBranchDH_SFNN
│   ├── MultiBranchDH_SFNN
│   └── AdaptiveDH_SFNN
└── MultiTimescaleModel
    ├── TwoBranchXOR
    └── MultiBranchXOR
```

### 2. 神经元组件
```
BaseNeuron
├── LIFNeuron
├── DH_LIFNeuron
│   ├── SingleBranchDH_LIF
│   └── MultiBranchDH_LIF
└── ReadoutIntegrator
```

### 3. 实验框架
```
BaseExperiment
├── DelayedXORExperiment
├── MultiTimescaleXORExperiment
├── SHDExperiment
├── SSCExperiment
└── Figure4Experiment
    ├── Figure4a_MultiTimescaleXOR
    ├── Figure4b_BranchComparison
    ├── Figure4c_TimingFactorAnalysis
    ├── Figure4d_NeuronVisualization
    ├── Figure4e_SingleBranchComparison
    └── Figure4f_DatasetComparison
```

## 🔧 配置管理

### 1. 基础配置
- 设备配置（CPU/GPU）
- 数据路径配置
- 输出路径配置
- 日志配置

### 2. 模型配置
- 网络架构参数
- 神经元参数
- 时间常数初始化
- 替代函数配置

### 3. 训练配置
- 优化器参数
- 学习率调度
- 批次大小
- 训练轮数

### 4. 实验配置
- 数据集选择
- 实验重复次数
- 评估指标
- 可视化选项

## 📊 数据流设计

### 1. 数据加载流程
```
原始数据 → 数据加载器 → 预处理器 → 批次生成器 → 模型输入
```

### 2. 训练流程
```
配置加载 → 模型创建 → 数据准备 → 训练循环 → 评估 → 结果保存
```

### 3. 实验流程
```
实验配置 → 多次运行 → 统计分析 → 可视化 → 报告生成
```

## 🎨 可视化框架

### 1. 实时监控
- 训练损失曲线
- 准确率变化
- 参数分布
- 梯度监控

### 2. 结果分析
- 性能对比图
- 统计显著性测试
- 参数演化分析
- 神经元活动可视化

### 3. 论文图表复现
- Figure 3: 延迟XOR实验
- Figure 4: 多时间尺度实验
- 所有子图的精确复现

## 🧪 测试框架

### 1. 单元测试
- 神经元功能测试
- 模型前向传播测试
- 数据加载测试
- 配置验证测试

### 2. 集成测试
- 端到端训练测试
- 实验流程测试
- 结果一致性测试

### 3. 性能测试
- 训练速度基准
- 内存使用监控
- GPU利用率测试

## 🚀 使用指南

### 1. 快速开始
```bash
# 安装依赖
pip install -r requirements.txt

# 运行延迟XOR实验
python experiments/delayed_xor/run_experiment.py

# 运行多时间尺度实验
python experiments/multi_timescale_xor/run_experiment.py

# 复现Figure 4
python experiments/figure_reproduction/figure4_reproduction.py
```

### 2. 自定义实验
```python
from core.models import DH_SFNN
from configs.experiment_configs import ExperimentConfig
from training.trainer import Trainer

# 创建配置
config = ExperimentConfig(
    model_type='dh_sfnn',
    num_branches=4,
    dataset='shd'
)

# 创建模型
model = DH_SFNN(config)

# 训练
trainer = Trainer(model, config)
results = trainer.train()
```

## 📈 扩展性设计

### 1. 新模型添加
- 继承BaseModel类
- 实现必要的接口
- 添加配置支持

### 2. 新实验添加
- 继承BaseExperiment类
- 定义实验流程
- 配置可视化

### 3. 新数据集支持
- 实现数据加载器
- 添加预处理器
- 更新配置文件

这个框架设计确保了代码的模块化、可扩展性和可维护性，为DH-SNN的研究和应用提供了坚实的基础。
