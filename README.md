# DH-SNN Ultimate

树突异质性脉冲神经网络（DH-SNN）实现

## 📖 项目简介

这是一个基于SpikingJelly框架的DH-SNN最小化实现，专注于复现论文《Dendritic heterogeneity spiking neural networks》的核心算法和实验结果。

### 🎯 主要特点

- **超精简架构**: 仅包含核心的DH-SNN实现
- **中文注释**: 所有代码都有详细的中文注释
- **基于SpikingJelly**: 使用成熟的SpikingJelly框架作为基础
- **完整实验**: 包含论文中的所有关键实验，包括创新的多时间尺度实验
- **易于使用**: 提供统一的实验运行器
- **创新扩展**: 包含多时间尺度处理等前沿创新实验

## 📁 项目结构

```
dh-snn-ultimate/
├── README.md                    # 项目说明文档
├── run_experiments.py          # 统一实验运行器
├── configs/                    # 配置文件目录
│   ├── __init__.py
│   ├── config.py              # 基础配置类
│   ├── config_manager.py      # 配置管理器
│   ├── delayed_xor_config.py  # 延迟异或实验配置
│   ├── multiscale_xor_config.py # 多时间尺度XOR配置
│   ├── neurovpr_config.py     # NeuroVPR实验配置
│   ├── sequential_config.py   # 序列任务配置
│   └── speech_config.py       # 语音识别配置
├── dh_snn/                    # DH-SNN核心实现
│   ├── __init__.py
│   ├── core/                  # 核心算法模块
│   │   ├── __init__.py
│   │   ├── layers.py          # DH-SNN层实现
│   │   ├── models.py          # 完整模型定义
│   │   ├── neurons.py         # 神经元模型
│   │   ├── soma_heterogeneity.py # 胞体异质性实现
│   │   └── surrogate.py       # 代理梯度函数
│   ├── data/                  # 数据处理模块
│   │   └── __init__.py
│   └── utils/                 # 工具函数
│       └── __init__.py
├── experiments/               # 实验脚本目录
│   ├── applications/          # 应用实验
│   │   ├── neurovpr.py       # NeuroVPR视觉位置识别
│   │   ├── shd.py            # SHD数字识别
│   │   ├── smnist.py         # Sequential MNIST
│   │   └── ssc.py            # 语音命令识别
│   ├── core/                  # 基础核心实验
│   │   └── delayed_xor.py    # 延迟异或实验
│   ├── core_validation/       # 核心验证实验
│   │   ├── delayed_xor.py    # 延迟异或验证
│   │   ├── multi_timescale.py # 多时间尺度实验⭐
│   │   └── multi_timescale_demo.py # 多时间尺度演示
│   └── innovations/           # 创新实验
│       ├── soma_heterogeneity.py # 胞体异质性实验
│       └── soma_vs_dendritic_heterogeneity.py # 胞体vs树突异质性对比
└── docs/                      # 文档目录（预留）
```

### 📁 目录说明

- **configs/**: 各种实验的配置文件，包含模型参数、训练超参数等
- **dh_snn/core/**: DH-SNN的核心实现，包括异质性神经元、层和模型
- **experiments/**: 分层组织的实验脚本
  - **core_validation/**: 核心算法验证实验，包含⭐多时间尺度创新实验
  - **applications/**: 真实数据集上的应用实验
  - **innovations/**: 算法创新和扩展实验
- **run_experiments.py**: 统一的实验运行入口，支持命令行参数

## 🚀 快速开始

### 1. 环境配置

```bash
# 克隆项目
git clone https://github.com/amadeus725/DH-SNN_Reproduce.git
cd DH-SNN_Reproduce

# 创建虚拟环境（推荐）
python -m venv venv
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# 安装依赖
pip install -r requirements.txt

# 确保SpikingJelly已正确安装
python -c "import spikingjelly; print('SpikingJelly安装成功')"
```

### 2. 运行实验

#### 查看可用实验
```bash
python run_experiments.py list
```

#### 🌟 运行核心创新实验
```bash
# 运行多时间尺度XOR实验（核心创新）
python run_experiments.py multi_timescale

# 运行所有核心验证实验
python run_experiments.py core_all

# 运行创新实验集合
python run_experiments.py innovations
```

#### 运行其他实验
```bash
# 运行延迟异或实验（基础对比）
python run_experiments.py delayed_xor

# 运行SSC语音命令识别实验
python run_experiments.py ssc

# 运行SHD数字识别实验
python run_experiments.py shd

# 运行NeuroVPR视觉位置识别实验
python run_experiments.py neurovpr
```

#### 运行所有实验
```bash
python run_experiments.py all
```

### 3. 指定设备和参数
```bash
# 使用GPU运行多时间尺度实验
python run_experiments.py multi_timescale --device cuda

# 使用CPU运行
python run_experiments.py multi_timescale --device cpu

# 指定随机种子
python run_experiments.py multi_timescale --seed 123
```

## 🔬 实验详细说明

### 🌟 核心验证实验

#### 多时间尺度XOR（Multi-Timescale XOR）⭐
- **地位**: DH-SNN的核心创新实验
- **目的**: 验证DH-SNN处理多时间尺度信息的能力
- **数据**: 包含快速和慢速XOR逻辑的混合时间序列
- **创新点**: 
  - 不同分支自动特化处理不同时间尺度
  - 展示树突异质性的真正价值
- **预期结果**: DH-SNN显著优于普通SNN，准确率提升>15%

#### 延迟异或（Delayed XOR）
- **目的**: 基础时间信息处理能力验证
- **数据**: 人工生成的延迟异或序列
- **评估**: 不同延迟步数下的分类准确率
- **预期结果**: DH-SNN在长延迟任务中优于普通SNN

### 📱 应用实验

#### SSC - 脉冲语音命令识别
- **数据集**: Spiking Speech Commands
- **任务**: 35类语音命令识别
- **特点**: 真实的脉冲音频数据

#### SHD - 脉冲海德堡数字识别
- **数据集**: Spiking Heidelberg Digits
- **任务**: 20类数字识别
- **特点**: 测试不同时间常数配置的影响

#### NeuroVPR - 神经视觉位置识别
- **数据集**: NeuroVPR（DVS事件相机数据）
- **任务**: 100类位置识别
- **特点**: 事件驱动的视觉数据处理

### 🧪 创新实验集合

通过`python run_experiments.py innovations`可以运行：

1. **Multi-Timescale Processing** - 多时间尺度处理 ✅
2. **Temporal Specialization Analysis** - 时间特化分析 🚧
3. **Adaptive Branch Selection** - 自适应分支选择 🚧
4. **Hybrid Heterogeneity** - 混合异质性 🚧

## 📚 参考文献

```bibtex
@article{zheng2023dendritic,
  title={Dendritic heterogeneity spiking neural networks},
  author={Zheng, Shen and others},
  journal={Neural Networks},
  year={2023}
}
```

## 📄 许可证

本项目采用 MIT 许可证。详情请参见 [LICENSE](LICENSE) 文件。

---

