# 🚀 DH-SNN 数据集基准实验

这个目录包含了DH-SNN论文中所有数据集的基准实验实现。

## 📋 实验列表

### 🎯 优先级 1 (快速验证)
- **Sequential MNIST** - 序列MNIST数字识别
- **Permuted MNIST** - 置换MNIST数字识别

### 🎯 优先级 2 (核心实验)  
- **GSC** - Google Speech Commands 语音命令识别
- **SHD** - Spiking Heidelberg Digits 脉冲数字识别
- **SSC** - Spiking Speech Commands 脉冲语音命令

### 🎯 优先级 3 (扩展实验)
- **TIMIT** - 语音识别
- **DEAP** - EEG情感识别
- **NeuroVPR** - 视觉位置识别

## 🏗️ 目录结构

```
dataset_benchmarks/
├── sequential_mnist/          # Sequential MNIST实验
│   ├── config.py             # 配置文件
│   ├── data_loader.py        # 数据加载器
│   ├── main_vanilla_srnn.py  # Vanilla SRNN实验
│   ├── main_dh_srnn.py       # DH-SRNN实验
│   └── results/              # 实验结果
├── permuted_mnist/           # Permuted MNIST实验
├── gsc/                      # GSC实验
├── shd/                      # SHD实验
├── ssc/                      # SSC实验
├── timit/                    # TIMIT实验
├── deap/                     # DEAP实验
├── neurovpr/                 # NeuroVPR实验
├── run_all_dataset_experiments.py  # 统一运行脚本
├── monitor_experiments.py    # 实验监控脚本
├── test_all_imports.py       # 导入测试脚本
└── README.md                 # 本文件
```

## 🚀 快速开始

### 1. 测试所有脚本导入
```bash
cd experiments/dataset_benchmarks/
python test_all_imports.py
```

### 2. 查看可用实验
```bash
python run_all_dataset_experiments.py --list
```

### 3. 运行单个实验
```bash
python run_all_dataset_experiments.py --experiment sequential_mnist --script main_dh_srnn.py
```

### 4. 按优先级运行实验
```bash
# 运行优先级1的所有实验
python run_all_dataset_experiments.py --priority 1

# 运行所有实验
python run_all_dataset_experiments.py --all
```

### 5. 监控实验状态
```bash
# 查看当前状态
python monitor_experiments.py

# 持续监控
python monitor_experiments.py --monitor

# 导出结果
python monitor_experiments.py --export results.json
```

## 📊 实验配置

每个实验目录都包含：

- **config.py** - 实验配置文件，包含网络参数、训练参数、DH-SNN参数等
- **data_loader.py** - 数据加载器，负责数据预处理和批次生成
- **main_*.py** - 主实验脚本，包含模型创建、训练和评估逻辑

### 配置文件结构
```python
# 基础配置
BASE_CONFIG = {
    'device': 'cuda',
    'seed': 42,
    'num_workers': 4
}

# 网络配置
NETWORK_CONFIG = {
    'input_size': 700,
    'hidden_size': 200,
    'output_size': 20,
    'v_threshold': 1.0
}

# DH-SNN配置
DH_CONFIG = {
    'num_branches': 2,
    'timing_init': 'medium',
    'tau_m_init': (0.0, 4.0),
    'tau_n_init': (0.0, 4.0)
}
```

## 🎯 实验参数说明

### 时间因子配置
- **Small**: (-4.0, 0.0) - 快速响应
- **Medium**: (0.0, 4.0) - 中等响应  
- **Large**: (2.0, 6.0) - 慢速响应

### 数据集特定参数
- **Sequential MNIST**: input_size=1, output_size=10
- **GSC**: input_size=700, output_size=15
- **SHD**: input_size=700, output_size=20
- **SSC**: input_size=700, output_size=35

## 📈 结果分析

实验结果保存在各自的`results/`目录中：

- **模型文件**: `*.pth` - 训练好的模型权重
- **结果文件**: `*_results.pth` - 包含准确率和配置信息
- **日志文件**: `*.log` - 训练过程日志

### 查看结果
```python
import torch

# 加载结果
results = torch.load('results/dh_srnn_results.pth')
print(f"最佳准确率: {results['best_accuracy']:.3f}")
print(f"模型类型: {results['model_type']}")
```

## 🔧 故障排除

### 常见问题

1. **导入错误**
   ```bash
   python test_all_imports.py --experiment sequential_mnist
   ```

2. **CUDA内存不足**
   - 减小batch_size
   - 使用CPU: 修改config.py中的device设置

3. **数据集缺失**
   - 检查数据路径配置
   - 确保数据集已下载

### 调试模式
```bash
# 模拟运行，不执行实际训练
python run_all_dataset_experiments.py --priority 1 --dry-run
```

## 📚 参考

- [DH-SNN原论文](https://arxiv.org/abs/2404.08013)
- [SpikingJelly文档](https://spikingjelly.readthedocs.io/)
- [项目主页](https://github.com/your-repo/DH-SNN_reproduce)

## 🤝 贡献

欢迎提交Issue和Pull Request来改进实验代码！

## 📄 许可证

本项目遵循MIT许可证。
