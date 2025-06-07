# GSC 实验

## 📋 概述

Google Speech Commands - 谷歌语音命令数据集

## 🏗️ 实验结构

```
gsc/
├── main_vanilla_sfnn.py    # Vanilla SFNN实验
├── main_dh_sfnn.py         # DH-SFNN实验  
├── main_vanilla_srnn.py    # Vanilla SRNN实验
├── main_dh_srnn.py         # DH-SRNN实验
├── data_loader.py          # 数据加载器
├── preprocessing.py        # 数据预处理
├── config.py               # 配置文件
└── README.md               # 本文件
```

## 🚀 快速开始

### 1. 运行Vanilla SFNN
```bash
python main_vanilla_sfnn.py
```

### 2. 运行DH-SFNN
```bash
python main_dh_sfnn.py
```

### 3. 运行SRNN实验
```bash
python main_vanilla_srnn.py
python main_dh_srnn.py
```

## 📊 实验配置

详见 `config.py` 文件中的配置参数。

## 📈 预期结果

根据原论文，DH-SNN应该在该数据集上显著优于Vanilla SNN。

## 📚 参考

- 原论文代码: `original_paper_code/gsc/`
- SpikingJelly文档: https://github.com/fangwei123456/spikingjelly
