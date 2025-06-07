#!/usr/bin/env python3
"""
按照原论文分类创建SpikingJelly实验结构
"""

import os
from pathlib import Path

def create_experiment_structure():
    """创建完整的实验结构"""
    
    print("🏗️ 创建SpikingJelly实验结构...")
    
    # 基础路径
    base_path = Path("spikingjelly_implementation")
    experiments_path = base_path / "experiments"
    
    # 实验分类 - 按照原论文代码结构
    experiment_categories = {
        "shd": {
            "description": "Spiking Heidelberg Digits - 脉冲海德堡数字识别",
            "files": [
                "main_vanilla_sfnn.py",
                "main_dh_sfnn.py", 
                "main_vanilla_srnn.py",
                "main_dh_srnn.py",
                "data_loader.py",
                "config.py",
                "README.md"
            ]
        },
        "ssc": {
            "description": "Spiking Speech Commands - 脉冲语音命令识别", 
            "files": [
                "main_vanilla_sfnn.py",
                "main_dh_sfnn.py",
                "main_vanilla_srnn.py", 
                "main_dh_srnn.py",
                "data_loader.py",
                "config.py",
                "README.md"
            ]
        },
        "gsc": {
            "description": "Google Speech Commands - 谷歌语音命令数据集",
            "files": [
                "main_vanilla_sfnn.py",
                "main_dh_sfnn.py",
                "main_vanilla_srnn.py",
                "main_dh_srnn.py", 
                "data_loader.py",
                "preprocessing.py",
                "config.py",
                "README.md"
            ]
        },
        "timit": {
            "description": "TIMIT Speech Recognition - TIMIT语音识别数据集",
            "files": [
                "main_dh_sfnn.py",
                "main_dh_srnn.py",
                "data_loader.py",
                "preprocessing.py",
                "config.py", 
                "README.md"
            ]
        },
        "deap": {
            "description": "DEAP EEG Emotion Recognition - EEG情感识别",
            "files": [
                "main_vanilla_sfnn.py",
                "main_dh_sfnn.py",
                "main_vanilla_srnn.py",
                "main_dh_srnn.py",
                "preprocessing.py",
                "data_loader.py",
                "config.py",
                "README.md"
            ]
        },
        "sequential_mnist": {
            "description": "Sequential MNIST - 序列MNIST数据集",
            "files": [
                "main_vanilla_srnn.py",
                "main_dh_srnn.py", 
                "data_loader.py",
                "config.py",
                "README.md"
            ]
        },
        "permuted_mnist": {
            "description": "Permuted Sequential MNIST - 置换序列MNIST",
            "files": [
                "main_vanilla_srnn.py",
                "main_dh_srnn.py",
                "data_loader.py", 
                "config.py",
                "README.md"
            ]
        },
        "neurovpr": {
            "description": "NeuroVPR - 神经形态视觉位置识别",
            "files": [
                "main_dh_sfnn.py",
                "data_loader.py",
                "preprocessing.py",
                "config.py",
                "README.md"
            ]
        }
    }
    
    # 创建目录结构
    for category, info in experiment_categories.items():
        category_path = experiments_path / category
        category_path.mkdir(parents=True, exist_ok=True)
        
        print(f"📁 创建目录: {category}/ - {info['description']}")
        
        # 创建文件
        for filename in info['files']:
            file_path = category_path / filename
            if not file_path.exists():
                if filename == "README.md":
                    create_readme(file_path, category, info['description'])
                elif filename == "config.py":
                    create_config(file_path, category)
                elif filename.startswith("main_"):
                    create_main_template(file_path, category, filename)
                elif filename == "data_loader.py":
                    create_data_loader_template(file_path, category)
                elif filename == "preprocessing.py":
                    create_preprocessing_template(file_path, category)
                else:
                    file_path.touch()
                
                print(f"  📄 创建文件: {filename}")
    
    # 创建通用工具目录
    create_common_tools(experiments_path)
    
    print(f"\n✅ 实验结构创建完成!")

def create_readme(file_path, category, description):
    """创建README模板"""
    
    content = f"""# {category.upper()} 实验

## 📋 概述

{description}

## 🏗️ 实验结构

```
{category}/
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

- 原论文代码: `original_paper_code/{category}/`
- SpikingJelly文档: https://github.com/fangwei123456/spikingjelly
"""
    
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)

def create_config(file_path, category):
    """创建配置文件模板"""
    
    # 根据不同数据集设置不同的默认配置
    configs = {
        "shd": {
            "input_size": 700,
            "hidden_size": 64, 
            "output_size": 20,
            "batch_size": 100,
            "learning_rate": 1e-2
        },
        "ssc": {
            "input_size": 700,
            "hidden_size": 200,
            "output_size": 35, 
            "batch_size": 200,
            "learning_rate": 1e-2
        },
        "gsc": {
            "input_size": 120,
            "hidden_size": 200,
            "output_size": 15,
            "batch_size": 100,
            "learning_rate": 1e-2
        },
        "timit": {
            "input_size": 40,
            "hidden_size": 256,
            "output_size": 61,
            "batch_size": 128, 
            "learning_rate": 1e-3
        },
        "deap": {
            "input_size": 32,
            "hidden_size": 200,
            "output_size": 3,
            "batch_size": 200,
            "learning_rate": 1e-2
        },
        "sequential_mnist": {
            "input_size": 1,
            "hidden_size": 256,
            "output_size": 10,
            "batch_size": 128,
            "learning_rate": 1e-3
        },
        "permuted_mnist": {
            "input_size": 1, 
            "hidden_size": 256,
            "output_size": 10,
            "batch_size": 128,
            "learning_rate": 1e-3
        },
        "neurovpr": {
            "input_size": 128,
            "hidden_size": 64,
            "output_size": 1,
            "batch_size": 32,
            "learning_rate": 1e-3
        }
    }
    
    config = configs.get(category, configs["shd"])
    
    content = f"""#!/usr/bin/env python3
\"\"\"
{category.upper()}数据集配置文件
\"\"\"

import torch

# 基础配置
BASE_CONFIG = {{
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'seed': 42,
    'num_workers': 4,
    'pin_memory': True
}}

# 网络配置
NETWORK_CONFIG = {{
    'input_size': {config['input_size']},
    'hidden_size': {config['hidden_size']},
    'output_size': {config['output_size']},
    'v_threshold': 1.0,
    'dt': 1.0
}}

# 训练配置
TRAINING_CONFIG = {{
    'batch_size': {config['batch_size']},
    'learning_rate': {config['learning_rate']},
    'epochs': 100,
    'weight_decay': 0.0,
    'grad_clip': None
}}

# 时间因子配置
TIMING_CONFIG = {{
    'small': (-4.0, 0.0),
    'medium': (0.0, 4.0), 
    'large': (2.0, 6.0)
}}

# DH-SNN配置
DH_CONFIG = {{
    'num_branches': 8,
    'timing_init': 'large',
    'learnable_timing': True,
    'beneficial_init': True
}}

# 数据配置
DATA_CONFIG = {{
    'data_path': '../datasets/{category}/',
    'train_samples': None,  # None表示使用全部数据
    'test_samples': None,
    'validation_split': 0.1
}}

# 实验配置
EXPERIMENT_CONFIG = {{
    'num_trials': 5,
    'save_models': True,
    'save_results': True,
    'output_dir': 'outputs/{category}/'
}}
"""
    
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)

def create_main_template(file_path, category, filename):
    """创建主实验文件模板"""
    
    model_type = filename.replace("main_", "").replace(".py", "")
    
    content = f"""#!/usr/bin/env python3
\"\"\"
{category.upper()} - {model_type.upper()} 实验
\"\"\"

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import sys
from pathlib import Path

# 添加路径
sys.path.append(str(Path(__file__).parent.parent.parent))

from core.models import *
from core.neurons import *
from core.utils import *
from config import *
from data_loader import load_{category}_data

def create_model(config):
    \"\"\"创建模型\"\"\"
    
    if '{model_type}' == 'vanilla_sfnn':
        model = VanillaSFNN(
            input_size=config['input_size'],
            hidden_size=config['hidden_size'], 
            output_size=config['output_size'],
            v_threshold=config['v_threshold'],
            device=config['device']
        )
    elif '{model_type}' == 'dh_sfnn':
        model = DH_SFNN(
            input_size=config['input_size'],
            hidden_size=config['hidden_size'],
            output_size=config['output_size'],
            num_branches=DH_CONFIG['num_branches'],
            v_threshold=config['v_threshold'],
            device=config['device']
        )
    elif '{model_type}' == 'vanilla_srnn':
        model = VanillaSRNN(
            input_size=config['input_size'],
            hidden_size=config['hidden_size'],
            output_size=config['output_size'],
            v_threshold=config['v_threshold'],
            device=config['device']
        )
    elif '{model_type}' == 'dh_srnn':
        model = DH_SRNN(
            input_size=config['input_size'],
            hidden_size=config['hidden_size'],
            output_size=config['output_size'],
            num_branches=DH_CONFIG['num_branches'],
            v_threshold=config['v_threshold'],
            device=config['device']
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    return model

def train_model(model, train_loader, test_loader, config):
    \"\"\"训练模型\"\"\"
    
    device = config['device']
    model = model.to(device)
    
    # 优化器
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
    criterion = nn.CrossEntropyLoss()
    
    best_acc = 0.0
    
    for epoch in range(config['epochs']):
        # 训练
        model.train()
        train_loss = 0.0
        train_acc = 0.0
        
        for batch_idx, (data, targets) in enumerate(train_loader):
            data, targets = data.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            pred = outputs.argmax(dim=1)
            train_acc += (pred == targets).float().mean().item()
        
        # 测试
        model.eval()
        test_acc = 0.0
        with torch.no_grad():
            for data, targets in test_loader:
                data, targets = data.to(device), targets.to(device)
                outputs = model(data)
                pred = outputs.argmax(dim=1)
                test_acc += (pred == targets).float().mean().item()
        
        train_loss /= len(train_loader)
        train_acc /= len(train_loader)
        test_acc /= len(test_loader)
        
        if test_acc > best_acc:
            best_acc = test_acc
        
        if epoch % 10 == 0:
            print(f"Epoch {{epoch:3d}}: Loss={{train_loss:.4f}}, Train={{train_acc:.3f}}, Test={{test_acc:.3f}}, Best={{best_acc:.3f}}")
    
    return best_acc

def main():
    \"\"\"主函数\"\"\"
    
    print(f"🚀 {category.upper()} - {model_type.upper()} 实验")
    print("="*60)
    
    # 合并配置
    config = {{**BASE_CONFIG, **NETWORK_CONFIG, **TRAINING_CONFIG}}
    
    print(f"📱 设备: {{config['device']}}")
    print(f"🏗️ 模型: {model_type.upper()}")
    
    # 加载数据
    train_loader, test_loader = load_{category}_data(
        DATA_CONFIG['data_path'],
        config['batch_size'],
        config['num_workers']
    )
    
    # 创建模型
    model = create_model(config)
    print(f"📊 参数数量: {{sum(p.numel() for p in model.parameters())}}")
    
    # 训练
    best_acc = train_model(model, train_loader, test_loader, config)
    
    print(f"\\n🎉 训练完成!")
    print(f"📈 最佳准确率: {{best_acc:.3f}}")
    
    # 保存结果
    os.makedirs(EXPERIMENT_CONFIG['output_dir'], exist_ok=True)
    results = {{
        'model_type': '{model_type}',
        'best_accuracy': best_acc,
        'config': config
    }}
    
    torch.save(results, f"{{EXPERIMENT_CONFIG['output_dir']}}/{model_type}_results.pth")
    
    return best_acc

if __name__ == '__main__':
    main()
"""
    
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)

def create_data_loader_template(file_path, category):
    """创建数据加载器模板"""
    
    content = f"""#!/usr/bin/env python3
\"\"\"
{category.upper()}数据加载器
\"\"\"

import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np
import os

class {category.upper()}Dataset(Dataset):
    \"\"\"
    {category.upper()}数据集类
    \"\"\"
    
    def __init__(self, data_path, split='train', transform=None):
        self.data_path = data_path
        self.split = split
        self.transform = transform
        
        # 加载数据
        self.data, self.labels = self._load_data()
    
    def _load_data(self):
        \"\"\"加载数据\"\"\"
        # TODO: 实现具体的数据加载逻辑
        # 这里需要根据具体数据集格式实现
        
        if self.split == 'train':
            data_file = os.path.join(self.data_path, f'{category}_train.h5.gz')
        else:
            data_file = os.path.join(self.data_path, f'{category}_test.h5.gz')
        
        # 示例代码 - 需要根据实际数据格式修改
        data = torch.randn(1000, 100, 700)  # [samples, time, features]
        labels = torch.randint(0, 20, (1000,))  # [samples]
        
        return data, labels
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        data = self.data[idx]
        label = self.labels[idx]
        
        if self.transform:
            data = self.transform(data)
        
        return data, label

def load_{category}_data(data_path, batch_size, num_workers=4):
    \"\"\"
    加载{category.upper()}数据
    
    Args:
        data_path: 数据路径
        batch_size: 批次大小
        num_workers: 工作进程数
    
    Returns:
        train_loader, test_loader
    \"\"\"
    
    # 创建数据集
    train_dataset = {category.upper()}Dataset(data_path, split='train')
    test_dataset = {category.upper()}Dataset(data_path, split='test')
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    print(f"📊 {category.upper()}数据加载完成:")
    print(f"  训练集: {{len(train_dataset)}} 样本")
    print(f"  测试集: {{len(test_dataset)}} 样本")
    
    return train_loader, test_loader

if __name__ == '__main__':
    # 测试数据加载器
    train_loader, test_loader = load_{category}_data('../datasets/{category}/', 32)
    
    for batch_idx, (data, targets) in enumerate(train_loader):
        print(f"Batch {{batch_idx}}: data={{data.shape}}, targets={{targets.shape}}")
        if batch_idx >= 2:
            break
"""
    
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)

def create_preprocessing_template(file_path, category):
    """创建预处理模板"""
    
    content = f"""#!/usr/bin/env python3
\"\"\"
{category.upper()}数据预处理
\"\"\"

import torch
import numpy as np
from typing import Tuple, Optional

def preprocess_{category}_data(raw_data: np.ndarray, 
                              sample_rate: int = 16000,
                              target_length: Optional[int] = None) -> torch.Tensor:
    \"\"\"
    预处理{category.upper()}数据
    
    Args:
        raw_data: 原始数据
        sample_rate: 采样率
        target_length: 目标长度
    
    Returns:
        预处理后的数据
    \"\"\"
    
    # TODO: 根据具体数据集实现预处理逻辑
    
    # 示例预处理步骤:
    # 1. 归一化
    data = (raw_data - raw_data.mean()) / (raw_data.std() + 1e-8)
    
    # 2. 长度调整
    if target_length is not None:
        if len(data) > target_length:
            data = data[:target_length]
        elif len(data) < target_length:
            data = np.pad(data, (0, target_length - len(data)), 'constant')
    
    # 3. 转换为张量
    data = torch.from_numpy(data).float()
    
    return data

def convert_to_spikes(data: torch.Tensor, 
                     dt: float = 1e-3,
                     max_time: float = 1.0) -> torch.Tensor:
    \"\"\"
    将数据转换为脉冲序列
    
    Args:
        data: 输入数据
        dt: 时间步长
        max_time: 最大时间
    
    Returns:
        脉冲序列
    \"\"\"
    
    # TODO: 实现数据到脉冲的转换
    # 这里需要根据具体数据类型实现
    
    time_steps = int(max_time / dt)
    
    if data.dim() == 1:
        # 时序数据
        spikes = torch.zeros(time_steps, len(data))
        # 简单的率编码
        for t in range(time_steps):
            spikes[t] = (torch.rand_like(data) < torch.abs(data)).float()
    else:
        # 其他格式数据
        spikes = data
    
    return spikes

def augment_data(data: torch.Tensor, 
                noise_level: float = 0.1) -> torch.Tensor:
    \"\"\"
    数据增强
    
    Args:
        data: 输入数据
        noise_level: 噪声水平
    
    Returns:
        增强后的数据
    \"\"\"
    
    # 添加噪声
    noise = torch.randn_like(data) * noise_level
    augmented_data = data + noise
    
    return augmented_data

if __name__ == '__main__':
    # 测试预处理函数
    test_data = np.random.randn(1000)
    processed = preprocess_{category}_data(test_data)
    spikes = convert_to_spikes(processed)
    
    print(f"原始数据: {{test_data.shape}}")
    print(f"预处理后: {{processed.shape}}")
    print(f"脉冲数据: {{spikes.shape}}")
"""
    
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)

def create_common_tools(experiments_path):
    """创建通用工具"""
    
    common_path = experiments_path / "common"
    common_path.mkdir(exist_ok=True)
    
    # 创建通用工具文件
    tools = [
        "__init__.py",
        "metrics.py", 
        "visualization.py",
        "utils.py",
        "trainer.py"
    ]
    
    for tool in tools:
        tool_path = common_path / tool
        if not tool_path.exists():
            if tool == "metrics.py":
                create_metrics_file(tool_path)
            elif tool == "trainer.py":
                create_trainer_file(tool_path)
            else:
                tool_path.touch()
    
    print(f"📁 创建通用工具目录: common/")

def create_metrics_file(file_path):
    """创建评估指标文件"""
    
    content = """#!/usr/bin/env python3
\"\"\"
通用评估指标
\"\"\"

import torch
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

def calculate_accuracy(predictions, targets):
    \"\"\"计算准确率\"\"\"
    pred_labels = predictions.argmax(dim=1)
    return (pred_labels == targets).float().mean().item()

def calculate_top_k_accuracy(predictions, targets, k=5):
    \"\"\"计算Top-K准确率\"\"\"
    _, top_k_pred = predictions.topk(k, dim=1)
    targets_expanded = targets.view(-1, 1).expand_as(top_k_pred)
    return (top_k_pred == targets_expanded).any(dim=1).float().mean().item()

def calculate_precision_recall_f1(predictions, targets, average='macro'):
    \"\"\"计算精确率、召回率和F1分数\"\"\"
    pred_labels = predictions.argmax(dim=1).cpu().numpy()
    targets_np = targets.cpu().numpy()
    
    precision, recall, f1, _ = precision_recall_fscore_support(
        targets_np, pred_labels, average=average, zero_division=0
    )
    
    return precision, recall, f1
"""
    
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)

def create_trainer_file(file_path):
    """创建通用训练器文件"""
    
    content = """#!/usr/bin/env python3
\"\"\"
通用训练器
\"\"\"

import torch
import torch.nn as nn
from typing import Dict, Any
from .metrics import calculate_accuracy

class BaseTrainer:
    \"\"\"基础训练器\"\"\"
    
    def __init__(self, model, config):
        self.model = model
        self.config = config
        self.device = config['device']
        
        # 优化器
        self.optimizer = torch.optim.Adam(
            model.parameters(), 
            lr=config['learning_rate']
        )
        
        # 损失函数
        self.criterion = nn.CrossEntropyLoss()
        
        # 学习率调度器
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, step_size=20, gamma=0.5
        )
    
    def train_epoch(self, train_loader):
        \"\"\"训练一个epoch\"\"\"
        self.model.train()
        total_loss = 0.0
        total_acc = 0.0
        
        for batch_idx, (data, targets) in enumerate(train_loader):
            data, targets = data.to(self.device), targets.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(data)
            loss = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            total_acc += calculate_accuracy(outputs, targets)
        
        return total_loss / len(train_loader), total_acc / len(train_loader)
    
    def evaluate(self, test_loader):
        \"\"\"评估模型\"\"\"
        self.model.eval()
        total_acc = 0.0
        
        with torch.no_grad():
            for data, targets in test_loader:
                data, targets = data.to(self.device), targets.to(self.device)
                outputs = self.model(data)
                total_acc += calculate_accuracy(outputs, targets)
        
        return total_acc / len(test_loader)
    
    def train(self, train_loader, test_loader, epochs):
        \"\"\"完整训练流程\"\"\"
        best_acc = 0.0
        
        for epoch in range(epochs):
            train_loss, train_acc = self.train_epoch(train_loader)
            test_acc = self.evaluate(test_loader)
            
            self.scheduler.step()
            
            if test_acc > best_acc:
                best_acc = test_acc
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch:3d}: Loss={train_loss:.4f}, Train={train_acc:.3f}, Test={test_acc:.3f}")
        
        return best_acc
"""
    
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)

if __name__ == "__main__":
    create_experiment_structure()
