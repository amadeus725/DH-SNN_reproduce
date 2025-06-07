#!/usr/bin/env python3
"""
工具函数 - SHD实验
"""

import torch
import numpy as np
import random
import os
import json
from typing import Dict, List, Tuple, Optional


def set_seed(seed: int):
    """设置随机种子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class AverageMeter:
    """计算和存储平均值和当前值"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output: torch.Tensor, target: torch.Tensor, topk: Tuple[int, ...] = (1,)) -> List[torch.Tensor]:
    """计算top-k准确率"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        
        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def save_checkpoint(state: Dict, save_dir: str, filename: str):
    """保存检查点"""
    os.makedirs(save_dir, exist_ok=True)
    filepath = os.path.join(save_dir, filename)
    torch.save(state, filepath)
    print(f"Checkpoint saved: {filepath}")


def load_checkpoint(filepath: str, model: torch.nn.Module, optimizer: Optional[torch.optim.Optimizer] = None) -> Dict:
    """加载检查点"""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Checkpoint not found: {filepath}")
    
    checkpoint = torch.load(filepath, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    print(f"Checkpoint loaded: {filepath}")
    return checkpoint


def count_parameters(model: torch.nn.Module) -> Tuple[int, int]:
    """计算模型参数数量"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params


def get_model_size_mb(model: torch.nn.Module) -> float:
    """计算模型大小(MB)"""
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    
    size_mb = (param_size + buffer_size) / 1024 / 1024
    return size_mb


def calculate_macs(model: torch.nn.Module, input_shape: Tuple[int, ...]) -> int:
    """计算模型的MACs (Multiply-Accumulate Operations)"""
    # 这是一个简化的MAC计算，实际应用中可能需要更精确的计算
    total_macs = 0
    
    def mac_hook(module, input, output):
        nonlocal total_macs
        if isinstance(module, torch.nn.Linear):
            total_macs += module.in_features * module.out_features
        elif isinstance(module, torch.nn.Conv1d):
            total_macs += (module.in_channels * module.out_channels * 
                          module.kernel_size[0] * output.shape[-1])
    
    # 注册钩子
    hooks = []
    for module in model.modules():
        if isinstance(module, (torch.nn.Linear, torch.nn.Conv1d)):
            hooks.append(module.register_forward_hook(mac_hook))
    
    # 前向传播
    with torch.no_grad():
        dummy_input = torch.randn(1, *input_shape)
        model(dummy_input)
    
    # 移除钩子
    for hook in hooks:
        hook.remove()
    
    return total_macs


def save_experiment_results(results: Dict, save_dir: str, filename: str):
    """保存实验结果"""
    os.makedirs(save_dir, exist_ok=True)
    filepath = os.path.join(save_dir, filename)
    
    with open(filepath, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved: {filepath}")


def load_experiment_results(filepath: str) -> Dict:
    """加载实验结果"""
    with open(filepath, 'r') as f:
        results = json.load(f)
    return results


def format_time(seconds: float) -> str:
    """格式化时间"""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds // 60
        seconds = seconds % 60
        return f"{int(minutes)}m {seconds:.1f}s"
    else:
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        seconds = seconds % 60
        return f"{int(hours)}h {int(minutes)}m {seconds:.1f}s"


def print_model_summary(model: torch.nn.Module, input_shape: Tuple[int, ...]):
    """打印模型摘要"""
    total_params, trainable_params = count_parameters(model)
    model_size = get_model_size_mb(model)
    
    print("=" * 60)
    print("MODEL SUMMARY")
    print("=" * 60)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Model size: {model_size:.2f} MB")
    
    try:
        macs = calculate_macs(model, input_shape)
        print(f"MACs: {macs:,}")
    except Exception as e:
        print(f"MAC calculation failed: {e}")
    
    print("=" * 60)


def create_experiment_config(base_config: Dict, **kwargs) -> Dict:
    """创建实验配置"""
    config = base_config.copy()
    config.update(kwargs)
    return config


def log_experiment_info(config: Dict, results: Dict):
    """记录实验信息"""
    print("\n" + "=" * 80)
    print("EXPERIMENT SUMMARY")
    print("=" * 80)
    
    print("Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    print("\nResults:")
    for key, value in results.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")
    
    print("=" * 80)


class ExperimentLogger:
    """实验日志记录器"""
    
    def __init__(self, log_dir: str, experiment_name: str):
        self.log_dir = log_dir
        self.experiment_name = experiment_name
        self.log_file = os.path.join(log_dir, f"{experiment_name}.log")
        
        os.makedirs(log_dir, exist_ok=True)
        
        # 清空日志文件
        with open(self.log_file, 'w') as f:
            f.write(f"Experiment: {experiment_name}\n")
            f.write("=" * 80 + "\n")
    
    def log(self, message: str):
        """记录日志"""
        print(message)
        with open(self.log_file, 'a') as f:
            f.write(message + "\n")
    
    def log_config(self, config: Dict):
        """记录配置"""
        self.log("Configuration:")
        for key, value in config.items():
            self.log(f"  {key}: {value}")
        self.log("")
    
    def log_results(self, results: Dict):
        """记录结果"""
        self.log("Results:")
        for key, value in results.items():
            if isinstance(value, float):
                self.log(f"  {key}: {value:.4f}")
            else:
                self.log(f"  {key}: {value}")
        self.log("")


def compute_efficiency_metrics(model: torch.nn.Module, input_shape: Tuple[int, ...], 
                              accuracy: float, baseline_accuracy: float = None) -> Dict:
    """计算效率指标"""
    total_params, _ = count_parameters(model)
    model_size = get_model_size_mb(model)
    
    try:
        macs = calculate_macs(model, input_shape)
    except:
        macs = None
    
    metrics = {
        'accuracy': accuracy,
        'parameters': total_params,
        'model_size_mb': model_size,
        'macs': macs
    }
    
    if baseline_accuracy is not None:
        metrics['accuracy_improvement'] = accuracy - baseline_accuracy
        if macs is not None:
            # 计算效率提升 (准确率提升 / MAC数量)
            metrics['efficiency_ratio'] = metrics['accuracy_improvement'] / (macs / 1e6)
    
    return metrics


if __name__ == "__main__":
    # 测试工具函数
    print("Testing utility functions...")
    
    # 测试AverageMeter
    meter = AverageMeter()
    for i in range(10):
        meter.update(i)
    print(f"Average: {meter.avg}")
    
    # 测试时间格式化
    print(f"Time format: {format_time(3661.5)}")
    
    print("Utility functions test completed!")
