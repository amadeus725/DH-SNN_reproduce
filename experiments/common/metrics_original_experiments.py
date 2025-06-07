#!/usr/bin/env python3
"""
通用评估指标
"""

import torch
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

def calculate_accuracy(predictions, targets):
    """计算准确率"""
    pred_labels = predictions.argmax(dim=1)
    return (pred_labels == targets).float().mean().item()

def calculate_top_k_accuracy(predictions, targets, k=5):
    """计算Top-K准确率"""
    _, top_k_pred = predictions.topk(k, dim=1)
    targets_expanded = targets.view(-1, 1).expand_as(top_k_pred)
    return (top_k_pred == targets_expanded).any(dim=1).float().mean().item()

def calculate_precision_recall_f1(predictions, targets, average='macro'):
    """计算精确率、召回率和F1分数"""
    pred_labels = predictions.argmax(dim=1).cpu().numpy()
    targets_np = targets.cpu().numpy()
    
    precision, recall, f1, _ = precision_recall_fscore_support(
        targets_np, pred_labels, average=average, zero_division=0
    )
    
    return precision, recall, f1
