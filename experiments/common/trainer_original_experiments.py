#!/usr/bin/env python3
"""
通用训练器
"""

import torch
import torch.nn as nn
from typing import Dict, Any
from .metrics import calculate_accuracy

class BaseTrainer:
    """基础训练器"""
    
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
        """训练一个epoch"""
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
        """评估模型"""
        self.model.eval()
        total_acc = 0.0
        
        with torch.no_grad():
            for data, targets in test_loader:
                data, targets = data.to(self.device), targets.to(self.device)
                outputs = self.model(data)
                total_acc += calculate_accuracy(outputs, targets)
        
        return total_acc / len(test_loader)
    
    def train(self, train_loader, test_loader, epochs):
        """完整训练流程"""
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
