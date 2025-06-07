#!/usr/bin/env python3
"""
结果管理器 - 负责保存模型、记录训练历史和生成可视化图表
"""

import os
import json
import pickle
import torch
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime
from pathlib import Path

class ResultsManager:
    def __init__(self, experiment_name, base_dir="results"):
        self.experiment_name = experiment_name
        self.base_dir = Path(base_dir)
        self.experiment_dir = self.base_dir / experiment_name
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 创建目录结构
        self.models_dir = self.experiment_dir / "models"
        self.plots_dir = self.experiment_dir / "plots"
        self.logs_dir = self.experiment_dir / "logs"
        
        for dir_path in [self.models_dir, self.plots_dir, self.logs_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # 训练历史记录
        self.training_history = {}
        
        print(f"📁 结果将保存到: {self.experiment_dir}")
    
    def save_model(self, model, model_name, epoch, accuracy, additional_info=None):
        """保存模型"""
        model_info = {
            'model_name': model_name,
            'epoch': epoch,
            'accuracy': accuracy,
            'timestamp': self.timestamp,
            'additional_info': additional_info or {}
        }
        
        # 保存模型权重
        model_path = self.models_dir / f"{model_name}_epoch{epoch}_{accuracy:.3f}_{self.timestamp}.pth"
        torch.save(model.state_dict(), model_path)
        
        # 保存模型信息
        info_path = self.models_dir / f"{model_name}_epoch{epoch}_{accuracy:.3f}_{self.timestamp}_info.json"
        with open(info_path, 'w') as f:
            json.dump(model_info, f, indent=2)
        
        print(f"💾 模型已保存: {model_path}")
        return model_path
    
    def log_training_step(self, model_name, epoch, train_loss, train_acc, test_acc, epoch_time):
        """记录训练步骤"""
        if model_name not in self.training_history:
            self.training_history[model_name] = {
                'epochs': [],
                'train_loss': [],
                'train_acc': [],
                'test_acc': [],
                'epoch_time': []
            }
        
        history = self.training_history[model_name]
        history['epochs'].append(epoch)
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['test_acc'].append(test_acc)
        history['epoch_time'].append(epoch_time)
    
    def save_training_history(self):
        """保存训练历史"""
        history_path = self.logs_dir / f"training_history_{self.timestamp}.json"
        with open(history_path, 'w') as f:
            json.dump(self.training_history, f, indent=2)
        
        # 也保存为pickle格式，便于后续分析
        pickle_path = self.logs_dir / f"training_history_{self.timestamp}.pkl"
        with open(pickle_path, 'wb') as f:
            pickle.dump(self.training_history, f)
        
        print(f"📊 训练历史已保存: {history_path}")
        return history_path
    
    def plot_training_curves(self, save_format='both'):
        """绘制训练曲线"""
        if not self.training_history:
            print("⚠️ 没有训练历史数据")
            return
        
        # 使用Plotly创建交互式图表
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('训练损失', '训练准确率', '测试准确率', '每轮训练时间'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        colors = px.colors.qualitative.Set1
        
        for i, (model_name, history) in enumerate(self.training_history.items()):
            color = colors[i % len(colors)]
            
            # 训练损失
            fig.add_trace(
                go.Scatter(x=history['epochs'], y=history['train_loss'],
                          name=f'{model_name} - 训练损失', line=dict(color=color)),
                row=1, col=1
            )
            
            # 训练准确率
            fig.add_trace(
                go.Scatter(x=history['epochs'], y=history['train_acc'],
                          name=f'{model_name} - 训练准确率', line=dict(color=color)),
                row=1, col=2
            )
            
            # 测试准确率
            fig.add_trace(
                go.Scatter(x=history['epochs'], y=history['test_acc'],
                          name=f'{model_name} - 测试准确率', line=dict(color=color, dash='dash')),
                row=2, col=1
            )
            
            # 每轮训练时间
            fig.add_trace(
                go.Scatter(x=history['epochs'], y=history['epoch_time'],
                          name=f'{model_name} - 训练时间', line=dict(color=color, dash='dot')),
                row=2, col=2
            )
        
        fig.update_layout(
            title=f'{self.experiment_name} 训练曲线',
            height=800,
            showlegend=True
        )
        
        # 保存交互式HTML图表
        if save_format in ['both', 'html']:
            html_path = self.plots_dir / f"training_curves_{self.timestamp}.html"
            fig.write_html(html_path)
            print(f"📈 交互式图表已保存: {html_path}")
        
        # 保存静态PNG图表
        if save_format in ['both', 'png']:
            png_path = self.plots_dir / f"training_curves_{self.timestamp}.png"
            fig.write_image(png_path, width=1200, height=800)
            print(f"📈 静态图表已保存: {png_path}")
        
        return fig
    
    def plot_comparison_bar(self, results_dict, title="模型性能对比"):
        """绘制模型性能对比柱状图"""
        models = list(results_dict.keys())
        accuracies = list(results_dict.values())
        
        fig = go.Figure(data=[
            go.Bar(x=models, y=accuracies, 
                   text=[f'{acc:.1f}%' for acc in accuracies],
                   textposition='auto',
                   marker_color=px.colors.qualitative.Set1[:len(models)])
        ])
        
        fig.update_layout(
            title=title,
            xaxis_title="模型",
            yaxis_title="准确率 (%)",
            yaxis=dict(range=[0, 100]),
            height=500
        )
        
        # 保存图表
        html_path = self.plots_dir / f"comparison_{self.timestamp}.html"
        png_path = self.plots_dir / f"comparison_{self.timestamp}.png"
        
        fig.write_html(html_path)
        fig.write_image(png_path, width=800, height=500)
        
        print(f"📊 对比图表已保存: {html_path}")
        return fig
    
    def save_experiment_summary(self, results_dict, config_dict=None, notes=""):
        """保存实验总结"""
        summary = {
            'experiment_name': self.experiment_name,
            'timestamp': self.timestamp,
            'results': results_dict,
            'config': config_dict or {},
            'notes': notes,
            'best_model': max(results_dict.items(), key=lambda x: x[1]) if results_dict else None
        }
        
        summary_path = self.logs_dir / f"experiment_summary_{self.timestamp}.json"
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        print(f"📋 实验总结已保存: {summary_path}")
        
        # 打印总结
        print("\n" + "="*60)
        print(f"🎉 {self.experiment_name} 实验总结")
        print("="*60)
        for model_name, accuracy in results_dict.items():
            print(f"{model_name:20s}: {accuracy:6.1f}%")
        
        if summary['best_model']:
            best_name, best_acc = summary['best_model']
            print(f"\n🏆 最佳模型: {best_name} ({best_acc:.1f}%)")
        
        return summary_path
    
    def generate_report(self, results_dict, config_dict=None, notes=""):
        """生成完整的实验报告"""
        print(f"\n📊 生成 {self.experiment_name} 实验报告...")
        
        # 保存训练历史
        self.save_training_history()
        
        # 绘制训练曲线
        self.plot_training_curves()
        
        # 绘制对比图表
        self.plot_comparison_bar(results_dict)
        
        # 保存实验总结
        self.save_experiment_summary(results_dict, config_dict, notes)
        
        print(f"✅ 实验报告生成完成，保存在: {self.experiment_dir}")
        return self.experiment_dir

# 使用示例
def example_usage():
    """使用示例"""
    # 创建结果管理器
    rm = ResultsManager("ssc_experiment")
    
    # 模拟训练过程
    for epoch in range(5):
        train_loss = 1.0 - epoch * 0.1
        train_acc = 0.5 + epoch * 0.1
        test_acc = 0.4 + epoch * 0.08
        epoch_time = 120.0
        
        rm.log_training_step("Vanilla SNN", epoch, train_loss, train_acc, test_acc, epoch_time)
        rm.log_training_step("DH-SNN", epoch, train_loss * 0.9, train_acc + 0.05, test_acc + 0.1, epoch_time * 1.1)
    
    # 生成报告
    results = {"Vanilla SNN": 70.5, "DH-SNN": 78.2}
    config = {"batch_size": 100, "learning_rate": 0.01}
    rm.generate_report(results, config, "SSC数据集实验")

if __name__ == "__main__":
    example_usage()
