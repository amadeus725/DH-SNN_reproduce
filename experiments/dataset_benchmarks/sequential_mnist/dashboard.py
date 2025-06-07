#!/usr/bin/env python3
"""
Sequential MNIST 实时监控仪表板
提供Web界面实时查看训练进度
"""

import os
import sys
import time
import torch
import json
from pathlib import Path
from datetime import datetime
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from flask import Flask, render_template_string, jsonify
import threading

class SequentialMNISTDashboard:
    def __init__(self, results_dir='./results', update_interval=30):
        self.results_dir = Path(results_dir)
        self.update_interval = update_interval
        self.app = Flask(__name__)
        self.latest_data = {}
        
        # 设置路由
        self.setup_routes()
        
    def load_current_results(self):
        """加载当前训练结果"""
        results = {}
        
        expected_files = [
            'S-MNIST_vanilla_srnn_results.pth',
            'S-MNIST_dh_srnn_results.pth', 
            'PS-MNIST_vanilla_srnn_results.pth',
            'PS-MNIST_dh_srnn_results.pth'
        ]
        
        for filename in expected_files:
            filepath = self.results_dir / filename
            if filepath.exists():
                try:
                    data = torch.load(filepath, map_location='cpu')
                    
                    # 解析实验信息
                    if 'S-MNIST' in filename and 'vanilla_srnn' in filename:
                        key = 'S-MNIST_Vanilla_SRNN'
                    elif 'S-MNIST' in filename and 'dh_srnn' in filename:
                        key = 'S-MNIST_DH-SRNN'
                    elif 'PS-MNIST' in filename and 'vanilla_srnn' in filename:
                        key = 'PS-MNIST_Vanilla_SRNN'
                    elif 'PS-MNIST' in filename and 'dh_srnn' in filename:
                        key = 'PS-MNIST_DH-SRNN'
                    else:
                        key = filename
                    
                    results[key] = {
                        'epochs_completed': len(data.get('train_losses', [])),
                        'best_acc': data.get('best_acc', 0),
                        'train_losses': data.get('train_losses', []),
                        'train_accs': data.get('train_accs', []),
                        'test_losses': data.get('test_losses', []),
                        'test_accs': data.get('test_accs', []),
                        'training_time': data.get('total_time', 0),
                        'last_updated': datetime.now().isoformat()
                    }
                    
                except Exception as e:
                    print(f"Error loading {filename}: {e}")
        
        return results
    
    def create_live_plots(self, results):
        """创建实时图表"""
        if not results:
            return None, None
        
        # 训练曲线图
        training_fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('训练损失', '测试损失', '训练准确率', '测试准确率'),
            vertical_spacing=0.1
        )
        
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
        
        for i, (name, data) in enumerate(results.items()):
            if not data['train_losses']:
                continue
                
            epochs = list(range(1, len(data['train_losses']) + 1))
            color = colors[i % len(colors)]
            
            # 训练损失
            training_fig.add_trace(
                go.Scatter(x=epochs, y=data['train_losses'],
                          name=name, line=dict(color=color),
                          showlegend=True),
                row=1, col=1
            )
            
            # 测试损失
            training_fig.add_trace(
                go.Scatter(x=epochs, y=data['test_losses'],
                          name=name, line=dict(color=color),
                          showlegend=False),
                row=1, col=2
            )
            
            # 训练准确率
            training_fig.add_trace(
                go.Scatter(x=epochs, y=data['train_accs'],
                          name=name, line=dict(color=color),
                          showlegend=False),
                row=2, col=1
            )
            
            # 测试准确率
            training_fig.add_trace(
                go.Scatter(x=epochs, y=data['test_accs'],
                          name=name, line=dict(color=color),
                          showlegend=False),
                row=2, col=2
            )
        
        training_fig.update_layout(
            title='实时训练进度',
            height=600,
            showlegend=True
        )
        
        # 进度条图
        progress_data = []
        for name, data in results.items():
            progress = (data['epochs_completed'] / 150) * 100
            progress_data.append({
                'experiment': name,
                'progress': progress,
                'epochs': data['epochs_completed'],
                'best_acc': data['best_acc']
            })
        
        progress_fig = go.Figure()
        
        for item in progress_data:
            progress_fig.add_trace(go.Bar(
                x=[item['progress']],
                y=[item['experiment']],
                orientation='h',
                text=f"{item['epochs']}/150 epochs (Best: {item['best_acc']:.1f}%)",
                textposition='inside',
                name=item['experiment']
            ))
        
        progress_fig.update_layout(
            title='实验进度',
            xaxis_title='完成百分比 (%)',
            xaxis=dict(range=[0, 100]),
            height=400
        )
        
        return training_fig, progress_fig
    
    def setup_routes(self):
        """设置Flask路由"""
        
        @self.app.route('/')
        def dashboard():
            return render_template_string(DASHBOARD_HTML)
        
        @self.app.route('/api/data')
        def get_data():
            """API端点：获取最新数据"""
            results = self.load_current_results()
            
            # 创建图表
            training_fig, progress_fig = self.create_live_plots(results)
            
            response_data = {
                'timestamp': datetime.now().isoformat(),
                'results': results,
                'training_plot': training_fig.to_json() if training_fig else None,
                'progress_plot': progress_fig.to_json() if progress_fig else None
            }
            
            return jsonify(response_data)
        
        @self.app.route('/api/summary')
        def get_summary():
            """API端点：获取实验摘要"""
            results = self.load_current_results()
            
            summary = {
                'total_experiments': 4,
                'completed_experiments': sum(1 for r in results.values() if r['epochs_completed'] >= 150),
                'in_progress_experiments': sum(1 for r in results.values() if 0 < r['epochs_completed'] < 150),
                'not_started_experiments': 4 - len(results),
                'best_accuracy': max([r['best_acc'] for r in results.values()]) if results else 0,
                'total_training_time': sum([r['training_time'] for r in results.values()]) / 3600,  # 小时
                'last_updated': datetime.now().isoformat()
            }
            
            return jsonify(summary)
    
    def run_dashboard(self, host='0.0.0.0', port=5000, debug=False):
        """运行仪表板"""
        print(f"🌐 启动 Sequential MNIST 监控仪表板...")
        print(f"📍 访问地址: http://{host}:{port}")
        print(f"🔄 数据更新间隔: {self.update_interval} 秒")
        
        self.app.run(host=host, port=port, debug=debug)

# HTML模板
DASHBOARD_HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>Sequential MNIST 监控仪表板</title>
    <meta charset="utf-8">
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }
        .header {
            background-color: #2c3e50;
            color: white;
            padding: 20px;
            border-radius: 5px;
            margin-bottom: 20px;
        }
        .summary-cards {
            display: flex;
            gap: 20px;
            margin-bottom: 20px;
            flex-wrap: wrap;
        }
        .card {
            background-color: white;
            padding: 20px;
            border-radius: 5px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            flex: 1;
            min-width: 200px;
        }
        .card h3 {
            margin: 0 0 10px 0;
            color: #2c3e50;
        }
        .card .value {
            font-size: 2em;
            font-weight: bold;
            color: #3498db;
        }
        .plots {
            display: flex;
            flex-direction: column;
            gap: 20px;
        }
        .plot-container {
            background-color: white;
            padding: 20px;
            border-radius: 5px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .status {
            color: #27ae60;
            font-weight: bold;
        }
        .last-updated {
            color: #7f8c8d;
            font-size: 0.9em;
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>🧠 Sequential MNIST 实时监控仪表板</h1>
        <p>DH-SRNN vs Vanilla SRNN 对比实验</p>
        <p class="last-updated" id="lastUpdated">最后更新: --</p>
    </div>
    
    <div class="summary-cards">
        <div class="card">
            <h3>总实验数</h3>
            <div class="value" id="totalExperiments">4</div>
        </div>
        <div class="card">
            <h3>已完成</h3>
            <div class="value" id="completedExperiments">0</div>
        </div>
        <div class="card">
            <h3>进行中</h3>
            <div class="value" id="inProgressExperiments">0</div>
        </div>
        <div class="card">
            <h3>最佳准确率</h3>
            <div class="value" id="bestAccuracy">0%</div>
        </div>
    </div>
    
    <div class="plots">
        <div class="plot-container">
            <div id="progressPlot"></div>
        </div>
        <div class="plot-container">
            <div id="trainingPlot"></div>
        </div>
    </div>
    
    <script>
        function updateDashboard() {
            fetch('/api/summary')
                .then(response => response.json())
                .then(data => {
                    document.getElementById('totalExperiments').textContent = data.total_experiments;
                    document.getElementById('completedExperiments').textContent = data.completed_experiments;
                    document.getElementById('inProgressExperiments').textContent = data.in_progress_experiments;
                    document.getElementById('bestAccuracy').textContent = data.best_accuracy.toFixed(1) + '%';
                    document.getElementById('lastUpdated').textContent = '最后更新: ' + new Date(data.last_updated).toLocaleString();
                });
            
            fetch('/api/data')
                .then(response => response.json())
                .then(data => {
                    if (data.progress_plot) {
                        Plotly.newPlot('progressPlot', JSON.parse(data.progress_plot));
                    }
                    if (data.training_plot) {
                        Plotly.newPlot('trainingPlot', JSON.parse(data.training_plot));
                    }
                });
        }
        
        // 初始加载
        updateDashboard();
        
        // 每30秒更新一次
        setInterval(updateDashboard, 30000);
    </script>
</body>
</html>
"""

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Sequential MNIST 监控仪表板')
    parser.add_argument('--host', type=str, default='0.0.0.0',
                       help='服务器主机地址')
    parser.add_argument('--port', type=int, default=5000,
                       help='服务器端口')
    parser.add_argument('--results-dir', type=str, default='./results',
                       help='结果目录路径')
    parser.add_argument('--update-interval', type=int, default=30,
                       help='数据更新间隔(秒)')
    
    args = parser.parse_args()
    
    dashboard = SequentialMNISTDashboard(
        results_dir=args.results_dir,
        update_interval=args.update_interval
    )
    
    dashboard.run_dashboard(host=args.host, port=args.port)

if __name__ == '__main__':
    main()
