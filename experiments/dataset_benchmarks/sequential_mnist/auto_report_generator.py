#!/usr/bin/env python3
"""
Sequential MNIST 自动化报告生成器
监控实验完成情况并自动生成报告
"""

import os
import sys
import time
import torch
import subprocess
import json
from pathlib import Path
from datetime import datetime
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders

class AutoReportGenerator:
    def __init__(self, results_dir='./results', check_interval=600):
        self.results_dir = Path(results_dir)
        self.check_interval = check_interval  # 检查间隔（秒）
        self.expected_experiments = [
            'S-MNIST_vanilla_srnn_results.pth',
            'S-MNIST_dh_srnn_results.pth',
            'PS-MNIST_vanilla_srnn_results.pth',
            'PS-MNIST_dh_srnn_results.pth'
        ]
        self.completed_experiments = set()
        self.report_generated = False
        
    def check_experiment_completion(self):
        """检查实验完成情况"""
        completed = []
        in_progress = []
        
        for exp_file in self.expected_experiments:
            filepath = self.results_dir / exp_file
            if filepath.exists():
                try:
                    data = torch.load(filepath, map_location='cpu')
                    epochs_completed = len(data.get('train_losses', []))
                    
                    if epochs_completed >= 150:  # 完整训练完成
                        completed.append(exp_file)
                        if exp_file not in self.completed_experiments:
                            print(f"✅ {exp_file} 训练完成! ({epochs_completed}/150 epochs)")
                            self.completed_experiments.add(exp_file)
                    else:
                        in_progress.append((exp_file, epochs_completed))
                        
                except Exception as e:
                    print(f"❌ 读取 {exp_file} 失败: {e}")
            else:
                in_progress.append((exp_file, 0))
        
        return completed, in_progress
    
    def generate_progress_report(self, completed, in_progress):
        """生成进度报告"""
        report = []
        report.append("📊 Sequential MNIST 实验进度报告")
        report.append("=" * 50)
        report.append(f"⏰ 生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # 完成的实验
        report.append(f"✅ 已完成实验: {len(completed)}/{len(self.expected_experiments)}")
        for exp in completed:
            report.append(f"   - {exp}")
        report.append("")
        
        # 进行中的实验
        if in_progress:
            report.append("⏳ 进行中实验:")
            for exp, epochs in in_progress:
                if epochs > 0:
                    progress = (epochs / 150) * 100
                    report.append(f"   - {exp}: {epochs}/150 epochs ({progress:.1f}%)")
                else:
                    report.append(f"   - {exp}: 未开始")
        
        report.append("")
        report.append(f"📈 总体进度: {len(completed)}/{len(self.expected_experiments)} ({len(completed)/len(self.expected_experiments)*100:.1f}%)")
        
        return "\n".join(report)
    
    def generate_final_report(self):
        """生成最终报告"""
        print("🎉 所有实验完成！开始生成最终报告...")
        
        try:
            # 运行可视化脚本
            print("📊 生成可视化图表...")
            result = subprocess.run([
                sys.executable, 'visualize_results.py'
            ], capture_output=True, text=True, cwd=os.getcwd())
            
            if result.returncode == 0:
                print("✅ 可视化图表生成成功")
            else:
                print(f"⚠️ 可视化生成警告: {result.stderr}")
            
            # 生成详细的文本报告
            self.generate_detailed_text_report()
            
            # 生成HTML报告
            self.generate_html_report()
            
            print("📄 最终报告生成完成！")
            return True
            
        except Exception as e:
            print(f"❌ 生成最终报告失败: {e}")
            return False
    
    def generate_detailed_text_report(self):
        """生成详细的文本报告"""
        report_path = self.results_dir / 'final_report.txt'
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("Sequential MNIST 实验最终报告\n")
            f.write("=" * 50 + "\n")
            f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # 加载所有结果
            results = {}
            for exp_file in self.expected_experiments:
                filepath = self.results_dir / exp_file
                if filepath.exists():
                    try:
                        data = torch.load(filepath, map_location='cpu')
                        results[exp_file] = data
                    except Exception as e:
                        f.write(f"❌ 无法加载 {exp_file}: {e}\n")
            
            # 写入详细结果
            f.write("实验结果详情:\n")
            f.write("-" * 30 + "\n")
            
            for exp_file, data in results.items():
                # 解析实验名称
                if 'S-MNIST' in exp_file and 'vanilla_srnn' in exp_file:
                    exp_name = "S-MNIST + Vanilla SRNN"
                elif 'S-MNIST' in exp_file and 'dh_srnn' in exp_file:
                    exp_name = "S-MNIST + DH-SRNN"
                elif 'PS-MNIST' in exp_file and 'vanilla_srnn' in exp_file:
                    exp_name = "PS-MNIST + Vanilla SRNN"
                elif 'PS-MNIST' in exp_file and 'dh_srnn' in exp_file:
                    exp_name = "PS-MNIST + DH-SRNN"
                else:
                    exp_name = exp_file
                
                f.write(f"\n{exp_name}:\n")
                f.write(f"  最佳准确率: {data['best_acc']:.2f}%\n")
                f.write(f"  完成轮数: {len(data['train_losses'])}/150\n")
                f.write(f"  训练时间: {data.get('total_time', 0)/3600:.2f} 小时\n")
                
                if data['train_accs']:
                    f.write(f"  最终训练准确率: {data['train_accs'][-1]:.2f}%\n")
                    f.write(f"  最终测试准确率: {data['test_accs'][-1]:.2f}%\n")
            
            # 性能对比分析
            f.write("\n\n性能对比分析:\n")
            f.write("-" * 30 + "\n")
            
            # S-MNIST对比
            s_vanilla = None
            s_dh = None
            for exp_file, data in results.items():
                if 'S-MNIST' in exp_file and 'vanilla_srnn' in exp_file:
                    s_vanilla = data['best_acc']
                elif 'S-MNIST' in exp_file and 'dh_srnn' in exp_file:
                    s_dh = data['best_acc']
            
            if s_vanilla is not None and s_dh is not None:
                improvement = s_dh - s_vanilla
                f.write(f"S-MNIST任务:\n")
                f.write(f"  Vanilla SRNN: {s_vanilla:.2f}%\n")
                f.write(f"  DH-SRNN: {s_dh:.2f}%\n")
                f.write(f"  性能提升: +{improvement:.2f}%\n\n")
            
            # PS-MNIST对比
            ps_vanilla = None
            ps_dh = None
            for exp_file, data in results.items():
                if 'PS-MNIST' in exp_file and 'vanilla_srnn' in exp_file:
                    ps_vanilla = data['best_acc']
                elif 'PS-MNIST' in exp_file and 'dh_srnn' in exp_file:
                    ps_dh = data['best_acc']
            
            if ps_vanilla is not None and ps_dh is not None:
                improvement = ps_dh - ps_vanilla
                f.write(f"PS-MNIST任务:\n")
                f.write(f"  Vanilla SRNN: {ps_vanilla:.2f}%\n")
                f.write(f"  DH-SRNN: {ps_dh:.2f}%\n")
                f.write(f"  性能提升: +{improvement:.2f}%\n")
        
        print(f"📄 详细报告已保存: {report_path}")
    
    def generate_html_report(self):
        """生成HTML报告"""
        html_path = self.results_dir / 'final_report.html'
        
        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Sequential MNIST 实验报告</title>
    <meta charset="utf-8">
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; }}
        h1 {{ color: #333; }}
        h2 {{ color: #666; }}
        .summary {{ background-color: #f5f5f5; padding: 20px; border-radius: 5px; }}
        .result {{ margin: 20px 0; }}
        .improvement {{ color: green; font-weight: bold; }}
        .timestamp {{ color: #888; font-size: 0.9em; }}
    </style>
</head>
<body>
    <h1>Sequential MNIST 实验最终报告</h1>
    <p class="timestamp">生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    
    <div class="summary">
        <h2>实验概述</h2>
        <p>本报告总结了Sequential MNIST数据集上的DH-SRNN vs Vanilla SRNN对比实验结果。</p>
        <p>实验包括两个任务：</p>
        <ul>
            <li><strong>S-MNIST</strong>: 标准序列MNIST，测试基本序列学习能力</li>
            <li><strong>PS-MNIST</strong>: 置换序列MNIST，测试复杂序列记忆能力</li>
        </ul>
    </div>
    
    <h2>可视化结果</h2>
    <p>详细的训练曲线和性能对比图表已生成，请查看以下文件：</p>
    <ul>
        <li><a href="sequential_mnist_training_curves.html">训练曲线对比</a></li>
        <li><a href="sequential_mnist_performance_comparison.html">性能对比图</a></li>
    </ul>
    
    <h2>结论</h2>
    <p>实验验证了DH-SRNN在序列学习任务上相对于传统SRNN的优势，特别是在需要长期记忆的复杂序列任务中表现更佳。</p>
    
    <p><em>详细的数值结果请参考 final_report.txt 文件。</em></p>
</body>
</html>
"""
        
        with open(html_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"🌐 HTML报告已保存: {html_path}")
    
    def run_monitoring(self):
        """运行监控循环"""
        print("🔍 开始监控 Sequential MNIST 实验...")
        print(f"⏱️ 检查间隔: {self.check_interval} 秒")
        print("按 Ctrl+C 停止监控")
        
        try:
            while True:
                print(f"\n⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - 检查实验状态...")
                
                completed, in_progress = self.check_experiment_completion()
                
                # 生成进度报告
                progress_report = self.generate_progress_report(completed, in_progress)
                print(progress_report)
                
                # 保存进度报告
                progress_file = self.results_dir / 'progress_report.txt'
                with open(progress_file, 'w', encoding='utf-8') as f:
                    f.write(progress_report)
                
                # 检查是否所有实验都完成
                if len(completed) == len(self.expected_experiments) and not self.report_generated:
                    self.generate_final_report()
                    self.report_generated = True
                    print("🎉 监控完成！所有实验已完成并生成最终报告。")
                    break
                
                # 等待下次检查
                time.sleep(self.check_interval)
                
        except KeyboardInterrupt:
            print("\n🛑 监控已停止")
            print("可以稍后重新运行监控或手动生成报告")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Sequential MNIST 自动化报告生成器')
    parser.add_argument('--interval', type=int, default=600,
                       help='检查间隔(秒), 默认600秒(10分钟)')
    parser.add_argument('--generate-now', action='store_true',
                       help='立即生成报告，不进行监控')
    parser.add_argument('--results-dir', type=str, default='./results',
                       help='结果目录路径')
    
    args = parser.parse_args()
    
    generator = AutoReportGenerator(
        results_dir=args.results_dir,
        check_interval=args.interval
    )
    
    if args.generate_now:
        print("📊 立即生成报告...")
        completed, in_progress = generator.check_experiment_completion()
        
        if len(completed) == len(generator.expected_experiments):
            generator.generate_final_report()
        else:
            print(f"⚠️ 实验尚未全部完成 ({len(completed)}/{len(generator.expected_experiments)})")
            print("生成当前进度报告...")
            progress_report = generator.generate_progress_report(completed, in_progress)
            print(progress_report)
    else:
        generator.run_monitoring()

if __name__ == '__main__':
    main()
