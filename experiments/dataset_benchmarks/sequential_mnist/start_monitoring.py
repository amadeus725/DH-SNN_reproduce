#!/usr/bin/env python3
"""
Sequential MNIST 一键启动监控系统
整合所有监控、可视化和报告功能
"""

import os
import sys
import time
import subprocess
import argparse
import signal
from pathlib import Path

class MonitoringSystem:
    def __init__(self):
        self.processes = []
        self.running = True
        
    def signal_handler(self, signum, frame):
        """处理中断信号"""
        print("\n🛑 收到中断信号，正在停止所有监控进程...")
        self.stop_all_processes()
        sys.exit(0)
    
    def start_process(self, command, name):
        """启动一个监控进程"""
        try:
            print(f"🚀 启动 {name}...")
            process = subprocess.Popen(
                command,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            self.processes.append((process, name))
            print(f"✅ {name} 已启动 (PID: {process.pid})")
            return process
        except Exception as e:
            print(f"❌ 启动 {name} 失败: {e}")
            return None
    
    def stop_all_processes(self):
        """停止所有监控进程"""
        for process, name in self.processes:
            try:
                print(f"🛑 停止 {name}...")
                process.terminate()
                process.wait(timeout=5)
                print(f"✅ {name} 已停止")
            except subprocess.TimeoutExpired:
                print(f"⚠️ 强制终止 {name}...")
                process.kill()
            except Exception as e:
                print(f"❌ 停止 {name} 失败: {e}")
    
    def check_dependencies(self):
        """检查依赖"""
        print("🔍 检查依赖...")
        
        required_packages = ['torch', 'plotly', 'flask', 'pandas', 'numpy']
        missing_packages = []
        
        for package in required_packages:
            try:
                __import__(package)
                print(f"✅ {package}")
            except ImportError:
                missing_packages.append(package)
                print(f"❌ {package} (缺失)")
        
        if missing_packages:
            print(f"\n⚠️ 缺失依赖包: {', '.join(missing_packages)}")
            print("请运行以下命令安装:")
            print(f"pip install {' '.join(missing_packages)}")
            return False
        
        print("✅ 所有依赖检查通过")
        return True
    
    def start_auto_report_generator(self, interval=600):
        """启动自动报告生成器"""
        command = f"python auto_report_generator.py --interval {interval}"
        return self.start_process(command, "自动报告生成器")
    
    def start_dashboard(self, host='0.0.0.0', port=5000):
        """启动Web仪表板"""
        command = f"python dashboard.py --host {host} --port {port}"
        return self.start_process(command, f"Web仪表板 (http://{host}:{port})")
    
    def start_periodic_visualization(self, interval=1800):
        """启动定期可视化更新"""
        command = f"""
while true; do
    python visualize_results.py
    echo "📊 可视化更新完成 - $(date)"
    sleep {interval}
done
"""
        return self.start_process(command, "定期可视化更新")
    
    def show_status(self):
        """显示当前状态"""
        print("\n📊 监控系统状态:")
        print("=" * 50)
        
        for process, name in self.processes:
            if process.poll() is None:
                print(f"✅ {name} - 运行中 (PID: {process.pid})")
            else:
                print(f"❌ {name} - 已停止")
        
        print(f"\n📁 结果目录: {os.path.abspath('./results')}")
        
        # 检查结果文件
        results_dir = Path('./results')
        if results_dir.exists():
            result_files = list(results_dir.glob('*_results.pth'))
            print(f"📄 结果文件: {len(result_files)} 个")
            
            for file in result_files:
                try:
                    import torch
                    data = torch.load(file, map_location='cpu')
                    epochs = len(data.get('train_losses', []))
                    best_acc = data.get('best_acc', 0)
                    print(f"   - {file.name}: {epochs}/150 epochs, 最佳准确率 {best_acc:.2f}%")
                except Exception as e:
                    print(f"   - {file.name}: 读取失败")
        else:
            print("📄 结果文件: 无")
    
    def run_monitoring_loop(self):
        """运行监控循环"""
        print("\n🔄 进入监控循环...")
        print("按 Ctrl+C 停止监控")
        
        try:
            while self.running:
                time.sleep(60)  # 每分钟检查一次
                
                # 检查进程状态
                for i, (process, name) in enumerate(self.processes):
                    if process.poll() is not None:
                        print(f"⚠️ {name} 意外停止，尝试重启...")
                        # 这里可以添加重启逻辑
                
        except KeyboardInterrupt:
            print("\n🛑 收到停止信号")
        finally:
            self.stop_all_processes()

def main():
    parser = argparse.ArgumentParser(description='Sequential MNIST 监控系统')
    parser.add_argument('--dashboard-port', type=int, default=5000,
                       help='Web仪表板端口')
    parser.add_argument('--dashboard-host', type=str, default='0.0.0.0',
                       help='Web仪表板主机')
    parser.add_argument('--report-interval', type=int, default=600,
                       help='报告生成间隔(秒)')
    parser.add_argument('--viz-interval', type=int, default=1800,
                       help='可视化更新间隔(秒)')
    parser.add_argument('--no-dashboard', action='store_true',
                       help='不启动Web仪表板')
    parser.add_argument('--no-auto-report', action='store_true',
                       help='不启动自动报告生成')
    parser.add_argument('--no-viz', action='store_true',
                       help='不启动定期可视化')
    parser.add_argument('--status-only', action='store_true',
                       help='只显示状态，不启动监控')
    
    args = parser.parse_args()
    
    print("🧠 Sequential MNIST 监控系统")
    print("=" * 50)
    
    # 创建监控系统
    monitor = MonitoringSystem()
    
    # 设置信号处理
    signal.signal(signal.SIGINT, monitor.signal_handler)
    signal.signal(signal.SIGTERM, monitor.signal_handler)
    
    # 检查依赖
    if not monitor.check_dependencies():
        sys.exit(1)
    
    # 创建结果目录
    os.makedirs('./results', exist_ok=True)
    
    if args.status_only:
        monitor.show_status()
        return
    
    print("\n🚀 启动监控组件...")
    
    # 启动自动报告生成器
    if not args.no_auto_report:
        monitor.start_auto_report_generator(args.report_interval)
    
    # 启动Web仪表板
    if not args.no_dashboard:
        monitor.start_dashboard(args.dashboard_host, args.dashboard_port)
        print(f"🌐 Web仪表板: http://{args.dashboard_host}:{args.dashboard_port}")
    
    # 启动定期可视化
    if not args.no_viz:
        monitor.start_periodic_visualization(args.viz_interval)
    
    # 显示初始状态
    time.sleep(2)  # 等待进程启动
    monitor.show_status()
    
    print(f"\n💡 使用说明:")
    print(f"   - Web仪表板: http://{args.dashboard_host}:{args.dashboard_port}")
    print(f"   - 结果目录: ./results/")
    print(f"   - 日志文件: ./results/progress_report.txt")
    print(f"   - 可视化图表: ./results/*.html")
    
    # 运行监控循环
    monitor.run_monitoring_loop()

if __name__ == '__main__':
    main()
