#!/usr/bin/env python3
"""
HTML转PNG转换器 - 尝试多种方法
"""

import os
import sys
import subprocess
from pathlib import Path
import time

def method1_selenium():
    """方法1: 使用Selenium + Chrome"""
    try:
        from selenium import webdriver
        from selenium.webdriver.chrome.options import Options
        
        print("🔧 尝试方法1: Selenium + Chrome")
        
        # 设置Chrome选项
        chrome_options = Options()
        chrome_options.add_argument('--headless')
        chrome_options.add_argument('--no-sandbox')
        chrome_options.add_argument('--disable-dev-shm-usage')
        chrome_options.add_argument('--window-size=1200,800')
        
        driver = webdriver.Chrome(options=chrome_options)
        return driver
        
    except Exception as e:
        print(f"❌ 方法1失败: {e}")
        return None

def method2_playwright():
    """方法2: 使用Playwright"""
    try:
        from playwright.sync_api import sync_playwright
        
        print("🔧 尝试方法2: Playwright")
        
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            return browser
            
    except Exception as e:
        print(f"❌ 方法2失败: {e}")
        return None

def method3_wkhtmltopdf():
    """方法3: 使用wkhtmltoimage"""
    try:
        print("🔧 尝试方法3: wkhtmltoimage")
        
        # 检查是否安装了wkhtmltoimage
        result = subprocess.run(['which', 'wkhtmltoimage'], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            return 'wkhtmltoimage'
        else:
            print("❌ wkhtmltoimage未安装")
            return None
            
    except Exception as e:
        print(f"❌ 方法3失败: {e}")
        return None

def method4_plotly_kaleido():
    """方法4: 重新尝试Plotly + Kaleido"""
    try:
        print("🔧 尝试方法4: Plotly + Kaleido")
        
        import plotly.graph_objects as go
        import plotly.io as pio
        
        # 测试简单图表
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=[1, 2, 3], y=[1, 2, 3]))
        
        # 尝试导出
        test_path = "test_plotly.png"
        fig.write_image(test_path, width=800, height=600)
        
        if os.path.exists(test_path):
            os.remove(test_path)
            return 'plotly'
        else:
            return None
            
    except Exception as e:
        print(f"❌ 方法4失败: {e}")
        return None

def convert_html_with_selenium(html_file, output_file):
    """使用Selenium转换HTML"""
    try:
        from selenium import webdriver
        from selenium.webdriver.chrome.options import Options
        
        chrome_options = Options()
        chrome_options.add_argument('--headless')
        chrome_options.add_argument('--no-sandbox')
        chrome_options.add_argument('--disable-dev-shm-usage')
        chrome_options.add_argument('--window-size=1200,800')
        
        driver = webdriver.Chrome(options=chrome_options)
        
        # 打开HTML文件
        file_url = f"file://{os.path.abspath(html_file)}"
        driver.get(file_url)
        
        # 等待页面加载
        time.sleep(3)
        
        # 截图
        driver.save_screenshot(output_file)
        driver.quit()
        
        return True
        
    except Exception as e:
        print(f"❌ Selenium转换失败: {e}")
        return False

def convert_html_with_wkhtmltoimage(html_file, output_file):
    """使用wkhtmltoimage转换HTML"""
    try:
        cmd = [
            'wkhtmltoimage',
            '--width', '1200',
            '--height', '800',
            '--quality', '100',
            html_file,
            output_file
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            return True
        else:
            print(f"❌ wkhtmltoimage错误: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"❌ wkhtmltoimage转换失败: {e}")
        return False

def find_html_files():
    """查找HTML文件"""
    html_files = []
    
    search_paths = [
        "results/",
        "experiments/dataset_benchmarks/figure_reproduction/figure3_delayed_xor/outputs/figures/",
        "experiments/legacy_spikingjelly/original_experiments/figure_reproduction/figure3_delayed_xor/outputs/figures/"
    ]
    
    for search_path in search_paths:
        if os.path.exists(search_path):
            for file in Path(search_path).glob("*.html"):
                html_files.append(file)
    
    return html_files

def main():
    """主函数"""
    print("🔄 HTML转PNG转换器")
    print("="*50)
    
    # 查找HTML文件
    html_files = find_html_files()
    print(f"📁 找到 {len(html_files)} 个HTML文件")
    
    if not html_files:
        print("⚠️ 未找到HTML文件")
        return
    
    # 创建输出目录
    output_dir = Path("DH-SNN_Reproduction_Report/figures")
    output_dir.mkdir(exist_ok=True)
    
    # 测试可用的转换方法
    print("\n🔍 测试转换方法...")
    
    methods = []
    
    # 测试方法3: wkhtmltoimage
    if method3_wkhtmltopdf():
        methods.append('wkhtmltoimage')
    
    # 测试方法1: Selenium
    if method1_selenium():
        methods.append('selenium')
    
    # 测试方法4: Plotly
    if method4_plotly_kaleido():
        methods.append('plotly')
    
    if not methods:
        print("❌ 没有可用的转换方法")
        print("\n💡 建议安装:")
        print("  • wkhtmltopdf: sudo apt-get install wkhtmltopdf")
        print("  • Chrome + Selenium: pip install selenium")
        print("  • 修复Kaleido: pip install --upgrade kaleido")
        return
    
    print(f"✅ 可用方法: {', '.join(methods)}")
    
    # 选择重要的HTML文件进行转换
    important_files = []
    for html_file in html_files:
        filename = html_file.name
        if any(keyword in filename for keyword in ['figure3_final', 'complete_figure4', 'performance_comparison']):
            important_files.append(html_file)
    
    if not important_files:
        important_files = html_files[:3]  # 取前3个
    
    print(f"\n🎨 转换重要文件 ({len(important_files)} 个)...")
    
    # 转换文件
    success_count = 0
    for html_file in important_files:
        output_file = output_dir / f"{html_file.stem}.png"
        
        print(f"📄 转换: {html_file.name} -> {output_file.name}")
        
        success = False
        
        # 尝试可用的方法
        for method in methods:
            if method == 'wkhtmltoimage':
                success = convert_html_with_wkhtmltoimage(str(html_file), str(output_file))
            elif method == 'selenium':
                success = convert_html_with_selenium(str(html_file), str(output_file))
            
            if success:
                print(f"  ✅ 成功 (使用 {method})")
                success_count += 1
                break
        
        if not success:
            print(f"  ❌ 转换失败")
    
    print(f"\n🎉 转换完成! 成功: {success_count}/{len(important_files)}")
    print(f"📁 输出目录: {output_dir}")

if __name__ == '__main__':
    main()
