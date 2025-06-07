#!/usr/bin/env python3
"""
简单的HTML转PNG转换器
"""

import os
import sys
import subprocess
from pathlib import Path

def method_wkhtmltoimage():
    """检查wkhtmltoimage"""
    try:
        result = subprocess.run(['which', 'wkhtmltoimage'], 
                              capture_output=True, text=True)
        return result.returncode == 0
    except:
        return False

def method_chrome():
    """检查Chrome"""
    chrome_paths = [
        '/usr/bin/google-chrome',
        '/usr/bin/chromium-browser', 
        '/usr/bin/chrome'
    ]
    
    for path in chrome_paths:
        if os.path.exists(path):
            return path
    return None

def convert_with_wkhtmltoimage(html_file, output_file):
    """使用wkhtmltoimage转换"""
    try:
        cmd = [
            'wkhtmltoimage',
            '--width', '1200',
            '--height', '800',
            '--quality', '100',
            str(html_file),
            str(output_file)
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        return result.returncode == 0
    except:
        return False

def convert_with_chrome(html_file, output_file, chrome_path):
    """使用Chrome转换"""
    try:
        cmd = [
            chrome_path,
            '--headless',
            '--disable-gpu',
            '--no-sandbox',
            '--window-size=1200,800',
            '--screenshot=' + str(output_file),
            'file://' + str(html_file.absolute())
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        return result.returncode == 0 and output_file.exists()
    except:
        return False

def find_html_files():
    """查找HTML文件"""
    html_files = []
    search_paths = [
        "../results/",
        "../experiments/dataset_benchmarks/figure_reproduction/figure3_delayed_xor/outputs/figures/",
        "../experiments/legacy_spikingjelly/original_experiments/figure_reproduction/figure3_delayed_xor/outputs/figures/"
    ]
    
    for search_path in search_paths:
        path = Path(search_path)
        if path.exists():
            html_files.extend(path.glob("*.html"))
    
    # 筛选重要文件
    important = []
    keywords = ['figure3_final', 'complete_figure4', 'performance_comparison']
    
    for html_file in html_files:
        if any(k in html_file.name for k in keywords):
            important.append(html_file)
    
    return important if important else html_files[:3]

def main():
    print("🎨 HTML转PNG转换器")
    print("="*40)
    
    # 查找文件
    html_files = find_html_files()
    print(f"📁 找到 {len(html_files)} 个HTML文件")
    
    if not html_files:
        print("⚠️ 未找到HTML文件")
        return
    
    # 创建输出目录
    output_dir = Path("../DH-SNN_Reproduction_Report/figures")
    output_dir.mkdir(exist_ok=True)
    
    # 检测转换工具
    print("\n🔍 检测转换工具...")
    
    has_wkhtmltoimage = method_wkhtmltoimage()
    chrome_path = method_chrome()
    
    if has_wkhtmltoimage:
        print("✅ wkhtmltoimage 可用")
    else:
        print("❌ wkhtmltoimage 不可用")
    
    if chrome_path:
        print(f"✅ Chrome 可用: {chrome_path}")
    else:
        print("❌ Chrome 不可用")
    
    if not has_wkhtmltoimage and not chrome_path:
        print("\n❌ 没有可用的转换工具")
        print("💡 请安装:")
        print("  sudo apt-get install wkhtmltopdf")
        print("  或安装 Google Chrome")
        return
    
    # 转换文件
    print(f"\n🔄 转换 {len(html_files)} 个文件...")
    success_count = 0
    
    for html_file in html_files:
        output_file = output_dir / f"{html_file.stem}.png"
        print(f"📄 {html_file.name} -> {output_file.name}")
        
        success = False
        
        if has_wkhtmltoimage:
            success = convert_with_wkhtmltoimage(html_file, output_file)
            if success:
                print("  ✅ wkhtmltoimage 成功")
        
        if not success and chrome_path:
            success = convert_with_chrome(html_file, output_file, chrome_path)
            if success:
                print("  ✅ Chrome 成功")
        
        if success:
            success_count += 1
        else:
            print("  ❌ 转换失败")
    
    print(f"\n🎉 完成! 成功: {success_count}/{len(html_files)}")
    
    # 列出生成的文件
    png_files = list(output_dir.glob("*.png"))
    if png_files:
        print(f"\n📋 生成的PNG文件:")
        for png_file in png_files:
            print(f"  • {png_file.name}")

if __name__ == '__main__':
    main()
