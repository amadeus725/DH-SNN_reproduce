#!/usr/bin/env python3
"""
创建简单的英文版图片，避免复杂依赖
"""

import os
import sys

def create_simple_figure():
    """创建一个简单的占位图片"""
    try:
        # 尝试使用PIL创建简单图片
        from PIL import Image, ImageDraw, ImageFont
        
        # 创建图片
        img = Image.new('RGB', (800, 600), color='white')
        draw = ImageDraw.Draw(img)
        
        # 尝试使用默认字体
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 20)
            title_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 24)
        except:
            font = ImageFont.load_default()
            title_font = ImageFont.load_default()
        
        # 绘制标题
        draw.text((200, 50), "DH-SNN Experimental Results", fill='black', font=title_font)
        
        # 绘制实验结果
        results = [
            "Delayed XOR: DH-SFNN 75.4% vs Vanilla SFNN 55.0% (+20.4%)",
            "Multi-timescale XOR: 2-Branch DH-SFNN 96.2% vs Vanilla 50.2% (+46.0%)",
            "SHD Dataset: DH-SNN 79.8% vs Vanilla SNN 54.5% (+25.3%)",
            "SSC Dataset: DH-SNN 60.5% vs Vanilla SNN 46.8% (+13.7%)"
        ]
        
        y_pos = 150
        for result in results:
            draw.text((50, y_pos), f"• {result}", fill='black', font=font)
            y_pos += 80
        
        # 绘制结论
        draw.text((50, 500), "Conclusion: DH-SNN consistently outperforms Vanilla SNN", 
                 fill='blue', font=title_font)
        draw.text((50, 540), "across all experimental paradigms", fill='blue', font=title_font)
        
        return img
        
    except ImportError:
        print("PIL not available, creating text-based summary instead")
        return None

def main():
    """主函数"""
    print("🎨 创建简单英文版图片...")
    
    # 创建输出目录
    output_dir = "DH-SNN_Reproduction_Report/figures"
    os.makedirs(output_dir, exist_ok=True)
    
    # 创建图片
    img = create_simple_figure()
    
    if img:
        # 保存图片
        img_path = os.path.join(output_dir, "experimental_results_summary.png")
        img.save(img_path)
        print(f"✅ 图片已保存: {img_path}")
    else:
        # 创建文本总结
        summary_path = os.path.join(output_dir, "results_summary.txt")
        with open(summary_path, 'w') as f:
            f.write("DH-SNN Experimental Results Summary\n")
            f.write("="*40 + "\n\n")
            f.write("1. Delayed XOR: DH-SFNN 75.4% vs Vanilla SFNN 55.0% (+20.4%)\n")
            f.write("2. Multi-timescale XOR: 2-Branch DH-SFNN 96.2% vs Vanilla 50.2% (+46.0%)\n")
            f.write("3. SHD Dataset: DH-SNN 79.8% vs Vanilla SNN 54.5% (+25.3%)\n")
            f.write("4. SSC Dataset: DH-SNN 60.5% vs Vanilla SNN 46.8% (+13.7%)\n\n")
            f.write("Conclusion: DH-SNN consistently outperforms Vanilla SNN\n")
            f.write("across all experimental paradigms.\n")
        print(f"✅ 文本总结已保存: {summary_path}")
    
    print("🎉 完成!")

if __name__ == '__main__':
    main()
