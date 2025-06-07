#!/usr/bin/env python3
"""
项目清理脚本
清除不必要的文件，整理代码结构
"""

import os
import shutil
from pathlib import Path

def cleanup_project():
    """清理项目结构"""
    
    print("🧹 开始清理项目结构...")
    
    # 当前目录
    project_root = Path(".")
    
    # 1. 清理重复和过时的实验文件
    experiments_to_remove = [
        "experiments/debug_ssc_data.py",           # 调试文件，已完成
        "experiments/detailed_ssc_analysis.py",    # 调试文件，已完成
        "experiments/demo_figure3.py",             # 演示文件，有更好版本
        "experiments/example_usage.py",            # 示例文件，有更好版本
        "experiments/final_shd_test.py",           # 旧版本，有更好版本
        "experiments/main.py",                     # 旧版主文件
        "experiments/quick_shd_test.py",           # 快速测试，已完成
        "experiments/train_example.py",            # 示例文件，有更好版本
        "experiments/quick_multi_timescale_test.py", # 有问题的版本
    ]
    
    print("\n📂 清理实验文件...")
    for file_path in experiments_to_remove:
        full_path = project_root / file_path
        if full_path.exists():
            print(f"  🗑️ 删除: {file_path}")
            full_path.unlink()
        else:
            print(f"  ⚠️ 未找到: {file_path}")
    
    # 2. 清理tasks目录（与core重复）
    tasks_dir = project_root / "tasks"
    if tasks_dir.exists():
        print(f"\n📂 清理tasks目录...")
        print(f"  🗑️ 删除整个tasks目录（与core重复）")
        shutil.rmtree(tasks_dir)
    
    # 3. 整理docs目录
    docs_to_remove = [
        "docs/ENHANCED_FIGURE3_SUMMARY.md",       # 过时文档
        "docs/FIGURE3_README.md",                 # 重复文档
        "docs/FIGURE3_REPRODUCTION_README.md",    # 重复文档
        "docs/IMPLEMENTATION_SUMMARY.md",         # 过时文档
        "docs/README.md",                         # 通用文档，有更好版本
    ]
    
    print("\n📂 清理文档文件...")
    for file_path in docs_to_remove:
        full_path = project_root / file_path
        if full_path.exists():
            print(f"  🗑️ 删除: {file_path}")
            full_path.unlink()
        else:
            print(f"  ⚠️ 未找到: {file_path}")
    
    # 4. 清理__pycache__目录
    print("\n📂 清理__pycache__目录...")
    for pycache_dir in project_root.rglob("__pycache__"):
        print(f"  🗑️ 删除: {pycache_dir}")
        shutil.rmtree(pycache_dir)
    
    # 5. 重命名和整理保留的文件
    print("\n📂 重命名和整理文件...")
    
    # 重命名核心实验文件
    renames = [
        ("experiments/spikingjelly_paper_equivalent.py", "experiments/shd_experiments.py"),
        ("experiments/ssc_figure4f_experiment.py", "experiments/ssc_experiments.py"),
        ("experiments/simple_multi_timescale_test.py", "experiments/multi_timescale_experiments.py"),
        ("experiments/figure3_plotly_reproduction.py", "experiments/figure3_experiments.py"),
    ]
    
    for old_name, new_name in renames:
        old_path = project_root / old_name
        new_path = project_root / new_name
        if old_path.exists() and not new_path.exists():
            print(f"  📝 重命名: {old_name} → {new_name}")
            old_path.rename(new_path)
        elif old_path.exists() and new_path.exists():
            print(f"  ⚠️ 目标文件已存在，跳过: {new_name}")
        else:
            print(f"  ⚠️ 源文件不存在: {old_name}")
    
    # 6. 创建新的目录结构
    print("\n📂 创建标准目录结构...")
    
    new_dirs = [
        "experiments/shd",
        "experiments/ssc", 
        "experiments/figure_reproduction",
        "data",
        "training",
        "visualization",
        "outputs/models",
        "outputs/logs",
    ]
    
    for dir_path in new_dirs:
        full_path = project_root / dir_path
        if not full_path.exists():
            print(f"  📁 创建目录: {dir_path}")
            full_path.mkdir(parents=True, exist_ok=True)
        else:
            print(f"  ✅ 目录已存在: {dir_path}")
    
    # 7. 移动文件到合适的目录
    print("\n📂 移动文件到合适目录...")
    
    moves = [
        ("experiments/shd_experiments.py", "experiments/shd/main_experiment.py"),
        ("experiments/ssc_experiments.py", "experiments/ssc/main_experiment.py"),
        ("experiments/figure3_experiments.py", "experiments/figure_reproduction/figure3.py"),
        ("experiments/paper_direct_test.py", "experiments/shd/paper_comparison.py"),
        ("experiments/fixed_ssc_experiment.py", "experiments/ssc/data_validation.py"),
    ]
    
    for old_path, new_path in moves:
        old_file = project_root / old_path
        new_file = project_root / new_path
        if old_file.exists() and not new_file.exists():
            print(f"  📦 移动: {old_path} → {new_path}")
            old_file.rename(new_file)
        elif old_file.exists() and new_file.exists():
            print(f"  ⚠️ 目标文件已存在，跳过: {new_path}")
        else:
            print(f"  ⚠️ 源文件不存在: {old_path}")
    
    print("\n🎉 项目清理完成!")
    
    # 8. 显示清理后的结构
    print("\n📊 清理后的项目结构:")
    show_directory_structure(project_root)

def show_directory_structure(path, prefix="", max_depth=3, current_depth=0):
    """显示目录结构"""
    if current_depth >= max_depth:
        return
    
    items = sorted([p for p in path.iterdir() if not p.name.startswith('.')])
    
    for i, item in enumerate(items):
        is_last = i == len(items) - 1
        current_prefix = "└── " if is_last else "├── "
        print(f"{prefix}{current_prefix}{item.name}")
        
        if item.is_dir() and current_depth < max_depth - 1:
            extension = "    " if is_last else "│   "
            show_directory_structure(item, prefix + extension, max_depth, current_depth + 1)

if __name__ == "__main__":
    cleanup_project()
