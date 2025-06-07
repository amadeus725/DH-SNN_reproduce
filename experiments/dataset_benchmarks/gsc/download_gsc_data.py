#!/usr/bin/env python3
"""
下载和准备GSC (Google Speech Commands) 数据集
"""

import os
import sys
import urllib.request
import tarfile
import shutil
from pathlib import Path
import argparse

# 数据集信息
GSC_URL = "http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz"
GSC_FILENAME = "speech_commands_v0.02.tar.gz"

def download_file(url, filename, target_dir):
    """下载文件"""
    target_path = os.path.join(target_dir, filename)
    
    if os.path.exists(target_path):
        print(f"✅ 文件已存在: {target_path}")
        return target_path
    
    print(f"📥 开始下载: {url}")
    print(f"   目标路径: {target_path}")
    
    def progress_hook(block_num, block_size, total_size):
        downloaded = block_num * block_size
        if total_size > 0:
            percent = min(100, downloaded * 100 / total_size)
            print(f"\r   进度: {percent:.1f}% ({downloaded // 1024 // 1024}MB / {total_size // 1024 // 1024}MB)", end="")
    
    try:
        urllib.request.urlretrieve(url, target_path, progress_hook)
        print(f"\n✅ 下载完成: {target_path}")
        return target_path
    except Exception as e:
        print(f"\n❌ 下载失败: {e}")
        if os.path.exists(target_path):
            os.remove(target_path)
        return None

def extract_tar_gz(tar_path, extract_dir):
    """解压tar.gz文件"""
    print(f"📦 解压文件: {tar_path}")
    print(f"   目标目录: {extract_dir}")
    
    try:
        with tarfile.open(tar_path, 'r:gz') as tar:
            tar.extractall(extract_dir)
        print(f"✅ 解压完成")
        return True
    except Exception as e:
        print(f"❌ 解压失败: {e}")
        return False

def create_silence_data(data_dir):
    """创建silence数据"""
    print("🔇 创建silence数据...")
    
    silence_dir = os.path.join(data_dir, "_silence_")
    noise_dir = os.path.join(data_dir, "_background_noise_")
    
    if not os.path.exists(noise_dir):
        print(f"⚠️  背景噪声目录不存在: {noise_dir}")
        return False
    
    if os.path.exists(silence_dir):
        print(f"✅ Silence目录已存在: {silence_dir}")
        return True
    
    os.makedirs(silence_dir, exist_ok=True)
    
    # 这里应该实现silence数据生成逻辑
    # 由于复杂性，先创建空目录
    print(f"✅ 创建silence目录: {silence_dir}")
    print("⚠️  Silence数据生成需要额外实现")
    
    return True

def verify_data(data_dir):
    """验证数据完整性"""
    print("🔍 验证数据完整性...")
    
    # 检查必要的文件
    required_files = [
        "testing_list.txt",
        "validation_list.txt"
    ]
    
    for filename in required_files:
        filepath = os.path.join(data_dir, filename)
        if not os.path.exists(filepath):
            print(f"❌ 缺少必要文件: {filename}")
            return False
        print(f"✅ 找到文件: {filename}")
    
    # 检查命令词目录
    expected_commands = [
        "yes", "no", "up", "down", "left", "right", 
        "on", "off", "stop", "go", "zero", "one", 
        "two", "three", "four", "five", "six", 
        "seven", "eight", "nine"
    ]
    
    found_commands = []
    for command in expected_commands:
        command_dir = os.path.join(data_dir, command)
        if os.path.exists(command_dir) and os.path.isdir(command_dir):
            wav_files = [f for f in os.listdir(command_dir) if f.endswith('.wav')]
            found_commands.append(command)
            print(f"✅ 命令词 '{command}': {len(wav_files)} 个音频文件")
    
    print(f"📊 数据统计:")
    print(f"   找到命令词: {len(found_commands)}/{len(expected_commands)}")
    print(f"   命令词列表: {found_commands}")
    
    # 检查背景噪声
    noise_dir = os.path.join(data_dir, "_background_noise_")
    if os.path.exists(noise_dir):
        noise_files = [f for f in os.listdir(noise_dir) if f.endswith('.wav')]
        print(f"✅ 背景噪声: {len(noise_files)} 个文件")
    else:
        print(f"❌ 缺少背景噪声目录")
    
    return len(found_commands) >= 10  # 至少需要10个命令词

def setup_gsc_data(data_disk="/data", force_download=False):
    """设置GSC数据集"""
    print("🎯 设置GSC数据集")
    print("=" * 50)
    
    # 创建目录
    download_dir = os.path.join(data_disk, "downloads")
    extract_dir = os.path.join(data_disk, "speech_commands")
    
    os.makedirs(download_dir, exist_ok=True)
    os.makedirs(data_disk, exist_ok=True)
    
    print(f"📁 下载目录: {download_dir}")
    print(f"📁 数据目录: {extract_dir}")
    
    # 检查是否已存在
    if os.path.exists(extract_dir) and not force_download:
        print("🔍 检查现有数据...")
        if verify_data(extract_dir):
            print("✅ 数据已存在且完整，跳过下载")
            return extract_dir
        else:
            print("⚠️  现有数据不完整，重新下载")
    
    # 下载数据
    tar_path = download_file(GSC_URL, GSC_FILENAME, download_dir)
    if not tar_path:
        print("❌ 下载失败")
        return None
    
    # 解压数据
    if os.path.exists(extract_dir):
        print(f"🗑️  删除现有目录: {extract_dir}")
        shutil.rmtree(extract_dir)
    
    os.makedirs(extract_dir, exist_ok=True)
    
    if not extract_tar_gz(tar_path, extract_dir):
        print("❌ 解压失败")
        return None
    
    # 创建silence数据
    create_silence_data(extract_dir)
    
    # 验证数据
    if verify_data(extract_dir):
        print("✅ GSC数据集设置完成")
        
        # 清理下载文件
        try:
            os.remove(tar_path)
            print(f"🗑️  清理下载文件: {tar_path}")
        except:
            pass
        
        return extract_dir
    else:
        print("❌ 数据验证失败")
        return None

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="下载和设置GSC数据集")
    parser.add_argument("--data-disk", default="/data", help="数据盘路径")
    parser.add_argument("--force", action="store_true", help="强制重新下载")
    
    args = parser.parse_args()
    
    print("🎤 GSC数据集下载工具")
    print("=" * 50)
    print(f"数据盘: {args.data_disk}")
    print(f"强制下载: {args.force}")
    
    # 检查数据盘
    if not os.path.exists(args.data_disk):
        print(f"❌ 数据盘不存在: {args.data_disk}")
        print("请确保数据盘已挂载")
        return False
    
    # 检查磁盘空间
    stat = shutil.disk_usage(args.data_disk)
    free_gb = stat.free / (1024**3)
    print(f"💾 可用空间: {free_gb:.1f} GB")
    
    if free_gb < 5:
        print("⚠️  磁盘空间不足，建议至少5GB")
        response = input("是否继续? (y/N): ")
        if response.lower() != 'y':
            return False
    
    # 设置数据
    data_dir = setup_gsc_data(args.data_disk, args.force)
    
    if data_dir:
        print(f"\n🎉 成功!")
        print(f"GSC数据集路径: {data_dir}")
        print("\n📝 使用说明:")
        print("1. 在实验脚本中设置 GSC_DATA_PATH 为上述路径")
        print("2. 运行 GSC 实验:")
        print("   cd experiments/dataset_benchmarks/gsc")
        print("   python gsc_spikingjelly_experiment.py")
        return True
    else:
        print("\n❌ 设置失败")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
