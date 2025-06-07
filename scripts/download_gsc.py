#!/usr/bin/env python3
"""
下载Google Speech Commands (GSC)数据集
"""

import requests
import tarfile
import os
from pathlib import Path
import time

def download_gsc():
    """下载GSC数据集"""
    
    print("📥 开始下载Google Speech Commands (GSC)数据集...")
    print("="*60)
    
    # 创建数据集目录
    datasets_dir = Path("../datasets")
    datasets_dir.mkdir(exist_ok=True)
    
    gsc_dir = datasets_dir / "gsc"
    gsc_dir.mkdir(exist_ok=True)
    
    print(f"📁 数据保存路径: {gsc_dir.absolute()}")
    
    # GSC v0.02下载链接
    gsc_url = "http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz"
    gsc_file = gsc_dir / "speech_commands_v0.02.tar.gz"
    
    try:
        if gsc_file.exists():
            print(f"✅ GSC压缩文件已存在: {gsc_file}")
            file_size = gsc_file.stat().st_size / (1024 * 1024)
            print(f"  文件大小: {file_size:.1f} MB")
        else:
            print(f"📥 开始下载GSC数据集...")
            print(f"  下载地址: {gsc_url}")
            print(f"  预计大小: ~2.3 GB")
            print(f"  这可能需要几分钟时间...")
            
            start_time = time.time()
            
            # 使用requests下载，支持断点续传
            headers = {}
            if gsc_file.exists():
                headers['Range'] = f'bytes={gsc_file.stat().st_size}-'
            
            response = requests.get(gsc_url, headers=headers, stream=True)
            total_size = int(response.headers.get('content-length', 0))
            
            if 'content-range' in response.headers:
                total_size = int(response.headers['content-range'].split('/')[-1])
            
            print(f"  总大小: {total_size / (1024*1024):.1f} MB")
            
            mode = 'ab' if 'Range' in headers else 'wb'
            downloaded = gsc_file.stat().st_size if gsc_file.exists() else 0
            
            with open(gsc_file, mode) as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        
                        # 显示进度
                        if total_size > 0:
                            percent = (downloaded / total_size) * 100
                            speed = downloaded / (time.time() - start_time + 1) / (1024*1024)
                            print(f"\r  📥 进度: {percent:5.1f}% ({downloaded/(1024*1024):6.1f}/{total_size/(1024*1024):6.1f} MB) 速度: {speed:5.1f} MB/s", 
                                  end='', flush=True)
            
            print(f"\n✅ GSC下载完成!")
            download_time = time.time() - start_time
            print(f"  下载时间: {download_time/60:.1f} 分钟")
        
        # 检查是否需要解压
        extracted_dir = gsc_dir / "speech_commands_v0.02"
        if not extracted_dir.exists():
            print(f"\n📂 解压GSC数据集...")
            print(f"  这可能需要几分钟时间...")
            
            extract_start = time.time()
            
            with tarfile.open(gsc_file, 'r:gz') as tar:
                # 获取所有成员
                members = tar.getmembers()
                total_members = len(members)
                
                print(f"  总文件数: {total_members}")
                
                # 解压所有文件
                for i, member in enumerate(members):
                    tar.extract(member, gsc_dir)
                    
                    if i % 1000 == 0 or i == total_members - 1:
                        percent = (i + 1) / total_members * 100
                        print(f"\r  📂 解压进度: {percent:5.1f}% ({i+1}/{total_members})", end='', flush=True)
            
            extract_time = time.time() - extract_start
            print(f"\n✅ GSC解压完成!")
            print(f"  解压时间: {extract_time/60:.1f} 分钟")
        else:
            print(f"✅ GSC数据集已解压: {extracted_dir}")
        
        # 验证数据
        print(f"\n🔍 验证GSC数据集...")
        
        # 检查目录结构
        if extracted_dir.exists():
            subdirs = [d for d in extracted_dir.iterdir() if d.is_dir()]
            wav_files = list(extracted_dir.rglob("*.wav"))
            
            print(f"  数据目录: {len(subdirs)} 个")
            print(f"  音频文件: {len(wav_files)} 个")
            
            # 显示前10个类别
            print(f"  前10个类别:")
            for i, subdir in enumerate(sorted(subdirs)[:10]):
                file_count = len(list(subdir.glob("*.wav")))
                print(f"    {subdir.name}: {file_count} 文件")
            
            if len(subdirs) > 10:
                print(f"    ... 还有 {len(subdirs)-10} 个类别")
        
        # 检查总大小
        print(f"\n💾 检查文件大小:")
        total_size = 0
        for file_path in gsc_dir.rglob("*"):
            if file_path.is_file():
                total_size += file_path.stat().st_size
        
        print(f"  压缩文件: {gsc_file.stat().st_size / (1024*1024):.1f} MB")
        print(f"  解压后总大小: {total_size / (1024*1024):.1f} MB")
        
        print(f"\n🎉 GSC数据集准备完成!")
        print(f"📍 数据位置: {gsc_dir.absolute()}")
        
        return True
        
    except Exception as e:
        print(f"\n❌ GSC下载失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def create_gsc_readme():
    """创建GSC使用说明"""
    
    readme_content = """# 🎤 Google Speech Commands (GSC) 数据集

## 📋 数据集信息
- **名称**: Google Speech Commands Dataset v0.02
- **类别数**: 35个语音命令
- **音频文件**: ~105,000个.wav文件
- **采样率**: 16kHz
- **时长**: 1秒
- **格式**: 16-bit PCM WAV

## 📊 主要类别
包含以下语音命令：
- 数字: zero, one, two, three, four, five, six, seven, eight, nine
- 方向: left, right, up, down
- 动作: go, stop, yes, no
- 其他: on, off, cat, dog, bird, bed, house, tree, happy, marvin, sheila, wow

## 🔄 数据预处理
GSC数据需要转换为脉冲序列：

### 1. 音频预处理
```python
import librosa
import numpy as np

# 加载音频
audio, sr = librosa.load(wav_file, sr=16000)

# 提取MFCC特征
mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)

# 转换为脉冲序列
spikes = convert_to_spikes(mfcc)
```

### 2. 脉冲编码
- **时间窗口**: 1秒音频 → 100个时间步
- **特征维度**: MFCC 13维 → 120维输入
- **编码方式**: 率编码或时间编码

## 📈 DH-SNN实验配置

### 网络架构
- **输入**: 120维 (MFCC特征)
- **隐藏层**: 200个神经元 × 3层
- **输出**: 15个类别 (选择主要命令)
- **分支数**: 1-8个树突分支

### 预期性能
- **Vanilla SNN**: ~85%
- **DH-SNN**: ~94%

## 📁 文件结构
```
gsc/
├── speech_commands_v0.02.tar.gz    # 原始压缩文件
├── speech_commands_v0.02/          # 解压后数据
│   ├── zero/                       # 各类别目录
│   ├── one/
│   ├── two/
│   └── ...
└── README.md                       # 本文件
```

## 🚀 使用示例

### 加载数据
```python
import os
import librosa
from pathlib import Path

gsc_dir = Path("../datasets/gsc/speech_commands_v0.02")
categories = ["yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go"]

for category in categories:
    category_dir = gsc_dir / category
    wav_files = list(category_dir.glob("*.wav"))
    print(f"{category}: {len(wav_files)} files")
```

### 特征提取
```python
def extract_features(wav_file):
    audio, sr = librosa.load(wav_file, sr=16000)
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
    return mfcc.T  # [time_steps, features]
```

## 💡 注意事项
1. 数据集较大 (~2.3GB)，下载需要时间
2. 包含背景噪声和未知词汇类别
3. 建议选择主要的10-15个类别进行实验
4. 需要音频预处理和特征提取
"""
    
    gsc_dir = Path("../datasets/gsc")
    readme_file = gsc_dir / "README.md"
    
    with open(readme_file, 'w', encoding='utf-8') as f:
        f.write(readme_content)
    
    print(f"📝 创建GSC使用说明: {readme_file}")

def main():
    """主函数"""
    
    print("🚀 Google Speech Commands (GSC) 下载器")
    print("="*60)
    
    # 下载GSC
    success = download_gsc()
    
    if success:
        # 创建使用说明
        create_gsc_readme()
        
        print(f"\n✅ GSC数据集准备完成!")
        print(f"💡 下一步:")
        print(f"  - 可以开始GSC分类实验")
        print(f"  - 需要音频预处理和特征提取")
        print(f"  - 数据已准备就绪，位于 ../datasets/gsc/")
    else:
        print(f"\n❌ 下载失败，请检查网络连接")

if __name__ == '__main__':
    main()
