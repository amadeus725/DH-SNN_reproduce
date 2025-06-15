@echo off
chcp 65001 >nul
echo 🚀 DH-SNN Ultimate GitHub 上传准备
echo ========================================

REM 检查是否在正确的目录
if not exist "run_experiments.py" (
    echo ❌ 错误：请在 dh-snn-ultimate 目录中运行此脚本
    pause
    exit /b 1
)

REM 1. 初始化 git 仓库（如果尚未初始化）
if not exist ".git" (
    echo 📦 初始化 Git 仓库...
    git init
    echo ✅ Git 仓库初始化完成
) else (
    echo 📦 Git 仓库已存在
)

REM 2. 添加所有文件到暂存区
echo 📝 添加文件到 Git...
git add .

REM 3. 提交更改
echo 💾 提交更改...
git commit -m "Initial commit: DH-SNN Ultimate Implementation

- 🎯 超精简DH-SNN实现
- 🌟 包含多时间尺度核心创新实验  
- 📱 支持SSC、SHD、NeuroVPR等应用实验
- 🔧 统一的实验运行器
- 📚 完整的中文文档和注释
- 🚀 基于SpikingJelly框架

主要特性：
- 树突异质性脉冲神经网络核心算法
- 多时间尺度信息处理能力
- 创新的胞体vs树突异质性对比实验
- 完整的实验配置管理系统"

REM 4. 设置远程仓库
echo 🌐 设置远程仓库...
git remote remove origin 2>nul
git remote add origin https://github.com/amadeus725/DH-SNN_Reproduce.git

REM 5. 强制推送到 GitHub（替换原有内容）
echo ⚠️  准备推送到 GitHub...
echo 注意：这将替换远程仓库的所有内容！
echo 按任意键继续，或关闭窗口取消...
pause >nul

echo 🚀 推送到 GitHub...
git branch -M main
git push -f origin main

echo.
echo 🎉 上传完成！
echo ========================================
echo 📋 上传摘要：
echo    - 项目名称: DH-SNN Ultimate
echo    - 仓库地址: https://github.com/amadeus725/DH-SNN_Reproduce
echo    - 分支: main
echo    - 状态: 已强制推送（替换原有内容）
echo.
echo 🔗 访问您的项目：
echo    https://github.com/amadeus725/DH-SNN_Reproduce
echo.
echo 📚 后续步骤：
echo    1. 在 GitHub 上查看项目
echo    2. 更新项目描述和标签
echo    3. 设置 GitHub Pages（如需要）
echo    4. 邀请协作者（如需要）
echo.
echo ✅ 所有步骤完成！
pause
