#!/bin/bash

# 全面启动所有DH-SNN实验
echo "🚀 启动所有DH-SNN实验"
echo "=========================="

BASE_DIR="/root/DH-SNN_reproduce/experiments/dataset_benchmarks"
cd "$BASE_DIR"

# 创建日志目录
mkdir -p logs

# 启动函数
launch_experiment() {
    local exp_name=$1
    local exp_dir=$2
    local script=$3
    local log_file="logs/${exp_name}_$(date +%Y%m%d_%H%M%S).log"
    
    echo "🔄 启动 $exp_name..."
    cd "$BASE_DIR/$exp_dir"
    
    # 使用screen启动实验
    screen -dmS "$exp_name" bash -c "python $script 2>&1 | tee $BASE_DIR/$log_file"
    
    if [ $? -eq 0 ]; then
        echo "✅ $exp_name 已启动 (日志: $log_file)"
    else
        echo "❌ $exp_name 启动失败"
    fi
    
    cd "$BASE_DIR"
    sleep 2
}

# 优先级1实验
echo ""
echo "🔥 优先级1实验"
echo "----------------"

launch_experiment "GSC_DH_SNN" "gsc" "gsc_spikingjelly_experiment.py"
launch_experiment "SHD_DH_SRNN" "shd" "main_dh_srnn.py"
launch_experiment "SHD_Vanilla_SRNN" "shd" "main_vanilla_srnn.py"
launch_experiment "Permuted_MNIST_DH" "permuted_mnist" "main_dh_srnn.py"
launch_experiment "Permuted_MNIST_Vanilla" "permuted_mnist" "main_vanilla_srnn.py"

# 等待一些实验启动
echo "⏳ 等待30秒让实验启动..."
sleep 30

# 优先级2实验
echo ""
echo "🔶 优先级2实验"
echo "----------------"

launch_experiment "SSC_DH_SRNN" "ssc" "main_dh_srnn.py"
launch_experiment "SSC_Vanilla_SRNN" "ssc" "main_vanilla_srnn.py"
launch_experiment "TIMIT_DH_SRNN" "timit" "main_dh_srnn.py"
launch_experiment "TIMIT_DH_SFNN" "timit" "main_dh_sfnn.py"

# 等待更多资源
echo "⏳ 等待60秒让实验稳定..."
sleep 60

# 优先级3实验
echo ""
echo "🔷 优先级3实验"
echo "----------------"

launch_experiment "DEAP_DH_SRNN" "deap" "main_dh_srnn.py"
launch_experiment "DEAP_Vanilla_SRNN" "deap" "main_vanilla_srnn.py"
launch_experiment "NeuroVPR_DH_SFNN" "neurovpr" "main_dh_sfnn.py"

# 检查所有screen会话
echo ""
echo "📋 当前运行的实验:"
echo "-------------------"
screen -ls

echo ""
echo "🎯 所有实验已启动!"
echo "使用以下命令监控:"
echo "  screen -ls                    # 查看所有会话"
echo "  screen -r <session_name>      # 连接到特定会话"
echo "  python monitor_all_experiments.py  # 运行监控脚本"

echo ""
echo "📊 预计完成时间:"
echo "  优先级1实验: 2-4小时"
echo "  优先级2实验: 4-8小时"
echo "  优先级3实验: 6-12小时"
echo "  总计: 12-24小时"

echo ""
echo "✅ 批量启动完成!"
