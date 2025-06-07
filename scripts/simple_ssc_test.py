#!/usr/bin/env python3
"""
简单的SSC实验测试
直接使用SpikingJelly实现，避免复杂的导入问题
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import sys
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))
sys.path.append(str(project_root / "experiments" / "dataset_benchmarks" / "figure_reproduction" / "figure3_delayed_xor"))

# 直接导入SpikingJelly实现
try:
    from experiments.dataset_benchmarks.figure_reproduction.figure3_delayed_xor.core.neurons import DH_LIF_Neuron
    from experiments.dataset_benchmarks.figure_reproduction.figure3_delayed_xor.core.layers import DendriticHeterogeneityDense, ReadoutIntegrator
    from experiments.dataset_benchmarks.figure_reproduction.figure3_delayed_xor.core.surrogate import MultiGaussianSurrogate
    print("✅ 成功导入SpikingJelly DH-SNN组件")
except Exception as e:
    print(f"❌ 导入失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

class SimpleDH_SFNN(nn.Module):
    """简化的DH-SFNN模型"""

    def __init__(self, input_dim=700, hidden_dim=200, output_dim=35, num_branches=2):
        super().__init__()

        self.hidden_layer = DendriticHeterogeneityDense(
            input_dim, hidden_dim, num_branches,
            tau_m_init=(0.0, 4.0), tau_n_init=(0.0, 4.0)
        )

        self.readout = ReadoutIntegrator(hidden_dim, output_dim)

    def forward(self, x):
        # x: [batch, time, features]
        batch_size, seq_len, _ = x.shape

        # 重置神经元状态
        self.hidden_layer.reset()

        # 逐时间步处理
        outputs = []
        for t in range(seq_len):
            hidden_out = self.hidden_layer(x[:, t, :])
            outputs.append(hidden_out)

        # 堆叠输出
        hidden_sequence = torch.stack(outputs, dim=1)  # [batch, time, hidden]

        # 读出层
        output = self.readout(hidden_sequence)

        return output

def create_mock_data(batch_size=32, seq_len=1000, input_dim=700, num_classes=35):
    """创建模拟SSC数据"""

    # 生成稀疏的脉冲数据
    data = torch.zeros(batch_size, seq_len, input_dim)

    for b in range(batch_size):
        # 每个样本随机生成一些脉冲
        num_spikes = np.random.randint(500, 1500)
        spike_times = np.random.randint(0, seq_len, num_spikes)
        spike_channels = np.random.randint(0, input_dim, num_spikes)

        for t, c in zip(spike_times, spike_channels):
            data[b, t, c] = 1.0

    # 随机标签
    labels = torch.randint(0, num_classes, (batch_size,))

    return data, labels

def train_model(model, train_data, train_labels, epochs=5):
    """训练模型"""

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    train_data = train_data.to(device)
    train_labels = train_labels.to(device)

    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    print(f"🚀 开始训练 (设备: {device})")

    for epoch in range(epochs):
        model.train()

        optimizer.zero_grad()
        outputs = model(train_data)
        loss = criterion(outputs, train_labels)
        loss.backward()
        optimizer.step()

        # 计算准确率
        pred = outputs.argmax(dim=1)
        acc = (pred == train_labels).float().mean().item()

        print(f"Epoch {epoch+1:2d}: Loss={loss.item():.4f}, Acc={acc:.3f}")

    return model

def main():
    """主函数"""
    print("🧪 简单SSC实验测试")
    print("="*50)

    # 创建模型
    print("🏗️ 创建DH-SFNN模型...")
    model = SimpleDH_SFNN(input_dim=700, hidden_dim=64, output_dim=35, num_branches=2)
    print(f"📊 参数数量: {sum(p.numel() for p in model.parameters())}")

    # 创建模拟数据
    print("📊 创建模拟数据...")
    train_data, train_labels = create_mock_data(batch_size=16, seq_len=100)  # 缩短序列长度
    print(f"   数据形状: {train_data.shape}")
    print(f"   标签形状: {train_labels.shape}")
    print(f"   平均脉冲数: {train_data.sum().item() / len(train_data):.1f}")

    # 训练模型
    print("🚀 开始训练...")
    trained_model = train_model(model, train_data, train_labels, epochs=3)

    print("\n🎉 测试完成！")
    print("✅ DH-SNN SpikingJelly实现工作正常")

    return True

if __name__ == "__main__":
    try:
        success = main()
        if success:
            print("\n🎯 下一步可以:")
            print("1. 运行完整的SSC数据集实验")
            print("2. 测试其他数据集 (SHD, GSC)")
            print("3. 生成论文复现报告")
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
