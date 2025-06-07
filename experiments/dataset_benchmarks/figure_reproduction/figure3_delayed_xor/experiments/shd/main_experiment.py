#!/usr/bin/env python3
"""
SpikingJelly等价实现 - 完全对应原论文的SNN框架
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import os
import sys

# 添加路径
# sys.path.append removed during restructure
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from spikingjelly.activation_based import neuron, functional, surrogate
from direct_gz_reader import read_gz_h5_file, convert_to_spike_tensor
from torch.utils.data import DataLoader, TensorDataset

print("🚀 SpikingJelly等价实现 - 对应原论文框架")
print("="*60)

# 原论文配置
CONFIG = {
    'learning_rate': 1e-2,
    'batch_size': 100,
    'epochs': 100,
    'hidden_size': 64,
    'v_threshold': 1.0,
    'dt': 1,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu'
}

# 时间因子配置 - 按照Table S3
TIMING_CONFIGS = {
    'Small': {'tau_m': (-4.0, 0.0), 'tau_n': (-4.0, 0.0)},
    'Medium': {'tau_m': (0.0, 4.0), 'tau_n': (0.0, 4.0)},
    'Large': {'tau_m': (2.0, 6.0), 'tau_n': (2.0, 6.0)}
}

class MultiGaussianSurrogate(torch.autograd.Function):
    """原论文的MultiGaussian替代函数"""

    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return input.gt(0).float()

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()

        # 原论文参数
        lens = 0.5
        scale = 6.0
        height = 0.15
        gamma = 0.5

        def gaussian(x, mu=0., sigma=0.5):
            return torch.exp(-((x - mu) ** 2) / (2 * sigma ** 2)) / torch.sqrt(2 * torch.tensor(math.pi)) / sigma

        # MultiGaussian - 完全按照原论文
        temp = gaussian(input, mu=0., sigma=lens) * (1. + height) \
             - gaussian(input, mu=lens, sigma=scale * lens) * height \
             - gaussian(input, mu=-lens, sigma=scale * lens) * height

        return grad_input * temp.float() * gamma

multi_gaussian_surrogate = MultiGaussianSurrogate.apply

class PaperEquivalentLIFNode(nn.Module):
    """等价于原论文的LIF神经元"""

    def __init__(self, size, tau_m_range=(0.0, 4.0), v_threshold=1.0, device='cpu'):
        super().__init__()
        self.size = size
        self.v_threshold = v_threshold
        self.device = device

        # 时间常数参数
        self.tau_m = nn.Parameter(torch.empty(size))
        nn.init.uniform_(self.tau_m, tau_m_range[0], tau_m_range[1])

        # 神经元状态
        self.register_buffer('mem', torch.zeros(1, size))
        self.register_buffer('spike', torch.zeros(1, size))

    def set_neuron_state(self, batch_size):
        """重置神经元状态"""
        self.mem = torch.rand(batch_size, self.size).to(self.device)
        self.spike = torch.rand(batch_size, self.size).to(self.device)

    def forward(self, input_current):
        """前向传播 - 完全按照原论文mem_update_pra"""
        # 原论文: alpha = torch.sigmoid(tau_m)
        alpha = torch.sigmoid(self.tau_m)

        # 原论文: mem = mem * alpha + (1 - alpha) * R_m * inputs - v_th * spike
        # R_m = 1 in original paper
        self.mem = self.mem * alpha + (1 - alpha) * input_current - self.v_threshold * self.spike

        # 原论文: inputs_ = mem - v_th
        inputs_ = self.mem - self.v_threshold

        # 原论文: spike = act_fun_adp(inputs_)
        self.spike = multi_gaussian_surrogate(inputs_)

        return self.mem, self.spike

class PaperEquivalentReadoutIntegrator(nn.Module):
    """等价于原论文的readout_integrator_test"""

    def __init__(self, input_dim, output_dim, tau_m_range=(0.0, 4.0), device='cpu'):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.device = device

        # 线性层
        self.dense = nn.Linear(input_dim, output_dim)

        # 时间常数
        self.tau_m = nn.Parameter(torch.empty(output_dim))
        nn.init.uniform_(self.tau_m, tau_m_range[0], tau_m_range[1])

        # 膜电位状态
        self.register_buffer('mem', torch.zeros(1, output_dim))

    def set_neuron_state(self, batch_size):
        """重置神经元状态"""
        self.mem = torch.rand(batch_size, self.output_dim).to(self.device)

    def forward(self, input_spike):
        """前向传播 - 完全按照原论文output_Neuron_pra"""
        # 突触输入
        d_input = self.dense(input_spike.float())

        # 原论文: alpha = torch.sigmoid(tau_m)
        alpha = torch.sigmoid(self.tau_m)

        # 原论文: mem = mem * alpha + (1-alpha) * inputs
        self.mem = self.mem * alpha + (1 - alpha) * d_input

        return self.mem

class PaperEquivalentVanillaSFNN(nn.Module):
    """等价于原论文的spike_dense_test_origin"""

    def __init__(self, config, tau_m_range=(0.0, 4.0)):
        super().__init__()
        self.config = config
        self.device = config['device']

        # 线性层
        self.dense = nn.Linear(700, config['hidden_size'])

        # LIF神经元
        self.lif_layer = PaperEquivalentLIFNode(
            config['hidden_size'],
            tau_m_range,
            config['v_threshold'],
            self.device
        )

        # 读出层
        self.readout = PaperEquivalentReadoutIntegrator(
            config['hidden_size'],
            20,
            tau_m_range,
            self.device
        )

        # 初始化权重
        torch.nn.init.xavier_normal_(self.readout.dense.weight)
        torch.nn.init.constant_(self.readout.dense.bias, 0)

    def forward(self, input_data):
        """前向传播 - 完全按照原论文Dense_test_1layer"""
        batch_size, seq_length, input_dim = input_data.shape

        # 设置神经元状态
        self.lif_layer.set_neuron_state(batch_size)
        self.readout.set_neuron_state(batch_size)

        output = 0
        for i in range(seq_length):
            input_x = input_data[:, i, :].reshape(batch_size, input_dim)

            # 线性变换
            d_input = self.dense(input_x.float())

            # LIF层
            mem_layer1, spike_layer1 = self.lif_layer.forward(d_input)

            # 读出层
            mem_layer2 = self.readout.forward(spike_layer1)

            # 累积输出 - 按照原论文
            if i > 10:
                output += F.softmax(mem_layer2, dim=1)

        return output

class PaperEquivalentDH_LIFNode(nn.Module):
    """等价于原论文的spike_dense_test_denri_wotanh_R"""

    def __init__(self, input_dim, output_dim, tau_m_range=(0.0, 4.0), tau_n_range=(2.0, 6.0),
                 num_branches=4, v_threshold=1.0, device='cpu'):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_branches = num_branches
        self.v_threshold = v_threshold
        self.device = device

        # 连接层 - 按照原论文
        self.pad = ((input_dim) // num_branches * num_branches + num_branches - input_dim) % num_branches
        self.dense = nn.Linear(input_dim + self.pad, output_dim * num_branches)

        # 时间常数参数
        self.tau_m = nn.Parameter(torch.empty(output_dim))
        self.tau_n = nn.Parameter(torch.empty(output_dim, num_branches))

        # 初始化
        nn.init.uniform_(self.tau_m, tau_m_range[0], tau_m_range[1])
        nn.init.uniform_(self.tau_n, tau_n_range[0], tau_n_range[1])

        # 神经元状态
        self.register_buffer('mem', torch.zeros(1, output_dim))
        self.register_buffer('spike', torch.zeros(1, output_dim))
        self.register_buffer('d_input', torch.zeros(1, output_dim, num_branches))

        # 创建连接掩码
        self.create_mask()

    def create_mask(self):
        """创建连接掩码 - 完全按照原论文"""
        input_size = self.input_dim + self.pad
        self.mask = torch.zeros(self.output_dim * self.num_branches, input_size).to(self.device)
        for i in range(self.output_dim):
            for j in range(self.num_branches):
                start_idx = j * input_size // self.num_branches
                end_idx = (j + 1) * input_size // self.num_branches
                self.mask[i * self.num_branches + j, start_idx:end_idx] = 1

    def apply_mask(self):
        """应用连接掩码"""
        self.dense.weight.data = self.dense.weight.data * self.mask

    def set_neuron_state(self, batch_size):
        """重置神经元状态"""
        self.mem = torch.rand(batch_size, self.output_dim).to(self.device)
        self.spike = torch.rand(batch_size, self.output_dim).to(self.device)
        self.d_input = torch.zeros(batch_size, self.output_dim, self.num_branches).to(self.device)

    def forward(self, input_spike):
        """前向传播 - 完全按照原论文"""
        # 树突时间常数
        beta = torch.sigmoid(self.tau_n)

        # 输入填充
        padding = torch.zeros(input_spike.size(0), self.pad).to(self.device)
        k_input = torch.cat((input_spike.float(), padding), 1)

        # 更新树突电流
        dense_output = self.dense(k_input).reshape(-1, self.output_dim, self.num_branches)
        self.d_input = beta * self.d_input + (1 - beta) * dense_output

        # 总输入电流
        l_input = self.d_input.sum(dim=2, keepdim=False)

        # 膜电位更新 - 按照原论文mem_update_pra
        alpha = torch.sigmoid(self.tau_m)
        self.mem = self.mem * alpha + (1 - alpha) * l_input - self.v_threshold * self.spike

        # 脉冲生成
        inputs_ = self.mem - self.v_threshold
        self.spike = multi_gaussian_surrogate(inputs_)

        return self.mem, self.spike

class PaperEquivalentDH_SFNN(nn.Module):
    """等价于原论文的DH-SFNN"""

    def __init__(self, config, tau_m_range=(0.0, 4.0), tau_n_range=(2.0, 6.0)):
        super().__init__()
        self.config = config
        self.device = config['device']

        # DH层
        self.dh_layer = PaperEquivalentDH_LIFNode(
            700,
            config['hidden_size'],
            tau_m_range,
            tau_n_range,
            4,  # num_branches
            config['v_threshold'],
            self.device
        )

        # 读出层
        self.readout = PaperEquivalentReadoutIntegrator(
            config['hidden_size'],
            20,
            tau_m_range,
            self.device
        )

        # 初始化权重
        torch.nn.init.xavier_normal_(self.readout.dense.weight)
        torch.nn.init.constant_(self.readout.dense.bias, 0)

    def forward(self, input_data):
        """前向传播"""
        batch_size, seq_length, input_dim = input_data.shape

        # 设置神经元状态
        self.dh_layer.set_neuron_state(batch_size)
        self.readout.set_neuron_state(batch_size)

        output = 0
        for i in range(seq_length):
            input_x = input_data[:, i, :].reshape(batch_size, input_dim)

            # 应用连接掩码
            self.dh_layer.apply_mask()

            # DH层
            mem_layer1, spike_layer1 = self.dh_layer.forward(input_x)

            # 读出层
            mem_layer2 = self.readout.forward(spike_layer1)

            # 累积输出
            if i > 10:
                output += F.softmax(mem_layer2, dim=1)

        return output

def load_data(num_train=2000, num_test=500):
    """加载数据"""
    print(f"📚 加载数据: 训练{num_train}, 测试{num_test}")

    train_times, train_units, train_labels = read_gz_h5_file(
        "../spikingjelly_shd/data/shd_train.h5.gz", max_samples=num_train
    )
    test_times, test_units, test_labels = read_gz_h5_file(
        "../spikingjelly_shd/data/shd_test.h5.gz", max_samples=num_test
    )

    print("🔄 转换为张量...")

    train_data = torch.zeros(len(train_times), 1000, 700)
    test_data = torch.zeros(len(test_times), 1000, 700)

    for i in range(len(train_times)):
        if i % 500 == 0:
            print(f"  处理训练样本 {i+1}/{len(train_times)}")
        tensor = convert_to_spike_tensor(train_times[i], train_units[i], dt=1e-3, max_time=1.0)
        train_data[i] = tensor

    for i in range(len(test_times)):
        if i % 100 == 0:
            print(f"  处理测试样本 {i+1}/{len(test_times)}")
        tensor = convert_to_spike_tensor(test_times[i], test_units[i], dt=1e-3, max_time=1.0)
        test_data[i] = tensor

    train_labels = torch.from_numpy(train_labels.astype(np.int64)).long()
    test_labels = torch.from_numpy(test_labels.astype(np.int64)).long()

    print(f"✅ 数据加载完成: 训练{train_data.shape}, 测试{test_data.shape}")
    return train_data, train_labels, test_data, test_labels

def train_equivalent_model(model, train_data, train_labels, test_data, test_labels, config, model_name):
    """训练等价模型 - 完全按照原论文方式"""
    print(f"🏋️ 训练{model_name}")

    device = config['device']
    model = model.to(device)

    train_dataset = TensorDataset(train_data, train_labels)
    test_dataset = TensorDataset(test_data, test_labels)

    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False)

    criterion = nn.CrossEntropyLoss()

    # 分组优化器 - 按照原论文
    base_params = []
    tau_m_params = []
    tau_n_params = []

    for name, param in model.named_parameters():
        if 'tau_m' in name:
            tau_m_params.append(param)
        elif 'tau_n' in name:
            tau_n_params.append(param)
        else:
            base_params.append(param)

    optimizer = torch.optim.Adam([
        {'params': base_params, 'lr': config['learning_rate']},
        {'params': tau_m_params, 'lr': config['learning_rate'] * 2},
        {'params': tau_n_params, 'lr': config['learning_rate'] * 2},
    ], lr=config['learning_rate'])

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

    best_acc = 0.0

    for epoch in range(config['epochs']):
        # 训练
        model.train()
        train_acc = 0
        sum_sample = 0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            # 应用掩码 (对DH-SFNN)
            if hasattr(model, 'dh_layer'):
                model.dh_layer.apply_mask()

            optimizer.zero_grad()
            predictions = model(images)
            loss = criterion(predictions, labels)
            loss.backward()
            optimizer.step()

            # 再次应用掩码
            if hasattr(model, 'dh_layer'):
                model.dh_layer.apply_mask()

            _, predicted = torch.max(predictions.data, 1)
            train_acc += (predicted.cpu() == labels.cpu()).sum().item()
            sum_sample += labels.size(0)

        scheduler.step()
        train_acc = train_acc / sum_sample * 100

        # 测试
        model.eval()
        test_acc = 0
        test_sum_sample = 0

        with torch.no_grad():
            for images, labels in test_loader:
                if hasattr(model, 'dh_layer'):
                    model.dh_layer.apply_mask()

                images = images.to(device)
                labels = labels.to(device)

                predictions = model(images)
                _, predicted = torch.max(predictions.data, 1)

                test_acc += (predicted.cpu() == labels.cpu()).sum().item()
                test_sum_sample += labels.size(0)

        test_acc = test_acc / test_sum_sample * 100

        if test_acc > best_acc:
            best_acc = test_acc

        if epoch % 20 == 0 or epoch == config['epochs'] - 1:
            print(f"  Epoch {epoch+1:2d}: 训练{train_acc:5.1f}%, 测试{test_acc:5.1f}%, 最佳{best_acc:5.1f}%")

    return best_acc

def main():
    """主函数"""
    try:
        print(f"🔧 使用设备: {CONFIG['device']}")

        # 加载数据
        train_data, train_labels, test_data, test_labels = load_data(2000, 500)

        results = {}

        for timing_name, timing_config in TIMING_CONFIGS.items():
            print(f"\n📊 测试 {timing_name} timing factors")
            print(f"  tau_m: {timing_config['tau_m']}, tau_n: {timing_config['tau_n']}")

            # Vanilla SFNN
            vanilla_model = PaperEquivalentVanillaSFNN(CONFIG, tau_m_range=timing_config['tau_m'])
            vanilla_acc = train_equivalent_model(
                vanilla_model, train_data, train_labels, test_data, test_labels,
                CONFIG, f"Vanilla SFNN ({timing_name})"
            )

            # DH-SFNN
            dh_model = PaperEquivalentDH_SFNN(
                CONFIG,
                tau_m_range=timing_config['tau_m'],
                tau_n_range=timing_config['tau_n']
            )
            dh_acc = train_equivalent_model(
                dh_model, train_data, train_labels, test_data, test_labels,
                CONFIG, f"DH-SFNN ({timing_name})"
            )

            results[timing_name] = {
                'vanilla': vanilla_acc,
                'dh_snn': dh_acc,
                'improvement': dh_acc - vanilla_acc
            }

            print(f"📈 {timing_name} 结果:")
            print(f"  Vanilla SFNN: {vanilla_acc:.1f}%")
            print(f"  DH-SFNN:      {dh_acc:.1f}%")
            print(f"  性能提升:     {dh_acc - vanilla_acc:+.1f} 个百分点")

        # 保存结果
        os.makedirs("outputs/results", exist_ok=True)
        torch.save(results, "outputs/results/spikingjelly_equivalent_results.pth")

        # 总结
        print(f"\n🎉 SpikingJelly等价实现测试完成!")
        print(f"📊 三种时间因子配置性能对比:")
        for timing_name in TIMING_CONFIGS.keys():
            vanilla_acc = results[timing_name]['vanilla']
            dh_acc = results[timing_name]['dh_snn']
            improvement = results[timing_name]['improvement']
            print(f"  {timing_name:6s}: Vanilla {vanilla_acc:5.1f}% → DH-SNN {dh_acc:5.1f}% (提升{improvement:+5.1f}%)")

        return True

    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == '__main__':
    success = main()
    print(f"\n🏁 测试完成，退出码: {0 if success else 1}")