#!/usr/bin/env python3
"""
直接使用原论文代码进行测试
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import sys

# 添加路径
# sys.path.append removed during restructure
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

# 导入原论文的实现
sys.path.append('../SHD')
from SNN_layers.spike_dense import spike_dense_test_denri_wotanh_R, spike_dense_test_origin, readout_integrator_test
from direct_gz_reader import read_gz_h5_file, convert_to_spike_tensor
from torch.utils.data import DataLoader, TensorDataset

print("🚀 直接使用原论文代码测试")
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

# 时间因子配置
TIMING_CONFIGS = {
    'Small': {'tau_m': (0.0, 4.0), 'tau_n': (-4.0, 0.0)},
    'Medium': {'tau_m': (0.0, 4.0), 'tau_n': (0.0, 4.0)},
    'Large': {'tau_m': (0.0, 4.0), 'tau_n': (2.0, 6.0)}
}

class OriginalVanillaSFNN(nn.Module):
    """使用原论文的Vanilla SFNN"""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.device = config['device']
        
        # 使用原论文的层
        self.dense_1 = spike_dense_test_origin(
            input_dim=700,
            output_dim=config['hidden_size'],
            tau_minitializer='uniform',
            low_m=0.0,
            high_m=4.0,
            vth=config['v_threshold'],
            dt=config['dt'],
            device=self.device
        )
        
        self.dense_2 = readout_integrator_test(
            input_dim=config['hidden_size'],
            output_dim=20,
            tau_minitializer='uniform',
            low_m=0.0,
            high_m=4.0,
            device=self.device,
            dt=config['dt']
        )
        
        # 初始化权重
        torch.nn.init.xavier_normal_(self.dense_2.dense.weight)
        torch.nn.init.constant_(self.dense_2.dense.bias, 0)
        
    def forward(self, input_data):
        batch_size, seq_length, input_dim = input_data.shape
        
        # 设置神经元状态
        self.dense_1.set_neuron_state(batch_size)
        self.dense_2.set_neuron_state(batch_size)
        
        output = 0
        for i in range(seq_length):
            input_x = input_data[:, i, :].reshape(batch_size, input_dim)
            
            # 隐藏层
            mem_layer1, spike_layer1 = self.dense_1.forward(input_x)
            
            # 读出层
            mem_layer2 = self.dense_2.forward(spike_layer1)
            
            # 累积输出
            if i > 10:
                output += F.softmax(mem_layer2, dim=1)
        
        return output

class OriginalDH_SFNN(nn.Module):
    """使用原论文的DH-SFNN"""
    
    def __init__(self, config, tau_n_range=(2.0, 6.0)):
        super().__init__()
        self.config = config
        self.device = config['device']
        
        # 使用原论文的DH层
        self.dense_1 = spike_dense_test_denri_wotanh_R(
            input_dim=700,
            output_dim=config['hidden_size'],
            tau_minitializer='uniform',
            low_m=0.0,
            high_m=4.0,
            tau_ninitializer='uniform',
            low_n=tau_n_range[0],
            high_n=tau_n_range[1],
            vth=config['v_threshold'],
            dt=config['dt'],
            branch=4,
            device=self.device,
            test_sparsity=False
        )
        
        self.dense_2 = readout_integrator_test(
            input_dim=config['hidden_size'],
            output_dim=20,
            tau_minitializer='uniform',
            low_m=0.0,
            high_m=4.0,
            device=self.device,
            dt=config['dt']
        )
        
        # 初始化权重
        torch.nn.init.xavier_normal_(self.dense_2.dense.weight)
        torch.nn.init.constant_(self.dense_2.dense.bias, 0)
        
    def forward(self, input_data):
        batch_size, seq_length, input_dim = input_data.shape
        
        # 设置神经元状态
        self.dense_1.set_neuron_state(batch_size)
        self.dense_2.set_neuron_state(batch_size)
        
        output = 0
        for i in range(seq_length):
            input_x = input_data[:, i, :].reshape(batch_size, input_dim)
            
            # 应用连接掩码
            self.dense_1.apply_mask()
            
            # DH隐藏层
            mem_layer1, spike_layer1 = self.dense_1.forward(input_x)
            
            # 读出层
            mem_layer2 = self.dense_2.forward(spike_layer1)
            
            # 累积输出
            if i > 10:
                output += F.softmax(mem_layer2, dim=1)
        
        return output

def load_data(num_train=2000, num_test=500):
    """加载更多数据"""
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

def train_original_model(model, train_data, train_labels, test_data, test_labels, config, model_name):
    """按照原论文方式训练"""
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
            if hasattr(model, 'dense_1') and hasattr(model.dense_1, 'apply_mask'):
                model.dense_1.apply_mask()
            
            optimizer.zero_grad()
            predictions = model(images)
            loss = criterion(predictions, labels)
            loss.backward()
            optimizer.step()
            
            # 再次应用掩码
            if hasattr(model, 'dense_1') and hasattr(model.dense_1, 'apply_mask'):
                model.dense_1.apply_mask()
            
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
                if hasattr(model, 'dense_1') and hasattr(model.dense_1, 'apply_mask'):
                    model.dense_1.apply_mask()
                
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
        
        # 加载更多数据
        train_data, train_labels, test_data, test_labels = load_data(2000, 500)
        
        results = {}
        
        for timing_name, timing_config in TIMING_CONFIGS.items():
            print(f"\n📊 测试 {timing_name} timing factors")
            print(f"  tau_n: {timing_config['tau_n']}")
            
            # Vanilla SFNN
            vanilla_model = OriginalVanillaSFNN(CONFIG)
            vanilla_acc = train_original_model(
                vanilla_model, train_data, train_labels, test_data, test_labels,
                CONFIG, f"Vanilla SFNN ({timing_name})"
            )
            
            # DH-SFNN
            dh_model = OriginalDH_SFNN(CONFIG, tau_n_range=timing_config['tau_n'])
            dh_acc = train_original_model(
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
        torch.save(results, "outputs/results/paper_direct_test_results.pth")
        
        # 总结
        print(f"\n🎉 原论文代码测试完成!")
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
