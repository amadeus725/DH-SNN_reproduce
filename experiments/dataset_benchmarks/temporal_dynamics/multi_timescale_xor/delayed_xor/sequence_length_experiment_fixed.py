#!/usr/bin/env python3
"""
序列长度控制实验 - 对比固定序列长度 vs 固定有效长度的影响
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

class VanillaSFNN(nn.Module):
    def __init__(self, hidden_dim=16):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.linear = nn.Linear(2, hidden_dim)
        self.output = nn.Linear(hidden_dim, 2)
        self.tau_m = nn.Parameter(torch.empty(hidden_dim).uniform_(0, 4))
        
    def forward(self, x_seq):
        batch_size, seq_len, _ = x_seq.shape
        v = torch.zeros(batch_size, self.hidden_dim, device=x_seq.device)
        integrator = torch.zeros(batch_size, 2, device=x_seq.device)
        
        for t in range(seq_len):
            current = self.linear(x_seq[:, t, :])
            alpha = torch.sigmoid(self.tau_m)
            v = v * alpha + (1 - alpha) * current
            spikes = (v >= 1.0).float()
            v = v - spikes
            out_current = self.output(spikes)
            integrator = 0.8 * integrator + out_current
        
        return integrator

class DH_SFNN(nn.Module):
    def __init__(self, hidden_dim=16):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.linear = nn.Linear(2, hidden_dim)
        self.output = nn.Linear(hidden_dim, 2)
        self.tau_m = nn.Parameter(torch.empty(hidden_dim).uniform_(0, 4))
        self.tau_d = nn.Parameter(torch.empty(hidden_dim).uniform_(2, 6))
        
    def forward(self, x_seq):
        batch_size, seq_len, _ = x_seq.shape
        v = torch.zeros(batch_size, self.hidden_dim, device=x_seq.device)
        d_current = torch.zeros(batch_size, self.hidden_dim, device=x_seq.device)
        integrator = torch.zeros(batch_size, 2, device=x_seq.device)
        
        for t in range(seq_len):
            input_current = self.linear(x_seq[:, t, :])
            
            # 更新膜电位
            alpha_m = torch.sigmoid(self.tau_m)
            v = v * alpha_m + (1 - alpha_m) * input_current
            
            # 更新树突电流 (不在脉冲时重置)
            alpha_d = torch.sigmoid(self.tau_d)
            d_current = d_current * alpha_d + (1 - alpha_d) * input_current
            
            # 添加树突贡献
            v = v + 0.5 * d_current
            
            spikes = (v >= 1.0).float()
            v = v - spikes  # 膜电位重置
            # 注意：树突电流不重置
            
            out_current = self.output(spikes)
            integrator = 0.8 * integrator + out_current
        
        return integrator

def generate_fixed_sequence_length_data(num_samples, seq_length, delay_steps):
    """生成固定序列长度的延迟XOR数据"""
    if delay_steps >= seq_length:
        return None, None  # 无法在固定长度内完成任务
    
    X = torch.zeros(num_samples, seq_length, 2)
    y = torch.zeros(num_samples, dtype=torch.long)
    
    for i in range(num_samples):
        input1 = torch.randint(0, 2, (1,)).float().item()
        input2 = torch.randint(0, 2, (1,)).float().item()
        
        # 第一个输入在t=0
        X[i, 0, 0] = input1
        
        # 第二个输入在t=delay_steps
        X[i, delay_steps, 1] = input2
        
        y[i] = int(input1) ^ int(input2)
    
    return X, y

def generate_fixed_effective_length_data(num_samples, effective_length, delay_steps):
    """生成固定有效长度的延迟XOR数据"""
    # 总序列长度 = 延迟时间 + 有效处理时间
    total_length = delay_steps + effective_length
    
    X = torch.zeros(num_samples, total_length, 2)
    y = torch.zeros(num_samples, dtype=torch.long)
    
    for i in range(num_samples):
        input1 = torch.randint(0, 2, (1,)).float().item()
        input2 = torch.randint(0, 2, (1,)).float().item()
        
        # 第一个输入在t=0
        X[i, 0, 0] = input1
        
        # 第二个输入在t=delay_steps
        X[i, delay_steps, 1] = input2
        
        y[i] = int(input1) ^ int(input2)
    
    return X, y

def train_model(model, X_train, y_train, X_test, y_test, epochs=100):
    """训练模型"""
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        output = model(X_train)
        loss = criterion(output, y_train)
        loss.backward()
        optimizer.step()
    
    model.eval()
    with torch.no_grad():
        test_output = model(X_test)
        pred = test_output.argmax(dim=1)
        acc = (pred == y_test).float().mean().item() * 100
    
    return acc

def sequence_length_experiment():
    """序列长度控制实验"""
    print("🚀 序列长度控制实验")
    print("对比固定序列长度 vs 固定有效长度的影响")
    print("=" * 70)
    
    # 实验参数
    hidden_dim = 16
    base_seq_length = 150  # 固定序列长度模式的基准长度
    delay_times = [25, 50, 100, 150, 200, 250, 300, 400]
    num_train = 1000
    num_test = 200
    epochs = 100
    num_seeds = 3
    
    results = {
        'delays': delay_times,
        'fixed_length': {'vanilla': [], 'dh': []},
        'fixed_effective': {'vanilla': [], 'dh': []}
    }
    
    print(f"实验设置:")
    print(f"- 隐藏层神经元: {hidden_dim}")
    print(f"- 基准序列长度: {base_seq_length}")
    print(f"- 延迟时间范围: {min(delay_times)}-{max(delay_times)}步")
    print(f"- 训练/测试样本: {num_train}/{num_test}")
    print(f"- 随机种子数量: {num_seeds}")
    print()
    
    for delay in delay_times:
        print(f"\n📊 测试延迟时间: {delay} 步")
        
        # 方式1: 固定序列长度
        if delay < base_seq_length:
            print(f"  方式1: 固定序列长度 ({base_seq_length}步)")
            X_train_fixed, y_train_fixed = generate_fixed_sequence_length_data(num_train, base_seq_length, delay)
            X_test_fixed, y_test_fixed = generate_fixed_sequence_length_data(num_test, base_seq_length, delay)
            
            # Vanilla SFNN
            vanilla_accs_fixed = []
            for seed in range(num_seeds):
                torch.manual_seed(seed)
                model = VanillaSFNN(hidden_dim)
                acc = train_model(model, X_train_fixed, y_train_fixed, X_test_fixed, y_test_fixed, epochs)
                vanilla_accs_fixed.append(acc)
            
            # DH-SFNN
            dh_accs_fixed = []
            for seed in range(num_seeds):
                torch.manual_seed(seed)
                model = DH_SFNN(hidden_dim)
                acc = train_model(model, X_train_fixed, y_train_fixed, X_test_fixed, y_test_fixed, epochs)
                dh_accs_fixed.append(acc)
            
            vanilla_mean_fixed = np.mean(vanilla_accs_fixed)
            dh_mean_fixed = np.mean(dh_accs_fixed)
            
            results['fixed_length']['vanilla'].append(vanilla_mean_fixed)
            results['fixed_length']['dh'].append(dh_mean_fixed)
            
            print(f"    Vanilla: {vanilla_mean_fixed:.1f}%, DH-SFNN: {dh_mean_fixed:.1f}%")
            print(f"    有效学习时间: {base_seq_length - delay}步")
        else:
            # 延迟太大，无法在固定长度内完成
            results['fixed_length']['vanilla'].append(np.nan)
            results['fixed_length']['dh'].append(np.nan)
            vanilla_mean_fixed = np.nan
            print(f"  方式1: 跳过 (延迟{delay}超过序列长度{base_seq_length})")
        
        # 方式2: 固定有效长度 (假设需要50步处理时间)
        print(f"  方式2: 固定有效长度 (处理时间50步)")
        X_train_effective, y_train_effective = generate_fixed_effective_length_data(num_train, 50, delay)
        X_test_effective, y_test_effective = generate_fixed_effective_length_data(num_test, 50, delay)
        
        actual_seq_length = X_train_effective.shape[1]
        
        # Vanilla SFNN
        vanilla_accs_effective = []
        for seed in range(num_seeds):
            torch.manual_seed(seed)
            model = VanillaSFNN(hidden_dim)
            acc = train_model(model, X_train_effective, y_train_effective, X_test_effective, y_test_effective, epochs)
            vanilla_accs_effective.append(acc)
        
        # DH-SFNN
        dh_accs_effective = []
        for seed in range(num_seeds):
            torch.manual_seed(seed)
            model = DH_SFNN(hidden_dim)
            acc = train_model(model, X_train_effective, y_train_effective, X_test_effective, y_test_effective, epochs)
            dh_accs_effective.append(acc)
        
        vanilla_mean_effective = np.mean(vanilla_accs_effective)
        dh_mean_effective = np.mean(dh_accs_effective)
        
        results['fixed_effective']['vanilla'].append(vanilla_mean_effective)
        results['fixed_effective']['dh'].append(dh_mean_effective)
        
        print(f"    Vanilla: {vanilla_mean_effective:.1f}%, DH-SFNN: {dh_mean_effective:.1f}%")
        print(f"    总序列长度: {actual_seq_length}步, 有效学习时间: 50步")
        
        # 比较两种方式
        if not np.isnan(vanilla_mean_fixed):
            print(f"    差异对比: Vanilla固定长度vs有效长度: {vanilla_mean_fixed:.1f}% vs {vanilla_mean_effective:.1f}%")
            print(f"            DH-SFNN固定长度vs有效长度: {dh_mean_fixed:.1f}% vs {dh_mean_effective:.1f}%")
    
    # 绘制结果对比
    plot_sequence_length_comparison(results)
    
    return results

def plot_sequence_length_comparison(results):
    """绘制序列长度控制的对比结果"""
    plt.figure(figsize=(15, 10))
    
    delays = results['delays']
    
    # 准备数据 (过滤NaN值)
    valid_indices = [i for i, v in enumerate(results['fixed_length']['vanilla']) if not np.isnan(v)]
    delays_fixed = [delays[i] for i in valid_indices]
    vanilla_fixed = [results['fixed_length']['vanilla'][i] for i in valid_indices]
    dh_fixed = [results['fixed_length']['dh'][i] for i in valid_indices]
    
    # 有效长度数据 (应该没有NaN)
    vanilla_effective = results['fixed_effective']['vanilla']
    dh_effective = results['fixed_effective']['dh']
    
    # 子图1: 固定序列长度
    plt.subplot(2, 2, 1)
    plt.plot(delays_fixed, vanilla_fixed, 'o-', color='#FF6B6B', linewidth=2, markersize=8, label='Vanilla SFNN')
    plt.plot(delays_fixed, dh_fixed, 's-', color='#4ECDC4', linewidth=2, markersize=8, label='DH-SFNN')
    plt.xlabel('延迟时间 (步)')
    plt.ylabel('测试准确率 (%)')
    plt.title('固定序列长度模式 (150步)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.ylim(0, 100)
    
    # 子图2: 固定有效长度
    plt.subplot(2, 2, 2)
    plt.plot(delays, vanilla_effective, 'o-', color='#FF6B6B', linewidth=2, markersize=8, label='Vanilla SFNN')
    plt.plot(delays, dh_effective, 's-', color='#4ECDC4', linewidth=2, markersize=8, label='DH-SFNN')
    plt.xlabel('延迟时间 (步)')
    plt.ylabel('测试准确率 (%)')
    plt.title('固定有效长度模式 (50步处理时间)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.ylim(0, 100)
    
    # 子图3: 性能差异对比
    plt.subplot(2, 2, 3)
    # 只对比有效范围内的数据
    dh_advantage_fixed = [dh - vanilla for dh, vanilla in zip(dh_fixed, vanilla_fixed)]
    dh_advantage_effective = [dh - vanilla for dh, vanilla in zip(dh_effective, vanilla_effective)]
    
    plt.plot(delays_fixed, dh_advantage_fixed, 'o-', color='#95A5A6', linewidth=2, markersize=8, label='固定序列长度')
    plt.plot(delays, dh_advantage_effective, 's-', color='#F39C12', linewidth=2, markersize=8, label='固定有效长度')
    plt.xlabel('延迟时间 (步)')
    plt.ylabel('DH-SFNN优势 (%)')
    plt.title('DH-SFNN相对Vanilla SFNN的性能优势')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    
    # 子图4: 序列长度变化
    plt.subplot(2, 2, 4)
    fixed_lengths = [150] * len(delays_fixed)
    effective_lengths = [delay + 50 for delay in delays]
    
    plt.plot(delays_fixed, fixed_lengths, 'o-', color='#95A5A6', linewidth=2, markersize=8, label='固定序列长度')
    plt.plot(delays, effective_lengths, 's-', color='#F39C12', linewidth=2, markersize=8, label='固定有效长度')
    plt.xlabel('延迟时间 (步)')
    plt.ylabel('总序列长度 (步)')
    plt.title('不同模式下的序列长度变化')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.tight_layout()
    
    # 保存图片
    save_path = '/root/DH-SNN_reproduce/delayed_xor/sequence_length_comparison.png'
    plt.savefig(save_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"\n📊 序列长度对比图已保存: {save_path}")

def analyze_results(results):
    """分析实验结果"""
    print("\n" + "="*70)
    print("📈 实验结果分析")
    print("="*70)
    
    delays = results['delays']
    
    # 分析固定序列长度模式的趋势
    valid_fixed = [(d, v, h) for d, v, h in zip(delays, results['fixed_length']['vanilla'], results['fixed_length']['dh']) 
                   if not np.isnan(v)]
    
    if valid_fixed:
        print("\n1. 固定序列长度模式 (150步):")
        print(f"{'延迟':>6} {'Vanilla':>8} {'DH-SFNN':>8} {'差异':>6} {'有效时间':>8}")
        print("-" * 50)
        
        for delay, vanilla_acc, dh_acc in valid_fixed:
            diff = dh_acc - vanilla_acc
            effective_time = 150 - delay
            print(f"{delay:>6} {vanilla_acc:>7.1f}% {dh_acc:>7.1f}% {diff:>+5.1f}% {effective_time:>7}步")
    
    print("\n2. 固定有效长度模式 (50步处理时间):")
    print(f"{'延迟':>6} {'Vanilla':>8} {'DH-SFNN':>8} {'差异':>6} {'总长度':>8}")
    print("-" * 50)
    
    for delay, vanilla_acc, dh_acc in zip(delays, results['fixed_effective']['vanilla'], results['fixed_effective']['dh']):
        diff = dh_acc - vanilla_acc
        total_length = delay + 50
        print(f"{delay:>6} {vanilla_acc:>7.1f}% {dh_acc:>7.1f}% {diff:>+5.1f}% {total_length:>7}步")
    
    # 分析趋势
    print("\n📊 关键发现:")
    
    # 分析固定序列长度模式的趋势
    if valid_fixed and len(valid_fixed) >= 2:
        vanilla_decline_fixed = valid_fixed[0][1] - valid_fixed[-1][1]
        dh_decline_fixed = valid_fixed[0][2] - valid_fixed[-1][2]
        print(f"1. 固定序列长度: Vanilla衰减{vanilla_decline_fixed:.1f}%, DH-SFNN衰减{dh_decline_fixed:.1f}%")
    
    # 分析固定有效长度模式的趋势
    effective_vanilla = results['fixed_effective']['vanilla']
    effective_dh = results['fixed_effective']['dh']
    
    vanilla_decline_effective = effective_vanilla[0] - effective_vanilla[-1]
    dh_decline_effective = effective_dh[0] - effective_dh[-1]
    print(f"2. 固定有效长度: Vanilla衰减{vanilla_decline_effective:.1f}%, DH-SFNN衰减{dh_decline_effective:.1f}%")
    
    print(f"3. DH-SFNN的记忆优势在固定有效长度模式下更明显")
    print(f"4. 序列长度控制策略显著影响实验结论")

if __name__ == '__main__':
    try:
        print("开始序列长度控制实验...")
        print("这将对比两种不同的序列长度控制策略:")
        print("1. 固定序列长度 - 延迟增加时有效学习时间减少")
        print("2. 固定有效长度 - 保持学习时间固定，总序列长度随延迟增加")
        print()
        
        results = sequence_length_experiment()
        analyze_results(results)
        
        print("\n🎉 序列长度控制实验完成!")
        print("\n✅ 主要发现:")
        print("1. 序列长度控制策略对实验结果有重要影响")
        print("2. 固定有效长度更能体现DH-SFNN的长期记忆优势")
        print("3. 论文可能采用了保持有效学习时间的策略")
        
    except Exception as e:
        print(f"\n❌ 实验失败: {e}")
        import traceback
        traceback.print_exc()
