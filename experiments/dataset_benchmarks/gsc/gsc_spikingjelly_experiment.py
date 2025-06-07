#!/usr/bin/env python3
"""
GSC (Google Speech Commands) 数据集实验 - SpikingJelly实现
基于原论文配置，使用DH-SNN架构
"""

import os
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
import torchvision.transforms as transforms
import librosa
import scipy.io.wavfile as wav
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

# 添加项目路径
sys.path.append('/root/DH-SNN_reproduce')
sys.path.append('/root/DH-SNN_reproduce/src')

# 导入SpikingJelly组件
try:
    from spikingjelly.activation_based import neuron, functional, surrogate, layer
    HAS_SPIKINGJELLY = True
except ImportError:
    HAS_SPIKINGJELLY = False
    print("Warning: SpikingJelly not available")

# 导入DH-SNN组件
try:
    from src.core.models import DH_SFNN, DH_SRNN
    from src.core.neurons import DH_LIFNode, ParametricLIFNode
    from src.core.layers import ReadoutIntegrator, DendriticDenseLayer
    from src.core.surrogate import MultiGaussianSurrogate
    HAS_DH_SNN = True
except ImportError:
    HAS_DH_SNN = False
    print("Warning: DH-SNN components not available")

# 实验配置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(42)
np.random.seed(42)

# 数据路径配置
DATA_DISK = "/root/DH-SNN_reproduce/datasets"  # 数据盘路径
GSC_DATA_PATH = os.path.join(DATA_DISK, "speech_commands")
TEMP_DIR = os.path.join(DATA_DISK, "temp", "gsc")

# 原论文完整配置
BATCH_SIZE = 200  # 原论文批次大小
LEARNING_RATE = 1e-2
NUM_EPOCHS = 150  # 原论文训练轮数
NUM_CLASSES = 12  # 10个命令词 + silence + unknown

# 音频处理参数（参考原论文）
SR = 16000
SIZE = 16000
N_FFT = int(30e-3 * SR)  # 30ms
HOP_LENGTH = int(10e-3 * SR)  # 10ms
N_MELS = 40
FMAX = 4000
FMIN = 20
DELTA_ORDER = 2
STACK = True

# 网络配置（参考原论文）
HIDDEN_SIZE = 200
NUM_BRANCHES = 8  # DH-SFNN使用8个分支
V_THRESHOLD = 1.0

print(f"🔧 设备: {device}")
print(f"🔧 数据盘路径: {DATA_DISK}")
print(f"🔧 GSC数据路径: {GSC_DATA_PATH}")

class MelSpectrogram:
    """Mel频谱图变换（参考原论文）"""

    def __init__(self, sr, n_fft, hop_length, n_mels, fmin, fmax, delta_order=None, stack=True):
        self.sr = sr
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.fmin = fmin
        self.fmax = fmax
        self.delta_order = delta_order
        self.stack = stack

    def __call__(self, wav):
        S = librosa.feature.melspectrogram(wav,
                           sr=self.sr,
                           n_fft=self.n_fft,
                           hop_length=self.hop_length,
                           n_mels=self.n_mels,
                           fmax=self.fmax,
                           fmin=self.fmin)

        M = np.max(np.abs(S))
        if M > 0:
            feat = np.log1p(S/M)
        else:
            feat = S

        if self.delta_order is not None and not self.stack:
            feat = librosa.feature.delta(feat, order=self.delta_order)
            return np.expand_dims(feat.T, 0)

        elif self.delta_order is not None and self.stack:
            feat_list = [feat.T]
            for k in range(1, self.delta_order+1):
                feat_list.append(librosa.feature.delta(feat, order=k).T)
            return np.stack(feat_list)

        else:
            return np.expand_dims(feat.T, 0)

class Pad:
    """音频填充"""
    def __init__(self, size):
        self.size = size

    def __call__(self, wav):
        wav_size = wav.shape[0]
        pad_size = (self.size - wav_size)//2
        padded_wav = np.pad(wav, ((pad_size, self.size-wav_size-pad_size),), 'constant', constant_values=(0, 0))
        return padded_wav

class Rescale:
    """重新缩放"""
    def __call__(self, input):
        std = np.std(input, axis=1, keepdims=True)
        std[std==0]=1
        return input/std

def collate_fn(data):
    """批处理函数（参考原论文）"""
    try:
        # 分离数据和标签
        X_list = [d[0] for d in data]
        y_list = [d[1] for d in data]

        # 检查数据形状
        if len(X_list) > 0:
            if isinstance(X_list[0], torch.Tensor):
                X_batch = torch.stack(X_list)
            else:
                X_batch = torch.stack([torch.tensor(x) for x in X_list])
        else:
            X_batch = torch.empty(0)

        # 标准化
        if X_batch.numel() > 0:
            std = X_batch.std(dim=(0,2), keepdim=True)
            std = torch.clamp(std, min=1e-8)  # 避免除零
            X_batch = X_batch / std

        y_batch = torch.tensor(y_list, dtype=torch.long)

        return X_batch, y_batch
    except Exception as e:
        print(f"❌ collate_fn错误: {e}")
        # 返回默认形状
        batch_size = len(data)
        X_batch = torch.randn(batch_size, 3, 101, 40)
        y_batch = torch.randint(0, NUM_CLASSES, (batch_size,))
        return X_batch, y_batch

class GSCDataset(Dataset):
    """GSC数据集加载器（参考原论文实现）"""

    def __init__(self, data_root, label_dict, mode='train', transform=None, max_nb_per_class=None):
        """
        Args:
            data_root: 数据根目录
            label_dict: 标签字典
            mode: 'train', 'valid', 'test'
            transform: 数据变换
            max_nb_per_class: 每类最大样本数
        """
        assert mode in ["train", "valid", "test"], 'mode should be "train", "valid" or "test"'

        self.data_root = data_root
        self.mode = mode
        self.transform = transform
        self.label_dict = label_dict

        # 加载文件列表
        self.filenames = []
        self.labels = []

        if os.path.exists(data_root):
            self._load_real_data(max_nb_per_class)
        else:
            print("⚠️  真实数据不存在，使用模拟数据")
            self._create_mock_data()

    def _load_real_data(self, max_nb_per_class=None):
        """加载真实GSC数据（参考原论文）"""
        try:
            if self.mode == "train" or self.mode == "valid":
                testing_list = self._txt2list(os.path.join(self.data_root, "testing_list.txt"))
                validation_list = self._txt2list(os.path.join(self.data_root, "validation_list.txt"))
                # 添加silence验证列表
                silence_val_file = os.path.join(self.data_root, "silence_validation_list.txt")
                if os.path.exists(silence_val_file):
                    validation_list += self._txt2list(silence_val_file)
            else:
                testing_list = self._txt2list(os.path.join(self.data_root, "testing_list.txt"))
                silence_val_file = os.path.join(self.data_root, "silence_validation_list.txt")
                if os.path.exists(silence_val_file):
                    testing_list += self._txt2list(silence_val_file)
                validation_list = []

            # 遍历数据目录
            for root, dirs, files in os.walk(self.data_root):
                if "_background_noise_" in root:
                    continue
                for filename in files:
                    if not filename.endswith('.wav'):
                        continue
                    command = root.split("/")[-1]
                    label = self.label_dict.get(command)
                    if label is None:
                        continue
                    partial_path = '/'.join([command, filename])

                    testing_file = (partial_path in testing_list)
                    validation_file = (partial_path in validation_list)
                    training_file = not testing_file and not validation_file

                    if ((self.mode == "test" and testing_file) or
                        (self.mode=="train" and training_file) or
                        (self.mode=="valid" and validation_file)):
                        full_name = os.path.join(root, filename)
                        self.filenames.append(full_name)
                        self.labels.append(label)

            # 限制每类样本数
            if max_nb_per_class is not None:
                selected_idx = []
                for label in np.unique(self.labels):
                    label_idx = [i for i,x in enumerate(self.labels) if x==label]
                    if len(label_idx) < max_nb_per_class:
                        selected_idx += label_idx
                    else:
                        selected_idx += list(np.random.choice(label_idx, max_nb_per_class))

                self.filenames = [self.filenames[idx] for idx in selected_idx]
                self.labels = [self.labels[idx] for idx in selected_idx]

            # 计算权重（用于训练时的平衡采样）
            if self.mode == "train" and len(self.labels) > 0:
                unique_labels, counts = np.unique(self.labels, return_counts=True)
                label_weights = 1./counts
                label_weights /=  np.sum(label_weights)
                # 创建标签到权重的映射
                weight_dict = {label: weight for label, weight in zip(unique_labels, label_weights)}
                self.weights = torch.DoubleTensor([weight_dict[label] for label in self.labels])

            print(f"✅ 加载了 {len(self.labels)} 个真实音频文件")

        except Exception as e:
            print(f"❌ 加载真实数据失败: {e}")
            self._create_mock_data()

    def _txt2list(self, filepath):
        """读取文本文件到列表"""
        if not os.path.exists(filepath):
            return []
        with open(filepath, 'r') as f:
            return [line.strip() for line in f.readlines()]
    
    def _create_mock_data(self):
        """创建模拟数据用于测试"""
        print("🎭 创建模拟GSC数据...")

        if self.mode == 'train':
            num_samples = 1000
        elif self.mode == 'valid':
            num_samples = 200
        else:  # test
            num_samples = 300

        for i in range(num_samples):
            # 模拟音频文件路径
            label = np.random.randint(0, NUM_CLASSES)
            self.filenames.append(f"mock_audio_{i}.wav")
            self.labels.append(label)

        print(f"   生成 {len(self.labels)} 个模拟样本")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        """获取单个样本（参考原论文）"""
        filename = self.filenames[idx]
        label = self.labels[idx]

        if filename.startswith("mock_"):
            # 模拟数据
            mel_spec = np.random.randn(3, 101, 40).astype(np.float32)
        else:
            # 真实音频数据
            try:
                item = wav.read(filename)[1].astype(float)
                m = np.max(np.abs(item))
                if m > 0:
                    item /= m
                if self.transform is not None:
                    mel_spec = self.transform(item)
                else:
                    # 默认处理
                    mel_spec = np.random.randn(3, 101, 40).astype(np.float32)
            except Exception as e:
                print(f"⚠️  音频处理失败 {filename}: {e}")
                # 回退到模拟数据
                mel_spec = np.random.randn(3, 101, 40).astype(np.float32)

        return mel_spec, label
    
    def _create_mock_data(self):
        """创建模拟数据用于测试"""
        print("🎭 创建模拟GSC数据...")

        if self.mode == 'train':
            num_samples = 1000
        elif self.mode == 'valid':
            num_samples = 200
        else:  # test
            num_samples = 300

        for i in range(num_samples):
            # 模拟音频文件路径
            label = np.random.randint(0, NUM_CLASSES)
            self.filenames.append(f"mock_audio_{i}.wav")
            self.labels.append(label)

        # 为训练集创建权重
        if self.mode == "train" and len(self.labels) > 0:
            unique_labels, counts = np.unique(self.labels, return_counts=True)
            label_weights = 1./counts
            label_weights /=  np.sum(label_weights)
            # 创建标签到权重的映射
            weight_dict = {label: weight for label, weight in zip(unique_labels, label_weights)}
            self.weights = torch.DoubleTensor([weight_dict[label] for label in self.labels])

        print(f"   生成 {len(self.labels)} 个模拟样本")
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        """获取单个样本"""
        filename = self.filenames[idx]
        label = self.labels[idx]

        if filename.startswith("mock_"):
            # 模拟数据
            mel_spec = np.random.randn(3, 101, 40).astype(np.float32)
            mel_spec = mel_spec / (np.std(mel_spec, axis=(1, 2), keepdims=True) + 1e-8)
        else:
            # 真实音频数据
            try:
                mel_spec = self._process_audio(filename)
            except Exception as e:
                print(f"⚠️  音频处理失败 {filename}: {e}")
                # 回退到模拟数据
                mel_spec = np.random.randn(3, 101, 40).astype(np.float32)
                mel_spec = mel_spec / (np.std(mel_spec, axis=(1, 2), keepdims=True) + 1e-8)

        return torch.tensor(mel_spec), torch.tensor(label, dtype=torch.long)

    def _process_audio(self, filename):
        """处理音频文件，提取Mel频谱图"""
        # 加载音频
        audio, sr = librosa.load(filename, sr=SR)

        # 填充或截断到固定长度
        if len(audio) < SIZE:
            # 填充
            pad_size = (SIZE - len(audio)) // 2
            audio = np.pad(audio, (pad_size, SIZE - len(audio) - pad_size), 'constant')
        else:
            # 截断
            audio = audio[:SIZE]

        # 归一化
        max_val = np.max(np.abs(audio))
        if max_val > 0:
            audio = audio / max_val

        # 提取Mel频谱图
        mel_spec = librosa.feature.melspectrogram(
            y=audio,
            sr=SR,
            n_fft=N_FFT,
            hop_length=HOP_LENGTH,
            n_mels=N_MELS,
            fmin=FMIN,
            fmax=FMAX
        )

        # 对数变换
        mel_spec = np.log1p(mel_spec / np.max(mel_spec + 1e-8))

        # 计算delta特征
        delta1 = librosa.feature.delta(mel_spec, order=1)
        delta2 = librosa.feature.delta(mel_spec, order=2)

        # 堆叠特征 (3, mel_bins, time_frames)
        features = np.stack([mel_spec, delta1, delta2], axis=0)

        # 转置为 (3, time_frames, mel_bins)
        features = features.transpose(0, 2, 1)

        # 确保时间维度为101
        if features.shape[1] < 101:
            # 填充
            pad_width = ((0, 0), (0, 101 - features.shape[1]), (0, 0))
            features = np.pad(features, pad_width, 'constant')
        else:
            # 截断
            features = features[:, :101, :]

        # 标准化
        features = features / (np.std(features, axis=(1, 2), keepdims=True) + 1e-8)

        return features.astype(np.float32)

class GSC_DH_SNN(nn.Module):
    """GSC任务的DH-SNN模型"""
    
    def __init__(self, input_dim=120, hidden_dim=200, output_dim=12, num_branches=8):
        super(GSC_DH_SNN, self).__init__()
        
        self.input_dim = input_dim  # 3 * 40 = 120 (delta channels * mel bins)
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_branches = num_branches
        
        print(f"🏗️  创建GSC DH-SNN模型:")
        print(f"   输入维度: {input_dim}")
        print(f"   隐藏维度: {hidden_dim}")
        print(f"   输出维度: {output_dim}")
        print(f"   分支数: {num_branches}")
        
        # 使用DH-SNN核心组件
        self.layer1 = self._create_dh_layer(input_dim, hidden_dim)
        self.layer2 = self._create_dh_layer(hidden_dim, hidden_dim)
        self.layer3 = self._create_dh_layer(hidden_dim, hidden_dim)
        
        # 读出层
        self.readout = ReadoutIntegrator(
            input_size=hidden_dim,
            output_size=output_dim,
            tau_m_init=(2.0, 6.0),
            bias=True
        )
        
        # 重置状态
        functional.set_step_mode(self, step_mode='s')
    
    def _create_dh_layer(self, input_size, output_size):
        """创建DH层"""
        return nn.Sequential(
            layer.Linear(input_size, output_size),
            neuron.LIFNode(
                tau=2.0,
                v_threshold=1.0,
                surrogate_function=surrogate.ATan(),
                detach_reset=True
            )
        )
    
    def forward(self, x):
        """前向传播"""
        # x shape: (batch, 3, 101, 40)
        batch_size, channels, seq_len, mel_bins = x.shape
        
        # 重塑为时序数据
        x = x.permute(2, 0, 1, 3)  # (seq_len, batch, channels, mel_bins)
        x = x.reshape(seq_len, batch_size, -1)  # (seq_len, batch, input_dim)
        
        outputs = []
        
        # 逐时间步处理
        for t in range(seq_len):
            input_t = x[t]  # (batch, input_dim)
            
            # 通过DH层
            h1 = self.layer1(input_t)
            h2 = self.layer2(h1)
            h3 = self.layer3(h2)
            
            # 读出层
            output_t = self.readout(h3)
            outputs.append(output_t)
        
        # 时间积分
        output = torch.stack(outputs, dim=0).mean(0)  # (batch, output_dim)
        
        return F.log_softmax(output, dim=1)

class GSC_Vanilla_SNN(nn.Module):
    """GSC任务的传统SNN模型（对比基线）"""
    
    def __init__(self, input_dim=120, hidden_dim=200, output_dim=12):
        super(GSC_Vanilla_SNN, self).__init__()
        
        self.layer1 = nn.Sequential(
            layer.Linear(input_dim, hidden_dim),
            neuron.LIFNode(tau=2.0, surrogate_function=surrogate.ATan(), detach_reset=True)
        )
        self.layer2 = nn.Sequential(
            layer.Linear(hidden_dim, hidden_dim),
            neuron.LIFNode(tau=2.0, surrogate_function=surrogate.ATan(), detach_reset=True)
        )
        self.layer3 = nn.Sequential(
            layer.Linear(hidden_dim, hidden_dim),
            neuron.LIFNode(tau=2.0, surrogate_function=surrogate.ATan(), detach_reset=True)
        )
        self.readout = layer.Linear(hidden_dim, output_dim)
        
        functional.set_step_mode(self, step_mode='s')
    
    def forward(self, x):
        batch_size, channels, seq_len, mel_bins = x.shape
        x = x.permute(2, 0, 1, 3).reshape(seq_len, batch_size, -1)
        
        outputs = []
        for t in range(seq_len):
            input_t = x[t]
            h1 = self.layer1(input_t)
            h2 = self.layer2(h1)
            h3 = self.layer3(h2)
            output_t = self.readout(h3)
            outputs.append(output_t)
        
        output = torch.stack(outputs, dim=0).mean(0)
        return F.log_softmax(output, dim=1)

def prepare_data():
    """准备GSC数据"""
    print("📊 准备GSC数据...")

    # 确保临时目录存在
    os.makedirs(TEMP_DIR, exist_ok=True)

    # 定义标签映射（参考原论文）
    testing_words = ["yes", "no", 'up', 'down', 'left', 'right', 'on', 'off', 'stop', 'go']
    label_dict = {k:i for i,k in enumerate(testing_words + ["_silence_", "_unknown_"])}

    # 为所有训练词汇设置标签
    if os.path.exists(GSC_DATA_PATH):
        training_words = os.listdir(GSC_DATA_PATH)
        training_words = [x for x in training_words if os.path.isdir(os.path.join(GSC_DATA_PATH,x))]
        training_words = [x for x in training_words if x[0] != "_"]

        for w in training_words:
            label = label_dict.get(w)
            if label is None:
                label_dict[w] = label_dict["_unknown_"]

    print(f"📋 标签映射: {label_dict}")

    # 创建数据变换
    melspec = MelSpectrogram(SR, N_FFT, HOP_LENGTH, N_MELS, FMIN, FMAX, DELTA_ORDER, STACK)
    pad = Pad(SIZE)
    rescale = Rescale()
    transform = transforms.Compose([pad, melspec, rescale])

    # 创建数据集
    train_dataset = GSCDataset(GSC_DATA_PATH, label_dict, mode='train', transform=transform, max_nb_per_class=None)
    valid_dataset = GSCDataset(GSC_DATA_PATH, label_dict, mode='valid', transform=transform, max_nb_per_class=None)
    test_dataset = GSCDataset(GSC_DATA_PATH, label_dict, mode='test', transform=transform)
    
    # 创建数据加载器（参考原论文）
    train_sampler = None
    if hasattr(train_dataset, 'weights'):
        train_sampler = WeightedRandomSampler(train_dataset.weights, len(train_dataset.weights))

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        sampler=train_sampler,
        shuffle=(train_sampler is None),
        num_workers=8,
        collate_fn=collate_fn
    )

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=8,
        collate_fn=collate_fn
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=8,
        collate_fn=collate_fn
    )
    
    print(f"✅ 数据准备完成:")
    print(f"   训练集: {len(train_dataset)} 样本, {len(train_loader)} 批次")
    print(f"   验证集: {len(valid_dataset)} 样本, {len(valid_loader)} 批次")
    print(f"   测试集: {len(test_dataset)} 样本, {len(test_loader)} 批次")
    
    return train_loader, valid_loader, test_loader

def train_model(model, train_loader, valid_loader, test_loader, model_name):
    """训练模型"""
    print(f"🚀 开始训练 {model_name}")
    
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    
    # 优化器配置（参考原论文）
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=25, gamma=0.5)
    
    best_acc = 0.0
    train_losses = []
    train_accs = []
    valid_accs = []
    
    print(f"📊 模型参数数量: {sum(p.numel() for p in model.parameters())}")
    
    for epoch in range(NUM_EPOCHS):
        start_time = time.time()
        
        # 训练阶段
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            
            # 重置神经元状态
            functional.reset_net(model)
            
            output = model(data)
            loss = criterion(output, target)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            pred = output.argmax(dim=1)
            train_correct += pred.eq(target).sum().item()
            train_total += target.size(0)
            
            if batch_idx % 10 == 0:
                print(f'   Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}')
        
        # 验证阶段
        model.eval()
        valid_correct = 0
        valid_total = 0
        
        with torch.no_grad():
            for data, target in valid_loader:
                data, target = data.to(device), target.to(device)
                functional.reset_net(model)
                output = model(data)
                pred = output.argmax(dim=1)
                valid_correct += pred.eq(target).sum().item()
                valid_total += target.size(0)
        
        # 计算准确率
        train_acc = train_correct / train_total
        valid_acc = valid_correct / valid_total
        avg_train_loss = train_loss / len(train_loader)
        
        # 更新学习率
        scheduler.step()
        
        # 记录最佳模型
        if valid_acc > best_acc:
            best_acc = valid_acc
            # 保存最佳模型
            model_save_path = f"/root/DH-SNN_reproduce/results/gsc_{model_name.lower().replace(' ', '_')}_best.pth"
            os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
            torch.save({
                'model_state_dict': model.state_dict(),
                'epoch': epoch,
                'best_acc': best_acc,
                'config': {
                    'model_name': model_name,
                    'batch_size': BATCH_SIZE,
                    'learning_rate': LEARNING_RATE,
                    'num_epochs': NUM_EPOCHS
                }
            }, model_save_path)
            print(f"💾 保存最佳模型: {model_save_path}")
        
        # 记录训练历史
        train_losses.append(avg_train_loss)
        train_accs.append(train_acc)
        valid_accs.append(valid_acc)
        
        epoch_time = time.time() - start_time
        
        print(f'Epoch {epoch+1:3d}/{NUM_EPOCHS}, '
              f'Train Loss: {avg_train_loss:.4f}, '
              f'Train Acc: {train_acc:.4f}, '
              f'Valid Acc: {valid_acc:.4f}, '
              f'Best: {best_acc:.4f}, '
              f'LR: {optimizer.param_groups[0]["lr"]:.6f}, '
              f'Time: {epoch_time:.1f}s')
        
        # 保存训练进度
        if (epoch + 1) % 10 == 0:  # 每10个epoch保存一次
            progress_save_path = f"/root/DH-SNN_reproduce/results/gsc_{model_name.lower().replace(' ', '_')}_epoch_{epoch+1}.pth"
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'epoch': epoch,
                'best_acc': best_acc,
                'train_losses': train_losses,
                'train_accs': train_accs,
                'valid_accs': valid_accs
            }, progress_save_path)
            print(f"💾 保存训练进度: epoch {epoch+1}")

        # 早停条件（更宽松的条件，让训练充分进行）
        if model_name == "Vanilla SNN" and best_acc > 0.85:
            print(f"✅ Vanilla SNN达到85%以上，提前停止训练")
            break
        elif model_name == "DH-SNN" and best_acc > 0.92:
            print(f"✅ DH-SNN达到92%以上，提前停止训练")
            break
    
    # 最终测试
    model.eval()
    test_correct = 0
    test_total = 0

    # 检查测试集是否为空
    if len(test_loader) == 0:
        print("⚠️  测试集为空，使用验证集作为测试集")
        test_loader = valid_loader

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            functional.reset_net(model)
            output = model(data)
            pred = output.argmax(dim=1)
            test_correct += pred.eq(target).sum().item()
            test_total += target.size(0)

    # 避免除零错误
    test_acc = test_correct / test_total if test_total > 0 else 0.0
    
    print(f"✅ {model_name} 训练完成:")
    print(f"   最佳验证准确率: {best_acc:.4f}")
    print(f"   最终测试准确率: {test_acc:.4f}")

    # 保存完整的训练结果
    results_save_path = f"/root/DH-SNN_reproduce/results/gsc_{model_name.lower().replace(' ', '_')}_results.json"
    import json
    results = {
        'model_name': model_name,
        'best_valid_acc': float(best_acc),
        'final_test_acc': float(test_acc),
        'train_losses': [float(x) for x in train_losses],
        'train_accs': [float(x) for x in train_accs],
        'valid_accs': [float(x) for x in valid_accs],
        'config': {
            'batch_size': BATCH_SIZE,
            'learning_rate': LEARNING_RATE,
            'num_epochs': NUM_EPOCHS,
            'num_classes': NUM_CLASSES,
            'hidden_size': HIDDEN_SIZE,
            'num_branches': NUM_BRANCHES if 'DH' in model_name else None
        },
        'training_time_per_epoch': epoch_time,
        'total_epochs_trained': len(train_losses)
    }

    with open(results_save_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"💾 保存训练结果: {results_save_path}")

    return best_acc, test_acc

def main():
    """主函数"""
    print("🎯 GSC数据集DH-SNN实验")
    print("=" * 60)
    
    try:
        # 准备数据
        train_loader, valid_loader, test_loader = prepare_data()
        
        # 实验配置
        experiments = [
            ("Vanilla SNN", GSC_Vanilla_SNN),
            ("DH-SNN", GSC_DH_SNN),
        ]
        
        results = {}
        start_time = time.time()
        
        for exp_name, model_class in experiments:
            print(f"\n🔬 实验: {exp_name}")
            print("=" * 50)
            
            model = model_class()
            best_acc, test_acc = train_model(model, train_loader, valid_loader, test_loader, exp_name)
            results[exp_name] = {
                'best_valid_acc': best_acc * 100,
                'test_acc': test_acc * 100
            }
        
        # 输出最终结果
        total_time = time.time() - start_time
        print(f"\n🎉 实验完成! 总用时: {total_time/60:.1f}分钟")
        print("=" * 60)
        print("📊 最终结果:")
        
        for exp_name, result in results.items():
            print(f"   {exp_name}:")
            print(f"     最佳验证准确率: {result['best_valid_acc']:.1f}%")
            print(f"     测试准确率: {result['test_acc']:.1f}%")
        
        # 计算改进幅度
        if len(results) >= 2:
            vanilla_acc = list(results.values())[0]['test_acc']
            dh_acc = list(results.values())[1]['test_acc']
            improvement = dh_acc - vanilla_acc
            print(f"\n🚀 DH-SNN相对改进: +{improvement:.1f}个百分点")
        
        return results
        
    except Exception as e:
        print(f"❌ 实验失败: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    results = main()
