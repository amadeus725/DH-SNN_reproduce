#!/usr/bin/env python3
"""
GSC DH-SNN 原论文代码的一一对应SpikingJelly实现
基于reference/original_paper_code/GSC/main_dense.py的精确复现
"""

import os
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.optim.lr_scheduler import StepLR
from torch.autograd import Variable
import warnings
warnings.filterwarnings("ignore")

# 添加SpikingJelly路径
sys.path.append('/root/DH-SNN_reproduce')
from spikingjelly.activation_based import neuron, functional, surrogate, layer

# 原论文设备和种子设置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(0)  # 原论文设置

print(f"🔧 设备: {device}")

# 原论文配置参数
batch_size = 200  # 原论文
learning_rate = 1e-2  # 原论文
epochs = 150  # 原论文
n = 200  # 原论文隐藏层大小
is_bias = True  # 原论文

# 原论文数据路径配置
train_data_root = "/root/DH-SNN_reproduce/datasets/speech_commands"
test_data_root = "/root/DH-SNN_reproduce/datasets/speech_commands"

# 原论文音频处理参数
sr = 16000
size = 16000
n_fft = int(30e-3*sr)
hop_length = int(10e-3*sr)
n_mels = 40
fmax = 4000
fmin = 20
delta_order = 2
stack = True

print(f"📊 原论文配置:")
print(f"  批次大小: {batch_size}")
print(f"  学习率: {learning_rate}")
print(f"  训练轮数: {epochs}")
print(f"  隐藏层大小: {n}")

# ============================================================================
# 原论文数据处理代码的精确对应
# ============================================================================

def txt2list(file_path):
    """原论文工具函数"""
    with open(file_path, 'r') as f:
        return [line.strip() for line in f.readlines()]

def get_random_noise(noise_files, size):
    """原论文噪声生成函数"""
    import scipy.io.wavfile as wav
    noise_file = np.random.choice(noise_files)
    _, noise = wav.read(noise_file)
    noise = noise.astype(float)
    
    if len(noise) < size:
        repeat_times = (size // len(noise)) + 1
        noise = np.tile(noise, repeat_times)
    
    start_idx = np.random.randint(0, len(noise) - size + 1)
    return noise[start_idx:start_idx + size]

def generate_random_silence_files(num_files, noise_files, size, prefix):
    """原论文静音文件生成"""
    import scipy.io.wavfile as wav
    for i in range(num_files):
        silence = get_random_noise(noise_files, size) * 0.1
        filename = f"{prefix}_{i:04d}.wav"
        wav.write(filename, sr, silence.astype(np.int16))

# 原论文标签字典设置
testing_words = ["yes", "no", 'up', 'down', 'left', 'right', 'on', 'off', 'stop', 'go']
label_dct = {k: i for i, k in enumerate(testing_words + ["_silence_", "_unknown_"])}

# 为未知词分配标签
training_words = os.listdir(train_data_root)
training_words = [x for x in training_words if os.path.isdir(os.path.join(train_data_root, x))]
training_words = [x for x in training_words if x[0] != "_"]

for w in training_words:
    label = label_dct.get(w)
    if label is None:
        label_dct[w] = label_dct["_unknown_"]

print(f"📋 标签字典: {label_dct}")

# 原论文噪声文件处理
noise_path = os.path.join(train_data_root, "_background_noise_")
noise_files = []
if os.path.exists(noise_path):
    for f in os.listdir(noise_path):
        if f.endswith(".wav"):
            full_name = os.path.join(noise_path, f)
            noise_files.append(full_name)

# 原论文静音数据生成
silence_folder = os.path.join(train_data_root, "_silence_")
if not os.path.exists(silence_folder) and noise_files:
    os.makedirs(silence_folder)
    generate_random_silence_files(2560, noise_files, size, os.path.join(silence_folder, "rd_silence"))
    
    silence_files = [fname for fname in os.listdir(silence_folder)]
    with open(os.path.join(train_data_root, "silence_validation_list.txt"), "w") as f:
        f.writelines("_silence_/" + fname + "\n" for fname in silence_files[:260])

class SpeechCommandsDataset(torch.utils.data.Dataset):
    """原论文数据集类的精确对应"""
    def __init__(self, data_root, label_dct, transform=None, mode="train", max_nb_per_class=None):
        assert mode in ["train", "valid", "test"], 'mode should be "train", "valid" or "test"'
        
        self.filenames = []
        self.labels = []
        self.mode = mode
        self.transform = transform
        
        # 原论文数据分割逻辑
        if self.mode == "train" or self.mode == "valid":
            testing_list = txt2list(os.path.join(data_root, "testing_list.txt"))
            validation_list = txt2list(os.path.join(data_root, "validation_list.txt"))
            if os.path.exists(os.path.join(data_root, "silence_validation_list.txt")):
                validation_list += txt2list(os.path.join(data_root, "silence_validation_list.txt"))
        else:
            testing_list = txt2list(os.path.join(data_root, "testing_list.txt"))
            if os.path.exists(os.path.join(data_root, "silence_validation_list.txt")):
                testing_list += txt2list(os.path.join(data_root, "silence_validation_list.txt"))
            validation_list = []
        
        # 原论文文件遍历逻辑
        for root, dirs, files in os.walk(data_root):
            if "_background_noise_" in root:
                continue
            for filename in files:
                if not filename.endswith('.wav'):
                    continue
                command = root.split("/")[-1]
                label = label_dct.get(command)
                if label is None:
                    print("ignored command: %s" % command)
                    break
                partial_path = '/'.join([command, filename])
                
                testing_file = (partial_path in testing_list)
                validation_file = (partial_path in validation_list)
                training_file = not testing_file and not validation_file
                
                if ((self.mode == "test" and testing_file) or 
                    (self.mode == "train" and training_file) or 
                    (self.mode == "valid" and validation_file)):
                    full_name = os.path.join(root, filename)
                    self.filenames.append(full_name)
                    self.labels.append(label)
        
        # 原论文类别限制逻辑
        if max_nb_per_class is not None:
            selected_idx = []
            for label in np.unique(self.labels):
                label_idx = [i for i, x in enumerate(self.labels) if x == label]
                if len(label_idx) < max_nb_per_class:
                    selected_idx += label_idx
                else:
                    selected_idx += list(np.random.choice(label_idx, max_nb_per_class))
            
            self.filenames = [self.filenames[idx] for idx in selected_idx]
            self.labels = [self.labels[idx] for idx in selected_idx]
        
        # 原论文权重计算逻辑
        if self.mode == "train":
            unique_labels, counts = np.unique(self.labels, return_counts=True)
            label_weights = 1. / counts
            label_weights /= np.sum(label_weights)
            # 创建标签到权重的映射
            label_to_weight = {label: weight for label, weight in zip(unique_labels, label_weights)}
            self.weights = torch.DoubleTensor([label_to_weight[label] for label in self.labels])
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        import scipy.io.wavfile as wav
        filename = self.filenames[idx]
        item = wav.read(filename)[1].astype(float)
        m = np.max(np.abs(item))
        if m > 0:
            item /= m
        if self.transform is not None:
            item = self.transform(item)
        
        label = self.labels[idx]
        return item, label

class Pad:
    """原论文Pad类的精确对应"""
    def __init__(self, size):
        self.size = size
    
    def __call__(self, wav):
        wav_size = wav.shape[0]
        pad_size = (self.size - wav_size) // 2
        padded_wav = np.pad(wav, ((pad_size, self.size - wav_size - pad_size),), 'constant', constant_values=(0, 0))
        return padded_wav

class MelSpectrogram:
    """原论文MelSpectrogram类的精确对应"""
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
        import librosa
        S = librosa.feature.melspectrogram(y=wav,
                                         sr=self.sr,
                                         n_fft=self.n_fft,
                                         hop_length=self.hop_length,
                                         n_mels=self.n_mels,
                                         fmax=self.fmax,
                                         fmin=self.fmin)
        
        M = np.max(np.abs(S))
        if M > 0:
            feat = np.log1p(S / M)
        else:
            feat = S
        
        if self.delta_order is not None and not self.stack:
            feat = librosa.feature.delta(feat, order=self.delta_order)
            return np.expand_dims(feat.T, 0)
        
        elif self.delta_order is not None and self.stack:
            feat_list = [feat.T]
            for k in range(1, self.delta_order + 1):
                feat_list.append(librosa.feature.delta(feat, order=k).T)
            return np.stack(feat_list)
        
        else:
            return np.expand_dims(feat.T, 0)

class Rescale:
    """原论文Rescale类的精确对应"""
    def __call__(self, input):
        std = np.std(input, axis=1, keepdims=True)
        std[std == 0] = 1
        return input / std

# 原论文数据变换组合
import torchvision
melspec = MelSpectrogram(sr, n_fft, hop_length, n_mels, fmin, fmax, delta_order, stack=stack)
pad = Pad(size)
rescale = Rescale()
transform = torchvision.transforms.Compose([pad, melspec, rescale])

# 原论文collate_fn
def collate_fn(data):
    X_batch = np.array([d[0] for d in data])
    std = X_batch.std(axis=(0, 2), keepdims=True)
    X_batch = torch.tensor(X_batch / std)
    y_batch = torch.tensor([d[1] for d in data])
    return X_batch, y_batch

# ============================================================================
# 原论文神经网络层的精确对应
# ============================================================================

# 原论文激活函数
class ActFun_adp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return input.gt(0.).float()

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        temp = abs(input) < 0.5
        return grad_input * temp.float()

act_fun_adp = ActFun_adp.apply

# 原论文神经元更新函数
R_m = 1  # 膜电阻

def mem_update_pra(inputs, mem, spike, v_th, tau_m, dt=1, device=None):
    """原论文神经元更新 - 软重置"""
    alpha = torch.sigmoid(tau_m)
    mem = mem * alpha + (1 - alpha) * R_m * inputs - v_th * spike
    inputs_ = mem - v_th
    spike = act_fun_adp(inputs_)
    return mem, spike

def output_Neuron_pra(inputs, mem, tau_m, dt=1, device=None):
    """原论文读出神经元 - 无脉冲积分器"""
    alpha = torch.sigmoid(tau_m).to(device)
    mem = mem * alpha + (1 - alpha) * inputs
    return mem

# 原论文读出积分器
class readout_integrator_test(nn.Module):
    """原论文readout_integrator_test的精确对应"""
    def __init__(self, input_dim, output_dim, tau_minitializer='uniform',
                 low_m=0, high_m=4, device='cpu', bias=True, dt=1):
        super(readout_integrator_test, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.device = device
        self.dt = dt

        # 原论文：self.dense = nn.Linear(input_dim,output_dim,bias=bias)
        self.dense = nn.Linear(input_dim, output_dim, bias=bias)

        # 原论文：self.tau_m = nn.Parameter(torch.Tensor(self.output_dim))
        self.tau_m = nn.Parameter(torch.Tensor(self.output_dim))

        # 原论文初始化
        if tau_minitializer == 'uniform':
            nn.init.uniform_(self.tau_m, low_m, high_m)
        elif tau_minitializer == 'constant':
            nn.init.constant_(self.tau_m, low_m)

    def set_neuron_state(self, batch_size):
        """原论文：self.mem = (torch.rand(batch_size,self.output_dim)).to(self.device)"""
        self.mem = torch.rand(batch_size, self.output_dim).to(self.device)

    def forward(self, input_spike):
        """原论文前向传播的精确对应"""
        # 原论文：d_input = self.dense(input_spike.float())
        d_input = self.dense(input_spike.float())

        # 原论文：self.mem = output_Neuron_pra(d_input,self.mem,self.tau_m,self.dt,device=self.device)
        self.mem = output_Neuron_pra(d_input, self.mem, self.tau_m, self.dt, device=self.device)

        return self.mem

# 原论文DH-SFNN层
class spike_dense_test_denri_wotanh_R(nn.Module):
    """原论文spike_dense_test_denri_wotanh_R的精确对应"""
    def __init__(self, input_dim, output_dim, tau_minitializer='uniform', low_m=0, high_m=4,
                 tau_ninitializer='uniform', low_n=0, high_n=4, vth=0.5, dt=1, branch=4,
                 device='cpu', bias=True, test_sparsity=False, sparsity=0.5, mask_share=1):
        super(spike_dense_test_denri_wotanh_R, self).__init__()

        # 原论文参数设置
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.device = device
        self.vth = vth
        self.dt = dt
        self.branch = branch
        self.test_sparsity = test_sparsity
        self.mask_share = mask_share

        # 原论文稀疏性设置
        if test_sparsity:
            self.sparsity = sparsity
        else:
            self.sparsity = 1 / branch

        # 原论文填充计算
        self.pad = ((input_dim) // branch * branch + branch - input_dim) % branch

        # 原论文：self.dense = nn.Linear(input_dim+self.pad,output_dim*branch)
        self.dense = nn.Linear(input_dim + self.pad, output_dim * branch, bias=bias)

        # 原论文参数
        self.tau_m = nn.Parameter(torch.Tensor(self.output_dim))
        self.tau_n = nn.Parameter(torch.Tensor(self.output_dim, branch))

        # 原论文连接掩码
        self.create_mask()

        # 原论文参数初始化
        if tau_minitializer == 'uniform':
            nn.init.uniform_(self.tau_m, low_m, high_m)
        elif tau_minitializer == 'constant':
            nn.init.constant_(self.tau_m, low_m)

        if tau_ninitializer == 'uniform':
            nn.init.uniform_(self.tau_n, low_n, high_n)
        elif tau_ninitializer == 'constant':
            nn.init.constant_(self.tau_n, low_n)

    def set_neuron_state(self, batch_size):
        """原论文神经元状态初始化的精确对应"""
        # 原论文：self.mem = Variable(torch.rand(batch_size,self.output_dim)).to(self.device)
        self.mem = Variable(torch.rand(batch_size, self.output_dim)).to(self.device)

        # 原论文：self.spike = Variable(torch.rand(batch_size,self.output_dim)).to(self.device)
        self.spike = Variable(torch.rand(batch_size, self.output_dim)).to(self.device)

        # 原论文树突电流初始化
        if self.branch == 1:
            self.d_input = Variable(torch.rand(batch_size, self.output_dim, self.branch)).to(self.device)
        else:
            self.d_input = Variable(torch.zeros(batch_size, self.output_dim, self.branch)).to(self.device)

        # 原论文：self.v_th = Variable(torch.ones(batch_size,self.output_dim)*self.vth).to(self.device)
        self.v_th = Variable(torch.ones(batch_size, self.output_dim) * self.vth).to(self.device)

    def create_mask(self):
        """原论文连接掩码创建的精确对应"""
        input_size = self.input_dim + self.pad
        self.mask = torch.zeros(self.output_dim * self.branch, input_size).to(self.device)

        for i in range(self.output_dim // self.mask_share):
            seq = torch.randperm(input_size)
            for j in range(self.branch):
                if self.test_sparsity:
                    if j * input_size // self.branch + int(input_size * self.sparsity) > input_size:
                        for k in range(self.mask_share):
                            self.mask[(i * self.mask_share + k) * self.branch + j, seq[j * input_size // self.branch:-1]] = 1
                            self.mask[(i * self.mask_share + k) * self.branch + j, seq[:j * input_size // self.branch + int(input_size * self.sparsity) - input_size]] = 1
                    else:
                        for k in range(self.mask_share):
                            self.mask[(i * self.mask_share + k) * self.branch + j, seq[j * input_size // self.branch:j * input_size // self.branch + int(input_size * self.sparsity)]] = 1
                else:
                    for k in range(self.mask_share):
                        self.mask[(i * self.mask_share + k) * self.branch + j, seq[j * input_size // self.branch:(j + 1) * input_size // self.branch]] = 1

    def apply_mask(self):
        """原论文掩码应用的精确对应"""
        self.dense.weight.data = self.dense.weight.data * self.mask

    def forward(self, input_spike):
        """原论文前向传播的精确对应"""
        # 原论文：beta = torch.sigmoid(self.tau_n)
        beta = torch.sigmoid(self.tau_n)

        # 原论文：padding = torch.zeros(input_spike.size(0),self.pad).to(self.device)
        padding = torch.zeros(input_spike.size(0), self.pad).to(self.device)

        # 原论文：k_input = torch.cat((input_spike.float(),padding),1)
        k_input = torch.cat((input_spike.float(), padding), 1)

        # 原论文：self.d_input = beta*self.d_input+(1-beta)*self.dense(k_input).reshape(-1,self.output_dim,self.branch)
        self.d_input = beta * self.d_input + (1 - beta) * self.dense(k_input).reshape(-1, self.output_dim, self.branch)

        # 原论文：l_input = (self.d_input).sum(dim=2,keepdim=False)
        l_input = (self.d_input).sum(dim=2, keepdim=False)

        # 原论文：self.mem,self.spike = mem_update_pra(l_input,self.mem,self.spike,self.v_th,self.tau_m,self.dt,device=self.device)
        self.mem, self.spike = mem_update_pra(l_input, self.mem, self.spike, self.v_th, self.tau_m, self.dt, device=self.device)

        return self.mem, self.spike

# 原论文Dense_test网络
class Dense_test(nn.Module):
    """原论文Dense_test网络的精确对应"""
    def __init__(self):
        super(Dense_test, self).__init__()

        # 原论文：n = 200
        n = 200

        # 原论文网络层的精确对应
        # self.dense_1 = spike_dense_test_denri_wotanh_R(40*3,n,vth= 1,dt = 1,branch = 8,device=device,bias=is_bias)
        self.dense_1 = spike_dense_test_denri_wotanh_R(40*3, n, vth=1, dt=1, branch=8, device=device, bias=is_bias)

        # self.dense_2 = spike_dense_test_denri_wotanh_R(n,n,vth= 1,dt = 1,branch = 8,device=device,bias=is_bias)
        self.dense_2 = spike_dense_test_denri_wotanh_R(n, n, vth=1, dt=1, branch=8, device=device, bias=is_bias)

        # self.dense_3 = spike_dense_test_denri_wotanh_R(n,n,vth= 1,dt = 1,branch = 8,device=device,bias=is_bias)
        self.dense_3 = spike_dense_test_denri_wotanh_R(n, n, vth=1, dt=1, branch=8, device=device, bias=is_bias)

        # self.dense_4 = readout_integrator_test(n,12,dt = 1,device=device,bias=is_bias)
        self.dense_4 = readout_integrator_test(n, 12, dt=1, device=device, bias=is_bias)

    def forward(self, input):
        """原论文前向传播的精确对应"""
        # 原论文：input.to(device)
        input = input.to(device)

        # 原论文：b,channel,seq_length,input_dim = input.shape
        b, channel, seq_length, input_dim = input.shape

        # 原论文神经元状态初始化
        self.dense_1.set_neuron_state(b)
        self.dense_2.set_neuron_state(b)
        self.dense_3.set_neuron_state(b)
        self.dense_4.set_neuron_state(b)

        output = 0
        input_s = input

        # 原论文时间步循环
        for i in range(seq_length):
            # 原论文：input_x = input_s[:,:,i,:].reshape(b,channel*input_dim)
            input_x = input_s[:, :, i, :].reshape(b, channel * input_dim)

            # 原论文：mem_layer1,spike_layer1 = self.dense_1.forward(input_x)
            mem_layer1, spike_layer1 = self.dense_1.forward(input_x)

            # 原论文：mem_layer2,spike_layer2 = self.dense_2.forward(spike_layer1)
            mem_layer2, spike_layer2 = self.dense_2.forward(spike_layer1)

            # 原论文：mem_layer3,spike_layer3 = self.dense_3.forward(spike_layer2)
            mem_layer3, spike_layer3 = self.dense_3.forward(spike_layer2)

            # 原论文：mem_layer4= self.dense_4.forward(spike_layer3)
            mem_layer4 = self.dense_4.forward(spike_layer3)

            output += mem_layer4

        # 原论文：output = F.log_softmax(output/seq_length,dim=1)
        output = F.log_softmax(output / seq_length, dim=1)
        return output

# ============================================================================
# 原论文训练和测试函数的精确对应
# ============================================================================

def test(data_loader, model, is_show=0):
    """原论文test函数的精确对应"""
    test_acc = 0.
    sum_sample = 0.
    fr_ = []
    for i, (images, labels) in enumerate(data_loader):
        # 原论文：apply the connection pattern
        model.dense_1.apply_mask()
        model.dense_2.apply_mask()
        model.dense_3.apply_mask()

        # 原论文：images = images.view(-1,3,101, 40).to(device)
        images = images.view(-1, 3, 101, 40).to(device)

        # 原论文：labels = labels.view((-1)).long().to(device)
        labels = labels.view((-1)).long().to(device)

        # 原论文：predictions= model(images)
        predictions = model(images)

        # 原论文：_, predicted = torch.max(predictions.data, 1)
        _, predicted = torch.max(predictions.data, 1)

        labels = labels.cpu()
        predicted = predicted.cpu().t()

        test_acc += (predicted == labels).sum()
        sum_sample += predicted.numel()

    return test_acc.data.cpu().numpy() / sum_sample

def train(epochs, criterion, optimizer, scheduler, model, train_dataloader, test_dataloader):
    """原论文train函数的精确对应"""
    acc_list = []
    best_acc = 0
    path = 'model/dense_layer3_200neuron_denri_branch8_initzero_MG'  # 原论文路径

    for epoch in range(epochs):
        train_acc = 0
        sum_sample = 0
        train_loss_sum = 0

        for i, (images, labels) in enumerate(train_dataloader):
            # 原论文：apply the connection pattern
            model.dense_1.apply_mask()
            model.dense_2.apply_mask()
            model.dense_3.apply_mask()

            # 原论文：images = images.view(-1,3,101, 40).to(device)
            images = images.view(-1, 3, 101, 40).to(device)

            # 原论文：labels = labels.view((-1)).long().to(device)
            labels = labels.view((-1)).long().to(device)

            optimizer.zero_grad()

            # 原论文：predictions= model(images)
            predictions = model(images)

            # 原论文：_, predicted = torch.max(predictions.data, 1)
            _, predicted = torch.max(predictions.data, 1)

            # 原论文：train_loss = criterion(predictions,labels)
            train_loss = criterion(predictions, labels)

            train_loss.backward()
            train_loss_sum += train_loss.item()
            optimizer.step()

            labels = labels.cpu()
            predicted = predicted.cpu().t()

            train_acc += (predicted == labels).sum()
            sum_sample += predicted.numel()

        if scheduler:
            scheduler.step()

        train_acc = train_acc.data.cpu().numpy() / sum_sample
        valid_acc = test(test_dataloader, model, 1)
        train_loss_sum += train_loss

        acc_list.append(train_acc)
        print('lr: ', optimizer.param_groups[0]["lr"])

        # 原论文保存条件
        if valid_acc > best_acc and train_acc > 0.890:
            best_acc = valid_acc
            os.makedirs('model', exist_ok=True)
            torch.save(model, path + str(best_acc)[:7] + '-srnn.pth')

        print('epoch: {:3d}, Train Loss: {:.4f}, Train Acc: {:.4f}, Valid Acc: {:.4f}'.format(
            epoch, train_loss_sum / len(train_dataloader), train_acc, valid_acc), flush=True)

    return acc_list

# ============================================================================
# 原论文主程序的精确对应
# ============================================================================

def main():
    """原论文主程序"""
    print("🚀 开始GSC DH-SNN训练 (原论文精确复现)")
    print("=" * 60)

    # 原论文数据集创建
    train_dataset = SpeechCommandsDataset(train_data_root, label_dct, transform=transform, mode="train", max_nb_per_class=None)
    train_sampler = torch.utils.data.WeightedRandomSampler(train_dataset.weights, len(train_dataset.weights))
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, num_workers=8, sampler=train_sampler, collate_fn=collate_fn)

    valid_dataset = SpeechCommandsDataset(train_data_root, label_dct, transform=transform, mode="valid", max_nb_per_class=None)
    valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True, num_workers=8, collate_fn=collate_fn)

    test_dataset = SpeechCommandsDataset(test_data_root, label_dct, transform=transform, mode="test")
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=8, collate_fn=collate_fn)

    print(f"📊 数据集大小:")
    print(f"  训练集: {len(train_dataset)} 样本")
    print(f"  验证集: {len(valid_dataset)} 样本")
    print(f"  测试集: {len(test_dataset)} 样本")

    # 原论文模型创建
    model = Dense_test()
    criterion = nn.CrossEntropyLoss()

    print("device:", device)
    model.to(device)

    print(f"📊 模型参数数量: {sum(p.numel() for p in model.parameters()):,}")

    # 原论文初始测试
    test_acc = test(test_dataloader, model)
    print(f"初始测试准确率: {test_acc}")

    # 原论文优化器设置
    if is_bias:
        base_params = [
            model.dense_1.dense.weight,
            model.dense_1.dense.bias,
            model.dense_2.dense.weight,
            model.dense_2.dense.bias,
            model.dense_3.dense.weight,
            model.dense_3.dense.bias,
            model.dense_4.dense.weight,
            model.dense_4.dense.bias,
        ]

    # 原论文优化器配置
    optimizer = torch.optim.Adam([
        {'params': base_params, 'lr': learning_rate},
        {'params': model.dense_4.tau_m, 'lr': learning_rate * 2},
        {'params': model.dense_1.tau_m, 'lr': learning_rate * 2},
        {'params': model.dense_1.tau_n, 'lr': learning_rate * 2},
        {'params': model.dense_2.tau_m, 'lr': learning_rate * 2},
        {'params': model.dense_2.tau_n, 'lr': learning_rate * 2},
        {'params': model.dense_3.tau_m, 'lr': learning_rate * 2},
        {'params': model.dense_3.tau_n, 'lr': learning_rate * 2},
    ], lr=learning_rate)

    # 原论文调度器
    scheduler = StepLR(optimizer, step_size=25, gamma=.5)

    # 原论文训练
    acc_list = train(epochs, criterion, optimizer, scheduler, model, train_dataloader, test_dataloader)

    # 原论文最终测试
    test_acc = test(test_dataloader, model)
    print(f"最终测试准确率: {test_acc}")

    return acc_list, test_acc

if __name__ == "__main__":
    main()
