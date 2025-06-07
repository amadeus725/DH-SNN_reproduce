#!/usr/bin/env python3
"""
Optimized NeuroVPR Experiment with DH-SNN
==========================================

This script implements the optimized NeuroVPR experiment using the identified
performance gap solutions:

1. Differentiated learning rates for time constants
2. Short time constants for short sequence tasks
3. Reduced model complexity
4. Temporal step fusion
5. Gradient clipping

Authors: DH-SNN Reproduction Study
Date: 2025
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
import json
from pathlib import Path

# Add project paths
sys.path.append('/root/DH-SNN_reproduce')
sys.path.append('/root/DH-SNN_reproduce/src')

from src.core.models import DH_SNN, create_dh_snn
from src.core.surrogate import MultiGaussianSurrogate
from spikingjelly.activation_based import neuron, functional, surrogate, layer
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
from pathlib import Path

# ==================== Optimized Models ====================

class OptimizedNeuroVPR_DH_SNN(nn.Module):
    """Optimized DH-SNN model for NeuroVPR task"""
    
    def __init__(self, input_dim=2752, hidden_dims=[512], output_dim=25, num_branches=2):
        super().__init__()
        
        print(f"🏗️  Creating Optimized NeuroVPR DH-SNN:")
        print(f"   - Input dim: {input_dim}")
        print(f"   - Hidden dims: {hidden_dims}")
        print(f"   - Output dim: {output_dim}")
        print(f"   - Branches: {num_branches} (reduced complexity)")
        
        # Optimized configuration for NeuroVPR task
        dh_snn_config = {
            'input_dim': input_dim,
            'hidden_dims': hidden_dims,
            'output_dim': output_dim,
            'num_branches': num_branches,  # Reduced branches for lower complexity
            'v_threshold': 0.3,  # Moderate threshold
            'tau_m_init': (0.2, 1.5),  # Short time constants for short sequences
            'tau_n_init': (0.2, 1.5),  # Short time constants for short sequences
            'tau_initializer': 'uniform',
            'sparsity': 1.0/num_branches,
            'mask_share': 1,
            'bias': True,
            'surrogate_function': MultiGaussianSurrogate(),
            'reset_mode': 'soft',
            'step_mode': 's'
        }
        
        self.dh_snn = create_dh_snn(dh_snn_config)
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.T = 3
        
        print("✅ Optimized DH-SNN model created successfully")
    
    def forward(self, inp):
        """Optimized forward pass with temporal fusion"""
        dvs_inp = inp[2]  # [batch, 3, 2, 32, 43]
        batch_size = dvs_inp.shape[0]
        
        # Reshape temporal dimension
        dvs_reshaped = dvs_inp.view(batch_size, self.T, -1)
        dvs_input = dvs_reshaped.transpose(0, 1)  # [3, batch, features]
        
        # Process all time steps and accumulate
        outputs = []
        for t in range(self.T):
            output = self.dh_snn(dvs_input[t])
            outputs.append(output)
        
        # Temporal fusion: weighted average with higher weights for later steps
        weights = torch.softmax(torch.tensor([0.5, 0.7, 1.0]), dim=0).to(outputs[0].device)
        final_output = sum(w * out for w, out in zip(weights, outputs))
        
        return final_output

class OptimizedNeuroVPR_Vanilla_SNN(nn.Module):
    """Optimized Vanilla SNN for comparison"""
    
    def __init__(self, input_dim=2752, hidden_dims=[512], output_dim=25):
        super().__init__()
        
        self.layers = nn.ModuleList()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.T = 3
        
        # Build network
        current_dim = input_dim
        for hidden_dim in hidden_dims:
            self.layers.append(layer.Linear(current_dim, hidden_dim))
            self.layers.append(neuron.LIFNode(
                tau=2.0,  # Moderate time constant
                v_threshold=0.3,  # Moderate threshold
                surrogate_function=surrogate.ATan(),
                step_mode='s'
            ))
            current_dim = hidden_dim
        
        # Output layer
        self.layers.append(layer.Linear(current_dim, output_dim))
    
    def forward(self, inp):
        dvs_inp = inp[2]  # [batch, 3, 2, 32, 43]
        batch_size = dvs_inp.shape[0]
        
        dvs_reshaped = dvs_inp.view(batch_size, self.T, -1)
        dvs_input = dvs_reshaped.transpose(0, 1)  # [3, batch, features]
        
        outputs = []
        for t in range(self.T):
            x = dvs_input[t]
            for layer_module in self.layers:
                x = layer_module(x)
            outputs.append(x)
        
        # Same temporal fusion strategy
        weights = torch.softmax(torch.tensor([0.5, 0.7, 1.0]), dim=0).to(outputs[0].device)
        final_output = sum(w * out for w, out in zip(weights, outputs))
        
        return final_output

# ==================== Optimized Training ====================

def create_optimized_optimizer(model, base_lr=1e-3):
    """Create optimizer with differentiated learning rates"""
    
    # Separate different types of parameters
    base_params = []
    tau_params = []
    
    for name, param in model.named_parameters():
        if 'tau' in name.lower():
            tau_params.append(param)
            print(f"Time constant parameter: {name}")
        else:
            base_params.append(param)
    
    # Create optimizer with differentiated learning rates
    optimizer = optim.Adam([
        {'params': base_params, 'lr': base_lr},
        {'params': tau_params, 'lr': base_lr * 0.5}  # Lower LR for time constants
    ])
    
    print(f"✅ Differentiated learning rate optimizer:")
    print(f"   - Base parameters: lr = {base_lr}")
    print(f"   - Time constants: lr = {base_lr * 0.5}")
    
    return optimizer

def train_optimized_model(model, train_dataset, test_dataset, model_name, num_epochs=30):
    """Optimized training process"""
    
    print(f"\n🚀 Starting optimized {model_name} training")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Use differentiated learning rate optimizer
    optimizer = create_optimized_optimizer(model, base_lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    
    best_test_acc = 0.0
    train_losses = []
    test_accuracies = []
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch_idx, (inputs, targets) in enumerate(train_dataset):
            targets = targets.to(device)
            inputs = [inp.to(device) for inp in inputs]
            
            optimizer.zero_grad()
            functional.reset_net(model)
            
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            
            # Gradient clipping to prevent gradient explosion
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += targets.size(0)
            train_correct += predicted.eq(targets).sum().item()
            
            if batch_idx % 20 == 0:
                print(f"   Batch {batch_idx}: Loss = {loss.item():.4f}")
        
        # Testing phase
        model.eval()
        test_correct = 0
        test_total = 0
        test_loss = 0.0
        
        with torch.no_grad():
            for inputs, targets in test_dataset:
                targets = targets.to(device)
                inputs = [inp.to(device) for inp in inputs]
                
                functional.reset_net(model)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                
                test_loss += loss.item()
                _, predicted = outputs.max(1)
                test_total += targets.size(0)
                test_correct += predicted.eq(targets).sum().item()
        
        train_acc = 100.0 * train_correct / train_total
        test_acc = 100.0 * test_correct / test_total
        avg_train_loss = running_loss / len(train_dataset)
        
        train_losses.append(avg_train_loss)
        test_accuracies.append(test_acc)
        
        if test_acc > best_test_acc:
            best_test_acc = test_acc
        
        scheduler.step()
        
        print(f"Epoch [{epoch+1}/{num_epochs}]:")
        print(f"  Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"  Test Acc: {test_acc:.2f}% (Best: {best_test_acc:.2f}%)")
        print(f"  Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
    
    return {
        'best_test_acc': best_test_acc,
        'final_test_acc': test_acc,
        'train_losses': train_losses,
        'test_accuracies': test_accuracies
    }

# ==================== Data Loading ====================

# Import from the original experiment
from collections import Counter
import random

class FixedNeuroVPRDataset:
    """Fixed NeuroVPR dataset loader from original experiment"""
    
    def __init__(self, data_path, exp_names, batch_size=16, is_shuffle=True, nclass=50, 
                 split_type='train', train_ratio=0.7, position_granularity=5.0):
        self.data_path = Path(data_path)
        self.exp_names = exp_names
        self.batch_size = batch_size
        self.is_shuffle = is_shuffle
        self.nclass = nclass
        self.split_type = split_type
        self.train_ratio = train_ratio
        self.position_granularity = position_granularity
        
        # Load data
        self.data_samples = []
        self.labels = []
        
        for exp_name in exp_names:
            exp_path = self.data_path / exp_name
            if not exp_path.exists():
                print(f"Warning: Dataset {exp_name} does not exist at {exp_path}")
                continue
                
            self._load_experiment_data(exp_path)
        
        print(f"Original data loaded: {len(self.data_samples)} samples")
        
        # Analyze class distribution
        self._analyze_class_distribution()
        
        # Perform train/test split
        self._perform_train_test_split()
        
        print(f"Data split completed ({split_type}): {len(self.data_samples)} samples")
        
        # Create batch indices
        self.indices = list(range(len(self.data_samples)))
        if self.is_shuffle:
            random.shuffle(self.indices)
        
        self.batch_indices = []
        for i in range(0, len(self.indices), batch_size):
            batch = self.indices[i:i+batch_size]
            if len(batch) == batch_size:  # Keep only complete batches
                self.batch_indices.append(batch)
    
    def _analyze_class_distribution(self):
        """Analyze class distribution"""
        label_counts = Counter(self.labels)
        print(f"Class analysis:")
        print(f"  Total classes: {len(label_counts)}")
        print(f"  Sample range: {min(label_counts.values())} - {max(label_counts.values())}")
        print(f"  Average per class: {np.mean(list(label_counts.values())):.1f}")
        
        # Filter classes with too few samples
        min_samples_per_class = 10
        valid_classes = set([label for label, count in label_counts.items() if count >= min_samples_per_class])
        
        if len(valid_classes) < len(label_counts):
            print(f"  Filtering classes with <{min_samples_per_class} samples: {len(label_counts) - len(valid_classes)} classes")
            
            # Re-filter data
            filtered_samples = []
            filtered_labels = []
            for sample, label in zip(self.data_samples, self.labels):
                if label in valid_classes:
                    filtered_samples.append(sample)
                    filtered_labels.append(label)
            
            self.data_samples = filtered_samples
            self.labels = filtered_labels
            
            print(f"  After filtering: {len(self.data_samples)} samples, {len(valid_classes)} classes")
    
    def _perform_train_test_split(self):
        """Perform train/test split stratified by class"""
        if len(self.data_samples) == 0:
            return
        
        # Set random seed for reproducibility
        random.seed(42)
        np.random.seed(42)
        
        # Group by class
        class_samples = {}
        for i, label in enumerate(self.labels):
            if label not in class_samples:
                class_samples[label] = []
            class_samples[label].append(i)
        
        # Stratified sampling for each class
        train_indices = []
        test_indices = []
        
        for label, indices in class_samples.items():
            random.shuffle(indices)
            train_size = int(len(indices) * self.train_ratio)
            
            train_indices.extend(indices[:train_size])
            test_indices.extend(indices[train_size:])
        
        # Select indices based on split type
        if self.split_type == 'train':
            selected_indices = train_indices
        else:  # test
            selected_indices = test_indices
        
        # Rebuild data based on selected indices
        selected_samples = [self.data_samples[i] for i in selected_indices]
        selected_labels = [self.labels[i] for i in selected_indices]
        
        self.data_samples = selected_samples
        self.labels = selected_labels
        
        print(f"Stratified {self.split_type} split: {len(self.data_samples)} samples")
    
    def _load_experiment_data(self, exp_path):
        """Load data from single experiment"""
        dvs_path = exp_path / "dvs_7ms_3seq" 
        position_file = exp_path / "position.txt"
        
        if not dvs_path.exists():
            print(f"Warning: DVS data path does not exist: {dvs_path}")
            return
        
        if not position_file.exists():
            print(f"Warning: Position file does not exist: {position_file}")
            return
        
        # Read all DVS file timestamps
        dvs_files = sorted(list(dvs_path.glob("*.npy")))  # Use .npy files
        dvs_timestamps = []
        dvs_file_map = {}
        
        for dvs_file in dvs_files:
            try:
                filename_base = dvs_file.stem
                timestamp = float(filename_base)
                dvs_timestamps.append(timestamp)
                dvs_file_map[timestamp] = dvs_file
            except (ValueError, IndexError):
                continue
        
        print(f"Found {len(dvs_timestamps)} valid DVS files")
        
        # Read position labels
        position_data = []
        with open(position_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 4:
                    try:
                        timestamp = float(parts[0])
                        x, y = float(parts[1]), float(parts[2])
                        position_data.append((timestamp, x, y))
                    except (ValueError, IndexError):
                        continue
        
        print(f"Loaded {len(position_data)} position labels")
        
        # Get time overlap range
        if not position_data or not dvs_timestamps:
            print("Warning: Position data or DVS data is empty")
            return
            
        pos_times = [p[0] for p in position_data]
        pos_min, pos_max = min(pos_times), max(pos_times)
        dvs_min, dvs_max = min(dvs_timestamps), max(dvs_timestamps)
        
        # Calculate overlap time range
        overlap_start = max(pos_min, dvs_min)
        overlap_end = min(pos_max, dvs_max)
        
        print(f"Time overlap range: {overlap_start:.2f} to {overlap_end:.2f} ({overlap_end-overlap_start:.2f}s)")
        
        # Filter to overlap range
        dvs_filtered = [(t, dvs_file_map[t]) for t in dvs_timestamps if overlap_start <= t <= overlap_end]
        pos_filtered = [(t, x, y) for t, x, y in position_data if overlap_start <= t <= overlap_end]
        
        print(f"After filtering: {len(dvs_filtered)} DVS files, {len(pos_filtered)} position labels")
        
        # Create position to class mapping
        position_list = []
        position_to_class = {}
        
        for _, x, y in pos_filtered:
            position_key = (round(x / self.position_granularity) * self.position_granularity, 
                          round(y / self.position_granularity) * self.position_granularity)
            if position_key not in position_to_class:
                class_id = len(position_list)
                position_to_class[position_key] = class_id
                position_list.append(position_key)
        
        print(f"Created {len(position_list)} position classes (granularity={self.position_granularity}m)")
        
        # Match DVS files and position labels
        loaded_count = 0
        tolerance = 5.0
        MAX_SAMPLES_PER_DATASET = 3000
        
        for dvs_timestamp, dvs_file in dvs_filtered:
            # Find closest position timestamp
            best_match = None
            min_diff = float('inf')
            
            for pos_timestamp, x, y in pos_filtered:
                diff = abs(dvs_timestamp - pos_timestamp)
                if diff < min_diff:
                    min_diff = diff
                    if diff <= tolerance:
                        best_match = (x, y, min_diff)
            
            if best_match is not None:
                x, y, time_diff = best_match
                position_key = (round(x / self.position_granularity) * self.position_granularity, 
                              round(y / self.position_granularity) * self.position_granularity)
                
                if position_key in position_to_class:
                    try:
                        # Load DVS data
                        dvs_data = np.load(dvs_file)  # Use np.load for .npy files
                        
                        # Ensure data is tensor format
                        if not isinstance(dvs_data, torch.Tensor):
                            if isinstance(dvs_data, np.ndarray):
                                dvs_data = torch.from_numpy(dvs_data)
                            else:
                                continue
                        
                        # Check data shape [2, 3, 260, 346]
                        if len(dvs_data.shape) == 4 and dvs_data.shape[0] == 2 and dvs_data.shape[1] == 3:
                            # Convert to [3, 2, 260, 346] time-first
                            dvs_data = dvs_data.permute(1, 0, 2, 3)  # [3, 2, 260, 346]
                            
                            # Downsample and normalize
                            dvs_data_flat = dvs_data.contiguous().view(-1, 260, 346)
                            dvs_data = nn.functional.avg_pool2d(dvs_data_flat, kernel_size=8, stride=8)
                            dvs_data = dvs_data.view(3, 2, dvs_data.shape[-2], dvs_data.shape[-1])
                            
                            # Data normalization and enhancement
                            dvs_data = dvs_data.float()
                            dvs_data = torch.clamp(dvs_data * 2.0, 0, 1)  # Enhance contrast
                            
                            self.data_samples.append({
                                'dvs': dvs_data,
                                'file_path': str(dvs_file)
                            })
                            self.labels.append(position_to_class[position_key])
                            loaded_count += 1
                            
                            if loaded_count >= MAX_SAMPLES_PER_DATASET:
                                break
                                    
                    except Exception as e:
                        continue
        
        print(f"Successfully loaded {loaded_count} samples")
    
    def __len__(self):
        return len(self.batch_indices)
    
    def __iter__(self):
        if self.is_shuffle:
            random.shuffle(self.batch_indices)
        
        for batch_idx in self.batch_indices:
            batch_dvs = []
            batch_labels = []
            
            for idx in batch_idx:
                dvs_data = self.data_samples[idx]['dvs']
                label = self.labels[idx]
                
                batch_dvs.append(dvs_data)
                batch_labels.append(label)
            
            # Convert to tensors
            dvs_tensor = torch.stack(batch_dvs)  # [batch, 3, 2, 32, 43]
            labels_tensor = torch.tensor(batch_labels, dtype=torch.long)
            
            # Return format: ([aps, gps, dvs], labels) compatible with original format
            dummy_aps = torch.zeros(len(batch_dvs), 3, 64)
            dummy_gps = torch.zeros(len(batch_dvs), 3, 3)
            
            yield ([dummy_aps, dummy_gps, dvs_tensor], labels_tensor)

def load_neurovpr_data():
    """Load NeuroVPR dataset using the fixed dataset loader"""
    
    print("📁 Loading NeuroVPR dataset...")
    
    # Data configuration
    DATA_PATH = '/root/autodl-tmp/neurovpr/datasets/'
    EXP_NAMES = ['floor3_v9', 'room_v5']  # Use actual dataset names
    BATCH_SIZE = 32
    N_CLASS = 25  # Reduced for optimization
    POSITION_GRANULARITY = 5.0
    
    # Create train and test datasets
    train_dataset = FixedNeuroVPRDataset(
        data_path=DATA_PATH,
        exp_names=EXP_NAMES,
        batch_size=BATCH_SIZE,
        is_shuffle=True,
        nclass=N_CLASS,
        split_type='train',
        train_ratio=0.7,
        position_granularity=POSITION_GRANULARITY
    )
    
    test_dataset = FixedNeuroVPRDataset(
        data_path=DATA_PATH,
        exp_names=EXP_NAMES,
        batch_size=BATCH_SIZE,
        is_shuffle=False,
        nclass=N_CLASS,
        split_type='test',
        train_ratio=0.7,
        position_granularity=POSITION_GRANULARITY
    )
    
    print(f"✅ Data loaded successfully:")
    print(f"   Train batches: {len(train_dataset)}")
    print(f"   Test batches: {len(test_dataset)}")
    
    return train_dataset, test_dataset

# ==================== Main Experiment ====================

def run_optimized_neurovpr_experiment():
    """Run the optimized NeuroVPR experiment"""
    
    print("="*80)
    print("OPTIMIZED NEUROVPR EXPERIMENT")
    print("="*80)
    
    # Load data
    train_dataset, test_dataset = load_neurovpr_data()
    
    # Initialize results
    results = {}
    
    # 1. Train Optimized DH-SNN
    print("\n" + "="*60)
    print("1. OPTIMIZED DH-SNN TRAINING")
    print("="*60)
    
    dh_snn_model = OptimizedNeuroVPR_DH_SNN(
        input_dim=2752,
        hidden_dims=[512],
        output_dim=25,
        num_branches=2  # Reduced complexity
    )
    
    dh_snn_results = train_optimized_model(
        dh_snn_model, train_dataset, test_dataset, 
        "Optimized DH-SNN", num_epochs=30
    )
    results['optimized_dh_snn'] = dh_snn_results
    
    # 2. Train Optimized Vanilla SNN for comparison
    print("\n" + "="*60)
    print("2. OPTIMIZED VANILLA SNN TRAINING")
    print("="*60)
    
    vanilla_snn_model = OptimizedNeuroVPR_Vanilla_SNN(
        input_dim=2752,
        hidden_dims=[512],
        output_dim=25
    )
    
    vanilla_snn_results = train_optimized_model(
        vanilla_snn_model, train_dataset, test_dataset,
        "Optimized Vanilla SNN", num_epochs=30
    )
    results['optimized_vanilla_snn'] = vanilla_snn_results
    
    # 3. Results Summary
    print("\n" + "="*80)
    print("OPTIMIZED EXPERIMENT RESULTS")
    print("="*80)
    
    print(f"\n📊 Performance Comparison:")
    print(f"Optimized DH-SNN:")
    print(f"  - Best Test Accuracy: {dh_snn_results['best_test_acc']:.2f}%")
    print(f"  - Final Test Accuracy: {dh_snn_results['final_test_acc']:.2f}%")
    
    print(f"\nOptimized Vanilla SNN:")
    print(f"  - Best Test Accuracy: {vanilla_snn_results['best_test_acc']:.2f}%")
    print(f"  - Final Test Accuracy: {vanilla_snn_results['final_test_acc']:.2f}%")
    
    # Performance improvement analysis
    improvement = dh_snn_results['best_test_acc'] - vanilla_snn_results['best_test_acc']
    print(f"\n🎯 Performance Analysis:")
    print(f"DH-SNN vs Vanilla SNN: {improvement:+.2f}%")
    
    if improvement > 0:
        print("✅ DH-SNN outperforms Vanilla SNN - Expected behavior!")
    else:
        print("❌ DH-SNN still underperforms - Further optimization needed")
    
    # Save results
    results_path = '/root/DH-SNN_reproduce/optimized_neurovpr_results.json'
    with open(results_path, 'w') as f:
        # Convert tensor lists to regular lists for JSON serialization
        json_results = {}
        for model_name, model_results in results.items():
            json_results[model_name] = {
                'best_test_acc': model_results['best_test_acc'],
                'final_test_acc': model_results['final_test_acc'],
                'train_losses': [float(x) for x in model_results['train_losses']],
                'test_accuracies': [float(x) for x in model_results['test_accuracies']]
            }
        json.dump(json_results, f, indent=2)
    
    print(f"\n💾 Results saved to: {results_path}")
    
    return results

if __name__ == "__main__":
    results = run_optimized_neurovpr_experiment()
