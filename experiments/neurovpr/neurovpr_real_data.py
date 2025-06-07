#!/usr/bin/env python3
"""
NeuroVPR Real Data Experiment for DH-SNN vs Vanilla SNN Comparison
================================================================

This script uses the real NeuroVPR dataset from Zenodo to compare 
DH-SNN against vanilla SNN performance on neuromorphic visual place 
recognition task using DAVIS 346 event camera data.

Dataset: https://zenodo.org/records/7827108
Paper: Zheng et al. "Dendritic heterogeneity spiking neural networks"

Authors: DH-SNN Reproduction Study
Date: 2025
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import time
import json
from pathlib import Path

# Add the original NeuroVPR code to path
sys.path.append('/root/DH-SNN_reproduce/experiments/neurovpr')
sys.path.append('/root/DH-SNN_reproduce/experiments/neurovpr/SNN_layers')

import torchvision

# Configuration - Force GPU usage (before importing SNN modules)
if torch.cuda.is_available():
    device = torch.device('cuda')
    print(f"✅ GPU available: {torch.cuda.get_device_name(0)}")
    print(f"Using device: {device}")
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    # GPU optimizations
    torch.cuda.empty_cache()  # Clear GPU cache
    torch.backends.cudnn.benchmark = True  # Optimize for consistent input sizes
    torch.backends.cudnn.deterministic = False  # Allow non-deterministic algorithms for speed
else:
    print("❌ GPU not available, falling back to CPU")
    device = torch.device('cpu')

# Import original models and utilities (after device setup)
from tiny_spiking_model import Dense_test_4layer, Dense_test_origin_4layer
from tool_function_bak import Data, setup_seed, accuracy

# Experiment parameters (optimized for fast GPU training)
BATCH_SIZE = 32  # Increased batch size for GPU efficiency
N_CLASS = 100
LEARNING_RATE = 1e-3
NUM_EPOCHS = 30  # Reduced for faster training
NUM_ITER = 50    # Reduced for faster training

# Data paths (updated to use data disk)
DATA_PATH = '/root/autodl-tmp/neurovpr/datasets/'
RESULTS_PATH = '/root/DH-SNN_reproduce/results'

# Experimental setup (updated for available datasets)
# Original code expects data_path + exp_idx format (e.g., '/data/room/room_v5')  
# We have floor3_v9 and room_v5 datasets
# We need to modify the approach to work with our dataset structure
AVAILABLE_DATASETS = ['floor3_v9', 'room_v5']
TRAIN_EXP_IDX = ['room_v5']     # Use room_v5 for training  
TEST_EXP_IDX = ['floor3_v9']    # Use floor3_v9 for testing

# Sequence lengths
SEQ_LEN_APS = 3
SEQ_LEN_DVS = 4
SEQ_LEN_GPS = 3
DVS_EXPAND = 3

def check_data_availability():
    """Check if NeuroVPR dataset is available"""
    print("Checking NeuroVPR dataset availability...")
    
    dataset_path = Path(DATA_PATH)
    if not dataset_path.exists():
        print(f"❌ Dataset path not found: {dataset_path}")
        return False
    
    # Check for available datasets
    found_datasets = []
    all_datasets = TRAIN_EXP_IDX + TEST_EXP_IDX
    
    for dataset_name in all_datasets:
        dataset_path_full = Path(DATA_PATH) / dataset_name
        if dataset_path_full.exists():
            found_datasets.append(dataset_name)
            print(f"✅ Found dataset {dataset_name}")
            
            # Check subdirectories and files
            dvs_frames = dataset_path_full / "dvs_frames"
            dvs_7ms = dataset_path_full / "dvs_7ms_3seq"
            position_file = dataset_path_full / "position.txt"
            
            if dvs_frames.exists():
                frame_count = len(list(dvs_frames.glob("*")))
                print(f"   - DVS frames: {frame_count} files")
            else:
                print(f"   - ⚠️  DVS frames directory missing")
                
            if dvs_7ms.exists():
                dvs_count = len(list(dvs_7ms.glob("*")))
                print(f"   - DVS sequences: {dvs_count} files")
            else:
                print(f"   - ⚠️  DVS 7ms sequences directory missing")
                
            if position_file.exists():
                print(f"   - Position file: ✅")
            else:
                print(f"   - ⚠️  Position file missing")
        else:
            print(f"❌ Dataset {dataset_name} not found")
    
    if len(found_datasets) >= 2:  # Need at least training and test data
        print(f"\n✅ Dataset check passed: {len(found_datasets)} datasets found")
        return True
    else:
        print(f"\n❌ Dataset check failed: Only {len(found_datasets)} datasets found")
        return False

def create_data_loaders():
    """Create data loaders for training and testing"""
    print("Creating data loaders...")
    
    # Normalization (from original code)
    normalize = torchvision.transforms.Normalize(
        mean=[0.3537, 0.3537, 0.3537],
        std=[0.3466, 0.3466, 0.3466]
    )
    
    try:
        train_loader = Data(
            data_path=DATA_PATH, 
            batch_size=BATCH_SIZE, 
            exp_idx=TRAIN_EXP_IDX, 
            is_shuffle=True,
            normalize=normalize, 
            nclass=N_CLASS,
            seq_len_aps=SEQ_LEN_APS, 
            seq_len_dvs=SEQ_LEN_DVS, 
            seq_len_gps=SEQ_LEN_GPS,
            dvs_expand=DVS_EXPAND
        )
        
        test_loader = Data(
            data_path=DATA_PATH, 
            batch_size=BATCH_SIZE, 
            exp_idx=TEST_EXP_IDX, 
            is_shuffle=False,
            normalize=normalize, 
            nclass=N_CLASS,
            seq_len_aps=SEQ_LEN_APS, 
            seq_len_dvs=SEQ_LEN_DVS, 
            seq_len_gps=SEQ_LEN_GPS,
            dvs_expand=DVS_EXPAND
        )
        
        print(f"✅ Data loaders created successfully")
        print(f"   - Training batches: {len(train_loader)}")
        print(f"   - Testing batches: {len(test_loader)}")
        
        return train_loader, test_loader
        
    except Exception as e:
        print(f"❌ Error creating data loaders: {e}")
        return None, None

class DeepSeqSLAM_DH(nn.Module):
    """DH-SNN model wrapper for NeuroVPR"""
    def __init__(self, num_classes=N_CLASS, branch=4):
        super(DeepSeqSLAM_DH, self).__init__()
        self.snn = Dense_test_4layer(branch)
        self.num_classes = num_classes

    def forward(self, inp, epoch=100):
        # Apply masks for structured sparsity
        self.snn.dense_1.apply_mask()
        self.snn.dense_2.apply_mask()
        self.snn.dense_3.apply_mask()
        
        # Get DVS sensory inputs (index 2 in the input tuple)
        dvs_inp = inp[2].to(device)
        output = self.snn(dvs_inp)
        return output

class DeepSeqSLAM_Vanilla(nn.Module):
    """Vanilla SNN model wrapper for NeuroVPR"""
    def __init__(self, num_classes=N_CLASS):
        super(DeepSeqSLAM_Vanilla, self).__init__()
        self.snn = Dense_test_origin_4layer()
        self.num_classes = num_classes

    def forward(self, inp, epoch=100):
        # Get DVS sensory inputs (index 2 in the input tuple)
        dvs_inp = inp[2].to(device)
        output = self.snn(dvs_inp)
        return output

def train_model(model, train_loader, test_loader, model_name, num_epochs=NUM_EPOCHS):
    """Train a model and return results"""
    print(f"\n{'='*50}")
    print(f"Training {model_name} Model")
    print(f"{'='*50}")
    
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    lr_schedule = torch.optim.lr_scheduler.StepLR(optimizer, step_size=25, gamma=0.5)
    criterion = nn.CrossEntropyLoss()
    
    # Training metrics
    train_losses = []
    test_accuracies = []
    best_test_acc1 = 0.0
    best_test_acc5 = 0.0
    
    train_iters = iter(train_loader)
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        train_acc1 = 0.0
        train_acc5 = 0.0
        
        for iter_idx in range(NUM_ITER):
            try:
                inputs, target = next(train_iters)
            except StopIteration:
                train_iters = iter(train_loader)
                inputs, target = next(train_iters)
            
            optimizer.zero_grad()
            outputs = model(inputs, epoch=epoch)
            loss = criterion(outputs, target.to(device))
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.cpu().item()
            
            # Calculate training accuracy
            acc1, acc5, _ = accuracy(outputs.cpu(), target, topk=(1, 5, 10))
            train_acc1 += acc1 / len(outputs)
            train_acc5 += acc5 / len(outputs)
        
        lr_schedule.step()
        
        # Testing phase
        model.eval()
        test_loss = 0.0
        test_acc1 = 0.0
        test_acc5 = 0.0
        test_batches = 0
        
        with torch.no_grad():
            for batch_idx, (inputs, target) in enumerate(test_loader):
                outputs = model(inputs, epoch=epoch)
                loss = criterion(outputs.cpu(), target)
                test_loss += loss.item()
                
                acc1, acc5, _ = accuracy(outputs.cpu(), target, topk=(1, 5, 10))
                test_acc1 += acc1 / len(outputs)
                test_acc5 += acc5 / len(outputs)
                test_batches += 1
        
        # Average metrics
        avg_train_loss = running_loss / NUM_ITER
        avg_train_acc1 = train_acc1 / NUM_ITER
        avg_train_acc5 = train_acc5 / NUM_ITER
        avg_test_loss = test_loss / test_batches
        avg_test_acc1 = test_acc1 / test_batches
        avg_test_acc5 = test_acc5 / test_batches
        
        train_losses.append(avg_train_loss)
        test_accuracies.append(avg_test_acc1)
        
        # Update best accuracies
        if avg_test_acc1 > best_test_acc1:
            best_test_acc1 = avg_test_acc1
        if avg_test_acc5 > best_test_acc5:
            best_test_acc5 = avg_test_acc5
        
        # Print progress
        if epoch % 10 == 0 or epoch == num_epochs - 1:
            print(f'Epoch [{epoch+1}/{num_epochs}]')
            print(f'  Train Loss: {avg_train_loss:.4f}, Train Acc1: {avg_train_acc1:.2f}%')
            print(f'  Test Loss: {avg_test_loss:.4f}, Test Acc1: {avg_test_acc1:.2f}%, Test Acc5: {avg_test_acc5:.2f}%')
            print(f'  Best Test Acc1: {best_test_acc1:.2f}%, Best Test Acc5: {best_test_acc5:.2f}%')
    
    return {
        'train_losses': train_losses,
        'test_accuracies': test_accuracies,
        'best_test_acc1': best_test_acc1,
        'best_test_acc5': best_test_acc5,
        'final_test_acc1': avg_test_acc1,
        'final_test_acc5': avg_test_acc5
    }

def main():
    """Main experimental function"""
    print("="*70)
    print("NeuroVPR Real Data Experiment: DH-SNN vs Vanilla SNN")
    print("="*70)
    
    # Setup
    setup_seed(42)
    os.makedirs(RESULTS_PATH, exist_ok=True)
    
    # Check dataset availability
    if not check_data_availability():
        print("\n❌ NeuroVPR dataset not found!")
        print("Please download the dataset from: https://zenodo.org/records/7827108")
        print(f"And place it in: {DATA_PATH.rstrip('room_v')}")
        return None
    
    # Create data loaders
    train_loader, test_loader = create_data_loaders()
    if train_loader is None or test_loader is None:
        print("❌ Failed to create data loaders. Please check your dataset structure.")
        return None
    
    # Initialize models and move to GPU
    print(f"\nInitializing models on {device}...")
    dh_snn_model = DeepSeqSLAM_DH(num_classes=N_CLASS, branch=4).to(device)
    vanilla_snn_model = DeepSeqSLAM_Vanilla(num_classes=N_CLASS).to(device)
    
    print(f"DH-SNN parameters: {sum(p.numel() for p in dh_snn_model.parameters()):,}")
    print(f"Vanilla SNN parameters: {sum(p.numel() for p in vanilla_snn_model.parameters()):,}")
    
    # Verify models are on GPU
    if torch.cuda.is_available():
        print(f"✅ DH-SNN model device: {next(dh_snn_model.parameters()).device}")
        print(f"✅ Vanilla SNN model device: {next(vanilla_snn_model.parameters()).device}")
    
    # Train models with reduced epochs for faster results
    print(f"\nStarting training experiments...")
    
    # Train DH-SNN (reduced epochs)
    dh_results = train_model(dh_snn_model, train_loader, test_loader, "DH-SNN", num_epochs=NUM_EPOCHS)
    
    # Train Vanilla SNN (reduced epochs)
    vanilla_results = train_model(vanilla_snn_model, train_loader, test_loader, "Vanilla SNN", num_epochs=NUM_EPOCHS)
    
    # Results comparison
    print("\n" + "="*70)
    print("FINAL RESULTS COMPARISON")
    print("="*70)
    
    print(f"DH-SNN Results:")
    print(f"  Best Test Accuracy: {dh_results['best_test_acc1']:.2f}%")
    print(f"  Final Test Accuracy: {dh_results['final_test_acc1']:.2f}%")
    print(f"  Best Top-5 Accuracy: {dh_results['best_test_acc5']:.2f}%")
    
    print(f"\nVanilla SNN Results:")
    print(f"  Best Test Accuracy: {vanilla_results['best_test_acc1']:.2f}%")
    print(f"  Final Test Accuracy: {vanilla_results['final_test_acc1']:.2f}%")
    print(f"  Best Top-5 Accuracy: {vanilla_results['best_test_acc5']:.2f}%")
    
    # Calculate improvements
    best_improvement = dh_results['best_test_acc1'] - vanilla_results['best_test_acc1']
    final_improvement = dh_results['final_test_acc1'] - vanilla_results['final_test_acc1']
    best_relative = (dh_results['best_test_acc1'] / vanilla_results['best_test_acc1'] - 1) * 100
    final_relative = (dh_results['final_test_acc1'] / vanilla_results['final_test_acc1'] - 1) * 100
    
    print(f"\nPerformance Improvements:")
    print(f"  Best Accuracy: +{best_improvement:.2f}% (relative: +{best_relative:.1f}%)")
    print(f"  Final Accuracy: +{final_improvement:.2f}% (relative: +{final_relative:.1f}%)")
    
    # Save comprehensive results
    all_results = {
        'experiment_info': {
            'dataset': 'NeuroVPR (Real Data)',
            'dataset_source': 'https://zenodo.org/records/7827108',
            'train_experiments': TRAIN_EXP_IDX,
            'test_experiments': TEST_EXP_IDX,
            'num_classes': N_CLASS,
            'batch_size': BATCH_SIZE,
            'num_epochs': NUM_EPOCHS,
            'num_iter_per_epoch': NUM_ITER,
            'device': str(device),
            'gpu_name': torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'
        },
        'dh_snn': dh_results,
        'vanilla_snn': vanilla_results,
        'comparison': {
            'best_accuracy_improvement': {
                'absolute': best_improvement,
                'relative_percent': best_relative
            },
            'final_accuracy_improvement': {
                'absolute': final_improvement,
                'relative_percent': final_relative
            }
        }
    }
    
    # Save results
    results_file = os.path.join(RESULTS_PATH, 'neurovpr_real_data_results.json')
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    # Save models
    torch.save(dh_snn_model.state_dict(), os.path.join(RESULTS_PATH, 'neurovpr_dh_snn_real.pth'))
    torch.save(vanilla_snn_model.state_dict(), os.path.join(RESULTS_PATH, 'neurovpr_vanilla_snn_real.pth'))
    
    print(f"\n✅ Results saved to: {results_file}")
    print(f"✅ Models saved to: {RESULTS_PATH}")
    
    return all_results

if __name__ == "__main__":
    results = main()
