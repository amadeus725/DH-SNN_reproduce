#!/usr/bin/env python3
"""
NeuroVPR Simulation for DH-SNN vs Vanilla SNN Comparison
========================================================

This script simulates the NeuroVPR (Neuromorphic Visual Place Recognition) task
to compare DH-SNN against vanilla SNN performance. Since the original DAVIS 346
event camera dataset is not readily available, we create synthetic neuromorphic
data that mimics the characteristics of event-based vision for robot navigation.

Key aspects simulated:
1. Event-based vision sequences (DVS-like data)
2. Temporal dynamics at different timescales
3. Place recognition task (100 locations)
4. Robot navigation context with sequential data

Authors: DH-SNN Reproduction Study
Date: 2025
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import time
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

# Set device and random seeds
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

def setup_seed(seed=42):
    """Setup random seeds for reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

setup_seed(42)

# Import DH-SNN layers (simplified versions)
class LIFNeuron(nn.Module):
    """Leaky Integrate-and-Fire Neuron"""
    def __init__(self, tau=0.25, vth=1.0, dt=1.0):
        super().__init__()
        self.tau = tau
        self.vth = vth
        self.dt = dt
        self.mem = None
        
    def forward(self, x):
        if self.mem is None:
            self.mem = torch.zeros_like(x)
            
        # LIF dynamics
        self.mem = self.mem + (self.dt / self.tau) * (-self.mem + x)
        
        # Spike generation
        spike = (self.mem >= self.vth).float()
        self.mem = self.mem * (1 - spike)  # Reset after spike
        
        return spike, self.mem
        
    def reset_state(self):
        self.mem = None

class DHSNNLayer(nn.Module):
    """Simplified DH-SNN Layer with multiple dendritic branches"""
    def __init__(self, input_dim, output_dim, num_branches=4, tau_range=(0.1, 2.0)):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_branches = num_branches
        
        # Multiple dendritic branches with different time constants
        self.branches = nn.ModuleList()
        self.branch_weights = nn.ModuleList()
        
        for i in range(num_branches):
            # Different tau for each branch
            tau = tau_range[0] + (tau_range[1] - tau_range[0]) * i / (num_branches - 1)
            branch = LIFNeuron(tau=tau)
            weight = nn.Linear(input_dim, output_dim)
            
            self.branches.append(branch)
            self.branch_weights.append(weight)
            
        # Integration weights for combining branches
        self.integration_weights = nn.Parameter(torch.ones(num_branches) / num_branches)
        
    def forward(self, x):
        branch_outputs = []
        
        for i, (branch, weight) in enumerate(zip(self.branches, self.branch_weights)):
            # Apply linear transformation
            weighted_input = weight(x)
            # Apply LIF dynamics
            spike, mem = branch(weighted_input)
            branch_outputs.append(spike)
            
        # Combine branches with learned weights
        combined_output = torch.zeros_like(branch_outputs[0])
        for i, output in enumerate(branch_outputs):
            combined_output += self.integration_weights[i] * output
            
        return combined_output
        
    def reset_state(self):
        for branch in self.branches:
            branch.reset_state()

class VanillaSNNLayer(nn.Module):
    """Standard SNN Layer with single time constant"""
    def __init__(self, input_dim, output_dim, tau=0.25):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        self.lif = LIFNeuron(tau=tau)
        
    def forward(self, x):
        x = self.linear(x)
        spike, mem = self.lif(x)
        return spike
        
    def reset_state(self):
        self.lif.reset_state()

class DHSNNModel(nn.Module):
    """DH-SNN Model for NeuroVPR task"""
    def __init__(self, input_dim=32*43*2, hidden_dim=512, num_classes=100, num_branches=4):
        super().__init__()
        
        self.layer1 = DHSNNLayer(input_dim, hidden_dim, num_branches)
        self.layer2 = DHSNNLayer(hidden_dim, hidden_dim, num_branches)
        self.layer3 = DHSNNLayer(hidden_dim, 256, num_branches)
        self.classifier = nn.Linear(256, num_classes)
        
    def forward(self, x):
        # x shape: (batch, seq_len, channels, height, width)
        batch_size, seq_len = x.shape[:2]
        
        # Reset neuron states
        self.layer1.reset_state()
        self.layer2.reset_state()
        self.layer3.reset_state()
        
        outputs = []
        for t in range(seq_len):
            # Flatten spatial dimensions
            x_t = x[:, t].reshape(batch_size, -1)
            
            # Forward through DH-SNN layers
            h1 = self.layer1(x_t)
            h2 = self.layer2(h1)
            h3 = self.layer3(h2)
            
            # Classification
            out = self.classifier(h3)
            outputs.append(out)
            
        # Average over time
        output = torch.stack(outputs, dim=1).mean(dim=1)
        return output

class VanillaSNNModel(nn.Module):
    """Vanilla SNN Model for comparison"""
    def __init__(self, input_dim=32*43*2, hidden_dim=512, num_classes=100):
        super().__init__()
        
        self.layer1 = VanillaSNNLayer(input_dim, hidden_dim)
        self.layer2 = VanillaSNNLayer(hidden_dim, hidden_dim)
        self.layer3 = VanillaSNNLayer(hidden_dim, 256)
        self.classifier = nn.Linear(256, num_classes)
        
    def forward(self, x):
        # x shape: (batch, seq_len, channels, height, width)
        batch_size, seq_len = x.shape[:2]
        
        # Reset neuron states
        self.layer1.reset_state()
        self.layer2.reset_state()
        self.layer3.reset_state()
        
        outputs = []
        for t in range(seq_len):
            # Flatten spatial dimensions
            x_t = x[:, t].reshape(batch_size, -1)
            
            # Forward through vanilla SNN layers
            h1 = self.layer1(x_t)
            h2 = self.layer2(h1)
            h3 = self.layer3(h2)
            
            # Classification
            out = self.classifier(h3)
            outputs.append(out)
            
        # Average over time
        output = torch.stack(outputs, dim=1).mean(dim=1)
        return output

class SyntheticNeuroVPRDataset(Dataset):
    """
    Synthetic dataset simulating neuromorphic visual place recognition data
    Mimics DAVIS 346 event camera data for robot navigation
    """
    def __init__(self, num_samples=5000, num_classes=100, seq_len=12, 
                 height=32, width=43, channels=2):
        self.num_samples = num_samples
        self.num_classes = num_classes
        self.seq_len = seq_len
        self.height = height
        self.width = width
        self.channels = channels
        
        # Generate synthetic data
        self.data = []
        self.labels = []
        
        for i in range(num_samples):
            # Create synthetic event-based sequence
            # Simulate different temporal patterns for different places
            label = i % num_classes
            
            # Generate base pattern for this location
            base_pattern = self._generate_place_pattern(label)
            
            # Create temporal sequence with variations
            sequence = self._generate_temporal_sequence(base_pattern, label)
            
            self.data.append(sequence)
            self.labels.append(label)
            
    def _generate_place_pattern(self, place_id):
        """Generate a characteristic pattern for a specific place"""
        np.random.seed(place_id)  # Consistent pattern for each place
        
        # Create structured patterns that represent different locations
        pattern = np.random.rand(self.height, self.width, self.channels)
        
        # Add some structure based on place_id
        center_x, center_y = place_id % self.width, (place_id // self.width) % self.height
        
        # Create radial pattern around center
        y, x = np.ogrid[:self.height, :self.width]
        dist = np.sqrt((x - center_x)**2 + (y - center_y)**2)
        mask = dist <= 5
        
        pattern[mask, 0] *= 2  # Enhance ON events
        pattern[mask, 1] *= 0.5  # Reduce OFF events
        
        return pattern
        
    def _generate_temporal_sequence(self, base_pattern, place_id):
        """Generate temporal sequence with multi-timescale dynamics"""
        sequence = []
        
        for t in range(self.seq_len):
            # Add temporal variations
            time_factor = np.sin(2 * np.pi * t / self.seq_len + place_id * 0.1)
            
            # Fast dynamics (high frequency)
            fast_noise = 0.1 * np.random.randn(*base_pattern.shape)
            
            # Slow dynamics (low frequency)
            slow_modulation = 0.3 * time_factor
            
            # Medium dynamics
            medium_noise = 0.2 * np.sin(2 * np.pi * t / 4 + place_id * 0.2)
            
            # Combine all temporal scales
            frame = base_pattern + fast_noise + slow_modulation + medium_noise
            frame = np.clip(frame, 0, 1)
            
            sequence.append(frame)
            
        return np.array(sequence)
        
    def __len__(self):
        return self.num_samples
        
    def __getitem__(self, idx):
        data = torch.tensor(self.data[idx], dtype=torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        
        # Reshape to (seq_len, channels, height, width)
        data = data.permute(0, 3, 1, 2)
        
        return data, label

def evaluate_model(model, dataloader, device):
    """Evaluate model performance"""
    model.eval()
    correct = 0
    total = 0
    top5_correct = 0
    
    with torch.no_grad():
        for data, labels in dataloader:
            data, labels = data.to(device), labels.to(device)
            outputs = model(data)
            
            # Top-1 accuracy
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Top-5 accuracy
            _, top5_pred = torch.topk(outputs, 5, dim=1)
            top5_correct += sum([labels[i] in top5_pred[i] for i in range(len(labels))])
    
    top1_acc = 100 * correct / total
    top5_acc = 100 * top5_correct / total
    
    return top1_acc, top5_acc

def train_model(model, train_loader, test_loader, device, num_epochs=50, lr=1e-3):
    """Train the model"""
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.5)
    
    train_losses = []
    test_accuracies = []
    
    print(f"Training {model.__class__.__name__}...")
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        
        for batch_idx, (data, labels) in enumerate(train_loader):
            data, labels = data.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
        scheduler.step()
        
        # Evaluate
        top1_acc, top5_acc = evaluate_model(model, test_loader, device)
        
        avg_loss = running_loss / len(train_loader)
        train_losses.append(avg_loss)
        test_accuracies.append(top1_acc)
        
        if epoch % 10 == 0 or epoch == num_epochs - 1:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}, '
                  f'Test Top-1 Acc: {top1_acc:.2f}%, Test Top-5 Acc: {top5_acc:.2f}%')
    
    return train_losses, test_accuracies

def main():
    """Main experimental function"""
    print("=" * 60)
    print("NeuroVPR Simulation: DH-SNN vs Vanilla SNN Comparison")
    print("=" * 60)
    
    # Create datasets
    print("Creating synthetic NeuroVPR dataset...")
    train_dataset = SyntheticNeuroVPRDataset(num_samples=4000, num_classes=100)
    test_dataset = SyntheticNeuroVPRDataset(num_samples=1000, num_classes=100)
    
    train_loader = DataLoader(train_dataset, batch_size=20, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=20, shuffle=False, num_workers=4)
    
    print(f"Dataset created: {len(train_dataset)} training, {len(test_dataset)} testing samples")
    
    # Initialize models
    dh_snn_model = DHSNNModel(num_branches=4).to(device)
    vanilla_snn_model = VanillaSNNModel().to(device)
    
    print(f"DH-SNN parameters: {sum(p.numel() for p in dh_snn_model.parameters()):,}")
    print(f"Vanilla SNN parameters: {sum(p.numel() for p in vanilla_snn_model.parameters()):,}")
    
    # Train DH-SNN
    print("\n" + "="*40)
    print("Training DH-SNN Model")
    print("="*40)
    dh_losses, dh_accuracies = train_model(dh_snn_model, train_loader, test_loader, device, num_epochs=30)
    
    # Train Vanilla SNN
    print("\n" + "="*40)
    print("Training Vanilla SNN Model")
    print("="*40)
    vanilla_losses, vanilla_accuracies = train_model(vanilla_snn_model, train_loader, test_loader, device, num_epochs=30)
    
    # Final evaluation
    print("\n" + "="*40)
    print("Final Evaluation Results")
    print("="*40)
    
    dh_top1, dh_top5 = evaluate_model(dh_snn_model, test_loader, device)
    vanilla_top1, vanilla_top5 = evaluate_model(vanilla_snn_model, test_loader, device)
    
    print(f"DH-SNN Final Results:")
    print(f"  Top-1 Accuracy: {dh_top1:.2f}%")
    print(f"  Top-5 Accuracy: {dh_top5:.2f}%")
    
    print(f"\nVanilla SNN Final Results:")
    print(f"  Top-1 Accuracy: {vanilla_top1:.2f}%")
    print(f"  Top-5 Accuracy: {vanilla_top5:.2f}%")
    
    print(f"\nPerformance Improvement:")
    print(f"  Top-1 Accuracy: +{dh_top1 - vanilla_top1:.2f}% ({(dh_top1/vanilla_top1-1)*100:.1f}% relative)")
    print(f"  Top-5 Accuracy: +{dh_top5 - vanilla_top5:.2f}% ({(dh_top5/vanilla_top5-1)*100:.1f}% relative)")
    
    # Save results
    results = {
        'dh_snn': {
            'top1_accuracy': dh_top1,
            'top5_accuracy': dh_top5,
            'training_losses': dh_losses,
            'test_accuracies': dh_accuracies
        },
        'vanilla_snn': {
            'top1_accuracy': vanilla_top1,
            'top5_accuracy': vanilla_top5,
            'training_losses': vanilla_losses,
            'test_accuracies': vanilla_accuracies
        },
        'improvement': {
            'top1_absolute': dh_top1 - vanilla_top1,
            'top1_relative': (dh_top1/vanilla_top1-1)*100,
            'top5_absolute': dh_top5 - vanilla_top5,
            'top5_relative': (dh_top5/vanilla_top5-1)*100
        }
    }
    
    # Save models and results
    torch.save(dh_snn_model.state_dict(), '/root/DH-SNN_reproduce/results/neurovpr_dh_snn_model.pth')
    torch.save(vanilla_snn_model.state_dict(), '/root/DH-SNN_reproduce/results/neurovpr_vanilla_snn_model.pth')
    
    import json
    with open('/root/DH-SNN_reproduce/results/neurovpr_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to /root/DH-SNN_reproduce/results/neurovpr_results.json")
    
    # Create visualization
    create_performance_plot(dh_accuracies, vanilla_accuracies)
    
    return results

def create_performance_plot(dh_accuracies, vanilla_accuracies):
    """Create performance comparison plot"""
    plt.figure(figsize=(12, 8))
    
    epochs = range(1, len(dh_accuracies) + 1)
    
    plt.subplot(2, 1, 1)
    plt.plot(epochs, dh_accuracies, 'b-', label='DH-SNN', linewidth=2)
    plt.plot(epochs, vanilla_accuracies, 'r--', label='Vanilla SNN', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Test Accuracy (%)')
    plt.title('NeuroVPR: DH-SNN vs Vanilla SNN Performance')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Performance improvement over time
    plt.subplot(2, 1, 2)
    improvements = [dh - vanilla for dh, vanilla in zip(dh_accuracies, vanilla_accuracies)]
    plt.plot(epochs, improvements, 'g-', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy Improvement (%)')
    plt.title('DH-SNN Performance Advantage')
    plt.grid(True, alpha=0.3)
    plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('/root/DH-SNN_reproduce/results/neurovpr_performance_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("Performance plot saved to /root/DH-SNN_reproduce/results/neurovpr_performance_comparison.png")

if __name__ == "__main__":
    main()
