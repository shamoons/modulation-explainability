#!/usr/bin/env python3
"""
Test training with actual split HDF5 data from data/split_hdf5/
This creates a simple loader for the split HDF5 format and runs a brief training test.
"""

import os
import sys
import torch
import h5py
import json
import numpy as np
from torch.utils.data import Dataset, DataLoader
sys.path.append('src')

class SplitHDF5Dataset(Dataset):
    """PyTorch Dataset for split HDF5 data format"""
    
    def __init__(self, split_hdf5_dir, modulations=None, snr_values=None, limit_per_file=None):
        """
        Args:
            split_hdf5_dir: Path to data/split_hdf5/
            modulations: List of modulation types to include
            snr_values: List of SNR values to include  
            limit_per_file: Max samples per HDF5 file (None = all)
        """
        self.data = []
        self.labels = []
        self.snr_labels = []
        
        # Load modulation mapping
        json_file = 'data/RML2018.01A/classes-fixed.json'
        with open(json_file, 'r') as f:
            all_modulations = json.load(f)
        
        self.mod2int = {mod: i for i, mod in enumerate(all_modulations)}
        
        # Use all modulations if not specified
        if modulations is None:
            modulations = all_modulations
        
        # Default SNR range
        if snr_values is None:
            snr_values = list(range(-20, 32, 2))  # -20 to 30 in 2dB steps
        
        print(f"Loading from {split_hdf5_dir}...")
        print(f"Modulations: {modulations}")
        print(f"SNR values: {snr_values}")
        
        total_samples = 0
        
        for mod in modulations:
            if mod not in self.mod2int:
                print(f"Skipping unknown modulation: {mod}")
                continue
                
            mod_dir = os.path.join(split_hdf5_dir, mod)
            if not os.path.exists(mod_dir):
                print(f"Skipping missing modulation directory: {mod}")
                continue
            
            mod_label = self.mod2int[mod]
            
            for snr in snr_values:
                snr_dir = os.path.join(mod_dir, f"SNR_{snr}")
                hdf5_file = os.path.join(snr_dir, f"{mod}_SNR_{snr}.h5")
                
                if not os.path.exists(hdf5_file):
                    continue
                
                # Load data from this HDF5 file
                with h5py.File(hdf5_file, 'r') as f:
                    X_data = f['X'][:]  # I/Q components
                    Y_data = f['Y'][:]  # Not used (we use mod_label)
                    Z_data = f['Z'][:]  # Not used (we use snr)
                    
                    # Apply limit if specified
                    if limit_per_file:
                        X_data = X_data[:limit_per_file]
                    
                    # Convert SNR to class index (for discrete SNR prediction)
                    snr_class = (snr + 20) // 2  # Maps -20->0, -18->1, ..., 30->25
                    
                    # Add to dataset
                    for x_sample in X_data:
                        self.data.append(x_sample)
                        self.labels.append(mod_label)
                        self.snr_labels.append(snr_class)
                        total_samples += 1
                
                print(f"  Loaded {len(X_data)} samples: {mod} @ SNR {snr}")
        
        # Convert to tensors
        self.data = torch.tensor(np.array(self.data), dtype=torch.float32)
        self.labels = torch.tensor(self.labels, dtype=torch.long)
        self.snr_labels = torch.tensor(self.snr_labels, dtype=torch.long)
        
        print(f"\nDataset created: {total_samples} total samples")
        print(f"Data shape: {self.data.shape}")
        print(f"Unique modulations: {len(set(self.labels.numpy()))}")
        print(f"Unique SNRs: {len(set(self.snr_labels.numpy()))}")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        # Note: I/Q data needs to be reshaped for constellation conversion
        # Original shape: (1024, 2) -> We'll treat this as raw I/Q for now
        return self.data[idx], self.labels[idx], self.snr_labels[idx]


def test_hdf5_training():
    print("=== Testing HDF5 Training ===\n")
    
    # Import training components
    from models.constellation_model import ConstellationResNet
    from losses.uncertainty_weighted_loss import AnalyticalUncertaintyWeightedLoss, DistancePenalizedSNRLoss
    
    # Use CPU for testing
    device = 'cpu'
    print(f"Using device: {device}")
    
    # Create dataset with a few modulations and SNRs for quick testing
    test_modulations = ['BPSK', 'QPSK', '8PSK']  # Just 3 modulations
    test_snrs = [0, 10, 20]  # Just 3 SNR values
    
    dataset = SplitHDF5Dataset(
        split_hdf5_dir='data/split_hdf5',
        modulations=test_modulations,
        snr_values=test_snrs,
        limit_per_file=32  # Only 32 samples per file for quick testing
    )
    
    if len(dataset) == 0:
        print("❌ No data loaded! Check if data/split_hdf5 exists and contains data.")
        return False
    
    # Create dataloader
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
    print(f"✅ Created dataloader with {len(dataset)} samples")
    
    # Create model - NOTE: This model expects constellation images, not raw I/Q
    # For now we'll test with the raw I/Q data to see the pipeline
    model = ConstellationResNet(num_classes=20, snr_classes=26)
    model = model.to(device)
    print("✅ Model created")
    
    # Create loss functions
    criterion_modulation = torch.nn.CrossEntropyLoss()
    criterion_snr = DistancePenalizedSNRLoss(snr_min=-20, snr_max=30, snr_step=2)
    uncertainty_weighter = AnalyticalUncertaintyWeightedLoss(num_tasks=2, device=device)
    uncertainty_weighter = uncertainty_weighter.to(device)
    print("✅ Loss functions created")
    
    # Create optimizer
    model_params = list(model.parameters()) + list(uncertainty_weighter.parameters())
    optimizer = torch.optim.Adam(model_params, lr=1e-4, weight_decay=1e-5)
    print("✅ Optimizer created")
    
    # Test a few training steps
    model.train()
    print("\n🔥 Starting training test...")
    
    for batch_idx, (data, mod_targets, snr_targets) in enumerate(dataloader):
        if batch_idx >= 3:  # Only test 3 batches
            break
            
        data = data.to(device)
        mod_targets = mod_targets.to(device) 
        snr_targets = snr_targets.to(device)
        
        print(f"\nBatch {batch_idx + 1}:")
        print(f"  Data shape: {data.shape}")
        print(f"  Modulation targets: {mod_targets.numpy()}")
        print(f"  SNR targets: {snr_targets.numpy()}")
        
        # NOTE: The model expects (batch, 1, 224, 224) constellation images
        # But we have (batch, 1024, 2) I/Q data
        # For testing, let's reshape the data to see what happens
        try:
            # Try to make it work temporarily by reshaping
            # This is NOT correct - just for pipeline testing
            batch_size = data.shape[0]
            # Flatten I/Q and pad/crop to match expected input
            data_flat = data.view(batch_size, -1)  # (batch, 2048)
            
            # Create dummy constellation-like data
            dummy_constellation = torch.zeros(batch_size, 1, 224, 224).to(device)
            # Fill with some I/Q derived values (very crude)
            for i in range(min(224, data_flat.shape[1])):
                dummy_constellation[:, 0, i % 224, i // 224] = data_flat[:, i]
            
            # Forward pass with dummy constellation data
            mod_pred, snr_pred = model(dummy_constellation)
            
            print(f"  ✅ Forward pass successful")
            print(f"     Mod prediction shape: {mod_pred.shape}")
            print(f"     SNR prediction shape: {snr_pred.shape}")
            
            # Compute losses
            mod_loss = criterion_modulation(mod_pred, mod_targets)
            snr_loss = criterion_snr(snr_pred, snr_targets)
            total_loss, task_weights = uncertainty_weighter([mod_loss, snr_loss])
            
            print(f"  📊 Losses:")
            print(f"     Modulation loss: {mod_loss:.4f}")
            print(f"     SNR loss: {snr_loss:.4f}")
            print(f"     Total loss: {total_loss:.4f}")
            print(f"     Task weights: {task_weights.cpu().numpy()}")
            
            # Backward pass
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            
            print(f"  ✅ Training step completed")
            
        except Exception as e:
            print(f"  ❌ Error in batch {batch_idx + 1}: {e}")
            return False
    
    print("\n" + "="*60)
    print("🎉 HDF5 TRAINING TEST COMPLETED!")
    print("="*60)
    print("✅ Successfully loaded split HDF5 data")
    print("✅ Created proper PyTorch dataset and dataloader")
    print("✅ Ran training pipeline with uncertainty weighting")
    print("✅ All components working together")
    print()
    print("⚠️  NOTE: This test used dummy constellation conversion.")
    print("   For real training, you need to:")
    print("   1. Convert I/Q data to constellation images using convert_to_constellation.py")
    print("   2. Or modify the model to work directly with I/Q data")
    print()
    print("The data pipeline and enhanced multi-task learning are ready!")
    
    return True


if __name__ == "__main__":
    success = test_hdf5_training()
    sys.exit(0 if success else 1)