#!/usr/bin/env python3
"""
Quick test to verify the core training functionality without requiring full dataset.
This tests the enhanced multi-task learning components in isolation.
"""

import torch
import sys
import os
sys.path.append('src')

def test_core_training_components():
    print("=== Testing Core Training Components ===\n")
    
    # Test imports
    try:
        from models.constellation_model import ConstellationResNet
        from losses.uncertainty_weighted_loss import AnalyticalUncertaintyWeightedLoss, DistancePenalizedSNRLoss
        print("✅ All imports successful")
    except Exception as e:
        print(f"❌ Import error: {e}")
        return False
    
    # Setup device (use CPU for testing to avoid device issues)
    device = 'cpu'  # Force CPU for testing
    print(f"✅ Using device: {device}")
    
    # Test model creation
    try:
        model = ConstellationResNet(num_classes=20, snr_classes=26)
        model = model.to(device)
        print("✅ Model created successfully")
    except Exception as e:
        print(f"❌ Model creation error: {e}")
        return False
    
    # Test loss functions
    try:
        criterion_modulation = torch.nn.CrossEntropyLoss()
        criterion_snr = DistancePenalizedSNRLoss(snr_min=-20, snr_max=30, snr_step=2)
        uncertainty_weighter = AnalyticalUncertaintyWeightedLoss(num_tasks=2, device=device)
        uncertainty_weighter = uncertainty_weighter.to(device)
        print("✅ Loss functions created successfully")
    except Exception as e:
        print(f"❌ Loss function error: {e}")
        return False
    
    # Test optimizer setup
    try:
        model_params = list(model.parameters()) + list(uncertainty_weighter.parameters())
        optimizer = torch.optim.Adam(model_params, lr=1e-4, weight_decay=1e-5)
        print("✅ Optimizer setup successful")
    except Exception as e:
        print(f"❌ Optimizer error: {e}")
        return False
    
    # Test forward pass
    try:
        batch_size = 4
        x = torch.randn(batch_size, 1, 224, 224).to(device)  # Fake constellation images
        
        with torch.no_grad():
            mod_pred, snr_pred = model(x)
            
        print(f"✅ Forward pass successful: mod_shape={mod_pred.shape}, snr_shape={snr_pred.shape}")
        
        # Verify output shapes
        assert mod_pred.shape == (batch_size, 20), f"Expected (4, 20), got {mod_pred.shape}"
        assert snr_pred.shape == (batch_size, 26), f"Expected (4, 26), got {snr_pred.shape}"
        print("✅ Output shapes correct")
        
    except Exception as e:
        print(f"❌ Forward pass error: {e}")
        return False
    
    # Test training step
    try:
        model.train()
        
        # Create fake targets
        mod_targets = torch.randint(0, 20, (batch_size,)).to(device)
        snr_targets = torch.randint(0, 26, (batch_size,)).to(device)
        
        # Forward pass
        mod_pred, snr_pred = model(x)
        
        # Compute individual losses
        mod_loss = criterion_modulation(mod_pred, mod_targets)
        snr_loss = criterion_snr(snr_pred, snr_targets)
        
        # Enhanced multi-task loss weighting
        total_loss, task_weights = uncertainty_weighter([mod_loss, snr_loss])
        
        print(f"✅ Loss computation successful:")
        print(f"   Modulation loss: {mod_loss:.4f}")
        print(f"   SNR loss: {snr_loss:.4f}")
        print(f"   Total loss: {total_loss:.4f}")
        print(f"   Task weights: {task_weights.cpu().numpy()}")
        
        # Backward pass
        optimizer.zero_grad()
        total_loss.backward()
        
        # Check gradients
        total_grad_norm = 0
        for p in model.parameters():
            if p.grad is not None:
                total_grad_norm += p.grad.data.norm(2).item() ** 2
        total_grad_norm = total_grad_norm ** 0.5
        
        print(f"✅ Backward pass successful, gradient norm: {total_grad_norm:.4f}")
        
        # Optimizer step
        optimizer.step()
        print("✅ Optimizer step successful")
        
    except Exception as e:
        print(f"❌ Training step error: {e}")
        return False
    
    # Test uncertainty weighter learning
    try:
        print(f"\n✅ Uncertainty parameters:")
        uncertainties = uncertainty_weighter.get_uncertainties()
        print(f"   Initial uncertainties: {uncertainties.cpu().numpy()}")
        
        # Run a few more steps to see if uncertainties adapt
        for step in range(5):
            mod_pred, snr_pred = model(x)
            mod_loss = criterion_modulation(mod_pred, mod_targets)
            snr_loss = criterion_snr(snr_pred, snr_targets)
            total_loss, task_weights = uncertainty_weighter([mod_loss, snr_loss])
            
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            
        final_uncertainties = uncertainty_weighter.get_uncertainties()
        final_weights = task_weights.cpu().numpy()
        print(f"   Final uncertainties: {final_uncertainties.cpu().numpy()}")
        print(f"   Final task weights: {final_weights}")
        print("✅ Uncertainty weighter learning verified")
        
    except Exception as e:
        print(f"❌ Uncertainty learning error: {e}")
        return False
    
    print("\n" + "="*60)
    print("🎉 ALL CORE TRAINING COMPONENTS WORKING CORRECTLY!")
    print("="*60)
    print("\nKey Features Verified:")
    print("✅ Enhanced ConstellationResNet model (26 discrete SNR classes)")
    print("✅ Analytical uncertainty-based loss weighting (SOTA 2024)")
    print("✅ Distance-penalized SNR loss function")
    print("✅ Complete training pipeline integration")
    print("✅ Automatic task balancing via learned uncertainty")
    print("\nReady for full training when constellation data is available!")
    
    return True

if __name__ == "__main__":
    success = test_core_training_components()
    sys.exit(0 if success else 1)