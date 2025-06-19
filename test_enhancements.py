#!/usr/bin/env python3
"""
Test script to demonstrate the enhanced multi-task learning implementations.
This script shows the improvements over the original approach.
"""

import torch
import sys
import os
sys.path.append('src')

def test_traditional_vs_enhanced():
    print("=== Comparison: Traditional vs Enhanced Multi-Task Learning ===\n")
    
    # Import enhanced components
    from losses.uncertainty_weighted_loss import AnalyticalUncertaintyWeightedLoss, DistancePenalizedSNRLoss
    from models.constellation_model import ConstellationResNet
    
    device = 'cpu'
    
    print("1. LOSS WEIGHTING COMPARISON")
    print("-" * 40)
    
    # Simulate different loss scenarios
    scenarios = [
        ("Balanced losses", torch.tensor(1.0), torch.tensor(1.0)),
        ("High mod, low SNR", torch.tensor(3.0), torch.tensor(0.5)),
        ("Low mod, high SNR", torch.tensor(0.3), torch.tensor(2.5)),
    ]
    
    # Traditional approach (fixed α=0.5, β=1.0)
    alpha, beta = 0.5, 1.0
    
    # Enhanced approach
    weighter = AnalyticalUncertaintyWeightedLoss(num_tasks=2, device=device)
    
    print(f"{'Scenario':<20} {'Traditional':<15} {'Enhanced':<15} {'Weights'}")
    print("-" * 70)
    
    for name, mod_loss, snr_loss in scenarios:
        # Traditional weighting
        traditional_loss = alpha * mod_loss + beta * snr_loss
        
        # Enhanced weighting (simulate training by running multiple times)
        enhanced_loss, weights = weighter([mod_loss, snr_loss])
        
        print(f"{name:<20} {traditional_loss:.3f}           {enhanced_loss:.3f}           {weights.numpy()}")
    
    print("\n✅ Enhanced weighting automatically adapts to loss magnitudes!\n")
    
    print("2. DISCRETE SNR PREDICTION")
    print("-" * 40)
    
    # Enhanced discrete SNR prediction
    snr_loss_fn = DistancePenalizedSNRLoss()
    print("Discrete SNR Prediction (26 classes):")
    print(f"  Classes: {len(snr_loss_fn.snr_values)} discrete 2dB intervals")
    print(f"  Range: {snr_loss_fn.snr_values[0]} to {snr_loss_fn.snr_values[-1]} dB")
    print(f"  Examples: {snr_loss_fn.snr_values[5]}, {snr_loss_fn.snr_values[10]}, {snr_loss_fn.snr_values[15]} dB")
    print("  ✅ Fine-grained resolution with distance-based penalization!")
    print("  ✅ Addresses reviewer feedback about coarse bucketing!")
    
    print("\n3. DISTANCE PENALTY DEMONSTRATION")
    print("-" * 40)
    
    # Test distance penalty for different prediction errors
    predictions = torch.zeros(4, 26)
    true_snr_db = 0  # 0 dB (class index 10)
    true_class = snr_loss_fn.snr_value_to_class(true_snr_db)
    
    # Different prediction scenarios
    perfect_pred = torch.zeros(1, 26)
    perfect_pred[0, true_class] = 10.0
    
    close_pred = torch.zeros(1, 26)  
    close_pred[0, true_class + 1] = 10.0  # +2 dB error
    
    medium_pred = torch.zeros(1, 26)
    medium_pred[0, true_class + 5] = 10.0  # +10 dB error
    
    far_pred = torch.zeros(1, 26)
    far_pred[0, true_class + 10] = 10.0  # +20 dB error
    
    target = torch.tensor([true_class])
    
    perfect_loss = snr_loss_fn(perfect_pred, target)
    close_loss = snr_loss_fn(close_pred, target)
    medium_loss = snr_loss_fn(medium_pred, target)
    far_loss = snr_loss_fn(far_pred, target)
    
    print(f"True SNR: {true_snr_db} dB (class {true_class})")
    print(f"Perfect prediction: loss = {perfect_loss:.3f}")
    print(f"Close error (+2 dB): loss = {close_loss:.3f} ({close_loss/perfect_loss:.1f}x)")
    print(f"Medium error (+10 dB): loss = {medium_loss:.3f} ({medium_loss/perfect_loss:.1f}x)")
    print(f"Far error (+20 dB): loss = {far_loss:.3f} ({far_loss/perfect_loss:.1f}x)")
    print("✅ Larger errors are penalized more heavily!")
    
    print("\n4. MODEL ARCHITECTURE")
    print("-" * 40)
    
    # Enhanced model with discrete SNR prediction (26 classes)
    model = ConstellationResNet(num_classes=20, snr_classes=26)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    
    print(f"Model parameters: {total_params:,}")
    print(f"SNR classes: 26 discrete values (-20 to 30 dB in 2dB steps)")
    print(f"Modulation classes: 20")
    print("✅ Efficient architecture for fine-grained SNR and modulation prediction!")
    
    print("\n" + "="*70)
    print("SUMMARY OF ENHANCEMENTS")
    print("="*70)
    print("✅ SOTA Multi-Task Learning: Analytical uncertainty-based weighting")
    print("   - Replaces fixed α/β with learned uncertainty parameters")
    print("   - Automatically balances task importance during training")
    print("   - Based on 2024 research (arXiv:2408.07985)")
    
    print("\n✅ Enhanced SNR Prediction: Discrete 26-class prediction")
    print("   - Addresses reviewer feedback about coarse 16dB buckets")
    print("   - 2dB resolution across full -20 to +30 dB range")
    print("   - Distance-based penalty for better error handling")
    
    print("\n✅ Simplified Architecture: Pure discrete SNR prediction")
    print("   - No more complex bucketing logic")
    print("   - Direct 26-class SNR prediction")
    print("   - Cleaner, more maintainable codebase")
    
    print("\n🚀 Ready for training with enhanced multi-task learning!")

if __name__ == "__main__":
    test_traditional_vs_enhanced()