#!/usr/bin/env python3
"""
Test the cycle-aware patience logic
"""

def test_cycle_aware_patience():
    """Test different configurations"""
    
    test_cases = [
        # (epochs, cycles_per_training, expected_cycle_length, expected_patience)
        (50, 5, 10, 20),    # 50 epochs, 5 cycles → 10 epochs/cycle → 20 patience
        (150, 5, 30, 60),   # 150 epochs, 5 cycles → 30 epochs/cycle → 60 patience
        (30, 3, 10, 20),    # 30 epochs, 3 cycles → 10 epochs/cycle → 20 patience
        (100, 4, 25, 50),   # 100 epochs, 4 cycles → 25 epochs/cycle → 50 patience
        (20, 5, 4, 10),     # 20 epochs, 5 cycles → 4 epochs/cycle → 10 (minimum)
    ]
    
    print("Testing cycle-aware patience logic:")
    print("=" * 60)
    
    for epochs, cycles_per_training, expected_cycle_length, expected_patience in test_cases:
        # Calculate cycle-aware patience (same logic as in training_constellation.py)
        cycle_length = epochs / cycles_per_training
        patience = max(int(cycle_length * 2), 10)  # At least 2 cycles, minimum 10
        
        step_size_up = int(cycle_length / 2)
        step_size_down = int(cycle_length / 2)
        
        print(f"Epochs: {epochs}, Cycles: {cycles_per_training}")
        print(f"  Cycle length: {cycle_length:.1f} epochs")
        print(f"  Patience: {patience} epochs")
        print(f"  Step sizes: up={step_size_up}, down={step_size_down}")
        print(f"  Expected: cycle_length={expected_cycle_length}, patience={expected_patience}")
        print(f"  ✓ Match: {abs(cycle_length - expected_cycle_length) < 0.1 and patience == expected_patience}")
        print()

if __name__ == "__main__":
    test_cycle_aware_patience()