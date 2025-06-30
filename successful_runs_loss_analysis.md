# Loss Analysis for Successful Runs Without Attractors

## super-plasma-180 (α=0.5) - DISTANCE-WEIGHTED CLASSIFICATION
**Status**: ✅ COMPLETED - NO ATTRACTORS
**Best Performance**: 46.31% combined (74.72% mod, 63.60% SNR)

### Loss Values:
- **Epoch 1**: Loss: 2.443 validation vs 3.485 training
- **Epoch 2**: Loss: 1.945 validation vs 2.358 training
- **Epoch 10 (Best)**: Validation loss 1.143

### Key Observations:
- Started with higher loss due to distance penalty term
- Rapid decrease: 2.443 → 1.945 (-20.4%) in one epoch
- Final best loss: 1.143 (significantly lower than start)
- Loss function: Classification + inverse-square distance penalty (1/d²)
- Successfully prevented 26-28 dB attractors

---

## honest-silence-181 (α=1.0) - STRONGER DISTANCE PENALTY
**Status**: ✅ COMPLETED - NO ATTRACTORS (but slightly worse performance)
**Best Performance**: 45.88% combined (73.92% mod, 63.42% SNR)

### Loss Values:
- **Epoch 1**: Loss: 3.735 validation vs 5.115 training
- **Epoch 2**: Loss: 3.097 validation vs 3.700 training
- **Epoch 20 (Best)**: Validation Loss: 1.3796
- **Final Test**: Test Loss: 1.3798

### Key Observations:
- Higher initial loss values (~2x) due to stronger penalty (α=1.0 vs 0.5)
- Loss values consistently higher than α=0.5 run
- Still achieved good performance without attractors
- Task imbalance more severe (83.5%/16.5%)

---

## super-cloud-175 - LOW LEARNING RATE REGRESSION
**Status**: ✅ COMPLETED - REDUCED ATTRACTOR (not eliminated)
**Best Performance**: 41.77% combined (74.06% mod, 58.65% SNR)

### Loss Values:
- **Epoch 16**: Validation loss 0.9807 (mentioned as "new best")
- **Loss Function**: SmoothL1Loss for SNR (regression, not classification)

### Key Observations:
- Used regression approach instead of classification
- Lower learning rate (1e-5) helped reduce 26 dB attractor
- Validation loss lower than classification approaches (0.9807)
- But still had some high SNR distribution issues

---

## Summary of Loss Patterns for Successful Runs:

1. **Distance-Weighted Classification (α=0.5)**:
   - Initial loss: 2.4-3.5 range
   - Converged to: ~1.1-1.2
   - No attractors formed

2. **Distance-Weighted Classification (α=1.0)**:
   - Initial loss: 3.7-5.1 range (higher due to stronger penalty)
   - Converged to: ~1.3-1.4
   - No attractors but task imbalance

3. **Regression with Low LR**:
   - Converged to: ~0.98 (lower absolute value)
   - Different loss scale due to SmoothL1Loss
   - Reduced but didn't eliminate attractors

## Key Insights:
- Distance-weighted classification with α=0.5 appears optimal
- Initial loss values 2-4 range are normal for classification with distance penalty
- Final loss values around 1.1-1.4 indicate good convergence without attractors
- Higher α leads to higher loss values but can cause task imbalance