# Negative Loss Investigation Report

## Problem Summary
During training, the loss value is showing as negative (-2.062 in the example), which is mathematically impossible for standard loss functions like CrossEntropyLoss. This indicates a bug in the loss calculation or weighting mechanism.

## Root Cause Analysis

### Primary Issue: Uncertainty Weighted Loss Regularization Term

The negative loss is caused by the `AnalyticalUncertaintyWeightedLoss` class in `src/losses/uncertainty_weighted_loss.py`. Specifically, the issue is in the regularization term:


```python
regularization = 0.5 * torch.sum(self.log_vars)
```

**Problem Details:**
1. `self.log_vars` are learnable parameters initialized as zeros
2. During training, these parameters can become negative
3. When `log_vars` are negative, the regularization term becomes negative
4. This negative regularization term can make the total loss negative

**Code Location:** `src/losses/uncertainty_weighted_loss.py`, line 58

### Loss Calculation Flow

1. **Individual Task Losses:** 
   - `loss_modulation = criterion_modulation(modulation_output, modulation_labels)` (CrossEntropyLoss)
   - `loss_snr = criterion_snr(snr_output, snr_labels)` (CrossEntropyLoss)
   - Both should be positive

2. **Uncertainty Weighting:**
   - `total_loss, task_weights = uncertainty_weighter([loss_modulation, loss_snr])`
   - This applies weights and adds the problematic regularization term

3. **Final Loss:**
   - `total_loss = torch.sum(weighted_losses) + regularization`
   - When `regularization < 0` and large enough, total_loss becomes negative

## Secondary Issues Identified

### 1. Inconsistent Loss Function Usage
- The training code uses `CrossEntropyLoss()` for both modulation and SNR classification
- However, there's a custom `DistancePenalizedSNRLoss` available that's not being used
- This suggests the SNR loss might not be optimal for the task

### 2. Potential Numerical Instability
- The uncertainty weighting uses `exp(-log_vars)` which can cause numerical issues
- No bounds or constraints on the `log_vars` parameters

### 3. Loss Monitoring Issues
- The progress bar shows `total_loss.item()` which includes the negative regularization
- Individual task losses are not separately monitored in the progress bar

## Impact Assessment

### Training Behavior
- **Negative loss values** are mathematically meaningless for optimization
- **Gradient flow** may be affected by the negative regularization term
- **Model convergence** could be compromised
- **Loss monitoring** becomes unreliable

### Model Performance
- The model might still learn due to the positive task losses
- However, the optimization process is not mathematically sound
- Validation loss comparisons may be misleading

## Recommended Solutions

### Immediate Fix (High Priority)

1. **Remove or Fix Regularization Term:**
   ```python
   # Option 1: Remove regularization entirely
   total_loss = torch.sum(weighted_losses)
   
   # Option 2: Use absolute value or exponential
   regularization = 0.5 * torch.sum(torch.abs(self.log_vars))
   # OR
   regularization = 0.5 * torch.sum(torch.exp(self.log_vars))
   ```

2. **Add Parameter Constraints:**
   ```python
   # Constrain log_vars to be non-negative
   self.log_vars = nn.Parameter(torch.zeros(num_tasks, device=device))
   # Add in forward pass:
   log_vars = torch.clamp(self.log_vars, min=0.0)
   ```

### Medium Priority Improvements

3. **Separate Loss Monitoring:**
   - Log individual task losses separately
   - Monitor uncertainty weights independently
   - Add loss sanity checks

4. **Use Appropriate SNR Loss:**
   - Consider using `DistancePenalizedSNRLoss` for SNR classification
   - This would be more appropriate for ordinal SNR values

5. **Add Loss Validation:**
   ```python
   def validate_loss(loss_value):
       if torch.isnan(loss_value) or torch.isinf(loss_value):
           raise ValueError(f"Invalid loss value: {loss_value}")
       if loss_value < 0:
           print(f"Warning: Negative loss detected: {loss_value}")
   ```

### Long-term Improvements

6. **Implement Proper Uncertainty Weighting:**
   - Review the uncertainty weighting approach
   - Consider using established methods like Kendall's uncertainty weighting
   - Add proper mathematical constraints

7. **Enhanced Logging:**
   - Log all loss components separately
   - Add loss distribution monitoring
   - Implement early stopping based on loss validity

## Implementation Plan

### Phase 1: Immediate Fix (1-2 hours)
1. Fix the regularization term in `uncertainty_weighted_loss.py`
2. Add loss validation checks
3. Test with a small training run

### Phase 2: Monitoring Improvements (2-3 hours)
1. Separate loss component logging
2. Add progress bar improvements
3. Implement loss sanity checks

### Phase 3: Loss Function Optimization (3-4 hours)
1. Evaluate SNR loss function options
2. Implement proper uncertainty weighting
3. Add comprehensive testing

## Testing Strategy

1. **Unit Tests:**
   - Test loss functions with known inputs
   - Verify loss values are always positive
   - Test edge cases (zero loss, large values)

2. **Integration Tests:**
   - Run short training sessions
   - Monitor loss progression
   - Verify model convergence

3. **Validation:**
   - Compare with baseline training (without uncertainty weighting)
   - Verify model performance metrics
   - Check for any regressions

## Conclusion

The negative loss issue is caused by an improperly implemented regularization term in the uncertainty weighted loss function. This is a critical bug that needs immediate attention as it affects the mathematical soundness of the training process. The fix is straightforward but should be implemented carefully to avoid introducing new issues.

The recommended approach is to fix the regularization term first, then gradually improve the loss monitoring and potentially switch to more appropriate loss functions for the specific tasks. 