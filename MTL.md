# Multi-Task Learning Implementation

## Current Approach

Our current implementation uses a shared backbone with task-specific heads and static loss weighting:

```python
total_loss = alpha * loss_modulation + beta * loss_snr
```

### Architecture
1. **Shared Backbone**: ResNet-based feature extractor
2. **Shared Layers**: 
   - Linear transformation
   - ReLU activation
   - Batch normalization
   - Dropout
3. **Task-Specific Heads**:
   - Modulation classification head
   - SNR prediction head

### Strengths
- Efficient parameter sharing
- Simple implementation
- Configurable loss weights
- Task-specific feature learning

### Limitations
- Static loss weights don't adapt to task difficulty
- No automatic task balancing
- Equal feature sharing might not be optimal
- Limited cross-task interaction

## Proposed Changes

### 1. Dynamic Loss Weighting (Kendall et al., 2018)
Reference: [Multi-Task Learning Using Uncertainty to Weigh Losses for Scene Geometry and Semantics](https://arxiv.org/abs/1705.07115)

Implementation approach:
```python
class DynamicWeightedLoss(nn.Module):
    def __init__(self, num_tasks):
        super().__init__()
        # Learnable log variances for each task
        self.log_vars = nn.Parameter(torch.zeros(num_tasks))
        
    def forward(self, losses):
        # losses: list of individual task losses
        weights = torch.exp(-self.log_vars)
        weighted_losses = [0.5 * torch.exp(-log_var) * loss + log_var 
                          for loss, log_var in zip(losses, self.log_vars)]
        return sum(weighted_losses)
```

### 2. Task-Specific Normalization
- Separate batch normalization layers for each task
- Allows tasks to learn their own feature statistics

### 3. Cross-Task Attention
- Add attention mechanisms between tasks
- Enable tasks to share information at the head level
- Reference: [Cross-Task Attention for Multi-Task Learning](https://arxiv.org/abs/2007.02808)

### 4. Progressive Training
- Start with shared features
- Gradually introduce task-specific features
- Reference: [Progressive Neural Networks](https://arxiv.org/abs/1606.04671)

## Implementation Checklist

### Phase 1: Dynamic Loss Weighting
- [ ] Clean up old files
  - [ ] Remove src/config/snr_buckets.json
  - [ ] Remove src/config/loss_weights.json
- [ ] Update code
  - [ ] Remove load_loss_config() import and usage
  - [ ] Remove alpha/beta parameters from training
  - [ ] Add DynamicWeightedLoss import
  - [ ] Initialize criterion_dynamic with num_tasks=2
  - [ ] Move criterion_dynamic to device
  - [ ] Update loss combination in training loop
- [ ] Add monitoring
  - [ ] Add logging for task weights
  - [ ] Add visualization of weight evolution
  - [ ] Add task-specific loss tracking
- [ ] Validation
  - [ ] Verify loss computation
  - [ ] Check weight evolution
  - [ ] Compare with baseline performance

### Phase 2: Task-Specific Normalization
- [ ] Modify model architecture for task-specific batch norm
- [ ] Update forward pass
- [ ] Add configuration options
- [ ] Benchmark performance impact

### Phase 3: Cross-Task Attention
- [ ] Design attention mechanism
- [ ] Implement attention module
- [ ] Integrate with existing architecture
- [ ] Add attention visualization
- [ ] Benchmark performance

### Phase 4: Progressive Training
- [ ] Implement progressive training schedule
- [ ] Add configuration options
- [ ] Create training visualization
- [ ] Benchmark against baseline

### Phase 5: Evaluation and Optimization
- [ ] Comprehensive benchmarking
- [ ] Ablation studies
- [ ] Hyperparameter optimization
- [ ] Documentation updates

### Additional Investigation
- [ ] Investigate low modulation accuracy (4.66%)
  - [ ] Check data preprocessing
  - [ ] Verify label encoding
  - [ ] Analyze class distribution
- [ ] Investigate high SNR MAE (12.10 dB)
  - [ ] Check SNR value normalization
  - [ ] Verify SNR label encoding
  - [ ] Analyze SNR distribution

## References
1. Kendall, A., Gal, Y., & Cipolla, R. (2018). Multi-Task Learning Using Uncertainty to Weigh Losses for Scene Geometry and Semantics. CVPR.
2. Liu, S., Johns, E., & Davison, A. J. (2019). End-to-End Multi-Task Learning with Attention. CVPR.
3. Rusu, A. A., et al. (2016). Progressive Neural Networks. arXiv.
4. Zhang, Z., et al. (2020). Cross-Task Attention for Multi-Task Learning. NeurIPS. 