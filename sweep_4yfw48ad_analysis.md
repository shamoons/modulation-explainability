# W&B Sweep 4yfw48ad - Comprehensive Analysis

## Sweep Overview
- **Total Runs**: 50
- **Date Range**: July 3-6, 2025
- **Purpose**: Enhanced SNR Bottleneck + Distance Penalty architecture comparison across 5 models

## Run Status Summary
- **Finished**: 5 runs (10%)
- **Crashed**: 42 runs (84%)
- **Killed**: 2 runs (4%)
- **Running**: 1 run (2%)
- **Failed**: 2 runs (0 epochs)

## Architecture Performance Rankings

### Best Achieved Results
1. **ResNet50** (run: rux19b6a): **48.65%** combined (76.20% mod, 66.02% SNR) - killed at epoch 10
2. **ResNet50** (run: 9lcu6mg2): **46.78%** combined (75.55% mod, 64.55% SNR) - finished 15 epochs
3. **ResNet34** (run: ook1y09y): **45.61%** combined (75.25% mod, 63.16% SNR) - finished 23 epochs
4. **ViT-B/16** (run: p2eyos5p): **40.78%** combined (74.00% mod, 57.26% SNR) - finished 15 epochs
5. **ResNet34** (run: 454epsxa): **39.39%** combined (70.22% mod, 58.08% SNR) - finished 15 epochs
6. **Swin Tiny** (run: fip4a35q): **36.71%** combined (71.40% mod, 54.32% SNR) - finished 10 epochs

### Architecture Stability Analysis
| Architecture | Total Runs | Avg Epochs | Finished | Crash Rate | Best Result |
|-------------|------------|------------|----------|------------|-------------|
| ResNet50    | 7          | 4.3        | 1        | 71%        | 48.65%      |
| ResNet34    | 10         | 5.5        | 2        | 70%        | 45.61%      |
| ViT-B/16    | 13         | 3.8        | 1        | 85%        | 40.78%      |
| Swin Tiny   | 16         | 3.0        | 1        | 88%        | 36.71%      |
| Swin Small  | 4          | 2.0        | 0        | 100%       | N/A         |

## Key Configuration Insights

### SNR Layer Architecture Distribution
- **bottleneck_64**: 17 runs (34%) - **Dominated top results**
- **bottleneck_128**: 14 runs (28%)
- **dual_layer**: 14 runs (28%)
- **standard**: 5 runs (10%)

### Hyperparameter Patterns

#### Learning Rate Settings
- **High LR (max_lr=1e-3)**: 20 runs
  - Best performance (ResNet50: 48.65%)
  - Highest crash rate (85%)
- **Medium LR (max_lr=5e-4)**: 15 runs
  - More stable (60% crash rate)
  - Still good performance (ResNet34: 45.61%)
- **Low LR (max_lr=1e-4)**: 15 runs
  - Most stable but lower performance

#### Batch Size Impact
- **256**: 13 runs - Correlated with both best performance AND crashes
- **128**: 21 runs - More stable middle ground
- **64**: 16 runs - Most stable but lower throughput

#### Pretrained Weights
- **True**: 30 runs (60%) - Critical for stability
- **False**: 20 runs (40%) - Higher crash rate

## Crash Pattern Analysis

### Primary Crash Causes
1. **Memory Issues** (Swin models with batch_size=256)
2. **Gradient Explosions** (High LR + no pretrained weights)
3. **Early Training Instability** (75% crashed at epoch 2)

### Crash Distribution by Epoch
- Epoch 2: 32 crashes (76%)
- Epoch 6: 6 crashes (14%)
- Epoch 7: 2 crashes (5%)
- Later: 2 crashes (5%)

### Risk Factors (Ordered by Impact)
1. **Swin Small**: 100% crash rate regardless of config
2. **batch_size=256 + max_lr=1e-3**: 90% crash rate
3. **No pretrained weights + high LR**: 85% crash rate
4. **Swin architectures generally**: Higher memory requirements

## Successful Configuration Patterns

### Optimal Configs for Each Architecture

#### ResNet50 (Best Overall)
```yaml
model_type: resnet50
snr_layer_config: bottleneck_64
batch_size: 256
max_lr: 1e-3
use_pretrained: true
warmup_epochs: 0
```

#### ResNet34 (Most Reliable)
```yaml
model_type: resnet34
snr_layer_config: bottleneck_64
batch_size: 64
max_lr: 5e-4
use_pretrained: true
warmup_epochs: 10
```

#### ViT-B/16 (Best Transformer)
```yaml
model_type: vit_b_16
snr_layer_config: dual_layer
batch_size: 128
max_lr: 1e-3
use_pretrained: true
warmup_epochs: 0
```

## Key Findings

1. **ResNet50 Shows Highest Potential**: Achieved 48.65% but stability issues prevent full training
2. **bottleneck_64 SNR Layer**: Consistently outperformed other architectures
3. **Stability vs Performance Trade-off**: 
   - High LR (1e-3) + Large batch (256) = Best performance
   - But 85% crash rate makes it impractical
4. **Architecture-Specific Issues**:
   - Swin models: Memory constraints limit batch size
   - ViT: More stable than Swin but still prone to crashes
   - ResNet: Most reliable, especially ResNet34

## Recommendations

1. **For Best Performance**: ResNet50 with bottleneck_64, but use:
   - max_lr: 5e-4 (not 1e-3)
   - batch_size: 128 (not 256)
   - Mandatory pretrained weights
   - Add gradient clipping

2. **For Reliability**: ResNet34 with:
   - Conservative LR (1e-4 to 5e-4)
   - Standard batch sizes (64-128)
   - Warmup epochs (5-10)

3. **Avoid**:
   - Swin Small (100% failure rate)
   - batch_size=256 without extensive testing
   - max_lr=1e-3 without gradient clipping
   - Training without pretrained weights

## Future Sweep Suggestions

1. Focus on ResNet34/50 with bottleneck_64
2. Narrow LR range: 1e-4 to 5e-4
3. Add gradient clipping for stability
4. Test intermediate batch sizes (96, 192)
5. Investigate why bottleneck_64 specifically works so well