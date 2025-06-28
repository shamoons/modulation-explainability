# Cyclic Learning Rate Guide

## Overview
CyclicLR is now the default scheduler. It varies the learning rate cyclically between base_lr and max_lr using the triangular2 mode.

## What to Expect

### Learning Rate Schedule (New Default Settings)
- **Base LR**: 1e-6 (0.000001) - Ultra-low for stability
- **Max LR**: 1e-4 (0.0001) - 100x base_lr for aggressive exploration
- **Mode**: triangular2 (amplitude halves each cycle)
- **Cycle**: 10 epochs (5 up, 5 down)
- **Updates**: Per batch (smooth, continuous transitions)

### Epoch-by-Epoch LR Pattern

**Cycle 1 (Epochs 1-10)**:
- Epochs 1-5: LR smoothly increases from 1e-6 to 1e-4
- Epochs 6-10: LR smoothly decreases from 1e-4 to 1e-6
- Peak amplitude: 1e-4 (similar to previous static LR runs)

**Cycle 2 (Epochs 11-20)**:
- Epochs 11-15: LR increases from 1e-6 to 5e-5 (halved amplitude)
- Epochs 16-20: LR decreases from 5e-5 to 1e-6
- Peak amplitude: 5e-5

**Cycle 3 (Epochs 21-30)**:
- Epochs 21-25: LR increases from 1e-6 to 2.5e-5
- Epochs 26-30: LR decreases from 2.5e-5 to 1e-6
- Peak amplitude: 2.5e-5

### Why Per-Batch Updates?
- **Standard practice**: PyTorch CyclicLR default
- **Smooth transitions**: No sudden jumps
- **More exploration**: ~400+ LR updates per epoch vs 1
- **Better optimization**: Continuous adaptation to loss landscape

### Expected Training Behavior

1. **Early Cycles (1-20 epochs)**:
   - Large LR swings for exploration
   - May see validation accuracy fluctuations
   - Helps escape local optima (especially at high SNRs)

2. **Mid Training (21-50 epochs)**:
   - Smaller amplitude cycles
   - More stable convergence
   - Fine-tuning with controlled exploration

3. **Late Training (50+ epochs)**:
   - Very small amplitude cycles
   - Essentially oscillating near base_lr
   - Final refinement phase

### Benefits for SNR Regression

1. **Escape Attractors**: High LR phases help escape 24/26/28 dB attractors
2. **Exploration**: Discovers better minima for high SNR predictions
3. **Stability**: Decreasing amplitude prevents divergence
4. **No Plateau**: Continuous learning rate variation

### Monitoring Tips

Watch for:
- **Validation spikes**: Normal during high LR phases
- **Overall trend**: Should improve despite oscillations
- **High SNR F1 scores**: Should improve during exploration phases
- **Task balance**: May shift during cycles

### Command Example

```bash
uv run python src/train_constellation.py \
    --model_type swin_tiny \
    --batch_size 256 \
    --snr_list "0,2,4,6,8,10,12,14,16,18,20,22,24,26,28,30" \
    --epochs 100
```

Note: No need to specify base_lr - it's automatically set to 1e-6 for cyclic.
Max LR defaults to 1e-4 unless overridden.

### Adjusting Parameters

- **Faster cycles**: `--step_size_up 3 --step_size_down 3` (6 epoch cycles)
- **Larger range**: `--max_lr 5e-4` (500x range - very aggressive)
- **Conservative**: `--max_lr 5e-5` (50x range - more stable)

## Key Differences from Previous Runs

1. **No more ReduceLROnPlateau**: Active exploration throughout training
2. **Per-batch updates**: LR changes smoothly every batch, not per epoch
3. **Automatic amplitude decay**: Natural annealing without manual intervention