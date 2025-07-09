# CLAUDE.md

## Project Overview
**Joint modulation-SNR prediction** using constellation diagrams and Swin Transformer. First comprehensive study of simultaneous classification (17 mods × 16 SNRs = 272 classes) with perturbation-based explainability.

## Key Innovation
- **SOTA**: 95%+ on modulation-only classification
- **Our Challenge**: Joint 272-class problem (~10x harder)
- **Current Best**: 46.48% combined (74.60% mod, 62.79% SNR)

## Tech Stack
Python 3.11, PyTorch 2.4.1, NumPy<2.0, H5PY, Weights & Biases, UV package manager

## Critical Discoveries

### 1. SNR-Preserving Preprocessing
```python
# CORRECT (preserves SNR info)
power = np.mean(I**2 + Q**2)
scale_factor = np.sqrt(power)
I_norm, Q_norm = I/scale_factor, Q/scale_factor
H = np.log1p(histogram2d(I_norm, Q_norm))

# WRONG (destroys SNR info)
H = H / H.max()  # Per-image normalization
```

### 2. SNR Regression (NEW DEFAULT)
- **Old**: 16-class classification → 28 dB black hole
- **New**: Continuous regression → smooth predictions
- **Loss**: SmoothL1Loss for robustness

### 3. Architecture Ranking
1. **Swin-Tiny**: 28M params, hierarchical attention ✅
2. **ResNet18/34**: 11-21M params, 23-26% ceiling ❌
3. **ViT**: Memory issues, unstable ❌

### 4. SNR Performance Paradox
- **Low (-20 to -2 dB)**: F1=0.000 (noise dominance)
- **Mid (0-14 dB)**: F1>0.73 (optimal discrimination)
- **High (16-30 dB)**: F1<0.31 (over-clarity paradox)

## Commands
```bash
# Train (default: SNR regression)
uv run python src/train_constellation.py --model_type swin_tiny --batch_size 256 --epochs 100

# Generate constellations
uv run python src/generate_snr_preserving_constellations.py --h5_dir data/split_hdf5 --output_dir constellation_diagrams

# Resume training
uv run python src/train_constellation.py --checkpoint checkpoints/best_model_swin_tiny_epoch_X.pth
```

## Current Status
- **Best Run**: 48.65% combined (76.20% mod, 66.02% SNR) - ResNet50
- **Stability**: Major issue - 79% crash rate in sweeps
- **Key Finding**: ResNet50 + bottleneck_64 dominates when stable

## Recent W&B Sweep Analysis (July 2025)

### Best Configurations Found
1. **ResNet50 + bottleneck_64**: 48.65% (1e-3 LR, batch 256)
2. **ResNet50 + bottleneck_128**: 46.58% (1e-4 LR, batch 128)
3. **ResNet34 + bottleneck_128**: 45.21% (5e-4 LR, batch 256)

### Architecture Rankings (168 runs analyzed)
1. **ResNet50**: avg 27.24%, max 48.65% (n=48)
2. **ViT-B/16**: avg 25.93%, max 38.84% (n=11) 
3. **ResNet34**: avg 25.77%, max 45.49% (n=44)
4. **Swin Tiny**: avg 18.89%, max 38.77% (n=31)
5. **ResNet18**: avg 18.76%, max 34.71% (n=4)
6. **Swin Small**: avg 17.96%, max 31.71% (n=13)

### Key Hyperparameter Insights
- **SNR Layer**: bottleneck_64/128 >> dual_layer > standard
- **Learning Rate**: 1e-4 optimal (1e-3 achieves best but 85% crash rate)
- **Batch Size**: 128-256 for stability (1024 good but n=2)
- **Pretrained**: Critical - always use pretrained weights
- **Warmup**: No significant benefit observed

### Stability Crisis
- **79.2% crash rate** across all sweeps (133/168 runs)
- High LR (1e-3) + large batch (256) = best performance but unstable
- Conservative settings (lr≤1e-4, batch≤128) = lower performance but stable

## Files
- **@RUNLOG.md**: Experimental results
- **@PAPER.md**: Academic analysis
- **src/**: Training/model code
- **constellation_diagrams/**: SNR-preserving dataset