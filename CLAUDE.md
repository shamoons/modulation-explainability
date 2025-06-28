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
- **Active Run**: balmy-waterfall-174 (SNR regression)
- **Progress**: Epoch 2 - 32.42% combined (68.15% mod, 50.41% SNR)
- **Key Result**: No more 28 dB black hole!

## Files
- **@RUNLOG.md**: Experimental results
- **@PAPER.md**: Academic analysis
- **src/**: Training/model code
- **constellation_diagrams/**: SNR-preserving dataset