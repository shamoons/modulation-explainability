# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a **joint modulation-SNR classification** research project combining enhanced multi-task learning with perturbation-based explainability for Automatic Modulation Classification (AMC). The framework addresses both performance and interpretability challenges by transforming I/Q signal data into constellation diagrams and employing hierarchical attention (Swin Transformer) to **simultaneously** classify modulation schemes and predict discrete SNR values.

### Novel Joint Prediction Approach

**Key Innovation**: Unlike existing SOTA approaches that focus on single-task modulation classification or train separate models per SNR range, our work tackles the significantly more challenging **joint modulation-SNR prediction** problem:

- **Traditional SOTA**: Single-task modulation classification achieving 95%+ accuracy
- **Our Approach**: Joint 272-class problem (17 modulations × 16 SNRs) with explainability
- **Challenge Level**: ~10x more complex than single-task approaches

### Research Contributions

1. **Perturbation-Based Explainability**: Novel PIS (Perturbation Impact Score) metric for constellation AMC
2. **Joint Multi-Task Learning**: First comprehensive study of simultaneous modulation-SNR prediction 
3. **Enhanced Constellation Generation**: Literature-standard preprocessing methodology
4. **Hierarchical Attention**: Swin Transformer breakthrough for constellation patterns
5. **Academic Rigor**: 500+ experimental runs with systematic architecture evaluation

## Technology Stack

- **Language**: Python 3.11
- **Deep Learning**: PyTorch 2.4.1, TorchVision 0.19.1
- **Data Processing**: NumPy (<2.0), Pandas, H5PY, SciPy
- **Visualization**: Matplotlib, Seaborn
- **Experiment Tracking**: Weights & Biases (wandb)
- **Package Management**: uv (UV package manager)

## Mathematical Framework

### Enhanced Constellation Generation

Power normalization preserving signal characteristics:
```
power = (1/N) ∑(I²ᵢ + Q²ᵢ)
scale_factor = √power
I_norm = I/scale_factor, Q_norm = Q/scale_factor
H = log(1 + histogram2d(I_norm, Q_norm))
```

### Multi-Task Loss with Uncertainty Weighting

Kendall homoscedastic uncertainty weighting for automatic task balancing:
```
L_total = (1/2σ²_mod)L_mod + (1/2σ²_snr)L_snr + log(σ_mod·σ_snr)
```

Where σ_mod and σ_snr are learned uncertainty parameters preventing task competition.

### SNR Distance-Penalized Loss

Ordinal relationship preservation for SNR prediction:
```
L_snr = α·L_CE + β·(1/N)∑|y_i - ŷ_i|²
```

### Perturbation Impact Score (PIS)

Novel explainability metric quantifying feature importance:
```
ΔA = A_original - A_perturbed
PIS = ΔA/f
```

Where f is the fraction of perturbed input data.

## Common Development Commands

```bash
# Install dependencies
uv sync

# Train with current best architecture (Swin Transformer)
uv run python src/train_constellation.py --model_type swin_tiny

# Train with bounded SNR range (current approach)
uv run python src/train_constellation.py \
    --model_type swin_tiny \
    --batch_size 256 \
    --snr_list "0,2,4,6,8,10,12,14,16,18,20,22,24,26,28,30" \
    --epochs 100

# Generate SNR-preserving constellation diagrams
uv run python src/generate_snr_preserving_constellations.py \
    --h5_dir data/split_hdf5 \
    --output_dir constellation_diagrams

# Resume training from checkpoint
uv run python src/train_constellation.py \
    --checkpoint checkpoints/best_model_swin_tiny_epoch_X.pth

# Test with perturbation analysis
uv run python src/test_constellation.py \
    --model_checkpoint <path> \
    --data_dir constellation_diagrams \
    --perturbation_dir <perturbation_path>

# Hyperparameter optimization
wandb sweep sweep.yml
wandb agent <sweep_id>
```

## Current Research Status

### 🚀 **Breakthrough Performance (iconic-serenity-164)**

**Active Run Achievements**:
- **SNR Accuracy**: 50.16% validation - **FIRST TIME ABOVE 50%**
- **Combined Accuracy**: 31.28% - **exceeds all previous runs**
- **Architecture**: Swin Transformer with bounded SNR range (0-30 dB)
- **Task Balance**: 55.4%/44.6% (optimal vs previous 60%/40% imbalance)

### Key Technical Discoveries

#### 1. **Joint vs Single-Task Complexity**
- **Single-Task SOTA**: 95%+ accuracy on modulation-only classification
- **Our Joint Task**: 31.28% on 272-class joint prediction (significantly harder)
- **Academic Significance**: First systematic study of joint complexity scaling

#### 2. **SNR-Performance Paradox**
Mid-range SNRs (0-14 dB) outperform both extremes:
- **Low SNR (-20 to -2 dB)**: F1 = 0.000 (noise dominance)
- **Mid SNR (0-14 dB)**: F1 > 0.73 (optimal discrimination)
- **High SNR (16-30 dB)**: F1 < 0.31 (over-clarity paradox)

#### 3. **Architecture Hierarchy for Constellations**
Empirical performance ranking:
1. **Swin Transformer**: Hierarchical attention, 31.28% combined accuracy
2. **ResNet18/34**: Traditional CNN, 23-26% ceiling
3. **ViT**: Global attention, memory constraints, instability

#### 4. **Perturbation Insights**
- **High-intensity regions**: Critical for classification (PIS up to 34.8)
- **Low-intensity regions**: Minimal impact (PIS < 1.0)
- **Validates**: Model focuses on constellation points, not noise artifacts

## Comparative Analysis vs SOTA

### Traditional AMC Approaches
Most existing work focuses on **single-task** modulation classification:
- WCTFormer (2024): 97.8% on modulation-only (RadioML2018.01a)
- TLDNN (2024): 62.83% on modulation-only (RadioML2016.10a)
- Ultralight (2024): 96.3% on modulation-only (synthetic data)

### Our Joint Approach
**Novel Problem Formulation**: Simultaneous modulation + SNR prediction
- **Complexity**: 272 classes vs typical 11-24 modulation classes
- **Challenge**: No SOTA baselines for direct comparison
- **Innovation**: First comprehensive joint prediction study with explainability

### Academic Positioning
1. **Problem Novelty**: Joint prediction largely unexplored in AMC literature
2. **Methodological Innovation**: Perturbation-based explainability for signal processing
3. **Architectural Discovery**: Hierarchical attention superiority for constellation tasks
4. **Experimental Rigor**: 500+ runs, systematic evaluation framework

## Development Guidelines

- **Research Updates**: Update @PAPER.md for academic findings, @RUNLOG.md for experimental results
- **Architecture Changes**: Test with current best (Swin Transformer) configuration first
- **SNR Range**: Use bounded 0-30 dB following literature precedent and current optimal performance
- **Explainability**: Implement perturbation analysis for model interpretability validation

## Project File Structure

- **@PAPER.md**: Academic research notes and literature positioning
- **@RUNLOG.md**: Detailed experimental run documentation and results
- **@CLAUDE.md**: Development guidance and project overview (this file)
- **papers/ELSP_Paper/**: Academic paper draft with ELSP template
- **src/**: Source code for training, testing, and constellation generation
- **constellation_diagrams/**: SNR-preserving constellation diagram dataset

---

*This compact guide focuses on current breakthrough methodology and joint prediction innovation. Detailed experimental history is maintained in @RUNLOG.md and academic analysis in @PAPER.md.*