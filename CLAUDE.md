# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a modulation explainability research project that combines **enhanced multi-task learning** with perturbation-based explainability for Automatic Modulation Classification (AMC). The framework addresses both performance and interpretability challenges by transforming I/Q signal data into constellation diagrams and employing a ResNet-based architecture to simultaneously classify modulation schemes and predict discrete SNR values.

### Research Paper Context
This work was submitted as "Constellation Diagram Augmentation and Perturbation-Based Explainability for Automatic Modulation Classification" and introduces novel perturbation-based explainability techniques using the Perturbation Impact Score (PIS) metric to analyze critical regions in constellation diagrams that drive model decisions.

## Technology Stack

- **Language**: Python 3.11
- **Deep Learning**: PyTorch 2.4.1, TorchVision 0.19.1
- **Data Processing**: NumPy (<2.0), Pandas, H5PY, SciPy
- **Visualization**: Matplotlib, Seaborn
- **Experiment Tracking**: Weights & Biases (wandb)
- **Package Management**: uv (UV package manager)

## Common Development Commands

```bash
# Install dependencies with UV
uv sync

# Train model with default settings (enhanced multi-task learning)
uv run python src/train_constellation.py

# Train with custom parameters and model architecture
uv run python src/train_constellation.py \
    --model_type vit_b_16 \
    --batch_size 32 \
    --snr_list "0,10,20" \
    --mods_to_process "BPSK,QPSK,8PSK" \
    --epochs 100 \
    --base_lr 1e-4 \
    --weight_decay 1e-5 \
    --dropout 0.3 \
    --patience 10

# Train with different architectures
uv run python src/train_constellation.py --model_type resnet18     # Default, fastest
uv run python src/train_constellation.py --model_type resnet34     # Deeper ResNet
uv run python src/train_constellation.py --model_type vit_b_16     # Vision Transformer ViT/16
uv run python src/train_constellation.py --model_type vit_b_32     # Vision Transformer ViT/32 (faster)
uv run python src/train_constellation.py --model_type swin_tiny   # Swin Transformer (fastest, hierarchical)
uv run python src/train_constellation.py --model_type swin_small  # Swin Transformer (balanced)
uv run python src/train_constellation.py --model_type swin_base   # Swin Transformer (largest)

# Resume training from checkpoint (now includes model name)
uv run python src/train_constellation.py --checkpoint checkpoints/best_model_resnet18_epoch_15.pth

# Convert HDF5 data to constellation images
uv run python src/convert_to_constellation.py \
    --h5_dir data/split_hdf5 \
    --snr_list -20,-18,-16,0,10,20,30 \
    --mod_list BPSK,QPSK,8PSK,16PSK

# Test model on perturbed and non-perturbed data
uv run python src/test_constellation.py \
    --model_checkpoint <path_to_checkpoint> \
    --data_dir constellation \
    --perturbation_dir <path_to_perturbation_dir>

# Run hyperparameter sweep with W&B
wandb sweep sweep.yml
wandb agent <sweep_id>

# Generate perturbed constellation data
uv run python src/perturb_constellations.py

# Calculate PID metrics
uv run python src/calculate_pid.py
```

## Project Workflow and Guidance

### Development Guidelines
- When making model/architectural/approach changes that are academic, be sure to update @PAPER.md accordingly

## Training Run History & Architecture Exploration

### 🚨 **Critical Discovery - Model Capacity Ceiling (June 2025)**

**Performance Plateau Issue**: Systematic evaluation revealed fundamental limitation at ~24-26% validation combined accuracy across all tested architectures.

#### Architecture Performance Ceiling Analysis

**Comprehensive Testing Results**:
- **ResNet18/34**: Consistent plateau at 23-26% validation accuracy (11-21M parameters)
- **ViT Transformers**: Memory constraints and training instability, limited progress
- **Swin Transformer**: Currently testing as potential solution

**Key Pattern - Training + Validation Plateau**:
- **Not Classic Overfitting**: Both training AND validation accuracy plateau together
- **Model Capacity Issue**: Suggests insufficient architectural capacity for 442-class task
- **Optimization Difficulty**: Models struggle to learn beyond initial convergence

#### Critical Learning Rate Insights

**Learning Rate Stability Analysis**:
- **1e-3**: Causes training instability and crashes across all architectures
- **1e-4**: Default rate, achieves 23-26% ceiling but plateaus
- **1e-5**: More stable but slower convergence, similar ceiling
- **1e-6**: Too conservative, minimal learning progress

#### Memory and Computational Constraints

**Architecture-Specific Limitations**:
- **Swin-Tiny Previous Failures**: Immediate crashes with large batches (>256) or high LR (1e-3)
- **ViT Memory Issues**: OOM errors even with batch=128 for ViT-B/16
- **ResNet Stability**: Most reliable but insufficient capacity for task complexity

### 🔬 **Current Experiment - Swin Transformer Capacity Test**

**Active Run**: snowy-valley-151 (Swin-Tiny)
- **Configuration**: batch=32, lr=1e-4 (default), epochs=100
- **Status**: Successfully running at 24.40 it/s (no crashes!)
- **Hypothesis**: Hierarchical attention may break through capacity ceiling
- **Early Metrics**: Training normally with stable memory usage

**Why Swin May Succeed**:
1. **Hierarchical Processing**: Better for multi-scale constellation patterns
2. **28M Parameters**: 2.5x ResNet18 capacity without extreme overfitting risk
3. **Efficiency**: Shifted windows reduce computational complexity vs standard attention
4. **Proven Architecture**: Strong performance on complex vision tasks

### 📊 **Sweep History Archive**

**Recent Sweep l6rqwlu2 Summary** (11 runs, mostly failed):
- **1 Successful Completion**: decent-sweep-8 (ResNet34) - 24.04% final accuracy
- **1 Currently Running**: hopeful-sweep-11 (ResNet18) - 23.19% at epoch 3
- **9 Failed Runs**: Consistent failures with Swin-Tiny, ViT, and high learning rates

**Failure Analysis**:
- **Architecture Reliability**: ResNet > ViT > Swin (historically)
- **Bayesian Learning**: Optimizer learning to avoid problematic configurations
- **Resource Allocation**: Moved to single runs for better monitoring

### 🎯 **Research Strategy Pivot**

**From Hyperparameter Optimization to Architectural Investigation**:
1. **Phase 1**: Individual architecture testing to find models that can train beyond plateau
2. **Phase 2**: Once viable architecture found, optimize hyperparameters
3. **Phase 3**: Implement curriculum learning on best-performing architecture

**Target Architectures for Testing**:
- ✅ **Swin-Tiny**: Currently testing (snowy-valley-151)
- 🔄 **ResNet50**: Higher capacity ResNet variant
- 🔄 **EfficientNet-B0/B1**: Parameter-efficient alternatives
- 🔄 **ViT-B/32**: Safer transformer option with larger patches

### 💡 **Academic Insights from Capacity Study**

**Methodological Contributions**:
1. **Capacity Ceiling Documentation**: First systematic study of architecture limits for 442-class AMC
2. **Training Pattern Analysis**: Distinction between overfitting vs insufficient capacity
3. **Architecture Reliability Ranking**: Empirical stability hierarchy for constellation tasks
4. **Resource Optimization**: Single-run strategy for architectural exploration

**Implications for Constellation-Based AMC**:
- Traditional computer vision architectures may be suboptimal for constellation patterns
- 442-class joint modulation-SNR classification requires specialized architectural considerations
- Hierarchical attention (Swin) may be better suited than standard convolution or global attention

### 🔄 **Next Steps**

**Immediate Priority**:
1. Monitor Swin-Tiny performance for breakthrough beyond 26%
2. If successful, optimize Swin hyperparameters
3. If unsuccessful, test larger capacity models (ResNet50, EfficientNet)

**Long-term Strategy**:
- Document architecture selection methodology for academic contribution
- Implement curriculum learning on best-performing architecture
- Explore domain-specific architectural modifications for constellation data

---

*This training history documents the evolution from hyperparameter optimization to fundamental architectural capacity investigation, representing a critical pivot in the research methodology.*