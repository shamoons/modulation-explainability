# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a modulation explainability research project that combines multi-task learning with perturbation-based explainability for Automatic Modulation Classification (AMC). The framework addresses both performance and interpretability challenges by transforming I/Q signal data into constellation diagrams and employing a ResNet-based architecture to simultaneously classify modulation schemes and estimate SNR buckets.

### Research Paper Context
This work was submitted as "Constellation Diagram Augmentation and Perturbation-Based Explainability for Automatic Modulation Classification" and introduces novel perturbation-based explainability techniques using the Perturbation Impact Score (PIS) metric to analyze critical regions in constellation diagrams that drive model decisions.

## Technology Stack

- **Language**: Python 3.9
- **Deep Learning**: PyTorch 2.4.1, TorchVision 0.19.1
- **Data Processing**: NumPy (<2.0), Pandas, H5PY, SciPy
- **Visualization**: Matplotlib, Seaborn
- **Experiment Tracking**: Weights & Biases (wandb)
- **Package Management**: Pipenv

## Common Development Commands

```bash
# Install dependencies
pipenv install

# Activate virtual environment
pipenv shell

# Train model with default settings
python src/train_constellation.py

# Train with custom parameters
python src/train_constellation.py \
    --batch_size 64 \
    --snr_list "0,5,10,15,20" \
    --mods_to_process "BPSK,QPSK,8PSK,OOK,8ASK,16QAM,256QAM,FM,GMSK,OQPSK" \
    --epochs 100 \
    --use_snr_buckets True \
    --base_lr 0.0000001 \
    --max_lr 0.0001 \
    --weight_decay 1e-5 \
    --test_size 0.2 \
    --patience 5

# Test model on perturbed and non-perturbed data
python src/test_constellation.py \
    --model_checkpoint <path_to_checkpoint> \
    --data_dir constellation \
    --perturbation_dir <path_to_perturbation_dir>

# Run hyperparameter sweep with W&B
wandb sweep sweep.yml
wandb agent <sweep_id>

# Generate perturbed constellation data
python src/perturb_constellations.py

# Calculate PID metrics
python src/calculate_pid.py
```

## High-Level Architecture

### Multi-Task Learning System
The project implements a multi-task learning approach with:
- **Shared Backbone**: ResNet18/34 extracts common features from constellation images
- **Task-Specific Heads**: 
  - Modulation classification (20 classes: BPSK, QPSK, 8PSK, various QAM, etc.)
  - SNR classification (3 buckets: low [-20 to -4 dB], medium [-2 to 14 dB], high [16 to 30 dB])

### Key Components

1. **Models** (`src/models/`):
   - `ConstellationResNet`: Primary model using ResNet backbone with dual heads
   - `ConstellationVisionTransformer`: Alternative ViT-based architecture

2. **Data Loading** (`src/loaders/`):
   - `ConstellationDataset`: Loads constellation images organized by modulation/SNR
   - `PerturbationDataset`: Handles perturbed images for robustness testing
   - Images are preprocessed to 224x224 and normalized

3. **Loss Functions** (`src/losses/`):
   - Multi-task loss: α × Modulation Loss + β × SNR Loss (α=0.5, β=1.0)
   - `DistancePenaltyCategoricalSNRLoss`: Custom loss that penalizes SNR predictions based on distance from true class

4. **Configuration** (`src/config/`):
   - `loss_weights.json`: Controls multi-task loss weighting
   - `snr_buckets.json`: Defines SNR categorization boundaries

### Data Organization
```
data/
├── RML2016.10a/          # Radio ML datasets
├── RML2018.01A/
└── split_hdf5/           # HDF5 files by modulation type and SNR
constellation/            # Constellation diagram images
perturbed_constellations/ # Perturbed test data
```

### Training Strategy
- Adam optimizer with weight decay (1e-5)
- ReduceLROnPlateau scheduler
- Mixed precision training (AMP)
- Gradient clipping (max_norm=1.0)
- Early stopping based on validation loss

### Testing and Evaluation
The testing pipeline evaluates models on:
- Original constellation images
- Perturbed images (various blackout percentages)
- Generates confusion matrices and F1 score plots
- Results saved to `confusion_matrices/` and `f1_scores/`

## Research Contributions

### Key Innovations
1. **Multi-Task Learning Framework**: Joint prediction of modulation type (20 classes) and SNR buckets, enhancing utility in dynamic wireless environments
2. **Constellation Diagram Augmentation**: Enhanced visual representations using binning, Gaussian smoothing, and normalization techniques
3. **Perturbation-Based Explainability**: Systematic analysis using PIS metric to identify critical constellation regions
4. **Progressive Perturbation Analysis**: Evaluation of classification degradation under varying perturbation levels

### Research Findings
- **High-intensity regions**: Critical for classification (PIS up to 34.8 for 1% brightest pixels)
- **Low-intensity regions**: Minimal impact on performance (PIS as low as 0.95)
- **Multi-task benefits**: Superior SNR prediction and combined accuracy compared to single-task learning
- **Robustness**: Maintains high accuracy across diverse modulation types under challenging noise conditions

## Areas for Future Development (Based on Reviewer Feedback)

### High Priority Improvements
1. **Enhanced Multi-Task Learning**: 
   - Implement task-specific attention mechanisms
   - Add dynamic weighting strategies beyond simple loss combination
   - Consider adversarial training to prevent task interference

2. **SNR Estimation Refinement**:
   - Reduce bucket width from current 16dB bins to 2-3dB for practical applications
   - Consider regression approach instead of classification for precise SNR estimation
   - Compare bucketed vs. precise SNR classification fairly

3. **Improved High-Order Modulation Performance**:
   - Address weak performance on 64QAM (66%) and 256QAM (79%)
   - Implement feature enhancement techniques
   - Add attention mechanisms for complex modulation schemes
   - Explore adaptive data augmentation

### Medium Priority Enhancements
4. **Comprehensive Explainability**:
   - Compare perturbation methods with Grad-CAM and other interpretability techniques
   - Extend beyond pixel intensity to temporal and frequency-domain characteristics
   - Investigate raw signal processing vs. constellation diagram conversion trade-offs

5. **Experimental Validation**:
   - Add comparisons with MCLDNN and CGDNet benchmarks
   - Include training accuracy curves alongside validation curves
   - Verify convergence behavior (current Fig 3b/3c show potential non-convergence)

6. **Implementation Details**:
   - Document α and β parameter selection rationale (currently α=0.5, β=1.0)
   - Clarify "augmentation" terminology usage
   - Improve reproducibility with detailed implementation specifications

## Important Notes

- The system can operate with either raw SNR values or SNR buckets (controlled by `use_snr_buckets` flag)
- Perturbation testing includes various scenarios like top/bottom percentage blackouts
- All experiments are tracked using Weights & Biases
- The project uses constellation diagrams (I/Q plots) converted to images as input
- Current work demonstrates proof-of-concept; reviewer feedback indicates need for more sophisticated multi-task learning and broader explainability comparisons