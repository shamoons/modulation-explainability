# Modulation Explainability

Enhanced multi-task learning framework for Automatic Modulation Classification (AMC) with perturbation-based explainability.

## Overview

This project implements a state-of-the-art multi-task learning approach for wireless signal classification, featuring:
- **Automatic Modulation Classification**: Identifies 17 digital modulation types (analog excluded by default)
- **Discrete SNR Estimation**: Predicts SNR values from -20 to +30 dB (26 classes)
- **Multi-Architecture Support**: ResNet18/34 and Vision Transformer (ViT) models
- **Analytical Uncertainty Weighting**: Automatically balances task losses during training
- **Perturbation-Based Explainability**: Analyzes critical constellation regions using PIS metric

## Installation

This project uses [uv](https://github.com/astral-sh/uv) for fast, reliable Python package management.

```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone the repository
git clone https://github.com/yourusername/modulation-explainability.git
cd modulation-explainability

# Install dependencies
uv sync
```

## Dataset Preparation

### 1. Convert HDF5 to Constellation Images

Transform raw I/Q signal data into constellation diagram images:

```bash
# Convert all modulations and SNRs (default)
uv run python src/convert_to_constellation.py

# Convert specific modulations and SNRs
uv run python src/convert_to_constellation.py \
    --h5_dir data/split_hdf5 \
    --snr_list -20,-18,-16,0,10,20,30 \
    --mod_list BPSK,QPSK,8PSK,16PSK
```

## Training

### Basic Training

```bash
# Train with default settings (digital modulations only, ResNet18)
uv run python src/train_constellation.py

# Train with specific model architecture
uv run python src/train_constellation.py --model_type vit      # Vision Transformer
uv run python src/train_constellation.py --model_type resnet34 # Deeper ResNet

# Train with specific parameters
uv run python src/train_constellation.py \
    --model_type vit \
    --batch_size 32 \
    --epochs 50 \
    --base_lr 1e-4 \
    --weight_decay 1e-5 \
    --patience 10
```

### Advanced Training Options

```bash
# Train on subset of modulations/SNRs with ViT
uv run python src/train_constellation.py \
    --model_type vit \
    --mods_to_process "BPSK,QPSK,8PSK" \
    --snr_list "0,10,20" \
    --test_size 0.2

# Resume from checkpoint (model type automatically detected)
uv run python src/train_constellation.py \
    --checkpoint path/to/checkpoint.pth

# Include analog modulations (override default exclusion)
uv run python src/train_constellation.py \
    --mods_to_process "BPSK,QPSK,8PSK,AM-DSB-SC,FM" \
    --model_type resnet34
```

### Hyperparameter Sweep with Weights & Biases

```bash
# Run sweep (requires W&B account)
wandb sweep sweep.yml
wandb agent <sweep_id>
```

## Testing and Evaluation

### Test Trained Model

```bash
# Test on original constellation images
uv run python src/test_constellation.py \
    --model_checkpoint <path_to_checkpoint> \
    --data_dir constellation

# Test with perturbed images
uv run python src/test_constellation.py \
    --model_checkpoint <path_to_checkpoint> \
    --data_dir constellation \
    --perturbation_dir perturbed_constellations
```

## Explainability Analysis

### 1. Generate Perturbed Constellations

Create systematically perturbed versions of constellation images:

```bash
# Generate with default settings (1%, 5%, 10% perturbations)
uv run python src/perturb_constellations.py

# Custom perturbation percentages
uv run python src/perturb_constellations.py \
    --percents 1 5 10 20 \
    --random  # Include random perturbations as baseline

# Use more CPU cores for faster processing
uv run python src/perturb_constellations.py \
    --workers 16  # Adjust based on your CPU
```

### 2. Calculate Perturbation Impact Score (PIS)

Analyze the impact of different perturbation types:

```bash
uv run python src/calculate_pid.py \
    --original_dir test_results/original \
    --perturbed_dir test_results/perturbed \
    --output_file perturbation_analysis.json
```

## Visualization Tools

### Generate Sample Visualizations

```bash
# Create scatter plot constellation diagrams
uv run python src/save_samples.py

# Create time-domain and constellation plots
uv run python src/visualization.py
```

## Model Architecture

The framework supports multiple architectures with:
- **Flexible Backbones**: ResNet18/34 or Vision Transformer (ViT-B/16) for feature extraction
- **Dual Task Heads**: 
  - Modulation classification (17 digital classes by default)
  - SNR estimation (26 discrete classes)
- **Uncertainty Weighting**: Automatic task balancing using learned parameters

### Model Performance Characteristics
- **ResNet18**: Fastest training (~94-99 it/s), good baseline performance
- **ResNet34**: Deeper model, moderate speed, potentially better accuracy
- **Vision Transformer**: Slowest (~3.8-4.0 it/s) but advanced attention mechanisms

## Key Features

### Analytical Uncertainty Weighting
Based on Liu et al. 2024 (arXiv:2408.07985), this method automatically balances multi-task losses without manual hyperparameter tuning.

### Distance-Penalized SNR Loss
Discrete SNR prediction with penalties proportional to prediction distance from true value, maintaining ordinal relationships.

### Perturbation-Based Explainability
Systematic pixel masking analysis revealing:
- **Top N% Brightest**: Critical signal regions
- **Bottom N% Dimmest**: Background/noise regions  
- **Random N%**: Baseline comparison

## Output Structure

```
modulation-explainability/
├── constellation/              # Generated constellation images
│   ├── BPSK/
│   │   ├── SNR_-20/
│   │   └── ...
│   └── ...
├── perturbed_constellations/   # Perturbation analysis images
├── checkpoints/                # Saved model checkpoints
├── confusion_matrices/         # Test results
└── f1_scores/                  # Performance metrics
```

## Configuration Files

- `pyproject.toml`: Project dependencies and metadata
- `sweep.yml`: W&B hyperparameter sweep configuration
- `CLAUDE.md`: Project documentation and instructions

## Dependencies

Core dependencies managed by uv:
- PyTorch (≥2.0.0)
- TorchVision (≥0.15.0)
- NumPy (<2.0)
- Matplotlib
- Weights & Biases
- H5PY
- Pillow
- scikit-learn
- tqdm

## Troubleshooting

### CUDA/GPU Issues
```bash
# Force CPU usage
uv run python src/train_constellation.py --device cpu

# Check CUDA availability
uv run python -c "import torch; print(torch.cuda.is_available())"
```

### Memory Issues
- Reduce batch size: `--batch_size 16`
- Use gradient accumulation
- Process fewer modulations/SNRs at once
- Switch to ResNet from ViT: `--model_type resnet18`

### Slow Training
- Use ResNet instead of ViT for faster training
- Enable mixed precision training (automatic on CUDA)
- Use larger batch sizes if memory allows
- Ensure data is on fast storage (SSD)

### Model Selection Guidelines
- **Quick experiments**: Use `--model_type resnet18`
- **Best performance**: Try `--model_type resnet34` or `--model_type vit`
- **Complex modulations**: Consider ViT for attention-based feature learning

## Citation

If you use this code in your research, please cite:

```bibtex
@article{siddiqui2024constellation,
  title={Enhanced Multi-Task Learning with Analytical Uncertainty Weighting for Automatic Modulation Classification and Discrete SNR Estimation},
  author={Siddiqui, Shamoon},
  journal={IEEE Transactions on Wireless Communications},
  year={2024}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.