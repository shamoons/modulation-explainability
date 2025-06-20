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
    --model_type vit \
    --batch_size 32 \
    --snr_list "0,10,20" \
    --mods_to_process "BPSK,QPSK,8PSK" \
    --epochs 50 \
    --base_lr 1e-4 \
    --weight_decay 1e-5 \
    --test_size 0.2 \
    --patience 10

# Train with different architectures
uv run python src/train_constellation.py --model_type resnet18  # Default, fastest
uv run python src/train_constellation.py --model_type resnet34  # Deeper ResNet
uv run python src/train_constellation.py --model_type vit       # Vision Transformer

# Resume training from checkpoint
uv run python src/train_constellation.py --checkpoint path/to/checkpoint.pth

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

# Test core training components (verification)
uv run python test_core_training_components.py

# Test with HDF5 data directly
uv run python test_hdf5_training.py
```

## High-Level Architecture

### Enhanced Multi-Task Learning System
The project implements a **state-of-the-art multi-task learning approach** with:
- **Flexible Backbone Architectures**: ResNet18/34 or Vision Transformer (ViT) for feature extraction
- **Task-Specific Heads**: 
  - Modulation classification (17 digital classes by default)
  - **Discrete SNR prediction** (26 classes: -20 to +30 dB in 2dB intervals)
- **Analytical Uncertainty Weighting**: SOTA 2024 method that automatically balances task losses using learned uncertainty parameters

### Key Components

1. **Models** (`src/models/`):
   - `ConstellationResNet`: Enhanced ResNet18/34 backbone with dual heads supporting discrete SNR classes
   - `ConstellationVisionTransformer`: Vision Transformer (ViT-B/16) architecture for complex pattern recognition
   - **Model Selection**: Choose architecture via `--model_type` (resnet18, resnet34, vit)
   - **Performance Trade-offs**: ResNet faster (~94-99 it/s), ViT slower (~3.8-4.0 it/s) but potentially better representation learning
   - **Default Dataset**: Excludes analog modulations (AM-DSB-SC, AM-DSB-WC, AM-SSB-SC, AM-SSB-WC, FM, GMSK, OOK)

2. **Data Loading** (`src/loaders/`):
   - `ConstellationDataset`: Loads constellation images organized by modulation/SNR
   - `SplitHDF5Dataset`: Direct loader for split HDF5 data format
   - `PerturbationDataset`: Handles perturbed images for robustness testing
   - Images are preprocessed to 224x224 and normalized

3. **Enhanced Loss Functions** (`src/losses/uncertainty_weighted_loss.py`):
   - **`AnalyticalUncertaintyWeightedLoss`**: SOTA uncertainty-based multi-task weighting
   - **`DistancePenalizedSNRLoss`**: Distance-aware loss for discrete SNR prediction
   - Replaces traditional α/β weighting with learned uncertainty parameters

4. **Data Pipeline**:
   - **Split HDF5**: Pre-organized data by modulation/SNR in `data/split_hdf5/`
   - **Constellation Conversion**: `convert_to_constellation.py` transforms I/Q to images
   - **Training Pipeline**: Direct training from constellation images or HDF5 data

### Data Organization
```
data/
├── RML2016.10a/          # Radio ML datasets
├── RML2018.01A/
└── split_hdf5/           # HDF5 files by modulation type and SNR
constellation/            # Constellation diagram images
perturbed_constellations/ # Perturbed test data
```

### Enhanced Training Strategy
- **Adaptive Multi-Task Learning**: Uncertainty weighting automatically balances modulation and SNR losses
- Adam optimizer with weight decay (1e-5) including uncertainty parameters
- ReduceLROnPlateau scheduler (patience-based)
- **Device-Adaptive Training**: CUDA mixed precision for GPU, optimized MPS/CPU training
- Gradient clipping (max_norm=1.0) for all model and uncertainty parameters
- Early stopping based on validation loss
- **Discrete SNR Training**: 26-class SNR prediction with distance-based penalties

### Testing and Evaluation
The testing pipeline evaluates models on:
- Original constellation images
- Perturbed images (various blackout percentages)
- Generates confusion matrices and F1 score plots
- Results saved to `confusion_matrices/` and `f1_scores/`

## Research Contributions

### Key Innovations
1. **Multi-Task Learning Framework**: Joint prediction of modulation type (17 digital classes by default) and SNR levels (26 classes), enhancing utility in dynamic wireless environments
2. **Constellation Diagram Augmentation**: Enhanced visual representations using binning, Gaussian smoothing, and normalization techniques
3. **Perturbation-Based Explainability**: Systematic analysis using PIS metric to identify critical constellation regions
4. **Progressive Perturbation Analysis**: Evaluation of classification degradation under varying perturbation levels

### Research Findings
- **High-intensity regions**: Critical for classification (PIS up to 34.8 for 1% brightest pixels)
- **Low-intensity regions**: Minimal impact on performance (PIS as low as 0.95)
- **Multi-task benefits**: Superior SNR prediction and combined accuracy compared to single-task learning
- **Robustness**: Maintains high accuracy across diverse modulation types under challenging noise conditions

## Recent Enhancements (2024 Updates)

### ✅ Completed High Priority Improvements
1. **✅ Enhanced Multi-Task Learning**: 
   - **IMPLEMENTED**: Analytical uncertainty-based weighting using 2024 SOTA methods
   - **IMPLEMENTED**: Dynamic task balancing that adapts during training
   - **IMPLEMENTED**: Learned uncertainty parameters that prevent task interference

2. **✅ SNR Estimation Refinement**:
   - **IMPLEMENTED**: Discrete SNR prediction with 26 classes (-20 to +30 dB in 2dB intervals)  
   - **IMPLEMENTED**: Distance-penalized loss function for accurate SNR classification
   - **REMOVED**: SNR bucket system entirely - now uses precise discrete prediction
   - **VERIFIED**: Training pipeline working with enhanced multi-task learning

## Areas for Future Development (Based on Reviewer Feedback)

### High Priority Improvements

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

## Important Implementation Notes

### Training System
- **Multi-Task Learning**: Now uses analytical uncertainty weighting (no manual α/β tuning required)
- **SNR Prediction**: System now operates with discrete 26-class SNR prediction (removed SNR buckets)
- **Device Support**: Optimized for CUDA (with mixed precision), MPS (Apple Silicon), and CPU
- **Package Management**: Uses UV instead of Pipenv for faster dependency management

### Data Pipeline
- **HDF5 Data**: Pre-split by modulation/SNR in `data/split_hdf5/` directory
- **Constellation Images**: Generated from HDF5 using `convert_to_constellation.py`
- **Training Data**: Can train directly from constellation images or HDF5 data
- **Image Format**: 224x224 grayscale constellation diagrams
- **Default Modulations**: Digital only (17 classes) - analog modulations excluded by default

### Key Features
- ✅ **Multi-Architecture Support**: ResNet18/34 and Vision Transformer (ViT) models
- ✅ **Uncertainty Weighting**: Automatically balances modulation vs. SNR loss
- ✅ **Discrete SNR Classes**: 26 classes from -20 to +30 dB (2dB intervals)
- ✅ **Distance-Penalized Loss**: SNR predictions penalized based on distance from true class
- ✅ **Device-Adaptive**: No CUDA-specific warnings on MPS/CPU devices
- ✅ **Verified Training**: Core pipeline tested and working with real constellation data

### Verification Status
- ✅ Core training components tested and verified
- ✅ HDF5 data loading pipeline working
- ✅ Enhanced multi-task learning active and learning
- ✅ Multi-architecture support (ResNet18/34, ViT) implemented and tested
- ✅ Device compatibility (CUDA/MPS/CPU) implemented
- ✅ Constellation image generation from split HDF5 data functional

## Current Training Configuration

### Updated Default Parameters (Dec 2024)
The training script now uses optimized defaults for full dataset training:
- **Batch Size**: 32 (memory-efficient for large dataset)
- **Learning Rate**: 1e-4 (increased from 1e-7 for faster convergence)
- **Epochs**: 50 (reasonable for initial full training)
- **Patience**: 10 (more stable for large dataset)
- **Dataset**: 17 digital modulations × 26 SNRs (442 classes total by default)
- **Test Split**: 20% validation

### Dataset Statistics
- **Digital Modulation Classes**: 17 (excludes 7 analog modulations by default)
- **Analog Modulations Excluded**: AM-DSB-SC, AM-DSB-WC, AM-SSB-SC, AM-SSB-WC, FM, GMSK, OOK
- **SNR Classes**: 26 (-20 to +30 dB in 2dB intervals)
- **Samples per Mod/SNR**: 4096 (excellent coverage)
- **Total Dataset Size**: ~1.8M samples (digital only)
- **Training Speed**: ~8-10 it/s on Apple M-series (MPS)
- **Estimated Training Time**: ~2 hours per epoch

### W&B Integration
- **Project**: modulation-explainability
- **Entity**: shamoons
- **MCP Integration**: Installed for advanced monitoring and analysis
- **Install Command**: `claude mcp add wandb -e WANDB_API_KEY=your-key -- uvx --from git+https://github.com/wandb/wandb-mcp-server wandb_mcp_server`

#### Efficient W&B Run Monitoring Commands
For quick and comprehensive run analysis, use these MCP queries:

```bash
# Get comprehensive run overview (all metrics, system info, config)
mcp__wandb__query_wandb_tool(
    query="""
    query GetRunDetails($entity: String!, $project: String!, $runId: String!) {
        project(name: $project, entityName: $entity) {
            run(name: $runId) {
                id name displayName state
                createdAt updatedAt heartbeatAt
                config summaryMetrics
                historyKeys
                systemMetrics
                tags { name }
            }
        }
    }""",
    variables={"entity": "shamoons", "project": "modulation-explainability", "runId": "eujpigwb"}
)

# Get training history for specific metrics
mcp__wandb__query_wandb_tool(
    query="""
    query GetRunHistory($entity: String!, $project: String!, $runId: String!, $specs: [JSONString!]!) {
        project(name: $project, entityName: $entity) {
            run(name: $runId) {
                id name
                sampledHistory(specs: $specs) {
                    step timestamp item
                }
            }
        }
    }""",
    variables={
        "entity": "shamoons", 
        "project": "modulation-explainability", 
        "runId": "eujpigwb",
        "specs": ["{\"keys\": [\"loss\", \"mod_accuracy\", \"snr_accuracy\", \"learning_rate\"]}"]
    }
)
```

### Recent Training Sessions
- **Current Run**: treasured-waterfall-89 (eujpigwb)
- **Status**: Active multi-task learning with uncertainty weighting (digital modulations only)
- **Digital Classes**: 17 modulation types (analog excluded by default)
- **Expected Final Performance**: Mod accuracy ~85-95%, SNR accuracy ~60-80%