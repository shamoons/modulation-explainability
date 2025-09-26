# PERTURBATION_TODO.md

## Phase 3B: Perturbation-Based Explainability Framework Implementation

### Overview
This document outlines the comprehensive approach for implementing the perturbation analysis framework to calculate PIS (Perturbation Impact Score) values for the ELSP paper. The framework will analyze how masking different regions of constellation diagrams affects model performance.

## Current Project State

### Existing Code Assets
1. **Perturbation Generation**: `src/perturb_constellations.py`
   - Creates top/bottom/random perturbations
   - Supports multiple percentage levels (1%, 2%, 5%, 10%)
   - Multiprocessing for efficient generation
   - Handles digital modulations only (excludes analog)

2. **Perturbation Loader**: `src/loaders/perturbation_loader.py`
   - PerturbationDataset class extending ConstellationDataset
   - Handles pre-perturbed images
   - Maintains same structure as original dataset

3. **PID Calculation**: `src/calculate_pid.py`
   - Calculates area differences between original and perturbed
   - Currently focused on physical area changes, not PIS

4. **Canonical Model**: `checkpoints/best_model_resnet50_epoch_14.pth`
   - ResNet50 with bottleneck_128 SNR layer
   - Test accuracy: 51.26% combined (76.39% mod, 68.71% SNR)

### Data Structure
- **Original constellations**: `constellation_diagrams/`
  - Structure: `{modulation}/SNR_{value}/grayscale_{modulation}_SNR_{value}_sample_{id}.png`
  - 17 digital modulations × 16 SNR levels (0-30 dB) × 4096 samples
  - 1-channel grayscale, 224×224 pixels

- **Perturbed constellations**: `perturbed_constellations/`
  - Same structure with suffix: `_{perturbation_type}.png`
  - Types: `top1_blackout`, `top5_blackout`, `bottom1_blackout`, etc.

## Implementation Plan

### Phase 1: Script Creation - `evaluate_perturbations.py`

```python
#!/usr/bin/env python3
"""
Evaluate perturbation impact on canonical model (Phase 3B).

This script:
1. Loads the canonical model checkpoint
2. Evaluates performance on original test set
3. Evaluates performance on each perturbation type
4. Calculates PIS scores
5. Generates visualizations and analysis

Usage:
    uv run python papers/ELSP_Paper/scripts/evaluate_perturbations.py
"""

# Key components:
# 1. Model loading (reuse from test_canonical_model.py)
# 2. Dataset creation for each perturbation type
# 3. Evaluation loop across perturbation types and percentages
# 4. PIS calculation: PIS = ΔA/f where ΔA = accuracy drop, f = fraction perturbed
# 5. Results aggregation and visualization
```

### Phase 2: Core Implementation Details

#### 2.1 Perturbation Types to Evaluate
```python
PERTURBATION_CONFIGS = [
    # High-intensity perturbations
    {'type': 'top1_blackout', 'percent': 1, 'description': '1% brightest pixels'},
    {'type': 'top2_blackout', 'percent': 2, 'description': '2% brightest pixels'},
    {'type': 'top3_blackout', 'percent': 3, 'description': '3% brightest pixels'},
    {'type': 'top4_blackout', 'percent': 4, 'description': '4% brightest pixels'},
    {'type': 'top5_blackout', 'percent': 5, 'description': '5% brightest pixels'},
    
    # Low-intensity perturbations
    {'type': 'bottom1_blackout', 'percent': 1, 'description': '1% dimmest non-zero pixels'},
    {'type': 'bottom2_blackout', 'percent': 2, 'description': '2% dimmest non-zero pixels'},
    {'type': 'bottom3_blackout', 'percent': 3, 'description': '3% dimmest non-zero pixels'},
    {'type': 'bottom4_blackout', 'percent': 4, 'description': '4% dimmest non-zero pixels'},
    {'type': 'bottom5_blackout', 'percent': 5, 'description': '5% dimmest non-zero pixels'},
    
    # Random perturbations (baseline)
    {'type': 'random1_blackout', 'percent': 1, 'description': '1% random pixels'},
    {'type': 'random2_blackout', 'percent': 2, 'description': '2% random pixels'},
    {'type': 'random3_blackout', 'percent': 3, 'description': '3% random pixels'},
    {'type': 'random4_blackout', 'percent': 4, 'description': '4% random pixels'},
    {'type': 'random5_blackout', 'percent': 5, 'description': '5% random pixels'},
]
```

#### 2.2 Evaluation Function
```python
def evaluate_perturbation(model, perturbation_type, test_indices, dataset_config):
    """Evaluate model on a specific perturbation type."""
    # Create PerturbationDataset
    dataset = PerturbationDataset(
        root_dir='constellation_diagrams',
        perturbation_dir='perturbed_constellations',
        perturbation_type=perturbation_type,
        image_type='grayscale',
        snr_list=SNR_LEVELS,
        mods_to_process=DIGITAL_MODULATIONS
    )
    
    # Create test loader with same indices as original
    test_loader = DataLoader(
        dataset,
        batch_size=256,
        sampler=SubsetRandomSampler(test_indices),
        num_workers=4,
        pin_memory=True
    )
    
    # Evaluate (reuse evaluate_model function)
    results = evaluate_model(model, test_loader)
    return results
```

#### 2.3 PIS Calculation
```python
def calculate_pis(original_accuracy, perturbed_accuracy, perturbation_fraction):
    """
    Calculate Perturbation Impact Score.
    PIS = ΔA/f where:
    - ΔA = accuracy drop (original - perturbed)
    - f = fraction of pixels perturbed
    """
    accuracy_drop = original_accuracy - perturbed_accuracy
    pis = accuracy_drop / (perturbation_fraction / 100)  # Convert percent to fraction
    return {
        'accuracy_drop': accuracy_drop,
        'pis': pis,
        'relative_drop_percent': (accuracy_drop / original_accuracy) * 100
    }
```

### Phase 3: Running the Analysis

#### 3.1 Pre-flight Checks
1. **Generate perturbations** (if not already done):
   ```bash
   uv run python src/perturb_constellations.py \
       --percents 1 2 3 4 5 \
       --random \
       --source constellation_diagrams \
       --output perturbed_constellations
   ```

2. **Verify data integrity**:
   - Check that all perturbation types exist
   - Verify same number of samples as original
   - Spot-check a few images visually

#### 3.2 Execute Evaluation
```bash
uv run python papers/ELSP_Paper/scripts/evaluate_perturbations.py
```

Expected runtime: ~2-3 hours for full evaluation (15 perturbation types × test set)

#### 3.3 Expected Outputs
```
papers/ELSP_Paper/results/perturbation_analysis/
├── pis_summary.json                    # Main PIS values for paper
├── detailed_results.json                # Full evaluation results
├── perturbation_impact_chart.png        # Visualization of PIS vs perturbation %
├── accuracy_degradation_curves.png      # Accuracy vs perturbation level
├── modulation_specific_analysis.json    # Per-modulation PIS breakdown
├── snr_specific_analysis.json          # Per-SNR PIS breakdown
└── example_perturbations/              # Sample perturbed images for paper
    ├── original_bpsk_10db.png
    ├── top1_bpsk_10db.png
    ├── top5_bpsk_10db.png
    ├── bottom1_bpsk_10db.png
    └── bottom5_bpsk_10db.png
```

### Phase 4: Integration into Paper

#### 4.1 Key Values to Extract
1. **Primary PIS values**:
   - Top 1% brightest: Expected ~30-40
   - Top 5% brightest: Expected ~15-25
   - Bottom 1% dimmest: Expected <1.0
   - Bottom 5% dimmest: Expected <1.0

2. **Accuracy drops**:
   - Modulation accuracy drop for top 5%
   - SNR accuracy drop for top 5%
   - Combined accuracy drop trends

3. **Comparative analysis**:
   - Top vs bottom perturbations
   - Targeted vs random perturbations
   - Per-modulation sensitivity variations

#### 4.2 Paper Updates
Replace TBD placeholders in `AMC_Constellation_Paper.tex`:
- Line 312: "modulation accuracy dropping from 76.39% to [TBD]%"
- Line 312: "PIS of [TBD]"
- Line 315: "PIS as low as [TBD]"
- Line 393: "PIS values up to 34.8" → actual value
- Line 395: "PIS < 1.0" → actual value
- Line 435: "PIS up to [TBD]"
- Line 435: "PIS < [TBD]"

### Phase 5: Visualization Requirements

#### 5.1 Main Figure - Perturbation Impact Analysis
- X-axis: Perturbation percentage (1-5%)
- Y-axis: PIS score
- Three lines: Top pixels, Bottom pixels, Random pixels
- Shows diminishing returns as percentage increases

#### 5.2 Accuracy Degradation Curves
- X-axis: Perturbation percentage
- Y-axis: Accuracy (separate for modulation/SNR/combined)
- Multiple lines for different perturbation types

#### 5.3 Example Perturbations Grid
- 3×5 grid showing:
  - Row 1: Original constellation
  - Row 2: Top 1%, 2%, 3%, 4%, 5% perturbations
  - Row 3: Bottom 1%, 2%, 3%, 4%, 5% perturbations
- Use high-SNR example (e.g., QPSK at 20 dB) for clarity

## Validation & Quality Checks

1. **Sanity checks**:
   - Original evaluation matches test_canonical_model.py results
   - Random perturbations have consistent impact across seeds
   - PIS decreases as perturbation percentage increases

2. **Statistical validity**:
   - Run evaluation 3 times with different random seeds
   - Report mean ± std for PIS values
   - Ensure consistent trends

3. **Edge cases**:
   - Very low SNR (-20 to 0 dB) where noise dominates
   - High-order modulations (256QAM) with dense constellations
   - Verify no division by zero in PIS calculation

## Timeline

1. **Day 1**: 
   - Create `evaluate_perturbations.py` script
   - Run perturbation generation if needed
   - Initial testing with subset of data

2. **Day 2**:
   - Full evaluation run
   - Generate all visualizations
   - Extract key PIS values

3. **Day 3**:
   - Update paper with actual values
   - Create publication-quality figures
   - Final validation and checks

## Notes & Considerations

1. **Memory management**: 
   - Load perturbations on-demand, not all at once
   - Clear GPU cache between evaluations
   - Monitor system resources

2. **Reproducibility**:
   - Set all random seeds
   - Document exact commands used
   - Save intermediate results frequently

3. **Paper narrative**:
   - Emphasize that high-intensity regions = constellation points
   - Low-intensity impact validates noise rejection
   - PIS provides quantitative explainability metric

4. **Future extensions**:
   - Gradient-based attribution methods
   - Perturbation patterns beyond uniform masking
   - Cross-modulation perturbation analysis