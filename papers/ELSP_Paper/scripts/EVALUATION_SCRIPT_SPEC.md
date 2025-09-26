# Perturbation Evaluation Script Specification

## Script Name: `evaluate_perturbations.py`

## Purpose
Evaluate the impact of different perturbation types on the canonical ResNet50 model to calculate PIS (Perturbation Impact Score) values for the ELSP paper. This quantifies how masking specific regions of constellation diagrams affects model performance.

## Core Functionality

### 1. Model Loading
- Load canonical checkpoint: `checkpoints/best_model_resnet50_epoch_14.pth`
- Initialize ResNet50 with bottleneck_128 SNR layer architecture
- Set to evaluation mode and move to GPU

### 2. Baseline Performance Establishment
- Load original test set (same indices used in test_canonical_model.py)
- Evaluate model on unperturbed constellation diagrams
- Store baseline accuracies:
  - Combined accuracy: 51.26%
  - Modulation accuracy: 76.39%
  - SNR accuracy: 68.71%
- These must match test_canonical_model.py results exactly for validation

### 3. Perturbation Evaluation Loop
For each perturbation type (15 total):
- Top 1-5% brightest pixels (non-zero only)
- Bottom 1-5% dimmest non-zero pixels
- Random 1-5% pixels (baseline comparison)

Create PerturbationDataset pointing to pre-generated perturbed images
Evaluate model performance on full test set
Record accuracies (combined, modulation, SNR)

### 4. PIS Calculation
For each perturbation:
```
PIS = ΔA / f
where:
  ΔA = accuracy_drop (baseline - perturbed)
  f = fraction_perturbed (percent / 100)
```

Calculate three PIS values:
- PIS_combined (based on combined accuracy)
- PIS_modulation (based on modulation accuracy)
- PIS_snr (based on SNR accuracy)

### 5. Statistical Analysis
- Calculate mean PIS across different percentages for each type
- Identify trends (expected: PIS decreases as percentage increases)
- Compare targeted (top/bottom) vs random perturbations
- Validate that top perturbations have higher impact than bottom

### 6. Per-Class Breakdown (Optional but Valuable)
- Calculate PIS for each modulation type separately
- Calculate PIS for each SNR level separately
- Identify which modulations/SNRs are most sensitive to perturbations

## Input Data Structure

### Original Constellations
```
constellation_diagrams/
├── BPSK/
│   ├── SNR_0/
│   │   ├── grayscale_BPSK_SNR_0_sample_0.png
│   │   └── ...
│   └── SNR_30/
├── QPSK/
└── ... (17 digital modulations total)
```

### Perturbed Constellations
```
perturbed_constellations/
├── BPSK/
│   ├── SNR_0/
│   │   ├── grayscale_BPSK_SNR_0_sample_0_top1_blackout.png
│   │   ├── grayscale_BPSK_SNR_0_sample_0_top2_blackout.png
│   │   ├── ... (15 perturbation variants per image)
```

## Output Data Structure

### Primary Results File: `pis_summary.json`
```json
{
  "baseline_accuracy": {
    "combined": 0.5126,
    "modulation": 0.7639,
    "snr": 0.6871
  },
  "perturbation_results": {
    "top1_blackout": {
      "accuracy": {"combined": 0.48, "modulation": 0.72, "snr": 0.65},
      "pis": {"combined": 32.6, "modulation": 43.9, "snr": 37.1},
      "accuracy_drop": {"combined": 0.0326, "modulation": 0.0439, "snr": 0.0371}
    },
    "top5_blackout": { ... },
    "bottom1_blackout": { ... },
    "random1_blackout": { ... }
  },
  "key_findings": {
    "top1_pis": 32.6,
    "top5_pis": 7.2,
    "bottom1_pis": 0.8,
    "bottom5_pis": 0.3,
    "top_vs_random_ratio": 5.4
  }
}
```

### Detailed Results: `detailed_results.json`
- Full evaluation metrics for each perturbation
- Per-class F1 scores
- Confusion matrices
- Processing timestamps

### Visualization Outputs
1. `perturbation_impact_chart.png`: PIS vs perturbation percentage
2. `accuracy_degradation_curves.png`: Accuracy curves for all types
3. `pis_comparison_bar_chart.png`: Side-by-side PIS comparison
4. `example_perturbations/`: Visual examples for paper

## Key Values for Paper

The script should extract and highlight:
1. **Top 1% PIS**: Expected ~30-40 (high impact from constellation centers)
2. **Top 5% PIS**: Expected ~15-25 (diminishing returns)
3. **Bottom 1% PIS**: Expected <1.0 (noise pixels have minimal impact)
4. **Bottom 5% PIS**: Expected <1.0 (confirms noise rejection)
5. **Random baseline PIS**: For comparison (should be between top and bottom)

## Implementation Considerations

### Memory Management
- Process perturbations sequentially, not in parallel
- Clear GPU cache between evaluations
- Use DataLoader with appropriate batch size (256 recommended)

### Reproducibility
- Set all random seeds (model, numpy, torch)
- Use same test indices as test_canonical_model.py
- Save configuration used for evaluation

### Error Handling
- Verify all perturbation files exist before starting
- Check for NaN/inf in PIS calculations
- Handle edge cases (0% accuracy scenarios)

### Progress Tracking
- Use tqdm for evaluation progress
- Print intermediate results after each perturbation type
- Save partial results frequently

## Expected Runtime
- ~10 minutes per perturbation type
- Total: ~2.5 hours for complete evaluation
- Can be interrupted and resumed using partial results

## Validation Checks

1. **Baseline Match**: Original accuracy must match test_canonical_model.py exactly
2. **Monotonicity**: PIS should generally decrease as perturbation % increases
3. **Ordering**: top_pis > random_pis > bottom_pis
4. **Sanity**: No negative PIS values (accuracy should drop, not improve)
5. **Completeness**: All 15 perturbation types evaluated successfully

## Command Line Interface
```bash
uv run python papers/ELSP_Paper/scripts/evaluate_perturbations.py \
    --checkpoint checkpoints/best_model_resnet50_epoch_14.pth \
    --perturbation-dir perturbed_constellations \
    --output-dir papers/ELSP_Paper/results/perturbation_analysis \
    --batch-size 256 \
    --seed 42
```

## Integration with Paper

Replace these placeholders in AMC_Constellation_Paper.tex:
- Line 312: Insert actual top 5% accuracy drop and PIS
- Line 315: Insert actual bottom perturbation PIS
- Line 393/395: Update with final PIS ranges
- Line 435: Insert summary PIS values

The script should generate LaTeX-formatted snippets for easy copy-paste.