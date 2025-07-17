# ELSP Paper Results Organization

This directory contains all experimental results and artifacts for the ELSP paper on Joint Modulation-SNR Classification with Perturbation-Based Explainability.

## Directory Structure

```
results/
├── checkpoints/                    # Model checkpoints
│   ├── canonical_model_epoch_14.pth    # Best model from run lmp0536i
│   └── checkpoint_metadata.json        # Checkpoint information
├── confusion_matrices/             # Confusion matrices
│   ├── modulation_confusion_matrix.png
│   ├── snr_confusion_matrix.png
│   ├── combined_confusion_matrix.png
│   └── confusion_matrix_data.json
├── f1_scores/                      # F1 score analyses
│   ├── modulation_f1_scores.json
│   ├── snr_f1_scores.json
│   ├── f1_by_snr_level.json
│   └── f1_score_plots.png
├── performance_metrics/            # Core performance data
│   ├── test_set_results.json
│   ├── validation_results.json
│   ├── training_curves.json
│   └── statistical_significance.json
├── figures/                        # Publication-ready figures
│   ├── architecture_diagram.png
│   ├── training_curves.png
│   ├── performance_comparison.png
│   └── constellation_examples.png
├── tables/                         # Publication-ready tables
│   ├── performance_comparison_table.tex
│   ├── ablation_study_results.tex
│   └── architecture_comparison.tex
├── raw_data/                       # Raw experimental data
│   ├── test_predictions.npy
│   ├── test_labels.npy
│   └── model_outputs.json
├── perturbation_analysis/          # Explainability results
│   ├── heatmaps/
│   │   ├── modulation_heatmaps/
│   │   └── snr_heatmaps/
│   ├── pis_scores/
│   │   ├── pis_by_modulation.json
│   │   └── pis_by_snr.json
│   └── constellation_regions/
│       ├── important_regions.json
│       └── region_visualizations.png
└── ablation_studies/               # Ablation study results
    ├── curriculum_learning/
    │   ├── with_curriculum_results.json
    │   └── without_curriculum_results.json
    ├── snr_preprocessing/
    │   ├── power_normalization_results.json
    │   └── per_image_normalization_results.json
    └── architecture_comparison/
        ├── bottleneck_128_results.json
        ├── bottleneck_64_results.json
        └── standard_results.json
```

## Key Files Expected

### Phase 3A: Test Set Evaluation
- `performance_metrics/test_set_results.json`: Combined, modulation, SNR accuracies
- `confusion_matrices/`: All confusion matrices with publication-ready formatting
- `f1_scores/`: Detailed F1 analysis by modulation type and SNR level
- `checkpoints/canonical_model_epoch_14.pth`: Best model checkpoint

### Phase 3B: Perturbation Analysis
- `perturbation_analysis/heatmaps/`: Constellation heatmaps showing important regions
- `perturbation_analysis/pis_scores/`: PIS metric quantification
- `perturbation_analysis/constellation_regions/`: Important region analysis

### Phase 3C: Ablation Studies
- `ablation_studies/curriculum_learning/`: With/without curriculum comparison
- `ablation_studies/snr_preprocessing/`: Power vs per-image normalization
- `ablation_studies/architecture_comparison/`: Different SNR layer architectures

### Publication Assets
- `figures/`: All publication-ready figures in high resolution
- `tables/`: LaTeX-formatted tables for direct inclusion in paper
- `raw_data/`: Raw experimental data for reproducibility

## Usage Notes

1. All JSON files should include metadata (timestamp, run ID, configuration)
2. Figure files should be saved in both PNG (for review) and PDF (for publication)
3. Raw data should be saved in compressed formats when possible
4. All results should be tagged with the corresponding W&B run ID for traceability

## Canonical Model Information

**Run ID**: lmp0536i (dainty-snow-230)
**Epoch**: 14
**Validation Accuracy**: 51.03% (76.14% mod, 68.65% SNR)
**Configuration**: ResNet50 + bottleneck_128 + curriculum learning
**Checkpoint Location**: To be saved from W&B run lmp0536i epoch 14