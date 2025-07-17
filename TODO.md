# ELSP Paper TODO: "Joint Modulation-SNR Classification with Perturbation-Based Explainability"

## Project Context

**Paper Topic**: Joint modulation-SNR classification for automatic modulation classification (AMC) using constellation diagrams with perturbation-based explainability

**Key Innovation**: First comprehensive study of simultaneous 272-class classification (17 modulations × 16 SNR levels) without oracle SNR or cascade approaches

**Best Result**: 51.03% validation accuracy (76.14% mod, 68.65% SNR) - Run lmp0536i at epoch 14

**Target Journal**: ELSP (Engineering, Life Sciences & Physical Sciences)

**Authors**: Shamoon Siddiqui (siddiq76@rowan.edu), Ravi Ramachandran (ravi@rowan.edu), Rowan University

**No funding to acknowledge**

## PHASE 1: Literature Review and Bibliography Enhancement

### Phase 1A: Core Bibliography Development
- [ ] Add cascade AMC papers showing oracle SNR limitations
  - [ ] Zhang et al. (2023): Multi-modal modulation recognition with SNR segmentation
  - [ ] Chen et al. (2024): Pre-classification and SNR-specific compensation
  - [ ] Wang et al. (2024): WCTFormer with WiFi channel state information
- [ ] Add joint prediction attempts for comparison
  - [ ] Liu & Wong (2022): Joint modulation and SNR classification (220-class baseline)
  - [ ] Chen et al. (2021): End-to-end joint prediction
- [ ] Add constellation-based methods establishing SNR range precedents
  - [ ] Peng et al. (2023): Constellation diagrams in practical SNR ranges (0-30 dB)
  - [ ] García-López et al. (2024): Ultralight signal classification (0-20 dB)
  - [ ] Gao et al. (2023): Robust constellation diagram representation

### Phase 1B: XAI and Perturbation Analysis Literature
- [ ] Add perturbation-based XAI methods for signal processing
  - [ ] Perturbation-based methods for explaining deep neural networks survey
  - [ ] Time series XAI with perturbation analysis
  - [ ] Validation of XAI explanations for multivariate time series classification
- [ ] Add multi-task learning references
  - [ ] Kendall et al. (2018): Multi-task learning using uncertainty weighting
  - [ ] Chen et al. (2018): GradNorm for adaptive loss balancing
- [ ] Add curriculum learning references
  - [ ] Bengio et al. (2009): Curriculum learning foundations
  - [ ] Li et al. (2019): Curriculum learning for automatic modulation classification

### Phase 1C: Technical Foundation References
- [ ] Add SNR-preserving preprocessing validation
  - [ ] Power normalization techniques in signal processing
  - [ ] Constellation diagram generation methods
- [ ] Add architecture comparison baselines
  - [ ] ResNet variants for signal classification
  - [ ] Transformer architectures for AMC (Swin, ViT comparisons)
- [ ] Update existing ref.bib with proper AMC-focused entries

## PHASE 2: Core Paper Writing

### Phase 2A: Abstract and Introduction (~2 pages)
- [ ] Write Abstract (0.3 pages)
  - [ ] Problem statement: Joint 272-class classification challenge
  - [ ] Method summary: ResNet50 + bottleneck_128 + curriculum learning
  - [ ] Key results: 51.03% validation, 76.14% mod, 68.65% SNR
  - [ ] Significance: First true joint approach vs cascade/oracle methods
- [ ] Write Introduction (1.5 pages)
  - [ ] AMC importance in cognitive radio and electronic warfare
  - [ ] Literature gap: Cascade vs joint approaches
  - [ ] Challenge: 272-class joint prediction complexity
  - [ ] Contributions: Joint architecture, SNR-preserving preprocessing, PIS metric
  - [ ] Paper organization

### Phase 2B: Related Work (~2 pages)
- [ ] Write Cascade vs Joint Approaches subsection
  - [ ] Oracle SNR limitations in practical scenarios
  - [ ] Two-stage methods (Zhang et al., Chen et al.)
  - [ ] Performance comparison: Our 272-class vs literature 220-class
- [ ] Write Constellation-Based AMC subsection
  - [ ] Traditional geometric feature extraction
  - [ ] Deep learning advances (CNN, ResNet, Transformers)
  - [ ] SNR range considerations (0-30 dB precedents)
- [ ] Write Explainable AI for Signal Processing subsection
  - [ ] Need for XAI in AMC applications
  - [ ] Perturbation-based explanation methods
  - [ ] Gap: Limited XAI work in joint modulation-SNR tasks

### Phase 2C: Methodology (~2.5 pages)
- [ ] Write Problem Formulation subsection
  - [ ] Joint prediction mathematical formulation
  - [ ] Multi-task learning framework
  - [ ] Uncertainty weighting approach
- [ ] Write SNR-Preserving Preprocessing subsection
  - [ ] Power normalization technique
  - [ ] Constellation diagram generation
  - [ ] Comparison to per-image normalization (destroys SNR info)
- [ ] Write Architecture Design subsection
  - [ ] ResNet50 backbone justification
  - [ ] Bottleneck_128 SNR layer design
  - [ ] Multi-task head architecture
- [ ] Write Training Strategy subsection
  - [ ] Curriculum learning sliding window approach
  - [ ] CyclicLR schedule (1e-6 → 7e-4)
  - [ ] Cycle-aware patience mechanism
- [ ] Write Perturbation Impact Score (PIS) subsection
  - [ ] Mathematical definition of PIS metric
  - [ ] Constellation region importance quantification
  - [ ] Explainability visualization approach

## PHASE 3: Experimental Validation

### Phase 3A: Test Set Evaluation (HIGH PRIORITY)
- [ ] Download canonical checkpoint from W&B run lmp0536i epoch 14
  - [ ] Save to `/papers/ELSP_Paper/results/checkpoints/canonical_model_epoch_14.pth`
  - [ ] Create checkpoint metadata file
- [ ] Run test set evaluation using `/papers/ELSP_Paper/scripts/test_canonical_model.py`
  - [ ] Expected: ~51-52% based on validation correlation
  - [ ] Generate: Combined, modulation, SNR accuracies
  - [ ] Save to: `/papers/ELSP_Paper/results/performance_metrics/test_set_results.json`
- [ ] Generate confusion matrices for both tasks
  - [ ] Modulation confusion matrix: `/papers/ELSP_Paper/results/confusion_matrices/modulation_confusion_matrix.png`
  - [ ] SNR confusion matrix: `/papers/ELSP_Paper/results/confusion_matrices/snr_confusion_matrix.png`
  - [ ] Confusion matrix data: `/papers/ELSP_Paper/results/confusion_matrices/confusion_matrix_data.json`
- [ ] Calculate F1 scores by class
  - [ ] Modulation F1 scores: `/papers/ELSP_Paper/results/f1_scores/modulation_f1_scores.json`
  - [ ] SNR F1 scores: `/papers/ELSP_Paper/results/f1_scores/snr_f1_scores.json`
- [ ] Save raw predictions and labels
  - [ ] Raw data: `/papers/ELSP_Paper/results/raw_data/test_predictions_*.npy`
- [ ] Statistical significance testing
  - [ ] Compare against cascade baseline
  - [ ] Confidence intervals for performance metrics

### Phase 3B: Perturbation Analysis Implementation (MEDIUM PRIORITY)
- [ ] Implement PIS metric for 272-class model
- [ ] Generate constellation heatmaps showing important regions
  - [ ] Per-modulation family analysis (ASK, PSK, QAM, APSK)
  - [ ] Per-SNR level analysis (low, mid, high SNR)
  - [ ] Save to: `/papers/ELSP_Paper/results/perturbation_analysis/heatmaps/`
- [ ] Quantify explainability metrics
  - [ ] Consistency across modulation types
  - [ ] Correlation with known signal characteristics
  - [ ] Save PIS scores: `/papers/ELSP_Paper/results/perturbation_analysis/pis_scores/`
- [ ] Generate constellation region analysis
  - [ ] Important regions: `/papers/ELSP_Paper/results/perturbation_analysis/constellation_regions/`

### Phase 3C: Ablation Studies (LOWER PRIORITY)
- [ ] Curriculum Learning Ablation
  - [ ] Train identical model without curriculum learning
  - [ ] Compare convergence speed and final performance
  - [ ] Quantify +3.65% benefit from sweep analysis
  - [ ] Save results: `/papers/ELSP_Paper/results/ablation_studies/curriculum_learning/`
- [ ] SNR-Preserving Preprocessing Ablation
  - [ ] Train with per-image normalization (destroys SNR)
  - [ ] Demonstrate 5.7x improvement quantification
  - [ ] Save results: `/papers/ELSP_Paper/results/ablation_studies/snr_preprocessing/`
- [ ] Architecture Ablation
  - [ ] Compare bottleneck_128 vs standard vs bottleneck_64
  - [ ] Validate architectural choices from sweep results
  - [ ] Save results: `/papers/ELSP_Paper/results/ablation_studies/architecture_comparison/`

## PHASE 4: Results and Analysis Writing

### Phase 4A: Experimental Results (~2.5 pages)
- [ ] Write Dataset and Setup subsection
  - [ ] RadioML2018 dataset description
  - [ ] 17 modulations × 16 SNR levels = 272 classes
  - [ ] 80/10/10 split rationale
  - [ ] Training configuration details
- [ ] Write Performance Comparison subsection
  - [ ] Test set results: Combined, modulation, SNR accuracies
  - [ ] Comparison with cascade approaches from literature
  - [ ] Statistical significance analysis
- [ ] Write Ablation Study Results subsection
  - [ ] Curriculum learning impact
  - [ ] SNR-preserving preprocessing benefit
  - [ ] Architecture comparison validation

### Phase 4B: Explainability Analysis (~1.5 pages)
- [ ] Write PIS Metric Validation subsection
  - [ ] Constellation heatmap visualizations
  - [ ] Quantitative importance scores
- [ ] Write Modulation Family Analysis subsection
  - [ ] ASK, PSK, QAM, APSK explainability patterns
  - [ ] Correlation with signal theory expectations
- [ ] Write SNR Impact Analysis subsection
  - [ ] Low vs mid vs high SNR explanation differences
  - [ ] Noise impact on perturbation sensitivity

## PHASE 5: Discussion and Finalization

### Phase 5A: Discussion and Conclusion (~1.5 pages)
- [ ] Write Discussion subsection
  - [ ] Key findings interpretation
  - [ ] Limitations and challenges
  - [ ] Practical implications for AMC systems
- [ ] Write Conclusion subsection
  - [ ] Summary of contributions
  - [ ] Future work directions
  - [ ] Broader impact on AMC field

### Phase 5B: Figure and Table Generation
- [ ] Create Figure 1: Problem formulation diagram
  - [ ] Save to: `/papers/ELSP_Paper/results/figures/problem_formulation.png`
- [ ] Create Figure 2: Architecture diagram (ResNet50 + bottleneck_128)
  - [ ] Save to: `/papers/ELSP_Paper/results/figures/architecture_diagram.png`
- [ ] Create Figure 3: SNR-preserving preprocessing comparison
  - [ ] Save to: `/papers/ELSP_Paper/results/figures/snr_preprocessing_comparison.png`
- [ ] Create Figure 4: Training curves and convergence
  - [ ] Save to: `/papers/ELSP_Paper/results/figures/training_curves.png`
- [ ] Create Figure 5: Confusion matrices (modulation and SNR)
  - [ ] Use: `/papers/ELSP_Paper/results/confusion_matrices/` (already generated)
- [ ] Create Figure 6: PIS heatmaps for key modulation types
  - [ ] Use: `/papers/ELSP_Paper/results/perturbation_analysis/heatmaps/` (from Phase 3B)
- [ ] Create Table 1: Performance comparison with literature
  - [ ] Save to: `/papers/ELSP_Paper/results/tables/performance_comparison.tex`
- [ ] Create Table 2: Ablation study results
  - [ ] Save to: `/papers/ELSP_Paper/results/tables/ablation_study_results.tex`
- [ ] Create Table 3: Architecture comparison summary
  - [ ] Save to: `/papers/ELSP_Paper/results/tables/architecture_comparison.tex`

### Phase 5C: Paper Formatting and Review
- [ ] Format paper according to ELSP template
- [ ] Add proper citations and reference formatting
- [ ] Proofread for technical accuracy
- [ ] Check figure/table numbering and captions
- [ ] Verify all placeholders are filled
- [ ] Final bibliography cleanup and formatting

## PHASE 6: Supplementary Materials (Optional)

### Phase 6A: Extended Experimental Results
- [ ] Extended confusion matrices for all 17 modulations
- [ ] SNR-wise performance breakdown (all 16 levels)
- [ ] Training hyperparameter sensitivity analysis
- [ ] Additional architecture comparisons (Swin, ViT results)

### Phase 6B: Code and Data Availability
- [ ] Prepare code repository for public release
- [ ] Document training scripts and configuration files
- [ ] Create reproducibility guidelines
- [ ] Prepare dataset preprocessing scripts

## Critical Dependencies

**Blocking Phase 3**: Test set evaluation must be completed before Results section
**Blocking Phase 4**: PIS implementation needed for complete explainability analysis
**Flexible**: Ablation studies can be abbreviated if time constrained

## Success Metrics

**Minimum Viable Paper**: Phases 1-2 + Phase 3A + Phase 4A + Phase 5A
**Complete Paper**: All phases through Phase 5C
**Comprehensive Paper**: All phases including Phase 6

## Key Files and Locations

- **Main Paper**: `/papers/ELSP_Paper/ELSP_Latex_Template.tex`
- **Bibliography**: `/papers/ELSP_Paper/ref.bib`
- **Figures**: `/papers/ELSP_Paper/fig/`
- **Results Directory**: `/papers/ELSP_Paper/results/` (organized structure for all artifacts)
- **Best Checkpoint**: `/papers/ELSP_Paper/results/checkpoints/canonical_model_epoch_14.pth`
- **Training Scripts**: `/src/train_constellation.py`
- **Test Script**: `/papers/ELSP_Paper/scripts/test_canonical_model.py`
- **Analysis Scripts**: `/scratch/` (for temporary analysis)

## Research Context Files

- **RUNLOG.md**: Complete experimental history and run comparisons
- **CLAUDE.md**: Technical specifications and key discoveries
- **PAPER.md**: Academic notes and literature analysis
- **Original Paper**: `/papers/constellation_amc_explainability.md` (reference for PIS metric)