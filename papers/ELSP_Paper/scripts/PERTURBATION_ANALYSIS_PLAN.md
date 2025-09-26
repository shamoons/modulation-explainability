# Perturbation Analysis Plan - What We're Looking For

## Executive Summary

We're quantifying the importance of different regions in constellation diagrams for AMC by measuring how their removal affects model performance. The key metric is PIS (Perturbation Impact Score), which normalizes accuracy drop by the fraction of pixels perturbed.

## Research Questions

### 1. Which pixels matter most for classification?
- **Hypothesis**: Constellation center points (brightest pixels) are critical
- **Test**: Compare top vs bottom perturbations
- **Expected**: High PIS for top pixels, low PIS for bottom pixels

### 2. Is the impact specific or general?
- **Hypothesis**: Targeted perturbations have higher impact than random
- **Test**: Compare top/bottom vs random perturbations
- **Expected**: top_PIS > random_PIS > bottom_PIS

### 3. How does impact scale with perturbation size?
- **Hypothesis**: Diminishing returns as more pixels are masked
- **Test**: Plot PIS vs perturbation percentage (1-5%)
- **Expected**: PIS decreases as percentage increases

### 4. Which modulations are most sensitive?
- **Hypothesis**: Dense constellations (256QAM) more sensitive than sparse (BPSK)
- **Test**: Calculate per-modulation PIS scores
- **Expected**: Higher-order modulations have higher PIS

### 5. How does SNR affect sensitivity?
- **Hypothesis**: Mid-SNR (0-14 dB) most sensitive to perturbations
- **Test**: Calculate per-SNR PIS scores
- **Expected**: Bell curve with peak at mid-SNR

## Data We Need to Extract

### Primary Metrics
1. **Overall PIS Scores**
   ```json
   {
     "top1": 35.2,    // High impact from constellation centers
     "top5": 18.7,    // Still significant but diminishing
     "bottom1": 0.8,  // Minimal impact from noise pixels
     "bottom5": 0.4,  // Even less impact
     "random1": 6.3,  // Baseline for comparison
     "random5": 3.1   // Baseline at higher percentage
   }
   ```

2. **Accuracy Drops**
   ```json
   {
     "top5": {
       "combined": "51.26% → 42.39% (-8.87%)",
       "modulation": "76.39% → 65.12% (-11.27%)",
       "snr": "68.71% → 61.45% (-7.26%)"
     }
   }
   ```

3. **Task-Specific Impact**
   - Which task (modulation vs SNR) is more affected?
   - Does perturbation type affect tasks differently?

### Secondary Analysis

1. **PIS Curves**
   - X-axis: Perturbation percentage (1, 2, 3, 4, 5)
   - Y-axis: PIS score
   - Three lines: Top, Bottom, Random
   - Shows scaling behavior

2. **Per-Modulation Sensitivity**
   ```
   256QAM: PIS_top5 = 25.3 (most sensitive)
   128QAM: PIS_top5 = 22.1
   ...
   BPSK: PIS_top5 = 12.4 (least sensitive)
   ```

3. **Per-SNR Sensitivity**
   ```
   SNR -20 to -2: PIS ≈ 0 (already random)
   SNR 0-14: PIS = 20-30 (peak sensitivity)
   SNR 16-30: PIS = 10-15 (less sensitive)
   ```

### Validation Checks

1. **Sanity Metrics**
   - Original accuracy matches baseline (51.26%)
   - No negative PIS values
   - Random perturbations consistent across seeds

2. **Expected Relationships**
   - PIS(1%) > PIS(2%) > ... > PIS(5%)
   - top_PIS >> random_PIS > bottom_PIS
   - Higher-order modulations more sensitive

## Interpretation Framework

### What High PIS Means
- **PIS > 20**: Critical regions for classification
- **PIS 10-20**: Important but not essential
- **PIS 5-10**: Moderate importance
- **PIS < 5**: Minimal importance

### For the Paper Narrative

1. **Explainability Story**
   - "The model relies heavily on constellation center points"
   - "Removing just 1% of brightest pixels causes X% accuracy drop"
   - "This confirms the model learned meaningful signal patterns"

2. **Comparison to Literature**
   - Most papers don't quantify explainability
   - PIS provides objective importance metric
   - Enables comparison across architectures

3. **Practical Implications**
   - Adversarial robustness: Know which pixels to protect
   - Data efficiency: Can potentially subsample less important regions
   - Model understanding: Confirms learning of physical signal properties

## Visualization Requirements

### Figure 1: PIS vs Perturbation Percentage
- Professional line plot with error bars
- Clear legend: "Top pixels", "Bottom pixels", "Random baseline"
- Highlight diminishing returns trend

### Figure 2: Example Perturbations
- Grid showing original and perturbed constellations
- Use high-SNR QPSK for clarity
- Annotate with accuracy drops

### Figure 3: Modulation Sensitivity Heatmap
- Rows: Modulation types (sorted by complexity)
- Columns: Perturbation types and percentages
- Color: PIS values (red = high impact)

### Table 1: Summary Statistics
| Perturbation | Accuracy Drop | PIS Score | Interpretation |
|-------------|---------------|-----------|----------------|
| Top 1%      | -3.52%        | 35.2      | Critical pixels |
| Top 5%      | -8.87%        | 17.7      | Important region |
| Bottom 1%   | -0.08%        | 0.8       | Negligible impact |
| Random 1%   | -0.63%        | 6.3       | Baseline comparison |

## Expected Outcomes

1. **Primary Finding**: Model correctly identifies constellation points as critical
2. **Quantification**: Top 1% pixels have ~40x more impact than bottom 1%
3. **Validation**: Random perturbations show intermediate impact
4. **Insights**: Higher-order modulations more fragile to perturbations

## Integration with Paper

### Replace Placeholders
- Line 312: "modulation accuracy dropping from 76.39% to 65.12%"
- Line 312: "PIS of 35.2"
- Line 315: "PIS as low as 0.8"
- Update all [TBD] with actual values

### Add New Insights
- Quantitative explainability metric (first for AMC)
- Validates model learned correct features
- Opens door for targeted defense strategies

## Next Steps

1. **Monitor perturbation generation** (~2.5 hours remaining)
2. **Implement evaluation script** following EVALUATION_SCRIPT_SPEC.md
3. **Run full analysis** (~2.5 hours)
4. **Generate visualizations**
5. **Update paper with results**
6. **Create supplementary materials** if interesting patterns emerge

## Success Criteria

- [ ] All 15 perturbation types evaluated
- [ ] PIS scores follow expected patterns
- [ ] Visualizations clearly communicate findings
- [ ] Paper updated with all actual values
- [ ] Results reproducible with documented commands