# Paper Completion Checklist

Canonical source: `papers/ELSP_Paper/AMC_Constellation_Paper.tex`
Data sources: `papers/ELSP_Paper/results/performance_metrics/test_set_results.json`, `papers/ELSP_Paper/results/perturbation_analysis/pis_summary.json`

## Front Matter & Venue
- [ ] Replace placeholders: Received/Accepted/Published (TBD), DOI (`https://doi.org/10.55092/xxxx`), first-page footer (“ELSP 2025: TBD”).
- [x] Confirm venue/header consistency: updated headers/footers to ELSP (main + first page).
- [x] Update corresponding author block (names/emails) per final author order.

## LaTeX Hygiene & References
- [x] Fix escaped section commands: change `\\subsection/\\subsubsection/\\textbf` to `\subsection/\subsubsection/\textbf` in perturbation sections.
- [x] Convert manual reference block to BibTeX: `\bibliographystyle{aiasbst}` + `\bibliography{ref}`; replace in-text cites with `\cite{...}` keys from `ref.bib`.
- [ ] Full compile and resolve warnings/overfull boxes.

## Figures (add includes + captions)
- [ ] Pipeline overview (constellation generation + training flow). Source: `fig/fig1.png` (or create `fig/pipeline.png`).
- [ ] Architecture diagram (shared trunk + modulation/SNR heads). File: `fig/architecture.png` (to be added).
- [ ] Training curves (loss/accuracy per task). File(s): `results/figures/training_curves.png` (generate if missing).
- [ ] Confusion matrices and F1 bars (modulation and SNR). Place under `results/figures/` (export from evaluation scripts).
 - [x] PIS plots moved to `fig/` and included: `fig/accuracy_degradation_curves.png`, `fig/pis_comparison_bar_chart.png`, `fig/perturbation_impact_chart.png`.
 - [x] F1 plots moved to `fig/`: `fig/f1_modulation_bar.png`, `fig/f1_snr_bar.png`.
 - [x] Confusion matrices exported under `results/confusion_matrices/` (optional figure pending).

## TikZ Block Diagrams
- [x] Add TikZ packages to preamble: `\usepackage{tikz}`, `\usetikzlibrary{arrows.meta,positioning,fit,shapes,calc}`.
- [x] Constellation & Training Pipeline diagram (Figure X):
  - Raw I/Q → Power Normalization → 2D Histogram → Log Scaling → Constellation Image → Dataset (224×224) → DataLoader → Model (Backbone + Heads) → Losses (CE, Uncertainty) → Training.
- [ ] Perturbation & PIS Pipeline diagram (Figure Y):
  - Constellation Images → Perturbation Masks (Top p%, Bottom p%, Random p%) → Perturbed Dataset → Evaluation (Acc_mod, Acc_snr, Acc_combined) → Compute PIS → Plots (Figs).
- [x] Perturbation & PIS Pipeline diagram (Figure Y).
- [x] Insert references to both diagrams in Methodology (Figure X) and Explainability (Figure Y) text.

## Tables & Numbers
- [ ] Architecture comparison table with best/avg metrics. Include: Swin-Tiny 45.45% (combined), ResNet50 51.26% (combined). Cite source.
- [ ] Ablations table: SNR head variants (standard vs bottleneck_64/128 vs dual_layer), curriculum on/off, preprocessing variants.
- [x] Insert exact test metrics in Results: combined=0.5126, modulation=0.7639, snr=0.6871.
- [x] Add SOTA comparator table (mod-only bounded vs joint baseline vs this work) with citations.

## PIS Values (replace [TBD])
- [x] From `pis_summary.json`, fill representative values in text:
  - Top1 PIS (combined): 11.75; Top5: 4.07
  - Bottom1 PIS (combined): 48.67; Bottom5: 9.72
  - Random1 PIS: ~0 (slightly negative); Random5: ~0.87
- [x] Replace all `[TBD]` placeholders in Results and Conclusion with actual numbers.

## Methods & Reproducibility
- [ ] Dataset details: 17 digital modulations, SNR 0–30 dB (16 levels), 4,096 samples/class → 1,114,112 total; 80/10/10 split; image size 224×224; power normalization + log scaling.
- [ ] Training config (final): optimizer, LR, batch, weight decay, dropout, pretrained weights; cycle-aware schedule; device; PyTorch/UV versions.
- [ ] Repro commands: provide `uv run` commands; checkpoint path; reference W&B run ID.
- [ ] Code/Data availability statement (repo URL; dataset prep steps).

## Citations & SNR Range Justification
- [x] Add explicit citations for SNR range precedent: `peng2023constellation`, `zhang2023multimodal`, `garcia2024ultralight`.
- [x] Ensure uncertainty weighting and curriculum learning cites: `kendall2018multitask`, `li2019curriculum`.
- [x] Add footnote clarifying dataset contains negative SNR bins (e.g., -20 to -2 dB) but experiments bound to 0–30 dB by design.

## Copy Editing & Submission
- [ ] Consistent terminology (PIS, SNR-preserving, joint modulation–SNR, curriculum, uncertainty weighting).
- [ ] Tense/grammar pass; unit/notation consistency; figure/table cross-references.
- [ ] Finalize formatting per ELSP template requirements.

## Deliverables Checklist
- [ ] Updated `.tex` with fixed sections, numbers, figure/table includes.
- [ ] All figures under `papers/ELSP_Paper/fig/` or `papers/ELSP_Paper/results/figures/`.
- [ ] BibTeX integrated; `.bbl` generated on compile; no manual `\bibitem` block.
