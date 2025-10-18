# Future Work + Reviewer Follow-Ups

## Motivation / Significance
- Emphasize why joint modulation+SNR with explainability matters for spectrum security, EW threat response, and 6G/NTN resilience.
- Highlight how current AMC deployments fail without calibrated SNR awareness and transparent reasoning.

## Statistical Rigor
- Re-run key configs (canonical ResNet50 + bottleneck_128, Swin-Tiny) with ≥3 seeds and report mean ± std for combined/mod/SNR accuracy.
- Document hardware used, wall-clock time, and variance to support reproducibility.

## Explainability Validation
- Compare PIS against Grad-CAM, Integrated Gradients, and random masking as baselines.
- Conduct a small user study (RF engineers) or quantitative sanity checks (e.g., correlation with accuracy drop) to justify PIS.

## Generalization / Channels
- Extend experiments beyond AWGN (e.g., Rayleigh/Rician fading, multipath, interference) to evaluate robustness.
- Explore curriculum or domain randomization to improve transfer to real-world captures.

## Documentation Enhancements
- Add a glossary/table describing SNR head variants (`standard`, `bottleneck_64`, `bottleneck_128`, `dual_layer`) and their layer structures.
- Provide a concise reproducibility appendix with commands, configs, and dataset preparation steps.
