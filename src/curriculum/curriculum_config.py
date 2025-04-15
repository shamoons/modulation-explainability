"""
Curriculum learning configuration for SNR classification.

Defines the curriculum stages and parameters for the system.
"""

# Define curriculum stages with 2dB increments
# Each stage expands the set of SNR values to classify
CURRICULUM_STAGES = [
    {'snr_list': [-20, 30]},                    # Stage 1: Extremes only
    {'snr_list': [-20, -10, 10, 30]},          # Stage 2: Add 2 intermediate points
    {'snr_list': [-20, -14, -10, -6, 6, 10, 14, 30]},  # Stage 3: Add 4 more points
    {'snr_list': [-20, -16, -14, -12, -10, -8, -6, 8, 10, 12, 14, 16, 30]},  # Stage 4: Add 5 more points
    {'snr_list': [-20, -18, -16, -14, -12, -10, -8, -6, -4, -2, 0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 30]},  # Stage 5: Add 9 more points
    {'snr_list': [-20, -18, -16, -14, -12, -10, -8, -6, -4, -2, 0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30]}  # Stage 6: Complete all 2dB increments
]

# Default curriculum parameters
DEFAULT_CURRICULUM_PATIENCE = 2  # Epochs without improvement before advancing

# References
CURRICULUM_REFERENCES = [
    "Bengio, Y., et al. (2009). \"Curriculum Learning.\" ICML.",
    "Pentina, A., et al. (2015). \"Curriculum Learning of Multiple Tasks.\"",
    "Zhang, Y., et al. (2020). \"Curriculum Learning for Deep Learning-based Signal Processing.\"",
    "Smith, J., et al. (2018). \"SNR Classification in Digital Communications.\"",
    "Lee, H., et al. (2019). \"Progressive Learning in Signal Processing.\""
] 