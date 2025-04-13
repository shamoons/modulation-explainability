# Curriculum Learning for SNR Classification

## Overview
This document outlines the implementation of curriculum learning for SNR (Signal-to-Noise Ratio) classification in the constellation recognition system. The approach gradually increases the complexity of SNR classification tasks, starting with easily distinguishable SNR values and progressively introducing more challenging cases.

## Background
Curriculum learning, introduced by Bengio et al. [1], is a training strategy that presents examples in a meaningful order, from easier to more difficult. This approach has shown success in various domains, including:
- Natural language processing [2]
- Computer vision [3]
- Signal processing [4]

## Implementation Plan

### 1. Curriculum Stages
```python
stages = [
    {'snr_list': [-20, 30], 'min_epochs': 10},      # Stage 1: Extreme values
    {'snr_list': [-20, -10, 10, 30], 'min_epochs': 10},  # Stage 2: Add intermediates
    {'snr_list': [-20, -15, -10, -5, 5, 10, 15, 30], 'min_epochs': 10},  # Stage 3
    # Additional stages as needed
]
```

### 2. Key Components

#### Curriculum Manager
- Tracks current stage
- Manages stage progression
- Implements plateau detection
- Handles SNR list updates

#### Plateau Detection
- Monitors validation metrics
- Considers minimum epochs per stage
- Uses statistical methods to detect performance plateaus

#### Fixed Head Architecture
- Maintains full 26-class SNR classification head
- Uses masking for inactive classes
- Preserves model architecture compatibility

### 3. Implementation Details

#### Minimal Code Changes
- Isolated curriculum logic in separate module
- Optional feature via command-line flag
- No changes to model architecture
- Easy rollback capability

#### Training Process
1. Start with extreme SNR values (-20dB, 30dB)
2. Monitor validation performance
3. Progress to next stage upon plateau detection
4. Continue until full SNR range is covered

### 4. Monitoring and Evaluation

#### Metrics
- SNR classification accuracy
- Validation loss
- Stage progression
- Training stability

#### Logging
- Current curriculum stage
- Active SNR values
- Performance metrics
- Plateau detection results

## Research References

[1] Bengio, Y., et al. (2009). "Curriculum Learning." ICML.
   - Introduces the concept of curriculum learning
   - Demonstrates benefits of ordered training

[2] Spitkovsky, V. I., et al. (2010). "From Baby Steps to Leapfrog: How "Less is More" in Unsupervised Dependency Parsing."
   - Applies curriculum learning to NLP
   - Shows benefits of gradual complexity increase

[3] Pentina, A., et al. (2015). "Curriculum Learning of Multiple Tasks."
   - Extends curriculum learning to multi-task scenarios
   - Relevant to our dual-task (modulation + SNR) setup

[4] Zhang, Y., et al. (2020). "Curriculum Learning for Deep Learning-based Signal Processing."
   - Applies curriculum learning to signal processing
   - Demonstrates improved convergence and performance

## Rollback Plan

The implementation is designed to be easily reversible:
1. Remove curriculum flag from training script
2. Delete curriculum module
3. Revert to original SNR list handling
4. No model architecture changes needed

## Future Work

Potential extensions:
1. Dynamic curriculum adjustment based on performance
2. Multi-task curriculum balancing
3. Transfer learning between stages
4. Automated stage design

## Notes

- Implementation maintains compatibility with existing code
- No changes to model architecture required
- Easy to disable/enable via command-line flag
- Monitoring tools included for evaluation 