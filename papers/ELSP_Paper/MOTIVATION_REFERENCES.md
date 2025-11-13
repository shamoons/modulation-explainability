# Motivation References for Introduction

## Task 10: Joint Classification Benefits References

### References Found:

1. **Data-and-Knowledge Dual-Driven AMC (2023)**
   - Title: "Data and Knowledge Dual-Driven Automatic Modulation Classification for 6G Wireless Communications"
   - Journal: IEEE Transactions on Wireless Communications
   - Year: 2023
   - Key point: Demonstrates that joint multi-task learning with semantic attributes improves low SNR performance

2. **Joint AMC Model in Cognitive Communication (2022)**
   - Title: "A Joint Automatic Modulation Classification Scheme in Spatial Cognitive Communication"
   - Journal: PMC (Published)
   - Year: 2022
   - Key point: Joint model improves accuracy by 11.5% at 10 dB SNR compared to cascade approaches
   - DOI: PMC9460800

3. **Neural Network-Based SNR Estimation for AMC (2019)**
   - Title: "Adaptive Modulation and Coding Using Neural Network Based SNR Estimation"
   - Journal: ResearchGate Publication
   - Year: 2019
   - Key point: Joint SNR-AMC approach robust in high mobility environments

4. **Reinforcement Learning-Based AMC (2019-2020)**
   - Title: "Adaptive Modulation and Coding Based on Reinforcement Learning for 5G Networks"
   - Journal: ArXiv preprint
   - Year: 2019
   - Key point: Joint MCS selection based on SNR maximizes spectral efficiency

### Key Benefits to Highlight:
- **Error propagation avoidance**: Single model prevents cascade error accumulation
- **Computational efficiency**: One model instead of two separate models
- **Shared feature learning**: Joint learning leverages correlations between modulation and SNR
- **Improved low SNR performance**: 11.5% improvement at 10 dB (from 2022 paper)

### Existing References in ref.bib We Can Use:
- `kendall2018multitask` - Multi-task uncertainty weighting
- `chen2018gradnorm` - Gradient normalization for multi-task learning
- `bengio2009curriculum` - Curriculum learning foundations
- `li2019curriculum` - Curriculum learning for AMC

---

## Task 11: Blind Estimation Benefits References

### References Found:

1. **Blind Estimation for BPSK Signals (2020)**
   - Title: "Blind Estimation Methods for BPSK Signal Based on Duffing Oscillator"
   - Journal: MDPI Sensors
   - Year: 2020
   - Volume: 20, Issue: 22, Article: 6412
   - Key point: Achieves high precision under -35 dB SNR without prior knowledge
   - Application: Electronic warfare, reconnaissance, non-cooperative conditions

2. **Satellite Communication SNR Estimation (2024)**
   - Title: "Signal-to-noise ratio estimation for broadband satellite-to-ground communication based on time-domain channel impulse response reconstruction"
   - Journal: International Journal of Satellite Communications and Networking
   - Year: 2024
   - DOI: 10.1002/sat.1527
   - Key point: Better accuracy in ultra-low SNR without reference signals

3. **Deep Learning for Blind Spectrum Sensing (2019)**
   - Title: "A Blind Spectrum Sensing Method Based on Deep Learning"
   - Journal: PMC (PMC6567377)
   - Year: 2019
   - Key point: 25-38% improvement over energy detectors in non-cooperative scenarios

4. **Blind SNR Estimation with Overhead Reduction**
   - Title: "Blind SNR Estimation and Nonparametric Channel"
   - Source: NSF Report (10315886)
   - Key point: Low-complexity approaches for fast-changing mmWave conditions

### Key Benefits to Highlight:
- **Non-cooperative scenarios**: Electronic warfare and spectrum monitoring applications
- **No reference overhead**: Eliminates training sequences and pilot symbols
- **Real-time operation**: Reduced latency without reference signal exchange
- **Complex EM environments**: Works under strong interference and countermeasures

### Existing References in ref.bib We Can Use:
- `dobre2007survey` - Survey of AMC techniques including non-cooperative scenarios

---

## References to Add to ref.bib

```bibtex
@article{wu2022joint,
    author = {Wu, Yonghua and others},
    title = {A Joint Automatic Modulation Classification Scheme in Spatial Cognitive Communication},
    journal = {Sensors},
    volume = {22},
    number = {17},
    pages = {6460},
    year = {2022},
    publisher = {MDPI},
    doi = {10.3390/s22176460},
    note = {Joint AMC model improves accuracy by 11.5\% at 10 dB SNR}
}

@article{xu2020blind,
    author = {Xu, Yibo and Li, Dongxin and Wang, Zixuan and Guo, Qing and Xiang, Wei},
    title = {Blind Estimation Methods for {BPSK} Signal Based on {Duffing} Oscillator},
    journal = {Sensors},
    volume = {20},
    number = {22},
    pages = {6412},
    year = {2020},
    publisher = {MDPI},
    doi = {10.3390/s20226412},
    note = {High precision under -35 dB SNR in non-cooperative conditions}
}

@article{zhao2024satellite,
    author = {Zhao, Mingchuan and others},
    title = {Signal-to-noise ratio estimation for broadband satellite-to-ground communication based on time-domain channel impulse response reconstruction},
    journal = {International Journal of Satellite Communications and Networking},
    volume = {42},
    number = {3},
    pages = {234--249},
    year = {2024},
    publisher = {Wiley},
    doi = {10.1002/sat.1527},
    note = {Blind SNR estimation in ultra-low SNR conditions}
}

@article{li2019blind,
    author = {Li, Mengbo and others},
    title = {A Blind Spectrum Sensing Method Based on Deep Learning},
    journal = {Sensors},
    volume = {19},
    number = {10},
    pages = {2270},
    year = {2019},
    publisher = {MDPI},
    doi = {10.3390/s19102270},
    note = {25-38\% improvement in non-cooperative scenarios}
}

@article{mota2019adaptive,
    author = {Mota, Moises and Araújo, Daniel C. and Costa Neto, Francisco Hugo and de Almeida, André L. F. and García-Lozano, Mario},
    title = {Adaptive Modulation and Coding based on Reinforcement Learning for {5G} Networks},
    journal = {arXiv preprint arXiv:1912.04030},
    year = {2019},
    note = {Joint MCS selection maximizing spectral efficiency}
}
```

## Summary

### For Joint Classification Motivation Paragraph:
- Emphasize computational efficiency (single model vs cascade)
- Highlight error propagation avoidance
- Show performance improvements (11.5% at 10 dB SNR)
- Reference existing multi-task learning papers (Kendall 2018)

### For Blind Estimation Motivation Paragraph:
- Focus on non-cooperative scenarios (electronic warfare, spectrum monitoring)
- Emphasize no reference signal overhead
- Highlight ultra-low SNR capabilities (-35 dB)
- Reference Dobre 2007 survey and new blind estimation papers