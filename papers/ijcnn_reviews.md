# View Reviews

Paper ID
3220

Paper Title
Constellation Diagram Augmentation and Perturbation-Based Explainability for Automatic
Modulation Classification

Track Name
Main Track

```
Reviewer #
```
### Questions

2. Relevance to IJCNN
Good
3. Technical quality
Fair
4. Novelty
Fair
5. Quality of presentation
Good
9. Comments to Authors
- conversion between complex signals and constellation diagram is really a standard
technique taught in any undergraduate course in communication; authors should not
take it as a part of their contributions
- the SNR bucketing is too coarse: typical communication systems require precise
estimation of the background SNR before proceeding to detection/precoding; the
tolerance of estimation error for general robust schemes should be about 2 or 3dB only
--- a bin-width of 16dB us too much

## Chairs have requested users to

## enter domain conflicts. Please

## click here to enter your conflict

## domains.


- in benchmarking with [23], authors claims that the proposed work achieve a higher
accuracy in SNR; but it does not seem to be exactly the case because [23] seeks to
classify SNR precisely without the coarse binning in this work; this implies the
comparison in the SNR accuracy is biased

Reviewer #

### Questions

2. Relevance to IJCNN
Good
3. Technical quality
Fair
4. Novelty
Fair
5. Quality of presentation
Very good
9. Comments to Authors
This paper proposes a method for modulation signal classification and SNR
classification while introducing certain explainability validation techniques. Experimental
results demonstrate that the proposed approach achieves excellent performance.
However, the multi-task learning method appears to be a simple addition of two loss
functions, lacking a more sophisticated strategy. Additionally, the explainability study is
primarily experimental and does not contribute to algorithmic optimization.
1. The paper first converts signals into constellation diagrams before classification.
However, it is unclear why this intermediate step is necessary. Would it not be possible
to directly process the raw signals for classification? The authors should clarify the
motivation behind this design choice.
2. Why choose to solve classification tasks instead of prediction tasks for SNR?
3. The proposed method claims to address a multi-task learning problem, yet it appears
to simply sum two loss functions without additional mechanisms to balance or optimize
the tasks jointly. Multi-task learning often requires strategies such as task-specific
attention mechanisms, dynamic weighting, or adversarial training to prevent
interference between tasks. The authors should consider incorporating such techniques
to improve the effectiveness of their multi-task framework.
4. The explainability study is purely experimental and does not contribute to algorithmic
improvements. Additionally, the perturbation analysis is quite limited in scope. The
perturbation method only considers pixel intensity variations in the constellation


```
diagram, neglecting other potential influencing factors such as temporal sequence
features and frequency-domain characteristics. The authors should explore more
comprehensive interpretability techniques that consider the broader signal
representation space.
```
5. The paper employs a perturbation-based explainability approach but does not
compare it against other widely used interpretability techniques, such as Grad-CAM.
6. The model performs weakly on high-order modulation modes. The experimental
results of the paper indicate that for high-order modulation modes such as 64-QAM and
256-QAM, the classification accuracy of the model is relatively low (64QAM: 66%,
256QAM: 79%). The article can further explore how to improve the classification
accuracy of high-order modulation modes through feature enhancement, attention
mechanisms, or adaptive data augmentation techniques.

Reviewer #3 | RegAC

### Questions

2. Relevance to IJCNN
Good
3. Technical quality
Good
4. Novelty
Good
5. Quality of presentation
Good
9. Comments to Authors
The paper is focused into proposing a Resnet based model for classification of different
types of constellation and 3 SNR levels (low, medium, high). Some explainability topics
are also addressed.
The paper is well structured, clearly written and fits the conference scope.

```
Comments :
```
- On Fig.3b) and Fig.1c) seem that the learning did not converge yet.
In such curves (over the training epochs) usually the accuracy trajectories with the
training data are also depicted.
- Not much about the choice of alfa and betta in eq. (5)?


- The reported very high combined accuracy may be rather misleading taking into
account the low accuracy of more complex modulations as 64QAM, 256QAM.
- The meaning of the word “augmentation” in the title and the text of the paper is not
clear.

Reviewer #

### Questions

2. Relevance to IJCNN
Good
3. Technical quality
Good
4. Novelty
Fair
5. Quality of presentation
Fair
9. Comments to Authors
Overall Review

```
This paper presents a novel framework for Automatic Modulation Classification (AMC)
that integrates multi-task learning, constellation diagram augmentation, and
perturbation-based explainability. The authors aim to enhance both the performance
and interpretability of deep learning models in AMC tasks, particularly in challenging
environments. By transforming raw I/Q signal data into enriched constellation diagrams,
the framework not only classifies modulation types but also estimates Signal-to-Noise
Ratio (SNR) buckets. The use of perturbation methods provides insights into the
model’s decision-making process, highlighting critical regions in the constellation
diagrams. Experimental results demonstrate high accuracy and robustness across
diverse modulation types, underscoring the framework’s practical implications for real-
world wireless communication systems.
```
```
Strength
```
1. Innovative approach: Combining multi-task learning with perturbation-based
interpretability to address both performance and interpretability issues is a major
advancement in AMC research.


2. Actionable insights: Perturbation analysis identifies key regions in the constellation
diagram, providing valuable insights that can inform further optimization and improve
model credibility.

Weaknesses

1. Implementation details: The paper lacks sufficient details on the implementation of
the proposed methodology, which may hinder reproducibility.
2. Experimental Evaluation: Due to the limited scope of the controlled experiments, a
comprehensive assessment of the relative performance of the proposed framework
cannot be achieved. It is recommended to include additional benchmark models for
comparison, such as MCLDNN and CGDNet.
3. Possible issues: It has been demonstrated in previous studies that iq to constellation
maps may lose some of the information leading to loss of accuracy. The resnet
based model lacks innovation.


