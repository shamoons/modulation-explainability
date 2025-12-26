# FEEDBACK TODO - Advisor Feedback Round 3

*Created: December 25, 2025*
*Status: IN PROGRESS*

---

## Ravi's Message

> Section 3.1 and 3.2 has serious problems. We need to meet on zoom to discuss. Please have a paper and pencil with you.

---

## Section 1 Review (Ravi's Edits)

### Changes Made by Ravi
- [x] Title: "Introduction" → "Introduction and Motivation"
- [x] Expanded "I/Q" to "in-phase/quadrature (I/Q)" on first use
- [x] Added clarification after "black boxes": "which do not reveal the crucial steps used to reach a decision"
- [x] Added "(comprehension of why an outcome is reached)" after "explainability techniques"
- [x] Changed "Joint modulation and SNR classification" → "Joint modulation classification and signal-to-Noise ratio (SNR) estimation"
- [x] Expanded "BPSK" to "Binary Phase-Shift Keying"
- [x] Restructured contributions into numbered lists (lines 149-164)
- [x] Restructured "In this work, we propose..." paragraph

### Issues Found
- [ ] **Typo (line 152)**: "assesed" → "assessed"
- [ ] **Capitalization (line 144)**: "signal-to-Noise ratio" should be "signal-to-noise ratio" (lowercase 'n')

---

## Section 2 Review (Ravi's Edits)

### Changes Made by Ravi
- [x] Added intro sentence: "This section summarizes the work related to the main aspects of this study."
- [x] Restructured Section 2.2 ending into numbered list (lines 180)
- [x] Minor grammar/flow improvements

### Issues Found
- None identified

---

## Section 3.1 Analysis - Potential "Serious Problems"

### Current Content Summary
Section 3.1 "Enhanced Constellation Diagram Generation" covers:
1. Power normalization equations (Eqs for power, scale_factor, I/Q normalization)
2. Histogram generation and log scaling
3. Reference to Figure 1 (pipeline)

### Likely Issues Ravi Will Raise

#### 1. Missing Explanation: WHY per-image normalization destroys SNR
**Current text**: "Our critical discovery revealed that standard preprocessing approaches destroy SNR information through per-image normalization"
**Problem**: No explanation of WHY this happens or HOW power normalization fixes it
**What's needed**:
- Mathematical explanation of what per-image max normalization does
- Show that it maps all images to same intensity range, erasing SNR differences
- Explain that power normalization preserves relative intensity across SNR levels

#### 2. Equations lack labels/references
**Current state**: Equations have no `\label{}` tags
**Problem**: Cannot reference them later in the paper
**Fix**: Add labels like `\label{eq:power}`, `\label{eq:scale}`, `\label{eq:inorm}`

#### 3. No clear problem statement before solution
**Current flow**: Jumps straight to equations
**Problem**: Reader doesn't understand what problem we're solving
**What's needed**: Clear statement like "Standard preprocessing applies per-image max normalization: $H_{norm} = H / \max(H)$. This maps all constellation diagrams to identical intensity ranges [0,1], destroying the intensity differences that encode SNR information."

#### 4. Missing connection between normalization and SNR preservation
**Problem**: The 1.73x improvement claim isn't explained here (where it should be)
**What's needed**: Explain that high-SNR signals have tighter constellation points (higher peak intensities) while low-SNR signals have spread points (lower peak intensities)

---

## Section 3.2 Analysis - Potential "Serious Problems"

### Current Content Summary
Section 3.2 "Task-Specific Heads for Modulation and SNR" is ONE paragraph covering:
- Shared backbone → task-specific heads
- Modulation branch: single linear classifier
- SNR branch: standard or bottleneck configurations (64/128 units)

### Likely Issues Ravi Will Raise

#### 1. Too brief and informal
**Current state**: Only 1 paragraph, no equations, no figures
**Problem**: Doesn't meet academic rigor standards
**What's needed**:
- Formal mathematical description of the architecture
- Equations for each head's forward pass
- Figure showing the branching architecture

#### 2. Implementation-specific terminology
**Current text**: "bottleneck_64 and bottleneck_128"
**Problem**: Sounds like code, not academic writing
**Fix**: "64-unit bottleneck layer" or "bottleneck layer with 64 hidden units"

#### 3. Vague claim without evidence
**Current text**: "empirically provided the most reliable convergence among the architectures we evaluated"
**Problem**: No evidence, no reference to results section
**What's needed**: Either remove claim or add forward reference to experimental results

#### 4. "decorrelates the feature streams" - undefined
**Problem**: Technical term used without definition
**What's needed**: Explain what this means or remove if not essential

#### 5. Missing architecture diagram
**Problem**: No visual representation of the multi-task head structure
**What's needed**: Figure showing backbone → split → mod head / SNR head

---

## Preparation for Zoom Meeting

### Questions to Clarify with Ravi
1. For Section 3.1: Does he want a full mathematical derivation of why power normalization preserves SNR?
2. For Section 3.2: Does he want a dedicated figure for the task heads architecture?
3. Should Section 3.2 be expanded significantly or kept concise?

### Things to Have Ready
- [ ] Diagram of per-image normalization vs power normalization effect
- [ ] Mathematical notation for the task-specific heads
- [ ] Understanding of the 1.73x improvement metric derivation

---

## Action Items From Zoom Call

### 1. Section 3.1: Equation 4 - histogram2d undefined

**Current text**: `H = log(1 + histogram2d(I_normalized, Q_normalized))`

**Problem**:
- `histogram2d` is a Python/NumPy function, NOT mathematical notation
- **Cannot use code/built-in routines in equations**
- Terms not defined

**Action needed**:
- Define histogram mathematically, e.g.:
  - Let the I-Q plane be divided into M×M bins
  - H(m,n) = count of samples where I falls in bin m and Q falls in bin n
  - Or: H(m,n) = |{i : I_i ∈ [a_m, a_{m+1}), Q_i ∈ [b_n, b_{n+1})}|
- Then apply log scaling: H_log = log(1 + H)

---

### 2. Figure 1 issues

**Problems**:
- "HDF5" not defined - what is it?
- "2D Histogram (I_norm, Q_norm)" notation inconsistent with equations
  - Equations use `I_normalized`, `Q_normalized`
  - Figure uses `I_norm`, `Q_norm`
- "val" should be "validation" (no abbreviations)
- Figure combines two different things:
  - Constellation image generation (belongs in Section 3.1)
  - Training pipeline (belongs in Section 4)

**Action needed**:
- Define HDF5 or just say "Raw I/Q data"
- Make figure notation match equation notation exactly
- Spell out "validation" not "val"
- **Split Figure 1 holistically** based on paper structure:
  - Section 3.1: Constellation generation only (Raw I/Q → normalization → histogram → log scaling → image)
  - Section 3.6: Uncertainty weighting (the loss equation diagram)
  - Section 4: Training pipeline (dataset split, DataLoader, backbone, heads, optimizer)
  - Think through what goes where based on section content
- **Undefined terms in figure**:
  - "DataLoader", "batch", "shuffle" - where do these belong? Section 4?
  - "filtered mods" - what modulations were filtered? Must be defined somewhere (Section 4 dataset description?)

---

### 3. Section 3.1: Need example constellation figures

**Problem**: No visual showing what the enhancement actually does

**Action needed**:
- Add figure showing constellation diagrams:
  - Before enhancement (per-image normalized)
  - After enhancement (power normalized)
- Show side-by-side for same signal at different SNRs to illustrate SNR preservation

---

### 4. Section 3.2: Dual heads are the CORE contribution

**Current state**: Section 3.2 is one brief paragraph, undersells the contribution

**Problem**:
- This is the core of the multi-task approach - needs emphasis
- Dual heads (modulation + SNR) is the key architectural innovation
- Currently reads like an implementation detail, not a contribution

**Action needed**:
- Emphasize that dual-head architecture is central to joint classification
- Explain WHY separate heads are needed (different task characteristics)
- Add mathematical formulation for each head
- **Define "shared backbone"** - what does this mean?
  - The CNN/Transformer that extracts features from constellation image
  - Shared = same weights used for both tasks
  - Then splits into separate heads
- **Add architecture diagram** showing:
  - Input image → Backbone (ResNet/Swin) → shared features → split → Mod head / SNR head

---

### 5. Section 3.1: "Our critical discovery" issue (line 194)

**Current text**: "Our critical discovery revealed that standard preprocessing approaches destroy SNR information through per-image normalization"

**Problem**:
- Self-congratulatory language ("our critical discovery")
- Claims a discovery but doesn't explain WHAT was discovered or WHY it matters

**Action needed**:
- Remove promotional language
- Elaborate on what the discovery actually was:
  - What is per-image normalization? (H_norm = H / max(H))
  - Why does it destroy SNR? (maps all histograms to same [0,1] range)
  - What does power normalization do differently? (normalizes I/Q before histogramming, preserves relative intensities)
  - How does this connect to SNR? (high SNR = tight points = high peaks; low SNR = spread = lower peaks)
- **Add references** for standard preprocessing approaches (Ravi emphasized this)

---

### 6. Figure 2: Perturbation and PIS Pipeline

**Problems**:
- `acc'_mod`, `acc'_snr` notation not defined
- Figure currently in Section 3, should move to new Section 4 (Perturbation Methodology)

**Action needed**:
- Define notation in text:
  - `acc_mod` = modulation accuracy (baseline)
  - `acc'_mod` = modulation accuracy after perturbation (prime = perturbed)
  - Same for SNR: `acc_snr`, `acc'_snr`
- **Simplify PIS formula in figure**: Don't repeat full equation in figure
  - Just show conceptual elements: ΔA, PIS, etc.
  - Reference the equation number in text instead
- Move figure to new Section 4 when restructuring
- Ensure all terms in figure are defined in the accompanying text

---

### 8. Section 3.5: ResNet50 missing from architecture list

**Current text** (line 379):
```
ResNet18/34: Convolutional baselines (11-21M parameters)
```

**Problem**:
- ResNet50 is not listed, but it's the architecture that achieved best results (51.48% test)
- Paper claims to evaluate architectures but omits the winning one

**Action needed**:
- Add ResNet50 to the architecture list with parameter count (~25M)
- Include reference to He et al. (same as ResNet18/34)
- Update text to reflect that ResNet50 was selected based on evaluation

---

### 9. Inflated run count claim

**Current text**: "Comprehensive evaluation across 500+ experimental runs"

**Problem**:
- Actual count from logs: ~168-200 runs total
- 500+ is not accurate

**Action needed**:
- Verify exact run count from W&B
- Update to accurate number (likely "200+" or "nearly 200")
- Or use vaguer language like "extensive experimentation"

---

### 10. Missing confidence intervals

**Problem**:
- Results reported as single values without uncertainty
- Need statistical rigor: 95% confidence intervals
- Reviewers/committee will ask: "How do you know ResNet50 is better than Swin Tiny?"

**Action needed**:
- Extract individual accuracies from W&B logs for each run
- Focus on 4 key architectures: ResNet18, ResNet34, Swin Tiny, ResNet50
- Send data to Ravi - he has MATLAB code to compute confidence intervals
- Add confidence interval visualization to results

---

### 11. Figure definitions: caption AND text

**Ravi's guidance**:
- Define ALL figure terms in BOTH the caption AND the text
- Reason: Some reviewers skip captions, some skip text
- Be redundant for clarity

---

### 12. Paper Structure: Split Section 3

**Current structure**:
- Section 3: Methodology (contains everything)
- Section 4: Experimental Setup
- Section 5: Results
- Section 6: Discussion
- Section 7: Limitations and Future Work
- Section 8: Conclusion

**New structure**:
- **Section 3: Learning Methodology**
  - 3.1: Enhanced Constellation Diagram Generation
  - 3.2: Task-Specific Heads (dual heads - CORE contribution)
  - 3.3: SNR Range Bounding Justification (was 3.4)
  - 3.4: Architecture Evaluation Framework (was 3.5)
  - 3.5: Multi-Task Learning with Uncertainty Weighting (was 3.6)
- **Section 4: Perturbation-Based Explainability** (NEW - was 3.3)
- Section 5: Experimental Setup (bumped from 4)
- Section 6: Results (bumped from 5)
- Section 7: Discussion (bumped from 6)
- Section 8: Limitations and Future Work (bumped from 7)
- Section 9: Conclusion (bumped from 8)

**Rationale**: Learning methodology and explainability are two distinct contributions deserving separate sections.

---

## Completion Checklist

- [ ] Section 1 typos fixed (assesed, capitalization)
- [ ] Section 3.1: Equation 4 rewritten with proper math notation
- [ ] Section 3.1: "critical discovery" expanded with references
- [ ] Section 3.1: Example constellation figures (before/after enhancement)
- [ ] Section 3.2: Expanded as CORE contribution with diagram
- [ ] Section 3.5: ResNet50 added to architecture list
- [ ] Section 3/4 restructuring done (split methodology + explainability)
- [ ] Figure 1 split into multiple figures (constellation gen, training pipeline, uncertainty)
- [ ] Figure 2 notation defined, PIS formula simplified, moved to Section 4
- [ ] All figure terms defined in BOTH caption AND text
- [ ] Run count verified and corrected (~200, not 500+)
- [ ] Confidence intervals: Extract W&B data, send to Ravi for MATLAB analysis
- [ ] **Self-review**: Proofread and self-critique entire paper
- [ ] Changes reviewed by Ravi
