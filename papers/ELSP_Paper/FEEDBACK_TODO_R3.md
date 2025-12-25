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

## Action Items After Meeting

*To be filled in after Zoom call*

---

## Completion Checklist

- [ ] Section 1 typos fixed
- [ ] Section 3.1 issues addressed
- [ ] Section 3.2 issues addressed
- [ ] Changes reviewed by Ravi
