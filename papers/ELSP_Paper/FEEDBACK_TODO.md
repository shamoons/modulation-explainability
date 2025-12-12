# FEEDBACK TODO - Advisor Feedback Round 2

*Created: December 12, 2025*
*Status: COMPLETE*

---

## Summary of Ravi's Changes (Already Applied)

### Section 3.3 - PIS Metric
- Added equation labels (E:deltaA, E:PIS)
- Expanded explanation of what PIS quantifies
- Added note about ΔA potentially being negative
- Clarified that f is always between 0 and 1

### Section 3.4 - SNR Range Bounding
- Rearranged text for better flow
- Added forward reference: "F1 score is defined in detail later"
- Split long information-theoretic sentence

### Section 4.3 - Evaluation Metrics
- Added F1 as separate equation (E:F1)
- Expanded definitions of precision and recall
- Added reference to sokolova2009systematic

### F1 Redundancy Fix (Our Change)
- **Section 3.4**: Simplified to "the F1 metric is defined formally in Section~\ref{sec:metrics}"
- **Section 4.3**: Added \label{sec:metrics}, removed "As mentioned earlier", kept full definition here
- **Rationale**: Section 4.3 (Evaluation Metrics) is the natural place for metric definitions; Section 3.4 just references it

### Typo Fix (Our Change)
- **Section 3.3**: Fixed "acciracy" → "accuracy"

### Minor Changes Throughout
- "SOTA" → "state of the art (SOTA)"
- "where" → "Note that" for equation definitions
- Grammar and flow improvements

---

## New Action Items from Ravi's Email

### Task 1: Add "black box" reference (Section 1) ✅ COMPLETED
**Location**: After "these models often operate as 'black boxes'"
**Reference**: Koh, Pang Wei and Percy Liang. "Understanding black-box predictions via influence functions." ICML 2017, pages 1885-1894. PMLR.
**Action Items**:
- [x] Look up the paper and verify it is pertinent to black-box ML models
- [x] Add citation if appropriate
- [x] Add to ref.bib
**Changes Made**:
- Added `\cite{koh2017understanding}` after "black boxes" in Section 1 (line 140)
- Added `koh2017understanding` entry to ref.bib

### Task 2: Add reference and examples for "high-stakes applications" (Section 1) ✅ COMPLETED
**Location**: After "This gap in understanding limits the deployment of AMC models in high-stakes applications"
**Action Items**:
- [x] Add a reference for high-stakes ML deployment
- [x] Give examples of high-stakes applications (e.g., autonomous vehicles, medical diagnosis, military communications, spectrum enforcement)
**Changes Made**:
- Added examples: "military communications, spectrum enforcement, and autonomous cognitive radio systems"
- Added `\cite{rudin2019stop}` citation (Rudin 2019, Nature Machine Intelligence)
- Added `rudin2019stop` entry to ref.bib

### Task 3: Fix -35 dB claim and distinguish from our work (Section 1, 4th paragraph) ✅ COMPLETED
**Location**: Sentence "Recent advances in blind estimation demonstrate robust performance even at ultra-low SNRs, achieving reliable parameter estimation at -35 dB without any prior knowledge..."
**Problem**: This sentence suggests the problem is already solved, which shouldn't be the impression
**Action Items**:
- [x] Add a reference for the -35 dB claim
- [x] Add a sentence distinguishing what is mentioned here from the work in this paper
- [x] Clarify that prior work is on different/simpler problems (e.g., single modulation parameter estimation, not joint classification)
**Changes Made**:
- Revised sentence to: "...with specialized methods achieving reliable single-parameter estimation at -35 dB for specific modulation types~\cite{wang2021duffing}"
- Added distinguishing sentence: "However, these approaches address isolated estimation tasks (e.g., carrier frequency for BPSK), not the substantially harder problem of joint modulation and SNR classification across diverse modulation families that we tackle in this work."
- Added `wang2021duffing` entry to ref.bib (Wang et al. 2021, Chinese Physics B)

### Task 4: Refer to and discuss Table 1 ✅ COMPLETED

**Location**: Proper place in text (likely Section 5.1 Results)
**Action Items**:
- [x] Add explicit reference to Table 1 in the text
- [x] Add discussion of what Table 1 shows
**Changes Made**:
- Added bridging sentence after results bullet list (line 434): "Table~\ref{tab:sota} contextualizes these results against prior constellation-based AMC work; while modulation-only approaches achieve higher accuracies under favorable conditions, they do not address the substantially harder joint prediction problem that we tackle here."

### Task 5: Refer to and discuss Figures 3 and 4 (Section 5) ✅ COMPLETED

**Location**: Section 5 Results
**Action Items**:
- [x] Add explicit reference to Figure 3 at proper place
- [x] Add discussion of Figure 3
- [x] Add explicit reference to Figure 4 at proper place
- [x] Add discussion of Figure 4
**Changes Made**:
- Added Figure 4 reference in Section 5.1 after key observations: "Figure~\ref{fig:f1_plots} illustrates this performance hierarchy through per-class F1 scores for both modulation types and SNR levels."
- Added Figure 3 reference at end of Section 5.2 PIS Trend Analysis: "Figure~\ref{fig:pis_plots} visualizes these perturbation effects across mask types and sizes."

### Task 6: Make Section 5.4 comprehensive evaluation a table ✅ COMPLETED

**Location**: Section 5.4 Architecture Comparison
**Current**: Inline text with architecture comparisons
**Action Items**:
- [x] Create a formal table for the architecture comparison
- [x] Refer to the table in the text
- [x] Ensure all architectures and metrics are included
**Changes Made**:
- Added `Table~\ref{tab:arch}` reference in intro sentence
- Added `\caption` and `\label{tab:arch}` to existing table
- Upgraded formatting to booktabs (`\toprule/\midrule/\bottomrule`)
- Fixed en-dashes for ranges (23--26%)

### Task 7: Match up primary/secondary evaluation metrics (Sections 4.3, 5, 6) ✅ COMPLETED

**Issue**: Section 4.3 mentions primary and secondary metrics; Sections 5 and 6 should explicitly reference these
**Action Items**:
- [x] Review Section 4.3 to identify primary metrics (combined accuracy)
- [x] Review Section 4.3 to identify secondary metrics (individual accuracies, F1, confusion matrix, task weights)
- [x] In Section 5, point out where primary metrics are discussed
- [x] In Section 5, point out where secondary metrics are discussed
- [x] In Section 6, point out where these metrics are discussed
- [x] Ensure metrics in 4.3 match what's actually reported in 5 and 6
**Changes Made**:
- Added "(primary)" after "Combined Accuracy" in Section 5.1 bullet list
- Added sentence: "Secondary metrics including per-class F1 scores and task weight evolution are detailed below."

### Task 8: Proofread Ravi's major changes ✅ COMPLETED

**Sections**: 3.3, 3.4, 4.3
**Action Items**:
- [x] Proofread Section 3.3 (PIS) for typos and clarity
- [x] Proofread Section 3.4 (SNR Range) for typos and clarity
- [x] Proofread Section 4.3 (Evaluation Metrics) for typos and clarity
- [x] Check mathematical notation consistency
- [x] Verify all equation references work
**Results**:
- All sections proofread - no typos or clarity issues found
- Equation references verified: E:deltaA, E:PIS, E:F1, sec:metrics all working
- Mathematical notation consistent throughout
- Forward reference from Section 3.4 to Section 4.3 (sec:metrics) working correctly

---

## Implementation Order

### Phase 1: Quick Fixes (15 mins)
1. Task 8: Proofread Ravi's changes
2. Task 1: Verify and add black-box reference
3. Task 2: Add high-stakes examples and reference

### Phase 2: Section 1 Fix (15 mins)
4. Task 3: Fix -35 dB claim with reference and distinction

### Phase 3: Tables and Figures (30 mins)
5. Task 4: Add Table 1 reference and discussion
6. Task 5: Add Figure 3 and 4 references and discussions
7. Task 6: Create architecture comparison table

### Phase 4: Metric Alignment (15 mins)
8. Task 7: Align primary/secondary metrics across sections

### Phase 5: Final Review (10 mins)
- Review all changes
- Compile change list for email

---

## Completion Checklist

- [x] Task 1: Black box reference added
- [x] Task 2: High-stakes examples added
- [x] Task 3: -35 dB claim fixed with distinction
- [x] Task 4: Table 1 referenced and discussed
- [x] Task 5: Figures 3 and 4 referenced and discussed
- [x] Task 6: Architecture comparison table created
- [x] Task 7: Metrics aligned across sections
- [x] Task 8: Ravi's changes proofread
- [x] Email response drafted

---

## Notes

### Current Figure/Table Numbering (need to verify)
- Figure 1: Pipeline diagram
- Figure 2: Perturbation pipeline
- Figure 3: PIS plots (accuracy degradation, PIS comparison, impact summary)
- Figure 4: F1 score plots (modulation, SNR)
- Table 1: SOTA comparison table

### References to Find/Add
- [ ] Koh & Liang (2017) - Black-box predictions via influence functions
- [ ] Reference for high-stakes ML applications
- [ ] Reference for -35 dB blind estimation claim
