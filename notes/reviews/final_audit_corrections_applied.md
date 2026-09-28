# Corrections being applied directly (errors only; no new analyses)

These will be in the notebook, write-up and log when the methods document is published. A document that describes the
notebook must describe it *after* these corrections.

## Problem 3 reasons and wording that change
1. **x11**: not "an exact combination of two included series". Correct: it has gaps (three countries stop early, a stale fill
   would be wrong by up to 10.1 pp) and it is nearly redundant with x86 − x12 (rebuild correlation 0.988), whose parts are both
   in the model. Excluded for those reasons.
2. **Boosting**: not run because of runtime and the exam's deadline-style budget (L8 p.55: "If you have an hour, or a deadline
   … the forest"), not because of leakage — tuning on forward folds would handle leakage. Stated in the notebook now.
3. **Interactions**: characteristic × macro interactions were not run in the linear models; the M3 forest could fit them
   automatically and scored worse than the characteristics-only forest (Table 3.18).
4. **Deep forest**: the evidence against it is the held-out comparison (Table 3.21: deep minus leaf-200 forest), not its
   in-sample R² (a forest fits its training rows closely by design, L8 p.44).
5. **Leaf 200** is our choice, a departure from L8 p.44's deep trees (already stated).
6. **PCA inside PCR** runs on standardised inputs, i.e. correlation-matrix PCA — our choice (L1 only demeans).
7. **Month bootstrap**: resamples out-of-sample months of fixed forecasts (no refit, unlike L2 p.80) and treats months as
   independent; serial dependence was not checked (the earlier claim that it "matters little" is withdrawn).
8. **Grid edges** (L5 p.27): ridge's pick sits at the heavy-penalty edge in 64% (M2) / 93% (M3) of refits; that edge is
   effectively the class-means model (P1 weights correlate 0.9994 with it); PCR's K = 1 edge was a real design gap (K = 0
   sensitivity reported).
9. **"Macro lowers R² in every model"** → true for the 14 pairs in Table 3.18; the curated-x10 variant (0.065%) is above ridge M2
   (0.042%, Table 3.10b); the losses are large for OLS and the forests.
10. **The paired characteristics-beyond-class-means delta** (−0.045%, SE 0.027%) is a non-rejection; its ±2 SE interval is about
    −0.10% to +0.01%. It is small partly because ridge shrank the characteristics almost to zero, so it tests the sign of what ridge
    used, not a bound on all within-class signal.
11. **α-regression β's** are not separate exposures: EW and RP returns are collinear (RP's β is 1.21 with EW, 0.72 alone; Table 3.27).
12. **Class blocks**: three of the four classes are tightly correlated (B 0.55, C 0.73, D 0.58); class A is not (0.21; Table 3.2).
13. **Per-asset trailing mean** is the easiest (noisiest) bar, not a harder one.
14. **Clustering (L7)**, **KNN**, **neural nets**, **covariance-optimised weights**, **outside data** and **information criteria**
    are now listed in the write-up as considered and not used, with reasons.
15. Lecture page numbers are **PDF pages**; in Lectures 1, 5 and 8 the printed slide numbers are lower after overlays.

## Citation corrections (notebook)
- Design: forest-may-beat-ridge → L8 p.54, p.62 (not p.55); schemes: static fit from the exam's workflow, expanding/rolling L5 p.48–52.
- Deep-forest overfitting → held-out evidence; L8 p.62 (weak, linear signal), not p.40.
- Permutation importance → L8 p.57 (not p.56–58).
- Ranks → "cap the leverage of any single characteristic value (L3 p.11–12)"; not "no extreme value can dominate a LS fit".
- Vol-scaling "class A would dominate a pooled fit" → our reasoning; constant variance L3 p.2–3 (L2 p.71–73 removed).
- Ridge intercept → the ridge objective penalises the slopes it contains (L5 p.12); keeping class intercepts out is our choice.
- Placebo rule → our construction in the spirit of L4 p.26–28's simulated null.
- Table 3.30 → "first eigenvalue of the return correlation matrix, as a share of the trace; our construction (eigenvalues: L1 p.43)".
- L1 p.63 → rank deficiency when T < n (36-month windows); not "badly conditioned".
- α as a performance benchmark → L2 p.49 added.
- Tercile sorts on training data → L4 p.43, L5 p.41 (L5 p.58 item 4 removed).
- P1: L8 p.54–55/p.61 cited as a contrast; L5 p.48 quote "Usually, CV is not valid…"; L6 p.57 narrowed.
