# Audit of Problems 1 and 2 (cells 3–43)

Scope: the executed `Final_Autumn_2026-1.ipynb` (read via `nb_text.md` and the embedded PNGs), `METHODOLOGY_LOG.md` §B–§D and §H,
the lecture PDFs and text. The original exam was compared at commit `02f222c`.
Independent recomputation scripts are in `scratchpad/audit/tmp/p12/` (`p1.py`, `p1b.py`, `p2.py`, `nums.py`).

## 0. Which cells are the exam's
- Exam text cells, unchanged from `02f222c`: 3 (P1 intro), 5 (1.1), 10 (1.2), 14 (1.3), 18 (1.4), 22 (1.5), 25 (P2 intro and rules), 27 (2.1), 29 (2.2), 33 (2.3), 39 (2.4).
- Exam setup code cells, edited by us:
  - cell 4: adds the `os.path.isdir` data-path fallback;
  - cell 26: the same fallback, plus `SEED = 2694` (the exam's placeholder was 0).
  - Nothing else in the exam's code was changed. `cv5` and `cv_acc` are the exam's own.
- Everything else in 6–43 is ours. Answer cells: 9, 13, 17, 21, 24, 28, 32, 38, 43.
- The 2.1 firewall holds. Commit `5086a8b` changes one line (cell 28's text), and every P2 code cell was still `# Your codes here.` at that commit. The current cell 28 matches it byte for byte.

## 1. Question-by-question table

| Q | Cells | Answered fully, in the form asked? | Every number printed? | Reasoning and citations | Code and figures | Issues (ID) |
|---|---|---|---|---|---|---|
| 1.1 | 6–9 | Yes: the baseline, the unpruned tree, the sweep, the preferred size with the tie rule, and both sides of the optimum. | Yes. 1.25 pp and 0.25 pp are 1/80 and 1/400 (arithmetic). | L6 p.57, L8 p.30/32/33/38 and L4 p.43/L5 p.57 all verified. "Within fold-to-fold noise" uses the wrong yardstick: the paired fold differences show the 3-leaf tree beats the 6-leaf tree in 4 of 5 folds. | Reproduced exactly: 64.25 / 85.00 / 83.50 / 90.75 / 90.75 / 89.50 / 88.25 / 87.50. The 3- and 4-leaf trees are identical out of fold. Fig 1.1 matches the text. | m1, n1 |
| 1.2 | 11–13 | Yes, in two sentences, for a lay reader. Gender unused. It describes group purchase rates rather than stating the rule ("the tree predicts…"). | Yes (Table 1.2). "$90,000" and "42" translate the thresholds correctly (salaries are all multiples of $1,000, and ages are integers). | L8 p.33 verified: the same tree, split by split. L5 p.36 on refitting on all data verified. | Figs 1.2a and 1.2b are correct, and the colours match. The five fold trees are printed. | n2 |
| 1.3 | 15–17 | Yes: it reports "whichever way", and the verdict is "no better". The "why" is 2 sentences but ~150 words (a long reading of the sentence limit). | Yes (91.6, 97.4 and 99.8 are roundings; 320 = 4/5 × 400). | L8 p.51 verified. Two problems: (i) "gaps inside the tree's fold range 85–95%" is an unpaired yardstick; (ii) the forest's 99.8% training-fold accuracy is presented as "chasing noise", against L8 p.44 ("not 'optimized to noise' because they are averages"). L8 p.54–55 and p.61 are cited *for* "the readable tree is the better choice", but those slides show the forest beating the tree and call it "the default answer". | Reproduced: RF 88.75 (355), GB 89.00 (356). Paired differences per fold: RF +1 +1 −2 **−6** −2; GB +1 0 0 **−7** −1. Fold 4 carries the whole gap (without it: RF −2, GB 0). | m2, m3, n3 |
| 1.4 | 19–21 | Yes: side by side (table and figure), the disagreement explained, and the choice for a decision maker. | Yes (all roundings). | L8 p.47/56/57/58/60 verified, and the quotes are verbatim. The mechanism is plausible, but "the diagnostic column **confirms** the mechanism" overclaims. The MDI flip is narrow (0.519 vs 0.470) and specific to the split: on 9 of 10 other stratified 70/30 splits (random_state 0–9), MDI ranks **Age** first. What is systematic is that MDI gives Salary more share than held-out permutation does (8 of 10 splits). The "117 vs 43 distinct values" are counted on all 400 rows; the forest saw 280 (108 vs 42). | Reproduced exactly: MDI .011/.470/.519; permutation 2.18/25.50/17.13 pp; held-out accuracy 0.917. The permutation call sets `scoring='accuracy'` explicitly. | m4, n4 |
| 1.5 | 23–24 | Yes: what goes wrong, what to do instead, and the direction of the bias (upward). | n/a | L5 p.27/28/48/56/57/58 #3, L6 p.30, L1 p.33/82 and AIG §4a verified (the 23 vs 15 variables and 0.036 match L5 p.56). "Stratifying makes it slightly worse" is our reasoning, unlabelled and unevidenced. | No code (the exam says "if any"). | n5 |
| 2.1 | 28 | Three sentences, committed before any code (verified in git). It never says an explicit "No" to "should VaR stay near 2.326?", but it predicts collapse. | No numbers. | This is the student's own prediction; no marks depend on it. | — | — |
| 2.2 | 30–32 | Yes: the plot, the one desk on night 2,500, the 1,000-desk statistics, the share below truth/10, where the one desk sits, and whether the story is typical. | Mostly. **2.58** is a double rounding: the printed 2.575 comes from 2.5749, so it should be 2.57. | Correct. VaR < truth/10 ⇔ σ̂² < 0.01. MC SE of the share: 1.6%. | Reproduced from the same streams: mean 2.344, median 0.005851, p5 2.89e-5, p95 1.225, max 1566, 57.8%. An independent χ² simulation (20,000 desks, another seed) gives median 0.0066, share below 0.01 of 55.0%, and the story at the 51.2nd percentile, all consistent. **The "mean" VaR (3.56σ; 4.29% of NAV) in Table 2.2 and in cell 32 is VaR at the mean σ̂², not the mean reported VaR, which is 0.775σ.** Fig 2.2 matches "near 0 for a few hundred nights" (~650), then the slide. | **M1**, n6 |
| 2.3 | 34–38 | Yes: the martingale check, the drift at n = 50/500/5,000, the formula, the histogram, and a single paragraph that reconciles the numbers and gives the top-10 share (88.8%). | Yes (all roundings). −1/(n−1) = −0.002004 is printed in cell 37. | L2 p.56/57/71–72 verified. The σ̂²_t = 1 check fails its own pre-set CI rule (p = 0.0088). This is reported honestly and read as MC luck: the exactness of `ddof=1` is a theorem, so the reading is defensible. "The same share of desks that go down (51.1% vs 50.7%)": these are 2.1 SE apart, so "same" breaks P2-04's "consistent with, never equals" rule. "Spread grows like √t, **so** roughly normal" is a non sequitur (normality comes from summing iid increments). | Code is correct. The SEs are computed across desks, and increments are iid given the scale-free step. The precise n = 50 run was pre-planned (P2_plan.md), not post hoc. The theory cell is labelled supplementary. Figs 2.3a and 2.3b match the text. | m5, n7, n8 |
| 2.4(a) | 40, 43 | Yes: the median and the 5th and 95th percentiles on night 2,500. | Yes. | Correct. The analytic stationary band 1 ± 1.645 × 0.0366 = 0.940–1.060 matches 0.9406–1.059. "Never widens after that" is loose (0.943–1.063 vs 0.938–1.064). | The implementation follows the exam: each desk keeps its own 500 real days (rescaled to variance 1) plus tonight's 500. The demeaned robustness row is on common random numbers. | n9 |
| 2.4(b) | 41–43 | Yes: the volatility, the kurtosis, the night-1 VaR comparison, the kurtosis comparison, and a single paragraph that answers all four sub-questions. (b) is labelled "loss of information" and (a) "accumulation of noise". | Mostly. "**0.55 pp**" is not printed, and it is mis-rounded: 0.03325 − 0.027803 = 0.5447 pp, so it should be 0.54. "The non-zero mean" is claimed but the mean is never printed (0.064%/day; t ≈ 2.1 under iid). | L2 p.57 and L2 p.80 quotes are verbatim, and the iid-bootstrap caveat is correctly labelled as ours. The difference is **1.7 SE** (0.545/0.320), and the SE is itself optimistic, yet "the normal model understates the 1-in-100 loss" is stated without that qualification. The printed exceedance rate (1.65%; 25 of 1,511 days) is the stronger evidence. The night-1 library is all 1,511 days (not the manual's 500); this is stated only in a code comment and the log. | Reproduced: sd 0.011951, kurtosis 24.075 (24.159 bias-corrected), pipeline VaR 0.027803, empirical 0.03325, exceedances 0.016545. The normal kurtosis reference uses n = 500, while the real series has n = 1,511 (conservative; NIT). Figs 2.4b and 2.4c match the text. | m6, m7, n10 |

## 2. Findings by severity

### (a) Errors in what exists
- **FATAL: none.** P1's shuffled stratified folds are on a cross-section of unrelated people, which is legitimate (L8 p.45). No transformation is fitted anywhere in P1 or P2.
- **M1 (MAJOR). Wrong number with a misleading label: Table 2.2 (cell 31) and the cell 32 answer table.** The row "reported VaR (σ units)" in the "mean" column shows 3.56 (and 4.286% of NAV). That is 2.326·√(mean σ̂²), not the mean reported VaR across desks, which is **0.775σ** (recomputed on the same stream). Read naturally, it says the average desk *over*-reports VaR, the opposite of the truth. The quantile columns are fine, because a monotone transform commutes with quantiles. *Fix:* in cell 31, compute the VaR column as `(Z01*np.sqrt(S2_T)).mean()` for the mean row (or label it "VaR at the mean σ̂²"), and update cell 32.
- **m1 (MINOR). Unpaired yardstick for paired comparisons** (cell 9, "3 vs 6 leaves … within fold-to-fold noise"; cell 17, "the gaps are inside the tree's own fold-to-fold range of 85% to 95%"). The folds are shared, so fold difficulty is common to all models. The relevant quantity is the per-fold difference:
  - 6-leaf minus 3-leaf: +1 −1 −1 −3 −1 (the 3-leaf tree wins 4 of 5 folds);
  - RF minus tree: +1 +1 −2 −6 −2;
  - GB minus tree: +1 0 0 −7 −1.

  The conclusion "no better" survives, and is better argued: fold 4 alone produces the whole ensemble deficit. *Fix:* print the per-fold differences (all numbers are already in Tables 1.1/1.3) and rephrase. This stays within P1-04 (fold counts won or lost), with no test needed.
- **m2 (MINOR). 1.3 "chase noise" evidence for the forest** (cell 17). A forest's ~100% training-fold accuracy is structural (deep trees on in-bag rows) and is not evidence of overfitting; L8 p.44 says forest predictions are "not 'optimized to noise' because they are averages". *Fix:* keep the training-accuracy argument for boosting only (L8 p.51: "More trees eventually hurt"). For the forest, argue that there is little variance to average away (the five fold trees are near-identical, 1.2), and that `max_features=1` forces splits on randomly drawn features, Gender included.
- **m3 (MINOR). Citation direction** (cell 17: "the readable tree is the better choice (L8 p.54–55, p.61)"). Those slides show averaging *beating* one tree (0.8327 vs 0.7220) and call the forest "the default answer". *Fix:* cite them as the contrast: "unlike L8 p.54, where 20,640 tracts reward averaging …". Keep p.54 for the point that untuned boosting falls short.
- **m4 (MINOR). 1.4 overclaim on mechanism** (cell 21: "The diagnostic column confirms the mechanism"). The MDI rank flip is narrow (0.519 vs 0.470), and on 9 of 10 alternative stratified 70/30 splits MDI ranks Age first. The flip is specific to the split; the Salary tilt is systematic. *Fix:* say "is consistent with" and add one sentence noting that the MDI margin is small. Optionally add a labelled robustness loop over splits (our construction).
- **m5 (MINOR). 2.3 martingale check fails its own rule at σ̂²_t = 1** (Table 2.3a, p = 0.0088). This is reported honestly, and the "1.7%" framing was added after the result (`p2/p2_fix1.py`). *Fix (optional, same draws, no seed-shopping):* add a post-hoc row, labelled as such, that pools both starts. The step is scale-free, so the ratios are identically distributed: mean ≈ 0.99986, t ≈ −1.4.
- **m6 (MINOR). 2.4(b) VaR gap is stated without using its SE** (cell 43). The gap is 0.545 pp against an SE of 0.320, i.e. 1.7 SE, and the SE is optimistic. *Fix:* "0.54 pp, 1.7 resampling SEs, so the quantile alone is suggestive; the exceedance rate (1.65%, 25 of 1,511 days, against 1%) and the kurtosis (24.1 against a maximum of 1.51 under normality) are decisive."
- **m7 (MINOR). Every-number rule** (cell 43). The "0.55 pp" is not printed and is mis-rounded (it should be 0.54). "The non-zero mean" is asserted with no printed mean. *Fix:* add `var_emp - var_pipe` and `r.mean()` to Table 2.4b.
- **NITs:**
  - n1: "2 to 4 leaves carry the signal" contradicts "2 leaves: underfitting" (cell 9).
  - n2: 1.2 could lead with the rule ("the tree predicts a purchase for …").
  - n3: 1.3 "why" is two ~75-word sentences.
  - n4: the distinct-value counts are from 400 rows, not the 280 training rows (108/42).
  - n5: "stratifying makes it slightly worse" is unlabelled own reasoning.
  - n6: 2.58 should be 2.57 (cell 32).
  - n7: "the same share" (2.1 SE apart) and "misses by 0.04%" (the CI misses by 0.009%).
  - n8: "√t, so normal" is a non sequitur.
  - n9: "never widens".
  - n10: the 1,511-day night-1 library is not stated in the answer; the kurtosis null uses n = 500.

### (b) Omissions worth proposing
1. 1.1/1.3: a per-fold paired-difference column (see m1).
2. 1.4: a split-robustness check of the MDI flip, labelled as our construction (see m4).
3. 2.2: the mean reported VaR across desks (0.775σ) makes the reconciliation sharper: even the *average* desk reports a third of the truth.
4. 2.4(b): print the real mean and the VaR gap (m7); interpret the SE (m6).

### (c) Methodology log (§B–§D, §H) against the notebook
- §D P2-05 claims the drift SE is "cross-checked against the SE across all increments". The notebook prints no such check (MINOR). The same entry omits the planned 2,000 × 2,000 precise n = 50 run, on which the −1/(n−1) conclusion rests.
- §D firewall says the theory note "(§D.2) will be added". It is §D.3 and it now exists. §D.3 is also stale ("When added, it will be labelled …") and repeats its "not taught" sentence (NIT).
- §C P1-06's rationale is garbled: it says the answer uses "mostly" because 9 of 241 low-salary young people bought. The notebook says "almost never" for that group and "mostly" for the buyers (NIT).
- §B G-05 says each problem opens with "shape, dtypes, counts and head". No `head()` is printed in P1 or P2 (NIT).
- §D P2-07 says a 500-day library is "offered as an optional robustness row". There is none (NIT).
- §C's known-answer check says the unpruned tree has "about 62 leaves" (L8 p.30 has 62). The notebook's has 61. That is consistent, not a contradiction.
- No other contradictions. §C and §D.2 results match the notebook to the printed digits.

### (d) Verified OK
- Every P1 accuracy, fold count, tree split, leaf count and importance was reproduced exactly.
- All P2 night-2,500 statistics were reproduced from the same SeedSequence streams. The shares (top-10 0.8879, top-1 0.6681, above 1: 0.059) match.
- P2's distributional conclusions agree with an independent χ² simulation and with theory (median exp(2,499·(ψ(k/2) − log(k/2))) = 0.00666; sd of log 3.168).
- The dj30 statistics were reproduced.
- The 2.4(a) band matches the analytic stationary AR(1)-in-variance band.
- All lecture quotes checked are verbatim on the cited pages: L8 p.44/47/51/54/55/56/57/58/60/61, L5 p.27/28/36/48/56/57/58, L6 p.30/57, L2 p.56/57/71/80, L1 p.33/82, L4 p.43/46.
- Seeds: `random_state=7034` on every tree, forest and boosting model and on the permutation call. P2 uses named, independent streams. The vectorised pipeline is unit-tested against the literal loop. `ddof=1` is set explicitly everywhere.
- Execution order is sequential (execution counts 2–22), with no warnings or errors.
