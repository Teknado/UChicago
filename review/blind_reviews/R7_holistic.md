# R7: Holistic blind grading, BUSN 41210 Final Exam (Solution A vs Solution B)

Grader stance: I wrote the exam and I am grading for the final mark. The binding standard is the exam text (including its sentence limits), the AI Coding Guide (a shuffled split or a transformation fitted before the split is fatal), the scope of the 8 lectures, the rule that every number must be printed by a cell, and honest reporting.

Cell numbers are notebook cell indices as they appear in `solution_*_P*.txt`. In Solution A, all of Problem 3 is one code cell (39) followed by the write-up (40).

**Fatal errors:** neither solution has one. Neither uses a shuffled split on time-ordered data. Every scaler, z-score and sigma-hat is either trailing or fitted inside the training window. A's full-sample PCA on the extended file (cell 39) is exploratory only and feeds no model.

**Independent checks I ran:**
- I re-ran A's 2.3 drift cell with seed 2694. It prints -0.018645 / -0.003788 / -0.000233, as in A's output, not the numbers A's answer quotes.
- I re-ran B's 1,000-desk simulation from its SeedSequence streams. It reproduces mean 2.3435, median 0.005851, max 1565.8, 57.8% below truth/10 and top-10 share 0.888 exactly.
- Where the two solutions compute the same thing, their numbers agree:
  - EW / RP / TSMOM gross Sharpe 0.282 / 0.308 / 0.199;
  - class-means R²_OOS +0.087%;
  - corr(x10, same-year x86) 0.998;
  - 99 missing x11 values;
  - 43 floored x5 months.

  So both pipelines are mechanically sound on these shared quantities.

---

## 1. Rubric (built from the exam) and scores

### Problem 1: Trees and ensembles (20)

| # | Full-credit answer must contain | pts | A | B | Justification (cells) |
|---|---|---|---|---|---|
| 1.1 | "Nobody purchases" = 64.25%; unpruned CV accuracy (85.00%); all six sweep values; choice of 3 leaves by the tie rule; underfitting at 2 leaves and overfitting beyond 4, with the unpruned tree as the extreme | 5 | 5 | 5 | A c6–7: all numbers correct, both sides explained. B c6–9: the same, plus training-fold diagnostic, proof that the 3- and 4-leaf trees give identical out-of-fold predictions, and fold-level margins |
| 1.2 | `plot_tree` of the chosen tree; one or two plain-English sentences (Age > 42.5 buy; younger buy only if salary > $90.5k) | 3 | 3 | 3 | A c9–10: clean two sentences. B c11–13: two sentences with counts, a clearly separated "technical note" and a fold-stability check |
| 1.3 | RF 88.75%, GB 89.00%; neither beats 90.75%; 2–3 sentences on why (n = 400, near-rectangular boundary, little variance or bias left to remove, max_features = 1 of 3) | 4 | 3.5 | 3.5 | A c12–13: correct. But the answer is a 4-part numbered template far beyond 2–3 sentences, and "GB overfits" is asserted without evidence. B c15–17: correct with fold-by-fold comparison and training accuracy 97.4% as evidence, but also well over the sentence limit |
| 1.4 | Side-by-side MDI vs held-out permutation; rankings disagree (MDI: Salary, permutation: Age); in-sample, split-cardinality mechanism; show permutation to the decision-maker | 4 | 3.5 | 4 | A c15–16: correct table and reasoning. But "≈116 vs ≈43 distinct values" is not printed (actually 117/43 full sample, 108/42 training), and "Gender zero signal" contradicts its own 0.019. B c19–21: split counts and distinct values printed, training-row permutation diagnostic, correlation ruled out, caveats |
| 1.5 | Future months leak into training; neighbouring months are near-duplicates (persistence); accuracy biased upward; walk-forward or expanding evaluation with a benchmark fixed in advance | 4 | 3.5 | 4 | A c18–19: look-ahead, upward bias and walk-forward are all there, but generic and it overclaims "collapses completely". B c23–24: all mechanisms, plus that `shuffle=False` and stratification do not fix it, and a training-only benchmark |
| | **Subtotal** | **20** | **18.5** | **19.5** | B loses 0.5 for exceeding sentence limits (1.3, 1.5), A loses 0.5 in 1.3 for the same reason plus content |

### Problem 2: Training on your own output (20)

| # | Full-credit answer must contain | pts | A | B | Justification (cells) |
|---|---|---|---|---|---|
| 2.1 | A committed prediction in 2–3 sentences (no marks for being right) | 2 | 2 | 2 | A c23: predicts collapse (VaR ≈ 0.19) via Jensen. B c28: predicts drift in the median of about −1/(n−1) and a mean that stays near 1 |
| 2.2 | One-desk plot of log σ̂²; night-2,500 σ̂² and VaR; 1,000-desk mean, median, p5, p95, max; share with VaR < truth/10; percentile of the one desk; story quantified: (0.24/2.8)² = 0.0073, which is about the median, so the story is typical | 6 | 5 | 6 | A c25–26: all statistics printed (median 0.0075, 54.1% below truth/10, desk at the 79.3rd percentile). Deductions: the story is called "typical" without converting it to σ̂²; "over 90% collapsed below the start" is not printed; 2,500 rather than 2,499 draw steps (negligible). B c30–32: everything, including the story at the 52.7th percentile with a rounding band, and mean VaR vs VaR-at-mean |
| 2.3 | One-step unbiasedness verified in a cell; drift measured for n = 50/500/5,000 over hundreds of desks and nights; formula −1/n (≈ −1/(n−1)); histogram; top-10 share; one-paragraph reconciliation (log random walk with negative drift, expectation carried by a vanishing tail, no new information enters) | 6 | 3.5 | 6 | A c28–29: martingale 1.000002, histogram and top-10 share 68.04% are fine, and the lognormal reconciliation is correct. But the drift is measured with a single step (not "a few hundred nights"), and the answer misquotes its own output: it prints −0.018645 / −0.003788 / −0.000233 and reports "−0.0204 / −0.0020 / −0.0002, within Monte Carlo error". Its n = 500 value is nearly twice the formula. The answer is a numbered list, not a paragraph. B c34–38: 200k-desk one-step test (the CI miss at σ̂² = 1 is reported honestly with its probability), 500 × 500 runs plus a precise n = 50 run that separates −1/n from −1/(n−1), n × drift ≈ −1, a full reconciliation paragraph, and an explicit reconciliation with its 2.1 prediction |
| 2.4 | (a) median, p5, p95 with the real days kept; (b) vol 1.195%, excess kurtosis 24.1, empirical VaR 3.33% vs pipeline 2.78%, scenario kurtosis ≈ 0; paragraph: synthetic data contain the fitted model plus its estimation error and lack the tails and any new information; (b) = loss of information, (a) = accumulation of noise; any self-retrained pipeline; only fresh real data stops it | 6 | 4.5 | 6 | A c31–32: all numbers printed and the labels correctly assigned. But "what it contains" is given as "infinite volume and Gaussian tractability", missing that the synthetic sample carries the model's own estimation error. The real days are not rescaled to variance exactly 1 as the rules require. The answer is a 4-item list, not a paragraph. B c40–43: all numbers, a kurtosis null distribution, a bootstrap SE for the VaR gap, the 1.65% exceedance rate, and a complete one-paragraph answer |
| | **Subtotal** | **20** | **15** | **19.5** | B loses 0.5 for 2.3's answer running well past "one paragraph" |

### Problem 3: Research project (60), graded on the discipline of the evaluation

| # | Element: full-credit content | pts | A | B | Justification (cells) |
|---|---|---|---|---|---|
| 3.1 | **Know your data**: cumulative returns by class; per-asset table (mean, vol, Sharpe, worst month); class-ordered correlation matrix and what the blocks imply for pooling; rolling 12-month Sharpe by class and its persistence; scale, distribution and persistence of each characteristic *within class*; tercile sorts (preferably training only); a hypothesis for x1–x5 backed by printed evidence | 8 | 4 | 8 | A c39: has the cumulative plot, block correlations and training-block tercile spreads. It has class-portfolio stats only (no per-asset table), no rolling Sharpe, and characteristic stats pooled rather than within class. Its x1–x5 hypotheses are in the write-up, but the supporting correlations are never printed. B c54–64: every element, including a per-asset table, persistence of yearly Sharpe, a within-class characteristic table, exact identities x2 = R12 and x5 = SD36, and training-only terciles with t and Bonferroni |
| 3.2 | **Feature engineering**: correct lags; within-month standardisation with a reason; missing values diagnosed per series and country, treated with past information only; the stale-value trap addressed and the rebuild reported; extended-file duplicates found; back-fills handled; trailing z for macro; country mapping and differential with class A justified; vol-scaled target and how it maps to positions; a table of every predictor with the count | 12 | 8.5 | 12 | A c39/40: lags correct, within-class ranks justified, x10 same-year-average look-ahead found (good), x11 dropped with pooled rebuild corr 0.988, trailing z and 2-month lag, differential vs country 7, class A zeroed, vol-scaled target. Deductions: the audit prints are partly hard-coded strings (see §4); the exam-flagged duplicate columns are never searched for; there is no per-country rebuild quality; "exact linear combination" is an overclaim; the "97% redundancy" claim rests on an unstandardised PCA; the predictor "table" is counts only. B c55–57, c65–68: every trap is found (including stale x1 in five B assets and six exact duplicates plus four hidden global copies), with per-country rebuild quality and stale-fill error of up to 10.1 pp, a causality test by truncation, 20 lag spot checks, and a full predictor table |
| 3.3 | **Models and evaluation schemes**: benchmarks computed before the models; windows fixed in advance with reasons; several models; one or more schemes (comparison valued); R²_OOS vs trailing mean *and* zero, model × scheme, and by class; hyper-parameters chosen on training data only; which predictors carry the signal and whether that is stable; per-class vs pooled; some measure of uncertainty | 12 | 5 | 12 | A c39: the benchmark is the correct expanding pooled mean, plus zero; four models; expanding window only. The ridge penalties are hard-coded (α = 50, α = 200) on unscaled inputs, with no training-data selection and no stated rationale. By-class results for M2 only. Coefficient paths are collected but never shown. No per-class models, no SEs. B c50–52, c69–88: 38 pre-listed specifications across 5 model types × 3 schemes, forward inner-fold tuning with asserted order, a month bootstrap (plus a block-bootstrap check), paired scheme differences, class tables, coefficient-sign stability, OOS permutation importance, per-class vs pooled, and a hindsight upper bound on tuning |
| 3.4 | **Macro and placebo**: refit without macro on the same rows; placebo with *every* macro series circularly shifted by 36 months *and* other shifts, applied before feature construction; correct interpretation | 8 | 1.5 | 8 | A c39: M3 vs M2 exists, but its −2.97% reflects an arbitrary, weak penalty (B's tuned ridge M3 is −0.08%). The placebo cannot work: it replaces only the raw `g_*` columns, while M3 uses the interaction columns built earlier, and country macro is never shifted. All three placebos therefore print exactly M3's −2.972% / −3.303%. B c89–92: every raw series is shifted before the full pipeline, 8 shifts, paired SEs, real macro ranks 4 of 9, wrap-around disclosed, persistence table |
| 3.5 | **Portfolios and attribution**: rule stated; EW, RP and TSMOM at unit gross, monthly; Sharpe, return, vol, drawdown, turnover, net of costs; cumulative plot; sub-periods; exposure vs timing, correctly interpreted | 10 | 6 | 10 | A c39/40: correct rules and metrics at 10 bp, drawdown plot, attribution regression. No sub-periods, one cost level, and EW turnover ignores drift. The interpretation is wrong: "decisively beat" and "positive trend-following alpha" with α t = 1.56, and the gain is attributed to characteristics when the weights are essentially class-mean tilts. B c71, c94–100: all of that plus P3 long-short, a class risk budget, a 0–50 bp grid with break-evens, sub-periods, a bootstrap SE on the Sharpe difference, and a class-means regressor that absorbs α (t = 0.82). The frozen static version gives Sharpe 0.27 |
| 3.6 | **Write-up**: the paper structure the exam asks for; every number printed by a cell; honest; conclusions consistent with the evidence; methodology reproducible | 10 | 2 | 9.5 | A c40: the structure is present, but the markdown is corrupted in the notebook itself (`$R`, `$x_k$`, `\b`, `\r`, `\a` escapes lost, e.g. "^2_{OOS}", "($)", "egin{pmatrix}", "\x07sset_16"), so the characteristic names are unreadable. Many numbers are unprinted or fabricated. Its conclusions contradict its own tables. Real identities (US, Japan, Nikkei, Gilt, BoJ) are asserted for anonymised data, and penalty selection is not described. B c103–106: a complete paper; Table 3.31 prints every headline number; negative results are stated as "not detectable" with power; limitations are given. −0.5 for density, length and process meta-text |
| | **Subtotal** | **60** | **27** | **59.5** | |

### Totals

| | P1 (20) | P2 (20) | P3 (60) | **Total (100)** |
|---|---|---|---|---|
| **Solution A** | 18.5 | 15 | 27 | **60.5** |
| **Solution B** | 19.5 | 19.5 | 59.5 | **98.5** |

---

## 2. The five most consequential differences

**1. The macro placebo (P3, ~8 pts). B is right.**
- A's placebo panels overwrite the four raw `g_*` columns, but model M3 is fitted on `g_*_class_*` and `c_*_class_*` interaction columns built before the shift. Country macro is never shifted at all.
- The "placebo" is therefore the real macro model: all three shifts print −2.972% / −3.303%, identical to M3 (cell 39 output).
- The write-up (cell 40) then reports −2.715%, −3.084% and −2.890%, numbers that no cell prints. It concludes that the placebo "proves" macro is noise, adding the false statement that shifted macro is "by construction orthogonal".
- B rebuilds the whole pipeline from circularly shifted raw series at 8 shifts (c91). Real macro ranks 4 of 9 (gain −0.123%, SE 0.085%), and B discloses that the wrap-around reaches OOS months when s ≥ 72.

**2. How hyper-parameters are chosen (P3). B is right.**
- A fixes ridge α = 50 (M2) and α = 200 (M3) on unstandardised inputs, with no training-window selection and no stated rationale. The exam requires every penalty to be chosen on training data only, and a reader cannot rule out that these values were picked after seeing OOS results.
- The consequence is material. A's "macro collapses R² to −2.97%" measures a nearly unpenalised 39-slope fit. B shows that tuned ridge M3 scores −0.08%, while OLS M3 scores −4.7% expanding and −20.4% static.
- B scales inside the window, tunes on three forward annual folds with the order asserted at every refit, and also reports a hindsight upper bound (0.087%) showing that no penalty on any grid beats the class means.

**3. Where the portfolio gain comes from (Q3). B is right.**
- Both find a net Sharpe of about 0.43–0.44 for P1, against 0.30 for RP.
- A calls this "decisive outperformance", driven by characteristics, with "positive trend-following alpha". That is contradicted by A's own α of +0.52% with t = 1.56, not significant.
- B shows that P1's weights correlate 0.9994 with the class-means-only portfolio. Adding the class-means tilt as a regressor gives β 0.95, R² 0.998 and α t = 0.82. Frozen at 2010 the portfolio scores Sharpe 0.27. The within-class long-short P3 earns a gross Sharpe of 0.08.
- The "positive result" in A is precisely the kind the exam says earns no credit: unexplained, and here explained away by B.

**4. The answer to Q1 (forecastability). B is right.**
- A's abstract claims "resilient predictability within equities and bonds". Its own Table 6 gives bonds −0.306% and equities +0.054%, and its M2 ridge (−0.001%) is *below* the class-means-only M1 (+0.087%), even though the write-up labels M2 the "optimal parsimonious model".
- B's pre-registered statistic is 0.042% (SE 0.241%). Characteristics beyond class means add −0.045% (SE 0.027%, ±2 SE: −0.099% to +0.009%), and B states the power limit (0.48% needed to pass 50% of the time).
- An honest negative result is what the exam rewards.

**5. Reporting integrity in P2.3 and throughout. B is right.**
- A's 2.3 answer states measured drifts of −0.0204 / −0.0020 / −0.0002 that "match the formula". The cell prints −0.018645 / −0.003788 / −0.000233, and I re-ran it to confirm.
- A also measures a single step rather than "a few hundred nights".
- B runs 500 desks × 500 nights per n plus a precise 2,000 × 2,000 run at n = 50 (1/n and 1/(n−1) differ most at small n). It tabulates n × drift ≈ −1 with SEs, and reports honestly that its own unbiasedness CI misses 1 at one start (p = 0.0088).

*(Close sixth: the breadth of the data audit. B finds the stale x1 values, the six exact duplicate columns, the hidden global copies and per-country x11 rebuild errors. A's audit lines are partly hard-coded text.)*

---

## 3. Trust issues

### Solution A: reasons to distrust its results

1. **A test that cannot work as coded:** the placebo (cell 39). It is identical to M3 by construction, and the output shows three identical rows.
2. **Numbers in answers that no cell prints, or that contradict what cells print:**
   - 2.3 drift values (c29 vs c28 output).
   - Placebo R² values (c40).
   - Class statistics in c40: "31.0%" vol and Sharpe 0.194 / −0.023 / 0.282 / 0.405. The printed class portfolios give 0.412 / −0.030 / 0.327 / 0.508, and per-asset averages (my check) give 0.211 / −0.030 / 0.290 / 0.460, so the write-up matches neither. The best and worst months (−54.65%, +91.52%, −16.29%, −25.36%, −12.08%) are correct per-asset extremes but are never printed.
   - Characteristic-identity correlations (ρ = 1.0000, 0.9934, −0.93 to −0.98, 0.85–0.98), "69.4%", VIX skewness "+2.06", ρ = −0.295, F = 508.2, and the SE and p columns of the attribution table: none is printed.
   - 1.4's "≈116 / ≈43 distinct values"; 2.2's "over 90% below the starting value".
3. **Hard-coded "audit" output** presented as executed results (cell 39). The T3 line lists back-filled assets by hand and misses asset_22. "34 annual + 26 quarterly" is typed in (B's computation gives 35 and 26). "Sweden, Japan, Switzerland" and "Confirms 97% redundancy" are text strings. The PCA behind the redundancy claim is fitted on unstandardised columns, so it measures scale, not redundancy.
4. **Hyper-parameters with no documented origin** (ridge α = 50 / 200). This cannot be distinguished from tuning on the test window.
5. **Conclusions contradicted by its own numbers:**
   - "predictability within equities and bonds" (bonds −0.306%);
   - M2 called "optimal" although M1 beats it;
   - "decisively beat" with α t = 1.56;
   - "Gender zero signal" against a permutation importance of 0.019;
   - x11 "≡ x86 − x12 … exact collinearity" at ρ = 0.988 (per-country as low as 0.871, per B).
6. **Coefficient paths are computed but never displayed**, yet the write-up names the drivers of predictability (momentum, carry, reversal) as findings.
7. **The write-up is corrupted in the notebook itself** (lost `$…$` and backslash escapes), so key sentences do not render.
8. **Anonymised entities are asserted as real** (US, Japan 10Y JGB, Nikkei 225, UK Gilt mini-budget, BoJ YCC, Swiss unpeg). These are speculation presented as fact.

### Solution B: reasons for caution (none undermines a verdict)

1. **The pre-registration claim cannot be verified, and B itself partly undercuts it.** Section 3.10 admits the design first appears in a commit (591c8ff) made *after* the development ledger had run. Diagnostics were added "after an independent review of the first full run" and "approved by the student after the final audit", so the OOS window was consulted more than once. All of this is disclosed, and every verdict is negative, so it cannot have produced a false positive.
2. **The month bootstrap treats months as independent.** B's own block bootstrap gives SEs a median 1.19× larger. This is acknowledged.
3. **The placebo wrap-around puts end-of-sample macro into early OOS months** for s ≥ 72 (up to 49 months at s = 120). This is disclosed and quantified.
4. **Some audit-table entries are literals** ("20 lag spot checks", "causality True"). They are backed by `assert` statements in c68, which would have halted execution on failure, so they are acceptable.
5. **Outputs match the code:**
   - My re-run of B's 2.2 reproduces every printed statistic exactly.
   - B's shared quantities agree with A's independent computation.
   - I found no number in B's write-up that Table 3.31 or an earlier cell does not print.

---

## 4. Over-engineering vs missing elements (fairness in both directions)

**Where B is over-engineered or over-long:**
- **Problem 1:** 1.2 appends a "technical note" to the two required sentences. 1.3 ("two or three sentences") and 1.5 run to several paragraphs with lecture citations.
- **Problem 2:** 2.3 ("one paragraph") adds bullet lists, tables and a four-point reconciliation with 2.1. 2.4's paragraph is very long.
- **Problem 3:**
  - It has 38 specifications plus 9 post-hoc additions, a power analysis, Bonferroni bounds, block bootstraps, leverage diagnostics and an 8.5-minute runtime. Much of this goes beyond what a 60-point project needs.
  - The write-up is dense and carries process meta-text (commit hashes, "at the student's request", "independent review") that a research paper should not contain.
  - The exam sets no length limit for Problem 3, and the extra work is clearly labelled and makes the negative result credible, so I deduct only 0.5 there, plus 0.5 each in P1 and P2 for the sentence and paragraph limits.
- **Genuine omissions in B:** no boosted trees (justified by runtime), no cross-country macro transforms or widened panel (disclosed), no outside data (justified). None of these is required.

**Where A is missing required or expected elements:**
- **Know your data:** no per-asset table, no rolling 12-month Sharpe, no within-class characteristic distributions.
- **Tables:** no table listing each predictor.
- **Data traps:** no search for the extended-file duplicates the exam explicitly flags.
- **Models:** no hyper-parameter selection; a single evaluation scheme; no stability evidence; no per-class models; no uncertainty measures.
- **Macro:** no working placebo.
- **Portfolios:** no sub-periods; a single cost level.
- **Answer format:** the "Direct Verdict / Key Numbers / Mechanism / Actionable Conclusion" template breaks every sentence and paragraph limit in P1 and P2 while adding little.

**Credit A deserves:**
- Its core Problem 3 pipeline is leak-free and mirrors B's design (vol-scaled target, 2003 start, 2011–2024 OOS, within-class ranks, a 2-month macro lag, trailing z-scores, country-7 differentials).
- It independently found the x10 same-year-average look-ahead.
- It computes the benchmark portfolios correctly (its numbers match B's).
- It ran an attribution regression.
- Its Problem 1 and most of its Problem 2 numbers are correct.

**What costs A the marks:** unreliable reporting and interpretation rather than bad mechanics.

**Final marks: Solution A 60.5 / 100; Solution B 98.5 / 100.**
