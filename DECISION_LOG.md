# Decision Log — `Final_project_claude.ipynb`

This log records every choice, assumption and judgment call in the reference solution: what was chosen, why, what was considered instead, and what was *not* done. It was rewritten after a full re-evaluation of the notebook against the exam, the data, the eight lectures and the AI Coding Guide. `REVIEW.md` lists every error that re-evaluation found and how each was fixed. This file describes the design as it now stands.

Conventions: **Given** = dictated by the exam; **Choice** = our decision; **Not done** = a real gap you may want to close.

---

## Problem 1: Trees and Ensembles

### 1.1 — baseline, unpruned tree, leaf sweep
- **Given:** `StratifiedKFold(5, shuffle=True, random_state=7034)` and `random_state=7034` on every model; leaf grid {2, 3, 4, 6, 8, 12}; ties go to the smaller size. Shuffling is correct here: rows are unrelated people (this is the contrast 1.5 asks about).
- **Baseline** = 1 − mean(Purchased) = 64.25%.
- **Tie-break** `max(leaf_acc, key=lambda k: (leaf_acc[k], -k))`, which is correct for "take the smaller". 3 and 4 leaves tie exactly. We checked why: their out-of-fold predictions are identical for all 400 people, because the fourth leaf only splits a group that is already classified.
- **Extra (not required):** accuracy-vs-size plot; `export_text` of the 4-leaf tree.

### 1.2 — plot and description
- `plot_tree(..., feature_names=Xa.columns)` as required; `class_names` and `filled` for readability.
- **Description** read off the fitted tree: root `Age <= 42.5` (everyone older is predicted to buy); for 42-or-younger, `EstimatedSalary > 90,500` predicts buy. Gender is unused. Written as two sentences (the rule, then "Gender is never used"), within the exam's one or two.

### 1.3 — RF(300), GBRT(100)
- **Deliberately untuned.** Only the exam's settings; everything else is sklearn default. The question asks you to report the result "whichever way it comes out".
- **Added:** per-fold accuracies and paired fold t-statistics (1.24 and 0.98), so that "the ensembles do not beat the tree" is not overstated as "the tree is better".
- **Explanation** consistent with Lecture 8: forests average deep, unpruned trees (p.44); out-of-the-box boosting loses even to the forest (p.61).

### 1.4 — MDI vs permutation importance
- **Given:** 70/30 split, `random_state=7034`, `stratify=ya`, 300-tree forest.
- **Choice:** `n_repeats=30`, `random_state=7034` for `permutation_importance`. Lecture 8 p.57 says to repeat and average. The ranking (Age > Salary) holds in every repeat.
- **Mechanism** cited to Lecture 8 p.58 ("MDI rewards a column for merely offering many places to split"), with the actual cardinalities printed (Salary 117, Age 43).

### 1.5 — shuffled CV on time-ordered data
- **Written answer only.**
- **Mechanism:** persistent predictors and regimes, so a test month's neighbours share its information. Returns themselves are at most weakly autocorrelated (median lag-1 autocorrelation in the course panel ≈ 0.02).
- **Remedy:** expanding or rolling time-ordered validation with a gap.
- **Direction:** optimistic.
- **Lecture 5 p.56 example:** it is cited for what it measures, namely random folds under-regularizing the lasso and costing 0.036 of test $R^2$.

---

## Problem 2: Training on Your Own Output

### Global setup
- **`SEED = 0` is a placeholder.** Set it to the last four digits of your student ID. Every specific Problem 2 number is for seed 0. What does *not* depend on the seed:
  - the −1/n drift and its exact value $\log 2 + \psi(\tfrac{n-1}{2}) - \log(n-1)$;
  - the lognormal approximation at night 2,500;
  - the qualitative collapse.
- **Choice:** `np.random.default_rng`, with a fresh generator per experiment, all seeded with `SEED`. The "one desk" of 2.2 is a separate simulation, not desk #1 of the cohort. Threading one generator through the problem would be equally valid.

### 2.2 — one desk, then a thousand
- **Given:** night-1 library rescaled to sample variance exactly 1; fits with `var(ddof=1)`; draws from $N(0, \hat\sigma^2)$; full replacement. Night 2,500 = the 2,500th fit after 2,499 random steps.
- **Vectorized** 1,000 × 500 draws per night. This is a performance choice only.
- **Added:** the story's implied $\hat\sigma^2 = (0.24/2.8)^2 = 0.0073$ and its percentile in the cohort (48.8th, i.e. the median desk), and the one desk's percentile (41.6th). All numbers in the answer are printed by a cell.

### 2.3 — unbiasedness, drift, reconciliation
- **Unbiasedness check:** two starting levels (1 and 4), 5,000 one-step draws each, with Monte Carlo standard errors.
- **Drift:** `avg_log_change(n, n_desks=500, n_nights=300)`, inside the exam's "a few hundred". Standard errors are reported, plus the exact digamma value. The simulated drift is within 1.5 s.e. of it for all three n.
- **Reconciliation.** $\log\hat\sigma^2$ is an exact random walk with i.i.d. steps $\log(\chi^2_{n-1}/(n-1))$. Its night-2,500 distribution is ≈ $N(-5.01, 3.17^2)$: median $e^{\mu}$ ≈ 0.0067 and mean $e^{\mu+s^2/2} = 1$. Half the mean comes from desks above ≈ 152, which have probability 0.08% each. A lognormal cohort simulation (2,000 cohorts of 1,000) shows the sample mean has median 0.67 and is below 1 in 76% of cohorts.
- **Histogram:** it reuses the 2.2 cohort, so the 2.2 thousand-desk numbers and the 2.3 histogram and reconciliation describe the same draws. The unbiasedness check, the drift table, the cohort simulation and 2.4 each use their own generator.

### 2.4(a) — permanent real anchor
- **Reading of the exam:** night 1's library is the 500 real days (sample variance exactly 1, as the Rules require). From night 2 it is the 500 real days plus the latest 500 scenarios (1,000 returns).
- **Earlier version:** it duplicated the real half on night 1 (σ̂² = 0.999). The night-2,500 numbers are identical to machine precision, because the night-1 difference decays by about ½ per night.
- **Interpretation:** the anchor makes the update an AR(1) with coefficient about ½ and a floor of 499/999, so noise is bounded. The residual 0.944–1.059 band is essentially all synthetic sampling noise, since the real data alone give exactly 1.

### 2.4(b) — DJ30
- `dj.groupby('date').MrkRet.first()` (MrkRet is identical across stocks on a date); 1,511 days, 2016–2021.
- `kurtosis(fisher=True)` is excess kurtosis. We also report kurtosis excluding 15 Feb–30 Apr 2020 (5.28) to show where the fat tail comes from.
- The night-1 library is all 1,511 real days, which reproduces the story's 2.8% VaR (2.78%).
- A fresh `rng_b` draws the 500 night-1 scenarios.

---

## Problem 3: Research Project

### Target
- **Choice (the exam's recommendation):** $y_{i,t}=r_{i,t}/\hat\sigma_{i,t-1}$, with $\hat\sigma$ the 12-month standard deviation of returns through $t-1$ (`shift(1).rolling(12, min_periods=12)`).
- **Rejected:** raw returns (class A would dominate the loss); ranks (hard to turn back into positions).
- **Cost:** the first 12 months of each asset.

### Characteristics
- **Identification (data, not guesswork):**

  | char | what it is | evidence |
  |---|---|---|
  | `x2` | trailing 12m compounded return through t−1 (momentum) | corr 0.993 |
  | `x3` | ≈ −(trailing 60m return), long-term reversal / value | corr −0.94 |
  | `x5` | trailing 36m volatility | corr 1.000 |
  | `x1`, `x4` | class-specific fundamentals (carry / valuation type) | cross-sectional scales differ 100–1000x across classes |

- **Lags (given):** `x2`, `x5` as stored; `x1`, `x3`, `x4` shifted one month within asset.
- **Backfill (given warning):** leading runs of identical values in `x2`/`x3`/`x5` are copies of a later value. They are blanked, keeping the first genuine observation (`x3`: five assets, 30–31 months; `x2`: two; `x5`: one). Not done: filling them with a same-date cross-sectional median instead of dropping.
- **Standardization. Choice: within month and within asset class** (z-score), because an all-asset z-score of `x1` and `x4` mostly encodes class membership. The all-asset version is run as a robustness check. It is clearly worse for characteristics-only models and mixed with macro. Not done: rank transforms; class dummies.

### Macro
- **Lag.** One month for all series (the exam's minimum), **except `x10`, lagged 13 months.** `x10` changes only in January and matches the same-year mean of `x86` (corr 0.998). With a one-month lag it would feed forecasts the average of months not yet observed. Not done: longer publication lags for other series (we have no release calendar).
- **`x11` (stops for `country_9/11/12`, trailing gaps only):** rebuilt by **chain-linking**: the last observed value plus the subsequent change in the complete series `x86 − x12` (a real rate: a nominal rate minus inflation).
  - Pooled correlation of `x86 − x12` with `x11`: 0.988.
  - In a 36-month pseudo-out-of-sample test, chain-link MAE is 0.11 / 0.21 / 0.03 (`country_9/11/12`); a level rebuild with a mean offset gives 0.14 / 0.19 / 0.40, and forward-fill 1.22 / 1.52 / 0.64.
  - Rejected: forward-fill, which would freeze values through the 2021–23 inflation surge.
- **Duplicates:** ten extended columns duplicate six curated and all four global series exactly, and one extended pair is an exact sign flip. All eleven are dropped. The scan uses `allclose(atol=rtol=1e-6)` on ≥100 overlapping rows. Not done: a search for exact transforms (the extended file holds some pre-computed changes, e.g. of `x12` and `x6`), which is inert because the file is unused.
- **Extended file:** apart from `x86` (for the `x11` rebuild), not used as features. Its seven gappy columns (all trailing gaps) are forward-filled for completeness only. This is a scope decision, and a real gap if you have hypotheses for its series.
- **Global macro:** trailing 60-month z-score (`min_periods=24`) of the lagged series, i.e. information through $t-1$ only. Not done: an expanding z-score, which would start the macro sample earlier.
- **Country mapping:** own country for B/C/D; `country_7` for class A (the files' convention); plus differentials vs `country_7`. The differentials are identically zero for the 28 assets in `country_7` (26 A + 1 C + 1 D). Not done: cross-country averages or a widened panel for class A.

### Evaluation design
- **Out-of-sample window:** January 2010–December 2024, fixed in advance.
  - Initial training starts in 2001 for characteristics (the target needs 12 months) and 2002 for macro (the global z-score needs 24). The common-sample comparison is reported.
  - Not done: robustness to other split dates.
- **Refits:** every January (Lecture 5 uses `step = 12`); expanding, and rolling over 120 months. The rolling window is selected by month position (not calendar offset). Not done: other rolling lengths.
- **Benchmarks** computed once from all returns:
  - the pooled trailing mean through t−1 (the exam's / HW5's convention applied to the panel);
  - zero;
  - a class trailing mean for the by-class tables.

  $R^2_{OOS} = 1-\mathrm{SSE}_{model}/\mathrm{SSE}_{bench}$ (Lecture 4), plus a Newey–West t-statistic of the monthly loss difference. A vol-scaled $R^2$ is also reported, because class A carries 88% of the raw-return benchmark SSE.
- **Tuning (training data only).**
  - *Ridge and Lasso:* standardized inside a pipeline, with the penalty re-chosen at every refit on the last 36 months of that refit's training window (Lecture 5 pp.49–55). Lasso grid $2^{-10}$ to $2^{-1.3}$ (Lecture 5 p.53; the top sets every slope to 0). Ridge grid $10^0$ to $10^7$, spanning effectively OLS to effectively the null model. The choice hits both ends in different years, which we report as instability rather than hide.
  - *Boosting:* depth 2, learning rate 0.05, rounds (≤ 300) chosen on the same validation block (Lecture 8).
  - *Random forest:* 200 trees. Depth {2, 4, 8} × minimum leaf {20, 100, 500} chosen **once per feature set** on the initial training block (fit before 2007, validate 2007–09): chars-only (8, 100), chars+macro (4, 20). It is not re-tuned at each refit, because the forest is too slow for that; not done.
- **Reference model:** intercept-only (no predictors), under both schemes.
- **Global vs country macro (Q2 asks about each):** chars, chars+global, chars+country and chars+all on one common sample (training from 2002), for OLS/Ridge/Lasso (both schemes) and boosting (expanding), with Newey–West t-statistics. The positive country-only rolling cells get their own placebo test.
- **Search size reported:** 22 main backtests, 7 robustness runs, 28 macro-split runs, 20 + 10 placebo runs, 10 noise-control runs, per-class Lasso for 2 feature sets (8 class-level backtests), 10 forecast portfolios plus 2 rule variants.

### Placebo
- Every macro series is circularly shifted by 12, 24, 36, 48 and 60 **months** on the (country, month) grid. The exam names 36 months and asks for more than one shift; the rest are our choice. The roll runs over the months where macro exists, so the same rows are used as in the real run. A check line prints the effective shift.
- Run for OLS, Ridge, Lasso and boosting (expanding). Also run by class (OLS, Lasso), and for the country-only rolling models.
- **Noise control:** persistent AR(1) noise built exactly like the macro block, five seeds, for OLS and Lasso. It has 4 global series, 7 country levels, and 7 differentials against `country_7`'s noise (zero for the `country_7` assets). This tests whether real macro hurts merely because of its structure.
- Not done: a finer grid of shifts, or a distribution of placebo scores.

### Pooled vs per-class
- Lasso on both feature sets. Each class's penalty is tuned on its own validation block, and all runs are scored on the same benchmark.
- Not done: the other model families.

### Portfolios
- **Rule (choice):** $w \propto \hat y/\hat\sigma$ (= $\hat r/\hat\sigma^2$, the diagonal mean-variance weight), unit gross. With a constant forecast it collapses to risk parity, so deviations from risk parity measure the forecasts. Reported as diagnostics:
  - the original $\hat y\hat\sigma$ rule (concentrates in class A);
  - a demeaned, pure cross-sectional version.
- **Primary model:** chosen **before the out-of-sample period**, as the lowest 2007–09 validation MSE of $y$ among the ten model × feature-set candidates (each fitted on data before 2007). This picks the random forest with macro. Caveat: the forest's own depth and leaf size were chosen on the same 2007–09 block, so its score there is optimistic. The pick is robust, though: every leaf-20 forest configuration (validation MSE 1.539–1.550) beats the next candidate, boosting with macro (1.562). All ten forecasts are also reported, each with its alpha against the benchmarks. (The original notebook had picked its model from the out-of-sample grid.)
- **Benchmarks (given):** equal weight; risk parity $1/\hat\sigma$; TS momentum $\mathrm{sign}(\text{trailing 12m compounded return})/\hat\sigma$; all unit gross, monthly. They are computed in Section 3, before any model is fitted.
- **Costs:** 10 bps per dollar traded, times turnover $\sum_i |w_{i,t} - w^{drift}_{i,t-1}|$ (weights drifted by returns). Sensitivity at 0, 5, 10, 20, 30 and 50 bps. The order of the three benchmarks never changes (risk parity > equal weight > TS momentum). The forecast portfolio leads only below about 7 bps and falls behind equal weight above about 13 bps.
- **Exposure vs timing:** regression of each forecast portfolio on the three benchmarks (Newey–West).
- **Significance of Sharpe differences:** Jobson–Korkie test with Memmel's correction, for every pair.
- **Sub-periods:** 2010–14, 2015–19, 2020–24.

---

## What to challenge first

1. **`SEED = 0`**: set your own before comparing Problem 2 numbers.
2. **The portfolio Sharpe is fragile.** Fixing `x10`'s lag alone moved the Lasso-with-macro forecast portfolio from about 0 to 0.22 net Sharpe. The validation-chosen forest looks best before costs but has one of the worst out-of-sample $R^2$s. Trust the alpha regressions and the Sharpe-difference tests (none significant), not the ranking.
3. **Forest settings are tuned once, not per refit; macro lags are the minimum** (except `x10`); **the extended file is unused.**
4. **One out-of-sample window** (2010–2024). No alternative split was tested.
5. **The 2007–09 validation block is a crisis period.** It favoured the macro-driven forest, which then did poorly out of sample. A different validation block could pick a different primary model.
6. **Class-level $R^2$** depends heavily on the benchmark (pooled vs class mean vs zero). Read the three columns together with the intercept-only row.
