# Problem 3: judging the three research designs

**What I did.**
- Read all three drafts in full: A (linear-first, 754 lines), B (nonlinear learners, 761 lines) and C (evaluation and portfolio discipline).
- Read the authoritative exam text (notebook cells 33–40), the AI Coding Guide, the lecture digest, `exam_relevance.txt` and the data profile.
- Checked 16 citations per draft against `notes/Lecture_<n>.md`.
- Checked 17 slide pages against text extracted directly from the PDFs with PyMuPDF. Image rendering is not available here because `pdftoppm` is not installed. The pages were L1 p.35; L2 p.73, p.80; L3 p.54; L4 p.27, p.46; L5 p.48, p.51, p.56, p.57; L6 p.38, p.46; L7 p.6; L8 p.43, p.44, p.62. All matched the notes.
- Checked library defaults in the installed stack (sklearn 1.9.1, pandas 3.0.6, scipy 1.17.1, statsmodels 0.15.0):
  - `RandomForestRegressor` defaults to `max_features=1.0`;
  - `permutation_importance` defaults to `scoring=None` with `n_repeats=5`;
  - `partial_dependence` defaults to `method='auto'` and `grid_resolution=100`;
  - `GridSearchCV` defaults to `scoring=None` and refits on every row passed;
  - with `n_iter_no_change`, `GradientBoostingRegressor` calls `train_test_split` with its default shuffle.

No model was fitted on exam data.

## 1. Scores (1–10)

| Criterion | A: linear-first | B: nonlinear | C: evaluation/portfolio |
|---|---|---|---|
| (a) Lecture grounding: citations real and apt | 8 | 6 | 8 |
| (b) Leakage safety / AI-guide compliance | 9 | 8 | 9 |
| (c) Completeness vs the 4-step workflow, the 3 questions and "What to submit" | 9 | 8 | 9 |
| (d) Correctness given the data traps | 9 | 7 | 8 |
| (e) Feasibility / runtime on a class server | 7 | 5 | 8 |
| (f) Pre-registration and search-size disclosure | 8 | 7 | 9 |
| **Total (out of 60)** | **50** | **41** | **51** |

**Base for the synthesis: C (51).** A (50) is a close second. The final plan takes C's evaluation harness, pre-registration and portfolio machinery, and grafts on:
- from A: the windows, the feature engineering and trap handling, the placebo rule and the leakage tests;
- from B: a small number of specific elements.

### 1.1 Justifications

**Draft A**
- **(a) 8.** Citations are dense and almost all verified. Weaknesses:
  - L3 p.19–26 is cited for "the regression on PCs". Those pages cover MLR, and PCR is only a legend label (L1 p.36 boundary).
  - The primary M3 relies on correlation-matrix PCA, which L1 does not teach. The draft does flag this honestly.
- **(b) 9.**
  - Every pooled-across-dates fit lives in `fit_window()`, and the plan says where each AI-guide guard lives.
  - Leakage tests: a truncation test for causal features, a perturbation test for fitted objects, an `OOS_LOCKED` switch, a grep for banned APIs, and an equality unit test between the ridge fast path and `Pipeline(StandardScaler, Ridge)`.
  - It loses a point because the hand-written SVD ridge fast path is extra code that could itself leak or be wrong. It is unit-tested, but still a risk.
- **(c) 9.** Every workflow bullet is ticked:
  - M1 "class means" benchmark-plus row;
  - per-class vs pooled;
  - placebo (7 primary + 13 supplementary shifts);
  - scheme comparison;
  - portfolios P1–P6 with costs, break-evens and α regressions;
  - write-up table.
- **(d) 9.** The best trap handling of the three:
  - x10 is dropped and replaced by x86;
  - x11 is diagnosed and rebuilt but *not* added, because x86 − x12 is spanned (exact-collinearity argument, L3 p.40, L4 p.14);
  - back-fills become NaN, and a sample start of 2003-01 puts them out of reach;
  - frozen x1 is kept and flagged;
  - own σ̂ avoids the asset_16 floor;
  - extended-file duplicates and hidden global columns are handled;
  - class A gets no country macro;
  - the unique-row PCA stops c7 and the 26 A assets from reweighting the covariance;
  - placebo shifts s ≤ 108 are proved never to wrap future data into OOS placebo values.
- **(e) 7.** Its 8–12 minute estimate needs the SVD fast path, because tuning 9 (k_G, k_C) combinations × 50 λ with sklearn in every placebo would take about 1 hour. RF with a minimum leaf of 400 on 4,800 initial rows gives very shallow trees.
- **(f) 8.** Pre-registered primaries have numeric decision rules, and all 39 specifications are listed with Bonferroni values. There is an OOS lock and a pseudo-OOS development protocol. Missing: a hash or time stamp of the pre-registration, and an ex-ante power statement.

**Draft B**
- **(a) 6.**
  - KNN *regression* is labelled "[LEC], adapted". L6 teaches only the KNN classifier; L6 p.21 gives E[Y|X] as the regression risk minimiser but never the estimator.
  - 2-D partial dependence is used unflagged. L8 p.59–60 teaches 1-D PD only.
  - Several soft cites: L1 p.42 for block structure, L1 p.39 for "compare at comparable risk", L6 p.25 for "large K because SNR is low".
  - Otherwise accurate.
- **(b) 8.**
  - Excellent API awareness: it verified in the installed source that GBRT `n_iter_no_change` shuffles, and it warns about `TimeSeriesSplit` (splits by row count), `RidgeCV()` (leave-one-out over rows), HistGB `early_stopping='auto'` and the `permutation_importance` default scoring.
  - Asserts `n_samples_seen_`.
  - The larger surface (7 model families, tuned RF and GBRT) multiplies the chances for mistakes.
- **(c) 8.** Complete, and adds the class-tilt diagnostic (CT) and `ts_mom`. Weaker points:
  - only 4 placebo shifts (the 150-month one wraps);
  - the Q1/Q2 answers rest on the least interpretable model (RF).
- **(d) 7.**
  - x11 is spliced (the original until the stop, then the rebuild), which creates a level break at the stop date (c12 overlap RMSE 0.78).
  - `ts_mom` divides by x5, which carries asset_16's 0.004 floor.
  - It adds x85 from the extended file (a term spread) that D's x1 already contains (corr 0.87).
  - Otherwise correct: x10 dropped, back-fills, within-class z-scores, c7 differential, class A = 0.
- **(e) 5.** 72 specifications and an estimated 40–45 minutes (FAST mode 18–20), dominated by GBRT depth 3 and 2-D PD. That is risky for "runs top to bottom on the class server".
- **(f) 7.** Primary, decision rules, written expectations (a strong commit-before-compute element) and a 72-specification inventory with Bonferroni 3.47. But the primary RF has a tuned leaf and 7 model families, so the search is large and the headline sits on the most flexible model, against L8 p.62's prior.

**Draft C**
- **(a) 8.** Accurate overall, with explicit "[EXAM-DEFINED, not taught]" labels. Errors:
  - "the largest of 20 null t-stats is typically near 2.9 (L4 p.27)". The slide's 2.88 is the best of **100**.
  - "best null t near 3.6 for ~100 tests". 3.65 is the Bonferroni/95th-percentile critical value; the best observed was 2.88.
  - The L4 p.50 "set seeds" cite is an inference.
- **(b) 9.**
  - Future-perturbation test at three dates.
  - Identity self-tests: P1(ŷ≡1) = RP and P2(ŷ≡R12) = TSMOM.
  - Explicit `scoring='neg_mean_squared_error'`.
  - Grep assertions and a matrix-rank assert.
  - Closed-form ridge recomputation of the headline.
  - A lag canary.
- **(c) 9.** The strongest portfolio and attribution section:
  - P1/P2/P3;
  - drift-adjusted turnover;
  - break-evens via `brentq`;
  - joint and single α regressions with a partial F;
  - weight overlap;
  - sub-periods;
  - benchmark tables printed before models (T10/T11).
- **(d) 8.**
  - Correct on x10, x11* (a full replacement, no splice), back-fills, duplicates, frozen x1 and the σ̂ floor.
  - But MODEL_START = 2004-01 throws away 2003 needlessly, because it uses `min_periods=36` on the trailing z. The 72-month initial block leaves thin per-class training sets (D: 504 rows).
  - Placebo shifts up to 204/264 wrap future data into OOS placebo values, and there is no non-wrapping subset.
- **(e) 8.** A realistic budget (about 35 minutes serial, 10–12 with 4 jobs) built on sklearn `GridSearchCV` without custom fast paths. RF is untuned, GBRT is excluded, and there is a `RUN_FULL_PLACEBO` flag.
- **(f) 9.** The best pre-registration:
  - `PREREG` dict with a SHA-256 hash and a `DEVIATIONS` log;
  - optional git time stamp;
  - `LEAK_ALARM`;
  - ex-ante power calculation;
  - Bonferroni table;
  - a 42-forecast-specification ledger.

  It loses a point because **no single forecast benchmark is pinned for the verdict**. It calls the per-asset mean "the" trailing mean, reports four benchmarks and says "all reported, so not a search dimension". That leaves room for benchmark switching (L5 p.57, p.58 #5).

## 2. Citation spot-checks (✓ verified and apt; ~ page correct but the use stretches it; ✗ wrong)

### Draft A

| # | Citation | Use in draft | What the notes / PDF say | Verdict |
|---|---|---|---|---|
| 1 | L5 p.51 | 12-month test step, 12-month step | code: initial 180, val 84, test 12, step 12 (PDF ✓) | ✓ |
| 2 | L5 p.36 | refit on all data at λ̂ | CV-lasso algorithm step 4, "re-fit … to all of the data" | ✓ |
| 3 | L5 p.56 | expanding folds beat random | −0.041 random / −0.005 expanding / −0.019 rolling (PDF ✓) | ✓ |
| 4 | L5 p.16 | ridge path against λ/T | ridge path, λ/T from 0 to 10 | ✓ |
| 5 | L5 p.46 | `LassoLarsIC(criterion='bic')` code | verbatim on slide | ✓ |
| 6 | L5 p.60 | ridge "features weak and should still contribute" | verbatim | ✓ |
| 7 | L8 p.44 | RF avoids CV; set B and min leaf | verbatim (PDF ✓) | ✓ |
| 8 | L8 p.43 | p/3 per split | "√p for classification, p/3 or p for regression" (PDF ✓) | ✓ |
| 9 | L8 p.62 | signal weak and linear | verbatim (PDF ✓) | ✓ |
| 10 | L3 p.50 | common slope + class intercepts | OJ elasticity with brand intercepts | ✓ |
| 11 | L3 p.54 | separate slopes; "why not separately estimate" | verbatim (PDF ✓) | ✓ |
| 12 | L2 p.73 | var(b1) = σ²/((n−1)s_x²) | (PDF ✓) | ✓ |
| 13 | L7 p.6 | groups with similar characteristics → portfolios | "Finance: identify groups of stocks …" (PDF ✓) | ✓ |
| 14 | L1 p.46 | PCA as pre-treatment | verbatim | ✓ |
| 15 | L6 p.46–47 | threshold fixed ex ante (1/5 vs 0.195) | (PDF ✓) | ✓ |
| 16 | L3 p.19–26 | "the regression on PCs" | MLR only; PCR is a legend label (L1 p.36) | ✗ (stretch) |

### Draft B

| # | Citation | Use in draft | What the notes / PDF say | Verdict |
|---|---|---|---|---|
| 1 | L6 p.21 | E[Y\|X] minimiser, used to ground KNN regression | the table is there; KNN regression is never taught | ~ |
| 2 | L6 p.25 | large K because SNR is low | K trade-off only; the SNR link is inference | ~ |
| 3 | L8 p.39 | coarse geometric grid | verbatim | ✓ |
| 4 | L8 p.45 | B is a budget | verbatim | ✓ |
| 5 | L8 p.50–52 | ν, B tuned; held-out curve chooses B | verbatim | ✓ |
| 6 | L8 p.28, p.37 | trees find interactions | verbatim | ✓ |
| 7 | L8 p.59–60 | PD: brute force, 40-point grid from 5th to 95th percentile | ✓ for 1-D; **2-D PD not taught and not flagged** | ✗ (for 2-D) |
| 8 | L1 p.42 | block structure | biplot shows sector blocks in a crisis | ~ |
| 9 | L1 p.39 | compare at comparable risk | lesson is implicit (smooth or levered simulated record) | ~ |
| 10 | L5 p.4–5 | MSE = deviance / n | ✓ | ✓ |
| 11 | L2 p.78; L4 p.26 | commit before compute | ✓ | ✓ |
| 12 | L3 p.26 | same columns at predict | ✓ | ✓ |
| 13 | L7 p.23 | z-score formula | ✓ | ✓ |
| 14 | L4 p.25–28 | best noise t ≈ 2.9 | 2.88, best of 100 (PDF ✓) | ✓ |
| 15 | L5 p.27 | too much validation hinders training | ✓ | ✓ |
| 16 | L2 p.94–97 | α test; β = 1 by hand | ✓ | ✓ |

### Draft C

| # | Citation | Use in draft | What the notes / PDF say | Verdict |
|---|---|---|---|---|
| 1 | L5 p.35 | average fold MSE | ✓ | ✓ |
| 2 | L5 p.37–39 | grid edges: null / OLS-like | ✓ | ✓ |
| 3 | L5 p.18 (note) | sklearn α = λ/2n | implementation remark in the notes, not on the slide | ✓ (should say "notes") |
| 4 | L8 p.43–45, p.55 | RF untuned, p/3 | ✓ | ✓ |
| 5 | L4 p.27 | "largest of 20 null t near 2.9" | 2.88 is the best of **100** (PDF ✓) | ✗ |
| 6 | L4 p.26–28 (§2.6) | "best null t near 3.6 for ~100" | 3.65 = Bonferroni critical value; best observed 2.88 | ✗ |
| 7 | L2 p.63–64 | var(X̄) = σ²/n in the power calculation | ✓ (the calculation is a flagged construction) | ✓ |
| 8 | L2 p.100–101 | forecast error = noise + estimation error | ✓ | ✓ |
| 9 | L6 p.30 | do not use another model's output | ✓ | ✓ |
| 10 | L7 p.4 | known labels ≠ clustering | ✓ | ✓ |
| 11 | L6 p.56 | lift ↔ sorts | ✓ | ✓ |
| 12 | L6 p.57 | reporting checklist | ✓ | ✓ |
| 13 | L5 p.26 | prediction-driven selection | ✓ | ✓ |
| 14 | L4 p.50 | set seeds | the p.50 code has no seed; lesson inferred | ~ |
| 15 | L3 p.12–18 | leverage argues for ranks | ✓ | ✓ |
| 16 | L2 p.81 | t of a mean; Sharpe × √T | ✓ | ✓ |

## 3. What each draft contributes to the final plan

**Taken from C (base):**
- `PREREG` dict with hash and `DEVIATIONS` log;
- benchmark tables printed before models are imported (T10/T11);
- the `r2_oos` / `monthly_diff` / `diff_test` harness with self-tests and ρ₁(d);
- the ex-ante power statement, recomputed for T = 168;
- the Bonferroni table;
- `GridSearchCV` with explicit time folds and explicit scoring;
- untuned RF;
- the portfolio machinery: drift-adjusted turnover, P1/P2/P3, identity tests, break-evens, attribution with partial F, weight overlap;
- the future-perturbation test, the lag canary and `LEAK_ALARM`;
- the ledger discipline.

**Grafted from A:**
- windows: initial block 2003-01..2010-12 (96 months), OOS 2011-01..2024-12 (168 months), rolling window 96 months, 14 annual refits;
- **pooled trailing mean of y as the single pinned primary benchmark**;
- the M1 (class dummies only) benchmark-plus row;
- within-class rank transform;
- the lag table for every frequency;
- x11 diagnosed and rebuilt but not added as a column (spanned);
- country block {x86, x12, x13, log x14, log x15, x16}, differenced vs c7, zero for class A;
- class-interacted macro;
- non-wrapping placebo shifts {36, …, 108} as the decision set, inside a 20-shift distribution;
- fitted-object perturbation check;
- OOS lock with pseudo-OOS development;
- unique-row PCA (kept as robustness R5 only);
- within-(month, class) permutation of characteristics;
- the per-class row counts.

**Grafted from B:**
- written ex-ante expectations;
- `ts_mom` as one robustness specification (R6), computed with our own σ̂ instead of the floored x5;
- the within-class z ±3 variant as robustness R2;
- Bonferroni for the 25 tercile-sort tests (3.178);
- the extended banned-API list (`TimeSeriesSplit`, `RidgeCV()`, HistGB, GBRT `n_iter_no_change`);
- the `n_samples_seen_` assertion;
- the hand-rolled GBRT early-stopping recipe, as an *optional* specification that is off by default;
- the class-tilt diagnostic, implemented as P1 on M1 forecasts;
- the paired bootstrap of ΔSharpe;
- the decile fit plot.

**Dropped:**
- KNN regression (not taught);
- 2-D PD (not taught);
- CART as a candidate;
- GBRT by default;
- extended-file PCA (no hypothesis; adds to the search);
- x85 term spread;
- tuned RF leaf;
- A's own-history z (F39), replaced by the single-column `ts_mom`;
- the static-vs-timing ex-post decomposition, now flagged OUTSIDE SCOPE and off by default.

## 4. Judging summary

- **C scores highest (51/60)** for evaluation discipline, pre-registration and portfolio evidence.
- **A (50) has the most correct feature engineering and trap handling.** The synthesis uses C as the base and takes A's windows, features and placebo rule.
- **B (41) has the richest menu but the weakest grounding and feasibility.** It supplies only targeted grafts.
- The two decisive conflicts were resolved as follows:
  - **One pinned primary benchmark.** The pooled trailing mean of the vol-scaled target, updated monthly, with zero also required. Per-class and per-asset means are always reported and never used for the verdict.
  - **Windows.** A 96-month initial block with a 168-month OOS window (not C's 72/180 or B's 120/144), because it balances power (SE of annual Sharpe ≈ 0.27) against training depth and per-class sample size.
