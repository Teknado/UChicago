# Final Project Plan — BUSN 41210 Financial Analytics, Autumn 2026

Status: **plan only.** Revised on 2026-09-27 after an external review (responses in `METHODOLOGY_LOG.md` §J). No notebook cell has been written or run. Coding starts only when you say go.
Companion document: [`METHODOLOGY_LOG.md`](METHODOLOGY_LOG.md) (the running justification log; every decision below has an entry there with alternatives, assumptions and evidence).

**How to read the citations.** `L5 p.56` = Lecture 5 PDF, page 56 (PDF page = slide). `cell 36` = cell 36 of `Final_Autumn_2026-1.ipynb` (0-based). `AIG §4` = AI Coding Guide section 4. Items tagged **[EXAM-DEFINED]** are required or defined by the exam notebook but not taught in any lecture (VaR, kurtosis, Sharpe ratio, risk parity, TSMOM, transaction costs, the placebo). Items tagged **[OUR CHOICE]** are design decisions with no single lecture source; the log records why. Nothing outside the lectures and the exam's own definitions is used unless it is listed under the open questions at the end.

---

## 0. Ground rules held throughout (from the AI Coding Guide and L5 p.58–59)

| Rule | Where it bites | How the plan honours it |
|---|---|---|
| **Fatal (a):** no shuffled split or shuffled folds on time-ordered data (AIG §4a; L5 p.58 #3, p.59) | P3 (monthly panel); P1.5 (explained) | P3 uses one hand-written date-mask splitter; every fold asserts `train.max() < validation.min()`. Shuffled folds appear only in P1, where the rows are unrelated people (L8 p.45) — and 1.5 explains why that would be wrong for months. |
| **Fatal (b):** no scaler / imputer / PCA fitted before the split (AIG §4b; L5 p.58 #1, p.47) | P3 | Every scaler is fitted inside each training window or fold only. Cross-sectional ranks are same-date; trailing z-scores and volatilities use past data only. No imputer is used (nothing is filled). |
| (c) `LogisticRegression()` is L2-penalised by default (AIG §4c; L6 p.38) | not used | Not used anywhere. If ever needed: `LogisticRegression(C=np.inf)` — on the installed sklearn 1.9.1, L6 p.38's `penalty=None` raises a FutureWarning (verified). |
| (d) `r2_score` / `.score` benchmark against the test mean (AIG §4d; L5 p.57) | P3 | Own `r2_oos(y, yhat, bench)` against a benchmark fixed before each forecast month (trailing mean) and against zero. `r2_score` and `.score` are never used for R². |
| pandas 3 / sklearn 1.9 traps (AIG §3) | all | `pd.concat` not `append`; `.ffill()` not `fillna(method=)`; `.items()`; no `sns.set()`; scaler in a Pipeline, never `normalize=`. |
| Describe data first; check against something known; verify the headline (AIG §6) | all | Each problem starts with shape/dtypes/counts checked against the data profile; known-answer checks are listed per part. |
| Every number comes from a cell (cell 1) | all | Every number quoted in an answer or the write-up is printed by a cell; a final P3 cell prints all headline numbers. |

**One data-path convention for all three setup cells (cells 4, 21, 35):** keep the exam's server line and add `import os; _DATA_DIR = _DATA_DIR if os.path.isdir(_DATA_DIR) else './'`. The same notebook then runs unchanged on the class server and on a local copy (cell 38).

---

## Problem 1 — Trees and Ensembles (Social_Network_Ads, 400 people)

All accuracies use the notebook's own `cv5 = StratifiedKFold(5, shuffle=True, random_state=7034)` and `cv_acc()`, so every model is scored on the **same folds** (L8 p.53: "the comparison is only meaningful because the split is shared"). `random_state=7034` on every tree, forest, boosting model, split and permutation. Shuffled folds are legitimate here because rows are unrelated people — a cross-section (L8 p.45). Trees need no scaling, and there are no missing values, so there is no preprocessing that could leak.

| Part | Method | Lecture grounding |
|---|---|---|
| **1.1** Baseline | "Nobody purchases" = majority-class accuracy (257/400 = 64.25% pooled; 63.75–65% fold by fold), stated first as the bar. | Accuracy "with the majority-class rate next to it" L6 p.57; root node [257, 143] L8 p.30/p.33; "compared to what?" L1 p.81 |
| | Unpruned tree: `DecisionTreeClassifier(random_state=7034)` (defaults = grown out), 5-fold CV accuracy; also its leaf count and in-sample accuracy as a labelled over-fitting diagnostic. | Grown-out tree on this very dataset overfits: L8 p.30 (the 400-row tree, ~62 leaves); choose size out of sample L8 p.29, p.32 |
| | Sweep `max_leaf_nodes` ∈ {2,3,4,6,8,12}; choose the best CV accuracy, **exact ties → smaller** (compare integer counts of correct people, not floats). Describe left side (too few leaves → underfit) and right side (more leaves → CV flat/falling while training accuracy rises → fitting noise). Show the 5 per-fold scores and their spread, not a formal test. | `max_leaf_nodes` as the size knob, L8 chose 3: L8 p.33; hump-shaped test accuracy L8 p.32; average fold scores L5 p.28, p.35; parsimony L5 p.25, L4 p.23; the chosen score won a 6-way search so is slightly optimistic L4 p.43, L5 p.57 |
| **1.2** Plot & explain | Refit the chosen size on **all 400 rows** (CV chose a size, not a tree), `plot_tree(..., feature_names=Xa.columns)`, plus `export_text` and an Age × Salary region plot. Answer in **exactly one or two plain sentences** (no "node", "gini"), e.g. "mostly buy / mostly don't buy" with each group's purchase rate; any technical note is kept outside those sentences. | Refit on all data at the chosen tuning value L5 p.36, as L8 p.33 does; reading `plot_tree` L8 p.25, p.33; leaf probability = class share L8 p.17; the data-plane picture L8 p.34 |
| **1.3** RF & GB | `RandomForestClassifier(n_estimators=300, random_state=7034)`, `GradientBoostingClassifier(n_estimators=100, random_state=7034)`, same `cv5`. Report either way, with fold spread. Explanation in 2–3 sentences, **split correctly**: a forest averages away *variance*; boosting adds shallow trees to remove *bias* (L8 p.51 — "Bagging and boosting are opposites"). With 400 people (1 person = 1.25 pp per fold) and a boundary that is close to one rectangle in Age × Salary, a small tree has little of either left to remove, and GB runs untuned (ν = 0.1, depth 3, 100 rounds). | RF = bootstrap + column subsampling, deep trees L8 p.41–45 (√p per split, p.43; B is a budget, p.45); boosting = shrunk residual fitting, ν and B are tuning parameters L8 p.50–52; out-of-the-box boosting lost to the forest L8 p.54–55 |
| **1.4** MDI vs permutation | `train_test_split(Xa, ya, test_size=0.3, random_state=7034, stratify=ya)`; RF(300) on the 70%; `feature_importances_` (in-sample MDI) vs `permutation_importance(rf, X_te, y_te, scoring='accuracy', n_repeats=50, random_state=7034)` on the held-out 30%; side-by-side table + bar chart; split counts per feature and distinct values per feature as evidence for the explanation. Recommend the **held-out permutation ranking** for a decision maker, with caveats (noisy on 120 people; no sign; not causal). | MDI is in-sample L8 p.47, p.56; permutation on held-out data, repeated and averaged, "answers the question you actually asked" L8 p.57; MDI rewards columns with many split points and splits credit across correlated columns L8 p.58; importance has no sign L8 p.60 |
| **1.5** Time-series CV (text) | Rows as months of one stock: shuffled folds train on the future (look-ahead) and put near-identical neighbouring months on both sides of the split; the CV then under-regularises and the reported accuracy is **biased upward** (optimistic). Fix: expanding (or rolling) walk-forward — train → validation → test in time order, hyper-parameters on validation only, test touched once, predictors lagged to t−1, any preprocessing refit inside each training window, benchmark = the "always up/down" rule learned from training months. | Order must be respected L5 p.27; "CV is not valid with time-series data" L5 p.48; expanding scheme L5 p.49–52; "the random fold sees the future … under-regularizes", −0.036 R² L5 p.56; fatal L5 p.58 #3, p.59; split by date "as it would be if the model were put to use" L6 p.30; L1 p.82 |

Non-CV accuracies (in-sample, training folds, the 1.4 held-out 30%) are always labelled **diagnostic**, because cell 3 says every *reported* accuracy is 5-fold CV.

---

## Problem 2 — Training on Your Own Output (VaR pipeline)

> **Commit-before-compute.** Part 2.1 is your own prediction, written *before* any Problem 2 code runs (L2 p.78: "Write down a number"; L4 p.26). This plan deliberately contains **no expected results** for Problem 2, and the theory notes are withheld from the log until 2.1 is written.

| Part | Method | Lecture grounding |
|---|---|---|
| **Setup** | `SEED = 2694` (the last four digits of your student ID). One `np.random.SeedSequence(SEED)` spawns named independent streams (one desk; 1,000 desks; each 2.3 experiment; 2.4a; 2.4b), so every cell is reproducible on re-run. Night indexing: night 1 fits the real library (sample variance exactly 1); nights 2…2,500 each fit the previous night's 500 scenarios → 2,499 random draw-and-refit steps. | Unbiased s² with ddof = 1 L2 p.56–57; VaR = 2.326 σ̂ **[EXAM-DEFINED]** cell 20 |
| **2.1** | Your prediction, supplied on 2026-09-27, goes into cell 23 **verbatim** before any Problem 2 code runs. It is never edited afterwards. 2.3 quotes it and reconciles each of its claims with the measured numbers, whichever way they come out. | L2 p.78; L4 p.26 |
| **2.2** | One desk exactly as the manual (fit `var(ddof=1)`, draw 500 from N(0, σ̂²) — scale = σ̂, not σ̂² — replace, report). Plot log σ̂² vs night; report σ̂² and VaR on night 2,500. Then 1,000 desks vectorised (one 1000 × 500 matrix per night, ≈ 35 s for 2,500 nights); night-2,500 mean, median, 5th/95th percentiles, max of σ̂²; fraction with VaR < truth/10 (with its Monte-Carlo SE as the sample SD of the 0/1 indicator / √1000); percentile rank of the one desk; where the story's 2.8% → 0.24% sits. A unit test checks the literal loop equals the vectorised version. | Monte-Carlo sampling distributions from a known truth L2 p.59–70, p.78–79, L4 p.26–27; log scale for multiplicative change L3 p.48–49; SE of a mean L2 p.63 |
| **2.3** | (i) Martingale check: 200,000 desks from the same σ̂²ₜ (two starting values), one step, average; report mean ratio with a 95% CI and say whether it contains 1 (reporting rule fixed in advance; non-rejection is "consistent with", not "equals"). (ii) Average nightly change in log σ̂² for n = 50, 500, 5,000 (500 desks × 500 nights each, SEs computed), and propose the formula relating it to n from the measurements. (iii) Histogram of log σ̂² across the 1,000 desks on night 2,500. (iv) Reconciliation table: expectation, median, 1,000-desk mean, share of the sum held by the ten largest desks; one paragraph; reconcile with your 2.1. Any theory beyond the lectures is a clearly labelled supplementary note, with any number it quotes computed in a cell. | Unbiasedness L2 p.56; CI / t for a mean L2 p.81, p.84–86; unbiased ≠ precise L2 p.71–72; CLT L2 p.63–64; logs L3 p.48–49 |
| **2.4(a)** | Each desk gets its own 500 real days = N(0,1) draws rescaled to sample variance exactly 1 (not demeaned — the smallest change that meets the rule). Library = those 500 real + tonight's 500 scenarios (1,000; last night's scenarios are replaced). Night-2,500 median, 5th/95th percentiles across 1,000 desks, side by side with 2.2; bands over nights on one plot. | same simulation design; degrees of freedom as information L2 p.57 |
| **2.4(b)** | `dj30.csv`: `groupby('date').MrkRet.first()` → 1,511 days. Daily sample volatility (`std(ddof=1)`) and excess kurtosis (`scipy.stats.kurtosis`, the exam's import, defaults `fisher=True, bias=True` — stated). Night-1 library = **all 1,511 real days** (this reproduces the story's 2.8% VaR). Compare empirical 1% VaR (−first percentile, `np.percentile` linear, stated) with 2.326 σ̂; kurtosis of real returns vs the 500 night-1 scenarios, with a reference band from 10,000 simulated normal samples of 500. Time-series plot of the real returns with both VaR lines. Then the paragraph (contains / lacks; loss of information vs accumulation of noise; implication for self-retrained pipelines; what stops it). | Normal-model answer vs resampling the actual data, "the gap between them is the size of the assumption" L2 p.80; simulated null band L4 p.27–28; plot the raw series L1 p.38; t is a fat-tailed normal L2 p.81. VaR, empirical VaR, kurtosis **[EXAM-DEFINED]** cells 20, 30 |

Runtime ≈ 3–4 minutes (measured on throwaway random matrices).

---

## Problem 3 — Cross-country asset return prediction (research project)

### 3.0 Design fixed before anything is fitted (cell 36 rule 2)

A markdown "Design fixed before fitting" cell plus one printed constants dict, placed **above** the benchmark cell. Deviations after that point go into a markdown list with what/why/effect.

| Item | Choice | Grounding |
|---|---|---|
| Sample | First target month **2003-01** (the 36-month trailing volatility first exists; every back-filled cell, last one 2002-08, is out of every design matrix) | data profile; cells 34, 37 |
| Initial training block | **2003-01 … 2010-12** (96 months, 4,800 rows; contains the GFC) | cell 36 rule 2; L4 p.43; L1 p.34 (nonstationarity) |
| Out-of-sample window | **2011-01 … 2024-12** (168 months, 8,400 rows), touched once; sub-periods 2011–17, 2018–24 | L4 p.43; L5 p.48–52 |
| Schemes | static (fit once at 2010-12); **expanding** (primary; fixed start, refit each December → 14 refits); rolling (96-month window) | L5 p.48–51 (step 12, as in L5 p.51); cell 37 |
| Tuning | Inside each training window only: three annual validation folds (fit on data up to year b−j, validate on year b−j+1, j = 3,2,1), mean fold MSE, ties → larger penalty, then refit on the whole window. Penalty parametrised as α/n so it means the same thing at every sample size. Chosen value must not sit at the grid edge (logged). | L5 p.48–52 (train → validation → test; alpha from validation only), p.35–36, p.27 & p.53 (grid edges), p.25 |
| Target | y = r_{i,t} / σ̂_{i,t−1}, σ̂ = 36-month SD of the asset's own past excess returns (ddof = 1), computed by us (avoids `x5`'s 0.004 floor) | cell 37 (recommended target); constant-variance assumption L3 p.2–3; var(b) ∝ σ² L2 p.71–73; "calm months and panics" L2 p.80 |
| Forecast benchmarks | Trailing mean of y through t−1, updated monthly (**pooled over assets = primary**, as you confirmed), per-class and per-asset versions always reported, and **zero**. Raw-return R² by class also reported against the literal per-asset trailing mean of raw returns. | cell 36; L4 p.45–46 ("why not out-of-sample Ȳ?"); L5 p.55–57 |
| Portfolio benchmarks | EW, risk parity (∝ 1/σ̂), TSMOM (∝ sign(12-month return)/σ̂); unit gross exposure; monthly rebalancing; computed and printed **before** any model | **[EXAM-DEFINED]** cell 36 rule 1 |
| Search size | The full list of specifications is fixed here (29 core + 9 pre-listed extras = 38 candidates; §3.3–3.4) and printed next to every headline, with the Bonferroni critical value computed in a cell | L4 p.25–28; L5 p.57 ("the size of the search belongs next to the number") |
| Expectations & power | Written down before computing (commit-before-compute); a short power calculation computed in a cell | L2 p.78; L4 p.26; var(X̄) = σ²/n L2 p.63 (flagged as our construction) |

### 3.1 Know your data (cell 37 step 1)

Integrity checks against the data profile. **Trap audit first:**
- 7 assets with leading back-filled `x2/x3/x5` → set to missing.
- `x10` is the **same-year average** of the monthly short rate stamped from January, so it looks up to 11 months ahead and a one-month lag does not fix it.
- `x11` stops early for countries 12, 9 and 11 at the 2021–23 inflation shock; rebuild it as `x86 − x12` and report the fit, pooled correlation 0.988.
- Six exact duplicate columns between the curated and extended files, and 26 hidden global columns in the extended file.
- `x1` is frozen for 5 currency-class assets in late 2024 (10 target rows).
- `asset_16`'s `x5` is floored at 0.004.

Then the full-sample return and predictor descriptives the exam lists:
- cumulative excess returns by class;
- a per-asset table of annualised mean, volatility, Sharpe and worst month;
- the correlation heat map ordered by class, with a 4 × 4 block table;
- rolling 12-month Sharpe by class and its year-to-year persistence;
- characteristic scale, distribution and persistence within class;
- identity checks: `x2` = the compounded past-12-month return and `x5` = the past-36-month SD.

**Tercile sorts are run on the training block only (2003–2010).** They relate predictors to future returns, so running them on 2011–2024 before the design is fixed would be snooping (L4 p.43; L5 p.41, p.58 #4). Hypotheses: `x1` carry, `x2` 12-month momentum, `x3` value / long-horizon reversal, `x4` 12-month change in the carry fundamental, `x5` 36-month volatility (the risk measure).

Grounding: L1 p.37–38, p.44 (`df.corr()` heat map); L1 p.35 (sorting as nonlinearity); L2 p.91 (t of a mean); L4 p.25–28 (Bonferroni for the 25 sort tests).

### 3.2 Feature engineering (cell 37 step 2)

| Block | Construction | Why / grounding |
|---|---|---|
| Lags | `x2`, `x5` as stored; `x1`, `x3`, `x4` shifted 1 month within asset; **every macro series shifted 2 months** (1-month minimum + release delay; 1 month as robustness); low-frequency annual/quarterly columns not used | cell 34; merge on publication date, not period end L5 p.58 #2 |
| Characteristics | **Within-class, within-month rank** of each lagged characteristic, scaled to [−0.5, 0.5] | Units differ by up to 3 orders of magnitude across classes; ranks are robust, bounded, parameter-free (cannot leak) and comparable for the penalty (L5 p.23; outliers L3 p.12–18). Within-class z-score as robustness (L7 p.23) |
| Characteristic levels (secondary spec) | Each asset's own lagged characteristic as a trailing 60-month z-score of its own history | Ranks can only rank assets within a class; this adds a time-series channel so Q2's "macro adds" test is not the only timing channel |
| Global macro `x6–x9` | log `x6`; trailing 60-month z-score (min 24 obs), then lag; enters with **class-specific slopes** | "Trailing information only" cell 37; interactions L3 p.53–54 |
| Country macro | 6 series: short rate `x86` (from the extended file, replacing the look-ahead `x10`), and from the curated file inflation `x12`, FX `x13`, log `x14`, log `x15`, `x16`. Each is taken as a **differential vs country 7**, then turned into a **trailing z-score per country** so that static country gaps (fixed effects) are removed and only time variation enters. Classes B, C, D get their own country's values × class. **Class A gets none** (no natural country, cell 34; parsimony L4 p.23) | cell 37 (country 7 the natural base; differentials); country fixed effects would otherwise pose as macro (verification finding) |
| `x11` | Diagnosed and rebuilt (`x86 − x12`, rebuild quality reported), **not added as a column**: it is an exact linear combination of two included series | cell 37; exact collinearity L3 p.40 |
| Extended file | Used only for the audit and for `x86`. Never screened against returns | multiple testing: "you will always find something" L4 p.25–28; screening then validating is cheating L5 p.41 |
| `x10` sensitivity (curated file only) | A pre-listed extra replaces `x86` with the curated `x10` at a **correct** lag of 14 months, so each month uses a completed prior-year average. A separate **leak demonstration** uses `x10` at the naive 1-month lag and is reported only in the leakage appendix, never as a result or a candidate. | Shows the result does not depend on using the extended file, and shows what the T1 trap would have done. L5 p.58 #2; cell 34 ("Getting this wrong silently introduces look-ahead") |
| Class intercepts | Class dummies (A reference) **left unpenalised**; ridge/lasso penalise only the predictor slopes (implemented exactly by within-class demeaning — verified on synthetic data) | Ridge penalises slopes, not the intercept L5 p.12; class intercepts L3 p.50; dummy coding L3 p.45, p.51–52 |
| Outside data | None | search size; release-date documentation (cell 34) |

The notebook ends this section with the predictor table and counts, as cell 37 asks. Main sets:

| Set | Contents | Columns |
|---|---|---|
| M1 | class intercepts only | 3 |
| M2 | M1 + 5 characteristic ranks | 8 |
| M3 | M2 + global × class + country × class | 42 |

### 3.3 Models × evaluation schemes (cell 37 step 3)

- **Models (core):**
  - OLS: the unpenalised baseline and the kitchen-sink warning (L4 p.47–48).
  - **Ridge, the primary model:** dense shrinkage, "suitable when features are weak and should still contribute" (L5 p.60). Standardised inside the window (L5 p.23).
  - **Lasso:** sparse selection, "best when you believe only a few features matter" (L5 p.60, p.17–24).
  - **PCR, added after review:** dense factors. It uses a StandardScaler, PCA(K) and OLS on the components, all fitted inside each training window and inside each validation fold. K is a tuning parameter chosen on the same validation folds as α, from {1, 2, 3, 4, 5, 6, 8, 10}. PCA is taught in L1 p.45–74, including as "pre-treatment for further ML" (L1 p.46). PCR is listed by the exam (cell 33) and appears in the GKX horse race (L1 p.36). L8 p.61 recommends fitting on principal components in high dimension. Standardising before PCA is our choice, flagged: L1 only demeans, and the columns here have different scales (L5 p.23).
  - Random forest: 300 trees, max_features = 1/3, minimum leaf 200, untuned (L8 p.43–45, p.55, p.61). The 200-row leaf is our low-signal prior; the leaf mean's standard error is about 0.07 in y units against about 0.45 for a 5-row leaf. A deep L8-recipe forest runs alongside it as an extra (§3.4), so the bias–variance trade-off is shown empirically.
  - The lecture's out-of-the-box default is the tree ensemble (L8 p.61–62). Linear wins "when the signal really is weak and linear — which, in asset pricing, is more often than you would like". So the forest runs on the identical harness as the check.
  - Ridge, lasso and PCR together give the dense-vs-sparse comparison. PCR and lasso run on M3 only: with five characteristics, M2 leaves nothing to compress or select (PCR with K = 5 is OLS).
  - Not used: KNN regression (not taught; curse of dimensionality, L6 p.37), neural nets, boosting (off; the tuning budget is a leakage risk, L8 p.55), elastic net (formula only, L5 p.11), and clustering (the class labels are known, L7 p.4).
- **Grid (core, 29 specifications):**
  - M1-OLS, expanding;
  - {OLS, ridge, RF} × {M2, M3} × {static, expanding, rolling} (18);
  - {lasso, PCR} × M3 × {static, expanding, rolling} (6);
  - ridge-expanding on global-only and on country-only macro (2);
  - per-class ridge on M2 and on M3, expanding (2). This tests pooled against segmented models for both the characteristic and the macro slopes.
- **Reported:**
  - R²_OOS = 1 − SSE_model / SSE_benchmark against the trailing mean(s) and zero (L5 p.55–57), model × scheme, by class, by sub-period, with in-sample R² next to it (L4 p.47–48);
  - uncertainty as a **month-resampling bootstrap SE** (each month keeps its whole cross-section) and estimate ± 2 SE (L2 p.80, p.85). The month is the unit because all assets share each month's shocks.
- **Which predictors carry the signal:** ridge coefficient paths over the 14 refits with sign stability (L5 p.22 style), and RF out-of-sample permutation importance per OOS year with `scoring='neg_mean_squared_error'` (L8 p.57).
- **Leak alarm:** any R²_OOS above ~2% triggers an audit before interpretation (monthly stock-level R²_OOS is about 0.3–0.5%, L5 p.57).

### 3.4 Macro, placebo, robustness

- **Macro vs no macro:** ridge M3 vs M2 under the same scheme, rows, folds and benchmark (refit without macro, as cell 37 asks), split into global vs country.
- **Placebo:**
  - every raw macro series is circularly shifted by s ∈ {36, 48, …, 120} months before any transform;
  - then the identical pipeline is run;
  - 8 shifts. Every placebo value used at an OOS date is a real past observation, because no shift above 130 is used. For s ≥ 72, the trailing-z window at early OOS dates reaches wrapped values; this affects normalisation only and is disclosed;
  - a persistence table shows how correlated each shifted series stays with the truth.
  - Macro is credited only if its gain beats **all 8** placebos and is positive against the bootstrap SE (null logic from L4 p.26–28).
- **Pre-listed extras (9, all under the expanding scheme):**
  - PCR with fixed K = 3 and with fixed K = 5, as a sensitivity check on the tuned K (2);
  - a deep forest on M2 and on M3, minimum leaf 5 ("grown deep", L8 p.44) (2);
  - macro lag 1 month (1);
  - within-class z-score instead of rank (1);
  - M2 and M3 plus characteristic levels, the timing channel (2);
  - curated-only macro, with `x10` at a correct 14-month lag instead of `x86` (1).
- **Diagnostic only, never a candidate:** `x10` at the naive 1-month lag (the leak demonstration).
- **Search size:** 38 candidate specifications. The Bonferroni critical value for 38 is computed in the notebook and printed next to every headline.
- **Evaluation-only cuts:** excluding March–May 2020, and excluding the frozen-`x1` rows.

### 3.5 From forecasts to portfolios (cell 37 step 4) [EXAM-DEFINED metrics; costs are OUR CHOICE]

- **P1 (primary):** w ∝ ŷ/σ̂, unit gross exposure, monthly rebalancing. If ŷ is constant it collapses to risk parity, so risk parity is the natural bar.
- **P3 (secondary):** within-class long–short, (ŷ − class mean)/σ̂.
- **Metrics:**
  - annualised mean, volatility, Sharpe (√12·mean/sd), max drawdown, worst month, turnover;
  - gross and net of costs;
  - cumulative-return and drawdown plots;
  - sub-periods.
- **Costs:** one proportional cost per unit of one-way turnover, headline 10 bp, grid {0, 5, 10, 25, 50} bp, plus the break-even cost. Costs are charged to the benchmarks too. Lectures do not cover costs; see Q5.
- **Exposure vs timing:**
  - regress net P1 on net EW, RP and TSMOM, and report α, β and R² (L2 p.94–97, L3 MLR);
  - hand test β = 1 against RP;
  - average correlation of P1's weights with RP's and TSMOM's;
  - maximum single-asset weight and `asset_16` share reported.
- **Cross-asset correlation and diversification (added after review):**
  - Correlation block tables (4 × 4, class-averaged) and the share of variance explained by the first principal component of the return correlation matrix, for 2003–2010, 2011–2019 and 2020–2024. Block structure can change in crises (L1 p.42–44; eigenvalue shares L1 p.43). These are descriptive only; no portfolio rule is chosen from them.
  - A class-risk-budget variant of RP and of P1. Inverse-volatility weights are used within each class, and each class sleeve is scaled by its own trailing 36-month volatility, so every class contributes equal ex-ante risk. The sleeve's volatility already reflects the correlation inside the class, so no covariance matrix is inverted. This addresses the concentration of unit-gross RP in the low-volatility classes (about 35–45% of gross in class D). [OUR CHOICE]
  - Not done: covariance-optimised (mean–variance or minimum-variance) weights. Mean–variance optimisation is not taught. A 50 × 50 sample covariance from 36–96 months of data is rank-deficient or badly conditioned ("might have rank less than n if T < n", L1 p.63), so inverting it needs shrinkage machinery the course does not cover.

### 3.6 Leakage audit, runtime, write-up

- **Leakage audit:** a markdown table maps each of L5 p.58's five leakage questions and AIG §4(a)–(d) to the cell where its guard lives. The executable checks are:
  - date-order asserts in every fold;
  - `n_samples_seen_` equals the training rows;
  - 20 random lag spot checks;
  - the `x2`/`x5` identities;
  - one causality test: truncate the raw data after 2010-12, rebuild, and assert that features, σ̂, benchmarks and benchmark weights up to 2010-12 are unchanged;
  - identity tests: P1(ŷ ≡ 1) = RP.
- **Runtime:** about 15 minutes estimated on a 4-core machine, serial except for the forest's own `n_jobs` (from synthetic-array timings; the deep forests add about 2 minutes). No process pool. The real runtime of each section will be printed.
- **Write-up** (cell 38):
  - Introduction: the question; why it matters for cross-country allocation; a preview; and what the literature leads us to expect. The literature is the lecture sources (GKX, L1 p.36; the 0.3–0.5% monthly R²_OOS, L5 p.57; L8 p.62) plus these papers, which you approved and will verify:
    - Gu, Kelly and Xiu (2020), *Empirical Asset Pricing via Machine Learning*, Review of Financial Studies;
    - Asness, Moskowitz and Pedersen (2013), *Value and Momentum Everywhere*, Journal of Finance;
    - Moskowitz, Ooi and Pedersen (2012), *Time Series Momentum*, Journal of Financial Economics.
  - Data: treatment of every trap; summary statistics; plots.
  - Methodology: reproducible from this section alone.
  - Results: in question order.
  - Interpretation: strong vs fragile evidence, and what we tried that did not work (every row of the ledger).
  - Conclusion: a half-page a portfolio manager can read alone.

---

## Your answers (received 2026-09-27)

| # | Question | Answer | Effect on the plan |
|---|---|---|---|
| 1 | SEED | **2694** | P2 streams come from `SeedSequence(2694)`. P3 keeps its own `P3_SEED = 7034` and never rebinds `SEED`. |
| 2 | 2.1 prediction | Supplied | It goes into cell 23 verbatim before any P2 code runs. It is not edited afterwards, and 2.3 reconciles it with the numbers. |
| 3 | P3 scope | **(b)** core plus extras, with PCR and the class-specific models added | 29 core + 9 extras = 38 candidates (§3.3–3.4) |
| 4 | Primary benchmark | **Pooled trailing mean**, with per-asset, per-class and zero also reported | P3-05 settled |
| 5 | Costs | **10 bp headline**, 0–50 bp grid, break-even cost | P3-15 settled |
| 6 | Literature | GKX (2020), the 0.3–0.5% benchmark, Asness–Moskowitz–Pedersen (2013), Moskowitz–Ooi–Pedersen (2012) | §3.6 write-up |
| 7 | Notebook | Either is allowed | In place in `Final_Autumn_2026-1.ipynb`; git history keeps the blank original |

Remaining before coding: your go-ahead.

---

## Errata from the final audit (2026-09-27)

The plan above is kept as approved. The final audit found these errors in it; the notebook and the methodology log use the corrected versions.
- Citations use PDF pages; in Lectures 1, 5 and 8 the printed slide number can be lower.
- "L2 p.91 (t of a mean)" should be L2 p.81; L2 p.91 is the t-statistic of a regression coefficient.
- "Within-class z-score as robustness (L7 p.23)": L7 p.23 is about scaling for k-means; the z-score variant is our choice.
- "ranks … comparable for the penalty (L5 p.23; outliers L3 p.12–18)": ranks cap the leverage of a characteristic value (L3 p.11–12); they do not bound the target.
- "Ridge penalises slopes, not the intercept (L5 p.12)": the ridge objective penalises the slopes it contains; leaving the class intercepts out is our choice.
- "boosting (off; the tuning budget is a leakage risk, L8 p.55)": the reason is runtime; tuning on forward folds would handle leakage.
- "clustering (the class labels are known, L7 p.4)": L7 p.4 also lists discovery as a use of clustering; the decision not to cluster is a scope and value judgement (proposal P-J).
- "coefficient paths … (L5 p.22 style)": L5 p.22 plots coefficients against the penalty, not across refits.
- `x11`: not an exact combination of two included series; excluded for gaps and near-redundancy.

