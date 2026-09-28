# Problem 3 research design, Draft A: "parsimonious linear-first"
BUSN 41210 Final, Autumn 2026. Problem 3, "Cross-country asset return prediction" (60 points).

**Status: plan only.** No model was fitted on the exam data. No tercile sort, R², backtest or Sharpe ratio was computed. The only things run were:
- API, signature and default checks in the installed environment (Python 3.11, pandas 3.0.6, sklearn 1.9.1, scipy 1.17.1, statsmodels 0.15.0, joblib 1.6.0, seaborn 0.13.2);
- timing runs on **synthetic** arrays of panel size (§7);
- the Bonferroni-type critical values in §8.

Every data fact comes from `notes/data_profile.md`, which never compared a predictor with the return it would forecast. Exam cell 1 says "every number must come from a cell in this notebook", so the notebook has to **recompute** each of these facts. §1 and §2 say where.

**Citation keys.**
- "L<n> p.<page>" is the PDF page, as used in `notes/Lecture_<n>.md`.
- "Exam cell N" is the 0-based cell index in `Final_Autumn_2026-1.ipynb`: 33 = question, 34 = data and "Read this before modelling", 35 = setup code, 36 = "Two rules", 37 = suggested workflow, 38 = "What to submit", 39 = project code, 40 = write-up.
- "Profile §X" is a section of `notes/data_profile.md`.
- Anything taught in no lecture is marked **[EXAM-DEFINED]** (with the defining cell) or **[OUTSIDE SCOPE: needs approval]**.

**Design A in one sentence.** The primary specification is a pooled ridge regression of the vol-scaled next-month excess return on five within-class cross-sectional ranks of the lagged characteristics plus class dummies. Macro enters only through a principal-components compression (PCR device) fitted inside each training window. It is evaluated on an expanding window with annual refits, and every penalty and number of components is chosen by time-ordered validation inside the training window. Nonlinear models (random forest) are secondary robustness only (L8 p.62: "in asset pricing the signal is often weak and linear").

---

## 0. Global design (applies to every part)

### 0.1 Pre-registered primaries and decision rules (frozen before any fitting)

| Question (exam cell 33) | Primary specification (ID, §8) | Metric | Decision rule, fixed now |
|---|---|---|---|
| Q1: How much of next month's excess return is forecastable from lagged characteristics? | **F08**: M2 (5 within-class ranks + 3 class dummies) × Ridge × expanding window, annual refit | R²_OOS on y = r/σ̂ over the OOS rows 2011-01..2024-12 (168 months × 50 = 8,400 rows), against the **pooled trailing mean** (primary) and **zero** | "Forecastable" if R²_OOS vs pooled trailing mean > 0 **and** the t-stat of the mean monthly loss differential is > 1.96. The magnitude is always reported, even if it is negative (L5 p.57). |
| Q2: Does macro add anything? | **F20** (M3 = M2 + PCR-compressed global and country macro) vs **F08**, same scheme, window and benchmark | ΔR²_OOS = R²(F20) − R²(F08), plus the placebo rank | "Macro adds" only if ΔR²_OOS > 0, **and** the loss-differential t(F08 errors − F20 errors) > 1.96, **and** the real Δ exceeds all 7 primary placebo Δ's (36..108-month circular shifts, §3). |
| Q3: Does a forecast portfolio beat the simple rules out of sample, after costs? | **P1**: weights ∝ ŷ/σ̂ from F08, unit gross exposure, monthly rebalanced | Net Sharpe at 10 bp per unit one-way turnover. α from regressing on the EW, RP and TSMOM returns jointly. | "Beats benchmark B" if net Sharpe(P1) > net Sharpe(B) at 10 bp. "Beats beyond exposure" only if α > 0 with t(α) > 1.96. Break-even costs are reported regardless. |

A family-wise note is reported next to the three primaries: the Bonferroni value for 3 tests is z = 2.394. The search over all 39 forecasting specifications is also reported: the Bonferroni value for 39 is z ≈ 3.22 (§8, L4 p.28).

### 0.2 Timing, sample and windows (exam cell 36, rule 2: "fix … in advance … and say why")

| Item | Value | Why |
|---|---|---|
| Month index | m = 0 ↔ 2000-01-31, …, m = 299 ↔ 2024-12-31. `panel` sorted by (asset_id, date), 300 consecutive month-ends per asset (assert). | Profile §C: the panel is complete, so `groupby().shift(k)` is a k-month shift. |
| Target month t | The row's `date`. Every predictor must be known at the end of t−1 (exam cell 36). | Exam cells 34 and 36 |
| Modelling sample | Target months **2003-01..2024-12** (m = 36..299): 264 months × 50 = **13,200 rows**. | σ̂_{t−1} is a 36-month SD, so it first exists at 2003-01. Starting here also puts **all leading back-fills** of x2/x3/x5 (7 assets, last filled cell 2002-08; profile §D) before the first target month, so they cannot enter even through the 1-month lag of x3 (assert this). |
| Initial training block | **2003-01..2010-12** (96 months, 4,800 rows) | Covers one full cycle including the 2008 crisis, so the first models see a stress episode. That gives 600 rows per estimated coefficient for M2 (8 features). With 14 years left it matches the L5 p.51 design idea (long initial block, 12-month test steps). |
| OOS (test) window | **2011-01..2024-12** (168 months, 8,400 rows), touched once (L4 p.43) | It contains the 2011 euro stress, the 2014–16 commodity collapse, 2020 COVID and the 2021–23 inflation regime, which is the regime change at which x11 stops (profile §F). Two equal sub-periods: 2011–2017 and 2018–2024. |
| Refit dates | End of each December b ∈ {2010-12, …, 2023-12}: **14 refits**. The model fitted at b forecasts Jan..Dec of b+1. | L5 p.51 (test_size 12, step 12). Characteristic premia are slow-moving. §7 shows annual refits are cheap, so this choice is about interpretability (14 coefficient-path points), not compute. |
| Schemes | **Static**: fit once on 2003-01..2010-12. **Expanding**: fit on 2003-01..b. **Rolling**: fit on the 96 months ending at b. At b = 2010-12 all three coincide (2010-12 − 95 months = 2003-01, verified with `pd.DateOffset`). | L5 p.48–50 (expanding with a fixed start, and rolling); L4 p.47–48 (a single chronological split = static) |
| Inner (validation) folds | For an outer window [a, b], folds j = 3, 2, 1 fit on [a, b − 12j] and validate on the next 12 months. The criterion is the pooled validation MSE over the 3 × 12 months (1,800 rows). Ties go to the larger penalty or smaller k (parsimony). Then **refit on all of [a, b]** at the chosen value. | L5 p.48–49 ("expanding window CV", train → validation → test in time order), L5 p.56 (expanding folds beat random folds), L5 p.36 (refit on all data at λ̂), L5 p.25 (parsimony). The p.52 code does not refit on train+validation. Refitting is our choice, because otherwise the most recent 3 years would be thrown away. |
| Seed | `SEED = 7034` for RF, permutations and bootstrap. | L6 p.18 (set seeds); same seed as Problem 1 (exam cell 3) |

### 0.3 Target, benchmarks and the R²_OOS formula

- σ̂_{i,t−1} = std(r_{i,t−36}, …, r_{i,t−1}, ddof=1), computed from `excess_return`:
  `panel.groupby('asset_id')['excess_return'].transform(lambda s: s.rolling(36).std().shift(1))`
  ddof=1 follows L2 p.56–57. This equals stored x5 in 13,157/13,200 rows. The exceptions are asset_16's 43 months floored at 0.004 (profile §D). We use our own σ̂, **not** x5, for scaling and weights, so the floor never enters a denominator.
- Target: y_{i,t} = r_{i,t} / σ̂_{i,t−1}.
- The forecast ŷ_{i,t} maps back to a return forecast r̂ = ŷ σ̂_{i,t−1}, and to a position ∝ r̂/σ̂² = ŷ/σ̂ (§4).
- **R²_OOS** (L5 p.55, L4 p.45–46) is computed by hand; `r2_score` and `.score` are never used (AI guide 4(d), L5 p.57):
  $$R^2_{OOS}=1-\frac{\sum_{(i,t)\in OOS}(y_{i,t}-\hat y_{i,t})^2}{\sum_{(i,t)\in OOS}(y_{i,t}-b_{i,t})^2}$$
- Benchmarks b, all fixed now (L5 p.57: "do not quietly switch benchmarks"):
  1. **Pooled trailing mean (PRIMARY).** b_t = ȳ^{pool}_{t−1} = the mean of y_{j,s} over all 50 assets and all s from 2003-01 to t−1. It is **updated every month**, and the same number is used for every asset in month t. This is exam cell 36's "mean of returns through month t−1 only, updated each month", applied to the variable being forecast (the vol-scaled return). **Why pooled:** it is exactly what a pooled intercept-only model refitted monthly would forecast. So R²_OOS vs this benchmark measures the value of the predictors over "no information", which is Q1.
  2. **Zero** (exam cell 36; L5 p.56–57; L4 p.46).
  3. **Per-asset trailing mean** ȳ_{i,t−1} (secondary, always reported, never used for any choice). It shows whether persistent asset-level means (for example D bonds' higher Sharpe) would already beat our models. With about 0.25 annual Sharpe and 8–21 years per asset, these means are noisy (L2 p.62, p.72–73).

  Implementation: build cumulative sums over months, shifted by one month. Unit test: at 2011-01 the benchmark equals `y[(date>='2003-01')&(date<='2010-12')].mean()`.
- **Loss-differential t-test** (L2 p.59–70 CLT and sampling distribution of a mean; L2 p.88–93 t-test). Let d_t = (1/50) Σ_i [(y_{i,t} − b_{i,t})² − (y_{i,t} − ŷ_{i,t})²], and t = d̄ / (s_d/√168). With 50 rows every month, sign(d̄) = sign(R²_OOS). Months are the independent units, which absorbs the cross-sectional correlation (C block 0.725; profile §C). Serial correlation in d_t is ignored. We flag this: HAC is not taught (L2 p.80 names robust SEs only). The same construction on (ŷ_A − y)² − (ŷ_B − y)² compares two models.
- **Raw-return R²_OOS** (supplementary, by class only): benchmark r̂ = ȳ^{pool}_{t−1} σ̂_{i,t−1} and forecast ŷ σ̂. This lets the desk see return-scale fit without class A dominating the pooled number.
- **IS R²** (L4 p.44) for every refit, reported next to OOS (AI guide §6.4: IS should exceed OOS).

### 0.4 Where each AI-guide guard lives

| Guard | Where it lives in this design |
|---|---|
| (a) no `train_test_split` or shuffled folds on time data | The only splitter is the date-based generator `inner_folds(a, b)` / `outer_windows(scheme)` (§3). Every fold asserts `train_dates.max() < val_dates.min()` and `val_dates.max() < test_dates.min()`. `KFold`, `train_test_split` and `cross_val_score` are not imported in P3. A final cell greps the notebook source for these names (§6). |
| (b) scaler/imputer/PCA fitted before the split | **Everything that pools rows across dates** (StandardScaler, PCA loadings, winsorisation cutoffs, λ, k) is fitted inside `fit_window()` on training dates only, and refitted for every inner fold and every outer refit. **Causal per-date transforms** (within-month ranks, trailing z-scores, trailing σ̂, lags, differentials vs country 7) use only data dated ≤ the row's information date, so they are computed once on the panel. A **truncation test** proves this (§6). The ridge fast path fits mean/sd on training rows only, and a unit test shows it equals `Pipeline([('sc', StandardScaler()), ('m', Ridge(alpha))])` to 1e-12. A synthetic check gave a max difference of 7e-17. |
| (c) `LogisticRegression()` is L2-penalised | No classifier is used in P3 (the target is continuous). If a sign classifier were ever added, it would be `LogisticRegression(penalty=None)` described as the MLE (L6 p.38). Default: not added. |
| (d) `r2_score` / `.score` use the test-set mean | A custom `r2_oos(y, yhat, bench)` is used with the benchmarks of §0.3. `r2_score` is never imported. |
| pandas 3 / sklearn 1.9 pitfalls | `pd.concat` (no `df.append`); `.ffill()` (no `fillna(method=)`); `.items()`; `StandardScaler` in a Pipeline (no `normalize=`); no `sns.set()` (call `sns.heatmap` directly). Strings load as `str` dtype and dates as `datetime64[us]` (profile header). `.corr().values` is read-only, so `.to_numpy(copy=True)`. `sort_values('country')` is lexicographic, so **merge on (country, date) keys** (profile §F, §I.10). `lasso_path(X, y, alphas=grid)` returns alphas **sorted descending**, so index coefficients by the returned alphas (verified). The year-end alias is `freq='YE'` and month-end is `'ME'` (verified). |

---

## Part 1. Know your data

**(i) Goal.** Describe the panel before modelling (exam cell 37 step 1). The required items are:
- cumulative excess returns by class;
- a per-asset table of annualised mean, volatility, Sharpe and worst month;
- the correlation matrix ordered by class, with the block structure and what it implies for pooling;
- rolling 12-month Sharpe by class and its persistence;
- the distribution, scale and persistence of each characteristic within each class;
- a first look at predictive content from tercile sorts;
- a hypothesis for what x1..x5 measure.

**(ii)–(iii) Method steps and lecture grounding.**

| # | Step | Exact construction | Grounding |
|---|---|---|---|
| 1.0 | Describe the data first | `panel.shape` (15,000 × 10), `dtypes`, `head()`. Assert 50 assets × 300 month-ends, no NaN, asset_info matches the panel, class counts A 26 / B 8 / C 9 / D 7, all A in country_7, max\|wide − panel\| = 0. Print the country × class table. | AI guide §6.1; L1 p.81–82 (know where the data came from); profile §C |
| 1.1 | Cumulative excess returns by class | Equal-weight within class, W_t = ∏(1 + r̄_{class,s}), log y-axis, 2000-01..2024-12, 4 lines. The 2003-01 and 2011-01 boundaries are drawn as vertical lines. | L1 p.37–41 (visualisation), L1 p.38 (plot the raw data) |
| 1.2 | Per-asset table (50 rows) | Annualised mean = 12·mean; vol = √12·sd(ddof=1); Sharpe = mean/sd·√12; worst month (value and date); class; country. Sorted by class, then Sharpe. Class medians appended (profile §C values are the check). | Sharpe and annualisation are **[EXAM-DEFINED]** (cell 37 step 1). Not in L2 (L2 boundaries). ddof=1 per L2 p.56–57. |
| 1.3 | Correlation matrix ordered by class | `wide[order].corr()` as a 50×50 `sns.heatmap` (vmin −1, vmax 1, diverging), with class boundary lines. A 4×4 table of average pairwise correlations within and between classes, for the full sample **and** the training block 2003–2010. The training-block version is what justifies the pooling design. | L1 p.44 (`df.corr()` heat map; correlated columns carry little separate information) |
| 1.4 | Rolling 12-month Sharpe by class | For each class EW portfolio: √12·mean/sd over a trailing 12-month window, one plot with 4 lines. Persistence = corr(SR_{12m} at t, SR_{12m} at t+12) using non-overlapping consecutive windows, one number per class. | [EXAM-DEFINED] (cell 37 step 1). The correlation is L1 p.44. |
| 1.5 | Characteristic distributions and scale by class | A 5×4 grid of box plots (characteristic × class) in raw units (units differ, so the panels do not share axes). A table of mean/sd/min/max by class; median within-asset autocorrelation at lags 1/6/12/24; median within-class cross-sectional sd per month; share of within-class variance explained by asset means. Profile §D gives the check values (x1 and x4 differ by up to 3 orders of magnitude). | L1 p.37–41; L3 p.48 ("think about scale") |
| 1.6 | Identification checks (hypotheses for x1..x5) | These compare predictors with **past** returns or macro, never with the target. (a) x2 vs the own compounded return r_{t−12..t−1}: exact in 14,347/14,400 rows from 2001-01. (b) x5 vs own SD36: exact in 13,157/13,200 rows from 2003-01; exceptions are asset_16 at 0.004. (c) x1 (B) vs (x86 − x86_c7): corr 0.96; x1 (D) vs x85 − x86: corr 0.87. (d) x3 vs the 60-month cumulative log return: about −0.93/−0.98/−0.97 (A/B/D), −0.45 (C). (e) x4 (B) vs Δ12(x86 − x86_c7): corr 0.9996. | L1 p.44 (correlation + scatter); profile §D |
| 1.7 | **Tercile sorts, training block only** | For each characteristic j and each target month t from 2003-01 to 2010-12 (96 months), within each class, sort assets on the lag-aligned x_j. With n assets in the class and k = n // 3: bottom = the k lowest (`rank(method='first')`), top = the k highest, middle = the rest. That gives 2/3/2, 2/4/2, 3/3/3 and 8/10/8 for n = 7, 8, 9, 26. Record the EW next-month **raw** excess return (the notebook's wording) and the **vol-scaled** return y of each tercile. Report the monthly T−B spread per class, and the class-equal-weighted average across the 4 classes: mean, sd and t = mean/(sd/√96). Output is a table plus a bar chart. | L1 p.35 ("sorting returns by characteristics"); L2 p.88–93 (t-test of a mean); L7 p.6 (groups with similar characteristics form portfolios) |

**Why training block only (decision).** A tercile sort is a predictor-vs-future-return statistic. Running it over 2011–2024 before the design is frozen would let OOS returns shape the feature and spec choices. That is the "never use the same data twice" violation (L5 p.41) and the garden of forking paths L4 p.25–28 warns about, and it breaks the untouched-test-set rule (L4 p.43). So Part 1's predictive look uses **only 2003–2010**. After the freeze, the same sort on 2011–2024 is shown in Results as descriptive portfolio evidence and is never used for a choice. The return-only descriptives (1.1–1.4) and predictor-only descriptives (1.5–1.6) involve no predictor–target link, so they are shown on the full sample.

**(iv) Assumptions.**
- A1.1: EW within class is the class aggregate for descriptive plots only (it is not the EW benchmark).
- A1.2: The asset-class labels are **hypotheses**, not facts (profile §C): A = commodities (26 contracts, no country, a33 −54.7%/+91.5% in 2020 looks like crude); B = currencies vs the country-7 base (country 7 has no B asset; Sharpe about 0); C = equity indices; D = government bonds.
- A1.3: The characteristic hypotheses come from the identification checks:
  - x1 = carry (rate differential in B, term spread in D, dividend-yield-type in C, roll yield in A);
  - x2 = 12-month momentum;
  - x3 = value / 5-year reversal;
  - x4 = 12-month change in the carry fundamental ("carry momentum");
  - x5 = 36-month volatility (the risk measure).

**(v) Pitfall guards.**
- No tercile sort or predictor–return statistic touches 2011–2024 before the freeze. An `OOS_LOCKED` flag enforces this (§8).
- Rolling-Sharpe persistence uses non-overlapping windows. Overlapping 12-month windows are mechanically autocorrelated.
- The correlation matrix uses returns only. It gets a training-block version because it drives the pooling decision.
- Class-scale warning (exam cell 34): 1.2 and 1.5 show why neither a pooled raw-return regression nor an EW portfolio is scale-neutral.

**(vi) Outputs.**
- Figures: F1 cumulative returns by class; F2 50×50 heat map; F3 rolling 12-month Sharpe by class; F4 box-plot grid; F5 tercile T−B bars (training block).
- Tables: T1 per-asset (50 rows); T2 4×4 block correlations (full sample and training); T3 characteristic summary by class with ACF and a hypothesis column; T3b identification-check correlations; T4 tercile spreads (training block).

**(vii) The written answer must cover.**
- ☐ Cumulative returns by class, and what they show (D steady, B about 0).
- ☐ The per-asset table and the vol range (0.82% to 17.40%).
- ☐ Block structure (B 0.55, C 0.725, D 0.58 within; A 0.21; D negative with A and C), and why it implies (a) month-level dependence in pooled errors, which is why months are the resampling and test unit, and (b) pooled slopes with class dummies instead of one fully common model.
- ☐ Persistence of class Sharpe.
- ☐ Characteristic scale, distribution and persistence per class (x2 and x4 show the 12-month overlapping-window ACF signature).
- ☐ Tercile evidence (training block only, and why).
- ☐ A hypothesis for each of x1..x5, with the evidence.

**(viii) Open questions.**
- Q1.a: Show the OOS-period tercile sorts in Results after the freeze? Default: **yes, labelled descriptive**.
- Q1.b: Use the raw or vol-scaled return in the sorts? Default: **both**; raw is the notebook's wording.

---

## Part 2. Feature engineering

**(i) Goal.** Turn the files into a predictor matrix in which every value is known at the end of t−1 (exam cells 34, 36, 37 step 2). This covers:
- lag rules, including release delay and low-frequency look-ahead;
- missing values handled with past information only;
- cross-sectional standardisation of characteristics;
- trailing standardisation of global macro;
- the country-to-asset mapping, including class A;
- differentials vs country 7 and cross-sectional macro transforms;
- interactions and class dummies;
- a disciplined use of the extended file;
- the target and its mapping to positions;
- one predictor table with counts.

### 2.1 Lag rules (per variable)

Notation: "shift s" means that the value stored at month m is used for target month m + s, via `groupby(asset or country).shift(s)` on complete, sorted panels.

| Series | Stored as | Frequency (profile) | Shift used | Reason |
|---|---|---|---|---|
| x2, x5 | already lagged one month | monthly | **0** | Exam cell 34. x2 = r_{t−12..t−1} compounded and x5 = SD(r_{t−36..t−1}), verified in 1.6. |
| x1, x3, x4 | contemporaneous | monthly | **1** (within asset) | Exam cell 34 |
| x6..x9 (global) | contemporaneous | monthly | **2** | Exam cell 34: "a one-month lag is the minimum, and you may argue for more". We assume a one-month publication delay and use the value in the following month: data for month s are known at the end of s+1 and used for target s+2. The names are anonymised, so we cannot tell market series (known at month end) from statistical releases (CPI, activity, delayed). A **uniform** 2-month rule is the conservative choice (L5 p.58 #2, merge on publication date, not period end). Robustness spec F32 uses shift 1. |
| x86 (monthly short rate; extended file), x12..x16 (curated) | contemporaneous | monthly (x16 and ratings are step series) | **2** | Same as above |
| **x10 / x73** | January-stamped annual **same-year mean of x86** (exact in 208/300 country-years) | annual | **not used** | Up to 11 months of look-ahead. A 1-month lag does not fix it (profile §F). Replaced by x86 at shift 2, which carries the same information on time. The notebook demonstrates the look-ahead (country 7 in 2022: x10 = 2.228 all year while x86 goes 0.08 → 4.10). |
| **x11** | contemporaneous real rate, stops early for c12 (2020-10), c9 (2021-08), c11 (2024-03) | monthly | **not used as stored** | See 2.2 |
| extended, monthly / irregular | contemporaneous | 60 + 22 columns | **2** | Same as monthly |
| extended, quarterly | stamped at quarter start = same-quarter mean (x90, x147, x149, x150 verified) | 26 columns | **5** | The quarter ends at stamp + 2; with a 2-month release delay it is known at stamp + 4 and used for target stamp + 5. For example, Q1 (stamp Jan) is used from June. |
| extended, annual | January-stamped same-year mean (x71, x74, x118, x153 verified) | 39 columns | **14** | The year ends at stamp + 11; with a 2-month release delay it is used from March of Y+1 (Jan Y + 14). Rule of thumb from profile §G: ≥ 12 months plus the release delay. |

Frequency classification (code): for each extended column, collect the calendar months in which it changes, across all countries. If the set is {1} → annual; if it is within {1, 4, 7, 10} → quarterly; otherwise monthly or irregular. Assert that the counts match profile §G (39 / 26 / 82 / 1 constant).

### 2.2 Missing values and data defects (only-past-information treatments)

| Series / defect | Pattern (diagnosed in the notebook) | Treatment | Why it uses only past information |
|---|---|---|---|
| Leading back-fills of x2 (asset_3, asset_22), x3 (asset_1, 2, 5, 26, 21), x5 (asset_3), ending by 2002-08 | A constant run from 2000-01 equal to the first genuine value | (1) **Set the filled cells to NaN in the raw panel** (run length − 1 cells per asset, per profile §D). No trailing statistic can then ever use them; this matters for F39's own-history z-scores, which use min_periods=24, with any remaining NaN set to 0 (neutral) and the count reported. (2) **Excluded from the rank features by the sample start** (targets from 2003-01; x3 at shift 1 uses stored ≥ 2002-12). Assert that no filled cell is in any design matrix. | A back-fill uses the future (for example, asset_3's filled x2 contains returns it would "predict"). Removing it is the only clean fix (exam cell 37 step 2). |
| x11 stops early for c12 / c9 / c11 (no interior holes) | Trailing NaN of 50 / 40 / 9 months. c12 and c9 stop **at the 2021–23 inflation regime change**. | **Rebuild** x11 as x86 − x12. Report the rebuild quality on the overlap: pooled corr 0.988, RMSE 0.354; c7 exact (RMSE 0.010); c12 RMSE 0.78, corr 0.871. Also report the **stale-ffill error** the rebuild avoids: c12 mean 3.67 pp and max 10.1 pp; c9 mean 2.74 pp and max 4.76 pp; c11 0.21 pp. Figure F6b: x11 vs the rebuild for c9, c11 and c12, with stop dates marked. The rebuilt series is **not added as a separate column**, because x86 − x12 is an exact linear combination of two included series. That would be exact collinearity (L3 p.40; L4 p.14 rank deficiency). Its information is therefore fully in the model through x86 and x12. | The rebuild uses same-month x86 and x12 (both complete), shifted 2 like every monthly series. Forward-filling a stale value through a regime change is explicitly the trap (exam cell 37 step 2). |
| x1 frozen 2024-09..2024-12 for 5 of 8 B assets (a8, a17, a30, a31, a34) | Stale end-of-sample values while the rate differential moves (a17: −0.71 → −0.04) | **Keep as stored** and flag it. It affects 15 target rows (2024-10..12). | A stale value is what a desk would have received, so it is not look-ahead. Rebuilding x1 from x86 would be an approximation with an unidentified scale (/≈1190). |
| asset_16 x5 floored at 0.004 (2019-09..2023-03) | 43 rows | Keep x5 as the characteristic (the rank within D is unaffected if it stays the lowest in D; print the check). **Use own σ̂ for all scaling and weights.** | Both are trailing quantities |
| Extended gap columns x55, x82, x83, x113, x136, x163, x164 | All stop early (last observations 2022-11..2024-10), with no holes | **Dropped** from the extended block | Dropping is legitimate (exam cell 37 step 2). x113 stops in 2022-11 for 4 countries, so a forward-fill would carry 25 stale months into the OOS window. |
| Global and trailing z-score warm-up | The first 23 stored months lack a z-score | Covered by the sample start (target 2003-01 uses stored 2002-11 with 35 observations) | — |

### 2.3 Cross-sectional standardisation of characteristics (decision)

Within each (target month t, class) cell, for each lag-aligned characteristic j:
u^{(j)}_{i,t} = (rank_avg(x^{(j)}) − 1)/(n_class − 1) − 0.5 ∈ [−0.5, 0.5], using `groupby(['date','asset_class'])[col].rank(method='average')` (verified in pandas 3).

- **Within class, not across all assets.** Raw units differ by class. x1 is in percent for D and a monthly decimal for B; x1 and x4 differ by up to 1,000× across classes (profile §D). Ranking across all 50 assets would sort **classes** (D carry vs the rest; A vol vs D vol) rather than assets within a comparable market. That would turn characteristics into class-identity proxies, which the class dummies already cover (L3 p.57 confounding).
- **Rank, not z-score.** (a) It is robust to outliers: x3 in A ranges from −9.50 to 0.96; x2 has winsorised extremes; class A is fat-tailed (L3 p.17 outlier policy, without deleting anything). (b) It is bounded with near-equal variance across characteristics, so the penalty is fair (L5 p.23). (c) It is interpretable: the OLS coefficient on u^{(j)} is the predicted **top-minus-bottom-in-class** difference in vol-scaled return. (d) It is the standard in the cross-asset factor literature (background, not a method).
- **Cost:** ranks discard magnitudes and the class-wide time-series level of a characteristic (for example, "all bonds have high carry now"). That component is addressed by (1) the macro block (class-specific global slopes), and (2) robustness F39 (M2-TS), which adds each characteristic's **own trailing z-score** (expanding mean/sd of the asset's own history up to its information date, min 36 observations). F33 checks z-score vs rank (within-class z-score with ddof=1).
- Causality: a rank uses only the same-date cross-section, known at t−1, so it is computed once on the panel.

### 2.4 Global macro (x6..x9): trailing standardisation

- Transform: log(x6), because x6 is VIX-type and right-skewed (10.1–62.7). This follows L3 p.48–49: log for series on a percentage scale. x7, x8 and x9 stay as they are.
- On the **stored** series, compute a trailing z-score: z_s = (v_s − mean(v_{s−59..s})) / sd(v_{s−59..s}, ddof=1), using `.rolling(60, min_periods=24)`. Then shift by 2.
- Justification: exam cell 37 ("rolling z-score, … never with the full sample"). A rolling window adapts to slow level drift (x9 ranges from 0.022 to 0.129 with AC12 0.73; x7 AR1 0.984; profile §E).
- Causal, so it is computed once.
- Inside the model these four series are compressed by PCA (2.7).

### 2.5 Country macro and the country-to-asset mapping

- **Curated country set (6 series):**
  - SR = x86 (the short rate, replacing x10);
  - INF = x12;
  - FX = x13 (YoY % change of the effective exchange rate);
  - EPU = log x14;
  - GPR = log x15 (x15 > 0; min 0.004; heavily skewed);
  - RISK = x16.

  All are shifted by 2.
- **Transform: the differential vs country 7:** D^{(k)}_{c,m} = x^{(k)}_{c,m} − x^{(k)}_{7,m}. The notebook calls country 7 "the natural base" (exam cell 37). It is the base currency of the B assets: x1 (B) ≈ the rate differential vs c7, and x4 (B) = Δ12 of that differential exactly (profile §D). So the differential is the economically correct object for FX and a relative-value object for C and D.
- **Mapping:**
  - Classes B, C, D: the asset's own country differential (direct mapping via `country`; merge on (country, date) keys).
  - Country-7 assets (asset_24 C, asset_27 D) have a differential of 0 by construction. They are the base.
  - **Class A: no country macro** (its country-macro interaction columns are identically 0). Its macro information comes from class-specific global slopes. Reasons: A has no natural country (exam cell 34); the c7 convention would give a differential of 0 anyway; a cross-country average of levels would be another global series, which we already have (x6..x9); and parsimony (L4 p.23).
- **Cross-sectional macro transforms considered:**
  - Differential vs c7: **primary**.
  - Deviation from the 12-country cross-sectional mean: **robustness F31**. It gives the c7 assets non-zero information.
  - Rank across the 12 countries: **not used**. It is coarse and full of ties: x10 is identical for c2/c4/c6/c8, x14 for c10/c11, and x86 for c2/c6.
  - Widened panel (each asset sees every country's value): **not used**. That is 6 × 12 = 72 extra columns per row, a multiple-testing and n/p problem (L4 p.25–28; L1 p.34).
- Levels of country macro are **not used**. They are mostly country fixed effects (cross-sectional variance share 0.92 for x16 and 0.99 for x15), so they would act as noisy asset-identity dummies.

### 2.6 Interactions and class dummies

- **Class dummies:** 1[B], 1[C], 1[D] with A as the reference (R − 1 dummies; L3 p.45, p.51–52). The dummy trap is avoided (L3 p.52). Pooled slopes with class intercepts is exactly the L3 p.50 structure. The class-specific-slope alternative (L3 p.54) is estimated as separate per-class models (§3).
- **Macro × class:**
  - Global PCs enter as PC_g × 1[class] for all 4 classes (class-specific timing, no common main effect). A global risk state need not move bonds and commodities the same way (the correlation of D with A is −0.08).
  - Country PCs enter as PC_c × 1[B], × 1[C], × 1[D].
  - This is the interaction mechanism of L3 p.53–54.
- **Characteristics × macro state (secondary, F28 = M4):** u^{(j)} × (global PC1) for j = 1..5. The hypothesis, stated before fitting, is that carry and momentum premia are state-dependent (carry and momentum "crashes" in high-uncertainty states). L3 p.53: "the effect of x_j depends on x_k".
- **PCA sign convention** (so paths are comparable across refits): after each fit, flip each component so that its loading with the largest |value| is positive. This rule uses training data only.

### 2.7 Macro compression by PCA (PCR device), fitted inside each training window

- **Global block:** a T_train × 4 matrix of unique training months of the lagged, trailing-z global series → `Pipeline([('sc', StandardScaler()), ('pca', PCA(n_components=k_G))])`, with k_G ∈ {1, 2, 3}.
- **Country block:** unique (country ≠ 7, training month) rows of the 6 differentials (11 × T_train rows) → the same pipeline with k_C ∈ {1, 2, 3}.
- Fit on **unique** rows, not panel rows, so that country 7 and its 26 A assets do not reweight the covariance.
- k_G and k_C are chosen jointly with the penalty on the inner validation folds (§3). A scree plot of the initial training block is shown for description (L1 p.43).
- Grounding:
  - PCA as "pre-treatment of the data for further ML algorithms" and noise reduction (L1 p.46);
  - the PCA mechanics, eigenvectors of the covariance of **centred** data, keeping k PCs (L1 p.45, p.63–74);
  - the regression on PCs (L3 p.19–26);
  - L8 p.61 (fit models to low-dimensional principal-component factors).
- **Boundary note:** L1 PCA only demeans. Standardising before PCA (correlation-matrix PCA) is **not** in L1. It is justified here by the scale argument of L5 p.23 and L7 p.23: the differentials mix percent, index points and log units. L1 also gives no formal rule for k, which is why k is chosen by time-ordered validation (L5 p.48–52).

### 2.8 Extended file (x17..x164): hypothesis-driven use only

The multiple-testing problem (L4 p.25–28): among 148 columns of noise, "you will essentially always find something". So the extended file is used in only **two** ways, fixed now.
1. **Targeted substitutions with a stated hypothesis:** x86 replaces the look-ahead x10 (profile: x10 = the same-year mean of x86), and x86 − x12 rebuilds x11. This is one column, and it is disclosed.
2. **One exploratory spec (F29 = M5):** M3 plus k_E ∈ {1, …, 5} PCs of the cleaned extended differentials, × {B, C, D}. The whole block counts as one entry in the search disclosure. It is labelled exploratory and cannot change the Q2 conclusion. It gets its own placebo (7 shifts).

   Cleaning, deterministic and done before any return is looked at. The notebook recomputes each list with max|diff| = 0 comparisons (exam cell 34: "find them before you concatenate").
   - (a) Drop the 6 exact duplicates of curated columns: x73 = x10, x89 = x12, x114 = x13, x17 = x14, x92 = x15, x18 = x16.
   - (b) Drop the 26 columns that are identical across all 12 countries (global series in disguise, whose c7 differential is identically 0): x88 (= x6), x91, x93–x109 (x99 = x9, x104 = x7, x109 = x8), x137–x140, x143, x148 (constant), x160 (= −x143).
   - (c) Drop the 7 gap columns.
   - (d) Drop x86, which is already in M3.
   - (e) Drop the near-duplicates x57 and x58 (|corr| > 0.999 with x56).
   - (f) Transform level series with |values| > 1e4 (x40–x48, x80, x81, x123–x126): if strictly positive, take the 12-month log change on the stored series; otherwise drop. This follows L3 p.48–49.
   - (g) Apply the frequency-specific lags (2.1).
   - (h) Take the differential vs c7.
   - (i) Inside the window: `StandardScaler` + `PCA` on unique (country ≠ 7, month) training rows.

   The expected count is about 106 columns. Print it and assert that no NaN remains in the sample. **No screening of extended columns by their correlation with returns** is ever done: that is the post-screening trap of L4 p.39 and L5 p.41.
- **No outside data** are added (exam cell 34 allows it). Every added series would add to the search and needs a documented release date. The extended file already holds more candidates than can be tested honestly. This is recorded as a choice (Open question Q2.c).

### 2.9 Target and mapping back to positions

- The target is y = r/σ̂_{t−1} with the 36-month SD (§0.3).
- **Why vol-scale:**
  - (1) Monthly vol ranges from 0.82% to 17.40% (exam cell 34, profile §C). A pooled OLS on raw returns would be dominated by class A: the constant-variance assumption of L3 p.2–3 and L2 p.37, 41 fails, and var(b) ∝ σ² (L2 p.73).
  - (2) y is in comparable "Sharpe units" across assets.
  - (3) It maps directly to a risk-scaled position.
- **Why 36 months:** it matches the provided risk characteristic x5 and exists for every asset from 2003-01. The purpose is to equalise **cross-asset** scale; time-series vol timing is not the target. Vol clustering inside assets (2008, 2020) remains, and is acknowledged (L2 p.80).
- **Positions:** r̂ = ŷ σ̂, and w ∝ r̂/σ̂² = ŷ/σ̂. This is a mean-variance rule with a diagonal covariance. The diagonal covariance is an assumption that ignores the B/C/D correlation blocks; it is disclosed.
- **Outliers:** there is no winsorisation in the primary spec. Following L3 p.17 ("run with and without"), F34 winsorises the **training** target at the fixed constant ±4. A constant, not an estimated cutoff, cannot leak (L5 p.58 #1). Evaluation is always on the raw y.

### 2.10 Final predictor table (the notebook prints this with counts)

| Block | Raw inputs | Transform | Shift | Columns entering the regression |
|---|---|---|---|---|
| Characteristics | x1, x2, x3, x4, x5 | within-class rank → [−0.5, 0.5] | 1, 0, 1, 1, 0 | 5 |
| Class dummies | asset_class | 1[B], 1[C], 1[D] (A reference) | – | 3 |
| Global macro | log x6, x7, x8, x9 | 60-month trailing z → StandardScaler + PCA (k_G ∈ {1,2,3}) × 1[A,B,C,D] | 2 | 4·k_G = 4–12 |
| Country macro | x86, x12, x13, log x14, log x15, x16 | differential vs c7 → StandardScaler + PCA (k_C ∈ {1,2,3}) × 1[B,C,D]; 0 for class A | 2 | 3·k_C = 3–9 |
| **M2 total (Q1 primary)** | 5 raw series + class | | | **8** |
| **M3 total (Q2 primary)** | 15 raw series + class | | | **15–29** |
| M4 add-on | u^{(j)} × global PC1 | | | +5 |
| M5 add-on | about 106 cleaned extended differentials | PCA k_E ∈ {1..5} × 1[B,C,D] | 2 / 5 / 14 | +3 to +15 |
| M3-raw (F30) | as M3 without PCA | 4 global z × 4 classes + 6 differentials × 3 classes | 2 | 8 + 16 + 18 = 42 |
| M2-TS (F39) | x1..x5 own-history trailing z | expanding own mean/sd, min 36 | as characteristics | 8 + 5 = 13 |
| Excluded | x10/x73 (look-ahead), stored x11 (gaps; its rebuild is spanned), 42 extended columns (duplicates/global/gaps/near-duplicates), x148 | | | – |

**(iv) Assumptions.**
- A2.1: The uniform 2-month macro lag (release delay) and 5/14 months for quarterly/annual.
- A2.2: Class A gets no country macro.
- A2.3: The differential vs c7 is the primary cross-sectional macro transform.
- A2.4: Within-class ranks.
- A2.5: Stale x1 values are kept.
- A2.6: The target uses a 36-month σ̂.
- A2.7: No outside data.
- A2.8: The extended file is used only as in 2.8.

**(v) Pitfall guards.**
- Every shift is a `groupby` on a sorted, complete panel. Assert: for a random (asset, t), feature(x1) equals stored x1 at t−1, and feature(x2) equals stored x2 at t.
- A **truncation test** proves all causal transforms: rebuild the whole feature pipeline on data truncated at month m* (for example 2015-06), and assert that the feature rows for target m* + 1 are identical to the full-data build (§6).
- The x10 look-ahead demonstration and the x11 rebuild are computed in the notebook.
- The duplicate and global detection is recomputed.

**(vi) Outputs.**
- Figures: F6a x10 vs the monthly x86 for c7 (2020–2023); F6b x11 vs the rebuild for c9/c11/c12; F6c the four lagged, trailing-z global series; F7 scree plots and first-PC loadings (initial training block, descriptive).
- Tables: T5 per-series treatment (frequency, shift, missing pattern, treatment, reason); T5b the extended-file cleaning log (column lists and counts); T6 the predictor table above.

**(vii) The written answer must cover.**
- ☐ The lag rule per variable, with the release-delay argument.
- ☐ The low-frequency look-ahead and its fix.
- ☐ Missing values per series (pattern → treatment → why it is past-only), including the x11 regime-change trap and how close the rebuild is.
- ☐ The within-class rank choice and why (all four options considered).
- ☐ Trailing standardisation of global macro.
- ☐ Country-to-asset mapping, including the class A choice.
- ☐ Differentials vs c7, and the other cross-sectional transforms considered.
- ☐ Interactions and class dummies.
- ☐ Extended-file use and the multiple-testing argument.
- ☐ Why vol-scaled, which σ̂, and how forecasts become positions.
- ☐ The predictor table with counts.

**(viii) Open questions.**
- Q2.a: Macro release lag. Default: **2 months** (1 month reported as F32).
- Q2.b: Class A country macro. Default: **none**.
- Q2.c: Bring in outside data? Default: **no**.
- Q2.d: Stale x1 in B for 2024-10..12. Default: **keep as stored and flag**.
- Q2.e: σ̂ window. Default: **36 months** (12 months is not run, because it would change the target and so the R² metric itself).

---

## Part 3. Models × evaluation schemes

**(i) Goal.** Measure out-of-sample forecastability (Q1) and the incremental value of macro (Q2). This requires:
- R²_OOS vs the trailing mean and vs zero, model × scheme, and by class;
- which scheme wins and why;
- which predictors carry the signal and whether that is stable;
- macro vs refitted-without-macro;
- the 36-month circular-shift placebo with more than one shift;
- per-class vs pooled models (exam cell 37 step 3).

### 3.1 Model menu (and what is deliberately not used)

| Model | Role | Implementation | Grounding |
|---|---|---|---|
| OLS | Parsimonious baseline; interpretable coefficients (units: top-minus-bottom spread in y) | `LinearRegression()` or `np.linalg.lstsq` on the scaled design. M3's k_G, k_C are chosen on inner folds. | L3 p.19–26 (MLR); L2 p.3, p.33 |
| **Ridge (primary)** | Many weak, correlated predictors ("suitable when features are weak and should still contribute", L5 p.60). Nests OLS as λ → 0 and the benchmark as λ → ∞. | Fast path: the closed form (X'X + λI)^{-1}X'y (L5 p.12) via one SVD per fold for all 50 λ. Scaler mean/sd from training rows only. Unit test equals `Pipeline([('sc', StandardScaler()), ('m', Ridge(alpha=λ))])` (sklearn `Ridge.alpha` = the slide's λ). | L5 p.11–16, p.23, p.60 |
| Lasso | Interpretability: which characteristics and macro PCs survive, and how stable the selection is (L5 p.21–22 paths; L4 p.50 instability) | `lasso_path(Xs, y − ȳ_train, alphas=grid)` on training-scaled X, exactly the L5 p.52 pattern. Mind the descending alphas returned. sklearn's objective is (1/2n)‖·‖² + α‖w‖₁, so the grid is not the slide's λ (L5 notes). | L5 p.17–24, p.47, p.52 |
| Lasso-BIC (tuning robustness) | Information-criterion tuning instead of validation | `Pipeline([('sc', StandardScaler()), ('m', LassoLarsIC(criterion='bic'))])` fitted on the outer training window (L5 p.46 code). For M3, k_G = k_C = 3 is fixed and lasso selects. Caveat: rows are cross-sectionally correlated, so the effective n is smaller than the row count and BIC's likelihood gain is overstated (flag, L3 p.2–3). | L5 p.42–46; L4 p.29–33 |
| PCR-macro | The macro compression device inside M3/M4/M5 (2.7). "OLS on M3" is PCR on the macro block with the characteristics as extra regressors; "Ridge on M3" is ridge on [characteristics, macro PCs × class]. | as 2.7 | L1 p.45–74, p.46; L8 p.61 |
| Random forest (secondary, nonlinear robustness) | Checks for nonlinearity and interactions the linear model misses (L8 p.28, p.37) | `RandomForestRegressor(n_estimators=300, max_features=0.33, min_samples_leaf=400, n_jobs=-1, random_state=SEED)`. **No tuning**, fixed a priori (L8 p.44: "RF avoids the need for CV … set B as large as patience allows, set a minimum leaf size"). The p/3 rule is L8 p.43. A leaf of 400 rows has a mean SE of about 1/√400 = 0.05 in y units, the order of a plausible signal. Features for M2: 5 ranks + 3 dummies. For M3: add k_G global PCs and k_C country PCs **without** class interactions (trees find interactions, L8 p.28), with (k_G, k_C) taken from the ridge's training-only choice for that window. | L8 p.41–47, p.61 |
| Not used | KNN: curse of dimensionality, no variable selection, K unstable (L6 p.37). GBRT: needs a validation budget, which is a leakage risk (L8 p.55); optional only if approved, and would add 2 specs. Neural nets, SVM, elastic net beyond its formula, XGBoost, SHAP, OOB, HAC: passing mentions only (L8 p.62, L5 p.11, L2 p.80). Clustering: classes are given labels (L7 p.4). | – | lecture boundaries |

### 3.2 Hyper-parameter grids (chosen on inner folds only; grid-edge rule fixed now)

- **Ridge:** λ = n_fit · 10^linspace(−3, 3, 50), i.e. λ/n ∈ [1e−3, 1e3] (L5 p.16 plots against λ/T). The shrinkage factor on a unit-variance direction is 1/(1 + λ/n), so the grid runs from essentially OLS to essentially the intercept-only benchmark.
- **Lasso:** α ∈ 10^linspace(−5, −0.5, 50). With standardised X and var(y) ≈ 1, α_max ≈ max|corr| < 0.3, so the null model is inside the grid.
- **k_G, k_C** ∈ {1, 2, 3} (9 combinations; M3 always contains macro). **k_E** ∈ {1, …, 5}.
- **Edge rule** (L5 p.27, p.37–39, p.53: "the optimum must lie in the middle of the grid"):
  - Record the chosen index for every refit (table T10).
  - At the **lower** edge (near-OLS): extend by 3 decades down and rerun that refit, disclosing it.
  - At the **upper** edge (for lasso, the all-zero region): accept it as the honest "no signal, shrink to the benchmark" answer and flag it.

### 3.3 Harness (pseudo-code; the only splitter in P3)

```python
REFIT_ENDS = pd.date_range('2010-12-31', '2023-12-31', freq='YE')      # 14
def outer_window(b, scheme):
    a = pd.Timestamp('2003-01-31') if scheme in ('static', 'expanding') else b - pd.DateOffset(months=95)
    return a, b                                    # static: only b = 2010-12 is used, and it predicts 2011-01..2024-12
def inner_folds(a, b):                             # 3 expanding folds, 12-month validation each
    for j in (3, 2, 1):
        fit_end = b - pd.DateOffset(years=j)       # month-end arithmetic via MonthEnd(0)
        val = (fit_end, fit_end + 12 months]
        assert fit_end < val.min()
        yield (a, fit_end), val
def fit_window(spec, model, rows_train, hyper):   # every pooled-across-dates fit happens here
    macro = fit_macro_pca(dates=rows_train.dates.unique(), kG, kC)   # unique-row PCA, training dates only
    X_tr = build_X(spec, rows_train, macro); X_te = build_X(spec, rows_te, macro)
    scaler = StandardScaler().fit(X_tr)  ...       # or Pipeline; ridge fast path equals the Pipeline (unit test)
```

- **Order of operations per outer refit b:**
  - (1) For each hyper-parameter combination, loop over the 3 inner folds: fit on the fold's training rows (scaler and PCA refitted on those rows only) and score the validation MSE.
  - (2) Pick the arg-min of the pooled validation MSE.
  - (3) Refit on all rows in [a, b].
  - (4) Predict the 12 months of year b+1. For static, predict all 168 months from b = 2010-12.
- Test rows never enter (1)–(3) (L5 p.52: "y_pred from the testing dataset has nothing to do with obtaining optimal alpha").
- **OOS lock:** `score_oos()` asserts `OOS_UNLOCKED`. During development the harness runs with pseudo-OOS years 2009–2010 inside the training block (§8).

### 3.4 Specifications evaluated (IDs used in the disclosure, §8)

M1 (class dummies only; OLS), M2, M3, M4, M5, M3-raw, M3-dev, M2-TS, as defined in §2.10 and §8. The notebook produces a **model × scheme table** (T7) with cells R²_OOS vs pooled trailing mean / vs zero / vs per-asset trailing mean, plus the average IS R²:

| spec \ model × scheme | OLS S/E/R | Ridge S/E/R | Lasso S/E/R | RF S/E/R |
|---|---|---|---|---|
| M1 class means | ✓ | – | – | – |
| M2 characteristics | ✓ | ✓ (**E = F08, primary Q1**) | ✓ | ✓ |
| M3 + macro PCR | ✓ | ✓ (**E = F20, primary Q2**) | ✓ | ✓ |
| M4, M5, M3-raw, M3-dev, M3 lag 1, M2 z-score, M2 winsorised, M2-TS, per-class M2 and M3, Lasso-BIC M2 and M3 | – | E only | (BIC: E only) | – |

M1 shows how much of any gain is just class-mean differences relative to the pooled trailing mean (a "benchmark-plus" row).

### 3.5 By class, per-class vs pooled

- **By class:** for F08, F20, F05 (M2 OLS expanding) and F14 (M2 RF expanding), R²_OOS is computed on each class's OOS rows with the **same** pooled benchmark (L8 p.53: same split, same benchmark), plus raw-return R² by class (§0.3). This is table T8.
- **Per-class models (F35, F36):** four separate ridge fits (one per class) on that class's rows only, each with its own inner-fold tuning. Class A has no country block. Forecasts are stitched together and scored overall and by class against the pooled model.
- Grounding: L3 p.54 ("why not separately estimate … models?", separate slopes = fully interacted); L3 p.50 (common slopes + class intercepts = pooled M2); L2 p.73 (var(b) = σ²/((n−1)s_x²)): pooling raises n and s_x, while per-class allows different slopes (the bias–variance trade-off, L2 p.72; "need to get lucky" with small n, L2 p.62).
- At the first refit the per-class training sizes are only 672 (D) to 2,496 (A) rows.

### 3.6 Macro vs no-macro, and the placebo

- **Refit without macro:** F08 vs F20, and F05 vs F17 (the OLS pair). Each pair uses the same scheme, window, folds and benchmark. Report ΔR²_OOS and the model-vs-model loss-differential t-stat.
- **Why no in-sample partial F-test:** the rows are cross-sectionally correlated, which breaks its iid-error premise (L3 p.2–3), and the course says to judge by prediction (L4 p.28, "stop asking whether it fits and start asking whether it predicts").
- **Placebo [EXAM-DEFINED, cell 37 step 3]:**
  - Before any transform, circularly shift **every raw macro series** (x6..x9, x86, x12..x16, and for M5 the extended columns) within country by s months, with `np.roll(values, s)` on each country's 300-month stored series. All countries use the same s, so the cross-country structure is real but from the wrong date.
  - Then run the **identical** pipeline (lags, trailing z, differentials, inner-fold tuning of λ, k_G, k_C, refits).
  - **Primary shifts: s ∈ {36, 48, 60, 72, 84, 96, 108}** (7 shifts).
    - They are multiples of 12, so January-stamped annual and quarterly series keep their stamping pattern and the frequency-specific lag rules still apply.
    - s ≥ 36 includes the notebook's 36.
    - s ≤ 108 guarantees that every placebo value used at an **OOS** target date comes from the past. The earliest OOS target (m = 132) under the longest lag (14) uses stored m = 118 ≥ s, so there is no wrap-around of future data into the test window.
    - Caveat: the 60-month trailing z-score window and the Δ12-log transforms are computed on the shifted series, so for s ≥ about 72 they can reach wrapped (end-of-sample) values. This affects only normalisation. It can only *help* the placebo, which makes crediting real macro harder (a conservative bias). Disclose it.
  - **Supplementary shifts:** {120, 132, …, 264} (13 more; these wrap). They are reported in an appendix.
  - A persistent series stays partly correlated with a shifted copy of itself (exam cell 37), which is exactly why several shifts are needed.
  - Output: F11, a strip plot of placebo ΔR²_OOS (placebo M3 − M2) with the real Δ marked. Report the rank of the real Δ among the 7 (and among 20).
  - Unit test: s = 0 reproduces F20 exactly.
  - Grounding: the honest-null design (L4 p.26–27: predictors "by construction" with no timely signal) and the simulated-null logic (L2 p.78–79, 88–93).
  - Characteristics are **not** shifted. The placebo isolates macro timing.

### 3.7 Which predictors carry the signal, and stability across windows

- **Coefficient paths (F9a, F9b).**
  - F08 (ridge): standardised coefficients (per 1 training-SD) at each of the 14 refits.
  - F05 (OLS): raw rank coefficients with ±2 SE. The SEs come from a **month-resampling bootstrap** (B = 500 per refit, seed SEED): resample training months with replacement, keeping each month's cross-section intact. This follows L2 p.78–80, "resampling the actual data". Nonrobust formula SEs (L3 p.33–35) are shown alongside, flagged as too small because the rows are cross-sectionally correlated (L2 p.80).
  - Expanding vs rolling paths are overlaid (rolling adapts to nonstationarity, L1 p.34).
- **Lasso selection heatmap (F9c):** for F11/F23 (lasso M2/M3 expanding), a 14 refits × features grid of nonzero indicators (L5 p.21–22; L4 p.50 "dropping one year … changes the selected sets").
- **Permutation importance, out of sample (F10)** (L8 p.57: permute on held-out data, repeat and average; L8 p.47, p.58: MDI is in-sample and is **not** reported as importance):
  - For each OOS year, use that year's fitted model and its test rows.
  - Characteristic j: permute u^{(j)} **within each (month, class) cell** (this preserves the rank marginal exactly).
  - Global macro block: permute the lagged global 4-vector **across OOS months**, then pass it through each year's fitted PCA.
  - Country block: permute each country's lagged differential vector across months.
  - R = 20 repeats (seeds SEED + r).
  - Importance = R²_OOS(original) − mean_r R²_OOS(permuted), with the SD across repeats. A per-year version gives a 14 × groups heatmap for stability.
  - Permuting all five characteristics jointly is the **negative control**: R²_OOS should fall to about 0 or below.
- **Partial dependence** (L8 p.59–60) for F14 (RF M2), for sign and shape: the brute-force average prediction on a 40-point grid from the 5th to 95th percentile of u^{(j)}, computed on the last expanding fit's training rows (`method='brute'`).
- **Fit diagnostics** (L3 p.58, p.8/18): the OOS ŷ vs y scatter and the residual-vs-fitted plot for F08. Print sd(ŷ)/sd(y) as a sanity check on how small the predictable part is.

### 3.8 Which scheme wins (interpretation to write)

- Compare static, expanding and rolling for M2 and M3 (ridge and OLS).
- Expected logic, stated as hypotheses:
  - Expanding benefits from more data at low SNR (L1 p.33).
  - Rolling adapts to nonstationarity (L1 p.34) at the cost of variance.
  - Static cannot adapt at all.
- Report the scheme ranking **as found**, with the loss-differential t-stats between schemes.

**(iv) Assumptions.**
- A3.1: Annual refit.
- A3.2: 3 expanding inner folds of 12 months each, then refit on the full window.
- A3.3: Pooled trailing mean as the primary benchmark.
- A3.4: RF hyper-parameters fixed a priori.
- A3.5: Months are the independent unit for tests and bootstrap; serial correlation is ignored and flagged.
- A3.6: The placebo shifts all macro by the same s across countries.

**(v) Pitfall guards.**
- Test rows are never used for λ, k or RF settings.
- There is one shared OOS window and one benchmark vector for all models (L8 p.53).
- Grid-edge flags are recorded.
- IS vs OOS are printed side by side.
- Sanity check: monthly R²_OOS far above about 0.5% is treated as a prompt to hunt for a leak **before** interpreting it (L5 p.57).
- The headline F08 R²_OOS is recomputed by an independent `np.linalg.lstsq` + closed-form ridge path, and must agree to 1e-10.

**(vi) Outputs.**
- Tables: T7 (model × scheme); T8 (by class, raw and vol-scaled; per-class vs pooled); T9 (macro vs no-macro with placebo ranks, including M4, M5, M3-raw, M3-dev, lag-1); T10 (chosen λ/α/k per refit with edge flags).
- Figures:
  - F8: R²_OOS bars, plus the cumulative SSE-difference plot Σ_{s≤t} Σ_i [(y − b)² − (y − ŷ)²] over OOS months. This visualises the components of the L5 p.55 formula over time.
  - F9a–c: paths and heatmap.
  - F10: permutation importance, full and per-year.
  - F11: placebo.
  - PD plots.
  - The fit plot.

**(vii) The written answer must cover.**
- ☐ R²_OOS vs the trailing mean and vs zero, model × scheme.
- ☐ By class for the models we care about.
- ☐ Which scheme wins and why.
- ☐ Which predictors carry the signal and whether that is stable across windows.
- ☐ Whether macro adds anything once refitted without it.
- ☐ The placebo with more than one shift, and why several.
- ☐ Per-class vs pooled.
- ☐ How penalties and hyper-parameters were chosen, and that they come from training data only.
- ☐ IS vs OOS.
- ☐ Negative numbers reported as results (L5 p.57).

**(viii) Open questions.**
- Q3.a: Primary trailing-mean benchmark: pooled or per-asset? Default: **pooled primary, per-asset always reported**.
- Q3.b: OOS start 2011-01 or 2008-01 (to put the GFC in the test window, with only 60 training months)? Default: **2011-01**.
- Q3.c: Run GBRT? Default: **no** (it would add 2 specs and a tuning budget).
- Q3.d: Run the supplementary wrapping placebo shifts? Default: **yes, in the appendix**.

---

## Part 4. From forecasts to portfolios

**(i) Goal.** Test whether a forecast-based portfolio beats EW, risk parity and TSMOM out of sample and after costs. Report Sharpe at a minimum, plus return, volatility, drawdown, turnover, net-of-cost performance, cumulative plots and sub-periods. Attribute any outperformance to benchmark exposure vs timing (exam cells 33, 36 rule 1, 37 step 4).

**Scope flag.** Portfolio construction, Sharpe ratio, drawdown, turnover, risk parity, TSMOM and transaction costs are **not taught in any lecture** (L2, L5 and L8 boundaries). They are **[EXAM-DEFINED]**: the benchmarks by cell 36 rule 1, and Sharpe, drawdown, turnover and costs by cell 37 step 4. The **cost model is our assumption** and is flagged below. The attribution regression is lecture-grounded (L2 p.47–49, p.94–97; L3 p.19–35).

### 4.1 Benchmarks, computed first (exam cell 36 rule 1: "before you fit anything")

All benchmarks are built from `excess_return` alone, formed at the end of t−1, held over month t, rebalanced monthly, and normalised to **unit gross exposure** Σ_i |w_{i,t}| = 1:
- **EW:** s_i = 1, so w = 1/50. Note: 26/50 of the weight, and most of the risk, sits in class A (exam cell 34's scale warning).
- **RP:** s_i = 1/σ̂_{i,t−1} (36-month SD, ddof=1, own σ̂, not the floored x5), with w = s/Σs.
- **TSMOM:** s_i = sign(R12_{i,t−1})/σ̂_{i,t−1}, where R12_{i,t−1} = ∏_{s=t−12}^{t−1}(1 + r_{i,s}) − 1, and w = s/Σ|s|. sign(0) = 0.

Their performance is reported for the OOS window 2011–2024 (the comparison window) and for 2003–2010 (context), **before any model cell runs**. x2 and x5 contain exactly the TSMOM and RP ingredients (profile §D), which is why the attribution in 4.5 matters.

### 4.2 Forecast-based rules (fixed now; only the forecasts of F08 and F20 are used)

| ID | Rule | Signal s_{i,t} | Motivation |
|---|---|---|---|
| **P1 (primary)** | proportional | ŷ_{i,t}/σ̂_{i,t−1} | Mean-variance with a diagonal covariance (r̂/σ̂² = ŷ/σ̂); the notebook's "proportional to the forecast scaled by volatility". **With a no-signal model (ŷ = a positive constant) P1 collapses to RP**, so RP is the natural bar. |
| P2 | sign | sign(ŷ_{i,t})/σ̂_{i,t−1} | Same form as TSMOM with the model's sign, so it is directly comparable to TSMOM |
| P3 | class-neutral cross-sectional | (ŷ_{i,t} − mean_{j∈class} ŷ_{j,t})/σ̂_{i,t−1} | Removes class-mean (beta-like) exposure, leaving only within-class selection. It is the portfolio analogue of within-class ranks. |

Every rule uses w = s/Σ_j|s_j|, and cash (w = 0) if Σ|s| = 0. P1–P3 come from F08; P4–P6 are the same three rules from F20. That is 6 forecast portfolios, all disclosed. The rule and its parameters are fixed ex ante, not tuned on the test window. This is the L6 p.46–47 lesson: the cost threshold of 1/5 is fixed in advance, and the test-optimal 0.195 is only a check.

### 4.3 Costs [OUTSIDE LECTURES; EXAM-REQUIRED (cells 33, 37); transparent assumption]

- Pre-trade (drifted) weights: w̃_{i,t+1} = w_{i,t}(1 + r_{i,t})/(1 + r^p_t).
- One-way turnover at rebalancing: TO_{t+1} = Σ_i |w_{i,t+1} − w̃_{i,t+1}|. This counts EW's rebalancing trades, which a no-drift definition would miss. The initial build month is excluded from the averages.
- Net return: r^{p,net}_t = r^p_t − c·TO_t, a proportional cost c per unit of one-way turnover. It is **uniform across assets**. Class-specific costs (commodities and EM FX cost more) are more realistic but would need outside data; flagged.
- **Cost grid** c ∈ {0, 2, 5, 10, 20, 50} bp. **Headline: 10 bp.**
- **Break-evens:** c₀ = mean(r^p)/mean(TO), the cost at which the net mean is zero; and c*_B, the cost at which net Sharpe(P) = net Sharpe(B), both paying c, found on a 0.5 bp grid from 0 to 200 bp with linear interpolation.

### 4.4 Performance metrics (definitions written in the Methodology)

| Metric | Definition |
|---|---|
| Annualised return | 12 · mean(r^p_t) |
| Annualised vol | √12 · sd(r^p_t), ddof=1 |
| Sharpe | √12 · mean/sd (returns are already excess, so there is no r_f) |
| Max drawdown | max_t (1 − W_t / max_{s≤t} W_s), with W_t = ∏(1 + r^p_s) |
| Worst month | min r^p_t |
| Turnover | mean monthly TO, and ×12 annualised |
| Net versions | at each c in the grid |
| Sub-periods | 2011-01..2017-12 and 2018-01..2024-12 (84 months each) |
| Sharpe-difference uncertainty | iid bootstrap over OOS months (2,000 draws, seed SEED) of SR(P) − SR(B), with the 2.5/97.5 percentiles. L2 p.78–80 resampling. The iid-month caveat is flagged; a block bootstrap is not taught. |

### 4.5 Attribution: benchmark exposure vs timing (lecture-grounded)

- Estimate r^{P}_t = α + β_EW r^{EW}_t + β_RP r^{RP}_t + β_TS r^{TSMOM}_t + ε_t by OLS over 2011–2024, for gross returns and for net returns at 10 bp:
  `sm.OLS(rP, sm.add_constant(B)).fit()`
- Also estimate each single-benchmark regression.
- Report 12α, t(α) = α/s_α with t_{n−p} p-values, the β's, and R².
  - This is the L2 p.94–95 intercept test (α as a performance benchmark, L2 p.49) in the MLR form (L3 p.19–35).
  - For the single-RP regression, also test β = 1 by hand (L2 p.96–97): does P1 just lever RP?
  - R² = the share of P's variance explained by exposure to the rules. α = what remains, attributed to timing and selection beyond linear benchmark exposure.
- Nonrobust SEs are flagged (L2 p.80). A month-resampling bootstrap SE of α (2,000 draws) is shown as the check.
- Multicollinearity among the three benchmark returns (EW and RP are both long-only) makes the individual β's unstable (L3 p.39–44), so the interpretation focuses on α and R².
- *Optional, [OUTSIDE SCOPE: needs approval]:* a static-vs-dynamic decomposition, r^P_t = Σ w̄_i r_{i,t} + Σ (w_{i,t} − w̄_i) r_{i,t}. It uses ex-post average weights, so it is attribution only, not a strategy.

**(iv) Assumptions.**
- A4.1: 36-month σ̂ for RP, TSMOM and P1–P3.
- A4.2: Uniform proportional costs; headline 10 bp.
- A4.3: Drift-adjusted turnover.
- A4.4: A diagonal-covariance position rule; correlations are ignored.
- A4.5: Unit gross exposure for everything (Sharpe is scale-free; returns and drawdowns are not, so all strategies are compared at the same gross exposure, as the notebook prescribes).

**(v) Pitfall guards.**
- Weights at t use only σ̂_{t−1}, R12_{t−1} and ŷ_t, which is formed from information at t−1. Assert that each weight row depends only on rows ≤ t−1 (the truncation test covers σ̂ and R12).
- Assert gross = 1 each month.
- Benchmarks are computed before the models.
- No rule parameter is tuned on 2011–2024.
- The primary portfolio (P1 from F08) is pre-registered; P2–P6 are disclosed.
- asset_16's very low σ̂ concentrates RP and TSMOM weight in D. Report the maximum single-asset weight per strategy.

**(vi) Outputs.**
- Tables: T11 (performance, full OOS and by half; EW, RP, TSMOM, P1–P6; gross and net at 10 bp); T12 (cost grid, c₀ and c*_B); T13 (alpha regressions, joint and single, with bootstrap SEs).
- Figures: F12 (cumulative log wealth, OOS, gross and net at 10 bp); F13 (drawdowns); F14 (average |w| by class per strategy, which exposes risk concentration).

**(vii) The written answer must cover.**
- ☐ How each portfolio is formed and rebalanced (exact formulas).
- ☐ Sharpe vs all three benchmarks.
- ☐ Return, vol, drawdown and turnover.
- ☐ Net of costs, with the cost assumption stated as ours, plus break-evens.
- ☐ Cumulative plots.
- ☐ Sub-periods.
- ☐ Exposure vs timing (α, β, R²).
- ☐ The answer to Q3, whichever way it comes out.

**(viii) Open questions.**
- Q4.a: Headline cost level. Default: **10 bp one-way**.
- Q4.b: Include P3 (class-neutral)? Default: **yes**.
- Q4.c: Run the static-vs-dynamic decomposition (outside scope)? Default: **no**.

---

## Part 5. Write-up structure (exam cell 38; cell 40 onward)

| Section | Must contain (ticked against the exam text) | Tables / figures |
|---|---|---|
| **Introduction** | ☐ The three questions. ☐ Why they matter to a cross-country, cross-asset allocator (diversification across the B/C/D blocks; the signals systematic funds trade). ☐ What the literature leads us to expect: small but positive cross-sectional predictability from carry, momentum and value; low SNR (L1 p.33); ML portfolios in GKX (L1 p.36); signals that are "often weak and linear" (L8 p.62); published monthly stock-level R²_OOS of about 0.3–0.5% (L5 p.57). Literature is cited as background only. ☐ A one-paragraph preview of the findings, whichever way they come out. | – |
| **Data** | ☐ Panel contents. ☐ The class/characteristic hypotheses. ☐ Treatment: missing values (back-fills, x11 rebuild and its quality, gap columns), lags (the table in 2.1, release delay, low-frequency look-ahead), standardisation (within-class ranks; trailing-z global), and the country mapping (differentials vs c7; class A none). ☐ Summary statistics. ☐ Plots of structure. | T1–T3, T5, T6; F1–F7 |
| **Methodology** | ☐ Benchmarks and why (trailing mean updated monthly and zero; EW/RP/TSMOM, unit gross, monthly). ☐ Models (OLS/ridge/lasso/PCR-macro; RF secondary). ☐ Schemes and the fixed OOS window with dates and reasons. ☐ Tuning (inner expanding folds; grids; edge rule; BIC check). ☐ Target and positions. ☐ Performance measures and costs. ☐ The pre-registration and the search size. A reader must be able to reproduce the design from this section alone. | T14 |
| **Results** | In the order of the questions. ☐ Q1: T7 and T8, with F08's headline and t-stat, IS vs OOS, and the sanity check. ☐ Q2: T9 and F11 (placebo), then F9 and F10 (signal and stability). ☐ Q3: T11–T13 and F12–F14. | T7–T13; F8–F14 |
| **Interpretation and discussion** | ☐ Economic meaning: which characteristics carry signal and whether that matches carry, momentum or value priors; why macro does or does not help. For example, x1 and x4 already embed country rate information (profile §D), and x2 and x5 duplicate the TSMOM and RP ingredients. ☐ Where the evidence is strong and where it is fragile (stability across windows, placebo ranks, class concentration, 2020 and 2022 extrapolation, L2 p.100–105). ☐ What we tried that did not work: every spec in T14. ☐ What we would do next: class-specific costs, a block bootstrap, a longer history. | – |
| **Conclusion** | ☐ A summary a PM can read alone: how much is forecastable, whether macro helps, whether the portfolio beats the rules after costs, and how confident we are. | – |
| **Appendix** | ☐ Specification-search disclosure (T14). ☐ Leakage audit (T15). ☐ Supplementary placebo shifts. ☐ Hyper-parameter table (T10). | T10, T14, T15 |

---

## Part 6. Leakage audit checklist (printed as table T15 in the notebook)

| # | Failure mode | Where the design prevents it | Executable check |
|---|---|---|---|
| L5 p.58 #1, AI guide 4(b) — **FATAL** | Transformation fitted before the split (scaler, imputer, PCA, winsor cutoff) | All pooled fits live in `fit_window()` on training dates only. Macro PCA is refitted per inner fold and per refit. Winsorisation uses a fixed constant (F34). There is no imputation (removed defects are dropped or rebuilt). | (a) Ridge fast path == `Pipeline(StandardScaler, Ridge)` on the first window. (b) Perturbation test: replace every raw value dated after 2010-12 (returns, x1..x5, macro) with random noise (seeded), rebuild the features, refit window b = 2010-12, and assert that the fitted coefficients, PCA loadings and chosen λ/k are unchanged. Scaling by a constant would not do: ranks are invariant to it, so that test would pass trivially. |
| Causal pre-computation | Ranks, trailing z, σ̂, lags, differentials | Computed per date from data ≤ the information date | **Truncation test:** rebuild the features after deleting all contemporaneous information dated after m* ∈ {2008-06, 2015-06, 2022-06}: returns, x1, x3, x4, and all macro rows. Keep row m*+1's pre-lagged x2 and x5, which by exam cell 34 are known at m*. Assert that the feature rows for target m* + 1 (and σ̂, R12, and the benchmark value) equal the full build (atol 0). |
| L5 p.58 #2 | Merge on period end instead of publication date | Shift table 2.1 (release delay 2; quarterly 5; annual 14); x10/x73 excluded; merges on (country, date) keys | Assert shifts per column from a dict; the x10 look-ahead demonstration figure; assert that no extended annual column enters with a shift < 14 |
| L5 p.58 #3, AI guide 4(a) — **FATAL** | Random fold on time-ordered data | Only the date-based `inner_folds` / `outer_window` | Per-fold asserts max(train) < min(val) < min(test); grep the notebook for `train_test_split`, `KFold(`, `shuffle=True`, `cross_val_score`: must find none in P3 cells |
| L5 p.58 #4 | Hyper-parameter or threshold chosen on the test block | λ, α, k on inner folds; RF fixed a priori; portfolio rules and cost grid fixed in §4; tercile sorts on the training block only | T10 logs the choices and the validation MSE only; `score_oos` asserts `OOS_UNLOCKED` |
| L5 p.58 #5 | Benchmark changed after the fact | Pooled trailing mean (primary), zero and per-asset, all pre-registered in §0.3 and PREREG; all always reported | PREREG dict printed before results; the headline is pulled from PREREG programmatically |
| AI guide 4(d) | `r2_score` / `.score` against the test mean | Custom `r2_oos` | Unit test: the benchmark at 2011-01 equals the training-period mean of y; `r2_score` is not imported |
| AI guide 4(c) | Default-penalised LogisticRegression called "plain" | No classifier in P3 | – |
| Data traps | Back-fills; x11 stale ffill; stale x1; x5 floor; exact duplicates and global columns in the extended file; x10/x73 annual averages | §2.2, §2.8 | Assert that no back-filled cell is in the design; print the duplicate and global lists recomputed with max\|diff\| = 0 |
| Placebo integrity | The placebo pipeline must differ only in timing | Same function with a `shift` argument | s = 0 reproduces F20 exactly |
| Headline verification (AI guide §6.5) | A number that is too good | Sanity bound (L5 p.57); independent recomputation; negative control (joint permutation of characteristics → R²_OOS ≤ about 0) | Printed with the headline |

---

## Part 7. Compute budget (class server; measured on synthetic arrays of panel size, 4 cores here)

| Component | Unit cost (measured) | Count | Estimate |
|---|---|---|---|
| Ridge, all 50 λ via SVD, 12,600 × 29 | 0.026 s (4,800 × 8: 0.002 s) | M3-type: 14 refits × (3 folds × 9 (k_G, k_C) combinations + 1 final) = 392 SVDs per spec-scheme | ≈ 15 s per M3-type spec-scheme; ≈ 1 s for M2 |
| `lasso_path`, 50 α | < 0.05 s | as above | ≈ 15 s per M3 lasso spec-scheme |
| `LassoLarsIC('bic')` | 0.02 s | 14 × 2 | < 1 s |
| sklearn `Ridge` in a Pipeline, loop over 50 λ | 0.41 s (**too slow for placebos**: about 200 s per placebo run). Hence the SVD fast path with its equality unit test. | – | – |
| RF, 300 trees, min leaf 400, 13k rows | 1.0 s (8 features) to 3.4 s (30 features), n_jobs=-1 | M2 and M3 × (1 static + 14 expanding + 14 rolling) = 58 fits | ≈ 2–3 min |
| Placebos | as an M3 ridge expanding run (≈ 15 s) | 20 shifts (F20) + 7 (F29) | ≈ 7 min serial; ≈ 2 min with `joblib.Parallel(n_jobs=4)` over shifts |
| Permutation importance | linear predict about 1 ms | 14 years × 20 repeats × about 8 groups × 2 models | < 30 s (RF M2 with R = 10: about 1 min) |
| Bootstraps (coefficient SE, Sharpe, α) | numpy | 14 × 500 OLS; 2,000 × a few | < 30 s |
| **Total** | | | **≈ 8–12 min** top to bottom (≈ 15 min worst case, serial) |

- **Refit frequency decision:** annual refits (14) are cheap, and a monthly refit (168) would still take only minutes for the linear models. Annual is chosen for alignment with L5 p.51 and for interpretability, not compute.
- The benchmark trailing mean is updated **monthly** regardless (exam cell 36). This slightly favours the benchmark, which is conservative.
- Flags: `RUN_SUPP_PLACEBO=True` and `N_JOBS=4`. There is no on-disk caching, so the notebook runs top to bottom as required (exam cell 38).

---

## Part 8. Specification search disclosure and pre-registration (L4 p.25–28; L5 p.57)

### 8.1 Pre-registration protocol (against the garden of forking paths)

1. Cell P3-0 (markdown + a `PREREG` dict) records §0.1–§0.3, the spec list below and the portfolio rules **before any modelling cell**. The headline numbers are read from `PREREG` programmatically.
2. Development and debugging run with `OOS_LOCKED = True`: `score_oos()` raises. Pseudo-OOS checks use 2009–2010, with the training data ending 2008-12, inside the initial block.
3. Unlock once, and run all specs, placebos and portfolios top to bottom. Spec definitions are not edited after unlocking. Any post-unlock change becomes a **new, disclosed spec** with its reason, listed under "what we tried that did not work".
4. All specs are reported in T14 whatever the outcome. None is dropped because it looks bad.

### 8.2 Forecasting specifications evaluated OOS (39)

| IDs | Spec | Model | Scheme(s) | Role |
|---|---|---|---|---|
| F01–F03 | M1 class dummies | OLS | S/E/R | benchmark-plus |
| F04–F06 | M2 | OLS | S/E/R | secondary |
| F07–F09 | M2 | Ridge | S/E/R | **F08 = PRIMARY Q1** |
| F10–F12 | M2 | Lasso (validation) | S/E/R | secondary |
| F13–F15 | M2 | RF | S/E/R | secondary (nonlinear) |
| F16–F18 | M3 | OLS (PCR-macro) | S/E/R | secondary |
| F19–F21 | M3 | Ridge | S/E/R | **F20 = PRIMARY Q2** |
| F22–F24 | M3 | Lasso | S/E/R | secondary |
| F25–F27 | M3 | RF | S/E/R | secondary |
| F28 | M4 (M3 + characteristics × global PC1) | Ridge | E | secondary |
| F29 | M5 (M3 + extended PCR) | Ridge | E | exploratory |
| F30 | M3-raw (no PCA) | Ridge | E | robustness |
| F31 | M3-dev (deviation from the cross-country mean) | Ridge | E | robustness |
| F32 | M3 with macro shift 1 | Ridge | E | robustness |
| F33 | M2 with within-class z-scores | Ridge | E | robustness |
| F34 | M2 with training target winsorised at ±4 | Ridge | E | robustness |
| F35 | M2 per class | Ridge | E | per-class |
| F36 | M3 per class | Ridge | E | per-class |
| F37 | M2 | Lasso-BIC | E | tuning robustness |
| F38 | M3 | Lasso-BIC | E | tuning robustness |
| F39 | M2-TS (M2 + own-history trailing z of x1..x5) | Ridge | E | robustness |

- **Null draws (not candidates):** 20 placebo runs of the F20 pipeline and 7 of F29.
- **Portfolios:** 6 forecast rules (P1–P6), 3 benchmarks, and 6 cost levels, all reported.
- **Inner-fold search, disclosed but inside training only:** 50 λ (× 9 k combinations for M3) per refit.
- **Critical values** (normal approximation; t_{167} values in brackets):
  - a single pre-registered test: 1.96 [1.97];
  - Bonferroni for the 3 primaries: 2.394 [2.418];
  - Bonferroni for the best of 39: **3.21** [3.27] (L4 p.28: "the fix is … a different critical value").

  Any non-primary spec that "wins" is interpreted against 3.21, not 1.96.

---

## 9. Notebook cell layout (inside cell 39 onward, before the write-up in cell 40)

| Cell | Content |
|---|---|
| P3-0 | Pre-registration (markdown) + `PREREG` dict + `OOS_LOCKED = True` |
| P3-1 | Load (exam cell 35 objects), assertions, data description (1.0) |
| P3-2 | Return EDA: F1–F3, T1, T2 |
| P3-3 | Characteristic EDA and identification checks: F4, T3, T3b; tercile sorts on the training block: T4, F5 |
| P3-4 | Macro diagnostics: x10 look-ahead (F6a), x11 rebuild quality and stale-ffill error (F6b), duplicates / global / frequency / gap detection in the extended file (T5, T5b) |
| P3-5 | Feature construction (2.1–2.9), truncation test, predictor table T6 |
| P3-6 | **Benchmarks first:** trailing-mean and zero benchmark vectors; EW / RP / TSMOM weights and performance (T11 benchmark rows) |
| P3-7 | Harness and unit tests (fold asserts, ridge == Pipeline, placebo s = 0, benchmark check, grep for forbidden calls) |
| P3-8 | `OOS_LOCKED = False`, then run F01–F39: T7, T10, F8 |
| P3-9 | By class and per-class vs pooled: T8 |
| P3-10 | Macro vs no-macro and placebos: T9, F11 |
| P3-11 | Signal and stability: F9a–c, F10, PD, fit plot |
| P3-12 | Portfolios: T11–T13, F12–F14 |
| P3-13 | T14 (search disclosure) and T15 (leakage audit), printed |

---

## 10. Consolidated open questions (defaults in bold)

1. Trailing-mean benchmark for the panel: **pooled y trailing mean (primary)**, or per-asset (always reported as secondary)?
2. Macro release lag: **2 months** (1 month as robustness F32)?
3. σ̂ for the target and weights: **36-month SD of own excess returns**, or 12 months?
4. Class A country macro: **none**, or c7's, or the cross-country average?
5. Stale x1 for B assets in 2024-10..12: **keep and flag**, or set to the class-month median rank?
6. OOS window: **2011-01..2024-12**, or start 2008-01?
7. Extended file: **one exploratory PCR spec (F29) only**, or none?
8. Headline transaction cost: **10 bp per unit of one-way turnover**, with grid {0, 2, 5, 10, 20, 50} bp?
9. Seed: **7034** (as in Problem 1), or the student-ID seed used in Problem 2?
10. Outside data: **none**?
11. GBRT and the static-vs-dynamic attribution (both outside the default): **not run** unless approved.

## 11. Scope notes

- **Lecture-grounded:**
  - OLS/MLR, dummies, interactions, logs, multicollinearity, diagnostics (L2, L3);
  - R²_OOS, train/validation/test, multiple testing, IC (L4);
  - ridge, lasso, scaling, time-series CV, expanding and rolling windows, the leakage checklist (L5);
  - PCA (L1);
  - RF, permutation importance, partial dependence (L8);
  - t-tests, the bootstrap and α tests (L2).
- **[EXAM-DEFINED]** (the notebook's definition is used, and the write-up says so):
  - the vol-scaled target recommendation, the trailing-mean benchmark updated monthly, EW/RP/TSMOM, unit gross exposure and monthly rebalancing (cells 36–37);
  - the Sharpe ratio, drawdown, turnover and "net of transaction costs" (cell 37);
  - the 36-month circular-shift placebo (cell 37);
  - cross-sectional standardisation within month (cell 37);
  - rolling z-scores for global macro (cell 37);
  - differentials vs country 7 (cell 37).
- **Our assumptions, flagged:** the cost model (proportional, uniform, drift-adjusted turnover), the 2/5/14-month release-delay lags, the placebo shift set, and the months-as-independent-units tests.
- **Dropped as outside scope:** HAC/Newey–West, Diebold–Mariano/Clark–West by name (replaced by a t-test of a mean loss differential, L2), block bootstrap, Sharpe-difference tests such as Jobson–Korkie, neural nets, SVM, XGBoost, SHAP, OOB error, elastic net beyond its formula, KNN (dominated, L6 p.37), clustering as a modelling device.
