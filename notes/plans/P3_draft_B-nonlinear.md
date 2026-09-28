# Problem 3: Research design, Draft B ("flexible nonlinear learners")

This is a plan only. Nothing here was fitted or computed on the exam data. The only code that was run is (a) API and signature checks in the installed environment (Python 3.11, pandas 3.0.6, sklearn 1.9.1, scipy 1.17, statsmodels 0.15) and (b) one timing run on **synthetic random data** of exam-like shape, used for the compute budget. Facts about the data come from the pre-modelling profile (cited as "DP §X").

**How sources are cited.**
- "L<n> p.<k>" means a lecture page, taken from the slide notes.
- "cell N" means a cell of `Final_Autumn_2026-1.ipynb`: 33 = the question, 34 = the data and "Read this before modelling", 36 = "Two rules", 37 = the suggested workflow, 38 = "What to submit".
- "AIG §x" means the AI Coding Guide.
- "DP §x" means `notes/data_profile.md`.

**Scope labels used throughout.**
- **[LEC]**: taught in the lecture cited.
- **[EXAM]**: not taught in any lecture, but defined or required by the exam notebook (the cell is cited).
- **[FLAG: outside scope]**: in neither source. Proposed only with the student's approval, and never part of a headline number.

---

## 0. Design at a glance

### 0.1 The three questions and how each is answered

| Question (cell 33) | Primary evidence (pre-registered) | Decision rule, fixed now |
|---|---|---|
| Q1. How much of next month's excess return is forecastable from lagged characteristics? | $R^2_{OOS}$ of **RF-F0-expanding** on the vol-scaled target, measured against the pooled trailing mean and against zero, over 2013-01..2024-12. **Ridge-F0-expanding** is the pre-registered linear comparator. | Report the number, whatever it is. Call it "forecastable" only if the loss-differential t-stat exceeds 1.98 (a single pre-registered test). Show the search-adjusted value 3.47 alongside it (L4 p.28). |
| Q2. Does macro (global or country) add anything? | $\Delta R^2_{OOS}$ = RF-F1 − RF-F0 (expanding). Split into global and country parts by refitting (F0+G, F0+C) and by block-permutation importance. Compare with a placebo using 4 circular shifts. | "Macro adds" requires two things: $\Delta R^2_{OOS}>0$ with t > 1.98, **and** RF-F1 beating every placebo shift. Anything else is written up as "no evidence that macro adds". |
| Q3. Does a forecast portfolio beat EW, RP and TSMOM out of sample, after costs? | Portfolio P1 (w ∝ ŷ/σ̂, unit gross exposure) built from RF-F1-expanding forecasts. Net of 10 bp per unit traded. Same 144 months as the benchmarks. | "Beats" requires two things: a higher net Sharpe than each benchmark with the paired-bootstrap 95% CI of ΔSharpe excluding 0, **and** a positive α (t > 1.98) in the regression on the three benchmark returns (L2 p.94-97). |

### 0.2 Key dates (fixed before anything is fitted, as cell 36 rule 2 requires)

| Item | Value | Rows (50 assets) | Why |
|---|---|---|---|
| Raw data | 2000-01..2024-12 | 300 months, 15,000 rows | DP §C |
| First usable target | 2003-01 | – | σ̂ needs 36 past months (2000-01..2002-12). This also clears every leading back-fill of x2/x3/x5, the last of which is used as a predictor in 2002-09 (DP §D). |
| Modelling frame | 2003-01..2024-12 | 264 months, 13,200 rows | – |
| Initial training block | 2003-01..2012-12 | 120 months, 6,000 rows | 10 years that include one full cycle (the 2003-07 boom, the GFC, the 2010-12 euro crisis), so the first model has seen a crisis. |
| Validation inside each training window | the last 36 months of that window | 1,800 rows | Time-ordered train → validation → test (L5 p.48-52). 36 of 120 months is 30%, which avoids starving the inner training set (L5 p.27). |
| OOS test window | **2013-01..2024-12** | 144 months, 7,200 rows (A 3,744 / B 1,152 / C 1,296 / D 1,008) | 12 years gives an annualised-Sharpe SE of about 1/√12 ≈ 0.29. The window spans distinct regimes (taper 2013, oil 2014-16, COVID 2020, inflation 2021-23). |
| Refits | every January, 2013..2024 | 12 refits | Annual step as in L5 p.51. Refitting monthly would cost 12× for a 1/12 change in the data. |

### 0.3 The pre-registered primary specification

This is written into the notebook as a markdown cell **before** any model cell, and a config dict is printed next to it (see §9).

- **Target:** $y_{i,t}=r_{i,t}/\hat\sigma_{i,t-1}$, where $\hat\sigma_{i,t-1}$ = sd(ddof=1) of $r_{i,t-36..t-1}$.
- **Primary forecaster:** `RandomForestRegressor(n_estimators=300, max_features=1/3, min_samples_leaf∈{0.005,0.02,0.08} tuned on validation, random_state=7034)`. It is pooled across all 50 assets, uses feature set **F1** (characteristics + class dummies + global + country macro; 20 features), and is refit each January on an expanding window.
- **Linear comparator:** `Pipeline([StandardScaler(), Ridge(alpha)])` on the same rows, with α tuned on the same validation blocks.
- **Benchmarks:** the pooled trailing mean of y through t−1, updated monthly, and zero.
- **Portfolio:** P1, w ∝ ŷ/σ̂ with unit gross exposure, rebalanced monthly. Costs are 10 bp per unit of drift-adjusted traded notional.
- **Expectations, written down before computing** (L2 p.78, L4 p.26):
  - pooled $R^2_{OOS}$ between −1% and +1% (L5 p.57 gives 0.3-0.5% as the stock-level norm; L1 p.33 notes the low signal-to-noise ratio);
  - RF ≈ Ridge (L8 p.62);
  - macro adds about 0, and the placebo does about as well as the real macro (L4 p.47-49);
  - no significant net-Sharpe win over TSMOM or RP.

---

## 1. Global conventions (apply to every part)

### 1.1 Configuration cell (first code cell of P3)

```python
SEED        = 7034                      # all random_state, bootstrap, permutation, PD subsample
N_JOBS      = min(4, os.cpu_count())    # shared class server
DATES       = pd.date_range('2000-01-31', '2024-12-31', freq='ME')   # datetime64[us] in pandas 3 (checked)
EST_START   = pd.Timestamp('2003-01-31')
OOS_START, OOS_END = pd.Timestamp('2013-01-31'), pd.Timestamp('2024-12-31')
REFIT_YEARS = range(2013, 2025)         # refit each January
VAL_MONTHS  = 36; ROLL_MONTHS = 120; SIG_WIN = 36; MACRO_LAG = 2; GZ_WIN, GZ_MIN = 60, 24
CLIP_Z      = 3.0
PLACEBO_SHIFTS = [36, 60, 96, 150]
COST_GRID_BP   = [0, 2, 5, 10, 25, 50]; COST_PRIMARY_BP = 10
FAST        = False                     # True: re-tune every 3 years, GBRT depth {1,2} (see §8)
```

### 1.2 Units and scales
- Returns are decimals and **excess** returns (cell 34).
- Macro rates and inflation are in percent. x1 is in percent for class D and in monthly decimals for class B. Some extended-file levels reach 1e12 (DP §D, §G, §I.6).
- Consequences:
  - No characteristic is ever pooled in raw units. §3.3 standardises them within class and month.
  - Level macro series that are strictly positive and skewed get a log transform (L3 p.48-49).

### 1.3 Library and pandas-3 guards (AIG §3, §4; DP header)
- Never use `df.append`, `fillna(method=)`, `.iteritems()`, `LassoCV(normalize=)` or `sns.set()`. Use `pd.concat`, `.ffill()` and `.items()` instead, and put a `StandardScaler` inside a `Pipeline`.
- String columns load as `StringDtype`, and dates as `datetime64[us]`. `pd.date_range` is also `[us]` (checked), so key merges are safe.
- `DataFrame.corr().values` is **read-only**. Take a `.copy()` before editing it, for example before masking the diagonal of a heat map.
- Country files are in natural order, and `sort_values('country')` sorts lexicographically. **Always merge on `(country, date)` keys with `validate='many_to_one'`, never by position** (DP §F, §I.10).
- `groupby().shift()` only after `sort_values(['asset_id','date'])`, then an assertion that dates are monotone within each asset.
- Cell 35 runs `warnings.filterwarnings('ignore')`, which hides Lasso ConvergenceWarnings. So set `max_iter=10_000` and assert `n_iter_ < max_iter`.
- **Never** call `train_test_split`, `KFold`, `cross_val_score` (with default CV), `RidgeCV()` (default = leave-one-out GCV over rows), `LassoCV(cv=int)`, `GradientBoostingRegressor(n_iter_no_change=…)` or `HistGradientBoostingRegressor` (default `early_stopping='auto'`).
  - This was checked in the sklearn 1.9.1 source: `GradientBoostingRegressor.fit` with `n_iter_no_change` calls `train_test_split(X, y, …, test_size=validation_fraction)` with the default `shuffle=True`. That is a random fold on time-ordered data, a fatal error (AIG §4a; L5 p.58 #3).
- `TimeSeriesSplit` is not used either. It splits by **row count**, and in an asset-month panel that can cut a month in half. All splits use explicit **date masks**.
- No `r2_score` and no `model.score`. Every $R^2_{OOS}$ goes through the hand-written function in §4.6 (AIG §4d; L5 p.57).
- `LogisticRegression` is not used anywhere, because nothing is classified. If a sign classifier were ever added, it would use `penalty=None` for the MLE (AIG §4c; L6 p.38).

### 1.4 Scope ledger

| Item | Status | Grounding |
|---|---|---|
| OLS / MLR, dummies, interactions, logs, multicollinearity | [LEC] | L3 p.19-26, p.45-55, p.48-49, p.39-44 |
| Ridge, lasso, standardisation, grids, time-ordered validation, $R^2_{OOS}$, leakage checklist | [LEC] | L5 p.12-24, p.23, p.27, p.48-59 |
| Train/validation/test roles, IS vs OOS $R^2$, benchmark = training mean or 0, multiple testing, Bonferroni | [LEC] | L4 p.43-48, p.25-28 |
| CART, RF, GBRT, early stopping on a held-out curve, permutation importance, PD | [LEC] | L8 p.19-22, p.29, p.41-60 |
| KNN (as a *regression* local average) | [LEC], adapted | L6 p.21-27. L6 shows the classifier. Its regression analogue replaces the vote by the neighbours' mean of y, which estimates $E[Y\mid X]$, the regression risk minimiser in L6 p.21. KNN is also listed in cell 33. |
| PCA (only for the exploratory extended-file spec) | [LEC] | L1 p.45-74, p.43 (scree); L8 p.61 (trees on PCs) |
| t-tests, CLT, bootstrap of periods, α regressions | [LEC] | L2 p.59-70, p.78-80, p.91-97, p.47-49 |
| Vol-scaled target, cross-sectional standardisation, trailing macro z-score, tercile sorts, placebo circular shift, EW/RP/TSMOM, unit gross exposure, monthly rebalancing | [EXAM] | cells 36, 37. Conceptual lecture support: L1 p.35 (sorting), L3 p.2-3 and L2 p.80 (unequal variances), L4 p.26-28 (an honest null). |
| Sharpe ratio, max drawdown, turnover, transaction costs, break-even cost | [EXAM]. These are requested in cell 37 step 4, but no lecture defines them (see the L2/L5 "boundaries"). | The formulas are defined in §5 and labelled "defined here". |
| Drift-adjusted turnover, a proportional cost grid, a class-tilt diagnostic portfolio, block (group) permutation | [FLAG: outside scope, minor]. Needed to operationalise exam requests. | Stated as transparent assumptions. Block permutation is the L8 p.57 algorithm applied to a set of columns. |
| HAC/Newey-West SEs, Diebold-Mariano and Clark-West tests, elastic net beyond its formula, neural nets, XGBoost/LightGBM/HistGB, OOB error, SHAP/ICE, PCR as a model | **Not used.** These are mentions only, or absent (L2 p.80; L5 p.11; L8 p.62; L8 "boundaries"; L1 "boundaries"). | – |

---

## 2. Part 1: Know your data

### (i) Goal
Describe the returns and their block structure, how persistent performance is, and the distribution, scale and persistence of x1..x5 within each class. Take a *snooping-free* first look at predictive content through tercile sorts. Form hypotheses about what x1..x5 are. Audit the macro files, because Part 2 depends on it.

### (ii) Method steps, (iii) lecture grounding

| # | Step (exact computation) | Grounding |
|---|---|---|
| 1.1 | **Load and validate.** Assert: `panel.shape==(15000,10)`; 300 unique month-ends equal to `DATES`; (asset, date) keys unique; `asset_info` matches the panel; `wide` equals the pivoted panel (max abs diff = 0); `mac_c` and `mac_x` have 3,600 rows with unique (country, date) keys. | AIG §6 steps 1 and 4 ("check against something you know"); L1 p.82 ("ask what the data are") |
| 1.2 | **Cumulative excess returns by class.** For each class c: $r_{c,t}$ = mean over $i\in c$ of $r_{i,t}$. Plot $\sum_{s\le t}\log(1+r_{c,s})$ for 2000-2024 (4 lines). Panel (b): the class average of vol-scaled y, cumulated from 2003-01, so that classes are compared at comparable risk. | L1 p.37-38 (EDA; plot the raw series; model returns); L1 p.39 (compare at comparable risk) |
| 1.3 | **Per-asset table** (50 rows). Columns: class, country, annualised mean (12·mean), annualised vol (√12·sd, ddof=1), Sharpe = ann. mean / ann. vol, worst month (value and date). Add a class-median summary row. | ddof=1: L2 p.56-57. Sharpe: [EXAM] (cell 37 step 1) |
| 1.4 | **Correlation matrix ordered by class** (A, B, C, D, then country, then asset id). Heat map with vmin=−1, vmax=1. Block table of mean off-diagonal correlations, within and between classes. Two versions: **2000-01..2012-12**, which is the one used for design statements, and the full sample for description. | `df.corr()` + heat map: L1 p.44; block structure: L1 p.42 |
| 1.5 | **Rolling 12-month Sharpe by class** (class EW): √12·mean₁₂/sd₁₂. Persistence: correlation between calendar-year Sharpe in year k and k+1 (24 pairs per class), plus the autocorrelation of the rolling series at lag 12. | [EXAM] cell 37 step 1; L1 p.34 (short, nonstationary samples) |
| 1.6 | **Characteristics by class** (raw units):<br>• table of mean, sd, p1, p50, p99, min, max<br>• median within-asset ACF at lags 1, 6, 12, 24<br>• median within-class monthly cross-sectional sd<br>• share of variance due to asset means<br>• boxplots, 5 characteristics × 4 classes<br>• time series of the class-month cross-sectional mean | L1 p.37 (statistics first); L3 p.48 (think about scale) |
| 1.7 | **Identification checks.** These compare predictors only with *past* returns and with macro, never with future returns (DP §D):<br>• x2 against compounded $R12_{t-1}$: share within 1e-5<br>• x5 against $SD36_{t-1}$: share exact, and the asset_16 floor at 0.004<br>• x3 against the 60-month log cumulative return (corr)<br>• x1 (B) against the short-rate differential x86 − x86(c7)<br>• x1 (D) against the term spread x85 − x86<br>• x4 (B) against Δ12 of the rate differential | L1 p.44 (correlation and scatter); AIG §6.4 |
| 1.8 | **Leading back-fill detector.** For each asset and each of x2/x3/x5, find the run of identical values starting 2000-01. Filled months = run − 1. Expect 7 assets, all ending by 2002-08 (DP §D table). | L5 p.58 #1 (imputation counts as a transform); cell 34 |
| 1.9 | **Tercile sorts, TRAINING BLOCK ONLY (2003-01..2012-12, 120 months).**<br>• Each month, within each class, rank the *lagged* characteristic (x1, x3, x4 shifted +1; x2, x5 as stored).<br>• T1 = bottom third by rank (rank ≤ n/3), T3 = top third (rank > 2n/3). Class D (7 assets) gives 2/3/2 assets.<br>• Record next-month **y** (vol-scaled, so that class A does not dominate) for T1, T2, T3, and the monthly spread T3 − T1.<br>• Report the 120-month mean of each, the t-stat of the spread (sd with ddof=1 over √120, compared with $t_{119}$), per class and as an equal-weighted average over classes.<br>• 25 tests (5 characteristics × 4 classes + 5 pooled), so show the Bonferroni critical value $t_{119}^{-1}(1-0.05/50)\approx3.1$ next to them. | Sorting: L1 p.35; t-test of a mean: L2 p.91-93; multiple testing: L4 p.25-28 |
| 1.10 | **Macro audit.**<br>• Curated file: the x11 NaN pattern is trailing only (c12 from 2020-11, c9 from 2021-08, c11 from 2024-04; 99 NaN in total). x10 is constant within each year and equals the same-year mean of x86 (show country_7 in 2022). Shared series: x10 in c2/4/6/8, x14 in c10/c11.<br>• Global file: AR(1) and time plots.<br>• Extended file: find exact duplicates by `(a==b)|(a.isna()&b.isna())` over all rows; list the columns that are identical across countries; classify frequency (share of months changing, whether changes happen only in January or only at quarter starts); list the gap columns; flag magnitudes above 1e6. | L3 p.40 and L4 p.7/p.14 (exact duplicates cause rank deficiency); L5 p.58 #2 (publication-date merges) |

### (iv) Assumptions and why
- **A1.1** Descriptive return statistics (1.2-1.5) use the **full sample**, as cell 37 asks. They do not relate any predictor to a later return, so they cannot select a predictor.
  - Pre-registered constraint: the universe is always all 50 assets, and no asset is dropped because of anything seen in these tables.
- **A1.2** **Tercile sorts use the training block only.**
  - Reason: sorts relate predictors to future returns. Running them over 2013-2024 would put the test window into the design ("never use the same data twice", L5 p.41; choosing on the test block, L5 p.58 #4; screening on the full sample "is cheating", L5 p.41; searching finds noise, L4 p.25-28).
  - The same sorts are recomputed on 2013-2024 **only after the design is frozen**, and reported in Results as an ex-post check that is used for nothing.
  - Pre-registered: no characteristic is dropped or selected on the basis of the sorts. All five enter every model.
- **A1.3** Correlation-based design statements (for example "B/C/D blocks are strong, so compare per-class and pooled models") cite the 2000-2012 matrix.

### (v) Pitfall guards
- Every sort uses the lagged alignment of §3.1. There is an assertion that `sort_frame.date.max() <= '2012-12-31'`.
- `.corr().values.copy()` before masking.
- No `sns.set()`.
- Heat-map order is built from `info` sorted by `(asset_class, country, asset_id)`, and the ids are checked against the panel.

### (vi) Outputs
- Fig D1: cumulative returns by class (2 panels).
- Table D1: per-asset statistics (appendix) plus a class summary (text).
- Fig D2: correlation heat maps (training block and full sample).
- Table D2: class-block average correlations.
- Fig D3: rolling 12-month Sharpe by class.
- Table D3: persistence of Sharpe.
- Table D4: characteristic summary by class.
- Fig D4: boxplots of the characteristics.
- Table D5: ACF persistence.
- Table D6: identification checks (x2 = R12 exact, x5 = SD36 exact, …).
- Table D7: tercile sorts on the training block.
- Table D8: macro audit (frequency, missingness, duplicates, look-ahead, shared series).
- Table D9: back-fill runs.

### (vii) What the write-up must cover (cell 37 step 1)
- ☐ cumulative returns by class
- ☐ per-asset mean / vol / Sharpe / worst month
- ☐ correlation matrix ordered by class, how strong the blocks are, and **what that implies for pooling** (B/C/D blocks → shared common shocks → effective cross-sectional N is much smaller than 50; this motivates the per-class check and the class-balanced sort averages)
- ☐ rolling Sharpe persistence
- ☐ distribution, scale and persistence of each characteristic by class
- ☐ tercile-sort first look, **and why it was run on the training block only**
- ☐ a hypothesis for each of x1..x5:
  - x1 = carry (rate differential for B, term spread for D, dividend-yield-minus-rate for C, roll/basis for A)
  - x2 = 12-month momentum, pre-lagged (exact)
  - x3 = value / 5-year reversal (B/M-like for C)
  - x4 = 12-month change in the carry fundamental
  - x5 = 36-month volatility, pre-lagged (exact; asset_16 floored)

### (viii) Open questions
- OQ1.1: show the full-sample correlation matrix in the Data section, or only the training-block one? **Default: both. Design statements cite the training block.**

---

## 3. Part 2: Feature engineering

### (i) Goal
Build a predictor set where every value is known at the end of month t−1, is comparable across classes, and handles the traps listed in DP §I. Define the target and how a forecast becomes a position. End with one table of every predictor and the counts (cell 37 step 2).

### 3.1 Lag rules (by variable)

**Rule:** a value describing period P may be used to predict target month t only if P ends no later than t − L. Here L = 1 month for characteristics (cell 34) and L = 2 months for macro (1 month minimum plus 1 month release delay; cell 34 "you may argue for more"). This is the "merge on publication date, not period end" rule (L5 p.58 #2).

| Variable | Stored as | Frequency | Shift applied (within asset or country) | Note |
|---|---|---|---|---|
| x2, x5 | already lagged (cell 34) | monthly | 0 | Check: x2 = $R12_{t-1}$ and x5 = $SD36_{t-1}$ exactly (DP §D) |
| x1, x3, x4 | contemporaneous | monthly | +1 | `groupby('asset_id').shift(1)` |
| x6..x9 (global) | contemporaneous | monthly | +2 | Market-priced x6 could argue for L=1 (OQ2.2). The uniform L=2 is conservative and gives a single rule. |
| x11, x12, x13, x14, x15, x16 (curated) | contemporaneous | monthly | +2 | x16 is monthly with step changes |
| **x10** (curated) | **same-calendar-year mean of x86, stamped from January** | annual | **dropped** | Up to 11 months of look-ahead, which a 1-month lag does not fix (DP §F). It is replaced by x86 at L=2, which carries the same information at a more timely frequency. |
| x85, x86 (extended) | contemporaneous | monthly | +2 | Monthly long yield and short rate (DP §G), both complete. |
| x73 (extended copy of x10) and all other annual/quarterly extended columns | same-year or same-quarter averages | annual / quarterly | not used | If ever used: annual year-Y value becomes usable from March Y+1; quarterly Q value from 2 months after the quarter ends. |

### 3.2 Missing values, per series (only past information)

| Series | Pattern (DP) | Treatment | Why it uses only past information |
|---|---|---|---|
| x2/x3/x5 leading runs (7 assets, 2000-01..≤2002-08) | backward fill with the first genuine value (look-ahead) | Set the filled cells to NaN. They never enter any model because estimation starts in 2003-01. Assert: after lagging, no filled cell falls on or after 2003-01. | Removes a backward fill (cell 37: "filling backward … uses the future") |
| x11 for c12 / c9 / c11 | stops at 2020-10 / 2021-08 / 2024-03; no interior holes | **Rebuild** each trailing gap as x86 − x12 of the same month, then apply the 2-month lag like every other series. Report overlap quality per country from the notebook, expecting c12 RMSE 0.78 / corr 0.871, c9 0.20 / 0.992, c11 0.33 / 0.972, pooled 0.988 (DP §F). Plot for c12: x11, stale forward fill, and the rebuild. | Uses only same-month values of two complete columns. **Not** forward-filled, because a stale fill during the 2021-23 inflation shock is wrong by up to 10.1 pp (c12). **Not** x10 − x12, because x10 carries look-ahead (DP §F). |
| x1, B assets a8/a17/a30/a31/a34, 2024-09..12 | frozen (stale) | Keep as stored and flag it. It affects ≤ 15 OOS rows (0.2%). OQ2.5 offers a rebuild instead. | Stale values are known at the time and are what a desk would have seen. |
| x5 of asset_16 floored at 0.004 (2019-09..2023-03) | provider floor | Keep in cs_x5, where the effect on a within-class z-score is negligible. **σ̂ is computed from returns, so it is unaffected.** | – |
| Extended gap columns x55, x82, x83, x113, x136, x163, x164 | trailing gaps | Not used (§3.9) | – |
| Global x6..x9, the asset panel | no gaps | – | – |

### 3.3 Standardising the characteristics (cross-sectionally, within month)
- **Choice:** z-score **within asset class and within month**, clipped at ±3:
  $$cs\_x_k[i,t]=\mathrm{clip}\!\left(\frac{x_k[i,t]-\bar x_{k,c(i),t}}{sd_{k,c(i),t}},-3,3\right)$$
  using the lagged values, sd with ddof=1, and 0 if $sd=0$. Formula from L7 p.23, applied cross-sectionally.
- **Why within class, not across all 50 assets.** Units differ by up to three orders of magnitude across classes: x1 in D is in percent while x1 in B is a monthly decimal, and x4 in B/C is about 100× x4 in A/D (DP §D). A z-score across all 50 assets would mostly rank classes, not assets. Class-level differences are carried by class dummies instead (§3.8). Penalised and distance-based methods need comparable scales (L5 p.23, L6 p.23).
- **Why z-score, not rank.** The z-score keeps magnitude (how *much* higher the carry is), which is economically meaningful for carry and value.
  - Outliers (x3 in A has a minimum of −9.5; x2 has winsorised extremes) are handled by a **fixed** ±3 clip. The clip is not estimated from data, so it is not a fitted winsorisation (L5 p.58 #1).
  - For n = 7 (class D) the largest attainable |z| is 2.27, so the clip only binds in class A.
  - Within-class **rank** (uniform on [−0.5, 0.5], ties averaged) is a disclosed robustness spec (E1). This is L3 p.17's "run with and without".
- **No leakage:** each month uses only that month's cross-section of lagged values. Nothing is fitted across time.

### 3.4 One time-series characteristic (pre-registered)
- `ts_mom = clip(x2 / (√12 · x5), −3, 3)`: the past 12-month return divided by the annualised 36-month vol. It is unit-free and comparable across classes, so it is pooled **without** cross-sectional demeaning.
- **Why it is included.** Cross-sectional demeaning removes the time-series level of every characteristic. Without this feature the pooled model could not learn TSMOM-type timing, which is exactly one of the Q3 benchmarks. With it, the Q3 comparison asks whether the model beats a rule it could have learned (L8 p.28: the tree can find the nonlinearity or interaction).
- Both inputs are already pre-lagged (cell 34).

### 3.5 Global macro (x6..x9)
- Lag +2, then a **rolling 60-month z-score**: `(m − m.rolling(60, min_periods=24).mean()) / m.rolling(60, min_periods=24).std(ddof=1)`. This uses only values up to the lagged date and first becomes available in 2002-02, before the first target (cell 37 "trailing information only").
- The trailing z-score is invariant to any affine full-sample standardisation the provider may have applied (x8 looks z-scored, with mean 0.23 and sd 1.01, DP §E). Vendor look-ahead in the level or scale therefore cancels.
- **4 features:** `g_x6, g_x7, g_x8, g_x9`, the same for all assets in a month.

### 3.6 Country macro and the country-to-asset mapping
- **Series.** Seven economically motivated country series, chosen before looking at any returns:

  | Feature | Definition | Transform |
  |---|---|---|
  | `sr` | x86 (short rate) | none |
  | `ts` | x85 − x86 (term spread) | none |
  | `rr` | x11 with the §3.2 rebuild (real rate) | none |
  | `fx` | x13 (YoY change in the effective exchange rate) | none |
  | `epu` | x14 (EPU-type index; L3 p.48-49) | log |
  | `gpr` | x15 (GPR-type index; skewed, min 0.004, max 13.2) | log1p |
  | `rat` | x16 (country risk rating) | none |

  Inflation enters implicitly (sr − rr ≈ infl), which avoids an exact linear dependence among sr, rr and x12 (L3 p.40; L4 p.14).
- **Transform: differential against country 7**, $d\_k[i,t]=M_{k,c(i),t-2}-M_{k,7,t-2}$, for assets in B, C and D.
  - Reason: B assets look like currencies against country 7's currency (country 7 has no B asset and B's Sharpe is about 0, DP §C). For B, x1 and x4 are *exactly* functions of the rate differential (DP §D). The differential is therefore the natural carry and macro state, and cell 37 names country 7 as the natural base.
  - C and D assets use the same transform for uniformity. For the two country-7 assets (asset_24 C, asset_27 D) the differential is identically 0. This is a known cost, and OQ2.3 records the alternative.
- **Class A:** the country block is set to **0**. Under the differential convention, "give A country 7's macro" and "give A no country macro" are *the same thing* (0 either way). Class A still receives the global block, which is the cross-country information that is sensible for commodity-like assets with no country (cell 34). An alternative for A is recorded in OQ2.4.
- Differentials enter in their natural units. Linear and KNN models scale them inside the Pipeline on training rows only (L5 p.23, p.47). Trees need no scaling.
- **7 features:** `d_sr, d_ts, d_rr, d_fx, d_epu, d_gpr, d_rat`.

### 3.7 Cross-sectional macro transforms
- Considered: rank across the 12 countries, deviation from the cross-country mean, and a widened panel (cell 37).
- The primary spec uses the c7 differential only.
- **Disclosed robustness E3 ("xsdev"):** replace the differential with $M_{k,c(i)}-\frac1{12}\sum_c M_{k,c}$. This version is non-zero for country-7 assets. Class A stays 0.
- Ranks across 12 countries have ties (x10 is identical for c2/4/6/8, x14 for c10/11, x86 for c2 and c6), so they are not used.
- The widened panel is not used, because of the parameter count (L1 p.34).

### 3.8 Interactions and class dummies
- **Class dummies:** `pd.get_dummies(asset_class, drop_first=True, dtype=float)` gives B, C, D with A as the reference (L3 p.45, p.51-52; L8 p.26). **3 features**, used by every model except KNN (L6 p.37: no natural handling of categoricals).
- **Trees** (CART, RF, GBRT) get the raw features. "Recursive splitting *is* interaction" (L8 p.37, p.28), so characteristic × macro and characteristic × class effects need no explicit terms.
- **Linear models, set F2:** add the products $cs\_x_k\times g\_x_m$ (5×4) and $ts\_mom\times g\_x_m$ (1×4), giving **24 interaction terms** (L3 p.53-54). F2 against F1 is the linear test of "do characteristic × macro interactions add?". RF/GBRT-F1 against Ridge-F1/F2 is the test of "does *learned* nonlinearity add?".

### 3.9 The extended file and multiple testing
- **Count:** 148 columns.
  - 6 exact duplicates of curated columns (x73=x10, x89=x12, x114=x13, x17=x14, x92=x15, x18=x16) are **dropped before any concatenation**, with an assertion of exact equality including the NaN pattern.
  - 26 columns are identical across countries (x88=x6, x104=x7, x109=x8, x99=x9 exactly; x148 is constant; x160 = −x143). They are not used.
  - 39 annual and 27 quarterly columns carry same-period look-ahead stamping. Not used.
  - 22 irregular/stale columns. Not used.
  - 7 gap columns. Not used.
- **Why not screen ~150 series against returns.** At the 5% level about 7 would come out "significant" by chance, and the best noise t is about 2.9 (L4 p.25-28). Screening is also unstable (L4 p.39, p.50), and screening on the full sample and then "validating" is cheating (L5 p.41).
- **Policy:**
  - (a) **Primary:** use only x85 and x86. They are chosen on economic grounds (rates) and are needed to replace the contaminated x10 and to rebuild x11.
  - (b) **One exploratory spec (E2 "extPCA")**, counted in the search size:
    - Take the 16 complete, monthly, country-varying extended columns that are not duplicates or transforms of used series: x37, x38, x39, x84, x110, x111, x115, x117, x141, x142, x144, x151, x152, x154, x158, x159.
    - Log x37, x39, x84, then lag by 2.
    - `Pipeline([StandardScaler(), PCA(n_components=3, random_state=SEED)])`, **fitted at every refit, and again inside every tuning split, on the training-window country-month rows only** (L1 p.45-74; L8 p.61 "fitting [trees] to low-dimensional factors").
    - PC scores become differentials against c7, and are 0 for class A. That gives 3 extra features for RF-F1.
    - k = 3 is fixed in advance. A scree plot on the first training window is shown for information only (L1 p.43).
  - Correlation screening (L4 p.37-39) is **not** used anywhere.

### 3.10 Target, σ̂, and mapping forecasts to positions
- **σ̂:** $\hat\sigma_{i,t-1}$ = `r.rolling(36, min_periods=36).std(ddof=1).shift(1)` within asset.
  - Computed from `excess_return` only, so the benchmarks can use the same σ̂ (cell 36).
  - It equals x5 except for asset_16's floor. Assert that the share of |σ̂ − x5| < 1e-5 is ≥ 99% from 2003 on.
  - ddof=1 follows L2 p.56-57.
- **Target:** $y_{i,t}=r_{i,t}/\hat\sigma_{i,t-1}$. No clipping of the target in the primary spec (L3 p.17).
- **Why vol-scale.** Monthly vol runs from 0.8% to 17.4% (DP §C). An unscaled pooled squared-error fit would essentially model the 26 class-A assets. It would violate the equal-variance assumption behind OLS inference (L3 p.2-3, p.20; L2 p.37, p.41, p.80) and inflate noise in every estimate (L2 p.71-73). Scaling by lagged vol makes the pooled squared loss roughly homoskedastic and turns the forecast into a predicted Sharpe-type number.
- **Back to positions.** $\hat y$ is the forecast of $r/\hat\sigma$, so $\hat r=\hat y\,\hat\sigma$. With a diagonal risk model, mean/variance weights are $\hat r/\hat\sigma^2=\hat y/\hat\sigma$. That is rule P1 in §5. Each asset's risk budget $w_i\hat\sigma_i$ is then proportional to its forecast $\hat y_i$.
- **Diagnostic only:** forecasts are also mapped to raw units ($\hat r=\hat y\hat\sigma$) to report a raw-return $R^2_{OOS}$ in the appendix, which is dominated by class A.

### 3.11 Placebo construction (cell 37 step 3)
- Take the **cleaned, pre-lag** monthly macro panel: x6..x9, plus sr, ts, rr, fx, epu, gpr, rat per country, with x11 already rebuilt.
- Circularly shift every column by k months within each country (global series as one block): `np.roll(values_sorted_by_date, k)`. Checked: `np.roll(a, 2)[t] = a[t−2]`, wrapping around.
- Then run the **identical** pipeline: lag 2, rolling z-score, c7 differential, mapping.
- Characteristics are **not** shifted. Note that x1 and x4 for B contain real macro at the correct time (DP §D), so the placebo tests only the *separate* macro block.
- Shifts k ∈ {36, 60, 96, 150}. With AR(1) ≈ 0.98, a 36-month shift keeps about 0.98³⁶ ≈ 0.48 of a persistent series's autocorrelation, while 150 months is half the sample and essentially independent (cell 37: "try more than one shift").
- The placebo deliberately uses mis-timed (sometimes future) data. It is a null, not a forecast (L4 p.26-28: "we know there is no signal because we made them up").

### 3.12 Final predictor table (the "one table with counts")

| Block | Features | Count | F0 | F1 | F2 (linear only) | KNN-F0k | KNN-F1k |
|---|---|---|---|---|---|---|---|
| Cross-sectional characteristics (within-class z, clipped) | cs_x1, cs_x2, cs_x3, cs_x4, cs_x5 | 5 | ✓ | ✓ | ✓ | ✓ | ✓ |
| Time-series momentum | ts_mom | 1 | ✓ | ✓ | ✓ | ✓ | ✓ |
| Class dummies (A = reference) | cls_B, cls_C, cls_D | 3 | ✓ | ✓ | ✓ | – | – |
| Global macro (lag 2, rolling-60 z) | g_x6, g_x7, g_x8, g_x9 | 4 | – | ✓ | ✓ | – | ✓ |
| Country macro, differential vs c7 (lag 2; 0 for A) | d_sr, d_ts, d_rr, d_fx, d_epu, d_gpr, d_rat | 7 | – | ✓ | ✓ | – | – |
| Characteristic × global interactions | (cs_x1..5, ts_mom) × g_x6..9 | 24 | – | – | ✓ | – | – |
| **Total** | | | **9** | **20** | **44** | **6** | **10** |

Variants:
- F0+G = 13 features. F0+C = 16 features.
- E2 extPCA = F1 + 3 = 23 features.
- E3 xsdev = 20 features (same count, different country transform).
- Placebo(k) = 20 features.

Assertions:
- The modelling frame has 13,200 rows and **no NaN** in any feature from 2003-01.
- `X.columns` is identical and in the same order at fit and at predict (L3 p.26).

### (iv) Assumptions (summary)
- A2.1 Uniform 2-month macro lag.
- A2.2 x10 dropped and replaced by x86.
- A2.3 x11 rebuilt after it stops.
- A2.4 Within-class z-score, clipped at ±3.
- A2.5 ts_mom added.
- A2.6 Country differential vs c7; class A = 0.
- A2.7 Extended file used only through x85/x86, plus one exploratory PCA spec.
- A2.8 σ̂ = 36-month sd with ddof=1, no floor.
- A2.9 Target not clipped.

### (v) Pitfall guards
- Every transformation is either (a) same-month cross-sectional, (b) trailing (rolling z, σ̂, R12), or (c) fitted inside the training window (scalers, PCA). There is no full-sample mean, sd, imputation or winsorisation anywhere (L5 p.58 #1; AIG §4b).
- Merges on keys only, with `validate=`.
- Assertions for the lag conventions (x2 = R12, x5 = σ̂ checks).
- The duplicate-drop assertion.
- A NaN assertion.

### (vi) Outputs
- Table F1: lag and publication rule per variable.
- Table F2: missing-value treatment, with x11 rebuild quality per country.
- Fig F1: x11 against the rebuild against a stale fill, for c12, c9 and c11.
- Table F3: extended-file disposition, with counts per category.
- Table F4: the final predictor table (§3.12).
- Fig F2: global macro rolling z-scores over time.

### (vii) Write-up checklist (cell 37 step 2)
- ☐ lag rule, including the release delay and the low-frequency fix
- ☐ standardisation choice and why
- ☐ missing values per series: diagnosis, treatment, and why it uses only past information, including the x11 regime trap and rebuild quality
- ☐ added data (none external; only x85/x86 from the provided extended file) with timing
- ☐ trailing standardisation of global macro
- ☐ country mapping, differential, class-A choice
- ☐ cross-sectional transforms considered
- ☐ interactions and class dummies
- ☐ how the extended file was used, and the multiple-testing argument
- ☐ target, why it is vol-scaled, which σ̂, and how forecasts become positions
- ☐ predictor table with counts

### (viii) Open questions (with defaults)
- OQ2.1 σ̂ window: 36 months (default; matches x5 and a stable risk-parity σ̂) or 12 or 24 months (more responsive in 2008/2020)?
- OQ2.2 Macro lag: uniform 2 months (default) or 1 month for market-priced series (x6, x85, x86, x13)?
- OQ2.3 For C/D assets, add own-country **levels** (for example the own term spread for bonds) alongside the differentials? Default: no (fewer forking paths).
- OQ2.4 Class A country block: 0 (default) or the cross-country average minus c7?
- OQ2.5 Rebuild the frozen late-2024 x1 for B from (x86 − x86_c7)? Default: leave as stored and flag it.
- OQ2.6 Include ts_mom? Default: yes (pre-registered). If the student prefers purely cross-sectional characteristics, drop it before any fitting.

---

## 4. Part 3: Models × evaluation schemes

### (i) Goal
Measure $R^2_{OOS}$ for a menu of lecture models under static, expanding and rolling schemes, on the same OOS rows and against the same benchmarks (L8 p.53). Test whether nonlinearity and characteristic × macro interactions add. Test whether macro adds, and whether the placebo does as well. Compare per-class with pooled models. Identify which predictors carry the signal, and how stable that is.

### 4.1 Benchmarks first (cell 36 rule 1), computed before any model cell
- **Pooled trailing mean (PRIMARY).** $b^{pool}_t=\frac{\sum_{s=2003\text{-}01}^{t-1}\sum_j y_{j,s}}{\sum_{s} N_s}$. One number per month, the same for every asset, updated monthly. This is the HW5 1.1 "mean of the data seen so far", applied to the pooled vol-scaled target the pooled model forecasts. Code: `(frame.groupby('date').y.sum().cumsum() / frame.groupby('date').y.count().cumsum()).shift(1)`.
- **Zero:** $b=0$ (L4 p.46; L5 p.56-57).
- **Secondary benchmarks, reported alongside the primary and never substituted for it (L5 p.57):**
  - per-asset trailing mean $b^{asset}_{i,t}$ = mean of $y_{i,s}$ for s ≤ t−1;
  - per-class trailing mean $b^{class}_{c,t}$.
- **Why pooled is primary.**
  - (1) It is the mean of the data seen so far for the object being forecast.
  - (2) On a vol-scaled target the pooled mean is an average Sharpe-type level, so pooling is meaningful. Pooling raw returns would not be.
  - (3) The pooled model's intercept estimates exactly this quantity, so $R^2_{OOS}$ against it measures what the predictors add.
  - (4) A per-asset mean from 10 years of data is very noisy (the SE of a monthly mean of y is about 0.09). It could lose to zero and flatter the models.
  - The per-class column shows how much of any gain is just class intercepts.
- Portfolio benchmarks EW, RP and TSMOM (§5) are computed and tabulated on 2013-2024 at the same point.
- Output: Table R1, with $R^2_{OOS}$ of each trailing-mean variant against zero, and the OOS Sharpe of EW/RP/TSMOM gross and net.

### 4.2 Schemes (all refits in January of Y ∈ 2013..2024; test = the 12 months of Y, 600 rows)

| Scheme | Training window $T_Y$ | Inner-train (tuning) | Validation | Rows (train / inner / val) |
|---|---|---|---|---|
| Static | fixed at 2003-01..2012-12. Fitted once and used for all 144 test months. | 2003-01..2009-12 | 2010-01..2012-12 | 6,000 / 4,200 / 1,800 |
| Expanding (fixed start, L5 p.50) | 2003-01..(Y−1)-12 | 2003-01..(Y−4)-12 | (Y−3)-01..(Y−1)-12 | 600·(Y−2003) / 600·(Y−2006) / 1,800 |
| Rolling (120 months) | (Y−10)-01..(Y−1)-12 | the first 84 months | the last 36 | 6,000 / 4,200 / 1,800 |

- In 2013 all three schemes produce the same model. They diverge after that, which gives a clean scheme comparison.
- Features (σ̂, cross-sectional z, rolling macro z) are computed monthly for every scheme. Only *model parameters* are frozen in the static scheme.
- No embargo is needed. Targets are single-month returns, so neither validation nor test targets overlap any training target. Overlapping features such as x2 contain only past returns.

### 4.3 Model menu, hyper-parameter grids and why each is on the menu

| Model | sklearn call (all `random_state=SEED` where applicable) | Tuned on validation (grid) | Feature sets | Why / grounding |
|---|---|---|---|---|
| OLS | `Pipeline([('sc',StandardScaler()),('m',LinearRegression())])` | – | F0, F1, F2 | Linear baseline (L3 p.19-26). Expected to overfit F2 (L4 p.47-49). |
| Ridge | `Pipeline([...,('m',Ridge(alpha=a))])` | `a ∈ 10**np.linspace(-1, 6, 50)` | F0, F1, F2 | Many weak, correlated predictors (L5 p.60, p.12-16). Standardisation needed (L5 p.23). |
| Lasso | scaler fitted on inner-train, `lasso_path(Xs, y−ȳ, alphas=grid)` for tuning (L5 p.52 idiom), final `Pipeline([...,('m',Lasso(alpha=a*, max_iter=10_000))])` | `a ∈ 10**np.linspace(-5, -0.5, 50)` (sklearn α = λ/2n) | F0, F1, F2 | Sparse alternative, with selection frequency across refits (L5 p.17-24) |
| KNN (regression) | `Pipeline([...,('m',KNeighborsRegressor(n_neighbors=K, weights='uniform', n_jobs=N_JOBS))])` | `K ∈ {50,100,200,400,800,1600}` | F0k (6), F1k (10) | Nonparametric local average (L6 p.21-27). Scale matters (L6 p.23). No dummies and no 7-dimensional country block because of the curse of dimensionality and categoricals (L6 p.37). Large K because the signal-to-noise ratio is low (L6 p.25). |
| CART | `DecisionTreeRegressor(max_leaf_nodes=L, min_samples_leaf=0.01)` | `L ∈ {2,4,8,16,32,64}` | F0, F1 | One readable tree. Size chosen out of sample (L8 p.19-22, p.29-33). Coarse geometric grid (L8 p.39). |
| **Random forest** | `RandomForestRegressor(n_estimators=300, max_features=1/3, min_samples_leaf=f, bootstrap=True, n_jobs=N_JOBS)` | `f ∈ {0.005, 0.02, 0.08}` (fractions: sklearn uses `ceil(f·n)`, about 21-864 rows) | F0, F1 (+ variants) | Deep trees plus averaging; decorrelated by p/3 columns per split (L8 p.41-47, p.43). B is a budget, not a tuning parameter (L8 p.45). Only the leaf size is tuned (L8 p.44). The fractional leaf keeps the grid valid for per-class fits. |
| **Gradient boosting** | `GradientBoostingRegressor(loss='squared_error', learning_rate=0.05, n_estimators=300, max_depth=d, subsample=0.8, min_samples_leaf=0.01)`. **No** `n_iter_no_change`. | `d ∈ {1,2,3}`. B* = argmin of the validation MSE along `staged_predict` over 1..300. Choose (d, B*) jointly. Refit on $T_Y$ with `n_estimators=B*`. | F0, F1 | Shallow trees, shrinkage ν, B chosen on the held-out curve (L8 p.50-52). Stochastic subsample 0.8 as in L8 p.52. |

**Grid-edge rule** (L5 p.27, p.53). Every chosen value is logged per refit.
- If the chosen value is at a grid edge in more than 4 of 12 refits, extend the grid one step in that direction in a disclosed rerun, which adds +1 to the search count.
- For GBRT, if B* ≥ 290 then re-tune that refit with `n_estimators=600`.

### 4.4 Tuning protocol (inside each training window $T_Y$, strictly time-ordered)
1. Split $T_Y$ by **date** into inner-train and validation (the last 36 months). Assert `inner.date.max() < val.date.min()` and `val.date.max() < test.date.min()`.
2. For each grid value: fit the whole Pipeline (scaler included, and PCA in E2) on **inner-train only**, then compute the validation MSE of y. MSE is the Gaussian deviance over n (L5 p.4-5, p.52).
3. Choose the argmin. Never look at test rows. Test predictions have "nothing to do with obtaining optimal α" (L5 p.52, p.55; L4 p.43; L8 p.55).
4. **Refit the chosen configuration on the whole $T_Y$** (inner-train + validation) (L5 p.36 step 4). This uses the most recent 3 years. L5 p.52's code does not refit, and that alternative is noted. For GBRT, B* is kept on refit (conservative).
5. Predict the 12 test months of Y (600 rows). Store: date, asset, class, y, ŷ, model, feature set, scheme, refit year, chosen hyper-parameters, IS $R^2$ on $T_Y$ (against the $T_Y$ mean).
6. `FAST=True`: re-tune only in 2013, 2016, 2019 and 2022, and refit only in the other years with the carried-over hyper-parameters. This stays time-ordered.

### 4.5 Evaluation metrics
- **$R^2_{OOS}$** (L4 p.44-46; L5 p.55-57), computed by hand:
  ```python
  def r2_oos(y, yhat, bench):   # sums over all OOS asset-months (or a subset: class, sub-period)
      return 1 - np.sum((y - yhat)**2) / np.sum((y - bench)**2)
  ```
  reported against $b^{pool}$ (primary) and 0, plus $b^{class}$ and $b^{asset}$ columns.
- **Significance.**
  - Monthly loss differential: $d_t=\sum_i[(y_{i,t}-b_{i,t})^2-(y_{i,t}-\hat y_{i,t})^2]$, so $R^2_{OOS}=\sum_t d_t/\sum_{t,i}(y-b)^2$.
  - t-stat = mean(d)/(sd(d, ddof=1)/√144), compared with $t_{143}$ (CLT and the t-test of a mean: L2 p.63-70, p.91-93). Critical values: 1.977 for a single pre-registered test, and the Bonferroni value **3.47 for m = 72 specifications**, $t_{143}^{-1}(1-0.05/144)$ (L4 p.27-28).
  - Plus a percentile CI from resampling the 144 months with replacement (all assets of a month together), B = 2,000, SEED (L2 p.78-80). This relaxes normality but assumes independence across months (L2 p.80 caveat). Monthly return AR(1) is about 0.03 (DP §C).
- **Cuts:** by class (A/B/C/D), by sub-period (2013-18, 2019-24), and excluding 2020-03..05 as a sensitivity check (the value is reported either way).
- **IS vs OOS:** the mean IS $R^2$ over refits sits next to $R^2_{OOS}$ (AIG §6.4; L4 p.47-48). **Leak alarm:** any $R^2_{OOS} > 3\%$ triggers a leak hunt before interpretation (L5 p.57).
- **Fit plot:** OOS deciles of ŷ (pooled), mean realised y against mean ŷ, with a 45° line (L3 p.58; L1 p.35).
- **Cumulative loss-differential plot** $\sum_{s\le t}d_s$ for the primary models, showing *when* any gain is earned. This is a presentation device and needs no new statistic.

### 4.6 Model × scheme table (the main Results table, R2)
- Rows: OLS / Ridge / Lasso / KNN / CART / RF / GBRT × F0 / F1 (plus F2 for the three linear models) = 17 rows.
- Columns: static / expanding / rolling × {vs $b^{pool}$, vs 0}, plus IS $R^2$ (expanding).
- A t-stat is shown under each primary cell.
- "Which scheme wins and why" is discussed through bias and variance:
  - expanding uses more data, so lower variance;
  - rolling adapts to nonstationarity (L1 p.34) at the cost of variance (L5 p.7-8, p.56; L8 p.40);
  - static is a check of whether refitting helps at all.

### 4.7 Does macro add? (Q2)
- Refit designs on the same OOS rows: F0 / F0+G / F0+C / F1 for **RF and Ridge (expanding)**. From the main grid, F0 against F1 for every model and scheme. Report ΔR² with the loss-differential t-stat of F1 against F0 (the same $d_t$ construction with F0 as the "benchmark").
- **Placebo:** RF-F1 and Ridge-F1 (expanding) with macro shifted by k ∈ {36, 60, 96, 150}, giving 8 fits. Table R6 lists $R^2_{OOS}$ for real, for each k, and ΔR² (real − placebo_k). Fig R3 is a dot plot.
- Pre-registered reading: real macro counts only if it beats **all** four shifts.
- **Block-permutation importance** (§4.9) of the global and country blocks in the RF-F1 test years gives a decomposition without refitting.
- In-sample, a nested-model F-test (`anova_lm(small, big)`, L4 p.11-15) of Ridge/OLS F0 against F1 on the *first training window only* may be shown for contrast. **The verdict is OOS** (L4 p.28: "stop asking whether it fits").
- Interpretation caveat: for class B, x1 and x4 already *are* rate differentials (DP §D), so "macro beyond characteristics" is partly already absorbed into the characteristics (confounding, L3 p.57).

### 4.8 Per-class against pooled
- Fit RF and Ridge separately for each class on that class's rows (F0 and F1, expanding, tuned the same way on the class's validation rows). Class A has no country block, so its F1 equals F0+G.
- Compare with the pooled model on the **same class rows** (same benchmark $b^{pool}$; also $b^{class}$). Table R7: class × {pooled, per-class} × {RF, Ridge}.
- Reading:
  - pooling raises n and $s_x$ and cuts estimation variance (L2 p.73);
  - per-class allows separate slopes (L3 p.54: "why not separately estimate three models?");
  - B/D validation blocks are small (252-288 rows), so tuning is noisy (L6 p.37; L8 p.40).
- The pooled RF can itself split on class dummies, so it is an adaptive compromise between the two.

### 4.9 Which predictors carry the signal, and is that stable?
- **OOS permutation importance** (L8 p.56-58), for RF-F1, GBRT-F1 and Ridge-F1 (expanding):
  - For each refit year Y: `permutation_importance(model_Y, X_test_Y, y_test_Y, scoring='neg_mean_squared_error', n_repeats=10, random_state=SEED, n_jobs=1)`. The result is the increase in test MSE when a column is shuffled (signature checked: `n_repeats` defaults to 5, so set it to 10).
  - Convert to $R^2_{OOS}$ units: $\Delta R^2_j=\sum_Y 600\cdot\Delta MSE_{j,Y}/SSE_{bench}$.
  - Fig R4a: aggregate bar chart. Fig R4b: feature × year heat map (stability). Stability statistics: Spearman rank correlation of importances between consecutive years, and the number of years with importance > 0.
  - **Block version** [FLAG: minor extension of L8 p.57]: apply one permutation to all columns of a block at once (cs-characteristics, ts_mom, class dummies, global, country). This avoids splitting credit among correlated columns (L8 p.58) and keeps the one-hot class block consistent.
  - MDI `feature_importances_` is shown **only** as an in-sample contrast, labelled as such (L8 p.47, p.56).
- **Partial dependence** (L8 p.59-60):
  - For RF-F1 at the 2013, 2018 and 2024 refits (stability), and GBRT-F1 at 2024.
  - `partial_dependence(model, X_sub, features=[f], grid_resolution=40, percentiles=(0.05,0.95), method='brute', kind='average')` on a seeded 3,000-row subsample of that refit's training rows (the fitting sample, as in L8 p.60). Signature and return keys (`grid_values`, `average`) were checked.
  - Features: cs_x1..cs_x5, ts_mom, g_x6..g_x9 (10 panels).
  - 2-D PD (20×20 grid) for (cs_x2, g_x6) and (ts_mom, g_x6) at 2024. This is the direct picture of a characteristic × macro interaction. PD reports "what the model says, not a causal effect" (L8 p.59). Importance has no sign, and PD supplies it (L8 p.60).
- **Coefficient paths:** standardised Ridge-F1 and Lasso-F1 coefficients against refit year (12 × 20), plus the Lasso selection frequency (L5 p.22, p.60). Read with the multicollinearity caveat (L3 p.39-44; L4 p.19-21).
- **Hyper-parameter paths:** chosen α / K / leaves / leaf-fraction / (d, B*) per year (Table A2), with edge flags.

### (iv) Assumptions (Part 3)
- A3.1 OOS window 2013-2024 and annual refit (see §0.2).
- A3.2 36-month validation; refit on train + validation.
- A3.3 Pooled trailing mean is the primary benchmark.
- A3.4 RF is the primary nonlinear model. It is "the default answer" and nearly tuning-free (L8 p.55, p.61). GBRT is secondary because it needs a validation budget (L8 p.55).
- A3.5 ν = 0.05 and B ≤ 300 for boosting (compute; L8 p.52: halving ν roughly doubles the rounds needed).
- A3.6 SEED 7034 for all randomness. A 2-seed replicate of RF-F1 is disclosed (not counted as search).

### (v) Pitfall guards
- All tuning uses date masks with assertions. No shuffled or random folds anywhere (L5 p.48, p.56, p.58 #3; L8 p.45).
- Scalers and PCA live inside Pipelines fitted on inner-train or $T_Y$ rows only (L5 p.47, p.58 #1).
- GBRT early stopping is hand-rolled on the time-ordered validation block (sklearn's internal version shuffles; see §1.3).
- One OOS row set, one benchmark and one $R^2$ function for every model (L8 p.53).
- Grid-edge logging.
- IS > OOS check.
- Leak alarm above 3%.
- Permutation importance and PD are computed *after* all forecasts are frozen, and are used only for interpretation, never for selection.

### (vi) Outputs
- Table R1: benchmarks.
- Table R2: model × scheme $R^2_{OOS}$ (F0, F1, F2).
- Table R3: by class, for RF / GBRT / Ridge / KNN with F0 and F1, expanding.
- Table R4: IS against OOS.
- Table R5: macro decomposition (F0, F0+G, F0+C, F1).
- Table R6: placebo.
- Table R7: per-class against pooled.
- Figs R1-R7: cumulative $d_t$, the decile fit plot, the placebo dot plot, permutation importance (aggregate and heat map), PD panels, 2-D PD, coefficient paths.
- Table A1: all 72 specifications.
- Table A2: hyper-parameter paths.

### (vii) Write-up checklist (cell 37 step 3)
- ☐ $R^2_{OOS}$ against the trailing mean **and** zero, model × scheme
- ☐ by class for the models that matter
- ☐ which scheme wins and why
- ☐ which predictors carry the signal and how stable that is
- ☐ whether macro adds once refitted without it
- ☐ the placebo with several shifts
- ☐ per-class against pooled
- ☐ nonlinearity and interactions (RF/GBRT against Ridge F1/F2; 2-D PD)
- ☐ how penalties and hyper-parameters were chosen (training data only)
- ☐ the search size next to the headline

### (viii) Open questions
- OQ3.1 OOS start 2013-01 (default) or 2011-01 (168 months, but only 96 training months)?
- OQ3.2 Primary benchmark: pooled trailing mean (default) or per-asset?
- OQ3.3 Should GBRT include depth 3? This is a compute question. Default: yes. FAST mode uses {1, 2}.
- OQ3.4 Refit on train + validation after tuning (default), or use the inner-train fit as in L5 p.52?

---

## 5. Part 4: From forecasts to portfolios

### (i) Goal
Build the three benchmark rules exactly as cell 36 specifies, plus forecast-based rules. Evaluate them all on 2013-01..2024-12, gross and net of costs. Attribute any outperformance to benchmark exposure versus timing.

### (ii) Construction. All weights use information through t−1, unit gross exposure $\sum_i|w_{i,t}|=1$, monthly rebalancing; $R_{p,t}=\sum_i w_{i,t}r_{i,t}$.

| Rule | Raw weight $\tilde w_{i,t}$, then $w=\tilde w/\sum|\tilde w|$ | Source |
|---|---|---|
| EW | 1 | [EXAM] cell 36 |
| RP | $1/\hat\sigma_{i,t-1}$ (36-month sd, ddof=1, from `excess_return`) | [EXAM] cell 36; ddof per L2 p.56-57 |
| TSMOM | $\mathrm{sign}(R12_{i,t-1})/\hat\sigma_{i,t-1}$, with $R12_{i,t-1}=\prod_{s=t-12}^{t-1}(1+r_{i,s})-1$ from `excess_return` (= x2); sign(0) = 0 | [EXAM] cell 36 |
| **P1 (primary)** | $\hat y_{i,t}/\hat\sigma_{i,t-1}$ | cell 37 step 4 ("proportional to the forecast scaled by volatility"); derived in §3.10 |
| P2 (secondary) | $\mathrm{sign}(\hat y_{i,t})/\hat\sigma_{i,t-1}$. Same form as TSMOM with the model's sign in place of the 12-month sign, so it isolates the signal. | cell 37 step 4 ("the sign of the forecast") |
| CT (diagnostic) | $b^{class}_{c(i),t}/\hat\sigma_{i,t-1}$ (the class trailing mean of y). The static class-premium tilt, built from returns only. | [FLAG: our diagnostic], used only for attribution |

Which forecasts are turned into portfolios:
- **Primary:** RF-F1-expanding, rule P1.
- **Secondary table (disclosed):** P1 and P2 for RF-F0, GBRT-F1, Ridge-F0 and Ridge-F1 (expanding), plus P2 for RF-F1. That makes 10 forecast-portfolio variants.

### (iii) Performance measures. These are [EXAM] items (cell 37 step 4); none is defined in a lecture, so each is defined here.
- Annualised mean = 12·mean($R_p$). Annualised vol = √12·sd($R_p$, ddof=1). **Sharpe** = ann. mean / ann. vol. Returns are already excess, so nothing is subtracted.
- Worst month.
- **Max drawdown** on $W_t=\prod(1+R_{p,s})$: $\max_t(1-W_t/\max_{s\le t}W_s)$.
- Maximum single-asset |w|, and the average net exposure $\sum w$. RP-type rules concentrate in asset_16 (σ̂ as low as 0.3%), which must be shown (DP §D).
- **Turnover** [FLAG]: $TO_t=\sum_i|w_{i,t}-w_{i,t-1}(1+r_{i,t-1})/(1+R_{p,t-1})|$ (drift-adjusted, traded notional per unit of capital). It is counted from 2013-02, so the initial build is excluded (stated).
- **Costs** [FLAG, transparent assumption]: $R^{net}_t=R_t-c\cdot TO_t$ with c ∈ {0, 2, 5, 10, 25, 50} bp per unit traded. The primary is c = 10 bp, a round number between typical FX/bond and commodity futures costs. It is an assumption, not an estimate.
- **Break-even cost** $c^*$: the c at which net Sharpe(P1) = net Sharpe(benchmark) (both pay costs), solved on a 0-200 bp grid with a 0.25 bp step. Also $c_0=\overline{R}/\overline{TO}$, where the net mean is 0.
- **Uncertainty:** a paired bootstrap over months (B = 5,000, SEED) for Sharpe and ΔSharpe against each benchmark, with 95% percentile CIs (L2 p.78-80; iid-months caveat).

### (iv) Attribution: exposure against timing (L2 p.47-49, p.94-97; L3 p.19-36)
- `smf.ols('P1 ~ EW + RP + TSMOM', data=oos).fit()` plus the three single-benchmark regressions.
  - Report α (monthly, and ×12 in % per year), its t-stat and p-value from $t_{n-p}$ (L2 p.91-95), the β's and $R^2$.
  - The share of P1's variance explained by benchmark returns ($R^2$) is the "exposure" part. α is return not explained by the benchmark rules, the "timing/selection" part.
  - Test β = 1 against a single benchmark by hand (L2 p.97) to see whether P1 is just a levered benchmark.
  - EW and RP are highly correlated (both long-only), so individual β's are unstable while α remains interpretable (L3 p.39-44; L4 p.19-21).
- Add a regression on **CT** to measure the static class-tilt component. If $\alpha$ vanishes once CT is included, the "outperformance" is class risk premia, not characteristics.
- Nonrobust OLS SEs (house style), with the month bootstrap of α as a check (L2 p.80). HAC is out of scope.
- [FLAG, optional, only if approved] Ex-post average-weight decomposition, $R^{static}_t=\sum_i\bar w_i r_{i,t}$ with $\bar w_i$ the OOS average weight and timing = $R_{P1}-R^{static}$. It is descriptive only, because $\bar w$ uses the whole OOS period.

### (v) Pitfall guards
- Weights at t use only σ̂, R12 and ŷ dated t−1 or earlier. Assertion: the weight frame's date index equals the return date, and the inputs are the lagged columns.
- The same 144 months for every strategy.
- Costs are applied to all strategies, benchmarks included.
- The primary cost level and the rule are fixed now and not moved after results are seen (L6 p.47: fix the threshold ex ante; L5 p.57: do not switch benchmarks).
- Plots scaled to 10% vol are labelled "ex-post scaled for display only".

### (vi) Outputs
- Table R8: gross and net performance for EW, RP, TSMOM, P1, P2 and CT (mean, vol, Sharpe gross, Sharpe net at 10 bp, MDD, worst month, average TO, max |w|, net exposure, $c^*$, $c_0$).
- Table R9: net Sharpe over the cost grid.
- Table R10: sub-periods 2013-16, 2017-20, 2021-24 (Sharpe).
- Table R11: attribution regressions.
- Table R12: bootstrap CIs of ΔSharpe.
- Fig R8: cumulative log wealth, gross and net at 10 bp.
- Fig R9: drawdowns.
- Fig R10: rolling 36-month Sharpe of P1 minus TSMOM.
- Appendix: the secondary portfolio variants.

### (vii) Write-up checklist (cell 37 step 4)
- ☐ how the portfolio is formed and rebalanced
- ☐ Sharpe, at a minimum, plus return, vol, drawdown, turnover, net of costs, cumulative plots, sub-periods
- ☐ comparison with each benchmark
- ☐ how much of any outperformance is benchmark exposure and how much is timing (α, $R^2$, CT)
- ☐ that costs and turnover are an assumption (lectures do not cover them)

### (viii) Open questions
- OQ4.1 Primary cost 10 bp uniform (default) or class-specific (for example A 10 / B 3 / C 5 / D 3 bp)?
- OQ4.2 Floor σ̂ in the weights (for example at 0.5% per month) to limit asset_16's concentration? Default: no floor, report max |w|.
- OQ4.3 Add a within-class top-minus-bottom-tercile rule (cell 37 mentions sorts)? Default: no, to keep the forking paths down.
- OQ4.4 Approve the ex-post average-weight decomposition [FLAG]? Default: off.

---

## 6. Part 5: Write-up structure (cell 38)

| Section | Content | Figures and tables |
|---|---|---|
| **Introduction** | The question (cell 33). Why it matters to a cross-country, cross-asset allocator: characteristics such as carry, momentum and value are traded systematically. What the literature leads you to expect: small but positive cross-sectional predictability, with low signal-to-noise (L1 p.33) and monthly OOS $R^2$ of a fraction of a percent (L5 p.57). ML portfolios in GKX (L1 p.36), but in asset pricing "the linear model can win" (L8 p.62). A one-paragraph preview with the headline numbers and search size. | – |
| **Data** | Panel contents, classes and countries. The identity of x1..x5 with evidence. Treatment of lags (incl. x10 look-ahead), missing values (back-fills, the x11 rebuild), standardisation, the macro mapping, extended-file triage. Summary statistics and structure plots. | D1-D9, F1-F4 |
| **Methodology** | Benchmarks and why (rule 1). Target and σ̂. The model menu and grids. Schemes, dates and the OOS window with reasons. Time-ordered tuning. The $R^2_{OOS}$ definition and tests. Placebo design. Forecast → position rules. Performance measures and the cost model. The pre-registration and the search size. A reader should be able to reproduce everything from this section alone (cell 38). | M1 (dates), M2 (models/grids), M3 (spec inventory) |
| **Results** | In the order of the questions: (1) forecastability, R1-R4, R2 as the headline; (2) the contribution of macro, R5-R7, placebo, importance, PD; (3) portfolios, R8-R12. Negative numbers are reported as results (L5 p.57). | R1-R12, Figs R1-R10 |
| **Interpretation and discussion** | Why the numbers come out as they do:<br>• x2 and x5 carry the same information as TSMOM and RP;<br>• x1 and x4 already contain country macro for B;<br>• blocks and effective N;<br>• nonstationarity and scheme choice;<br>• what "nonlinearity" bought, from PD shapes.<br>Where the evidence is strong and where it is fragile (sub-periods, 2020, class dependence, Bonferroni). What was tried and did not work (Table A1). What to do next: longer histories, point-in-time macro vintages, class-specific costs, own-country levels. | – |
| **Conclusion** | A half-page summary a PM can read alone: forecastability, whether macro helps, whether the portfolio beats the rules net of costs, and how much of it is just exposure. | – |
| **Appendix** | A1 all specifications. A2 hyper-parameter paths. A3 leakage audit (§7). A4 runtime log. Secondary benchmarks and portfolios. | – |

---

## 7. Part 6: Leakage audit checklist (lives in an appendix cell with executable asserts)

| # | Risk | Where the design prevents it | Executable check |
|---|---|---|---|
| 1 | **FATAL:** a transform fitted before the split (scaler, imputer, PCA, winsorisation) (L5 p.58 #1; AIG §4b) | StandardScaler and PCA sit inside Pipelines fitted on inner-train or $T_Y$ rows only (§4.4, §3.9). Cross-sectional z uses same-month data (§3.3). Macro z and σ̂ are trailing (§3.5, §3.10). The clip is a fixed constant. No imputation: back-fills become NaN and fall outside the sample; the x11 rebuild is a same-month identity. | `assert pipe.named_steps['sc'].n_samples_seen_ == len(train_rows)`; grep the notebook for `fit_transform(` outside Pipelines (should find none) |
| 2 | Merge on period end rather than publication date (L5 p.58 #2) | Lag table §3.1: characteristics +1, macro +2. x10/x73 dropped, annual/quarterly columns excluded. Key merges only. | x2 = R12 and x5 = σ̂ identity checks; a spot check that `d_sr` at 2022-03 equals x86(2022-01) − x86_c7(2022-01) |
| 3 | **FATAL:** random fold on time-ordered data (L5 p.58 #3; AIG §4a; L8 p.45) | Date-mask splits only. Validation = the last 36 months of each window. No `train_test_split`, KFold, `RidgeCV()`, `LassoCV(cv=int)`, GBR `n_iter_no_change` or HistGB. | `assert inner.date.max() < val.date.min() <= val.date.max() < test.date.min()` at every refit; a notebook-wide search for banned names |
| 4 | Hyper-parameter or threshold chosen on the test block (L5 p.58 #4; L8 p.55) | Selection only on validation MSE. Primary spec, cost level and portfolio rule pre-registered (§0.3). Importance and PD computed after forecasts are frozen. | The tuning function receives no test arrays (its signature enforces this); Table A2 logs the choices |
| 5 | Benchmark quietly changed (L5 p.58 #5, p.57) | Pooled trailing mean and zero fixed in §0.3. Per-asset and per-class shown alongside no matter how they come out. | The benchmark is computed once in the "Benchmarks first" cell and reused by name |
| 6 | Test-set-mean $R^2$ (AIG §4d) | Hand-written `r2_oos`. | Search for `r2_score` and `.score(` (should find none) |
| 7 | Mislabelled "plain" logistic regression (AIG §4c) | Not used. If ever used, `penalty=None`. | – |
| 8 | Look-ahead from low-frequency stamping (DP §F-G) | x10/x73 dropped. Annual/quarterly extended columns excluded. | Assert those columns are absent from every X |
| 9 | Back-filled characteristics (DP §D) | Set to NaN, and the sample starts 2003-01. | Assert no NaN in X, and that every filled cell is dated < 2003-01 after lagging |
| 10 | Stale x11 during the regime change | Rebuild instead of forward fill. | Rebuild RMSE table |
| 11 | Snooping through EDA sorts | Sorts on 2003-2012 only; no feature selection from them. | `assert sorts.date.max() <= '2012-12-31'` |
| 12 | Forking paths | Pre-registration cell; all 72 specifications reported (A1); Bonferroni 3.47 shown. | Spec counter printed at the end |
| 13 | pandas-3 silent errors | Key merges with `validate=`; sort before groupby-shift; `.copy()` of `.corr().values` | Assertions in §1.3 |
| 14 | Portfolio look-ahead | Weights use only t−1 inputs. The same months for all strategies. | Assert the weight inputs are the lagged columns. Spot check: a deliberately wrong TSMOM that uses R12 through t (look-ahead) must give different weights from the production code, which proves the code path uses t−1 |
| 15 | "Too good to be true" | Leak alarm at $R^2_{OOS}>3\%$ (L5 p.57). Headline verification: does the result survive dropping 2020, each class in turn, and an extra lag of all predictors (diagnostic spec E4)? | Printed diagnostic table |

---

## 8. Part 7: Compute budget

**Evidence.** A timing run on **synthetic** data with 12,000 rows × 19 features (the largest pooled training window), on this 4-core machine:

| Fit | Time |
|---|---|
| RF, 300 trees, max_features 1/3: min_samples_leaf 50 / 200 / 800 | 3.3 / 2.1 / 1.2 s (n_jobs=4); 12.3 / 7.9 / 4.4 s (n_jobs=1) |
| GBR, 600 rounds, subsample 0.8, depth 1 / 2 / 3 | 11.7 / 22.6 / 32.7 s (single-threaded) |
| Ridge, 50 α | 0.35 s |
| Lasso, 50 α | 0.31 s |
| KNN fit + predict | 0.33 s |
| Permutation importance, 600 rows × 19 features × 10 repeats | 18 s |
| 1-D brute PD, 3,000 rows × 40 grid points | 3.8 s |
| 2-D PD, 20 × 20 | 36 s |

Scaling assumptions:
- Expanding windows average about 62% (inner) and 78% (final) of the largest window.
- Rolling windows are about 35% and 50%.
- GBRT uses 300 rounds, so about half of the times measured above for 600 rounds.

| Block | Specs × refits | Estimated time (4 cores) |
|---|---|---|
| OLS / Ridge / Lasso / KNN / CART, full grid (33 specs) + placebo, per-class and robustness for Ridge | ~50 × 25 | ~5 min |
| RF: F0, F1 × 3 schemes | 6 | ~3.5 min |
| RF: F0+G, F0+C, 4 placebo shifts, per-class F0/F1, E1 rank, E2 extPCA, E3 xsdev, E4 extra lag, 2 seed replicates | 14 (expanding) | ~14 min |
| GBRT: F0, F1 × 3 schemes (depth {1,2,3}, ν = 0.05, B ≤ 300) | 6 | ~12 min |
| Permutation importance (RF, GBRT, Ridge; single and block), PD (1-D × 3 refits, two 2-D) | – | ~8 min |
| EDA, benchmarks, portfolios, bootstraps (vectorised over monthly SSE sums) | – | < 1 min |
| **Total (FULL)** | | **about 40-45 min** |
| **FAST=True** (re-tune every 3 years; GBRT depth {1,2}; PD at the 2024 refit only) | | **about 18-20 min** |

Decisions this supports:
- **Annual refits.** Monthly refits would multiply the model time by 12.
- Small geometric grids (L8 p.39).
- RF with B = 300 as a budget (L8 p.45: flat after about 50 trees).
- GBRT with a hand-rolled early-stopping curve from a single fit per depth (`staged_predict` costs about 0.02 s).
- Placebos and robustness specs run only for RF and Ridge under the expanding scheme.
- `N_JOBS = min(4, cpu_count)`.
- The runtime is logged for every block (Table A4).
- Optional: `CACHE_DIR = None` by default. If set, predictions are saved as parquet so the write-up cells can be re-run, but the notebook still runs top to bottom from scratch (cell 38).

---

## 9. Part 8: Specification search and pre-registration (L4 p.28; L5 p.57)

**Pre-registration cell.** A markdown cell placed *before* the first model fit. It contains:
- the primary spec (§0.3);
- the decision rules (§0.1);
- the expectations (commit before compute; L2 p.78, L4 p.26);
- the full inventory below.

A `CONFIG` dict with every constant is printed beside it. A result can never be promoted to "headline" after the fact. The headline is always the primary spec, even if a secondary spec does better (L5 p.57: "do not quietly switch").

**Inventory of OOS-evaluated forecasting specifications** (disclosed next to the headline):

| Group | Specs | Count |
|---|---|---|
| Main grid: OLS / Ridge / Lasso × F0/F1/F2 × 3 schemes | 27 | 27 |
| KNN × F0k/F1k × 3; CART × F0/F1 × 3; RF × F0/F1 × 3; GBRT × F0/F1 × 3 | 6 + 6 + 6 + 6 | 24 |
| Macro decomposition: RF, Ridge × {F0+G, F0+C}, expanding | 4 | 4 |
| Placebo: RF, Ridge × 4 shifts, expanding | 8 | 8 |
| Per-class: RF, Ridge × {F0, F1}, expanding (each is a set of 4 class models) | 4 | 4 |
| Robustness: E1 rank standardisation (RF, Ridge, F1); E2 extPCA (RF); E3 xsdev (RF); E4 extra lag (RF-F1, a leak diagnostic) | 5 | 5 |
| **Total** | | **72**. Bonferroni with m = 72 gives a $t_{143}$ critical value of 3.47. If the student adds or drops a spec before fitting, recompute it with `stats.t.ppf(1-0.05/(2*m), 143)`. |

Not counted as search, but disclosed: 2 RF seed replicates, and the portfolio variants (10 forecast portfolios + CT + 3 benchmarks) × 6 cost levels. The α tests for the portfolios use Bonferroni with m = 10, giving 2.85.

**Guards against the garden of forking paths:**
- (a) Every specification above is run and reported in A1, whatever the result.
- (b) No feature is chosen from sorts or test results.
- (c) Grid extensions happen only under the pre-stated edge rule and are counted.
- (d) Anything added after results are seen is labelled "post-hoc", counted in the search, and excluded from the headline.
- (e) The search-adjusted critical value is printed next to every t-stat in R2, R5, R6 and R11 (L4 p.27-28).

---

## 10. Consolidated open questions for the student (each has a default)

| # | Question | Default |
|---|---|---|
| OQ1.1 | Full-sample or training-only correlation matrix in the Data section | both; design cites 2000-2012 |
| OQ2.1 | σ̂ window | 36 months, ddof=1 |
| OQ2.2 | Macro lag | uniform 2 months |
| OQ2.3 | Own-country levels for C/D | no |
| OQ2.4 | Class-A country block | 0 (≡ c7 differential) |
| OQ2.5 | Rebuild frozen late-2024 x1 (B) | no, flag it |
| OQ2.6 | Include ts_mom | yes |
| OQ3.1 | OOS start | 2013-01 |
| OQ3.2 | Primary trailing-mean benchmark | pooled mean of y |
| OQ3.3 | GBRT depth 3 | yes (FAST: no) |
| OQ3.4 | Refit on train + validation after tuning | yes |
| OQ4.1 | Cost level | 10 bp uniform, 0-50 bp grid |
| OQ4.2 | σ̂ floor in weights | none, report max \|w\| |
| OQ4.3 | Tercile long-short rule | no |
| OQ4.4 | Ex-post weight decomposition [FLAG] | off |
| OQ5 | SEED | 7034 (as in Problem 1). Problem 2 uses the student ID; confirm. |

---

## 11. Notebook cell map (implementation order under "# Your project starts here")

1. **P3-0 Config and imports.** Constants from §1.1. Imports: `Pipeline`, `StandardScaler`, `Ridge`, `Lasso`, `lasso_path`, `LinearRegression`, `KNeighborsRegressor`, `DecisionTreeRegressor`, `RandomForestRegressor`, `GradientBoostingRegressor`, `PCA`, `permutation_importance`, `partial_dependence`, `statsmodels.formula.api as smf`, `scipy.stats`.
2. **P3-1 Load and validate** (§2 step 1.1). Extended-file duplicate check and drop.
3. **P3-2 EDA** (§2, steps 1.2-1.10). Sorts restricted to 2003-2012.
4. **P3-3 Cleaning.** Back-fills → NaN. x11 rebuild and its quality table. x10 dropped. Extended-file triage table.
5. **P3-4 `build_features(macro_raw, shift_k=0, cs='z', country='c7diff', ext_pca=False)`** returns the 13,200-row frame with the target, σ̂, R12, all feature blocks, and NaN and lag assertions.
6. **P3-5 Benchmarks first.** $b^{pool}, b^{class}, b^{asset}$, zero. EW/RP/TSMOM weights, returns and OOS table (R1). Printed **before** any model.
7. **P3-6 Pre-registration** markdown cell and printed `CONFIG`.
8. **P3-7 Harness:** `windows(scheme)` gives date masks; `tune(model, Xi, yi, Xv, yv)` returns (params, val_curve) and cannot see test data; `fit_final`; `run_spec(model, fset, scheme)` returns the prediction frame (7,200 rows) and a log.
9. **P3-8 Run the inventory** (§9), with a runtime log and optional cache.
10. **P3-9 Evaluation:** `r2_oos`, loss-differential t-test, month bootstrap, tables R2-R4, figures R1-R2.
11. **P3-10 Macro and placebo** (R5, R6, Fig R3). **P3-11 Per-class against pooled** (R7).
12. **P3-12 Interpretation:** permutation importance (single and block), PD 1-D and 2-D, coefficient and hyper-parameter paths (Figs R4-R7, A2).
13. **P3-13 Portfolios:** weights, drift-adjusted turnover, costs, metrics, break-even, bootstrap ΔSharpe, attribution regressions, plots (R8-R12, Figs R8-R10).
14. **P3-14 Leakage audit asserts and runtime table** (§7, A4).
15. **Write-up** (markdown, §6).
