# Problem 3 research design, Draft C: evaluation discipline and portfolio evidence

**Scope.** This is a plan only. No model has been fitted and no exam quantity has been computed. The environment checks behind it were limited to these:
- library signatures and defaults in the installed stack: Python 3.11, pandas 3.0.6, sklearn 1.9.1, scipy 1.17.1, statsmodels 0.15.0, joblib 1.6.0; pyarrow is **not** installed;
- one timing run on *synthetic noise* with the exam's array shapes;
- Bonferroni critical values and an ex-ante power formula evaluated at hypothetical R² values.

Data facts come from the pre-modelling data profile (`notes/data_profile.md`). They must be re-derived in the notebook and printed there.

**Citation rules.**
- "L*n* p.*k*" means the lecture notes for Lecture *n*, PDF page *k*.
- "Exam cell *N*" means cell *N* of `Final_Autumn_2026-1.ipynb`: 33 is the question, 34 the data/traps, 36 the two rules, 37 the workflow, 38 what to submit.
- Anything the exam requires that no lecture teaches is labelled **[EXAM-DEFINED, not taught]**.
- Anything outside both sources is labelled **[OUTSIDE SCOPE — only if the user approves]**.

**Design philosophy.** The harness is built and reported before anything else:
- the pre-registered windows;
- the forecast benchmarks and the three portfolio benchmarks, computed and tabulated before any model is imported;
- the tests, the placebo machinery and the cost and attribution code.

The models are then deliberately simple and few. The model menu matters less than having "one history, one draw" (L8 p.40-41) evaluated honestly.

---

## 0. The pre-registration block (frozen before any fit)

The first code cell of Problem 3 is a `PREREG` dict. It is printed together with its `hashlib.sha256(json.dumps(PREREG, sort_keys=True, default=str).encode()).hexdigest()`. A `DEVIATIONS = []` list follows it, and every later change is appended there with a reason and a date.

Suggestion (student's call, see OQ-12): commit the notebook with this cell before the first out-of-sample (OOS) run. The commit then serves as a time stamp.

| key | frozen value | why (short) | grounding |
|---|---|---|---|
| `SEED` | 7034 | same seed as P1; used for RF, permutations and bootstrap | L4 p.50, L6 p.18 (set seeds) |
| `PRE_END` | 2009-12-31 | everything before the OOS window. Every *design* decision may use only PRE data | exam cell 36 rule 2; L4 p.43 |
| `FIRST_Y` | 2003-01-31 | first month with a 36-month σ̂ from returns, so the first target y and the first portfolio benchmark | exam cell 36 |
| `MODEL_START` | 2004-01-31 | first January after all warm-ups (see §3). Back-fills end 2002-08, σ̂ exists from 2003-01, lag-2 trailing global z has ≥36 obs from 2003-03 | data profile C/D/E |
| `INIT_TRAIN` | 2004-01 … 2009-12 (72 months, 3,600 asset-months) | contains the 2004-07 expansion **and** the 2008-09 crisis | L1 p.34 (nonstationarity) |
| `OOS` | 2010-01 … 2024-12 (180 months, 9,000 asset-months) | longest window leaving ≥6 years of training; power (§3.3); several regimes | L5 p.48-52, L4 p.43 |
| `SUBPERIODS` | 2010-14, 2015-19, 2020-24 | three equal 5-year blocks fixed ex ante | exam cell 37 ("sub-periods") |
| `SCHEMES` | static, expanding, rolling-72 | the notebook's three | exam cell 37; L5 p.48-50, p.56 |
| `REFIT` | annual, each January 2010…2024 (15 refits) | L5 p.51 uses step 12 | L5 p.51 |
| `INNER_CV` | 3 annual validation folds (years Y-3, Y-2, Y-1), each fit on all window data before that year; λ = argmin of the average fold MSE; then refit on the whole window | time-series CV | L5 p.48-49, p.36 (refit), p.35 (average MSE) |
| `RIDGE_GRID` | `10**np.linspace(-1, 7, 50)` | spans OLS-equivalent to null-equivalent | L5 p.27, p.37-39 |
| `LASSO_GRID` | `10**np.linspace(-5, 0, 50)` | same logic under sklearn's 1/(2n) scaling | L5 p.18 note, p.27 |
| `RF` | `n_estimators=300, min_samples_leaf=200, max_features=1/3`, not tuned | "set B large, set a min leaf size, stop" | L8 p.43-45, p.55 |
| `SIGMA` | σ̂_{i,t-1} = sd(r_{i,t-36..t-1}), ddof=1 | shared by the target and all portfolios | exam cell 36; L2 p.56-57 |
| `R12` | ∏(1+r_{i,t-12..t-1}) − 1 | TSMOM signal from excess_return | exam cell 36 |
| `MACRO_LAG` | 2 months for every monthly macro series (sensitivity: 1) | 1-month minimum plus 1 month of release delay | exam cell 34 |
| `GZ` | trailing z of lagged global macro, window 60 months, min_periods 36 | "trailing information only" | exam cell 37 |
| `BENCH_FCST` | zero, per-asset trailing mean, per-class trailing mean, pooled trailing mean (all through t-1, updated monthly) | notebook rule plus explicit panel versions | exam cell 36; L4 p.46; L5 p.55-57 |
| `BENCH_PORT` | EW, RP (1/σ̂), TSMOM (sign(R12)/σ̂), all with unit gross exposure and monthly rebalancing | exactly as specified | exam cell 36 |
| `PRIMARY` | Q1: Ridge, CHAR, pooled, expanding. Q2: Ridge CHAR+MACRO vs Ridge CHAR, pooled, expanding. Q3: rule P1 on Ridge-CHAR-pooled-expanding forecasts, net of 10 bp | one pre-named test per question | L5 p.57, L4 p.28 |
| `PLACEBO_K` | {36, 60, 96, 132, 168, 204}; full set 36, 48, …, 264 (20 shifts) for the primary Q2 model | more than one shift; all multiples of 12 keep calendar stamping | exam cell 37 |
| `COST_BP` | {0, 5, 10, 25, 50} bp per unit notional traded; headline at 10 bp; break-evens reported | not taught, so a transparent grid | exam cell 37 |
| `BOOT_B` | 10,000 iid resamples of months | SE by resampling | L2 p.78-80 |
| `LEAK_ALARM` | any R²_OOS > 2% (vs zero or vs trailing mean) ⇒ stop and audit before interpreting | monthly R²_OOS of 0.3-0.5% is typical | L5 p.57 |
| `N_JOBS` | `min(4, os.cpu_count())` | compute budget (§7) | — |

Timeline:

```
2000-01 ─ 2002-08  back-fill zone (x2/x3/x5 leading fills) → never enters any model
2003-01            first σ̂(36m) → first y; portfolio benchmarks start; trailing means start
2004-01 ─ 2009-12  INITIAL TRAINING BLOCK (72 m); inner validation years 2007, 2008, 2009
2010-01 ─ 2024-12  OOS WINDOW (180 m); refit every January; sub-periods 2010-14 | 2015-19 | 2020-24
```

---

## Part 1 — Know your data

### (i) Goal
Describe the panel and its traps before any modelling. Form a hypothesis for x1..x5, and take a first look at predictive content **without looking at the OOS window**.

### (ii) Method steps (notebook cells P3-B, P3-C, P3-D)

1. **Integrity checks** (T1).
   - Shapes: 15,000 × 10; 300 month-ends 2000-01-31…2024-12-31; 50 assets; class counts A 26 / B 8 / C 9 / D 7; all A assets are `country_7`; no NaN in the panel or the global file.
   - `asset_returns_wide` equals the panel exactly.
   - Merge the country files on the `(country, date)` keys, never by position. A lexicographic sort reorders `country_10` before `country_2`. Assert that the shapes are unchanged after every merge.
   - pandas 3 notes: strings load as `StringDtype`; `parse_dates` gives `datetime64[us]`.
2. **Trap audits** (T6, T7, T8, F6).
   - (a) Leading constant runs of x2/x3/x5 starting 2000-01. Rule: a run of identical values of length ≥2 starting at the first date. All but the last cell become NaN. Expect 7 assets, all ending by 2002-08.
   - (b) The x10 same-year-mean check. Share of country-years where x10 equals the mean of x86 in the same calendar year, compared with the prior year. Print 208/300 and corr 0.998 vs 0.839.
   - (c) The x11 stop pattern per country (c12 last 2020-10, c9 2021-08, c11 2024-03). Rebuild x11* = x86 − x12. Report overlap RMSE and corr per country, and the error a stale forward-fill would have made after each stop.
   - (d) Extended file:
     - exact duplicates (x73=x10, x89=x12, x114=x13, x17=x14, x92=x15, x18=x16);
     - global series hidden in it (x88=x6, x104=x7, x109=x8, x99=x9, 26 constant-across-country columns);
     - x148 constant; x160 = −x143;
     - frequency classification (monthly / annual-January / quarterly / irregular);
     - the 7 gap columns.
   - (e) x1 frozen for 5 of 8 B assets 2024-09…12; asset_16 x5 floor 0.004 (2019-09…2023-03).
3. **Returns EDA** on the full sample for description. Each item is also computed on PRE, and every design decision cites the PRE version only.
   - F1: cumulative excess returns by asset class. Equal-weighted within class, `(1+r).cumprod()`, log y-axis, dashed `axvline` at 2010-01.
   - T2: 50-row table of asset, class, country, annualised mean (12·mean), annualised volatility (√12·sd, ddof=1), Sharpe, worst month and its date. Extra columns give the PRE-block Sharpe.
   - F2: `df.corr()` heat map ordered by class, then country. `sns.heatmap` only, never `sns.set()`.
   - T3: class-block average correlations (within the diagonal, between off the diagonal).
   - F3: rolling 12-month Sharpe by class (class-EW, `rolling(12)` mean/sd × √12). Persistence = the correlation between consecutive non-overlapping 12-month Sharpe ratios, per class.
4. **Characteristic EDA** (T4, F4). Per class, give mean/sd/min/max, boxplots (5 characteristics × 4 classes), the median within-asset ACF at lags 1/6/12/24, and the within-class cross-sectional dispersion over time.
5. **Hypotheses for x1..x5** (T4b). These are predictor-side identities only; no target return is involved.
   - x2 equals the compounded return r_{t-12..t-1} in 14,347/14,400 rows.
   - x5 equals sd(r_{t-36..t-1}) in 13,157/13,200 rows (except asset_16's floor).
   - x3 vs the 60-month cumulative log return.
   - x1 vs the rate differential (B) and the term spread (D).
   - x4 vs Δ12 of the rate differential.
   - Expected hypotheses: x1 = carry, x2 = 12-month momentum (pre-lagged), x3 = value / long-horizon reversal, x4 = 12-month change in the carry fundamental, x5 = 36-month volatility (**the risk measure**).
6. **Tercile sorts (T5, F5), PRE block only: target months 2003-01…2009-12, 84 months.**
   - Each month, rank each lagged characteristic within class and cut at u < −1/6 (T1) and u > 1/6 (T3).
   - Average next-month y (vol-scaled) and raw r by tercile, pooled and per class.
   - Report the monthly T3−T1 spread with mean and t = mean/(sd/√T) against t_{T−1}.
   - A full-sample version of the same table is produced **only after** the OOS run. It goes in an appendix labelled "ex post, not used for design".

### (iii) Lecture grounding
- EDA as statistics before design: L1 p.37-38.
- Correlation matrix and heat map: L1 p.44.
- Sorting returns by characteristics as the original ML: L1 p.35.
- Low SNR prior: L1 p.33.
- t-test of a mean: L2 p.81, p.91-93.
- Multiple testing across 5 characteristics × 4 classes: L4 p.25-27.
- Test set untouched: L4 p.43.
- Never use the same data twice: L5 p.41.
- Threshold chosen on the test block is leakage: L5 p.58 #4.
- Known labels mean classification, not clustering; no asset clustering: L7 p.4.
- Log scale for multiplicative series: L3 p.48-49.
- Cumulative return plots, rolling Sharpe and tercile sorts are listed in exam cell 37. **The Sharpe ratio is [EXAM-DEFINED, not taught]** (L2 boundaries: not covered).

### (iv) Assumptions
- Full-sample *descriptive* statistics are allowed because the exam requires them (cell 37), but no modelling decision may depend on them. The rule is stated in the Data section, and T2/T3/F3 carry PRE columns.
- **Tercile sorts are done on PRE only.** They are a predictor-to-future-return analysis. Doing them on 2010-24 would let OOS returns shape the characteristic set, signs or interactions: garden of forking paths, L4 p.28 / L5 p.41.
  - As a further insulator, the sorts are **not** used to drop characteristics. All five enter every model, as pre-registered.
- Asset classes are taken from `asset_info`. The economic labels (A commodities, B currencies against base country 7, C equity indices, D government bonds) are hypotheses and are presented as such.

### (v) Pitfall guards
- No predictor-vs-return statistic uses dates ≥ 2010-01 before the OOS run (enforced by slicing on `PRE_END`).
- Leading back-fills are set to NaN before any EDA. Plots of x2/x3/x5 start in 2003-01.
- The `.corr().values` array is read-only under pandas 3 copy-on-write; use `.to_numpy(copy=True)`.
- The t-stats in T5 carry the multiple-testing caveat. The largest of 20 null t-stats is typically near 2.9: L4 p.27.

### (vi) Outputs

| output | contents |
|---|---|
| T1 | integrity checks |
| T2 | per-asset statistics |
| T3 | block correlations |
| T4 / T4b | characteristic summaries and identities |
| T5 | PRE tercile sorts |
| T6 | missing values and fills log |
| T7 | x11 rebuild quality |
| T8 | extended-file audit |
| F1 | cumulative returns by class |
| F2 | correlation heat map |
| F3 | rolling Sharpe by class |
| F4 | characteristic distributions and ACFs |
| F5 | tercile bars |
| F6 | x11 vs rebuild with stop dates |

Sanity anchors to check against the data profile:
- monthly vol ranges from 0.82% (asset_16) to 17.40% (asset_47);
- within-class correlation: A 0.21, B 0.55, C 0.725, D 0.58;
- D is negatively correlated with A and C.

### (vii) The write-up must cover
- [ ] cumulative returns by class
- [ ] per-asset table: annualised mean, vol, Sharpe, worst month
- [ ] correlation matrix ordered by class: how strong the B/C/D blocks are, and what that implies for pooling. The effective cross-section is far below 50, so tests aggregate to one number per month (§3.3)
- [ ] rolling 12-month Sharpe by class and its persistence
- [ ] distribution, scale and persistence of each characteristic within class. Units differ by up to 3 orders of magnitude (x1, x4), which motivates the within-class standardisation
- [ ] tercile sorts, with the reason they are PRE-only
- [ ] a hypothesis for each of x1..x5, including which one is the risk measure (x5)
- [ ] the traps found (fills, x10/x73, x11, duplicates, hidden global series, frozen x1, x5 floor)

### (viii) Open questions
- None essential. The default for the ex-post full-sample tercile table is to produce it in an appendix after the OOS run (see OQ list).

---

## Part 2 — Feature engineering

### (i) Goal
Build a predictor set in which every value is known at the end of t−1. Choose standardisation and macro mappings with reasons, fix the target and how forecasts map to positions, and finish with one predictor table with counts.

### (ii) Method steps (cell P3-E; pure functions `build_features(raw, cfg) -> DataFrame` indexed by (date, asset_id))

**2.1 Lag rules**
- Sort by `['asset_id', 'date']`.
- x2 and x5 are used as stored: they are pre-lagged (exam cell 34), which the identities in Part 1 confirm.
- x1, x3, x4: `groupby('asset_id').shift(1)`.
- Every monthly macro series (global x6-x9; country x11*, x12, x13, x14, x15, x16; x86 used to build x11*): `shift(2)` within country along the sorted date axis.
- Low-frequency series are **not used**. The rule is recorded in case the student extends the set:
  - annual (January-stamped, same-year averages): shift 14 months, so the year-Y value first becomes usable at target March Y+1;
  - quarterly (quarter-start stamped): shift 5 months, i.e. 2 months of release delay after the quarter ends.
  - Because the stored annual series is constant within each calendar year, a uniform shift of 14 implements the rule "prior year's value from March onwards" exactly.
- **x10 and x73 are dropped.** They hold the same-year mean of x86, i.e. up to 11 months of look-ahead that a 1-month lag does not remove. Their information is spanned by x86 = x11* + x12.

**2.2 Missing values and stale data (T6)**, per series and country, using only past or same-date information:

| series | pattern (diagnosis) | treatment | why it uses only information available at the time |
|---|---|---|---|
| x2, x3, x5 leading fills (7 assets, ≤2002-08) | backward-filled, so look-ahead | set to NaN; the model sample starts 2004-01, so they never enter | a back-fill carries the first genuine value into earlier months |
| x11 (c12 after 2020-10, c9 after 2021-08, c11 after 2024-03) | stops early at the 2021-23 inflation regime change, no interior holes | **replace x11 by the rebuild x11* = x86 − x12 for all countries and all months.** A same-date identity with no fitted parameters and complete inputs. Report the overlap fit per country (pooled corr ≈ 0.988, RMSE ≈ 0.35; c12 weakest ≈ 0.87) | forward-filling would be wrong by up to ≈10 pp for c12 in 2022; a fitted per-country rebuild would be a fitted transform (L5 p.58 #1). The identity uses only same-month x86 and x12 |
| x1 frozen, 5 B assets, 2024-09..12 | stale vendor values; not future information | keep as stored (it is what a desk saw); flag; robustness R3 excludes those 15 asset-months from evaluation | stale is not look-ahead |
| x5 asset_16 floor 0.004 | vendor floor | keep as a characteristic (within-class rank is barely affected); σ̂ is computed from returns, unfloored | — |
| extended gap columns (x55, x82, x83, x113, x136, x163, x164) | trailing stops, no holes | not used (§2.6) | — |

Rejected treatments (exam cell 37): backward fill, interpolation across gaps, full-sample mean imputation, all of which use the future. Forward-fill of x11 is legitimate in principle but wrong in size at a regime change.

**2.3 Cross-sectional standardisation of characteristics**
- Within month, **within class, rank to uniform scores**: u_{i,t} = (rank_{i,t} − 1)/(n_g − 1) − 0.5 ∈ [−0.5, 0.5], using `groupby(['date','asset_class']).rank(method='average')`.
- Why within class: x1 is in percent for D and in monthly decimals for B, and x4 is ≈1.3 for B/C against 0.004 for D. Ranks across all 50 assets would therefore mostly sort *by class*. Class-level differences are carried by class dummies instead.
- Why rank rather than z-score:
  - robust to extreme values (x3 A min −9.5, x2 A max 2.67, the vendor-winsorised x2 extremes);
  - bounded, so no single asset gets leverage in a pooled fit (L3 p.12-18);
  - has **no parameters estimated across time**, so it cannot leak; it uses only the date-t cross-section of already-lagged values.
- Cost: magnitude and the time-series level of each characteristic are discarded. By design the characteristic block is purely *cross-sectional*, which is what the question calls the literature's signal (exam cell 33). Time-series timing is left to macro and to the TSMOM benchmark (see OQ-8).
- Robustness R4: within-class z-score clipped at ±3 (a fixed constant, not fitted).

**2.4 Global macro (x6..x9)**
- Lag 2, then a trailing z-score on the lagged series:
  `z = (xl - xl.rolling(60, min_periods=36).mean()) / xl.rolling(60, min_periods=36).std()`
- Then interact with the four class indicators, giving 16 columns. Main effects are omitted because the four interactions span them.
- Reason for class-specific slopes: a VIX-type shock plausibly predicts opposite signs for bonds (D) and equities (C), and a common slope would average them (L3 p.53-54, p.57 confounding).

**2.5 Country macro: mapping, differential, class A**
- Series:
  - x11* (real short rate);
  - x12 (CPI inflation);
  - x13 (effective FX, year-on-year);
  - log x14 (policy uncertainty);
  - log x15 (geopolitical risk);
  - x16 (risk rating).
  - Logs for x14 and x15 because they are skewed positive indices that move on a percentage scale (L3 p.48-49).
- Lag 2. Map to assets via `country` (merge on keys).
- Take the **differential against country 7 at the same date**: d_{c,t} = x_{c,t} − x_{7,t}. Country 7 is the natural base: B has no c7 asset, so it is the base currency, and x1 for B is the rate differential against c7 (data profile D).
- Interact with the B, C and D indicators, giving 18 columns. These are standardised by the Pipeline's `StandardScaler`, fitted on training rows only.
- **Class A:** the files' convention maps A to country 7, so its differential is identically **0** and A receives **no country macro**, only global macro. Reasons:
  - A has no natural country (commodities hypothesis);
  - giving A one country's cycle has no economic rationale;
  - a cross-country average would largely duplicate the global block.
  - Side effect, stated honestly: c7's own C and D assets (asset_24, asset_27) also get zero country features. They still see the global block, which is plausibly c7-centred.
- Transforms considered and rejected:
  - rank across 12 countries: coarse, with ties because x10 is identical for c2/4/6/8 and x14 for c10/c11, and it discards the magnitude of, for example, the 2022 inflation gaps;
  - deviation from the 12-country mean: breaks the base-currency logic for B;
  - widened panel with every country's value: 72+ columns per asset on 3,600 initial observations, a multiple-testing and variance problem (L4 p.25-28; L5 p.6, p.9).

**2.6 Extended file policy**
- Used only for (a) the audit (T8) and (b) **x86**, which rebuilds x11 and replaces the look-ahead x10. This is hypothesis-driven: x86 is the documented source of x10, of x1 (B) and of x4 (B).
- No other extended column enters any model. It has 148 columns, 60 of them monthly. Screening them against returns would be the L4 p.26-28 experiment run on ourselves: about 5% "significant" by chance, a best null t near 3.6 for ~100 tests, and adjusted R² gives no protection.
- Rejected option: PCA of the monthly extended columns fitted within each training window (L1 p.45-74; L8 p.61). No hypothesis supports it, and it multiplies the search (OQ list).

**2.7 Interactions and dummies**
- Class dummies d_B, d_C, d_D, with A as the reference (L3 p.45, p.51-52).
- One hypothesis-driven characteristic × macro-state block: u_k × z(x6) for k = 1..5, giving 5 columns. Hypothesis: momentum and carry premia unwind in high-uncertainty states.
- Trees receive the *uninteracted* inputs and find interactions themselves (L8 p.28, p.35-37).

**2.8 Target and positions**
- y_{i,t} = r_{i,t} / σ̂_{i,t-1}, with σ̂_{i,t-1} = sd(r_{i,t-36..t-1}), ddof=1, computed from excess_return:
  `groupby('asset_id')['excess_return'].transform(lambda s: s.rolling(36, min_periods=36).std().shift(1))`
- Why vol-scaled:
  - (a) monthly vol runs from under 1% to over 17%, so a pooled least-squares fit on raw r would be dominated by class A (constant-variance assumption L3 p.2-3, p.20; "calm months and panics" L2 p.80);
  - (b) R² on y weights every asset equally in risk units;
  - (c) ŷ is a "monthly Sharpe" forecast that maps straight into risk-budgeted positions.
- Why this σ̂:
  - it is the provider's own risk definition (x5 = SD36) but without the 0.004 floor;
  - it is built from excess_return alone, as the benchmarks must be;
  - 36 months gives stable weights (lower turnover) at the cost of a slow response.
- **Forecast → position**: expected raw return r̂ = ŷ·σ̂, so the position is ∝ r̂/σ̂² = ŷ/σ̂. Normalise to unit gross.
  - This rule nests the benchmarks: ŷ ≡ constant > 0 gives risk parity, and ŷ ≡ sign(R12) gives TSMOM.
  - The rule is the exam's "positions proportional to the forecast scaled by volatility" (cell 37). The μ/σ² intuition is standard portfolio theory **[not taught; intuition only]**.

**2.9 Future-perturbation test (cell P3-E, before any model)**
- For t0 ∈ {2006-06-30, 2012-01-31, 2020-03-31}:
  - replace every stored value dated ≥ t0 with random draws (returns, x1/x3/x4, all macro including x86), and x2/x5 dated > t0 (row t0 of x2/x5 is t0−1 information);
  - rebuild everything;
  - `assert` that all features, σ̂, trailing-mean benchmarks and portfolio weights for target months ≤ t0 are **bit-identical**.
- This catches a missing shift, a backward fill, a full-sample z-score or a centred rolling window.
- It cannot catch vendor-side stamping such as x10. That is why the audit in Part 1 exists.

### (iii) Lecture grounding
- Scale matters before pooling or penalising: L5 p.23.
- The z formula is L7 p.23. Standardisation within month, trailing global z and the vol-scaled target are **[EXAM-DEFINED]** (cell 37).
- Transformations fitted after the split: L5 p.27, p.47, p.58 #1, p.59; L6 p.30.
- Publication-date merge: L5 p.58 #2.
- Dummies: L3 p.45, p.51-52. Interactions: L3 p.53-55. Logs: L3 p.48-49.
- Multicollinearity from x11* + x12 + x86, so x86 is not added separately: L3 p.39-44; L4 p.14 (rank).
- Multiple testing for the extended file: L4 p.25-28.
- Leverage and outliers argue for ranks: L3 p.12-18.
- Heteroskedasticity motivates vol scaling: L2 p.80; L3 p.2-3.
- Do not use another model's output as a predictor. Characteristics are vendor-constructed but are *inputs*, not forecasts: L6 p.30.

### (iv) Assumptions
- Macro releases arrive within one month (lag 2). x6 is market-based and could be lag 1; sensitivity R1 uses lag 1 for all.
- x86 is known at month end.
- The identity x11 ≈ x86 − x12 holds after the stop at the same quality as before it.
- The frozen x1 values were what a desk would have seen.
- Class labels are fixed.
- No external data (OQ-9).

### (v) Pitfall guards
- `groupby().shift()` only after an explicit sort.
- Merges only on keys, with row counts asserted.
- No `fillna(method=)`; use `.ffill()` if ever needed.
- No `df.append`; use `pd.concat`.
- No `.iteritems()`.
- The perturbation test and a lag unit test: for 20 random (asset, date) pairs, assert feature = stored value at t − lag.
- Duplicated or collinear columns: assert `np.linalg.matrix_rank(Xs) == Xs.shape[1]` on every training design (L4 p.14).

### (vi) Outputs
- T9 (final predictor table with counts):

| block | columns | source → lag → transform | classes | count |
|---|---|---|---|---|
| characteristics | u1..u5 | x1 (lag 1), x2 (0), x3 (1), x4 (1), x5 (0) → within-class, within-month rank score | all | 5 |
| class dummies | d_B, d_C, d_D | asset_info, A reference | all | 3 |
| global × class | z(x6..x9) × {A, B, C, D} | lag 2 → trailing 60m z | all | 16 |
| char × state | u_k × z(x6) | product | all | 5 |
| country × class | {x11*, x12, x13, log x14, log x15, x16} differential vs c7 × {B, C, D} | lag 2 → minus c7 → Pipeline scaler | B, C, D (0 for A and c7 assets) | 18 |
| **CHAR** | u1..u5 + dummies | | | **8** |
| **CHAR+G** | CHAR + global×class + char×state | | | **29** |
| **CHAR+C** | CHAR + country×class | | | **26** |
| **CHAR+MACRO** | all of the above | | | **47** |
| RF inputs | u1..u5, 3 dummies, z(x6..x9), 6 country differentials (uninteracted) | | | **18** (CHAR-only: 8) |
| target | y = r/σ̂_{t-1} | | | 1 |

- Excluded: x10, x73 (look-ahead); raw x11 (replaced); raw x86 (spanned); every other extended column.
- The perturbation-test log.

### (vii) The write-up must cover
- [ ] lag rule per variable, including the release delay and the annual/quarterly look-ahead fix
- [ ] standardisation choice (within class, rank) and why
- [ ] missing values: diagnosis per series and country, treatment, only-past justification, the regime-change trap, rebuild quality
- [ ] own data: none, and why
- [ ] trailing standardisation of global macro
- [ ] country mapping, the differential against c7, and the class A choice
- [ ] cross-sectional macro transforms considered and rejected
- [ ] interactions, class dummies, and the extended-file policy with the multiple-testing argument
- [ ] target choice, which σ̂ exactly, and how forecasts become positions
- [ ] predictor table with counts

### (viii) Open questions
- OQ-4 (macro lag 2 vs 1)
- OQ-6 (class A country macro)
- OQ-8 (add a time-series characteristic channel)
- OQ-9 (external data)

---

## Part 3 — Evaluation harness, models × schemes

### (i) Goal
Measure how much of next month's (vol-scaled) excess return is forecastable OOS, whether macro adds anything, and how sure we can be. Every model is scored on the **same** OOS rows, against the **same** pre-registered benchmarks (L8 p.53).

### (ii) Method steps

**3.1 Windows (cell P3-G), coded once and shared by every model**

```python
YEARS = range(2010, 2025)
def outer_window(scheme, Y):
    if scheme == 'static':    return ('2004-01-31', '2009-12-31')        # one fit, reused for all Y
    if scheme == 'expanding': return ('2004-01-31', f'{Y-1}-12-31')
    if scheme == 'rolling':   return (f'{Y-6}-01-31', f'{Y-1}-12-31')    # 72 months
def inner_folds(train_dates, Y):                                         # positional indices
    folds = []
    for v in (Y-3, Y-2, Y-1):
        fit = train_dates <= pd.Timestamp(f'{v-1}-12-31')
        val = (train_dates >= pd.Timestamp(f'{v}-01-31')) & (train_dates <= pd.Timestamp(f'{v}-12-31'))
        assert train_dates[fit].max() < train_dates[val].min()
        folds.append((np.flatnonzero(fit), np.flatnonzero(val)))
    return folds
# every window: assert train_end < test_start; test = Jan..Dec of Y; static reuses the Y=2010 model
```

- Rolling-72 in 2010 is the same window as expanding in 2010, and both equal static. The schemes differ only from 2011, which gives a clean comparison.
- The first inner fold always has ≥36 months (1,800 rows pooled).
- There are 29 unique fits per specification.

**3.2 Forecast benchmarks (cell P3-F, computed and tabulated before any model is imported)**

All benchmarks are for the target y, use data through t−1 only and are updated monthly:
- b⁰ = 0 (notebook: "or zero"; L4 p.46; L5 p.56-57).
- **b^asset_{i,t} = mean(y_{i,s} : 2003-01 ≤ s ≤ t−1).** This is *the* trailing mean of exam cell 36 for a panel: each asset's own "mean of the data seen so far", as in the HW5 1.1 harness.
  - For a vol-scaled target it is the trailing mean **of y itself** (a trailing monthly Sharpe), not r̄/σ̂_{t−1}, because R² compares two forecasts of the same variable.
  - Code: `y.groupby(asset).transform(lambda s: s.expanding().mean().shift(1))`.
- b^class_{g,t} = the mean of y over all assets in class g and all months ≤ t−1. This is the forecast of an intercept-plus-class-dummy model, i.e. the "no-characteristics" comparator.
- b^pool_t = the mean over all assets and months ≤ t−1. It is the λ→∞ limit of the ridge model, up to annual versus monthly updating.
- Secondary, raw-return versions for Q1's "excess return" wording:
  - r̂ = ŷ·σ̂;
  - benchmarks zero and the per-asset expanding mean of r from 2000-01;
  - reported **by class only**, because pooled raw SSE is dominated by class A.
- **T10 (before models):** R²_OOS of each benchmark against each other benchmark, over the OOS window, pooled and by class. This shows which benchmark is hardest *before* any model is seen.
  - Expectation stated ex ante: b^asset is noisy (≈ 84-264 months per asset), so it is the easiest trailing mean to beat. Zero and b^class are harder. That is why all are reported.

**3.3 R²_OOS and statistical judgement (cell P3-G)**

```python
def r2_oos(y, yhat, ybench):                      # never r2_score / model.score
    assert y.index.equals(yhat.index) and y.index.equals(ybench.index) and not np.isnan(yhat).any()
    return 1 - ((y - yhat)**2).sum() / ((y - ybench)**2).sum()
def monthly_diff(y, ya, yb):                      # >0 means forecast a beats b in month t
    return ((y - yb)**2 - (y - ya)**2).groupby(level='date').mean()   # T=180 numbers
def diff_test(d):
    T = len(d); t = d.mean() / (d.std(ddof=1) / np.sqrt(T))
    return t, 2*stats.t.sf(abs(t), df=T-1), d.autocorr(1)
```

- **Aggregate to one number per month first.** All 50 assets share each month's shocks (C block ρ ≈ 0.73), so asset-months are not independent. The test is the L2 t-test of the mean of 180 monthly loss differentials (L2 p.81, p.91-93). A DM-type statistic without HAC is the closest lecture-grounded analogue.
  - Diebold-Mariano with HAC, Clark-West and block bootstraps are **[OUTSIDE SCOPE — only if the user approves]**.
- **Dependence caveat:** report ρ₁(d_t). If it is material, the iid SE is too small (L2 p.80). An iid bootstrap of months (B = 10,000, `rng = np.random.default_rng(7034)`, resampling rows of the monthly SSE sums) gives a percentile CI for R²_OOS. It relaxes normality but still assumes independent months (L2 p.80).
- **Nested-comparison caveat** (macro vs no macro): under the null, the extra coefficients add estimation error (forecast error = noise + estimation error, L2 p.100-101). The plain differential test is therefore *conservative* against the bigger model. The placebo (§3.6) is the like-for-like fix: same parameter count, wrong timing.
- **Multiple testing:** Bonferroni critical values t_{179}(1 − 0.05/(2m)) (L4 p.27-28), evaluated at the design stage:

| number of tests m | critical t |
|---|---|
| 1 | 1.97 |
| 2 | 2.26 |
| 3 | 2.42 |
| 6 | 2.67 |
| 12 | 2.90 |
| 42 | 3.29 |
| 126 | 3.61 |

  The pre-registered primary test uses 1.97. Any "best of the grid" claim uses the m = 42 value.
- **Is an R²_OOS of 0.3% distinguishable from zero?** Ex-ante calculation from L2 p.63-64 (var(X̄) = σ²/n):
  - Suppose y = s + e with var(e) ≈ 1 and var(s) = R², and even a perfect forecast of s.
  - Then d_it = s² + 2se, so E[d] = R² and sd(d) ≈ 2R. Hence t ≈ R·√(N_eff·T)/2.
  - At T = 180 and R² = 0.3%: t ≈ 2.6 if N_eff = 50, 1.6 if N_eff = 20, 1.2 if N_eff = 10.
  - Given the documented block correlations, N_eff is far below 50. **A single R²_OOS of 0.3% will usually not be statistically distinguishable from zero, even for a perfect forecaster**, and estimated forecasts are worse (L2 p.100-101).
  - So the write-up reports CIs, not only point estimates, and leans on the placebo and portfolio evidence. This is written into the Methodology section *before* results.
- Sanity rules:
  - IS R² (vs the training mean, L4 p.44) must exceed OOS R² (L4 p.47-48; AI guide §6);
  - `LEAK_ALARM` at 2% (L5 p.57);
  - a cumulative SSE-difference plot, Σ_{s≤t}[SSE_bench,s − SSE_model,s] over OOS months (F8), shows *when* any gain accrues. It visualises the same quantity as the R² numerator versus denominator (L4 p.44-46; L1 p.37).

**3.4 Models (cell P3-H)**

The menu is deliberately small:
- **OLS**: `Pipeline([('s', StandardScaler()), ('m', LinearRegression())])`. The unpenalised comparator and kitchen-sink warning (L4 p.47-48; L5 p.6, p.9).
- **Ridge (primary)**: `Pipeline([('s', StandardScaler()), ('m', Ridge())])`, tuned by
  `GridSearchCV(pipe, {'m__alpha': RIDGE_GRID}, cv=inner_folds(...), scoring='neg_mean_squared_error', refit=True, n_jobs=1)`
  - Why ridge: many weak, correlated predictors (within-class x2-x3 −0.14…−0.41, C x3-x5 0.57) fit L5 p.60's case for ridge. In asset pricing the signal is often weak and linear (L8 p.62).
  - y is not standardised; it is already unit-free with variance ≈ 1.
  - Dummies are standardised and penalised, so λ→∞ gives the pooled training mean (b^pool).
- **Lasso**: the same with `Lasso(max_iter=10000)` and `LASSO_GRID`. The few-strong alternative (L5 p.17-24, p.60).
- **Random forest**: `RandomForestRegressor(n_estimators=300, min_samples_leaf=200, max_features=1/3, random_state=7034, n_jobs=N_JOBS)` on the 18 uninteracted inputs.
  - No tuning, so there is no validation-search leakage (L8 p.44-45, p.55).
  - p/3 per split (L8 p.43). sklearn's default `max_features=1.0` would be bagging, not a forest.
  - Big leaves because the SNR is tiny.
  - The bootstrap happens inside the training window, so it is not a validation step. OOB error is **[OUTSIDE SCOPE]** and would be a random fold on time data anyway.
- **Pooled vs per-class**:
  - The pooled model with a *common* characteristic slope is the direct test of the literature claim that the same characteristics work "across many markets and asset classes" (exam cell 33). Pooling raises n and s_x and lowers var(b) (L2 p.73).
  - Per-class models are the same pipelines fitted separately per class:
    - CHAR: 5 ranks;
    - CHAR+MACRO for A: 5 + 4 global z + 5 char×state = 14;
    - CHAR+MACRO for B/C/D: 14 + 6 country = 20.
  - Per-class fits are equivalent to a fully interacted pooled model (L3 p.54, "why not separately estimate three models?").
  - OLS and ridge only. Each class is tuned on its own inner folds, and predictions are stacked so R² is computed on the same 9,000 rows.
- **Excluded, and why:**
  - GBRT: its tuning budget is a leakage risk and costly (L8 p.55). Also, sklearn's `n_iter_no_change` early stopping calls a **shuffled** `train_test_split` internally (checked in the installed 1.9.1 source), which is fatal error (a). See OQ-7.
  - KNN: curse of dimensionality (L6 p.37).
  - Neural networks, elastic net estimator, PCR/PLS, XGBoost, SHAP: not taught (L1 p.36 legend only, L5 p.11 formula only, L8 p.62).
- BIC for tuning (L5 p.42-46) was considered and rejected. Its i.i.d. Gaussian likelihood treats 12,000 cross-sectionally correlated rows as independent, so the log n penalty is miscalibrated. The time-ordered CV estimates OOS loss directly (L5 p.26).

**3.5 Scheme rules**
- Static: fit once on INIT_TRAIN, tuned on inner folds 2007/2008/2009.
- Expanding: refit each January on 2004-01…(Y−1)-12. The start is fixed, which is L5 p.50's requirement.
- Rolling: 72-month window.
- Annual refits: L5 p.51 uses step 12; with low SNR the coefficients drift slowly; annual refits keep the budget small (§7).
  - The trailing-mean benchmarks update monthly while model coefficients update annually. This is conservative for the models.
- After tuning, refit on the whole training window (L5 p.36). This deviates from L5 p.52's code, which predicts with train-only coefficients; the deviation is stated.
- Grid-edge log (T16): the chosen α and its grid index for every refit.
  - The grids run from OLS-equivalent to null-equivalent, so an edge solution is interpretable: the upper edge is the null / pooled mean (L5 p.38), the lower edge is OLS (L5 p.39). It is reported rather than "fixed".
  - Note the lecture's own edge case (L5 p.53).

**3.6 Macro vs no macro, global vs country, placebo, per-class vs pooled (cell P3-I)**
- **Identical schemes and rows:**
  - CHAR and CHAR+MACRO (and CHAR+G, CHAR+C for ridge) use the same windows, folds, grids, seeds and the **same row mask** (all 47 features non-missing), so any difference comes from the features only.
  - Primary Q2 test: Ridge pooled expanding, ΔR² = R²(CHAR+MACRO) − R²(CHAR) against each benchmark, with the paired monthly-differential test.
  - The global/country split uses m = 3 Bonferroni (MACRO, G, C).
  - The x1 and x4 characteristics for B are near-deterministic functions of country rates, so "macro" is partly inside the characteristics already. Q2 measures macro *beyond* that; stated ex ante.
- **Placebo (exam cell 37) [EXAM-DEFINED; honest-null logic from L4 p.26-28 and L2 p.88-93]:**
  - Circularly shift **every raw stored macro series** (x6-x9, x11 inputs x86 and x12, x13-x16) by k months *before* the lag/transform pipeline: `np.roll(values_sorted_by_date, k, axis=0)` within each country, with the same k for all countries and series.
  - Characteristics are untouched, including x1 and x4 that embed macro.
  - Rerun the identical pipeline, including the trailing z-scores, which are recomputed on the shifted series.
  - K_main = {36, 60, 96, 132, 168, 204} is run for OLS/Ridge/Lasso CHAR+MACRO under all 3 schemes, for RF expanding, and for Ridge CHAR+G and CHAR+C expanding.
  - K_full = {36, 48, …, 264} (20 shifts) is run for the primary Ridge pooled expanding model. This gives an empirical null distribution of ΔR² (F10, drawn like L4 p.27's null max-t histogram).
  - Shifts stay ≥36 months from zero in *both* directions (264 ≡ a 36-month lead). A near-zero *forward* shift would inject "returns anticipate macro" reverse causality, which is not a placebo.
  - **Persistence caveat:** a circular shift by k is, for most dates, the real series lagged k months. For slow series (x9 AC12 ≈ 0.73, x16 ≈ 0.62) it stays partly correlated with the truth. T14b reports corr(x_t, x^{(k)}_t) per series and shift, so a placebo that "works" can be read correctly.
  - **Pre-registered reading:** real macro is credited only if (1) its ΔR² > 0 with a paired t above the Bonferroni value for m = 3, **and** (2) it exceeds the ΔR² of most placebo shifts. Report its rank among 21 (real plus 20). A rank of 1/21 corresponds to p ≈ 0.048. Neighbouring shifts are correlated, so this p is approximate.
  - The placebo also neutralises the nested-model estimation-noise handicap, because it has exactly the same parameter count.
- **Per-class vs pooled:** R² pooled and by class, with the paired monthly test (T15).
- **Macro-lag robustness R1** (lag 1) and **x11 splice R2** (original x11 until the stop, then the rebuild) for the primary Q2 model only.

**3.7 Which predictors carry the signal, and is it stable? (cell P3-J)**
- **Coefficient paths**: standardised ridge coefficients per refit (15 expanding, 15 rolling), plotted against year for u1..u5 and the 8 largest macro terms (F11, in the style of the L5 p.22 path plot). T-sign: the share of refits with the same sign as the average.
- **Lasso selection frequency** over the 15 expanding refits.
- **Grouped permutation importance, OOS** (L8 p.56-58):
  - For each test year Y, take the model fitted through Y−1 and permute a *group* of columns jointly across that year's rows. Use 10 repeats with `rng(7034)`, since a single permutation is itself a random draw (L8 p.57).
  - Importance = ΔSSE / SSE_zero-benchmark, in R² points.
  - Groups: u1, …, u5; dummies; char×state; each global series (its 4 class columns); each country series (its 3 class columns). That is 17 groups for linear models; 15 for RF (u1..u5, dummies, x6..x9, 6 country).
  - sklearn's `permutation_importance` permutes single columns and defaults to `estimator.score`, i.e. R² against the test mean. We use custom grouped code with MSE, or `scoring='neg_mean_squared_error'` for singletons.
  - Heat map group × year (F12). MDI is not used as evidence because it is in-sample (L8 p.47, p.56).
- **Partial dependence** (RF, top 2 groups): the brute-force L8 p.59 function on a 40-point 5th-95th percentile grid, on the final expanding refit's training rows. It shows sign and shape, and it is "what the model says, not causal" (L8 p.59-60).

### (iii) Lecture grounding (by step)
- Windows: L5 p.48-52, p.56; L4 p.43; L1 p.82.
- R²_OOS and benchmarks: L4 p.44-46; L5 p.55-57.
- Tests: L2 p.63-64, p.78-81, p.88-93, p.100-101.
- Search size: L4 p.25-28; L5 p.57.
- Models: L3 p.19-36; L5 p.12-24, p.60; L8 p.42-45, p.55, p.62.
- Tuning: L5 p.27, p.35-36, p.41, p.47, p.53.
- Pooled vs per-class: L2 p.72-73; L3 p.54.
- Placebo logic: L4 p.26-28.
- Importance: L8 p.56-60.
- Same split for every model: L8 p.53.
- Placebo, vol-scaled target, the per-month trailing mean and the three schemes are **[EXAM-DEFINED]** (cells 36-37).

### (iv) Assumptions
- Validation MSE on three recent years is an honest (noisy, L5 p.41) estimate of next-year loss.
- The coefficients of year Y−1 apply throughout year Y.
- Cross-sectional dependence is absorbed by monthly aggregation; time dependence is diagnosed via ρ₁.
- The OOS window is long enough to be informative but not decisive at R² ≈ 0.3% (power calculation above).

### (v) Pitfall guards
- No `train_test_split`, `KFold`, `cross_val_score`, `LassoCV`, `RidgeCV` or GBRT `n_iter_no_change` anywhere in P3 (a grep assertion in P3-L).
- Every scaler lives inside the Pipeline and is refitted per fold by GridSearchCV.
- `scoring='neg_mean_squared_error'` is set explicitly. GridSearchCV's default `scoring=None` would use R² against the *fold* mean.
- Own `r2_oos`.
- Index-alignment asserts.
- Seeds set.
- `Lasso` `ConvergenceWarning`s are logged. The notebook's global `warnings.filterwarnings('ignore')` would hide them.
- Harness self-tests (cell P3-G, run before any model):
  - (1) `r2_oos(y, b_asset, b_asset) == 0` and the same for zero;
  - (2) random N(0, 0.05²) "forecasts" must give a negative R² vs zero;
  - (3) `r2_oos(y, y, b) == 1`;
  - (4) an optional **lag canary**: OLS CHAR with x1/x3/x4 *unshifted*, reported only in the leakage appendix, to show why the lag rule matters (not a candidate specification).
- **Verify the headline** (AI guide §6):
  - recompute the primary ridge forecasts for one refit year from the closed form (X'X + λI)⁻¹X'y on training-scaled X with centred y (L5 p.12) and assert agreement with sklearn to 1e-8;
  - recompute the headline R² from scratch with numpy.

### (vi) Outputs
- T10 benchmark-vs-benchmark R² table.
- **T12, model × scheme** (42 forecast specifications). Columns:

| column group | contents |
|---|---|
| identification | model, features, pooling, scheme |
| in-sample | IS R² (mean over refits) |
| OOS R² | vs b^asset, vs b^class, vs b^pool, vs 0 |
| tests | t(d) vs b^asset and vs 0; Bonferroni flags |
| uncertainty | bootstrap 95% CI (vs 0); ρ₁(d) |
| tuning | count of edge solutions |

- T13 by class and by sub-period for the primary models.
- T14 macro vs no macro and G vs C, with placebo ΔR² for K_main. T14b placebo persistence correlations. T14c real-macro rank among the 21.
- T15 per-class vs pooled.
- T16 α per refit.
- T21 raw-return R² by class.
- F8 cumulative SSE difference.
- F9 R² bar chart, model × scheme.
- F10 placebo histogram with the real value marked.
- F11 coefficient paths.
- F12 importance heat map.
- F16 partial dependence.
- An OOS forecast-tercile table (the L6 p.56 lift-style check): average realised y by within-class forecast tercile.

### (vii) The write-up must cover
- [ ] the models and why this menu
- [ ] static / expanding / rolling definitions
- [ ] the OOS window and initial block with dates and reasons (§0), fixed in advance
- [ ] tuning on training data only (inner time-ordered folds), grid-edge report
- [ ] R²_OOS against the trailing mean *and* zero, what "trailing mean" means for this panel, model × scheme, by class
- [ ] which scheme wins and why (bias-variance L5 p.7-8; nonstationarity L1 p.34)
- [ ] which predictors carry the signal and whether it is stable across windows
- [ ] whether macro adds when refit without it, and global vs country
- [ ] placebo with several shifts and the persistence caveat
- [ ] per-class vs pooled
- [ ] significance with the dependence caveat
- [ ] search size (§8)
- [ ] IS vs OOS

### (viii) Open questions
- OQ-1 (OOS start 2010 vs 2012)
- OQ-7 (GBRT)
- OQ-11 (placebo set size vs runtime)

---

## Part 4 — Portfolios, costs and attribution

### (i) Goal
Answer Q3: does a forecast-based portfolio beat EW, risk parity and TSMOM out of sample, after costs? Where it does, how much of that is benchmark exposure and how much is timing?

### (ii) Method steps (benchmarks in cell P3-F *before models*; forecast portfolios in P3-K)

**4.1 Common machinery.** Returns matrix R (date × 50 assets). Weights W_t are formed from information through t−1 and held during month t.

```python
def unit_gross(raw):                       # Σ|w| = 1; all-zero row → cash
    g = raw.abs().sum(axis=1)
    return raw.div(g.where(g > 1e-12), axis=0).fillna(0.0)
def port_ret(W, R):  return (W * R).sum(axis=1)
def traded(W, R):                          # one-way traded notional per unit capital (buys + sells)
    Rp = port_ret(W, R)
    drift = (W.shift(1) * (1 + R.shift(1))).div(1 + Rp.shift(1), axis=0)
    return (W - drift).abs().sum(axis=1)
net = lambda W, R, c: port_ret(W, R) - c * traded(W, R)
```

- The drift formula treats excess-return positions like fully funded holdings, an approximation for futures-style overlays; it is stated.
- The January 2010 initial build is excluded from the turnover and costs of **every** strategy, so all start equally.

**4.2 Benchmarks, exactly as specified (exam cell 36) [EXAM-DEFINED, not taught]**
- EW: w = 1/50.
- RP: w ∝ 1/σ̂_{t−1}.
- TSMOM: w ∝ sign(R12_{t−1})/σ̂_{t−1}, with R12 from `np.expm1(np.log1p(r).rolling(12).sum()).shift(1)` per asset.
- All use unit gross exposure, monthly rebalancing and excess_return only.
- Computed from 2003-01. The headline evaluation is 2010-01…2024-12, and a full-period line is descriptive.
- Report:
  - average gross weight share by class;
  - maximum single-asset weight. With unfloored σ̂, asset_16 (true SD36 down to ≈0.003 in 2019-23) may take a very large RP/TSMOM share. This is a property of the exam's rule and is disclosed; robustness R5 applies a 0.004 floor (the provider's own) to *all* portfolios alike;
  - the class risk shares of EW: A's share of EW variance, cov(R_A-part, R_EW)/var(R_EW). This addresses the exam's warning about pooling classes in one equal-weighted portfolio (cell 34).

**4.3 Forecast rules (pre-registered; all use ŷ at t from features ≤ t−1 and the same σ̂)**
- **P1 (primary)**: w ∝ ŷ_{i,t}/σ̂_{i,t−1}, unit gross. With ŷ ≡ c > 0 it equals RP *exactly*.
- **P2**: w ∝ sign(ŷ_{i,t})/σ̂_{i,t−1}. This is TSMOM's shape with the model's sign, so the comparison with TSMOM is like-for-like.
- **P3**: w ∝ (ŷ_{i,t} − mean_{g(i)} ŷ_t)/σ̂_{i,t−1}. Within-class long-short, the pure cross-sectional signal: the literature's "long top, short bottom" (exam cell 37; L1 p.36; L7 p.6 group portfolios). It is scale-free, so it bets on the ranking regardless of how shrunk ŷ is.
- **Identity self-tests** (cell P3-G): assert P1(ŷ ≡ 1) == RP and P2(ŷ ≡ R12) == TSMOM to 1e-12, and that the perturbation of r_t leaves W_t unchanged.
- Headline portfolios: P1/P2/P3 on Ridge CHAR pooled expanding (Q3 primary = P1) and on Ridge CHAR+MACRO pooled expanding (the portfolio side of Q2). The full grid of 42 specifications × 3 rules goes to an appendix table (gross and net Sharpe only).

**4.4 Performance metrics [EXAM-DEFINED in cell 37; formulas stated because no lecture defines them]**
- Monthly excess return R_t.
- Annualised mean 12·R̄; annualised volatility √12·sd(R), ddof=1 (L2 p.56-57).
- **Sharpe** = √12·R̄/sd(R). No risk-free rate is subtracted because the returns are already excess.
- t(mean) = R̄/(sd/√T), which equals the monthly Sharpe × √T (L2 p.81). Hence **SE(annual Sharpe) ≈ √(12/T) ≈ 0.26 at T = 180**, stated ex ante: Sharpe gaps below ≈0.5 are not distinguishable.
- Maximum drawdown on W_t = ∏(1+R): max_t(1 − W_t/max_{s≤t} W_s).
- Worst month.
- Average traded notional per month.
- Maximum weight.
- Sub-period Sharpe and mean (2010-14, 2015-19, 2020-24).
- Figures:
  - F13: cumulative log wealth at unit gross, with a second panel scaled to 10% ex-post annual volatility, labelled "display only; ex-post scaling". Scaling does not change the Sharpe. This follows L1 p.39's warning about comparing levered curves.
  - Drawdown panel.
  - F15: rolling 36-month Sharpe of P1 minus each benchmark.

**4.5 Transaction costs [NOT TAUGHT; transparent assumption]**
- Proportional cost c per unit notional traded (one-way): net_t = R_t − c·TN_t, with c ∈ {0, 5, 10, 25, 50} bp. The headline "after costs" verdict is at 10 bp, pre-registered.
- Break-evens:
  - c*₀ = mean(R)/mean(TN), the cost that sets the net mean to zero;
  - c*_b = the cost at which net Sharpe(strategy) = net Sharpe(benchmark b), solved with `scipy.optimize.brentq` on [0, 0.02], or reported as "no crossing in [0, 200 bp]".
  - Break-evens let readers plug in their own cost.
- Class-specific costs: OQ-3.
- F14: net Sharpe against c for all strategies, with crossings marked.
- The rules and cost grid are fixed ex ante, like L6 p.46-47's cost-derived threshold. It is not optimised on the test sample.

**4.6 Statistical judgement and attribution: exposure vs timing**
- **Regression attribution** (L2 p.47-49 single-factor form; L3 p.19-36 MLR; L2 p.94-95 α test):
  `smf.ols('S ~ EW + RP + TS', data=monthly).fit()` (default nonrobust covariance, as in the lectures).
  - The β's are the exposure to the benchmark rules. α (×12 annualised) with t(α) is the part the rules cannot replicate, i.e. timing and selection.
  - Partial F-test that all β = 0 via `anova_lm(smf.ols('S ~ 1', ...).fit(), full)` (L4 p.11-15).
  - Single-benchmark regressions too, because EW and RP are both long-only and collinear (L3 p.39-44). Report their correlation (L1 p.44).
  - For P2 on TSMOM: test H0 β = 1 by hand (L2 p.96-97) and α = 0. "Is P2 just TSMOM?"
  - Run the regressions gross and net at 10 bp (net strategy on net benchmarks).
  - iid month-bootstrap CI for α (L2 p.80), with the dependence caveat.
  - Bonferroni for the two headline portfolios gives m = 2, t = 2.26.
- **Position-space overlap**: for each month, the cross-sectional correlation of P1's weight vector with the RP and TSMOM weight vectors, averaged over OOS (L1 p.44). This answers "how much of P1 *is* risk parity?".
- **Static-vs-timing decomposition** [EXAM asks, cell 37; identity not taught, so descriptive and ex post]:
  - w̄_i = the OOS average P1 weight;
  - R_static,t = Σ w̄_i r_{i,t}, and timing_t = R_t − R_static,t;
  - report the annualised mean of each and each class's contribution.
  - Flagged: w̄ uses the whole OOS window, so this is accounting, not a tradable strategy.
- **Placebo portfolios**: P1 Sharpe for Ridge CHAR+MACRO under each K_main shift, next to the real-macro P1. Does the portfolio gain survive the placebo?
- **Pre-registered verdict rule for Q3**: "beats" requires both
  - a higher net Sharpe at 10 bp than each benchmark over 2010-24, **and**
  - α > 0 in the three-benchmark regression with t > 2.26.
  - Otherwise report "not distinguishable". The sub-periods show fragility.

### (iii) Lecture grounding
- Excess returns and α as a performance benchmark: L2 p.47-49.
- α and β tests: L2 p.94-97.
- MLR inference: L3 p.29-36.
- Multicollinearity: L3 p.39-44.
- Partial F: L4 p.11-15.
- Bootstrap: L2 p.78-80.
- t of a mean: L2 p.81.
- Forecast-sorted portfolios vs a benchmark: L1 p.36.
- Too-good track records: L1 p.39.
- Decision rules fixed ex ante from costs: L6 p.46-47.
- Rank-and-select lift as the classification cousin of sorts: L6 p.56.
- **Not taught:** EW/RP/TSMOM, unit gross exposure, Sharpe, drawdown, turnover, costs and break-even. These are exam-defined (cells 36-37) or flagged constructions.

### (iv) Assumptions
- Returns are excess returns on self-financing or futures-like positions, so cash earns nothing extra.
- A single proportional cost; no market impact.
- Drift-adjusted turnover.
- σ̂ unfloored in the primary.
- Headline cost 10 bp.
- Monthly rebalancing at month-end prices, with no execution lag beyond the information lag.

### (v) Pitfall guards
- Weights are built only from `.shift(1)` inputs and ŷ_t (features ≤ t−1), confirmed by the perturbation test.
- Identity self-tests.
- Cash when Σ|raw| = 0.
- The same σ̂ everywhere.
- No ex-post scaling in any reported Sharpe or α.
- Costs charged to the benchmarks too.
- The same OOS window for every strategy.
- No switching of the headline rule, cost or benchmark after seeing results (L5 p.57-58 #5).

### (vi) Outputs
- T11 benchmark portfolios, reported before models.
- T17 performance of 3 benchmarks + 6 headline forecast portfolios, full OOS and sub-periods.
- T18 cost grid and break-evens.
- T19 attribution regressions.
- T20 exposure/timing decomposition and weight overlaps.
- Appendix grid of 126 portfolio specifications.
- F7 benchmark cumulative returns.
- F13 cumulative returns and drawdowns.
- F14 net Sharpe vs cost.
- F15 rolling Sharpe differences.

### (vii) The write-up must cover
- [ ] how each portfolio is formed and rebalanced, with the exact formulas
- [ ] benchmarks built exactly as the notebook specifies
- [ ] Sharpe at a minimum, plus return, volatility, drawdown, turnover, net of costs, cumulative plots, sub-periods
- [ ] the cost assumption flagged as an assumption, with break-evens
- [ ] exposure vs timing (β's, α, weight overlap, decomposition)
- [ ] the after-costs verdict under the pre-registered rule
- [ ] the uncertainty of Sharpe differences (SE ≈ 0.26)

### (viii) Open questions
- OQ-2 (headline forecast set for Q3)
- OQ-3 (cost level / class-specific costs)
- OQ-10 (σ̂ floor)

---

## Part 5 — Write-up structure (exam cell 38)

- **Introduction**
  - The three desk questions.
  - Why they matter to a cross-asset allocator (costs of turnover, benchmark-relative evaluation).
  - Literature expectation: GKX forecast-sorted portfolios beat the market benchmark (L1 p.36); monthly R²_OOS of 0.3-0.5% is normal (L5 p.57); low SNR by market efficiency (L1 p.33). Further papers (value, momentum or carry "everywhere", TSMOM) are **[OUTSIDE SCOPE — cite only if the user verifies them]**.
  - A one-paragraph preview, written last.
- **Data**
  - Universe, classes, the country table.
  - T2, T3, F1-F3.
  - Characteristic hypotheses (T4/T4b).
  - Treatment: fills, x10/x73 dropped, x11 rebuild (T7/F6), frozen x1, x5 floor, extended audit (T8), lags, standardisation, macro mapping, class A.
  - Predictor table T9.
- **Methodology** (a reader must be able to reproduce the study from it)
  - Pre-registration table (§0).
  - Benchmarks and why (T10 and T11 reported first).
  - Target and position mapping.
  - Models and pipelines.
  - Schemes, OOS window, refits, inner CV, grids.
  - R² definitions, tests, power statement, Bonferroni, placebo design.
  - Portfolio rules, cost model and metric formulas.
  - The L6 p.57 reporting checklist (question/target; split with dates and sizes; estimator and tuning; test performance vs benchmark; decision rule and costs) is mapped onto these sections.
- **Results**, in question order:
  - (1) forecastability: T12, T13, F8, F9, T21, IS vs OOS;
  - (2) macro: T14/T14b/T14c, F10, T15, importance F11/F12;
  - (3) portfolios: T17-T20, F13-F15.
  - Negative numbers are reported as results (L5 p.57).
- **Interpretation and discussion**
  - The economics of any signal found (carry, momentum, value, low risk).
  - Why the numbers come out as they do: the power statement, pooling vs dependence, macro already inside x1/x4, momentum overlapping TSMOM.
  - Strong vs fragile evidence: sub-periods, placebo rank, CI width.
  - What did not work, from the full ledger (§8).
  - Next steps (OQ list).
- **Conclusion**: a PM-readable half page answering Q1-Q3 in plain words, each with one number and its uncertainty.
- **Appendices**: search ledger T22; deviations log; leakage audit T23 with assertion outcomes; ex-post full-sample tercile sorts; lag canary; robustness R1-R5.

---

## Part 6 — Leakage audit checklist (T23; each row has a code assertion or test)

| # | risk | where it could enter | guard in this design (cell) | check |
|---|---|---|---|---|
| a | shuffled split or folds (fatal; AI guide 4a; L5 p.58 #3, p.59) | train_test_split, KFold, GBRT early stopping, RF OOB | date-based windows and inner folds only (P3-G); GBRT and OOB not used | grep assertion; `train_end < val_start < test_start` asserts in every window |
| b | transform fitted before the split (fatal; AI guide 4b; L5 p.58 #1, p.47) | scaler, imputer, PCA, winsorising, z-scores | Pipeline(StandardScaler, model) refitted per fold; ranks are same-date; trailing z uses rolling past windows; x11* is an identity; no imputer or PCA | future-perturbation test (P3-E) |
| c | LogisticRegression called "plain" (AI guide 4c; L6 p.38) | a sign classifier | not used; if added: `penalty=None` or a tuned C, described as such | code review |
| d | r2_score / model.score as R²_OOS (AI guide 4d; L5 p.57) | GridSearchCV default scoring, permutation_importance default | own `r2_oos`; `scoring='neg_mean_squared_error'` explicitly | grep assertion |
| e | merge on period-end instead of publication date (L5 p.58 #2) | macro used contemporaneously; x10/x73 same-year means | lag 2 for all monthly macro; x10/x73 dropped; annual/quarterly unused | lag unit test; x10 audit print |
| f | hyper-parameter or threshold chosen on the test block (L5 p.58 #4; L8 p.55) | λ, RF parameters, portfolio rule, cost level | inner folds inside training; RF untuned; rules and costs pre-registered | index asserts; PREREG hash |
| g | benchmark changed after the fact (L5 p.58 #5, p.57) | switching trailing-mean definitions | four forecast and three portfolio benchmarks all reported | PREREG hash; T10 printed before models |
| h | double-lag or missed lag of characteristics | x2/x5 pre-lagged; x1/x3/x4 stored contemporaneously | shift only x1/x3/x4 | x2 identity; perturbation test |
| i | back-filled leading values | x2/x3/x5 ≤2002-08 | NaN; sample starts 2004-01 | assert no filled cell in the model frame |
| j | stale forward fill across a regime change | x11 | rebuild x11* | T7 |
| k | duplicate or hidden-global columns | concatenating the extended file | not concatenated; x86 only | T8; matrix-rank assert |
| l | row-order merge error | lexicographic country sort | key merges only | shape and NaN asserts |
| m | target or σ̂ look-ahead | σ̂ including month t | `rolling(36).std().shift(1)` | perturbation test |
| n | trailing mean including t | `expanding().mean()` without the shift | `.shift(1)` | spot-check of 20 random (i, t) |
| o | portfolio weights using month-t information | R12 or σ̂ not shifted | shifted inputs | perturbation test; identity tests |
| p | snooping via EDA | tercile sorts or return statistics on OOS | PRE-only sorts; decisions cite PRE tables | slicing on `PRE_END` |
| q | placebo contamination | near-zero forward shifts | shifts ≥36 months both ways; identical pipeline | shift-set assert |
| r | garden of forking paths | many specifications, best reported | PREREG primary, ledger, Bonferroni | T22 |
| s | fixed universe (ex-post survivors) | the 50 assets exist through 2024 | cannot be fixed; stated as a limitation | — |

---

## Part 7 — Compute budget (the notebook must run top to bottom)

Timing basis: synthetic-noise run with the exam's shapes on this 4-core machine (no exam data):
- tuned Ridge refit (50 α × 3 folds + refit, n = 12,000, p = 47): ≈1.9 s single-job;
- tuned Lasso: ≈1.7 s;
- RF(300, leaf 200, p = 18): ≈2.4 s on 4 cores; RF predict on 600 rows: ≈0.09 s.

| block | fits | single-core estimate |
|---|---|---|
| pooled OLS / Ridge / Lasso × CHAR, CHAR+MACRO × 29 windows | 174 (116 tuned) | ≈3 min |
| Ridge CHAR+G, CHAR+C × 29 | 58 tuned | ≈1.5 min |
| per-class OLS / Ridge × 2 feature sets × 4 classes × 29 | 464 (232 tuned, small n) | ≈1.2 min |
| RF × 2 feature sets × 29 | 58 | ≈2 min |
| placebo K_main: 6 × (3 linear × 29 + RF 15 + Ridge G/C 2 × 15) | ≈800 | ≈17 min |
| placebo K_full extra: 14 × Ridge 15 | 210 tuned | ≈5 min |
| robustness R1, R2, R4 (Ridge expanding) | 45 tuned | ≈1 min |
| grouped permutation importance (Ridge, RF; 15 years × ≤17 groups × 10 repeats) | — | ≈3 min |
| bootstraps, portfolios, costs, regressions | — | < 1 min |
| **total** | | **≈35 min single core → ≈10-12 min with `joblib.Parallel(n_jobs=4)` over refit years** |

- Choices that follow from the budget:
  - annual refits rather than monthly (a monthly-refit linear variant would be ≈12× the cost and is not pre-registered);
  - RF untuned;
  - GBRT excluded;
  - full placebo set only for the primary model.
- `N_JOBS = min(4, os.cpu_count())` and a `RUN_FULL_PLACEBO = True` flag. Setting the flag to False drops about 5 min and is logged.
- Optional pickle cache (`./p3_cache/<spec>_<prereg-hash>.pkl`; pyarrow is not installed, so no parquet). The cache is ignored if the hash changes, and the submitted run starts from an empty cache.

---

## Part 8 — Specification search and the garden of forking paths

**Ledger (T22), disclosed next to the headline** (L4 p.28; L5 p.57):
- Forecast specifications, 14 × 3 schemes = **42**:
  - pooled {OLS, Ridge, Lasso, RF} × {CHAR, CHAR+MACRO} = 8;
  - Ridge {CHAR+G, CHAR+C} = 2;
  - per-class {OLS, Ridge} × {CHAR, CHAR+MACRO} = 4.
- Benchmarks: 4 forecast, 3 portfolio. All are reported, so they are not a search dimension.
- Portfolio specifications: 42 × 3 rules = **126**, each at 5 cost levels (all reported).
- Robustness R1-R5, primary specification only: **5**.
- Placebo runs: 72 + 14 = **86**. These are null draws, not candidates.
- Diagnostic only: the lag canary.
- Nothing else is tried. Any addition is appended to `DEVIATIONS` and to the ledger with its result.

**How forking is prevented.**
1. `PREREG` cell with hash, written before any fit. One named primary test per question.
2. Harness-first development:
   - all code is debugged on PRE data, using a "pseudo-OOS" of the inner validation years 2007-09;
   - benchmarks (T10, T11) are printed before models are imported;
   - the first complete OOS run is the one reported.
3. No feature, model, rule, cost or benchmark change is informed by OOS results. Bug fixes are allowed, logged, and both versions are reported.
4. Every specification in the ledger is reported, including losers. Negative R² is a result.
5. "Best of grid" claims use the Bonferroni critical value for m = 42 (3.29) or m = 126 (3.61). The primary uses 1.97.
6. The placebo distribution and the lag canary show what "signal" looks like when there is none, or when leakage is present.
7. The characteristic set is not pruned by the tercile sorts. The extended file is not screened.

---

## Appendix A — What is taught, exam-defined, or outside scope

- **Lecture-taught and used**:
  - OLS/MLR, dummies, interactions, logs;
  - ridge and lasso with standardisation, grids and time-ordered CV;
  - expanding and rolling windows; R²_OOS against the trailing mean or zero;
  - t-tests, CIs, α/β tests, partial F, Bonferroni, bootstrap (iid);
  - random forests; permutation importance; partial dependence;
  - correlation heat maps; sorts as ML; the leakage checklist.
- **[EXAM-DEFINED, not taught]**:
  - vol-scaled target; per-month trailing mean; EW / RP / TSMOM; unit gross exposure;
  - Sharpe; drawdown; turnover; net-of-cost performance;
  - 36-month circular-shift placebo;
  - cross-sectional standardisation; trailing global z; differential vs country 7;
  - tercile sorts; rolling 12-month Sharpe; "exposure vs timing".
- **Constructions of this design (flagged)**:
  - the proportional cost grid and break-evens;
  - the drift-adjusted turnover;
  - the static-vs-timing weight decomposition;
  - the future-perturbation test;
  - the ex-ante power formula (from L2 p.63-64).
- **[OUTSIDE SCOPE — only if the user approves]**:
  - HAC/Newey-West SEs (L2 p.80 names robust SEs only), Diebold-Mariano, Clark-West, block bootstrap, Sharpe-difference tests (Jobson-Korkie, Ledoit-Wolf, Lo);
  - OOB error, SHAP, XGBoost/LightGBM, neural networks;
  - the elastic-net estimator (formula only, L5 p.11), PCR/PLS;
  - EWMA volatility, mean-variance optimisation beyond the ŷ/σ̂ intuition;
  - literature beyond GKX.

## Appendix B — Open questions for the student (each with a default)

| # | question | default |
|---|---|---|
| OQ-1 | OOS start 2010-01 (72-month initial block) or 2012-01 (96 months, less power)? | 2010-01 |
| OQ-2 | Q3 headline forecast set: Ridge CHAR, or CHAR+MACRO? | CHAR headline; CHAR+MACRO reported alongside as the portfolio side of Q2 |
| OQ-3 | Cost model: single grid with a 10 bp headline, or class-specific costs? | Single grid, 10 bp headline, break-evens. Class-specific costs need a sourced number per class |
| OQ-4 | Macro lag 2 for all series, or 1 for market-observed series (x6, x86, x13)? | 2 for all; R1 = lag 1 |
| OQ-5 | Seed | 7034 (or the student ID, as in P2) |
| OQ-6 | Country macro for class A? | None (differential vs c7 ≡ 0); alternative: cross-country average |
| OQ-7 | Include tuned GBRT (validation-curve early stopping on the inner folds, *not* `n_iter_no_change`)? | No (budget, tuning-leakage risk) |
| OQ-8 | Add a time-series characteristic channel (e.g. sign(R12) or own-history z of x1/x2)? | No; TSMOM stays an external benchmark |
| OQ-9 | External data? | None |
| OQ-10 | σ̂ floor in portfolios? | None; R5 floor 0.004 |
| OQ-11 | Size of the full placebo set (20 shifts, Ridge only)? | 20 shifts; flag to drop |
| OQ-12 | Git-commit the PREREG cell before the first OOS run as a time stamp? | Suggested; the user decides |
| OQ-13 | Ex-post full-sample tercile sorts in an appendix? | Yes, clearly labelled ex post |

## Appendix C — Citation index (all checked against the notes)

- **L1**: p.33-35, 36, 37-39, 44, 45-74, 80-82
- **L2**: p.47-49, 56-57, 63-64, 72-73, 78-81, 88-97, 100-101
- **L3**: p.2-3, 12-18, 19-36, 39-44, 45, 48-49, 51-55, 57-58
- **L4**: p.9, 11-15, 25-28, 43-48
- **L5**: p.6-9, 11-24, 25-27, 35-36, 37-39, 41-47, 48-53, 55-60
- **L6**: p.30, 37-38, 46-47, 56-57
- **L7**: p.4, 6, 23
- **L8**: p.28, 35-37, 40-47, 53-62
