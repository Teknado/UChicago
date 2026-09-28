# Problem 3: final research design (synthesis)
BUSN 41210 Final, Autumn 2026. Problem 3, "Cross-country asset return prediction" (60 points).

## Status and conventions

**This is a plan only.** No model has been fitted on the exam data. No tercile sort, R², backtest, Sharpe ratio or placebo has been computed. The only things run were:
- checks of library signatures and defaults in the installed stack (Python 3.11, pandas 3.0.6, sklearn 1.9.1, scipy 1.17.1, statsmodels 0.15.0);
- one timing run on **synthetic random arrays** of panel shape;
- critical values from `scipy.stats`.

Nothing under `/home/user/UChicago` was modified. Every data fact below comes from `notes/data_profile.md`, and the notebook must **recompute and print** each one, because exam cell 1 requires "every number must come from a cell".

**Where this plan comes from.**
- Base: **Draft C**, which contributes the pre-registration hash, the benchmarks-first tables, the evaluation harness and the portfolio and attribution machinery.
- Grafted from **Draft A**: the windows, the feature engineering and trap handling, the pinned pooled benchmark, the M1 row, the non-wrapping placebo rule and the truncation and fitted-object tests.
- Grafted from **Draft B**: the written expectations, `ts_mom` and the z-score variants as robustness checks, the Bonferroni cut-off for the sorts, the banned-API list and the class-tilt diagnostic.

Every conflict between the drafts is resolved in Appendix A.

**Citation keys.**
- "L<n> p.<k>" means the PDF page, as used in `notes/Lecture_<n>.md`.
- "cell N" means a cell of `Final_Autumn_2026-1.ipynb`: 33 is the question, 34 the data and "Read this before modelling", 35 the setup code, 36 "Two rules", 37 the workflow, 38 "What to submit", 39 the project code, 40 the write-up.
- "DP §X" means a section of the data profile.
- "AIG §x" means the AI Coding Guide.

**Scope labels.**
- [LEC]: taught, with a page cited.
- [EXAM-DEFINED, cell N]: not taught in any lecture, but defined or required by the exam notebook.
- [OUTSIDE SCOPE — proposed only if the user approves]: in neither source. Off by default and never part of a headline number.

**The design in brief.**
- **Primary model.** A pooled **ridge** regression. The target is the vol-scaled next-month excess return y = r/σ̂₋₁. The predictors are five **within-class, within-month rank scores** of the lagged characteristics plus class dummies (M2).
- **Macro.** Enters as class-specific slopes on trailing-z global macro, plus country macro expressed as differentials against country 7 for classes B, C and D (M3).
- **Estimation.** Expanding window, refit every December, with the penalty chosen on three time-ordered annual validation folds inside each training window.
- **Evaluation.** A single untouched OOS window, 2011-01..2024-12, scored against the **pooled trailing mean of y updated monthly** (the pinned primary benchmark) and against zero.
- **Other models.** OLS, lasso and an untuned random forest are secondary. L8 p.62 says that in asset pricing "the signal really is weak and linear" more often than not.
- **Portfolio.** P1: w ∝ ŷ/σ̂, unit gross exposure, rebalanced monthly, net of 10 bp per unit of one-way turnover. It is compared with EW, RP and TSMOM, and attributed with an α regression on the three benchmarks.

---

## 0. Pre-registration: primary specification, OOS window, benchmarks, search budget

### (i) Goal
Fix, before anything is fitted (cell 36 rule 2), every choice that could otherwise be tuned to the OOS window: the windows, the target, the benchmarks, the primary specifications, the decision rules, the placebo set, the cost assumption and the complete list of specifications. Also state what we expect to find, and how much power the design has.

### (ii) Method steps
1. **Cell P3-0 (markdown + code).**
   - Print the `PREREG` dict below.
   - Print `hashlib.sha256(json.dumps(PREREG, sort_keys=True, default=str).encode()).hexdigest()`.
   - Initialise `DEVIATIONS = []`, where every later change is recorded as (date, what, why, both results), and set `OOS_LOCKED = True`.
2. **Benchmarks first (cell P3-6).**
   - Build the forecast-benchmark vectors and the EW/RP/TSMOM portfolios, and print T10 and T11 (§3.2, §4.2). This happens **before any model class is imported into a model cell**.
   - These are the only OOS numbers seen before the lock is released. They involve no predictor, and cell 36 rule 1 requires them first.
3. **Development on pseudo-OOS.** Every model and harness cell is debugged with a pseudo-split inside the initial block: train on 2003-01..2008-12, inner folds 2006/2007/2008, "test" 2009-01..2010-12. That is all training data, and no design decision is taken from it. While `OOS_LOCKED` is True, `score_oos()` raises.
4. **Unlock once.**
   - Set `OOS_LOCKED = False` and run the full ledger top to bottom.
   - Specification definitions are never edited after unlocking. Bug fixes are allowed, but each goes into `DEVIATIONS` and both versions are reported.
   - Headline numbers are read from `PREREG['PRIMARY']` programmatically.
5. Optional: git-commit the notebook right after P3-0 to time-stamp it (OQ-12, the student decides).

**0.1 Frozen constants (`PREREG`)**

| Key | Frozen value | Why | Grounding |
|---|---|---|---|
| `SEED` | 7034, used for RF, bootstrap and permutations | Same seed as Problem 1 | cell 3; L6 p.18 (random draws differ run to run, so set seeds) |
| `DATES` | `pd.date_range('2000-01-31','2024-12-31',freq='ME')`: 300 month-ends, `datetime64[us]` | Equals the panel dates (DP §C, §H) | — |
| `FIRST_TARGET` | 2003-01-31 (month index m = 36) | The 36-month σ̂ first exists here. Every leading back-fill (the last filled cell is 2002-08) is then out of every design matrix, even through x3's 1-month lag (DP §D). | cell 34, 36 |
| `INIT_TRAIN` | 2003-01..2010-12: 96 months, **4,800 rows** | Contains a full cycle: the 2003–07 expansion, the GFC and the 2010 recovery. The inner validation years 2008, 2009 and 2010 therefore include a crisis and its reversal. That is 600 rows per M2 coefficient. | cell 36 rule 2; L1 p.34 (nonstationarity) |
| `OOS` | 2011-01..2024-12: 168 months, **8,400 rows** (A 4,368 / B 1,344 / C 1,512 / D 1,176) | 14 years, so the SE of annual Sharpe is about √(12/168) = 0.27. It spans the euro stress of 2011, the 2014–16 commodity collapse, COVID and the 2021–23 inflation regime, which is where x11 stops. It is touched once. | L4 p.43; L5 p.48–52 |
| `SUBPERIODS` | 2011-01..2017-12 and 2018-01..2024-12 (84 months each) | Two equal halves fixed in advance | cell 37 ("sub-periods") |
| `REFIT_ENDS` | `pd.date_range('2010-12-31','2023-12-31',freq='YE')`: **14 refits**. The model fitted at b predicts the 12 months of b+1 (600 rows). | Test step of 12 and step of 12, as in L5 p.51. Characteristic premia move slowly. | L5 p.51 |
| `SCHEMES` | static (fit once at b = 2010-12); expanding (start 2003-01, end b); rolling (the 96 months ending b). All three coincide at b = 2010-12. | These are the notebook's three schemes. A fixed start for expanding (L5 p.50); fixed length for rolling (L5 p.48); a single chronological fit for static (L4 p.47–48). | cell 37; L5 p.48–50 |
| `INNER` | For an outer window [a, b] and j = 3, 2, 1: fit on [a, b − 12j months], validate on the next 12 months. Criterion: mean of the 3 fold MSEs (equal-sized folds, 600 rows each). Ties go to the larger penalty, because the grid is ordered descending and `GridSearchCV` takes the first best. Then **refit on all of [a, b]**. | Time-ordered train → validation → test | L5 p.48–49 (expanding CV), p.35 (average fold MSE), p.36 (refit on all data), p.25 (parsimony); p.52 does not refit, and that deviation is disclosed |
| `RIDGE_GRID` | `10**np.linspace(7, -1, 50)` (descending). sklearn `Ridge.alpha` equals the slide's λ with an unscaled SSE. | α/n runs from about 3e3 (null-equivalent) to about 1e-5 (OLS-equivalent) for n in 3,000–12,600 | L5 p.27, p.37–39 |
| `LASSO_GRID` | `10**np.linspace(-0.5, -5, 50)` (descending), `Lasso(max_iter=10_000)`. sklearn minimises (1/2n)SSE + α‖w‖₁. | With standardised X and var(y) ≈ 1, α_max ≈ max\|corr(x_j, y)\|, expected well below the 0.316 top, so the null model is inside the grid. The edge log (§3.5) catches it otherwise. | L5 p.18 (notes remark), p.27 |
| `RF` | `RandomForestRegressor(n_estimators=300, max_features=1/3, min_samples_leaf=200, bootstrap=True, oob_score=False, random_state=SEED, n_jobs=N_JOBS)`, **not tuned** | "Set B as large as patience allows, set a minimum leaf size, and stop" (L8 p.44); p/3 for regression (L8 p.43). With a 200-row leaf the SE of a leaf mean is about 0.07 in y units. sklearn's default `max_features=1.0` would be bagging (verified). | L8 p.43–45, p.55 |
| `SIGMA` | σ̂_{i,t−1} = sd(r_{i,t−36..t−1}, ddof=1), from `excess_return` | Shared by the target, RP, TSMOM and every forecast portfolio. It avoids x5's 0.004 floor. | cell 36; L2 p.56–57 |
| `TARGET` | y_{i,t} = r_{i,t}/σ̂_{i,t−1} | §2.10 | cell 37 |
| `MACRO_LAG` | 2 months for every monthly macro series. Robustness: 1 month (R1). | A 1-month minimum plus 1 month of release delay | cell 34 ("you may argue for more"); L5 p.58 #2 |
| `GZ` | 60-month rolling z-score, `min_periods=24`, ddof=1, computed on the stored series and then shifted 2 | "Trailing information only" | cell 37 |
| `BENCH_FCST` | **PRIMARY: pooled trailing mean of y**, updated monthly. REQUIRED: zero. SECONDARY (always reported, never used for a verdict): per-class and per-asset trailing means of y. | §0.2 | cell 36; L4 p.46; L5 p.55–57 |
| `BENCH_PORT` | EW, RP (∝ 1/σ̂), TSMOM (∝ sign(R12)/σ̂); unit gross exposure; monthly rebalancing | As specified | cell 36 rule 1 |
| `PRIMARY` | Q1: **S08** (Ridge × M2 × expanding). Q2: **S20** (Ridge × M3 × expanding) vs S08. Q3: **P1 on S08**, net of 10 bp. | One named test per question | L5 p.57; L4 p.28 |
| `PLACEBO` | Primary Q2 model S20: s ∈ {36, 48, …, 264} (20 shifts). **Decision set** {36, …, 108} (7, non-wrapping). Secondary models: {36, 60, 84, 108}. | §5.2 | cell 37 ("try more than one shift") |
| `COST_BP` | {0, 2, 5, 10, 25, 50}, per unit of one-way traded notional. **Headline 10.** | Not taught; a transparent grid | cell 37 step 4 [EXAM-DEFINED] |
| `BOOT_B` | 10,000 iid resamples of OOS months (R², ΔSharpe, α); 500 per refit for OLS coefficient SEs | Resampling the actual data | L2 p.78–80 |
| `LEAK_ALARM` | Any R²_OOS above 2% (against any benchmark) means stop and audit before interpreting | Monthly stock-level R²_OOS is typically 0.3–0.5% | L5 p.57 |
| `N_JOBS` | `min(4, os.cpu_count())` | Shared class server | §7 |
| `FLAGS` | `RUN_FULL_PLACEBO=True`, `OPTIONAL_GBRT=False`, `OPTIONAL_DECOMP=False` | §7, §10 | — |

**0.2 Benchmarks (the forecast side is pinned now; details in §3.2)**
- **Pooled trailing mean (PRIMARY).**
  - Definition: b_t = mean of y_{j,s} over all assets j and all months 2003-01 ≤ s ≤ t−1. It is updated every month and is the same for every asset in month t.
  - It is cell 36's "mean of returns through month t−1 only, updated each month … the mean of the data seen so far", applied to the variable being forecast.
  - It is exactly what an intercept-only pooled model refitted monthly would forecast, so R²_OOS against it measures what the predictors add. It is also the λ → ∞ limit of the ridge model, because the intercept is unpenalised.
  - **Why not per-asset.** A mean of y from 8–22 years per asset has an SE of about 1/√n ≈ 0.06–0.10. It is the *easiest* benchmark to beat, so using it as the headline would flatter the models (C's own ex-ante reading). It is always reported as secondary.
- **Zero** is required by cell 36 ("or zero"; L4 p.46; L5 p.56–57).
- **Per-class trailing mean.** Secondary, and used for decomposition only. It approximates what the class dummies alone deliver, so the gap between R²(S08) against the pooled mean and against the per-class mean shows how much of the headline is class premia rather than characteristics.
- "Do not quietly switch benchmarks" (L5 p.57, p.58 #5).

**0.3 Pre-registered primaries and decision rules**

| Question (cell 33) | Primary evidence | Decision rule (fixed now) |
|---|---|---|
| **Q1**: How much of next month's excess return is forecastable from lagged characteristics? | R²_OOS of **S08** against the pooled trailing mean over the 8,400 OOS rows. Reported alongside: against zero, per-class and per-asset; the bootstrap 95% CI; IS R²; by class; **ΔR² of S08 over S02** (M1, class dummies only), which is the characteristic-specific part. | "Forecastable" requires R²_OOS > 0 **and** a monthly loss-differential t > **1.974** (t₁₆₇, single pre-registered test). "Characteristics add beyond class premia" requires ΔR²(S08 − S02) > 0 with a model-vs-model t > 1.974. The magnitude is always reported, including when it is negative (L5 p.57). |
| **Q2**: Does macro, global or country-level, add anything? | ΔR² = R²(S20) − R²(S08): same scheme, windows, folds, rows and benchmark. Split into global and country with S28 (M3-G) and S29 (M3-C). Placebo distribution for S20. | "Macro adds" requires (1) ΔR² > 0, (2) the S08-vs-S20 loss-differential t > 1.974, **and** (3) the real ΔR² exceeds **all 7** decision-set placebo ΔR²s. Global versus country uses Bonferroni over the 3 macro tests (M3, M3-G, M3-C): t > **2.418**. Any other outcome is written up as "no evidence that macro adds". |
| **Q3**: Does a forecast portfolio beat EW, RP and TSMOM out of sample, after costs? | **P1 on S08**, net of 10 bp, 2011-01..2024-12 (168 months, turnover counted from 2011-02) | "Higher Sharpe than benchmark B" means net Sharpe(P1) > net Sharpe(B), always reported with the paired-bootstrap 95% CI of ΔSharpe. "Beats B beyond exposure" requires α > 0 with t > **1.975** (t₁₆₄) in the regression of net P1 on net EW, RP and TSMOM. **"Beats the simple rules after costs"** requires both: a higher net Sharpe than all three, and a significant α. Break-even costs are reported regardless. |

Family note, printed next to the three primaries: the Bonferroni value for 3 tests is 2.418.

**0.4 Search budget (full ledger in Appendix B)**
- **39 OOS-evaluated forecasting specifications:**
  - main grid, 27: M1 × OLS × 3 schemes, plus {OLS, Ridge, Lasso, RF} × {M2, M3} × {static, expanding, rolling};
  - secondary, 5 (expanding): M3-G, M3-C, M4, per-class M2, per-class M3;
  - robustness, 7 (expanding): R1–R7.
- **Critical values** (two-sided 5%, t₁₆₇):

  | number of tests m | critical t |
  |---|---|
  | 1 | 1.974 |
  | 2 | 2.262 |
  | 3 | 2.418 |
  | **39** | **3.276** |

  Any "best of the ledger" claim is judged against 3.276 (L4 p.27–28: "the fix is … a different critical value").
- **Optional, off by default:** 2 GBRT specifications (m = 41 gives 3.283).
- **Not candidates, but disclosed:**
  - 40 placebo null runs (20 for S20, plus 4 × 5 secondary);
  - 1 deliberately leaky lag canary (leakage appendix only);
  - 1 RF seed replicate.
- **Portfolios:** 7 forecast portfolios (P1/P2/P3 on S08 and on S20, plus P1 on S02 as a class-tilt diagnostic) and 3 benchmarks, each at 6 cost levels. All are reported. For α tests across the 7 forecast portfolios, the Bonferroni value (t₁₆₄) is 2.724.
- **Inside training only:** 50 α per refit (and 9 (k_G, k_C) combinations for R5).

**0.5 Expectations and power, written before computing** (commit-before-compute: L2 p.78; L4 p.26)
- **Expectations:**
  - pooled R²_OOS(S08) between −1% and +1% (L5 p.57; low SNR, L1 p.33);
  - RF no better than ridge (L8 p.62);
  - macro ΔR² ≈ 0 and inside the placebo range, partly because x1 and x4 for class B already *are* rate differentials (DP §D);
  - no significant net-Sharpe gain over RP or TSMOM, because x2 and x5 carry the same information as those rules (DP §D).
- **Power** (a flagged construction from var(X̄) = σ²/n, L2 p.63–64). Take y = s + e with var(e) ≈ 1, var(s) = R², and a perfect forecast. Then E[d] = R² and sd(d) ≈ 2R, so t ≈ R·√(N_eff·T)/2. At T = 168 and R² = 0.3%:

  | effective cross-section N_eff | t |
  |---|---|
  | 50 | 2.51 |
  | 20 | 1.59 |
  | 10 | 1.12 |

  The B/C/D blocks (within-class correlation 0.55 / 0.725 / 0.58) put N_eff well below 50. **A true 0.3% will often not reach significance**, and an estimated model does worse than a perfect one (L2 p.100–101). The write-up therefore leads with magnitudes, CIs, placebos and portfolios, and this statement goes into the Methodology section *before* any results.

### (iii) Lecture grounding
Given in the tables above. The single test set and the grey zone of validation come from L4 p.43; "never use the same data twice" from L5 p.41; the search size belongs next to the number (L5 p.57, L4 p.28).

### (iv) Assumptions
A01–A09 in §9: the windows, annual refits, the refit on train+validation, the pooled benchmark, the grids, the untuned RF, the cost level, months as the unit of testing, and the seed.

### (v) Pitfall guards
- The `PREREG` hash is printed before any fit.
- `OOS_LOCKED` makes `score_oos()` raise.
- Benchmark tables are printed before models.
- The headline is read from `PREREG`.
- Every change goes into `DEVIATIONS`.

### (vi) Outputs
- The printed `PREREG` and its hash.
- T0: timeline and row counts (13,200 modelling rows; 4,800 initial; 8,400 OOS; the per-class split).
- The printed decision rules.
- The printed Bonferroni table.

### (vii) What the write-up must cover
- ☐ The windows and why they were fixed in advance (cell 36 rule 2).
- ☐ The primaries and decision rules.
- ☐ The benchmarks and why (cell 38 Methodology).
- ☐ The search size next to the headline.
- ☐ The power statement.

### (viii) Open questions
OQ-1 (OOS start), OQ-2 (primary benchmark), OQ-5 (seed), OQ-12 (git time stamp). See §10.

---

## 1. Know your data

### (i) Goal
Cell 37 step 1 asks for:
- cumulative excess returns by class;
- a per-asset table of annualised mean, volatility, Sharpe and worst month;
- the correlation matrix ordered by class: how strong the blocks are and what that implies for pooling;
- rolling 12-month Sharpe by class and its persistence;
- the distribution, scale and persistence of each characteristic within class;
- a first look at predictive content from tercile sorts;
- a hypothesis for each of x1..x5.

Also audit every data trap before any feature is built.

### (ii)–(iii) Method steps and lecture grounding

| # | Step: exact construction | Grounding |
|---|---|---|
| 1.0 | **Load and check integrity (T1).** Assert:<br>• `panel.shape == (15000, 10)`; 300 unique month-ends equal to `DATES`; (asset_id, date) unique;<br>• 50 assets; class counts A 26 / B 8 / C 9 / D 7; every A asset in `country_7`; `info` matches the panel;<br>• `max\|wide − pivot(panel)\| == 0`;<br>• `mac_g` 300 × 5 with no NaN; `mac_c` and `mac_x` have 3,600 rows with unique (country, date) keys.<br>Print `dtypes` (strings are `StringDtype`, dates `datetime64[us]`) and the country × class table (DP §C). | AIG §6.1 ("describe your data first"), §6.4 (check against something known); L1 p.81–82 |
| 1.1 | **Trap audit (T6, T8, F6a, F6b), before any EDA:**<br>(a) leading constant runs of x2/x3/x5 from 2000-01 (expect 7 assets, all ending by 2002-08; filled = run − 1);<br>(b) x10 against the same-year and prior-year means of x86 (expect 208/300 country-years exact; corr 0.998 vs 0.839). F6a plots country_7 2020–2023: x10 = 2.228 all of 2022 while x86 goes 0.08 → 4.10;<br>(c) x11: the last non-missing date per country (c12 2020-10, c9 2021-08, c11 2024-03; 99 NaN; no interior holes); the rebuild x86 − x12 with overlap RMSE and corr per country (pooled 0.988 / 0.354; c12 0.871 / 0.78); the stale-forward-fill error after each stop (c12 mean 3.67 pp, max 10.1; c9 2.74 / 4.76; c11 0.21 / 0.39). F6b plots x11, the stale fill and the rebuild for c9, c11, c12, with stop dates marked;<br>(d) extended file: exact duplicates via `((a==b)\|(a.isna()&b.isna())).all()` (expect x73=x10, x89=x12, x114=x13, x17=x14, x92=x15, x18=x16); columns identical across all 12 countries (expect 26: x88=x6, x104=x7, x109=x8, x99=x9, x148 constant, x160 = −x143); frequency classification (the set of calendar months in which a column changes: {1} → annual, within {1,4,7,10} → quarterly, otherwise monthly or irregular; expect 39/26/82/1); the 7 gap columns with last observation per country; magnitudes above 1e6;<br>(e) x1 frozen 2024-09..12 for a8, a17, a30, a31, a34; asset_16's x5 floored at 0.004 for 2019-09..2023-03;<br>(f) series shared across countries: x10 for c2/4/6/8, x14 for c10/c11, x86 for c2 = c6. | L5 p.58 #1, #2; L3 p.40 and L4 p.14 (exact duplicates cause rank deficiency); cell 34 ("find them before you concatenate"), cell 37 step 2 |
| 1.2 | **Cumulative excess returns by class (F1).**<br>• Class EW return r̄_{c,t} = mean over i ∈ c; W = `(1+r̄).cumprod()`; log y-axis; 2000-01..2024-12; dashed `axvline` at 2003-01 and at 2011-01.<br>• Panel (b): the class average of y, cumulated from 2003-01, so classes are compared at comparable risk. | L1 p.37–38 (EDA; plot the raw series); L1 p.39 (smooth or levered track records mislead, an implicit lesson) |
| 1.3 | **Per-asset table (T2, 50 rows).** Class, country, 12·mean, √12·sd (ddof=1), Sharpe = √12·mean/sd, worst month (value and date). Add the same Sharpe computed on the pre-OOS block 2000-01..2010-12, and class medians. Check: vol runs from 0.82% (a16) to 17.40% (a47). | Sharpe is [EXAM-DEFINED, cell 37]; no lecture defines it (L2 boundaries). ddof: L2 p.56–57 |
| 1.4 | **Correlation matrix ordered by class (F2, T3).**<br>• Order by (asset_class, country in natural numeric order, asset_id); `wide[order].corr()`.<br>• `sns.heatmap(C, vmin=-1, vmax=1, cmap='RdBu_r', center=0)`, with class boundary lines; `.to_numpy(copy=True)` before any masking.<br>• T3: a 4 × 4 table of mean off-diagonal correlations within and between classes, for 2000-01..2010-12 (the version design statements cite) and for the full sample (description). Expect within A .21, B .55, C .725, D .58; D vs A −.08 and D vs C −.10. | L1 p.44 (`df.corr()`, heat map; correlated columns carry little separate information); L1 p.42 (block structure can change in crises) |
| 1.5 | **Rolling 12-month Sharpe by class (F3) and persistence (T3b).**<br>• Class EW; `rolling(12)` √12·mean/sd; 4 lines.<br>• Persistence: corr(calendar-year Sharpe in year k, in year k+1), 24 non-overlapping pairs per class. Overlapping 12-month windows would be mechanically autocorrelated. | [EXAM-DEFINED, cell 37]; correlation L1 p.44; short nonstationary samples L1 p.34 |
| 1.6 | **Characteristics by class (T4, F4).** Forecast-aligned values (x1, x3, x4 shifted +1 within asset; x2, x5 as stored) for target months 2003-01..2024-12, so no filled cell is shown:<br>• mean, sd, p1, p50, p99, min, max by class;<br>• median within-asset autocorrelation at lags 1/6/12/24;<br>• median within-class monthly cross-sectional sd;<br>• share of within-class variance explained by asset means;<br>• F4: a 5 × 4 grid of box plots with unshared axes, because units differ by up to 3 orders of magnitude for x1 and x4 (DP §D). | L1 p.37; L3 p.48 ("think about scale") |
| 1.7 | **Identification checks (T4b).** These compare predictors with **past returns or macro only**, never with the target:<br>• x2 against compounded R12_{t−1} (share with \|diff\| < 1e-5 from 2001-01; expect 14,347/14,400);<br>• x5 against SD36_{t−1} (from 2003-01; expect 13,157/13,200; exceptions are asset_16's floor);<br>• x3 against the 60-month cumulative log return (corr by class; expect about −.93/−.98/−.45/−.97 for A/B/C/D);<br>• x1 (B) against x86 − x86_c7 (≈ .96); x1 (D) against x85 − x86 (≈ .87);<br>• x4 (B) against Δ12(x86 − x86_c7) (≈ .9996).<br>Hypothesis column: x1 = carry, x2 = 12-month momentum (pre-lagged), x3 = value / 5-year reversal, x4 = 12-month change in the carry fundamental, x5 = 36-month volatility, **the risk measure**. | L1 p.44 (correlation with a scatter); AIG §6.4; DP §D |
| 1.8 | **Tercile sorts on the TRAINING BLOCK ONLY (T5, F5).**<br>• Target months 2003-01..2010-12 (96). Each month, within each class, rank the lag-aligned x_j with `rank(method='first')`.<br>• With k = n // 3: bottom k, top k, middle the rest. That gives A 8/10/8, B 2/4/2, C 3/3/3, D 2/3/2.<br>• Record the EW next-month **y** for each tercile (the primary sort metric) and raw r (descriptive, the notebook's wording).<br>• Monthly top-minus-bottom spread per class, and the class-equal-weighted average of the 4 class spreads.<br>• Mean, sd (ddof=1), t = mean/(sd/√96), against t₉₅ = 1.985 **and** the Bonferroni value for the 25 tests (5 characteristics × (4 classes + average)): **3.178**.<br>• No characteristic is dropped or re-signed because of the sorts. | L1 p.35 ("sorting returns by characteristics"); L2 p.91–93 (t of a mean); L4 p.25–28 (multiple testing); L7 p.6 (groups with similar characteristics → portfolios) |
| 1.9 | **Macro descriptives (F7a).** Time plots and AR(1)/AC12 for x6..x9 and for each country series. These are predictor-only statistics, shown on the full sample. | L1 p.37 |

**Why the sorts use the training block only.** A tercile sort is a predictor-versus-future-return statistic. Running it on 2011–2024 before the design is frozen would let OOS returns shape the features, their signs or the interactions. That breaks the untouched-test rule (L4 p.43), uses the same data twice (L5 p.41), chooses on the test block (L5 p.58 #4) and invites forking paths (L4 p.25–28). After unlocking, the same sort is recomputed on 2011–2024 and shown in an appendix labelled "ex post, used for nothing". The return-only descriptives (1.2–1.5) and predictor-only descriptives (1.6, 1.7, 1.9) contain no predictor–target link, so they are shown on the full sample as cell 37 asks. Every design statement cites the pre-OOS versions.

### (iv) Assumptions
- A10: class labels are economic *hypotheses*: A = commodities, B = currencies against the country-7 base, C = equity indices, D = government bonds (DP §C).
- A11: full-sample return and predictor descriptives are allowed.
- A12: the universe is all 50 assets, and none is dropped because of anything seen in these tables.

### (v) Pitfall guards
- `assert sort_frame.date.max() <= pd.Timestamp('2010-12-31')`.
- Back-fills are set to NaN before any EDA (§2.2).
- `.corr().to_numpy(copy=True)`, because under copy-on-write `.values` is read-only.
- No `sns.set()`.
- Heat-map order is built from `info` and checked against the panel ids.
- Merges on (country, date) keys only, never by position: `sort_values('country')` sorts lexicographically.

### (vi) Outputs
- Tables: T1, T2, T3, T3b, T4, T4b, T5, T6, T8 (audit part).
- Figures: F1, F2, F3, F4, F5, F6a, F6b, F7a.

### (vii) What the write-up must cover
- ☐ Cumulative returns by class: what they show (D steady; B about 0).
- ☐ The per-asset table and the vol range.
- ☐ Block strength and **what it implies for pooling**:
  - months, not asset-months, are the independent unit for tests and resampling;
  - pooled common slopes with class intercepts, plus a per-class check;
  - an effective cross-section well below 50.
- ☐ Persistence of class Sharpe.
- ☐ Characteristic scale, distribution and persistence (the lag-12 ACF signature of x2 and x4).
- ☐ Tercile evidence, and why it is training-only.
- ☐ A hypothesis for each of x1..x5, with evidence, naming x5 as the risk measure.
- ☐ Every trap found.

### (viii) Open questions
OQ-13: ex-post OOS sorts in an appendix. Default: yes, labelled ex post.

---

## 2. Feature engineering

### (i) Goal
Build a predictor matrix in which every value is known at the end of t−1 (cells 34, 36, 37 step 2). Standardise characteristics cross-sectionally, treat missing values using past information only (including the x11 regime-change trap and how close the rebuild is), standardise global macro with trailing information only, map country macro to assets (including the choice for class A), consider cross-sectional macro transforms, interactions, class dummies and the extended file, define the target and how forecasts become positions, and end with one predictor table with counts.

### (ii)–(iii) Method steps and lecture grounding

**2.1 Lag rules (T7).** "Shift s" means `groupby(asset_id or country).shift(s)` on a complete panel sorted by date. Assert monotone dates within each group before every shift.

| Series | Stored as | Frequency | Shift | Reason |
|---|---|---|---|---|
| x2, x5 | already lagged one month | monthly | **0** | cell 34; confirmed exactly in 1.7 |
| x1, x3, x4 | contemporaneous | monthly | **1**, within asset | cell 34 |
| x6..x9 (global) | contemporaneous | monthly | **2** | A 1-month minimum (cell 34) plus 1 month of release delay. The names are anonymised, so market series cannot be told apart from statistical releases, and a uniform rule is the conservative choice. Merge on the publication date, not the period end (L5 p.58 #2). R1 uses 1. |
| x12..x16 (curated), x86 (extended, monthly short rate) | contemporaneous | monthly (x16 a step series) | **2**, within country | as above |
| x10, and its copy x73 | January-stamped **same-year mean of x86** | annual | **not used** | Up to 11 months of look-ahead that a 1-month lag cannot remove (DP §F). Replaced by x86 at shift 2. |
| x11 | contemporaneous; stops early for c12, c9, c11 | monthly | **not used as a column** | See 2.2. The rebuild x86 − x12 is spanned by two included series. |
| every other extended column | — | 60 monthly, 39 annual, 26 quarterly, 22 irregular | **not used** (2.8) | The rule for any future use, recorded now: monthly 2; quarterly (quarter-start stamp = same-quarter mean) 5; annual (January stamp = same-year mean) 14, i.e. usable from March of Y+1 |

**2.2 Missing values and data defects (T7, second part)**

| Series / defect | Diagnosed pattern | Treatment | Why it uses only information available at the time |
|---|---|---|---|
| Leading back-fills: x3 for a1, a2, a5, a26, a21; x2 for a3, a22; x5 for a3 | a constant run from 2000-01 equal to the first genuine value; the last filled cell is 2002-08 | Set the filled cells (run − 1 per asset) to **NaN in the raw panel** before any computation. Then exclude them by the sample start: targets begin 2003-01, and x3 at shift 1 uses stored values ≥ 2002-12. Assert no filled cell is in any design matrix. | A backward fill carries the future into the past. For example, the filled x2 of asset_3 contains the returns it would "predict" (DP §D; cell 37). |
| x11 stops (c12 after 2020-10, c9 after 2021-08, c11 after 2024-03) | trailing NaN only, no holes; c12 and c9 stop **at the start of the 2021–23 inflation regime** | **Diagnose and rebuild** x11* = x86 − x12, and report the overlap quality (1.1c). **Do not add it as a column**: x86 and x12 are both in the country block, so x11* would be an exact linear combination of them, which is rank deficiency (L3 p.40; L4 p.14). Its information is fully in the model. Forward-filling is rejected: it would be wrong by up to 10.1 pp for c12 in 2022. A per-country fitted rebuild is rejected: it would be a fitted transform (L5 p.58 #1). | The rebuild uses same-month x86 and x12, both complete, lagged 2 like every other monthly series |
| x1 frozen 2024-09..12, 5 B assets | stale vendor values while the rate differential moves (a17: −0.71 → −0.04) | **Keep as stored and flag.** It affects 15 target rows (2024-10..12). An evaluation-only cut excludes them (§5.3). | A stale value is what a desk would have seen, so it is not look-ahead |
| x5 floor for asset_16 (43 months at 0.004) | provider floor; the true SD36 goes as low as 0.00298 | Keep x5 as a characteristic: its within-D rank is unaffected if it stays lowest, and the notebook prints that check. **σ̂ comes from returns, so no denominator uses the floor.** | Both quantities are trailing |
| The 7 extended gap columns | early stops, no holes | Not used (2.8) | — |
| Warm-ups (σ̂ 36 months, trailing z `min_periods=24`) | — | Covered by `FIRST_TARGET`: target 2003-01 uses global stored 2002-11, which has 35 observations | — |

Rejected treatments (cell 37): backward fill, interpolation across a gap, full-sample mean imputation.

**2.3 Cross-sectional standardisation of characteristics (primary: within-class rank)**
- For each lag-aligned characteristic j, within each (target month t, class) cell:
  - u^{(j)}_{i,t} = (rank_avg − 1)/(n_class − 1) − 0.5 ∈ [−0.5, 0.5]
  - code: `panel.groupby(['date','asset_class'])[col].rank(method='average')` (verified in pandas 3)
- **Within class, not across all 50 assets.** Units differ by class (x1 is in percent for D and a monthly decimal for B; x4 in B/C is about 1.3 against 0.004 in D). Across-asset ranks would mostly sort classes, turning characteristics into class proxies that the dummies already carry (L5 p.23; confounding, L3 p.57).
- **Rank, not z-score:**
  - robust to extremes (x3 in A has a minimum of −9.5; x2 has vendor-winsorised extremes), without deleting anything (L3 p.12–18, p.17);
  - bounded with comparable variance, so the penalty is fair (L5 p.23);
  - **has no parameters estimated across time**, so it cannot leak;
  - interpretable: the OLS slope on u is the predicted top-minus-bottom-within-class difference in y;
  - u sums to exactly zero within each class-month, so it is **orthogonal to the class dummies**. The ΔR² of M2 over M1 therefore isolates the characteristics.
- **The cost**, stated honestly: ranks discard magnitude and the class-wide time-series level of each characteristic. Two robustness checks address this:
  - **R2**: within-class z-score, sd with ddof=1, clipped at a fixed ±3 (Draft B's primary; the z formula is L7 p.23). For n = 7 the largest attainable |z| is 2.27.
  - **R6**: adds one time-series characteristic (2.7).
- Standardising within month is [EXAM-DEFINED, cell 37]. The choice is argued above.

**2.4 Global macro (x6..x9)**
- log(x6), which is VIX-type, strictly positive (10.1–62.7) and right-skewed (L3 p.48–49). x7, x8 and x9 stay as they are.
- Trailing z-score on the **stored** series, then shift 2:
  ```python
  g = mac_g.set_index('date')[['x6','x7','x8','x9']].assign(x6=lambda d: np.log(d.x6))
  gz = (g - g.rolling(60, min_periods=24).mean()) / g.rolling(60, min_periods=24).std(ddof=1)
  gz = gz.shift(2)
  ```
- Why 60 months: it adapts to slow level drift (x7 AR1 0.984; x9 AC12 0.73), and it is invariant to any affine vendor standardisation (x8 looks z-scored). "Trailing information only, never the full sample" (cell 37).
- Entry into linear models: **class-specific slopes**, gz_k × 1[class] for A, B, C, D, giving 16 columns with no common main effect (spanned). A global risk state need not move bonds and commodities the same way (corr(D, A) = −0.08; L3 p.53–54, p.57).
- RF gets the 4 uninteracted columns, because trees find interactions (L8 p.28, p.37).

**2.5 Country macro and the country-to-asset mapping**
- **Six curated country series** (a hypothesis-driven set, fixed before looking at any returns):

  | Name | Series | Meaning | Transform |
  |---|---|---|---|
  | SR | x86 | short rate | replaces the look-ahead x10 |
  | INF | x12 | inflation | — |
  | FX | x13 | YoY change in the effective exchange rate | — |
  | EPU | x14 | policy uncertainty | log (min 16.6) |
  | GPR | x15 | geopolitical risk | log (min 0.004 > 0; skewed; L3 p.48–49) |
  | RISK | x16 | risk rating | — |

  The real rate x11* = SR − INF is spanned, as in 2.2.
- **Differential against country 7** at the same date, then shift 2: d_{k,c,m} = x_{k,c,m} − x_{k,7,m}.
  - Code: build `C` by merging `mac_c` with `mac_x[['country','date','x86']]` on keys (`validate='one_to_one'`), subtract country 7's rows aligned on `date`, `groupby('country').shift(2)`, then merge to the panel on (country, date) with `validate='many_to_one'`.
  - Why country 7 is the base: cell 37 calls it "the natural base". It has no B asset, so it is the base currency, and x1 and x4 for B are (functions of) the rate differential against c7 (DP §D).
- **Mapping:**
  - Classes B, C and D use their own country's differential.
  - **Class A gets no country macro.** Under the c7 convention its differential is identically 0 (the code asserts this), so "give A country 7's macro as a differential" and "give A nothing" are the same thing. Reasons:
    - A has no natural country (cell 34);
    - a cross-country average would duplicate the global block;
    - parsimony (L4 p.23).
  - Side effect, disclosed: c7's own assets (asset_24 in C, asset_27 in D) also get zero country features. They still see the global block. R4 gives them non-zero information.
- **Entry into linear models:** d_k × 1[B], × 1[C], × 1[D], giving 18 columns. A × 1[A] column would be identically zero and is omitted. Scaling happens inside the Pipeline's `StandardScaler`, fitted on training rows only (L5 p.23, p.47).
- **RF** gets the 6 uninteracted differentials (0 for A and for c7 assets).
- Levels are not used: they are mostly country fixed effects (the cross-sectional variance share is 0.92 for x16 and 0.99 for x15; DP §F).

**2.6 Cross-sectional macro transforms considered**

| Transform | Status | Reason |
|---|---|---|
| Differential against c7 | **primary** | 2.5 |
| Deviation from the 12-country mean at the same date | **robustness R4** | Class A is still set to 0; c7 assets become non-zero |
| Rank across the 12 countries | rejected | Coarse, and tied (x10 identical for c2/4/6/8, x14 for c10/c11, x86 for c2 = c6). It also discards the size of the 2022 inflation gaps. |
| Widened panel (every country's value for every asset) | rejected | 6 × 12 = 72 extra columns per row. A multiple-testing and variance problem (L4 p.25–28; L1 p.34; L5 p.6, p.9). |

**2.7 Interactions, class dummies, time-series channel**
- **Class dummies:** `pd.get_dummies(info.asset_class, drop_first=True, dtype=float)`, giving dB, dC, dD with A as the reference. That is R−1 dummies, which avoids the dummy trap (L3 p.45, p.51–52). Common slopes with class intercepts is the L3 p.50 structure. Per-class models are the fully interacted alternative (L3 p.54).
- **Macro × class:** as in 2.4 and 2.5.
- **Characteristic × macro state (secondary spec M4, S30):** u_j × gz6 (the trailing z of log x6) for j = 1..5, giving 5 columns. Hypothesis stated now: carry and momentum premia unwind in high-uncertainty states (L3 p.53: "the effect of x_j depends on x_k").
- **Time-series channel (robustness R6, S38):**
  - `ts_mom = clip(x2 / (sqrt(12)·σ̂), −3, 3)`: the past 12-month return over the annualised trailing vol, pooled without demeaning. Both inputs are known at t−1.
  - It uses our own σ̂, not x5, so asset_16's floor never enters.
  - Its purpose: with it, the pooled model *could* learn TSMOM-type timing. That lets Q3 ask whether the forecast beats a rule it could have learned.
  - The clip is a fixed constant, not a fitted cut-off (L5 p.58 #1). The TSMOM signal itself is [EXAM-DEFINED, cell 36].
- **Trees** get the uninteracted inputs (L8 p.28, p.35–37).

**2.8 Extended file (x17..x164): hypothesis-driven use only (T8)**
- It is used for (a) the audit in 1.1(d) and (b) **x86**, the documented source of x10, of x1 (B) and of x4 (B).
- No other extended column enters any model, and **no column is ever screened against returns**. With about 150 candidates "you will essentially always find something": about 5% come out significant by chance, and in the lecture's experiment the best of 100 noise t-statistics was 2.88 (L4 p.25–28). Screening then "validating" is cheating (L5 p.41), and post-screening sets are unstable (L4 p.39, p.50).
- Rejected: a PCA of the monthly extended columns fitted within each training window (L1 p.45–74; L8 p.61). No hypothesis supports it, and it adds to the search (OQ-9).

**2.9 Own data.** None. Every added series would need a documented source and release date and would enlarge the search. The provided files already contain more candidates than can be tested honestly (cell 34 permits outside data; OQ-10).

**2.10 Target and how forecasts become positions**
- σ̂_{i,t−1} = `panel.groupby('asset_id')['excess_return'].transform(lambda s: s.rolling(36, min_periods=36).std(ddof=1).shift(1))`. Assert agreement with x5 in at least 99.6% of rows from 2003-01 (13,157/13,200); the exceptions are asset_16's floor.
- y = r/σ̂₋₁.
- **Why vol-scale:**
  - Monthly vol runs from 0.82% to 17.40%. A pooled squared-error fit on raw r would essentially model the 26 class-A assets, violating the constant-variance assumption (L3 p.2–3, p.20; L2 p.37, p.41) and inflating the variance of every estimate (var(b) ∝ σ², L2 p.71–73). "Calm months and panics do not have the same variance" (L2 p.80).
  - y is in comparable "Sharpe units".
  - It maps straight into risk-budgeted positions.
- **Why 36 months:** it matches the provider's risk characteristic (x5 = SD36), exists for all assets from 2003-01 and gives stable weights (lower turnover). The cost is slow reaction in 2008 and 2020. Vol clustering within assets remains, and that is acknowledged.
- **Back to positions:** r̂ = ŷ·σ̂, and a mean-variance rule with a *diagonal* covariance gives w ∝ r̂/σ̂² = ŷ/σ̂ (rule P1, §4). The diagonal covariance ignores the B/C/D blocks, which is disclosed. The μ/σ² intuition is standard portfolio theory and is **not taught**, so it is used as intuition only; the rule itself is cell 37's "positions proportional to the forecast scaled by volatility". With ŷ ≡ c > 0 the rule collapses to RP exactly.
- **No winsorising of y** in the primary. R3 clips the *training* y at ±4 (fixed), with evaluation always on raw y (L3 p.17, "run with and without").

**2.11 Final predictor table (T9, printed with counts by the notebook)**

| Block | Columns | Source → shift → transform | Classes | Count |
|---|---|---|---|---|
| Characteristics | u1..u5 | x1 (1), x2 (0), x3 (1), x4 (1), x5 (0) → within-class, within-month rank score | all | 5 |
| Class dummies | dB, dC, dD | asset_info; A is the reference | all | 3 |
| Global × class | gz6..gz9 × {A, B, C, D} | log x6, x7, x8, x9 → 60-month trailing z → shift 2 | all | 16 |
| Country × class | {SR, INF, FX, EPU, GPR, RISK} differential against c7 × {B, C, D} | x86, x12, x13, log x14, log x15, x16 → minus c7 → shift 2 → Pipeline scaler | B, C, D (0 for A and c7 assets) | 18 |
| Char × state (M4 only) | u_j × gz6 | product | all | 5 |
| Time-series (R6 only) | ts_mom | x2/(√12·σ̂), clipped at ±3 | all | 1 |

Model feature sets:

| Set | Contents | Count |
|---|---|---|
| **M1** | class dummies only | **3** |
| **M2** (Q1) | u1..u5 + dummies | **8** |
| M3-G | M2 + global × class | 24 |
| M3-C | M2 + country × class | 26 |
| **M3** (Q2) | M2 + global × class + country × class | **42** |
| M4 | M3 + char × state | 47 |
| M2+TS | M2 + ts_mom | 9 |
| RF-M2 | u1..u5, dB, dC, dD | 8 |
| RF-M3 | RF-M2 + gz6..gz9 + 6 uninteracted differentials | 18 |
| Per-class M2 | u1..u5 | 5 |
| Per-class M3 | A: 5 + 4 = 9; B/C/D: 5 + 4 + 6 = 15 | 9 / 15 |
| Target | y = r/σ̂₋₁ | 1 |

- **Excluded:** x10 and x73 (look-ahead); stored x11 (gaps; the rebuild is spanned); every other extended column (the 6 duplicates, 26 hidden global columns, 7 gap columns, and the annual and quarterly look-ahead columns), plus x148 (constant).
- **Assertions:** the modelling frame has 13,200 rows × all features, with **no NaN** from 2003-01. The training design has full column rank (`np.linalg.matrix_rank(Xs) == p`). `X.columns` is identical and in the same order at fit and at predict (L3 p.26).

**2.12 Causality tests (T9b; they run before any model cell)**
- **Future-perturbation test.** For t0 ∈ {2006-06-30, 2012-01-31, 2020-03-31}:
  - replace with seeded random draws every stored value dated ≥ t0 (returns; x1, x3, x4; all macro including x86), and x2/x5 dated > t0 (row t0 of x2/x5 is information from t0−1);
  - rebuild everything;
  - assert **bit-identical** features, σ̂, R12, trailing-mean benchmarks and EW/RP/TSMOM weights for every target month ≤ t0.

  This catches a missing shift, a backward fill, a full-sample z-score or a centred rolling window. It cannot catch vendor stamping (x10), which is why the 1.1 audit exists.
- **Lag unit tests.** For 20 seeded random (asset, t) pairs, assert:
  - feature(x1) = stored x1 at t−1 and feature(x2) = stored x2 at t;
  - SR differential at t = x86_c(t−2) − x86_7(t−2);
  - gz at t uses stored values through t−2 only.

### (iv) Assumptions
A13–A24 in §9: the 2-month macro lag, x10 dropped, x11 spanned, within-class ranks, the country set, the c7 differential, class A = 0, stale x1 kept, no outside data, the extended file limited to x86, the 36-month σ̂, and the diagonal-covariance position rule.

### (v) Pitfall guards
- Every transform is one of three kinds:
  - (a) same-date cross-sectional (ranks, differentials);
  - (b) trailing (σ̂, R12, gz), computed once and proved causal by 2.12;
  - (c) fitted inside the training window by a Pipeline (scalers; PCA in R5 only).
- No full-sample mean, sd, imputation or winsorisation anywhere (AIG §4b; L5 p.58 #1).
- Merges use keys with `validate=`.
- `groupby().shift()` only after an explicit sort.
- `.ffill()` never appears, because nothing is filled. No `fillna(method=)`, no `df.append`, no `.iteritems()`.

### (vi) Outputs
- Tables: T7 (lag and treatment per series), T8 (extended-file disposition with counts), T9 (predictor table), T9b (causality-test log).
- Figures: F6a, F6b, F7b (the four lagged trailing-z global series).

### (vii) What the write-up must cover
- ☐ Lag rule per variable, including the release delay and the low-frequency look-ahead fix.
- ☐ Standardisation choice and why (all four options considered).
- ☐ Missing values per series and country (pattern → treatment → why it uses only the past), including the x11 regime-change trap and the rebuild quality.
- ☐ Own data: none, and why.
- ☐ Trailing standardisation of global macro.
- ☐ Country mapping, the differential against c7, and the class-A choice.
- ☐ Cross-sectional transforms considered.
- ☐ Interactions and dummies.
- ☐ Extended-file policy with the multiple-testing argument.
- ☐ Target: why vol-scaled, which σ̂ exactly, and how forecasts become positions.
- ☐ The predictor table with counts.

### (viii) Open questions
OQ-3 (macro lag), OQ-4 (class A), OQ-6 (stale x1), OQ-7 (σ̂ window), OQ-9 (extended PCA), OQ-10 (outside data).

---

## 3. Models and evaluation schemes

### (i) Goal
Report R²_OOS against the trailing mean **and** against zero, model × scheme and by class (cell 37 step 3). Establish which scheme wins and why, which predictors carry the signal and whether that is stable across windows, whether macro adds once refitted without it (§5), how the placebo compares (§5) and whether per-class models beat the pooled one. Also explain how penalties and hyper-parameters were chosen, from training data only.

### (ii)–(iii) Method steps and lecture grounding

**3.1 Harness (cell P3-7). This is the only splitter in Problem 3.**
```python
from pandas.tseries.offsets import MonthEnd
T0, INIT_END = pd.Timestamp('2003-01-31'), pd.Timestamp('2010-12-31')
REFIT_ENDS = pd.date_range('2010-12-31', '2023-12-31', freq='YE')          # 14
def outer_window(scheme, b):
    if scheme == 'static':    return T0, INIT_END                         # one fit; predicts 2011-01..2024-12
    if scheme == 'expanding': return T0, b
    if scheme == 'rolling':   return (b - pd.DateOffset(months=95)) + MonthEnd(0), b   # 96 months
def inner_folds(dates, b):                        # dates: the window's row dates, positional
    folds = []
    for j in (3, 2, 1):
        fit_end = (b - pd.DateOffset(years=j)) + MonthEnd(0)
        val_end = (fit_end + pd.DateOffset(years=1)) + MonthEnd(0)
        fit = dates <= fit_end
        val = (dates > fit_end) & (dates <= val_end)
        assert dates[fit].max() < dates[val].min()
        folds.append((np.flatnonzero(fit), np.flatnonzero(val)))
    return folds
def test_rows(b):  return (frame.date > b) & (frame.date <= b + pd.DateOffset(years=1) + MonthEnd(0))
# each window: assert train.date.max() < test.date.min(); static reuses the b = 2010-12 model for every year
```
- Shapes:
  - initial window: 4,800 rows × p;
  - inner fits at b = 2010-12: 3,000 / 3,600 / 4,200 rows, each validated on 600;
  - largest expanding window: 12,600 rows (2003-01..2023-12);
  - rolling: always 4,800;
  - each test block: 600 × p;
  - OOS prediction frame: 8,400 rows × (date, asset_id, class, y, ŷ, spec, scheme, refit_end, chosen hyper-parameters).
- Inner folds inside a rolling window expand from that window's own start (60, 72 and 84 months).
- Features (ranks, σ̂, gz) update monthly under every scheme. Only the *parameters* are frozen between refits.
- No embargo is needed. Targets are single-month returns, so no validation or test target overlaps a training target. Overlapping features such as x2 contain only past returns.

**3.2 Forecast benchmarks, computed first (cell P3-6, T10)**
```python
s = frame.groupby('date')['y']
b_pool  = (s.sum().cumsum() / s.count().cumsum()).shift(1)                   # PRIMARY, one value per month
b_asset = frame.groupby('asset_id')['y'].transform(lambda v: v.expanding().mean().shift(1))
b_class = (frame.groupby(['asset_class','date'])['y'].agg(['sum','count'])
                .groupby(level='asset_class').cumsum().pipe(lambda d: d['sum']/d['count'])
                .groupby(level='asset_class').shift(1))                      # merge back on (class, date)
b_zero  = 0.0
```
- Unit test: `b_pool[2011-01-31] == frame.loc[frame.date.between('2003-01-31','2010-12-31'),'y'].mean()`.
- T10 is printed before models. It reports the R²_OOS of each trailing-mean benchmark against zero and against each other, pooled and by class, over 2011–2024. It shows which benchmark is hardest before any model is seen.
- **Supplementary raw-return benchmark, by class only:** forecast r̂ = ŷ·σ̂; benchmarks b_pool·σ̂ and zero. This gives the desk return-scale fit without class A dominating a pooled number.

**3.3 R²_OOS, tests and uncertainty (cell P3-9)**
```python
def r2_oos(y, yhat, bench):                       # never r2_score / model.score (AIG §4d)
    assert y.index.equals(yhat.index) and not np.isnan(yhat).any()
    return 1 - ((y - yhat)**2).sum() / ((y - bench)**2).sum()
def monthly_diff(y, ya, yb):                      # > 0: forecast a beats b in month t; 168 values
    return ((y - yb)**2 - (y - ya)**2).groupby(level='date').mean()
def diff_test(d):
    T = len(d); t = d.mean() / (d.std(ddof=1) / np.sqrt(T))
    return t, 2 * stats.t.sf(abs(t), df=T - 1), d.autocorr(1)
```
- R²_OOS = 1 − SSE_model/SSE_bench over the OOS rows (L5 p.55; L4 p.45–46). It can be negative.
- **Aggregate to one number per month before testing.** All 50 assets share each month's shocks (block correlations up to 0.725), so the unit is the month. The t-test is the L2 t of a mean of 168 differentials (L2 p.63–70 CLT; L2 p.81, p.91–93). With 50 rows every month, sign(d̄) = sign(R²_OOS).
- Report ρ₁(d). If it is material, the iid SE is too small (L2 p.80). HAC, Diebold–Mariano, Clark–West and block bootstraps are **[OUTSIDE SCOPE — proposed only if the user approves]**.
- **Bootstrap CI** for R²_OOS and ΔR²: resample the 168 months with replacement (each month keeps its whole cross-section; `rng = np.random.default_rng(SEED)`; B = 10,000), recompute the ratio of summed SSEs, and take 2.5/97.5 percentiles (L2 p.78–80). It relaxes normality but still assumes independent months.
- **Nested-model caveat.** Under the null, extra coefficients add estimation error (L2 p.100–101), so the plain test is conservative against the bigger model. The placebo (§5.2) is the like-for-like control: same parameter count, wrong timing.
- **IS R²** per refit against the training mean (L4 p.44), printed next to OOS. IS should exceed OOS (AIG §6.4; L4 p.47–48).
- **F9:** cumulative Σ_{s≤t}[SSE_bench,s − SSE_model,s] over the OOS months, showing *when* any gain accrues.
- Bonferroni values as in §0.4.

**3.4 Model menu (cell P3-8; every scaler lives inside a Pipeline)**

| Model | Exact implementation | Tuning (inner folds only) | Role and grounding |
|---|---|---|---|
| OLS | `Pipeline([('sc', StandardScaler()), ('m', LinearRegression())])` | none | The unpenalised baseline and the kitchen-sink warning (L3 p.19–26; L4 p.47–48; L5 p.6, p.9). M1 with OLS = class means (benchmark-plus). |
| **Ridge (primary)** | `GridSearchCV(Pipeline([('sc', StandardScaler()), ('m', Ridge())]), {'m__alpha': RIDGE_GRID}, cv=inner_folds(dates, b), scoring='neg_mean_squared_error', refit=True, n_jobs=1)` | 50 α | Many weak, correlated predictors ("suitable when features are weak and should still contribute", L5 p.60; L5 p.12–16). Standardise first (L5 p.23). y is not standardised because it is unit-free with var ≈ 1. Dummies are standardised and penalised, so λ → ∞ gives the pooled training mean. |
| Lasso | the same with `Lasso(max_iter=10_000)` and `LASSO_GRID`; assert `n_iter_ < 10_000`, because cell 35 silences ConvergenceWarning | 50 α | The few-strong alternative; its selection frequency across refits shows which predictors survive (L5 p.17–24, p.60; L4 p.50). The per-fold Pipeline scaler is the correct form of L5 p.47's "scale using only the training data". |
| Random forest | `RF` in §0.1: 8 inputs (M2) or 18 (M3), no scaling | **none** | Checks for nonlinearity and interactions the linear models miss. "The default answer", nearly tuning-free (L8 p.41–47, p.55, p.61). No OOB, which is not taught and would be a random fold on time data anyway. |
| Per-class ridge | the same GridSearchCV per class; inner folds from `inner_folds` on that class's dates; predictions stacked onto the 8,400 rows | 50 α per class | Separate slopes = a fully interacted pooled model (L3 p.54). Pooling raises n and s_x and cuts var(b) (L2 p.73). First-refit rows: A 2,496 / B 768 / C 864 / D 672. |
| Lasso-BIC (R7 only) | `Pipeline([('sc', StandardScaler()), ('m', LassoLarsIC(criterion='bic'))])` fitted on each whole window | BIC | Tuning by information criterion, which cell 33 allows (L5 p.42–46; L4 p.29–33). Caveat: correlated rows mean the effective n is below the row count, so log n is miscalibrated. |
| M3-PCA (R5 only) | global: `StandardScaler`+`PCA(k_G)` on the training months' gz 4-vectors (one row per month); country: `StandardScaler`+`PCA(k_C)` on the **unique (country ≠ 7, month)** training rows of the 6 differentials; scores × class indicators; sign rule: flip each component so its largest-\|loading\| is positive | (k_G, k_C) ∈ {1,2,3}² jointly with α; PCA refitted within every inner fold | Compression as "pre-treatment for further ML" (L1 p.46, p.45–74; L8 p.61). Scree plot of the initial block, for description only (L1 p.43). **Boundary flag:** standardising before PCA is not in L1 (which only demeans); it is justified by scale (L5 p.23). L1 gives no rule for k, so k is validated. |
| GBRT (**optional**, off) | `GradientBoostingRegressor(loss='squared_error', learning_rate=0.05, n_estimators=500, max_depth=d, subsample=0.8, min_samples_leaf=200, random_state=SEED)`, **never** `n_iter_no_change` (verified: it calls a shuffled `train_test_split`) | d ∈ {1, 2}; B* = argmin of the mean inner-fold validation MSE along `staged_predict`; refit with `n_estimators=B*` | L8 p.50–52 (L, ν, B tuned; the held-out curve picks B); L8 p.55 (a tuning budget is a leakage risk). Off by default for budget reasons (OQ-8). |

**Not used, and why:**
- KNN: curse of dimensionality, no handling of categoricals (L6 p.37). KNN *regression* is not taught.
- CART as a candidate: dominated by the forest (L8 p.40–41).
- Neural nets, SVM, the elastic-net estimator (formula only, L5 p.11), PCR/PLS as named estimators, XGBoost/LightGBM/HistGB, OOB, SHAP and 2-D PD: passing mentions or absent (L1 p.36; L5 p.11; L8 p.62; the L8 boundaries).
- Clustering: classes are given labels (L7 p.4).
- `LogisticRegression`: nothing is classified. If a sign classifier were ever added it would be `LogisticRegression(penalty=None)`, described as the MLE (L6 p.38; AIG §4c).

**3.5 Tuning protocol and grid-edge rule (cell P3-7)**
Per outer refit b:
1. For each grid value, fit the whole Pipeline (the scaler, and the PCA in R5) on each inner fold's fit rows, and score validation MSE. MSE is Gaussian deviance over n (L5 p.4–5, p.52).
2. Choose the argmin of the mean fold MSE. Ties go to the larger penalty.
3. Refit on all of [a, b] (L5 p.36). `GridSearchCV(refit=True)` does this; verified to refit on every row passed.
4. Predict the 12 months of b+1. "y_pred from the testing dataset has nothing to do with obtaining optimal alpha" (L5 p.52).
5. Log to T14: the chosen α (and k), its grid index, the validation-MSE curve, and IS R².

**Edge rule** (L5 p.27, p.37–39, p.53):
- The grid spans OLS-equivalent to null-equivalent, so an edge solution is interpretable. The upper edge is "shrink to the benchmark", reported as the honest no-signal answer. The lower edge is OLS.
- If the lower edge is chosen in more than 4 of 14 refits, extend the grid 2 decades down in a disclosed rerun (+1 to the search, logged in `DEVIATIONS`).

**3.6 Specification grid (T12; full IDs in Appendix B)**
T12 is the main Results table:
- rows: M1-OLS, then {OLS, Ridge, Lasso, RF} × {M2, M3};
- columns: {static, expanding, rolling} × {R²_OOS against b_pool, against 0}, plus IS R² (expanding);
- under each primary cell: t(d) and the bootstrap CI.

An appendix version adds columns for b_class and b_asset, ρ₁(d) and the number of edge solutions.

**3.7 By class; per-class against pooled (T13, T15, T16)**
- **By class:** for S08, S20, S05 (OLS-M2-E), S14 (RF-M2-E) and S26 (RF-M3-E), compute R²_OOS on each class's OOS rows with the **same** pooled benchmark (and a b_class column), plus the raw-return R² by class. The same split and the same benchmark for every model (L8 p.53).
- **By sub-period:** 2011–17 and 2018–24.
- **Per-class against pooled:**
  - S31 (per-class M2) against S08, and S32 (per-class M3) against S20;
  - on all rows and on each class's rows;
  - model-vs-model loss differential, with m = 2 Bonferroni (2.262).
  - Reading: bias against variance (L2 p.72–73). B and D have small samples and noisy tuning (L6 p.37; L8 p.40).

**3.8 Which predictors carry the signal, and is that stable? (F11–F14)**
- **Coefficient paths (F11).**
  - Standardised ridge coefficients (per 1 training SD, from `best_estimator_.named_steps['m'].coef_`) at the 14 expanding refits (S08, S20) and the 14 rolling refits (S09, S21). Style of L5 p.22.
  - Sign-stability share: the fraction of refits with the same sign as the mean.
  - OLS S05 rank coefficients with ±2 SE from a **month-resampling bootstrap** (B = 500 per refit; resample training months with replacement, keeping each month's cross-section; L2 p.78–80), next to the nonrobust formula SEs (L3 p.33–35). The formula SEs are flagged as too small because rows are cross-sectionally correlated (L2 p.80).
  - Read with the multicollinearity caveat (L3 p.39–44; L4 p.19–21).
- **Lasso selection heatmap (F12):** refits × features for S11 and S23 (L5 p.21–22; L4 p.50, "dropping one year … changes the selected sets").
- **OOS permutation importance (F13)** (L8 p.56–58; MDI is in-sample, L8 p.47, so it is not reported as importance):
  - For each OOS year, use that year's fitted model and its test rows.
  - Characteristic u_j: permute **within each (month, class) cell**, which preserves the rank marginal exactly.
  - Class dummies: permute class labels across assets within month, for the dummy columns only.
  - Each global series, or the whole global block: permute the lagged gz vector **across the 168 OOS months** (all assets in a month get the same draw), then rebuild the class interactions.
  - Each country series, or the whole block: permute each country's lagged differential across OOS months, then rebuild.
  - R = 10 repeats with `default_rng(SEED + r)`, because "a single permutation is itself a random draw" (L8 p.57).
  - Importance = mean over repeats of (SSE_perm − SSE_orig)/SSE_bench, in R²_OOS points. Shown as a total and as a group × year heat map (stability).
  - **Negative control:** permuting all five u jointly should take the characteristic contribution to about 0.
  - Custom code, because sklearn `permutation_importance` permutes single columns across all rows and defaults to `estimator.score`, i.e. R² against the test mean (verified).
  - Models: S08, S20, S14, S26.
- **Partial dependence (F14)** (L8 p.59–60), 1-D only:
  - `partial_dependence(rf, X_sub, features=[j], grid_resolution=40, percentiles=(0.05, 0.95), method='brute', kind='average')`. The defaults are `method='auto'` and `grid_resolution=100`, so both must be set (verified).
  - On a seeded 3,000-row subsample of the training rows of the first (2010-12) and last (2023-12) expanding refits of S14 and S26 (a computational approximation, disclosed).
  - Features: u1..u5, gz6..gz9.
  - It shows sign and shape. It is "what the model says, not a causal effect".

**3.9 Which scheme wins (interpretation, written as found)**
- Compare static, expanding and rolling for each model and set, with model-vs-model t-statistics between schemes for ridge M2/M3.
- Hypotheses, stated now:
  - expanding gains from more data at low SNR (L1 p.33);
  - rolling adapts to nonstationarity (L1 p.34) at a variance cost (L5 p.7–8; L8 p.40);
  - static cannot adapt.
- Evidence: L5 p.56 found expanding (−0.005) ahead of rolling (−0.019) as the *fold* scheme on its data.

**3.10 Sanity checks and headline verification**
- `LEAK_ALARM` (2%).
- IS > OOS.
- **F15 decile fit plot:** pooled OOS ŷ deciles, mean realised y against mean ŷ, with a 45° line (L3 p.58; L1 p.35). Also print sd(ŷ)/sd(y).
- **Headline recomputation:**
  - for refit 2015-12 of S08 and S20, recompute ridge from the closed form (X′X + αI)⁻¹X′(y − ȳ) on training-scaled X (L5 p.12) and assert agreement with sklearn to 1e-8;
  - recompute the headline R² with numpy from the stored arrays.
- **Harness self-tests** on the pseudo-OOS split:
  - `r2_oos(y, b, b) == 0`;
  - `r2_oos(y, y, b) == 1`;
  - N(0, 0.05²) noise forecasts give R² < 0 against zero.

### (iv) Assumptions
A01–A09 and A25–A28 in §9: annual refits, three annual inner folds, refit on the full window, the pooled benchmark, the untuned RF, months as the testing unit, the coefficients of year b applying through year b+1, and the PD subsample.

### (v) Pitfall guards
- Date-mask folds only, with asserts.
- `scoring='neg_mean_squared_error'` is always explicit, because `GridSearchCV(scoring=None)` would score R² against the *fold* mean (verified default).
- Scalers and PCA live inside the Pipeline and are refitted per fold. Assert `best_estimator_.named_steps['sc'].n_samples_seen_ == len(train_rows)`.
- One OOS row set, one benchmark vector and one `r2_oos` for every model.
- Importance and PD are computed after all forecasts are frozen, and are never used for selection.

### (vi) Outputs
- Tables: T10, T12, T13, T14, T15, T16.
- Figures: F9, F10 (R² bars, model × scheme), F11–F15.

### (vii) What the write-up must cover
- ☐ R²_OOS against the trailing mean **and** zero, model × scheme.
- ☐ By class for the models that matter.
- ☐ Which scheme wins and why.
- ☐ Which predictors carry the signal and how stable that is.
- ☐ Per-class against pooled.
- ☐ How penalties and hyper-parameters were chosen, from training data only, with the edge log.
- ☐ IS against OOS.
- ☐ Negative numbers reported as results.
- ☐ The search size next to the headline.

### (viii) Open questions
OQ-8 (GBRT), OQ-11 (refit on train + validation).

---

## 4. Portfolios

### (i) Goal
Answer Q3. Build EW, RP and TSMOM exactly as cell 36 specifies, and forecast-based rules whose formation and rebalancing are stated precisely. Compare them over 2011-01..2024-12:
- Sharpe at a minimum;
- return, volatility, drawdown, turnover;
- net of costs;
- cumulative plots;
- sub-periods.

Where there is outperformance, measure how much is exposure to the benchmark rules and how much is timing (cell 37 step 4; §5.4).

**Scope flag.** Portfolio construction, Sharpe, drawdown, turnover, risk parity, TSMOM and costs are **not taught in any lecture** (L2, L5, L8 boundaries). They are [EXAM-DEFINED]: the benchmarks by cell 36, the metrics and costs by cell 37. The cost model and the drift-adjusted turnover are *our assumptions*. The inference (t of a mean, bootstrap, α regressions) is lecture-grounded.

### (ii)–(iii) Method steps and lecture grounding

**4.1 Common machinery.** R is the date × 50 return matrix. W_t is formed from information through t−1 and held during month t.
```python
def unit_gross(raw):                               # Σ|w| = 1; an all-zero row means cash
    g = raw.abs().sum(axis=1); return raw.div(g.where(g > 1e-12), axis=0).fillna(0.0)
def port_ret(W, R):  return (W * R).sum(axis=1)
def traded(W, R):                                  # one-way traded notional per unit capital
    Rp = port_ret(W, R)
    drift = (W.shift(1) * (1 + R.shift(1))).div(1 + Rp.shift(1), axis=0)   # pre-trade weights
    return (W - drift).abs().sum(axis=1)
net = lambda W, R, c: port_ret(W, R) - c * traded(W, R)
```
- The drift formula treats excess-return positions like funded holdings, an approximation for futures-style overlays. It is stated, and it is applied equally to every strategy.
- The 2011-01 initial build is excluded from turnover and costs for **every** strategy.

**4.2 Benchmarks (cell P3-6, T11, F16a), built before any model** [EXAM-DEFINED, cell 36]
- EW: w = 1/50.
- RP: w ∝ 1/σ̂_{t−1}.
- TSMOM: w ∝ sign(R12_{t−1})/σ̂_{t−1}, with `R12 = np.expm1(np.log1p(wide).rolling(12).sum()).shift(1)` and sign(0) = 0. Assert R12 = x2 in at least 99.6% of rows from 2001-01.
- All use unit gross exposure, monthly rebalancing and `excess_return` only, with the same σ̂ as the target.
- Computed from 2003-01. The headline is 2011–2024; 2003–2010 is shown as context.
- Also report:
  - gross weight share by class (EW puts 26/50 of its weight, and most of its risk, in class A: cell 34's warning);
  - the maximum single-asset weight (asset_16's true SD36 falls to about 0.003, so RP and TSMOM concentrate in it);
  - EW's variance share by class.

**4.3 Forecast rules (pre-registered; ŷ at t comes from features ≤ t−1)**

| ID | Raw weight → `unit_gross` | Built on | Why |
|---|---|---|---|
| **P1 (primary)** | ŷ_{i,t}/σ̂_{i,t−1} | S08 (**Q3 headline**); S20 | Diagonal mean-variance (§2.10), cell 37's "proportional to the forecast scaled by volatility". **ŷ ≡ c > 0 gives RP exactly**, so RP is the natural bar. |
| P2 | sign(ŷ_{i,t})/σ̂_{i,t−1} | S08; S20 | TSMOM's shape with the model's sign: a like-for-like comparison with TSMOM (cell 37, "the sign of the forecast") |
| P3 | (ŷ_{i,t} − mean_{j∈class(i)} ŷ_{j,t})/σ̂_{i,t−1} | S08; S20 | Within-class long-short, the pure cross-sectional signal. The continuous analogue of "long the top, short the bottom of a sort" (cell 37; L1 p.36; L7 p.6). Scale-free. |
| P1-M1 (diagnostic) | ŷ^{M1}/σ̂, from S02 (class means refitted annually) | S02 | The static class-premium tilt (Draft B's "CT"), used only for attribution (§5.4) |

- **Identity self-tests** (cell P3-7):
  - P1(ŷ ≡ 1) == RP and P2(ŷ ≡ R12) == TSMOM, to 1e-12;
  - perturbing r_t leaves W_t unchanged.
- Rule parameters are fixed now and not tuned on the OOS window. This is L6 p.46–47's lesson: the threshold is fixed from costs ex ante (1/5), and the test-optimal 0.195 is only a check.

**4.4 Performance metrics (T17, F16, F17, F19, F20)** [EXAM-DEFINED, cell 37; formulas stated because no lecture defines them]
- Annualised mean = 12·mean.
- Annualised vol = √12·sd (ddof=1; L2 p.56–57).
- **Sharpe** = √12·mean/sd. Nothing is subtracted, because the returns are already excess.
- t(mean) = mean/(sd/√T) = monthly Sharpe × √T (L2 p.81). Hence **SE(annual Sharpe) ≈ √(12/168) ≈ 0.27**, stated ex ante: gaps below about 0.5 are hard to distinguish.
- Max drawdown on W_t = ∏(1 + R): max_t(1 − W_t/max_{s≤t}W_s).
- Worst month.
- Average turnover (monthly and ×12).
- Maximum |w|, average net exposure Σw, and gross share by class.
- Everything for the full OOS window and both sub-periods.
- Figures:
  - F16: cumulative log wealth, gross and net at 10 bp. A second panel scales each strategy to 10% ex-post volatility, labelled "display only; ex-post scaling". Sharpe is unchanged by scaling, and L1 p.39 warns about levered curves.
  - F17: drawdowns.
  - F19: rolling 36-month Sharpe of P1 minus each benchmark.
  - F20: average |w| by class per strategy.

**4.5 Costs (T18, F18)** [NOT TAUGHT; transparent assumption; cell 37 requires net performance]
- net_t = R_t − c·TO_t, with c in `COST_BP`, the same for every asset and charged to the benchmarks too. **Headline c = 10 bp**, a round number between typical FX/bond and commodity-futures costs. It is an assumption, not an estimate.
- Break-evens:
  - c₀ = mean(R)/mean(TO), where the net mean is zero;
  - c*_B, where net Sharpe(P) = net Sharpe(B) with both paying c, solved with `scipy.optimize.brentq` on [0, 0.02], or reported as "no crossing in 0–200 bp".
- F18: net Sharpe against c for every strategy, with crossings marked.
- Class-specific costs are OQ-14.

**4.6 Uncertainty and the Q3 verdict (T19)**
- Paired iid month bootstrap (B = 10,000, `default_rng(SEED)`) of Sharpe and of ΔSharpe(P − B) for each benchmark, with 95% percentile CIs (L2 p.78–80, with the iid-months caveat). Sharpe-difference tests such as Jobson–Korkie or Ledoit–Wolf are OUTSIDE SCOPE.
- The verdict follows §0.3 exactly, whichever way it comes out.

### (iv) Assumptions
A29–A33 in §9: self-financing excess-return positions; one proportional cost; drift-adjusted turnover; no σ̂ floor in the primary; rebalancing at month-end prices with no execution lag beyond the information lag.

### (v) Pitfall guards
- Weights use only `.shift(1)` inputs and ŷ_t built from t−1 features. The perturbation test (2.12) and the identity tests prove it.
- Gross exposure = 1 is asserted every month.
- The same 168 months and the same σ̂ for every strategy.
- Costs are charged to the benchmarks.
- No ex-post scaling in any reported statistic.
- The headline rule, cost and benchmark are never switched after results (L5 p.57, p.58 #5).

### (vi) Outputs
- Tables: T11, T17, T18, T19.
- Figures: F16, F17, F18, F19, F20.
- Appendix: all 7 forecast portfolios × 6 cost levels.

### (vii) What the write-up must cover
- ☐ How each portfolio is formed and rebalanced, with the exact formulas.
- ☐ Benchmarks exactly as specified.
- ☐ Sharpe, plus return, vol, drawdown, turnover, net of costs, cumulative plots and sub-periods.
- ☐ The cost assumption, labelled as ours, with break-evens.
- ☐ The Q3 verdict under the pre-registered rule.
- ☐ The SE of Sharpe differences.

### (viii) Open questions
OQ-14 (cost level), OQ-15 (σ̂ floor), OQ-16 (a literal tercile long-short rule).

---

## 5. Robustness, placebo, attribution

### (i) Goal
Answer Q2 rigorously (macro against no macro, global against country, placebo) and stress-test Q1 and Q3:
- alternative lags, standardisations, target clipping, macro transforms, compression, a time-series channel, IC tuning;
- evaluation-only cuts;
- portfolio floors.

Then attribute any portfolio outperformance to benchmark exposure against timing.

### (ii)–(iii) Method steps and lecture grounding

**5.1 Macro against no macro (T20)**
- **Primary:** S20 against S08 (§0.3). The same comparison is made for every model and scheme: S16–S18 against S04–S06 (OLS), S22–S24 against S10–S12 (Lasso), S25–S27 against S13–S15 (RF), S19/S21 against S07/S09.
- **Global against country:** S28 (M3-G) and S29 (M3-C) against S08, Bonferroni m = 3.
- **Interactions:** S30 (M4) against S20.
- The same rows, windows, folds, grids and seeds; a pooled monthly-differential test for each pair.
- **No in-sample partial F for the verdict.** Cross-sectionally correlated rows break its iid premise (L3 p.2–3), and the course says to judge by prediction (L4 p.28: "stop asking whether it fits and start asking whether it predicts").
- **Caveat stated ex ante:** x1 and x4 for B are near-deterministic functions of country rates (DP §D), so Q2 measures macro *beyond* what the characteristics already embed (confounding, L3 p.57).

**5.2 Placebo (T21, T21b, F21)** [EXAM-DEFINED, cell 37; honest-null logic from L4 p.26–28 and simulated-null reasoning from L2 p.78–79, p.88–93]
- **Construction:**
  - Circularly shift **every raw stored macro series** (x6..x9 as one block; x86, x12, x13, x14, x15, x16 within each country) by s months, *before* any transform: `np.roll(values_sorted_by_date, s, axis=0)`. Verified: `np.roll(a, 2)[t] == a[t−2]`, with wrap-around.
  - The same s for every series and country, so the cross-country structure is real but from the wrong date.
  - Then run the **identical** pipeline: log, trailing z recomputed on the shifted series, c7 differential, shift 2, interactions, inner-fold tuning and refits.
  - One function, `build_macro(shift=s)`. Unit test: s = 0 reproduces S20 exactly.
  - Characteristics are **not** shifted; x1 and x4 keep their embedded, correctly timed macro.
- **Shift sets:**
  - S20: s ∈ {36, 48, …, 264}, 20 shifts, all multiples of 12 (calendar stamping preserved) and at least 36 months from zero in both directions (264 is equivalent to a 36-month lead).
  - **Decision set** s ∈ {36, …, 108}, 7 shifts. For these, every placebo value used at an OOS target is real past data: the earliest OOS target, m = 132, uses stored m = 130, and 130 − 108 = 22 ≥ 0. So no wrap-around into the test window.
  - Caveat: for s ≥ 72 the 60-month trailing z at OOS dates can reach wrapped values. That affects normalisation only, and can only help the placebo, which is conservative for crediting real macro.
  - Secondary models (S17, S23, S26, S28, S29): s ∈ {36, 60, 84, 108}.
- **Persistence table (T21b).** corr(x_t, x_{t−s}) per series and shift (pooled over countries for country series), predictor-only. A shifted persistent series stays partly correlated with the truth (x9 AC12 0.73, x16 0.62), which is exactly why several shifts are run.
- **Reading, pre-registered:**
  - real macro counts only under the §0.3 rule (beating **all 7** decision-set ΔR²s);
  - also report its rank among the 21 (real + 20). A rank of 1/21 corresponds to p ≈ 0.048, approximate because neighbouring shifts are correlated;
  - the placebo also neutralises the nested-model handicap, because it has the same parameter count.
- **F21:** strip plot of placebo ΔR² with the real value marked, drawn like L4 p.27's null histogram.
- **Placebo portfolios (T25):** P1 on S20's 7 decision-set placebos next to the real P1(S20). Does any portfolio gain survive the placebo?

**5.3 Robustness (T22; Ridge, expanding; each against its parent on the same rows)**

| ID | Spec | Changed element | Parent | Grounding |
|---|---|---|---|---|
| R1 / S33 | M3, macro lag 1 | release delay | S20 | cell 34; L5 p.58 #2 |
| R2 / S34 | M2, within-class z clipped at ±3 | standardisation | S08 | cell 37; L7 p.23; L3 p.17 |
| R3 / S35 | M2, training y clipped at ±4 | outlier policy | S08 | L3 p.17 (run with and without) |
| R4 / S36 | M3-dev: deviation from the 12-country mean (A still 0) | cross-sectional macro transform | S20 | cell 37 |
| R5 / S37 | M3-PCA (k_G, k_C validated) | compression | S20 | L1 p.46; L8 p.61 (flag: correlation-matrix PCA) |
| R6 / S38 | M2 + ts_mom | time-series channel | S08 | cell 36 (TSMOM signal); L3 p.48 |
| R7 / S39 | M3, Lasso-BIC | tuning by IC | S20 / S23 | L5 p.42–46; L4 p.29–33 |

**Evaluation-only cuts** (no refit; primaries only):
- excluding 2020-03..05;
- excluding the 15 frozen-x1 rows;
- by class;
- by sub-period.

**Portfolio robustness:**
- σ̂ floored at 0.004 (the provider's own floor), applied to *all* portfolios;
- the cost grid.

**Seed replicate:** S14 rerun with seed 7035, disclosed and not a candidate.

**5.4 Attribution: exposure against timing (T23, T24)** (L2 p.47–49 single-factor form, L2 p.94–97 α and β tests, L3 p.19–36 MLR, L4 p.11–15 partial F)
- **Joint regression**, gross and net at 10 bp (net strategy on net benchmarks), over 168 months: `smf.ols('P ~ EW + RP + TS', data=oos_monthly).fit()`, with default nonrobust covariance (house style).
  - Report 12α, t(α) against t₁₆₄, the β's, and R².
  - R² is the share of variance explained by exposure to the rules. α is what they cannot replicate, i.e. timing and selection.
- **Partial F** that all β = 0: `anova_lm(smf.ols('P ~ 1', data=...).fit(), full)`.
- **Single-benchmark regressions**, because EW and RP are both long-only and collinear (L3 p.39–44). Report their correlation (L1 p.44).
- **Hand tests of β = 1** (L2 p.96–97): P1 on RP ("does P1 just lever RP?") and P2 on TSMOM ("is P2 just TSMOM?").
- **Month-bootstrap CI for α** (L2 p.80), with the dependence caveat.
- **Class-tilt test:** add P1-M1 returns as a regressor. If α vanishes, the outperformance is class premia, not characteristics.
- **Weight overlap (T24):** the monthly cross-sectional correlation of P1's weight vector with RP's and with TSMOM's, averaged over the OOS window (L1 p.44). "How much of P1 *is* risk parity?" Also the class gross-weight shares.
- **Optional, [OUTSIDE SCOPE — proposed only if the user approves]** (`OPTIONAL_DECOMP=False`): the ex-post static-versus-timing decomposition R_t = Σ w̄_i r_{i,t} + Σ(w_{i,t} − w̄_i) r_{i,t}, with w̄ the OOS-average weight. It is accounting, not a tradable rule, because w̄ uses the whole OOS window.

### (iv) Assumptions
- A34: the same s for all series and countries.
- A35: the decision set is non-wrapping.
- A36: nonrobust OLS SEs plus a month bootstrap for α.
- A37: robustness specs are fitted only under the expanding scheme.

### (v) Pitfall guards
- The placebo uses the same code path (the shift argument), and s = 0 is tested.
- Shifts are at least 36 months in both directions.
- Placebo runs are null draws: never candidates, never the headline.
- Robustness specs are pre-listed; anything added later is labelled post-hoc and counted.

### (vi) Outputs
- Tables: T20, T21, T21b, T22, T23, T24, T25.
- Figure: F21.

### (vii) What the write-up must cover
- ☐ Whether macro adds once refitted without it, global against country, and interactions.
- ☐ The placebo with several shifts, why several, and the persistence caveat.
- ☐ Which robustness checks move the answer and which do not.
- ☐ Exposure against timing (α, β, R², class tilt, weight overlap).

### (viii) Open questions
OQ-11 (placebo set size), OQ-17 (static/timing decomposition).

---

## 6. Leakage audit (cell P3-14 prints T26 with the outcome of every assertion)

| # | Risk | Where it could enter | Where the guard lives | Executable check |
|---|---|---|---|---|
| a | **FATAL: shuffled split or folds** (AIG §4a; L5 p.58 #3, p.59) | `train_test_split`, KFold, `ShuffleSplit`, `TimeSeriesSplit` (splits by row count and can cut a month), `cross_val_score` or `cross_validate` with integer cv, `RidgeCV()` (leave-one-out over rows), `LassoCV(cv=int)`, GBRT `n_iter_no_change` (a shuffled `train_test_split`, verified), HistGB `early_stopping='auto'`, RF `oob_score` | `outer_window` and `inner_folds` (§3.1) are the only splitters | Asserts `train.max < val.min` and `val.max < test.min` in every window and fold. A final cell greps the notebook source for these names and must find none in Problem 3. |
| b | **FATAL: transform fitted before the split** (AIG §4b; L5 p.58 #1, p.47) | scaler, imputer, PCA, winsor cut-offs, z-scores | Every pooled-across-dates fit sits inside `Pipeline` / `GridSearchCV` / `fit_window` on training rows (§3.4–3.5). Ranks are same-date. gz and σ̂ are trailing. x11* is a spanned identity. Clips are fixed constants. There is no imputer. | 2.12 perturbation test; assert `n_samples_seen_ == len(train)`; **fitted-object check**: perturb every raw value dated after 2010-12, refit b = 2010-12 for S08 and S20, and assert identical coefficients, scaler means, chosen α and (R5) PCA loadings |
| c | `LogisticRegression()` called "plain" (AIG §4c; L6 p.38) | a sign classifier | none is used | code review; if ever added, `penalty=None` |
| d | `r2_score` or `.score` reported as R²_OOS (AIG §4d; L5 p.57) | the `GridSearchCV` default scoring; the `permutation_importance` default | own `r2_oos`; explicit `scoring='neg_mean_squared_error'`; custom permutation code | grep for `r2_score` and `.score(`: none. `b_pool` unit test (§3.2). |
| e | merge on period end instead of publication date (L5 p.58 #2) | macro used contemporaneously; x10/x73 same-year means | the §2.1 lag table; x10/x73 dropped; annual and quarterly columns unused | lag unit tests (2.12); the x10 audit figure F6a; assert no low-frequency column is in any X |
| f | hyper-parameter, threshold or rule chosen on the test block (L5 p.58 #4; L8 p.55) | α, k, RF settings, portfolio rules, cost | inner folds inside training only; RF untuned; rules and costs in `PREREG`; `OOS_LOCKED` | `score_oos()` raises while locked; the tuning function receives no test arrays (enforced by its signature); T14 logs choices |
| g | benchmark changed after the fact (L5 p.58 #5) | switching trailing-mean definitions | pooled mean pinned as primary; all four reported; T10 printed before models | `PREREG` hash; headline read from `PREREG` |
| h | double lag or missed lag of characteristics | x2/x5 pre-lagged; x1/x3/x4 contemporaneous | shift x1/x3/x4 only | x2 = R12 and x5 = SD36 identities (1.7); lag unit tests |
| i | back-filled leading values | x2/x3/x5 up to 2002-08 | set to NaN; `FIRST_TARGET` 2003-01 | assert no filled cell in any design matrix |
| j | stale forward fill through a regime change | x11 | not used; spanned by x86 and x12 | rebuild-quality table (1.1c) |
| k | duplicate or hidden-global columns | concatenating the extended file | not concatenated; only x86 is taken | T8; `matrix_rank` assert |
| l | row-order merge error | lexicographic `sort_values('country')` | key merges with `validate=` | shape and NaN asserts after every merge |
| m | σ̂, R12 or trailing-mean look-ahead | omitting `.shift(1)`; `expanding().mean()` without a shift | `.shift(1)` everywhere | perturbation test; 20 random spot checks |
| n | portfolio weights using month-t information | unshifted R12 or σ̂ | lagged inputs only | identity tests (4.3); perturbation test |
| o | snooping through EDA | sorts or predictor–return statistics on OOS | sorts on the training block; design statements cite pre-OOS tables | `assert sort_frame.date.max() <= INIT_END` |
| p | placebo contamination | a different code path; near-zero forward shifts | one function with `shift=`; shifts ≥ 36 both ways | s = 0 reproduces S20; shift-set assert |
| q | garden of forking paths (L4 p.25–28) | many specs, best one reported | `PREREG`, ledger, Bonferroni 3.276 | T28 ledger printed; spec counter at the end |
| r | "too good to be true" (L5 p.57; AIG §6.5) | any leak | `LEAK_ALARM`; headline recomputation (3.10); negative-control permutation | **lag canary D1**: OLS-M2-E with x1/x3/x4 *unshifted*, reported only here, to show what leakage looks like |
| s | pandas 3 / sklearn 1.9 silent errors (AIG §3) | `df.append`, `fillna(method=)`, `.iteritems()`, `LassoCV(normalize=True)`, `sns.set()`; read-only `.corr().values`; `StringDtype`; `datetime64[us]` | `pd.concat`, `.ffill()`, `.items()`, a scaler in a Pipeline, `sns.heatmap` directly, `.to_numpy(copy=True)`, keys built with `pd.Timestamp` | grep for the old APIs; dtype asserts |
| t | hidden warnings | cell 35's `warnings.filterwarnings('ignore')` hides Lasso ConvergenceWarning | `max_iter=10_000` | assert `n_iter_ < max_iter` on every lasso fit |
| u | fixed universe (the 50 assets survive to 2024) | survivorship | cannot be fixed | stated as a limitation |

---

## 7. Runtime budget (the notebook must run top to bottom on the class server; cell 38)

**Measured basis.** Synthetic N(0,1) arrays on this 4-core machine, no exam data:
- tuned ridge (50 α × 3 folds + refit, n = 12,600, p = 42): **2.46 s**, single job;
- tuned lasso: **2.17 s**;
- RF (300 trees, leaf 200, p = 18, n = 12,600): **2.45 s** with `n_jobs=4` (8.9 s with 1).

Scaling assumptions: an average expanding window is about 70% of the largest, rolling windows are 4,800 rows, and M2 designs (p = 8) cost about 30% of M3.

| Block | Fits | Estimate (serial) |
|---|---|---|
| EDA, audits, features, perturbation tests (3 rebuilds) | — | ~1 min |
| OLS × {M1, M2, M3} × 29 fits | 87 | < 0.2 min |
| Ridge and Lasso × {M2, M3} × 29 (tuned) | 116 | ~3 min |
| RF × {M2, M3} × 29 | 58 | ~2 min |
| Secondary + robustness (S28–S39), 14 refits each; R5 has 9 (k) combinations | ~170 tuned | ~6 min |
| Placebo S20 × 20 shifts × 14 refits (+ a feature rebuild per shift) | 280 tuned | ~8 min |
| Placebo secondary: 4 shifts × (OLS, Lasso, RF, Ridge-G, Ridge-C) × 14 | 280 | ~5 min |
| Permutation importance (4 models × 14 years × ≤ 16 groups × 10 repeats) | — | ~5 min |
| PD (2 RF models × 2 refits × 9 features × 40 points, 3,000 rows) | — | ~2 min |
| Bootstraps (vectorised on monthly SSE sums), portfolios, costs, regressions, OLS coefficient bootstrap | — | ~1 min |
| **Total** | | **~33 min serial; ~12–15 min with `joblib.Parallel(n_jobs=N_JOBS)` over refit years and placebo shifts** (RF keeps its own `n_jobs`; the pool is not nested) |

**Decisions that follow from the budget:**
- annual refits (monthly would be 12×);
- RF untuned;
- GBRT off (it would add about 8 min at depth ≤ 2);
- the full 20-shift placebo only for S20;
- robustness under expanding only.

**Flags:**
- `RUN_FULL_PLACEBO=False` drops the 13 wrapping shifts and saves about 5 min. The decision set is unaffected, and the change is logged.
- Runtime per block is logged in T27.
- No on-disk cache in the submitted run (it must run from scratch). pyarrow is not installed, so any development cache is pickle, keyed by the `PREREG` hash.

---

## 8. Write-up outline (cell 40 onwards; cell 38's research-paper form)

| Section | Must contain (ticked against cell 38) | Tables / figures |
|---|---|---|
| **Introduction** | ☐ The three desk questions (cell 33).<br>☐ Why they matter to someone allocating across countries and asset classes: diversification across the B/C/D blocks; the systematic signals funds trade; turnover costs; evaluation relative to simple rules.<br>☐ What the literature leads us to expect, as background only: small cross-sectional predictability at low SNR (L1 p.33); ML forecast-sorted portfolios (GKX, L1 p.36); linear often wins in asset pricing (L8 p.62); monthly stock-level R²_OOS of about 0.3–0.5% (L5 p.57). Further papers (value/momentum/carry "everywhere", TSMOM) are **[OUTSIDE SCOPE — cite only if the user verifies them]**.<br>☐ A one-paragraph preview, written last, with the headline numbers and the search size. | — |
| **Data** | ☐ Panel, classes, countries; the characteristic hypotheses with evidence.<br>☐ Treatment: fills, x10/x73, x11 diagnosis and rebuild quality, frozen x1, the x5 floor, the extended-file audit, lags, standardisation, the macro mapping, class A.<br>☐ Summary statistics and structure plots. | T1–T9, F1–F7 |
| **Methodology** (a reader can reproduce the study from it) | ☐ The pre-registration table and the power statement.<br>☐ Benchmarks and why: T10 and T11 first.<br>☐ Target and position mapping.<br>☐ Models and Pipelines.<br>☐ Schemes, OOS window, refits, inner folds, grids, edge rule.<br>☐ R² definitions, tests, bootstrap, Bonferroni, placebo design.<br>☐ Portfolio rules, cost model, metric formulas.<br>☐ The L6 p.57 reporting checklist mapped onto these sections: question and target; split with dates and sizes; estimator and tuning; test performance against a benchmark; decision rule and costs. | §0 table, T14 |
| **Results** (in question order) | ☐ (1) Forecastability: T12, T13, T16, F9, F10, F15; IS against OOS; the S08 − S02 split.<br>☐ (2) Macro: T20, T21/T21b, F21, T22; signal and stability F11–F14; per-class T15.<br>☐ (3) Portfolios: T17–T19, T23–T25, F16–F20.<br>☐ Negative numbers reported as results. | as listed |
| **Interpretation and discussion** | ☐ Economics: which characteristics carry signal and whether that matches carry, momentum, value or low-risk priors; why macro does or does not help (x1/x4 already embed rates; x2/x5 duplicate the TSMOM/RP ingredients).<br>☐ Why the numbers come out as they do: the power statement; pooling against dependence; 2020/2022 extrapolation (L2 p.103–105).<br>☐ Strong against fragile evidence: sub-periods, placebo rank, CI width, robustness.<br>☐ What we tried that did not work: every row of the ledger.<br>☐ What we would do next: class-specific costs, point-in-time macro vintages, longer histories, HAC/block methods if approved. | — |
| **Conclusion** | ☐ Half a page a PM can read alone: how much is forecastable, whether macro helps, whether the portfolio beats the rules after costs and how much of it is exposure, each with one number and its uncertainty. | — |
| **Appendix** | ☐ T28 search ledger; `DEVIATIONS`; T26 leakage audit with assertion outcomes; ex-post OOS tercile sorts; lag canary; supplementary placebo shifts; T14 hyper-parameter log; T27 runtime. | — |

**Notebook cell map (inside cell 39, before the write-up in cell 40):**

| Cell | Contents |
|---|---|
| P3-0 | `PREREG`, hash, `DEVIATIONS`, `OOS_LOCKED` |
| P3-1 | load and integrity (T1) |
| P3-2 | trap audit (T6, T8, F6) |
| P3-3 | return EDA (F1–F3, T2, T3) |
| P3-4 | characteristic EDA, identification checks, training-block sorts (T4, T4b, T5, F4, F5) |
| P3-5 | features (§2), causality tests, T7, T9 |
| P3-6 | **benchmarks first** (T10, T11) |
| P3-7 | harness and self-tests |
| P3-8 | unlock; run the ledger (T12, T14) |
| P3-9 | evaluation (T13, T15, T16, F9, F10, F15) |
| P3-10 | macro and placebo (T20, T21, F21) |
| P3-11 | signal and stability (F11–F14) |
| P3-12 | portfolios (T17–T19, F16–F20) |
| P3-13 | attribution and robustness (T22–T25) |
| P3-14 | leakage audit, ledger, runtime (T26–T28) |

---

## 9. Assumptions register

| ID | Assumption | Justification | Citation |
|---|---|---|---|
| A01 | Initial block 2003-01..2010-12 (96 months); OOS 2011-01..2024-12 (168 months) | Balances power (SE of Sharpe ≈ 0.27) against training depth. Contains the GFC. Per-class training at least 672 rows. | cell 36 rule 2; L4 p.43; L5 p.48–52; L1 p.34 |
| A02 | Annual refits, every December | L5's 12-month step; slow-moving premia; budget | L5 p.51 |
| A03 | Three annual inner folds; mean fold MSE; ties go to the larger penalty | Time-ordered validation; parsimony | L5 p.48–49, p.35, p.25 |
| A04 | Refit on the whole window after tuning | Uses the most recent 3 years; deviates from L5 p.52 (disclosed) | L5 p.36 |
| A05 | Pooled trailing mean of y is the primary benchmark; zero required; per-class and per-asset secondary | "Mean of the data seen so far" for the pooled target; the intercept-only forecast; per-asset is noisiest | cell 36; L4 p.46; L5 p.55–57 |
| A06 | Grids as in §0.1, ordered descending | Span OLS to null, so the optimum can sit mid-grid | L5 p.27, p.37–39, p.53 |
| A07 | RF fixed a priori (300, p/3, leaf 200) | "RF avoids the need for CV"; the tuning budget is a leakage risk | L8 p.43–45, p.55 |
| A08 | Months are the independent unit; serial correlation is diagnosed with ρ₁, not corrected | Cross-sectional dependence; HAC is not taught | L2 p.63–64, p.80 |
| A09 | SEED = 7034 | Same as Problem 1 | cell 3; L6 p.18 |
| A10 | The economic class labels are hypotheses | Anonymised data | DP §C; cell 34 |
| A11 | Full-sample return-only and predictor-only descriptives are allowed; design cites pre-OOS versions | cell 37 asks for them; no predictor–target link | cell 37; L4 p.43 |
| A12 | The universe is fixed at 50 assets | No selection on outcomes | L5 p.41 |
| A13 | Uniform 2-month macro lag (quarterly 5, annual 14 if ever used) | 1-month minimum plus release delay; names anonymised | cell 34; L5 p.58 #2 |
| A14 | x10/x73 dropped; x86 used instead | Same-year-mean look-ahead | DP §F; L5 p.58 #2 |
| A15 | x11 not used as a column; the rebuild is diagnosed only | Spanned by x86 and x12; exact collinearity | cell 37; L3 p.40; L4 p.14 |
| A16 | Within-class, within-month rank scores | Scale-free, robust, parameter-free, orthogonal to dummies | cell 37; L5 p.23; L3 p.12–18 |
| A17 | Country set {x86, x12, x13, log x14, log x15, x16} | Hypothesis-driven: rates, inflation, FX, uncertainty, risk. Logs for skewed positive indices. | L3 p.48–49; L4 p.23 |
| A18 | Differential against country 7 | The base currency of B; cell 37's "natural base" | cell 37; DP §D |
| A19 | Class A gets no country macro (≡ 0) | No natural country; the c7 differential is 0 anyway; the average duplicates the global block | cell 34, 37; L4 p.23 |
| A20 | Global macro enters with class-specific slopes | Opposite-signed risk-state effects across classes | L3 p.53–54, p.57 |
| A21 | Stale x1 for B (late 2024) is kept | What a desk saw; not look-ahead | cell 37 |
| A22 | No outside data; extended file limited to x86 | Search size; timing documentation | cell 34; L4 p.25–28 |
| A23 | σ̂ = 36-month sd, ddof=1, from returns, unfloored | Matches x5 without its floor; stable weights | cell 36, 37; L2 p.56–57 |
| A24 | Position rule w ∝ ŷ/σ̂ with a diagonal covariance | The notebook's rule; nests RP. The μ/σ² intuition is not taught. | cell 37 |
| A25 | Year-b coefficients apply through year b+1; benchmarks update monthly | Conservative for the models | cell 36; L5 p.51 |
| A26 | y is not standardised; dummies are penalised | y is unit-free; λ → ∞ gives the pooled mean | L5 p.23, p.52 |
| A27 | PD on a seeded 3,000-row subsample | Compute; the definition averages over rows | L8 p.59 |
| A28 | Per-class inner folds use the same date rule | Consistency; small samples disclosed | L3 p.54; L6 p.37 |
| A29 | Returns are excess returns on self-financing positions | The data are excess returns | cell 34 |
| A30 | One proportional cost, 10 bp headline, uniform across assets | Not taught; transparent; break-evens let readers substitute their own | cell 37 [EXAM-DEFINED] |
| A31 | Drift-adjusted one-way turnover; the initial build is excluded | Counts EW's rebalancing trades | cell 37 [EXAM-DEFINED]; flagged construction |
| A32 | No σ̂ floor in the primary portfolios; 0.004 floor as robustness | The exam's rule as specified; concentration disclosed | cell 36; DP §D |
| A33 | Rebalancing at month-end, no execution lag beyond the information lag | Standard monthly backtest; stated | cell 36 |
| A34 | The placebo uses the same s for all series and countries | Keeps the cross-sectional structure real, only the timing wrong | cell 37; L4 p.26–28 |
| A35 | The decision set is the non-wrapping shifts {36..108} | Real past data at OOS dates | cell 37 |
| A36 | Nonrobust OLS SEs for α, with a month-bootstrap check | House style; robust SEs are named only | L2 p.80, p.94–95; L3 p.28 |
| A37 | Robustness under the expanding scheme only | Budget; the primary scheme | L5 p.50 |

---

## 10. Open questions for the student (each with a proposed default)

| # | Question | Proposed default |
|---|---|---|
| OQ-1 | OOS start 2011-01 (96-month initial block), 2010-01 (84/180), or 2013-01 (120/144)? | **2011-01** |
| OQ-2 | Primary forecast benchmark: pooled trailing mean of y, or per-asset? | **Pooled**. Per-asset, per-class and zero are always reported. |
| OQ-3 | Macro lag: uniform 2 months, or 1 for market-observed series (x6, x86, x13)? | **2 for all**; R1 = 1 for all |
| OQ-4 | Class A country macro: none, c7 levels, or the cross-country average? | **None** (≡ c7 differential = 0) |
| OQ-5 | Seed: 7034 (Problem 1) or the student-ID seed of Problem 2? | **7034** |
| OQ-6 | Stale late-2024 x1 for B: keep and flag, or rebuild from x86 − x86_c7 (scale unidentified)? | **Keep and flag**; the evaluation cut excludes the 15 rows |
| OQ-7 | σ̂ window 36 months, or 12/24 (more responsive in 2008/2020, but changes the target itself)? | **36** |
| OQ-8 | Run the optional GBRT (depth {1,2}, hand-rolled early stopping on the inner folds)? | **No** (budget; the tuning budget is a leakage risk, L8 p.55) |
| OQ-9 | One exploratory PCA spec on the cleaned extended file? | **No** (no hypothesis; enlarges the search) |
| OQ-10 | Bring in outside data? | **None** |
| OQ-11 | Refit on train + validation after tuning (default), or use the inner-train fit as L5 p.52 codes it? Full 20-shift placebo or only the 7-shift decision set? | **Refit; 20 shifts** (flag to drop to 7) |
| OQ-12 | Git-commit the notebook after P3-0 as a time stamp? | **Suggested; the student decides** (nothing is committed by this plan) |
| OQ-13 | Ex-post OOS tercile sorts in an appendix? | **Yes**, labelled "ex post, used for nothing" |
| OQ-14 | Cost model: one 10 bp headline, or class-specific (for example A 10 / B 3 / C 5 / D 3 bp)? | **Uniform 10 bp** with the 0–50 bp grid and break-evens (class-specific costs need a sourced number per class) |
| OQ-15 | Floor σ̂ in portfolio weights (0.004) to limit asset_16's concentration? | **No floor** in the primary; the floor as robustness; max \|w\| reported |
| OQ-16 | Add a literal within-class top-minus-bottom-tercile rule on ŷ? | **No** (P3 is its continuous analogue; D has 7 assets) |
| OQ-17 | Include the ex-post static-versus-timing decomposition [OUTSIDE SCOPE]? | **No**; the α/β regression answers the exposure-versus-timing question |

---

## Appendix A. Conflict-resolution log (A vs B vs C → decision)

| Topic | A | B | C | **Decision and reason** |
|---|---|---|---|---|
| OOS window / initial block | 2011–24 / 2003–10 | 2013–24 / 2003–12 | 2010–24 / 2004–09 | **2011–24 / 2003–10.** Going 168 → 180 OOS months cuts the SE by only 3.5%, while training would shrink 96 → 72 and per-class D rows 672 → 504. Starting 2004 is unnecessary with `min_periods=24`. |
| Validation block | 3 annual expanding folds | the last 36 months, one block | 3 annual folds | **3 annual folds**, mean MSE. Uses the same 36 months as B but gives three draws (L5 p.35). |
| Refit | annual | annual | annual | annual, 14 refits |
| Refit after tuning | yes | yes | yes | yes, with the L5 p.52 deviation disclosed |
| Rolling length | 96 | 120 | 72 | **96** (= the initial block, so the schemes coincide at the first refit) |
| Primary benchmark | pooled | pooled | four reported, per-asset called "the" | **Pooled pinned**; zero required; the others secondary (avoids benchmark switching) |
| Target / σ̂ | r/σ̂36 own | the same | the same | r/σ̂36 from returns, ddof=1, unfloored |
| Characteristic standardisation | within-class rank | within-class z ±3 | within-class rank | **Rank** primary; z ±3 = R2 |
| Time-series channel | own-history z (F39) | ts_mom in the primary | none (OQ) | **ts_mom as robustness R6** (1 column, own σ̂) |
| Macro compression | PCA (k tuned) primary | none | none | **None in the primary** (ridge handles 42 columns); PCA = R5 with the not-in-L1 flag |
| Country set | x86, x12, x13, log x14, log x15, x16 | sr, ts, rr, fx, epu, gpr, rat | x11*, x12, x13, log x14, log x15, x16 | **A's set.** It has the same span as C's; it avoids B's splice and the x85 redundancy with D's x1. |
| x11 | not a column (spanned) | splice then rebuild | x11* replaces for all months | **Diagnose and rebuild, not a column** (spanned) |
| Class A macro | none | 0 | 0 | 0 (identical under the c7 differential) |
| Macro × class | global × 4, country × 3 | none (linear F1 main effects) | global × 4, country × 3 | **Class-specific slopes** |
| Char × state | M4 secondary | F2 (24 terms) | inside the primary | **M4 secondary** (5 terms, gz6) |
| Model menu | OLS/Ridge/Lasso/RF, Lasso-BIC, PCR | + KNN, CART, GBRT | OLS/Ridge/Lasso/RF | **OLS/Ridge/Lasso/RF**; Lasso-BIC = R7; GBRT optional and off; KNN and CART dropped |
| Primary model | Ridge | RF | Ridge | **Ridge** (L8 p.62; interpretable; cheap placebos) |
| RF leaf | 400 fixed | tuned fractions | 200 fixed | **200 fixed** (400 gives very shallow trees on 4,800 rows) |
| Placebo shifts | 7 (≤ 108) + 13 supplementary | 4 | 6 main + 20 for the primary | **20 for S20 with the non-wrapping 7 as the decision set; 4 for the secondary models** |
| Q3 portfolio source | Ridge M2 | RF-F1 | Ridge CHAR | **Ridge M2 (S08)**; S20 portfolios as the portfolio side of Q2 |
| Q3 rule | net SR > B; α t > 1.96 | + ΔSR CI excludes 0 | net SR > B; α t > 2.26 | **net SR > all three and α t > 1.975**; the ΔSR CI is always reported (it would almost never exclude 0 with an SE of 0.27) |
| Cost | 10 bp, grid 0–50 | 10 bp | 10 bp | **10 bp**, grid {0, 2, 5, 10, 25, 50} |
| Portfolio rules | P1–P3 × 2 models | P1, P2, CT | P1–P3 | **P1–P3 on S08 and S20, plus P1-M1** (= B's CT, built from the harness) |
| Static/timing decomposition | outside scope | flagged, off | included (descriptive) | **OUTSIDE SCOPE, off** (OQ-17) |
| Sorts | training block | training block + Bonferroni | pre-OOS | **training block, y metric, Bonferroni 3.178** |
| Pre-registration | `PREREG`, OOS lock | CONFIG + expectations | `PREREG` hash, DEVIATIONS, power | **all of it combined** |
| Leakage tests | truncation + fitted-object | asserts | perturbation, identity, canary | **perturbation (3 dates) + fitted-object + identity + canary** |
| Runtime | 8–12 min (SVD fast path) | 40–45 min | 10–12 min (4 jobs) | **~12–15 min with 4 jobs, sklearn Pipelines only** (no custom fast path) |

## Appendix B. Specification and portfolio ledger (T28)

| IDs | Features | Model | Schemes | Role |
|---|---|---|---|---|
| S01–S03 | M1 | OLS | S/E/R | benchmark-plus (class means); S02 feeds P1-M1 |
| S04–S06 | M2 | OLS | S/E/R | secondary |
| S07–S09 | M2 | Ridge | S/E/R | **S08 = PRIMARY Q1 and Q3** |
| S10–S12 | M2 | Lasso | S/E/R | secondary |
| S13–S15 | M2 | RF | S/E/R | secondary (nonlinear) |
| S16–S18 | M3 | OLS | S/E/R | secondary |
| S19–S21 | M3 | Ridge | S/E/R | **S20 = PRIMARY Q2** |
| S22–S24 | M3 | Lasso | S/E/R | secondary |
| S25–S27 | M3 | RF | S/E/R | secondary |
| S28 / S29 | M3-G / M3-C | Ridge | E | global against country |
| S30 | M4 | Ridge | E | characteristic × state |
| S31 / S32 | per-class M2 / M3 | Ridge × 4 classes | E | per-class against pooled |
| S33–S39 | R1–R7 | Ridge (R7 Lasso-BIC) | E | robustness |
| S40–S41 | M2 / M3 | GBRT | E | **optional, off** |
| D1 | M2 with x1/x3/x4 unshifted | OLS | E | lag canary; leakage appendix only |
| Null runs | S20 × 20 shifts; {S17, S23, S26, S28, S29} × 4 shifts | — | E | 40 placebo runs; never candidates |
| Portfolios | EW, RP, TSMOM; P1/P2/P3 × {S08, S20}; P1-M1 | — | — | 3 + 7, each × 6 cost levels + σ̂-floor robustness; placebo P1 × 7 |

That is 39 candidate forecasting specifications (41 if GBRT is approved). Bonferroni t₁₆₇: 3.276 (3.283).

## Appendix C. Scope ledger

- **Lecture-taught and used:**
  - EDA, the correlation heat map and PCA (L1);
  - OLS/MLR, t-tests, CIs, the bootstrap, α/β tests (L2);
  - diagnostics, logs, dummies, interactions, multicollinearity, the fit plot (L3);
  - train/validation/test, IS vs OOS R², Bonferroni, IC, partial F (L4);
  - ridge, lasso, standardisation, grids, time-series CV with expanding and rolling windows, R²_OOS against a trailing mean or zero, the leakage checklist (L5);
  - the ex-ante cost threshold idea and the reporting checklist (L6);
  - known labels ≠ clustering (L7);
  - RF, permutation importance, 1-D PD, boosting as an option (L8).
- **[EXAM-DEFINED, not taught]:**
  - vol-scaled target; per-month trailing-mean benchmark;
  - EW/RP/TSMOM; unit gross exposure; monthly rebalancing;
  - Sharpe, drawdown, turnover, net-of-cost performance;
  - cross-sectional standardisation within month; trailing global z; differential against country 7;
  - tercile sorts; rolling 12-month Sharpe;
  - the circular-shift placebo; "exposure against timing".
- **Constructions of this design (flagged):**
  - the proportional cost grid and break-evens;
  - drift-adjusted turnover;
  - the perturbation and fitted-object tests;
  - the ex-ante power formula (from L2 p.63–64);
  - `ts_mom`;
  - grouped permutation (the L8 p.57 algorithm applied to column sets);
  - correlation-matrix PCA in R5.
- **[OUTSIDE SCOPE — proposed only if the user approves]:**
  - HAC/Newey–West, Diebold–Mariano, Clark–West, block bootstrap, Sharpe-difference tests;
  - the ex-post static/timing decomposition;
  - literature beyond GKX;
  - OOB, SHAP, XGBoost/LightGBM/HistGB, neural nets;
  - KNN regression, 2-D PD, the elastic-net estimator, PCR/PLS as named estimators;
  - EWMA volatility, full mean-variance optimisation.
