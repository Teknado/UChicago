# AI Coding Guide audit of `Final_Autumn_2026-1.ipynb`

Scope: every code cell of Problems 1–3 (cells 2–94) against every item in `AI_Coding_Guide.pdf` (3 pages, read in full, PDF and text),
plus the technical items in the brief. Evidence is `cell N` (0-based, as in `nb_text.md`) with the line quoted.
The notebook was **not executed**. API behaviour was probed with small snippets in `scratchpad/audit/tmp/api_probe.py`
(this machine: Python 3.11.15, pandas 3.0.6, scikit-learn 1.9.1, numpy 2.4.6, matplotlib 3.11.2, seaborn 0.13.2, statsmodels 0.15.0, 4 cores).
The stored run executed cleanly: execution counts 1..58 run in order, with no `error` outputs and no `stderr` streams (checked in the .ipynb JSON).

Status key: PASS / FAIL / PARTIAL / N/A. Severity (for anything not PASS): FATAL / MAJOR / MINOR / NIT.

---

## Part A. The guide as a numbered checklist

| # | Guide § | Rule / pitfall / step (paraphrased; quotes where they matter) |
|---|---|---|
| G0.1 | Intro p.1 | "You must be able to defend every line you submit." "The AI wrote it" is not a defence. |
| G0.2 | Intro p.1 | Be ready for "why is there a `shuffle=True` there?" (it will cost marks). |
| G1.1–1.6 | §1 | Fine to use AI for syntax, plotting, tracebacks, explaining given code, explaining statistics, translating. (Permissions: nothing to audit.) |
| G2.1 | §2 | Always check the *method* it proposes: e.g. a random forest where a time-ordered split is needed, or a t-test where observations are not independent. |
| G2.2 | §2 | Always check claims about *your data*: column names, "assume a date is a date", row counts, numbers stored as strings (`' 10.65%'`). |
| G2.3 | §2 | Always check *validation design* ("the big one"). |
| G2.4 | §2 | Always check *interpretation*: whether an R² is good depends on the benchmark. |
| G2.5 | §2 | Always check *library versions*: code written for pandas 1.x; course runs pandas 3. |
| G3.1 | §3 | `df.append(...)` → AttributeError (removed pandas 2.0); use `pd.concat`. |
| G3.2 | §3 | `s.fillna(method='ffill')` → TypeError; use `s.ffill()`. |
| G3.3 | §3 | `s.iteritems()` → AttributeError; use `s.items()`. |
| G3.4 | §3 | `LassoCV(normalize=True)` → TypeError; put a `StandardScaler` in a `Pipeline`. |
| G3.5 | §3 | `sns.set(...)` / very old matplotlib kwargs → deprecation noise. |
| G3.6 | §3 | Implied mechanism: these "crash, you notice, you fix them" – i.e. errors/warnings must be visible. |
| G4a | §4(a), §7 | `train_test_split` (default `shuffle=True`) or any shuffled split/fold on time-ordered data. **Graded fatal.** |
| G4b | §4(b), §7 | Fitting a scaler, imputer or PCA (any transformation) before the split. Right shape: a `Pipeline` fitted after the split so the transformation only sees training rows. **Graded fatal.** |
| G4c | §4(c) | `LogisticRegression()` is L2-penalised at C = 1.0 by default; never call it "plain"/"unregularised"; ask for the textbook estimator explicitly. |
| G4d | §4(d) | `r2_score(y_test, y_pred)` is *not* the course's out-of-sample R²: it uses the test-set mean; score against a benchmark fixed before the split. |
| G5.1 | §5 | Use the tool to learn the statistics (SEs, coefficients, negative R²_OOS). |
| G5.2 | §5 | Ask "which lines could be leaking, and why?" |
| G5.3 | §5 | Ask "three ways this result could be wrong". |
| G5.4 | §5 | Before submitting: "pretend you are a sceptical referee. What would you attack first?" |
| G6.1 | §6.1 | Describe the data before asking for code (`df.head()`, `df.dtypes`). |
| G6.2 | §6.2 | One step at a time: "A 60-line block you cannot read is worse than five lines you can." |
| G6.3 | §6.3 | Read every line before running it. |
| G6.4 | §6.4 | Check against what you know: row counts, coefficient signs vs the lecture, in-sample fit > out-of-sample fit. |
| G6.5 | §6.5 | Verify the thing that matters: what must be true for the headline number to be believable; check it. |
| G6.6 | §6.6 | When it is confidently wrong, give it the evidence (the TypeError/traceback). |
| G7.1 | §7 | §4(a) and §4(b) are graded fatal in every assignment; AI code makes them more often (cross-sectional patterns). |
| G7.2 | §7 | The write-up must say what you tried and rejected. |
| G7.3 | §7 | The write-up is graded as well as the answer; a number you cannot explain is worth less. |
| G8 | Closing line | Check everything it says about *your data*, *your split* and *whether a number is good*; never submit a line you could not defend. |

---

## Part B. Guide checklist applied to the notebook

| # | P1 (cells 3–24) | P2 (cells 25–43) | P3 (cells 50–96) | Evidence |
|---|---|---|---|---|
| G0.1 | PASS | PASS | PARTIAL (MINOR) | Code is commented and asserted throughout, but several P3 cells are very long (see G6.2), and one "executable check" cited in the leakage audit cannot fail (Finding M3). |
| G0.2 | PASS | N/A | PASS | `cv5 = StratifiedKFold(5, shuffle=True, random_state=7034)` is the exam's own line (cell 4, unchanged from commit 174f164); cell 24 explains why it is valid for people and wrong for months. No shuffle anywhere in P3. |
| G2.1 | PASS | PARTIAL (MINOR) | PASS | P2 cell 41: iid resampling of 1,511 time-ordered daily returns for VaR SEs (`rv_all[rb.integers(0, len(rv_all), size=(500, len(rv_all)))]`); the caveat is stated in cell 43 ("iid resampling ignores the clustering … so the SE is optimistic"). P3 resamples whole months because assets share month shocks (cell 71 `BOOT_IDX`); serial dependence is only disclosed (Finding M5). |
| G2.2 | PASS | PASS | PASS | P1 cell 6 prints dtypes, class counts, duplicates. P2 cell 41 prints `dj.shape`, dtypes; the int `date` (20160104) is parsed with `format='%Y%m%d'` (cell 42); `MrkRet` constancy within a date is asserted. P3 cells 54–57: integrity asserts, back-fills, frozen `x1`, `x5` floor, `x10` look-ahead, `x11` gaps, duplicate and hidden-global columns in the extended file. Update frequency of the macro series actually used (x6–x9, x12–x16, x86) re-checked here: all monthly (x16 changes in 62% of months); none is an annual same-year average like `x10`. |
| G2.3 | PASS | N/A | PASS | See G4a/G4b and Part D. |
| G2.4 | PASS | PASS | PASS | P1 accuracies against the 64.25% "nobody purchases" bar (cell 6). P3 R²_OOS always against trailing benchmarks and zero (cells 68, 75). |
| G2.5 | PASS | PASS | PASS | pandas-3 idioms: `include_groups=False` (cell 56), `freq='ME'` (cell 51), `.ffill()` (cell 56), `str` dtypes handled. Probes: no warnings from any pandas/sklearn call used. |
| G3.1 | PASS | PASS | PASS | No `DataFrame.append`; only `list.append`; frames built with `pd.concat` / `pd.DataFrame(rows)` (e.g. cell 7 `pd.concat([base_row, show])`). |
| G3.2 | N/A | N/A | PASS | Only `.ffill()` (cell 56 `d.x11.ffill()`), used in a diagnostic, never as a feature. |
| G3.3 | PASS | PASS | PASS | No `iteritems`; `ledger.iterrows()` (cell 74) and `.items()` only. |
| G3.4 | N/A | N/A | PASS | No `*CV` estimator, no `normalize=`. Scaler fitted inside the fit rows (cell 71 `self.sc = StandardScaler().fit(Xd)`). Not an sklearn `Pipeline`, but equivalent (NIT). |
| G3.5 | PASS | PASS | PASS (NIT) | No `sns.set()` (comment in cell 6). The one seaborn call (cell 59 `sns.heatmap`) emits a matplotlib-3.11 `PendingDeprecationWarning` ("set_bad … will be deprecated") in this environment (probe); it is hidden by the exam's warnings filter. |
| G3.6 | PASS | PASS | PARTIAL (MINOR) | cell 46 (exam's own setup cell) `warnings.filterwarnings('ignore')` silences every warning for all of P3, and for any P1/P2 cell re-run afterwards. Kept, and disclosed in the write-up ("The exam's setup cell silences warnings. We kept it"). Finding M1. |
| G4a | PASS (cross-section) | N/A | PASS | P1: `StratifiedKFold(shuffle=True)` (cell 4) and `train_test_split(..., stratify=ya)` (cell 19) on 400 unrelated people, with unique `User ID` (checked). Legitimate and exam-mandated. P3: no `train_test_split`, `KFold`, `TimeSeriesSplit`, `GridSearchCV` or `*CV` estimator (grep of all code cells). The only splitter is `window()` / `inner_folds()` (cell 71), with `assert m[fit].max() < m[val].min()` and `assert m[win].max() < m[test].min()` at every fold and refit. |
| G4b | N/A (trees, no transforms) | N/A | PASS | Scaler, within-class demeaning and PCA (SVD) all fitted on the fit rows inside `LinearFE.fit` (cell 71), which only ever receives `X[fit]` or `X[win]`. Ranks use one date; z-scores are trailing; no statistical imputation. Truncation causality test (cell 66): all 64 feature columns for the 4,800 rows up to 2010-12 are identical when later data are deleted. |
| G4c | N/A | N/A | N/A | No `LogisticRegression` anywhere. Environment note: in sklearn 1.9.1 `penalty` is `'deprecated'`, `l1_ratio=0.0`, `C=1.0`, so the default is still L2 at C = 1. |
| G4d | N/A (accuracy) | N/A | PASS | No `r2_score`. Own `r2_oos(y, yhat, bench)` (cell 68) against trailing benchmarks; `IS R2` is labelled in-sample; `permutation_importance(scoring='neg_mean_squared_error')` (cell 71) avoids the default regressor score, which is `r2_score`. |
| G5.2–5.4 | PASS | PASS | PASS | Independent referee review recorded (cell 95 item 4; METHODOLOGY_LOG §E.6); leakage audit table (cell 91). |
| G6.1 | PASS | PASS | PASS | cells 6, 41, 46 (`panel.head()`), 54 (dtypes printed). |
| G6.2 | PASS | PASS | PARTIAL (MINOR) | Non-blank lines per code cell: cell 71 = 131, cell 64 = 87, cell 94 = 68, cell 51 = 62, cell 69 = 60, cells 40/83/56 > 50. Structured (functions, docstrings), but well beyond the guide's 60-line example. Finding M4. |
| G6.3 | PASS | PASS | PASS | Not observable directly; the code has no dead or unexplained lines found in this audit apart from unused exam imports (`sm`, `Lasso`; NIT). |
| G6.4 | PASS | PASS | PASS | Row counts asserted (cells 6, 41, 54, 65, 68). IS > OOS printed for 38/38 specs (cell 75). Coefficient signs tabulated (cell 77, Table 3.15). P1 training-fold vs CV accuracy (cell 7). |
| G6.5 | PASS | PASS | PASS | P3: power statement (cell 52/75), leak alarm (cell 75), placebo (cell 83), and the Q3 headline decomposed into the class-means tilt (cell 88, R² 0.998 with M1 as regressor). |
| G6.6 | N/A | N/A | N/A | Process item; no visible tracebacks. |
| G7.1 | PASS | N/A | PASS | See G4a/G4b. |
| G7.2 | PASS | PASS | PASS | Write-up "What we tried that did not work", "What we did not try", §3.10 deviations. |
| G7.3 | PASS | PASS | PASS | All answers explained in markdown. |
| G8 | PASS | PASS | PASS | Data (G2.2), split (G4a/b), and benchmarking (G2.4, G4d) all checked. |

---

## Part C. Technical checklist from the brief

| # | Item | Status | Evidence |
|---|---|---|---|
| T1 | `train_test_split` | PASS | Only cell 19 (P1, cross-sectional, `random_state=7034, stratify=ya`). Default `shuffle=True` confirmed by probe. |
| T2 | `KFold` / `StratifiedKFold` / `shuffle` | PASS | Only `cv5` in cell 4 (P1, exam's line). `KFold().shuffle` defaults to False (probe), not used. |
| T3 | `cross_val_score` / `cross_validate` / `cross_val_predict` | PASS | Cells 4, 7, 12, all P1 with `cv=cv5`. |
| T4 | every `*CV` estimator | PASS | None in the notebook (METHODOLOGY_LOG P3-03 rejects `RidgeCV`/`LassoCV` as row-wise folds). |
| T5 | `TimeSeriesSplit` | PASS (not used) | Custom month-mask splitter instead (a row-based `TimeSeriesSplit` on a stacked panel would split inside months). |
| T6 | `GridSearchCV` | PASS (not used) | — |
| T7 | `fit(` / `fit_transform(` on data including test rows | PASS | No `fit_transform` anywhere. P1 all-data fits are display/diagnostic only: `full_unpruned.fit(Xa, ya)` (cell 7, "diagnostic only"), `tree_star` (cell 11, "CV chose a size, not a particular tree"), `rf_full` (cell 15, reports settings). P3 fits only on `X[fit]`/`X[win]` (cell 71); unit tests fit on training rows only (cell 72 `tr = F.m <= INIT_END_M`). |
| T8 | `StandardScaler` | PASS | cell 71 only, on fit rows. The accompanying `assert self.sc.n_samples_seen_ == len(Xd)` is tautological (Finding M3). |
| T9 | PCA | PASS | PCR via SVD of the fit rows inside `LinearFE.fit` (cell 71); sklearn `PCA` only in the unit test on training rows (cell 72). `svd_solver='auto'` resolves to deterministic `covariance_eigh` for 4,800 × 42 (probe). |
| T10 | imputers | PASS (NIT) | No imputer. Constant fills only: `trailing_z(lagged[x]).fillna(0)` for the secondary `l1..l5` (cell 64), `F[xcols].fillna(0.0)` for the 2003-01 rows of the `x10` 14-month extra (cell 64), `unit_gross(...).fillna(0.0)` = cash (cell 69). None is fitted, so no leak; but cell 91's "Nothing is imputed" is imprecise (Finding N1). |
| T11 | `r2_score` | PASS | Not used. |
| T12 | `.score(` | PASS | cells 7, 11, 19: classification accuracy, labelled diagnostic. |
| T13 | `LogisticRegression` + penalty | N/A | Not used. |
| T14 | GradientBoosting `n_iter_no_change` | PASS | cell 15 `GradientBoostingClassifier(n_estimators=100, random_state=7034)`; default `n_iter_no_change=None` (probe), so no hidden random validation split; `subsample=1.0`. Boosting not used in P3 (METHODOLOGY_LOG P3-12). |
| T15 | `permutation_importance` scoring | PASS | P1 cell 19 `scoring='accuracy'` on the held-out 30%, `n_repeats=50, random_state=7034`. P3 cell 71 `scoring='neg_mean_squared_error'` on each test year, `random_state=P3_SEED`. NIT: `importances_std` is `np.std` with ddof = 0 (sklearn source), reported as "sd over 50 shuffles" (cell 19). |
| T16 | seeds / global numpy RNG | PASS (NIT) | No `np.random.seed` or global-RNG call (grep). P2: named `SeedSequence(SEED).spawn` streams (cell 30). P3: `default_rng(P3_SEED)` for spot checks (cell 66) and again for `BOOT_IDX` (cell 71) – same seed, two uses (NIT). All sklearn randomness seeded. RF `n_jobs` 1 vs 4: max prediction difference 3e-16 (probe). |
| T17 | `ddof` | PASS | Every `.std`/`.var` passes `ddof=1` except pandas `.var()` in a variance ratio (cell 82; pandas default ddof = 1 anyway). P2's unbiased variance uses `var(ddof=1)` as the manual requires (cells 30, 34, 40). |
| T18 | groupby / apply | PASS | cell 56 `groupby('country').apply(..., include_groups=False)` (pandas-3 safe); cell 60 apply grouped by an index-year array (no group columns). `b_asset` via `groupby('asset_id').y.transform(lambda v: v.expanding().mean().shift(1))` relies on F's time order within asset; unit-tested (cell 68). No warnings in probes. |
| T19 | chained assignment / copy-on-write | PASS | No `df[a][b] = …` pattern. In-place edits are on explicit copies (`W.copy()` cells 55, 64; `to.loc[idx].copy()` cell 69; `S_TO[k].copy()` cell 88). Under CoW `Series.to_numpy()` is read-only (probe); no code writes into such an array. |
| T20 | dtypes (`str`, `datetime64[us]`) | PASS | cell 54 prints `datetime64[us]` and `str`. In pandas 3.0.6 `pd.date_range(freq='ME')` is also `[us]` (probe), so `DATES` and parsed dates share a unit; label lookups and the `isin` on (asset, date) tuples (cell 84, asserted) work. `dj30.date` is int64 and parsed explicitly (cell 42). |
| T21 | deprecated / removed APIs (Py 3.14, pandas 3.0.5, sklearn 1.9.0) | PASS | None of the §3 patterns. All code cells compile under 3.11 with no `SyntaxWarning` (no invalid escapes). No removed NumPy aliases (`np.float`, `np.trapz`, `np.in1d`, …). `lasso_path(alphas=…, return_n_iter=True)` still supported; it re-sorts alphas descending, and the grid (cell 51) is already descending, so coefficient columns line up (probe: column 20 equals `Lasso(alpha=grid[20])`). Nothing found that is specific to 3.14 (not testable here). |
| T22 | matplotlib / seaborn deprecations | PASS (NIT) | `ax.set_xticks(x, labels)` needs matplotlib ≥ 3.5 (fine). `sns.heatmap` → matplotlib `PendingDeprecationWarning` (hidden). |
| T23 | warnings suppression | PARTIAL (MINOR) | cell 46 `warnings.filterwarnings('ignore')` (exam's code, unchanged from commit 174f164). Finding M1. |

---

## Part D. Problem 3 leakage audit (strict: data available at the forecast date; no test months in selection)

| # | Transformation | Status | Evidence |
|---|---|---|---|
| L1 | Lags of characteristics | PASS | cell 64 `lagged[x] = W.shift(1) if x in ('x1','x3','x4') else W`; identities `x2 = R12(t-12..t-1)`, `x5 = SD36(t-36..t-1)` hold in >99.6% of cells (cell 61); 20 spot checks (cell 66). |
| L2 | Back-filled leading cells | PASS | Set to NaN, never used (cells 55, 64); last fill 2002-07 < first target 2003-01. |
| L3 | Within-class monthly ranks (and z-score variant) | PASS | cell 64 `B.rank(axis=1, …)` / cross-sectional mean-sd of one month; no parameter estimated across time. |
| L4 | Global macro trailing z | PASS | `GZ = trailing_z(G).shift(macro_lag)`: 60-month window ending at t−2 (cell 64); spot-checked (cell 66). |
| L5 | Country macro differential + trailing z | PASS | `D = P.sub(P['country_7'])`, `trailing_z(D)`, `.shift(lag)` (cell 64); `x10` replaced by `x86` because it is a same-year average (cell 56); the `x10` extra uses a 14-month lag (correct: always a completed prior calendar year). |
| L6 | Own-history levels `l1..l5` | PASS | Trailing z of already-lagged values (cell 64). |
| L7 | 36-month σ̂ (target scale, P1, RP, TSMOM) | PASS | `Rw.rolling(36, min_periods=36).std(ddof=1).shift(1)` (cell 64) and `SD36` (cell 55); asserted equal to `R.iloc[m-36:m].std(ddof=1)` (cell 66). |
| L8 | Forecast benchmarks | PASS | `b_pool = (cumsum/cumcount).shift(1)`, `b_class` shifted within class, `b_asset` expanding-mean shifted, `rb_asset = R.expanding().mean().shift(1)` (cell 68); two brute-force unit tests. |
| L9 | Linear-model scaling / demeaning / PCA | PASS | Inside `LinearFE.fit` on fit rows only (cell 71). |
| L10 | Hyper-parameter selection | PASS | Three annual folds inside `[a, b]` (cell 71 `inner_folds`), ties → heavier penalty (`np.argmin` on a most-penalised-first grid); test rows `(m > b) & (m <= b+12)` never enter the tuning loop. Forest untuned. Post-hoc PCR K = 0 run (cell 84) is labelled "not a candidate". |
| L11 | Portfolio weights, TSMOM sign, turnover drift | PASS | `R12 = …rolling(12).sum()).shift(1)` (cell 61); turnover drift uses `W.shift(1)`, `Rs.shift(1)` (cell 69); truncation causality check of RP/TSMOM weights (cell 69). |
| L12 | Sleeve volatilities / class-risk-budget scales | PASS | `sleeve_vol = sleeve_ret.rolling(36, min_periods=36).std(ddof=1).shift(1)` (cell 86); sleeve returns use weights built from σ̂_{t−1}. Present since the first analysis commit (591c8ff), so not post hoc. |
| L13 | Permutation importance | PASS | Fitted forest scored on its own test year only (cell 71); evaluation only, feeds no choice (cell 78). |
| L14 | Placebo circular shift | PARTIAL (MINOR) | `np.roll(A, shift, axis=0)` (cell 64) moves the last s months to the start. Target months up to s + 60 use wrapped (i.e. later) macro, including 1–49 OOS months for s ≥ 72 (Table 3.19b, cell 83, verified by rebuild). No OOS return enters any fit, so no target leakage, and it is disclosed (cells 81, 96; METHODOLOGY_LOG P3-14). But the placebo inputs are not strictly causal. Finding M2. |
| L15 | Exploratory statistics | PASS | Predictor–return tercile sorts use 2003–2010 only (cell 62, asserted). Full-sample statistics involve returns alone or predictors alone (cells 58–61, 82–83, 90). Table 3.8 / 3.9 (benchmarks out of sample before any model) follow the exam's rule 1, after the benchmark was fixed in cell 50. |
| L16 | Pre-registration | PARTIAL (MINOR; not a guide rule) | cell 50 says it was "written before any model was fitted"; cell 95 item 6 concedes version control does not prove this ("the design first appears in a commit (591c8ff) made after the development ledger had run"). |

---

## Part E. Run-ability on the class server

| # | Item | Status | Evidence |
|---|---|---|---|
| R1 | Data path | PASS | Cells 4, 26, 46: `_DATA_DIR = '/classes/41210_MiF_fall2026/Data/'`, falling back to `'./'` if absent (student's 2-line addition to the exam cells). |
| R2 | Top-to-bottom order | PASS | Execution counts 1..58 run in order, with no errors or stderr in the stored outputs. P3 re-declares its style/constants (cell 51), so it does not depend on P1/P2 state. |
| R3 | Dependencies | PASS | pandas, numpy, matplotlib, seaborn, statsmodels, scipy, sklearn (the exam imports the same set); `Styler` needs jinja2; `display` needs IPython (a Jupyter kernel). |
| R4 | Runtime | PARTIAL (MINOR risk) | P3 took 9.0 min on this 4-core machine (cell 94). The ledger took 519 s (cell 74), `rf|M3|expanding` alone 300 s. P2 cells 31/35/40 took 27 + 31 + 73 s. Total about 12 min. `N_JOBS = min(4, cpu_count)` (cell 51), so a 1–2-core shared server would be slower on the forest part. |
| R5 | Memory | PASS | Largest arrays: 10,000 × 500 (cell 41), 2,500 × 1,000 (cell 31), 10,000 × 168 (cell 71). |
| R6 | Environment record | PARTIAL (NIT) | The notebook metadata has no `language_info`, and no cell prints library versions. Course env (3.14 / 3.0.5 / 1.9.0) vs this machine (3.11 / 3.0.6 / 1.9.1) cannot be confirmed from the file. |

---

## Part F. Findings

### (a) Errors in what exists

**FATAL: none.** No shuffled split or fold on time-ordered data. No transformation fitted before the split.
No `r2_score` read as R²_OOS. No `LogisticRegression`.

**MAJOR: none found.**

**MINOR**
- **M1 – All warnings silenced for Problem 3** (cell 46, `warnings.filterwarnings('ignore')`, the exam's own line). It hides exactly the §3 "deprecation noise" and any sklearn/pandas FutureWarning or ConvergenceWarning on the course environment, which differs from this machine (Python 3.14, sklearn 1.9.0). It is disclosed in the write-up. Of the API patterns probed on this machine (not a full re-run), only one warns: a matplotlib `PendingDeprecationWarning` from `sns.heatmap` (cell 59). Lasso convergence is asserted separately (`assert max(n_iter) < 10_000`, cell 71). Fix: add `warnings.resetwarnings()` (or `simplefilter('default')`) in cell 51. Alternatively, run once with warnings on and state that none appear.
- **M2 – The placebo is not strictly causal** (cell 64 `np.roll`, Table 3.19b). Wrapped end-of-sample macro enters training rows and, for s ≥ 72, the trailing-z windows of up to 49 OOS months. The circular shift is exam-defined and disclosed, and no return leaks into a fit. A strictly past-only alternative is a pure extra lag of s months (drop or zero the first s months), which could be reported as a robustness check on Q2.
- **M3 – A leakage "executable check" that cannot fail.** Cell 91 cites "`n_samples_seen_` asserted at every fit" against the fatal §4b error. But `assert self.sc.n_samples_seen_ == len(Xd)` (cell 71) is true by construction, because the scaler is fitted on `Xd`. The real guarantees are the fold/refit mask asserts and the truncation causality test (cell 66). Cell 95 says a can't-fail unit test was removed; this one remains. The audit table should name the mask asserts and the causality test instead.
- **M4 – Very long cells** against §6.2 ("A 60-line block you cannot read…"): cell 71 has 131 non-blank lines, cell 64 has 87, cell 94 has 68, cell 51 has 62, cell 69 has 60. This is a defensibility risk under G0.1, not a correctness error. Consider splitting the harness: splitter / `LinearFE` / `run_linear` / `run_rf` / bootstrap.
- **M5 – Dependence in resampling is only disclosed, not checked.** P2 cell 41 resamples daily returns iid; the caveat is stated. P3's month bootstrap ignores serial dependence. The write-up asserts this "matters little for R²_OOS" but no cell shows it, for example the autocorrelation of the monthly loss differential, `month_sums(...)`, in cell 80. All verdicts are "no", so an understated SE could not have created a false positive.
- **M6 – Runtime on a small server** (R4). About 12 min here on 4 cores. It is dominated by the forests plus per-year permutation importance.

**NIT**
- N1: cell 91's "Nothing is imputed" should say "no fitted imputation; secondary `l1..l5` and one `x10` month are filled with the constant 0".
- N2: `importances_std` is ddof = 0 (cell 19).
- N3: `P3_SEED` seeds two generators (cells 66, 71).
- N4: unused exam imports (`sm`, `Lasso`, cell 46).
- N5: no sklearn `Pipeline`; the manual equivalent in `LinearFE` is correct and unit-tested against `Ridge` and `PCA` (cell 72).
- N6: no printed library versions (R6).
- N7: P1's reported 90.75% is the winner of the same CV used for selection; it is optimistic, and cell 9 discloses this.

### (b) Omissions worth proposing
1. Re-enable warnings after the exam's setup cell and record that the run is warning-free (M1).
2. Add a cell printing `sys.version`, `pd.__version__`, `sklearn.__version__`, `np.__version__` (R6).
3. Add a non-circular placebo (macro lagged by s + 2 months, early months dropped) next to Table 3.19 (M2).
4. Print the lag-1..12 autocorrelation of the monthly loss differential for ridge M2, to support "the month bootstrap … matters little" (M5).
5. Replace the tautological `n_samples_seen_` citation in the audit table with the mask asserts and the causality test (M3).

### (c) Verified OK (explicitly checked)
- P1: shuffled stratified folds and the 70/30 split are legitimate here: 400 people with unique `User ID`, where 40 share feature values but are distinct people, and cell 6 checks this. All models are seeded. GB has no early stopping. Permutation importance runs on held-out rows with explicit accuracy scoring.
- P2: `var(ddof=1)` everywhere the manual says "unbiased". `N(0, σ̂²)` is drawn with scale `sqrt(s2)`. Independent named RNG streams. The vectorised pipeline is asserted equal to the literal loop (cell 30). No splits.
- P3: every item in Part D except L14 and L16. The truncation causality test covers all 64 feature columns. All macro series used are monthly (re-checked from the CSVs). The lasso alpha ordering is correct. The ridge and PCR paths are unit-tested against sklearn. The R²_OOS scorer is unit-tested. Positional alignment of macro and panel rests on asserted identical 300-month date sets (cell 54).
- pandas-3 / CoW / dtype behaviour of every API pattern used: probed, no warnings or errors apart from the seaborn one.
