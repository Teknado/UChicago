# Methodology Log — BUSN 41210 Final Project (Autumn 2026)

A running record of **which approach was used for each problem, which assumptions were made, and why**. It is written so that every analytical choice can be traced to a lecture slide, to the exam notebook's own definitions, or to an explicitly flagged choice of ours. It is updated as work proceeds: every implemented step changes its status, and every change of plan gets a dated entry in §H.

**Conventions**
- `L5 p.56` = Lecture 5 PDF, page 56 (PDF page = slide number). `cell 36` = 0-based cell of `Final_Autumn_2026-1.ipynb`. `AIG §4` = AI Coding Guide section. `DP` = the pre-modelling data profile (descriptive facts about the files, §A.3).
- Scope tags:
  - **[LEC]**: taught; the slide is cited.
  - **[EXAM-DEFINED]**: required or defined by the notebook, not taught.
  - **[OUR CHOICE]**: a design decision with no single source; the reason is given.
  - **[OUTSIDE SCOPE]**: used only with the student's approval.
- Status: `PLANNED` (agreed design, not yet coded) → `IMPLEMENTED` (in the notebook) → `VERIFIED` (checked against a known answer or a second route). `REVISED` marks an entry changed by review; the change is recorded in §G or §H.

---

## A. Sources, environment and process

### A.1 What was read (2026-09-26)
- **AI Coding Guide** (3 pp.). It is treated as binding. The two fatal errors are the shuffled split or fold on time-ordered data (§4a) and a transformation fitted before the split (§4b). Also: `LogisticRegression()` is penalised by default (§4c), and `r2_score` is not the course's R²_OOS (§4d). The pandas-3 API failures (§3) and the workflow of §6 apply throughout.
- **Lectures 1–8** (528 pages). Each was read in full, from the extracted text and from rendered slides for formulas, figures and code, into slide-cited notes. The notes define the method boundary: a method that a lecture only names (neural nets, SVM, the elastic-net estimator, OOB error, SHAP, XGBoost, HAC standard errors, t-SNE) is *not* treated as taught.
- **Exam notebook** (41 cells): Problems 1–3, all instructions, data table, lag rules, "Two rules", suggested workflow, "What to submit".

### A.2 Environment
The course environment is Python 3.14, pandas 3.0.5 and scikit-learn 1.9.0 (AIG §3). This machine runs Python 3.11, pandas 3.0.6, scikit-learn 1.9.1, numpy 2.4.6, scipy 1.17.1 and statsmodels 0.15.0.

Version-specific facts verified here:
- `LogisticRegression(penalty=None)` raises a FutureWarning in 1.9.1. `C=np.inf` gives identical coefficients.
- `GradientBoostingRegressor(n_iter_no_change=…)` internally calls a *shuffled* `train_test_split`, so it must never be used on time-ordered data.
- The `scoring=None` defaults of `GridSearchCV` and `permutation_importance` score R² against the fold or test mean, so scoring is always set explicitly.
- `RandomForestRegressor` defaults to `max_features=1.0`, which is bagging rather than a forest.
- pandas 3 loads strings as `StringDtype`, parses dates to `datetime64[us]`, and returns read-only `.corr().values` under copy-on-write.

### A.3 Process
1. **Understand.** Nine parallel readers: one per lecture, plus a data profiler. The profiler was barred from computing anything that relates a predictor to a future return, because Problem 3 requires the design to be fixed before anything is fitted (cell 36 rule 2). Its findings are in §E.1.
2. **Design.** One planner each for P1 and P2. Three competing P3 designs (linear-first, nonlinear-learners, evaluation-first) were scored by a judge (51 / 50 / 41 out of 60). The final P3 design is the evaluation-first draft with the linear-first draft's feature engineering and windows grafted on.
3. **Verify.** Nine adversarial reviewers:
   - one citation audit per problem, with every cited page checked against the PDF (91, 132 and 136 distinct pages);
   - one correctness and completeness audit per problem;
   - a sceptical referee and a defensibility critic for P3;
   - a cross-plan completeness critic that went through every exam cell.

   Result: **0 fatal, 18 major, 49 minor and 21 nit findings.** Every major finding changed the plan; see §G.

---

## B. Global decisions (all problems)

| ID | Decision | Justification | Status |
|---|---|---|---|
| G-01 | **Data path.** Keep the exam's `_DATA_DIR` server line in cells 4, 21 and 35, and add `import os; _DATA_DIR = _DATA_DIR if os.path.isdir(_DATA_DIR) else './'` after it. | Cell 38 requires the notebook to run top to bottom on the server *or* with `_DATA_DIR` pointed at a local copy. The path is set three times, so a one-cell override would be undone by the next cell. `./` is the exam's own commented alternative (cell 35). [OUR CHOICE] | PLANNED |
| G-02 | **Every number from a cell.** Each number in an answer cell or in the write-up must be printed by a code cell. Problem 3 ends with a cell that prints every headline number from stored results. | Cell 1: "Every number must come from a cell in this notebook". | PLANNED |
| G-03 | **Seeds.** P1 uses `random_state=7034` everywhere (cell 3). P2 uses `SEED`, the student ID, through named `SeedSequence` streams. P3 uses `P3_SEED = 7034` and never rebinds `SEED`, so re-running P2 after P3 cannot change P2's numbers. | Cells 3 and 20; set seeds because random draws differ run to run (L6 p.18). The P3 naming avoids one name carrying two meanings. [OUR CHOICE] | PLANNED |
| G-04 | **Variance convention.** Always `ddof=1`, passed explicitly. numpy defaults to `ddof=0`. | L2 p.56–57 (s² = SSE/(n−p)); P2's manual specifies `var(ddof=1)` (cell 20). | PLANNED |
| G-05 | **Describe before modelling.** Each problem opens with shape, dtypes, counts and head, asserted against the data profile. | AIG §6.1 and §6.4; L1 p.81–82 ("ask what the data are"). | PLANNED |
| G-06 | **Only lecture methods.** Anything taught only in passing is excluded. Exam-defined quantities (VaR, kurtosis, Sharpe, RP, TSMOM, costs, placebo) are implemented exactly as the exam defines them and labelled [EXAM-DEFINED]. | The student's instruction; the lecture boundaries noted in §A.1. | PLANNED |
| G-07 | **Honest reporting.** Results are reported whichever way they come out. A negative R² is a result. Benchmarks are never switched after seeing results, and the search size goes next to every headline. | L5 p.57; L4 p.28; cell 11 ("whichever way it comes out"); cell 33 ("A negative result, honestly established, earns full credit"). | PLANNED |
| G-08 | **Answer length limits are obeyed literally.** 1.2 gets one or two sentences; 1.3 two or three; 2.1 two or three; 2.3 and 2.4 one paragraph each. Technical notes sit outside those sentences and are labelled. | Cells 8, 11, 22, 27, 30. | PLANNED |

---

## C. Problem 1 — Trees and Ensembles

| ID | Decision | Alternatives considered | Assumptions | Justification | Status |
|---|---|---|---|---|---|
| P1-01 | Every accuracy comes from the notebook's `cv5` and `cv_acc()`, on shared folds. | `cv=5` as an integer. Rejected: that gives *unshuffled* folds, which are different folds. | Rows are unrelated people. | Shared split L8 p.53; K-fold for iid data L5 p.28, p.35; exam cells 3–4. A random split is legitimate for a cross-section (L8 p.45). | PLANNED |
| P1-02 | The baseline is the majority rule "nobody purchases". It scores 257/400 = 64.25% pooled; per fold it is 63.75–65%, because stratification cannot split 257 and 143 evenly into 5. Equality is asserted on integer counts. | `DummyClassifier`. Not taught and not needed. | — | L6 p.57; L8 p.30 root [257, 143]; L1 p.81. | PLANNED |
| P1-03 | Size selection: pick the highest CV accuracy over {2, 3, 4, 6, 8, 12}. **Exact ties** (the same integer number of people correct) go to the smaller size. The unpruned tree is a reference, not a candidate. | A 1-SE or "within noise" tolerance. Rejected: the 1-SE rule appears only in a figure legend (L5 p.46) and is not taught, and the exam's rule is literal. | — | Size chosen out of sample L8 p.29, p.32; parsimony L5 p.25, L4 p.23. The chosen score won a 6-way search, so it is disclosed as slightly optimistic (L4 p.43, L5 p.57). | PLANNED |
| P1-04 | **REVISED.** Differences between models are judged from the 5 per-fold scores, their spread (sd, min–max) and the count of folds won or lost, against the resolution of 1 person = 1.25 pp per fold and 0.25 pp pooled. | A paired per-person SE with a "< 2 SE means indistinguishable" rule. **Rejected in review**: it is effectively McNemar's test, which is not taught. | Folds share training rows, so the spread is descriptive only. | The course compares CV numbers directly and shows fold or tree spread (L5 p.46 error bars; L8 p.45 ±1 sd). | PLANNED |
| P1-05 | The chosen tree is refit on **all 400 rows** before plotting. | Plotting one CV fold's tree. Rejected: CV chose a *size*, and each fold tree saw only 320 people. | — | Refit on all data at the chosen tuning value, L5 p.36; L8 p.33 does exactly this (root `samples = 400`). | PLANNED |
| P1-06 | The 1.2 answer is exactly one or two plain sentences. Thresholds are translated (Age ≤ 42.5 → "42 or younger"; sklearn thresholds are midpoints and the data are integers or multiples of $1,000). It says "mostly" rather than "only", because 9 of the 241 low-salary young people bought in L8's version. | — | If k* = 3 the tree should reproduce L8 p.33; that is checked, not assumed. | L8 p.25, p.33–34 (reading the tree; data plane); L8 p.17 (leaf probability). | PLANNED |
| P1-07 | **REVISED.** The 1.3 explanation keeps the two ensembles apart. A forest *averages away variance*; boosting *adds shallow trees to remove bias*. With a near-rectangular 2-variable boundary and 400 people, a small tree leaves little of either. GB runs at untuned defaults (ν = 0.1, depth 3, 100 rounds). | "Ensembles mainly remove variance" for both. **Rejected in review**: it contradicts L8 p.51 for boosting. | Default hyper-parameters as fixed by the exam. | L8 p.40–45 and p.51 (bagging vs boosting are opposites); L8 p.50–52 (ν and B are tuning parameters); L8 p.54–55 (out-of-the-box boosting lost to the forest). | PLANNED |
| P1-08 | 1.4 uses `permutation_importance(..., scoring='accuracy', n_repeats=50, random_state=7034)` on the held-out 30%. It is compared with MDI through a side-by-side table and a bar chart, and the mechanism is tested with split counts and distinct values per feature (Salary 117, Age 43, Gender 2). | `n_repeats=5` (the default). Too few for a 120-row test set, where accuracy moves in 0.83 pp steps. | The held-out accuracy of the 70/30 forest is labelled *diagnostic*: cell 3 reserves "reported accuracy" for 5-fold CV. | L8 p.47, p.56 (MDI is in-sample); p.57 (repeat and average; "answers the question you actually asked"); p.58 (why the two rankings disagree); p.60 (importance has no sign). | PLANNED |
| P1-09 | 1.5 is text only. It covers look-ahead and dependence between neighbouring months; the direction of the error (upward, i.e. optimistic, in expectation); the fix (walk-forward train → validation → test, lagged predictors, per-window preprocessing, a training-period majority benchmark); and why removing `shuffle` alone is not enough. Mechanisms that are our own reasoning are labelled as such. | A simulated demonstration. Not needed ("if any"). A purging gap is [OUTSIDE SCOPE] and unnecessary for a one-month target. | — | L5 p.27, p.48, p.49–52, p.56, p.58 #3, p.59; L6 p.30; L1 p.82; L4 p.43. | PLANNED |

**Known-answer checks for P1:**
- 400 rows, 257/143;
- the baseline, asserted on counts;
- the unpruned 400-row tree against L8 p.30: about 62 leaves, with one conflicting cell, so 399/400 correct in sample;
- if k* = 3, the tree equals L8 p.33;
- in-sample > CV for every size;
- MDI sums to 1.

---

## D. Problem 2 — Training on Your Own Output

> **Firewall.** Part 2.1 is the student's own prediction, written before any Problem 2 code runs (L2 p.78, "Write down a number"; L4 p.26, "Say a number before I show you").
>
> **2.1 committed on 2026-09-27, before any Problem 2 code**, and recorded here verbatim. It goes into cell 23 unchanged, and 2.3 reconciles each claim with the measured numbers:
>
> *"In levels, sample variance $s^2$ is an unbiased estimator of population variance ($E[s_{t+1}^2 \mid s_t^2] = s_t^2$), so the desk average across 1,000 independent banks should remain approximately 1.0. However, because the log function is strictly concave, **Jensen's Inequality** implies that $E[\ln s_{t+1}^2 \mid s_t^2] < \ln E[s_{t+1}^2 \mid s_t^2] = \ln s_t^2$. Therefore, over 2,500 nights of sequential refitting on synthetic draws, the **median** variance and the trajectory of an individual desk will experience a systematic negative drift (proportional to $-1/(n-1)$), leading to a collapse in estimated volatility and a severe underestimation of VaR."*
>
> The prediction is deliberately not assessed until the numbers exist. The supplementary theory note (§D.2) will be added with the Problem 2 results.

| ID | Decision | Alternatives considered | Assumptions | Justification | Status |
|---|---|---|---|---|---|
| P2-01 | **Night indexing.** Night 1 fits the real library, whose sample variance is exactly 1. Each later night fits the previous night's 500 scenarios. So nights 1 → 2,500 are separated by 2,499 random draw-and-refit steps. | 2,500 random steps: a one-step difference, far inside Monte-Carlo noise; mentioned in one clause. | "2,500 refits in all" includes the night-1 fit on real data. | Cell 20 ("every desk starts on night 1 from a library whose sample variance is exactly 1"); the manual's order (fit, draw, replace, report). | PLANNED |
| P2-02 | **RNG design.** `SEED = 2694` (student-supplied, 2026-09-27). `SeedSequence(SEED).spawn(k)` gives named, independent streams: one desk; 1,000 desks; each 2.3 experiment; 2.4(a); 2.4(b). New streams are appended, never inserted. The one desk is independent of the 1,000. | The global `np.random.seed`. Rejected: legacy global state, and re-running cells out of order changes results. | — | Reproducibility; cell 20 ("your numbers will differ from your classmates'"). [OUR CHOICE] | PLANNED |
| P2-03 | **One implementation.** A vectorised `run_desks(n_desks, n_scen, nights, rng)` is used by 2.2 and by 2.3. A unit test checks that it equals the literal one-desk loop exactly. Scenarios are drawn as `sd * standard_normal`: the SD, not the variance, is the scale. | — | — | The manual, cell 20; AIG §6.4 (check against something known). | PLANNED |
| P2-04 | **Monte-Carlo uncertainty.** A fraction's SE is the sample SD of the 0/1 indicator over √M. A mean's CI uses t_{M−1}. The martingale check reports a 95% CI and whether it contains 1, with the reporting rule fixed before running. Non-rejection is reported as "consistent with", never as "equals". | p(1−p)/M. The Bernoulli variance is not on any slide, and the direct form is numerically the same. The "|t| < 2 means equal" wording was **rejected in review**: the course treats non-rejection as continuing under the null (L2 p.91, p.93). | Draws are independent. | L2 p.63 (var(X̄) = σ²/n); L2 p.81, p.84–86. | PLANNED |
| P2-05 | **Drift experiment design.** For n = 50, 500 and 5,000: 500 desks × 500 nights each. The SE is computed across desks and cross-checked against the SE across all increments. The formula is proposed from how the measured drift **scales with n across the three values**, judged against the Monte-Carlo SEs, with the decision rule fixed before running. The n = 500 row is cross-checked against the increments of the 2.2 run. | — | — | Cell 27 ("measure … and propose the formula"); L2 p.63; logs L3 p.48–49. | PLANNED |
| P2-06 | **2.4(a) real days.** Each desk has its *own* 500 N(0, 1) draws, **rescaled** to sample variance exactly 1 but **not demeaned**. The library is those 500 plus tonight's 500 scenarios (1,000 returns; last night's scenarios are replaced). Night 1 fits the real days alone. | Demeaning as well. It is not required by the rule, so it is reported as a robustness row on the same draws. One shared real sample for all desks was rejected: desks would not be independent. | "Holds 1,000 returns" means the scenarios do not accumulate. | Cell 30; cell 20 ("when a part needs the real days themselves, take them to be 500 draws from N(0, 1)"). | PLANNED |
| P2-07 | **2.4(b) library.** The night-1 library is **all 1,511 real days**. | A 500-day window, first or last. Arbitrary; offered as an optional robustness row. | — | 2.326 × sd(1,511 days) = 2.78% reproduces the story's "2.8% of NAV" (DP). The exam asks for the volatility and kurtosis "of the real returns". | PLANNED |
| P2-08 | **Estimators stated explicitly.** Daily volatility is `std(ddof=1)`. Excess kurtosis is `scipy.stats.kurtosis` with its defaults `fisher=True, bias=True`, the function the exam imports (cell 21); the bias-corrected value is a footnote. Empirical 1% VaR is `-np.percentile(r, 1)` with linear interpolation, with the `lower`/`higher` range shown. | pandas `.kurt()` (bias-corrected): 24.16 vs 24.07. The choice is stated, not hidden. | — | [EXAM-DEFINED] cells 20, 30. | PLANNED |
| P2-09 | **Kurtosis reference.** The normal-sample reference band comes from simulating 10,000 normal samples of 500 and reading off percentiles. | The analytic √(24/n) SE. Not taught. | — | Simulating a statistic under a known null, L4 p.27–28. | PLANNED |
| P2-10 | **Bootstrap SE** for the empirical VaR and the normal VaR on the real days: iid resampling of the 1,511 days, B = 10,000. **Our caveat, not the slide's:** iid resampling ignores volatility clustering, so these SEs are optimistic. | Block bootstrap. Not taught. | — | L2 p.80 ("resample the … actual weeks with replacement"). Earlier drafts wrongly attributed the independence caveat to L2 p.80; corrected in review. | PLANNED |
| P2-11 | **Wording of the conclusion.** "What stops it" is phrased as *real data, information from outside the model, kept in the training set at every generation*. In 2.4(a) even the same 500 original days suffice. It is not phrased as "fresh data". | "Fresh real data". **Revised in review**: the anchor in (a) is not fresh, and the exam rules out a changing market. | — | Reasoning from the 2.4 experiments. The L2 p.80 wording ("the gap between them is the size of the assumption") is quoted verbatim, not paraphrased. | PLANNED |

### D.1 Required imports added in P2 cells
`import os`, `import scipy.stats as stats` (the L2 p.87 style), and `from scipy.special import digamma, polygamma` only if the supplementary note uses them. The exam's cell 21 imports only `norm`, `kurtosis` and `time`, so without these additions the planned code would raise `NameError`.

### D.2 Supplementary theory note
2.1 is now committed (§D). The note will be added together with the Problem 2 results. When added, it will be labelled "supplementary, beyond the lectures". The χ² law of s², log-increment moments, the delta method and the lognormal approximation are not taught (Lecture 2 boundaries). Any number it quotes will be computed in a cell.

---

## E. Problem 3 — Cross-country asset return prediction

### E.1 Data-trap register (from the pre-modelling profile; the notebook recomputes and prints each item)

| # | Trap | Evidence | Treatment | Why it uses only information available at the time |
|---|---|---|---|---|
| T1 | **`x10` is a same-year average** of the monthly short rate (`x86`), stamped from January | Equals the same-year mean of `x86` exactly in 208 of 300 country-years; correlation 0.998 with the same-year mean vs 0.839 with the prior-year mean. Example: country 7 in 2022 has `x10` = 2.228 every month while `x86` went 0.08 → 4.10. | **Dropped.** `x86` from the extended file is used instead, lagged 2 months. The same applies to its duplicate `x73`. | A one-month lag leaves up to 11 months of look-ahead. The fix is to use the monthly source series at its release lag (L5 p.58 #2: merge on the publication date, not the period end). |
| T2 | **`x11` stops early** for c12 (after 2020-10), c9 (2021-08) and c11 (2024-03), with no interior holes | c12 and c9 stop at the start of the 2021–23 inflation shock. A stale forward-fill would be wrong by up to 10.1 pp for c12. | Diagnosed and rebuilt as `x86 − x12`, with the rebuild quality reported (pooled correlation 0.988, RMSE 0.354; c12 weakest at 0.871). **Not added as a column**: it is an exact linear combination of two included series. | Same-month `x86` and `x12` are both complete and are lagged like every other series. Forward-fill was rejected (regime change). A fitted per-country rebuild was rejected: it would be a transformation fitted on data (L5 p.58 #1). Exact collinearity, L3 p.40. |
| T3 | **Leading back-fills:** `x3` for a1, a2, a5, a26, a21; `x2` for a3, a22; `x5` for a3 | Constant runs from 2000-01 equal to the first genuine value; the last filled cell is 2002-08. | Set to missing. The first target month is 2003-01, so no filled cell enters any design matrix, and this is asserted. | A backward fill carries the future into the past (cell 37). For example, asset 3's filled `x2` contains the returns it would "predict". |
| T4 | **Exact duplicate columns** between the curated and extended files: `x73`=`x10`, `x89`=`x12`, `x114`=`x13`, `x17`=`x14`, `x92`=`x15`, `x18`=`x16`. Also 26 global columns hidden in the extended file (`x88`=`x6`, `x104`=`x7`, `x109`=`x8`, `x99`=`x9`; `x148` constant; `x160` = −`x143`). | max \|diff\| = 0 on all 3,600 rows | The extended file is never concatenated. Only `x86` is taken from it. | Cell 34 ("find them before you concatenate"); exact collinearity, L3 p.40. |
| T5 | **Low-frequency stamping** in the extended file: 39 annual and 26 quarterly columns carry same-period averages stamped at the period start | e.g. `x74` = the same-year mean of `x85` | Not used. The rule is recorded in case they are ever used: annual series ≥ 14 months, quarterly ≥ 5 months. | L5 p.58 #2 |
| T6 | **`x1` frozen** for 5 B assets in 2024-10..12 | Stale values while the rate differential kept moving | Kept as stored and flagged (10 target rows). An evaluation-only cut excludes them. | A stale value is what a desk would have seen, so it is not look-ahead. |
| T7 | **`x5` floored at 0.004** for `asset_16`, 2019-09..2023-03 | The true 36-month SD falls to 0.00298 | σ̂ is computed from returns, so no denominator uses the floor. `x5` is kept as a characteristic; its within-class rank is unaffected. | Both quantities are trailing. |
| T8 | **Scales differ** by up to 3 orders of magnitude across classes (`x1` for D is in percent, for B a monthly decimal); monthly volatility runs from 0.82% to 17.40% | DP §C, §D | Within-class ranks for characteristics; a vol-scaled target. | L5 p.23 (scale matters for penalties); L3 p.2–3 (constant variance). |
| T9 | **Row order.** `sort_values('country')` sorts lexicographically (country_1, country_10, …) | DP §F | Merge on (country, date) keys with `validate=`, never by position. | — |

### E.2 Characteristic hypotheses (checked against past returns or macro only, never against the target)
- **`x2`: 12-month momentum, pre-lagged.** It equals the compounded return r_{t−12..t−1} in 14,347 of 14,400 rows; the exceptions are vendor-winsorised extremes.
- **`x5`: 36-month volatility, pre-lagged; the risk measure.** It equals the SD of r_{t−36..t−1} in 13,157 of 13,200 rows; the exceptions are the T7 floor.
- **`x1`: carry.**
  - B: correlation 0.96 with the short-rate differential vs country 7.
  - D: correlation 0.87 with the term spread.
- **`x3`: value / long-horizon reversal.** Correlation −0.93 to −0.98 with the 60-month cumulative log return for A, B and D; for C it looks like a book-to-market measure.
- **`x4`: 12-month change in the carry fundamental.** For B it equals Δ12 of the rate differential exactly (correlation 0.9996).

**Implications, stated in advance:**
- Some country macro is *already inside* `x1` and `x4` for class B.
- `x2` and `x5` carry the same information as the TSMOM and risk-parity benchmarks.
- Both matter for interpreting Q2 and Q3.

### E.3 Design decisions

| ID | Decision | Alternatives considered | Justification | Status |
|---|---|---|---|---|
| P3-01 | **Windows.** First target 2003-01. Initial training block 2003-01..2010-12 (96 months). OOS 2011-01..2024-12 (168 months), touched once. Sub-periods 2011–17 and 2018–24. | 2010-01 start (84/180 months) or 2013-01 (120/144). Going from 168 to 180 OOS months cuts the SE by only about 3.5%, but shrinks training to 72 months and class D's rows from 672 to 504. | Cell 36 rule 2 (fixed in advance, with reasons). The training block contains a full cycle including the GFC. L4 p.43; L5 p.48–52. | PLANNED |
| P3-02 | **Schemes.** Static (fit once at 2010-12); expanding (primary; refit each December, 14 refits); rolling (96 months, equal to the initial block, so all three coincide at the first refit). | Monthly refits: 12× the compute for slowly moving premia. | L5 p.48–51 (step 12, as in L5 p.51; fixed start for expanding, L5 p.50); cell 37. | PLANNED |
| P3-03 | **Tuning.** Three annual validation folds inside each training window, mean fold MSE, ties to the larger penalty, then refit on the whole window. The penalty is parametrised as α/n (α = a × n_train), so the validated shrinkage is the shrinkage actually applied at refit. The grid spans OLS-like to null-like; edge solutions are logged. | `GridSearchCV` with a fixed α. **Revised in review**: sklearn's ridge α multiplies an unscaled SSE, so the effective shrinkage changes by about 25% between fold size and refit size. `RidgeCV()` (leave-one-out over rows) and `LassoCV(cv=int)` were rejected: they are random or row-wise folds on time data. | L5 p.48–52 (alpha chosen on validation only; "y_pred from the testing dataset has nothing to do with obtaining optimal alpha"); L5 p.35–36; L5 p.27 and p.53 (optimum inside the grid); L5 p.25. | PLANNED |
| P3-04 | **Target.** y = r/σ̂_{t−1}, with σ̂ = the 36-month SD (ddof = 1) of the asset's own past excess returns, computed by us. Forecasts map back to positions as w ∝ ŷ/σ̂. | Raw returns: a pooled squared-error fit would mostly model the 26 class-A assets. `x5` as σ̂: it carries the asset_16 floor. 12- or 24-month σ̂: more responsive, but it changes the target itself and raises turnover. | Cell 37 (recommended target); L3 p.2–3, p.20 (constant variance); L2 p.71–73 (var(b) ∝ σ²); L2 p.80. | PLANNED |
| P3-05 | **Forecast benchmarks.** The trailing mean of y through t−1, updated monthly, pooled across assets (primary, pending Q4). Per-class and per-asset trailing means are always reported, and so is zero. By class, raw-return R² is also scored against the literal per-asset trailing mean of *raw* returns. | Per-asset mean as primary: closer to the single-series HW5 harness, but noisier and easier to beat. The test-block mean is not legitimate. | Cell 36; L4 p.45–46; L5 p.55–57. Pinning one primary prevents benchmark switching (L5 p.58 #5). | PLANNED; pooled primary confirmed by the student, 2026-09-27 |
| P3-06 | **Characteristic standardisation.** Within-class, within-month rank, scaled to [−0.5, 0.5]. Within-class z-score as robustness. | Across all 50 assets: ranks would mostly sort classes, which the intercepts already carry (confounding, L3 p.57). z-score as primary: sensitive to extremes (`x3` in A reaches −9.5). | Cell 37; L5 p.23; L3 p.12–18. Ranks have no parameters estimated across time, so they cannot leak. | PLANNED |
| P3-07 | **REVISED. Unpenalised class intercepts.** Ridge and lasso penalise only predictor slopes. This is implemented exactly by within-class demeaning of y and X on the training rows, then adding back the class means (verified on synthetic data to match the closed form). | Penalised, standardised dummies. **Rejected in review**: shrinking the class means with the same α confounds "characteristics add beyond class premia" with dummy shrinkage. | Ridge penalises slopes, not the intercept (L5 p.12); common slopes with class intercepts (L3 p.50). | PLANNED |
| P3-08 | **Macro lag.** 2 months for every monthly macro series; 1 month as robustness. | Series-specific 1-month lags for market-type series. The names are anonymised, so market series cannot be told apart from statistical releases; the uniform, conservative rule is stated. | Cell 34 ("a one-month lag is the minimum, and you may argue for more"); L5 p.58 #2. | PLANNED |
| P3-09 | **Global macro.** log `x6`. Trailing 60-month z-score (min 24 obs) computed on the stored series, then lagged. Class-specific slopes (16 columns). | The full-sample z-score. Forbidden: it uses the future. | Cell 37 ("trailing information only"); log for skewed positive series L3 p.48–49; interactions L3 p.53–54. | PLANNED |
| P3-10 | **REVISED. Country macro.** Six curated series (`x86`, `x12`, `x13`, log `x14`, log `x15`, `x16`). Each is a differential vs country 7, **then a trailing 60-month z-score per country**, then lagged, × class for B, C and D (18 columns). Class A gets none. | Raw differentials. **Rejected in review**: country means explain 31–90% of the variance of several differentials, so the block could reproduce static asset premia and pose as "macro". Class A options: the country-7 convention (identically 0 as a differential), a cross-country average, or none. We choose none: A has no natural country (cell 34), and parsimony (L4 p.23). | Cell 37 (differential vs country 7, "the natural base"); trailing standardisation as for global macro. | PLANNED |
| P3-11 | **Extended file.** Used only for the audit and for `x86`; never screened against returns. The limitation is stated: credit, valuation and external-balance macro are not tested. | Screening about 150 columns, or a PCA of them. No hypothesis, and it would enlarge the search. | L4 p.25–28 ("you will always find something"); L5 p.41 (screening then validating is cheating). | PLANNED |
| P3-12 | **REVISED 2026-09-27. Models.** OLS; **ridge (primary)**; lasso and PCR (both core, on M3 only); random forest (300 trees, max_features = 1/3, min_samples_leaf = 200, untuned), plus a deep forest (leaf 5) as an extra. | Boosting: off, because the tuning budget is a leakage risk (L8 p.55) and for runtime. KNN regression is not taught. Neural nets, SVM and the elastic-net estimator are named only. | Ridge: "suitable when features are weak and should still contribute" (L5 p.60); low signal-to-noise (L1 p.33). The forest is run on the same harness because L8 p.61–62 makes the tree ensemble the out-of-the-box default. Linear wins when "the signal really is weak and linear — which, in asset pricing, is more often than you would like" (quoted exactly; an earlier draft misquoted this). The 200-row leaf is [OUR CHOICE] for low signal-to-noise, departing from L8 p.44's "grown deep". | PLANNED |
| P3-13 | **Uncertainty.** A month-resampling bootstrap SE: each resample keeps a month's whole cross-section. Report estimate ± 2 SE. | A monthly loss-differential t-test. **Rejected in review**: it is Diebold–Mariano, which is not taught. Percentile intervals were also rejected (not taught). | L2 p.80 (resample the actual observations); L2 p.85 (±2 SE). The month is the unit because all 50 assets share each month's shocks; that is [OUR CHOICE] and is stated. | PLANNED |
| P3-14 | **Placebo.** Circularly shift every raw macro series by s ∈ {36, 48, …, 120}, then run the identical pipeline. Macro is credited only if its gain beats all 8 placebos. A persistence table is reported. | 20 shifts, including wrapping ones (s > 130 would put future values at OOS dates); a rank p-value (neighbouring shifts are correlated, so the number cannot be defended). | Cell 37 ("try more than one shift"); null logic L4 p.26–28. For s ≥ 72, the trailing-z window at early OOS dates reaches wrapped values; that is a normalisation-only effect, disclosed. Training rows use wrapped (later) macro values but no OOS return enters any fit. | PLANNED |
| P3-15 | **Portfolios.** Benchmarks exactly as cell 36 defines them. P1 = ŷ/σ̂ at unit gross (primary; it collapses to RP if ŷ is constant). P3 = within-class long–short. Costs: one proportional cost per unit of one-way traded notional, 10 bp headline (confirmed by the student), 0–50 bp grid, break-even. Attribution by regression on the three benchmarks plus weight correlations. **Added 2026-09-27:** sub-period correlation diagnostics and class-risk-budget variants of RP and P1 (P3-22). | Class-specific costs: they need a sourced number per class (Q5). Mean-variance optimisation: not taught. | [EXAM-DEFINED] cells 36–37; α and β tests L2 p.94–97; multiple regression L3. The cost model is [OUR CHOICE], with break-evens so readers can substitute their own. | PLANNED |
| P3-16 | **Tercile sorts on the training block only (2003–2010), within class.** No characteristic is dropped or re-signed because of them. | Full-sample sorts. They would let OOS returns shape the design (snooping). | L4 p.43; L5 p.41, p.58 #4. | PLANNED |
| P3-17 | **REVISED. Pre-registration without theatre.** A markdown "design fixed before fitting" cell and one printed constants dict. Deviations are kept as a markdown list. | A SHA-256 hash, an `OOS_LOCKED` switch, self-grep audits and fitted-object refit tests. **Cut in review**: a hash recomputed in the same run proves nothing; the lock guarded an undefined function; the grep matched its own source. | The taught form of commit-before-compute is writing a number down first (L2 p.78; L4 p.26). AIG p.1: defend every line. | PLANNED |
| P3-18 | **Power statement**, computed in a cell and flagged as our construction from var(X̄) = σ²/n. Two cases: within-class signals, where class-common shocks cancel and power is higher, and class-level signals (macro), where the effective cross-section is about 9 and power is low. Q2 is therefore read from magnitudes, the bootstrap SE and the placebo, not from a single test. | One effective-N table. **Revised in review**: it was wrong in both directions. | L2 p.63–64; stated in Methodology before any results. | PLANNED |
| P3-19 | **ADDED 2026-09-27. PCR (core).** StandardScaler → PCA(K) → OLS on the components, with class intercepts unpenalised (within-class demeaning, as P3-07). Fitted inside each training window, and inside every validation fold during tuning. K is chosen on the same three annual validation folds as α, from {1, 2, 3, 4, 5, 6, 8, 10}. Fixed K = 3 and K = 5 are reported as sensitivity extras. Runs on M3 × {static, expanding, rolling}. | Fixed K ∈ {3, 5} as the core (the reviewer's proposal). Not adopted as the core: K is a tuning parameter and should be chosen out of sample like α (L5 p.48–52); a hard-coded K is exactly the kind of arbitrary constant the same review objected to for the forest. Fixed K is kept as sensitivity. PCR on M2 was rejected: with 5 characteristics there is nothing to compress. | Cell 33 lists PCR explicitly. PCA L1 p.45–74, "pre-treatment for further ML" L1 p.46; fitting on principal components in high dimension L8 p.61; PCR is in the GKX horse race L1 p.36; dense vs sparse L5 p.60; OLS L3 p.19–26. Standardising before PCA is [OUR CHOICE] (L1 only demeans) because the columns differ in scale (ranks vs z-scores; L5 p.23). A scree plot of the initial window is descriptive only (L1 p.43). | PLANNED |
| P3-20 | **ADDED 2026-09-27. Deep forest (extra).** The same forest with min_samples_leaf = 5 on M2 and M3 under the expanding scheme, next to the leaf-200 forest. | Tuning the leaf size in every fold. Timed on synthetic panel-size arrays: 2.4–5.6 s per deep fit, so about 15–20 extra minutes; and L8 calls the forest nearly tuning-free. | Shows the bias–variance trade-off empirically (L8 p.40–44, p.51). The leaf-200 prior stays the primary forest: at R² ≈ 0.3%, the standard error of a 5-row leaf mean (≈ 0.45 in y units) is about 9 times a plausible signal, against ≈ 0.07 for a 200-row leaf. | PLANNED |
| P3-21 | **ADDED 2026-09-27. `x10` sensitivity and leak demonstration.** (a) An extra in which the curated `x10` replaces `x86`, lagged 14 months, so every month uses a completed prior-calendar-year average. (b) A diagnostic with `x10` at the naive 1-month lag, reported only in the leakage appendix and never a candidate. | Using `x10` at a 1-month lag as the primary input (the reviewer's proposal). Rejected: it is a known look-ahead (T1). | (a) shows the result does not hinge on the extended file; (b) shows what the trap would have done. L5 p.58 #2; cell 34. | PLANNED |
| P3-22 | **ADDED 2026-09-27. Diversification diagnostics.** (a) Class-averaged 4 × 4 correlation block tables and the first-PC variance share for 2003–2010, 2011–2019 and 2020–2024. (b) Class-risk-budget versions of RP and P1: inverse-vol weights within each class, each class sleeve scaled by its trailing 36-month volatility, so each class carries equal ex-ante risk. | Covariance-optimised weights: rejected (§F). | Block structure changes in crises (L1 p.42–44); eigenvalue shares L1 p.43. The sleeve volatility reflects within-class correlation without inverting a covariance matrix. (a) is descriptive and chooses nothing; (b) is [OUR CHOICE]. | PLANNED |
| P3-23 | **ADDED 2026-09-27. Per-class ridge on M3** (expanding) next to per-class ridge on M2. | — | Completes pooled-vs-segmented for both the characteristic and the macro slopes (cell 37). Small per-class samples (B/C/D: 672–864 initial rows) raise estimation variance, disclosed (L2 p.73). | PLANNED |

### E.4 Specification ledger (revised 2026-09-27; printed next to every headline)

- **Core, 29 specifications:**
  - class-means OLS (expanding);
  - {OLS, ridge, RF} × {M2, M3} × {static, expanding, rolling} (18);
  - {lasso, PCR} × M3 × {static, expanding, rolling} (6);
  - ridge-expanding with global-only and with country-only macro (2);
  - per-class ridge on M2 and on M3, expanding (2).
- **Pre-listed extras, 9 (all expanding):**
  - PCR with K = 3 and with K = 5 (2);
  - a deep forest (leaf 5) on M2 and on M3 (2);
  - macro lag 1 month (1);
  - within-class z-score (1);
  - M2 and M3 with characteristic levels (2);
  - curated-only macro with `x10` at a 14-month lag (1).
- **Not candidates:** the 8 placebo runs, and the `x10` 1-month leak demonstration.
- **Bonferroni:** 38 candidates. The critical value is computed with `scipy.stats.t.ppf` in the notebook, never hard-coded.

---

## F. Considered and rejected (for the "what we tried and rejected" paragraph)

| Considered | Why rejected | Source |
|---|---|---|
| `train_test_split` or shuffled K-fold for P3 | Fatal on time-ordered data | AIG §4a; L5 p.58–59 |
| `r2_score` / `.score` as R²_OOS | Benchmarks against the test mean | AIG §4d; L5 p.57 |
| Lagging `x10` by 1 month | Leaves up to 11 months of look-ahead (T1) | DP; L5 p.58 #2 |
| Forward-filling `x11` | Stale through a regime change; errors up to 10 pp | cell 37; DP |
| Screening the extended file | Multiple testing; unstable post-screening sets | L4 p.25–28, p.39, p.50 |
| KNN regression | Not taught (L6 teaches the KNN classifier); curse of dimensionality | L6 p.37 |
| Boosted trees (P3) | Needs a tuning budget, which is a leakage risk; runtime | L8 p.55 |
| Covariance-optimised weights (mean–variance, minimum variance) | Not taught. A 50 × 50 sample covariance from 36–96 months is rank-deficient or badly conditioned (L1 p.63: rank < n if T < n). Correlation is handled descriptively and through class-level risk budgets instead (P3-22) | L1 p.63; review §J |
| Tuning the forest's leaf size in every validation fold | About 15–20 extra minutes on a shared server; L8 treats the forest as nearly tuning-free. A fixed deep forest runs alongside instead (P3-20) | L8 p.44–45, p.55; timing §J |
| `x10` at a 1-month lag as the primary macro input | Known look-ahead (T1). Kept only as a labelled leak demonstration | L5 p.58 #2; cell 34 |
| ~~PCA compression~~ **Reversed 2026-09-27**: PCR on M3 is now a core model (P3-19). What stays rejected is only the earlier separate-PCA-per-macro-block design (country-month rows, 9 (k_G, k_C) combinations): too complex to defend, and a leakage risk if any step escapes the fold | cell 33 lists PCR; L1 p.45–74; L8 p.61 |
| Paired SE / McNemar (P1) | Not taught | review (§G) |
| Diebold–Mariano-type test, percentile bootstrap CIs | Not taught | review (§G) |
| Wrapping placebo shifts, rank p-values | Future values at OOS dates; indefensible p-value | review (§G) |

---

## G. Verification record: review findings that changed the plan (2026-09-26)

| Finding (severity) | Change made |
|---|---|
| P1: the paired per-person SE is effectively McNemar's test, not taught (major) | Replaced by fold-level spread (P1-04) |
| P1: "ensembles remove variance" is wrong for boosting (major) | Explanation split between RF and GB (P1-07) |
| P1: the unpruned-tree check cited the training-subset tree (L8 p.31) (minor) | Compare with the 400-row tree on L8 p.30 |
| P1: fold baselines are not identical; exact float `==` fails (minor) | Integer-count asserts (P1-02) |
| P1: 1.2 and 1.3 answers would overrun the sentence limits (minor) | G-08 |
| P1: non-CV accuracies could read as "reported" accuracies (minor) | Labelled as diagnostics |
| P1/P3: `LogisticRegression(penalty=None)` is deprecated in sklearn 1.9.1 (nit) | Use `C=np.inf` if ever needed |
| P2: L2 p.80 was cited for an independence caveat it does not contain (major) | Caveat labelled as ours (P2-10) |
| P2: SE of a fraction via p(1−p) is not on the slides; "|t| < 2 means equal" misstates the convention (minor) | P2-04 |
| P2: missing imports (`os`, `scipy.stats`) (minor) | §D.1 |
| P2: "fresh data" contradicted by experiment (a) (minor) | P2-11 |
| P3: L8 p.62 misquoted to justify ridge as primary (major) | Exact quote; ridge grounded in L5 p.60 and L1 p.33 (P3-12) |
| P3: penalised class dummies confound Q1 (major) | Unpenalised intercepts (P3-07) |
| P3: country differentials carry country fixed effects (major) | Trailing z per country (P3-10) |
| P3: ranks + fixed intercepts leave macro as the only timing channel (major) | Characteristic-levels spec pre-listed (§E.4) |
| P3: power statement wrong in both directions (major) | P3-18 |
| P3: Diebold–Mariano-type decision test not taught (major) | Bootstrap SE ± 2 SE (P3-13) |
| P3: hash, lock and grep machinery indefensible; 39 specs and 40 placebo runs too many to defend; runtime relied on a process pool (major) | P3-17; ledger cut to 22 + ≤ 5; 8 placebo runs; about 8–12 min serial. (Later expanded to 29 + 9 by the §J review, all run through the same single harness loop; the cut custom machinery stays cut.) |
| P3: no data-path handling; `SEED` reused (major / minor) | G-01, G-03 |
| P3: literal raw-return trailing-mean benchmark missing (minor) | Added (P3-05) |
| P3: ridge α not comparable across sample sizes (minor) | α/n parametrisation (P3-03) |
| P3: frozen-`x1` row count was 15, actually 10 (minor) | T6 |
| P3: L4 p.14 cited for rank deficiency (wrong page) (minor) | Cite L3 p.40 only |

Independent checks I ran on the two findings that most change the design:
- Ridge with unpenalised class intercepts equals ridge on within-class-demeaned data: slopes and intercepts agree with the closed form (synthetic data).
- `LogisticRegression(C=np.inf)` reproduces the `penalty=None` coefficients without the FutureWarning.

---

## H. Deviations and implementation log

Each change is recorded with its date, what changed, why, and its effect on results.

| Date | What changed | Why | Effect |
|---|---|---|---|
| 2026-09-27 | Added PCR (core), lasso moved to core (M3 × 3 schemes), per-class ridge M3, deep-forest extras, the `x10` sensitivity and leak demonstration, and diversification diagnostics | External review, §J | Plan-stage only; search size 27 → 38 candidates; estimated runtime ≈ 15 min |
| 2026-09-27 | SEED = 2694, 2.1 committed, scope (b), pooled benchmark, 10 bp costs, literature list | Student's answers | Settles P2-02, P3-05, P3-15 and the write-up sources |

---

## I. Open questions: answered 2026-09-27
- SEED 2694 (P2-02).
- 2.1 committed verbatim (§D).
- Scope (b), expanded per §J.
- Pooled trailing-mean primary benchmark (P3-05).
- 10 bp headline cost with the grid and break-even (P3-15).
- Literature: GKX (2020), Asness–Moskowitz–Pedersen (2013), Moskowitz–Ooi–Pedersen (2012), plus the lecture sources. The student verifies the references.
- Notebook filled in place.

Nothing is still open except the go-ahead to code.

---

## J. External review (2026-09-27): what was accepted, what was rejected, and why

The student had a second AI model review the plan. Each point was checked against the exam notebook, the lecture PDFs and the data before a verdict.

| # | Reviewer's point | Verdict | Reason (evidence) | Change |
|---|---|---|---|---|
| 1 | Add PCR / dense factor models | **Accepted, modified** | Cell 33 lists PCR explicitly ("OLS, ridge, lasso, PCR, KNN, …"). The reviewer cited cell 37, which does not mention PCA. PCA is taught (L1 p.45–74) and L8 p.61 recommends it in high dimension, so rejecting PCR because it is "not taught as a named estimator" was too strict: it is two taught tools composed. Modification: K is tuned out of sample rather than fixed at {3, 5}, with fixed K kept as sensitivity. The HW6 claim could not be verified, since HW6 is not in the repository. The description of GKX (2020) as "centred on PCA factor representation" is inaccurate: it is a horse race of many learners in which PCR is one entry (L1 p.36 legend). | P3-19 |
| 2 | Keep `x10` from the "curated_features.csv" as primary for autograder schema safety | **Rejected; a narrower version adopted** | The file names are wrong (the files are `macro_country.csv` and `macro_country_extended.csv`). There is no autograder: Problem 3 is an open-ended project ("There is no answer key"; "the discipline of the evaluation is what is graded", cell 33). Nothing consumes a feature schema, and the exam itself invites the extended file and outside data (cells 34, 37). A 1-month-lagged `x10` carries up to 11 months of look-ahead (T1), the "merge keyed on a period-end" leak of L5 p.58 #2 that cell 34 warns about. Making it primary would trade a hypothetical format risk for a real leak. | P3-21: a curated-only extra with `x10` at a correct 14-month lag, plus a labelled leak demonstration |
| 3 | `min_samples_leaf = 200` is arbitrary; tune it or run both | **Partly accepted** | L8 p.44 says "grown deep … set a minimum leaf size, and stop", so a fixed a-priori leaf is the lecture's recipe. L8 p.44–45 and p.55 treat the forest as nearly tuning-free. The value 200 is not arbitrary: it follows from the leaf-mean standard error at a low signal-to-noise ratio. The request to show the trade-off empirically is fair and cheap, so a deep forest (leaf 5) runs alongside it. Tuning in every fold was rejected on measured runtime (about 15–20 extra minutes). | P3-20 |
| 4 | Plan "totally rejects" cross-asset covariance and diversification | **Premise rejected, useful additions accepted** | The plan never ignored correlation: the class-ordered correlation heat map and the 4 × 4 block table were already in §3.1, and the inverse-volatility comparison the reviewer asks for *is* the risk-parity benchmark, which P1 is tested against. The only thing rejected was covariance-*optimised* weights, which are not taught and ill-posed with 50 assets and 36–96 months (L1 p.63). Sub-period correlation tables and a correlation-aware class risk budget are worthwhile. The reviewer's "16 assets" and class labels (A = equities, …) do not match the data: there are 50 assets (A 26, B 8, C 9, D 7), the labels are anonymised, and the evidence points to A commodity-like, B currencies, C equity indices, D government bonds (DP §C). | P3-22 |
| 5 | Add asset-class-specific ridge models | **Already in the plan; extended** | Per-class ridge on M2 was already a core specification (PROJECT_PLAN §3.3), and M3 already has class-specific macro slopes. Extending per-class estimation to M3 completes the pooled-vs-segmented comparison at little cost. | P3-23 |
| 6 | Answers to the open questions | **Recorded** | The reviewer's closing instruction to "proceed to execute the code" is superseded by the student's instruction to wait for the go-ahead. | §I, §D, §H |
