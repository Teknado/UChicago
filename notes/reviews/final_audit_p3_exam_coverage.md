# Problem 3 coverage audit: what the exam asks for and invites, against what was delivered

Scope: exam cells 1, 44–49 (read line by line) against our cells 50–96 of the executed notebook (`scratchpad/audit/nb_text.md`),
METHODOLOGY_LOG §E–F and the pre-modelling profile (`scratchpad/notes/data_profile.md`). Nothing in the repo was modified.
Spot checks run in the scratchpad: the update frequency of the curated and global macro series, and boosting fit times on synthetic data.

Status key: **D** = delivered, **P** = partial, **M** = missing, **S** = deliberately skipped (with the reason and whether it is sound).
Severity for P/M/S: FATAL / MAJOR / MINOR / NIT (as defined in CONTEXT.md). "WU §n" = section n of the write-up in cell 96.

---

## A. Numbered requirement table

### A.1 General instructions (cell 1) and the project framing (cell 44)

| # | Requirement / statement (exam text, shortened) | Status | Where delivered | Notes, severity |
|---|---|---|---|---|
| G1 | "Every number must come from a cell in this notebook" | D | cell 94 Table 3.31 collects the numbers the write-up quotes; others are in Tables 3.1–3.30 | NIT: WU §3 "20-fold spread in volatility" is not printed. 17.40%/0.82% = 21.2, and that ratio is computed by the reader. "Three to ten years" is derived from 36–120 months, which is fine |
| G2 | "an answer you cannot explain or reproduce is your error" | D | fixed seeds (cell 51), harness unit tests (cell 72), truncation causality test (cell 66), leakage audit (cells 91–92) | — |
| P0a | Design is yours; "the write-up must defend it" | P | cell 50 (design), cell 63 (feature table with reasons), WU §3 | MINOR: the defence of *skipped* options is thin in the write-up (see S-items, A.7 and W2.7/W3.1) |
| P0b | "A negative result, honestly established, earns full credit; an unexplained positive one does not" (grading) | D | Every positive number is explained or labelled: P1 Sharpe 0.44 (class-mean tilt, Tables 3.27–3.28b); rolling P1 0.60 ("not a result"); the per-asset-benchmark t = 2.18 (shrinkage toward class means, Table 3.10c); x10 at 14 months +0.15% at about 2 SE (one of nine checks); the 2018–24 α t = 2.47 (post hoc) | Verified OK |
| Q1 | "How much of next month's excess return is forecastable from lagged characteristics?" | D | cell 75 Tables 3.10/3.10b/3.10c; cell 76 Tables 3.11–3.13; cell 80 Fig 3.6a/b; WU §4.1 | Also answered in raw units by class (Table 3.13). The detectable effect is printed (0.48%/0.68%) |
| Q2 | "Does macroeconomic information — global or country-level — add anything?" | D | cell 82 Table 3.18 (global-only and country-only separately); cell 83 Tables 3.19–3.20, Fig 3.7; cell 84 Table 3.21; WU §4.2 | Country macro reaches only 22 of the 50 assets: class A gets none, and the two country-7 assets (asset_24 C, asset_27 D) have a differential identically 0. The write-up does not say this about the country-7 assets (NIT) |
| Q3 | "Does a forecast-based portfolio beat the simple rules … out of sample, after costs?" | D | cells 86–89 Tables 3.24–3.29, Fig 3.8a–c; WU §4.3 | — |
| M0 | "Any method is allowed: OLS, ridge, lasso, PCR, KNN, trees, random forests, boosting, neural networks, …" | P / S | Done: OLS, ridge, lasso, PCR, RF (leaf 200), deep RF (leaf 5), per-class ridge (cells 71, 74). Not done: KNN, a single CART tree, boosting, NN | S. Reasons are in METHODOLOGY_LOG §F only, never in the write-up. **KNN** regression is not taught (L6 teaches the classifier): sound. **NN** is only named (L8 p.62): sound. **CART** alone: no reason given; the forest dominates it (L8 p.61), so this is defensible. **Boosting** is taught (L8 p.50–55); the reason given ("a tuning budget is a leakage risk", L8 p.55, plus runtime) is **weak**, because L8 p.55's own remedy ("Tune on a validation split, report on a test split you touched once") is exactly the harness already built. MINOR. See LOT #4 |
| M1 | "static, expanding-window or rolling-window estimation" | D | all three schemes for OLS, ridge, RF, lasso and PCR (cell 71 `window`; Table 3.10) | — |
| M2 | "cross-validation or information criteria for tuning" | D | three forward annual validation folds per window (cell 71 `inner_folds`); ties go to the heavier penalty | Information criteria (L4) were not used; this is optional |
| GR | "the discipline of the evaluation is what is graded" | D | pre-registration (cells 50–52), a 38-spec ledger with Bonferroni 3.27, one forward-only splitter with asserts, `n_samples_seen_` asserts, truncation test, leakage audit table (cell 91), leak alarm | Strongest part of the project. The honest timing caveat (design first committed after the development ledger ran) is disclosed in §3.10 item 6 and WU §3 |

### A.2 Data section (cell 45), including its hints, lag rules and data traps

| # | Requirement / hint | Status | Where | Notes, severity |
|---|---|---|---|---|
| D1 | Load all files from `_DATA_DIR`; month-ends 2000-01..2024-12; excess returns in decimals | D | cell 46; cell 54 integrity asserts | — |
| D2 | `asset_returns_wide.csv` = the same returns | D | cell 54 (max abs diff = 0 asserted) | — |
| D3 | Extended file: "a few of its columns duplicate curated series exactly; find them before you concatenate" | D | cell 57: 6 exact duplicates (x73=x10, x89=x12, x114=x13, x17=x14, x92=x15, x18=x16), plus 26 series common to all countries (4 of them copies of x6–x9); never concatenated | Verified OK |
| D4 | "Treat [extended] as exploratory material of uneven quality" | D | cell 57 (frequencies: 35 annual, 26 quarterly; 7 gap columns); only x86 used | — |
| D5 | Universe: 50 assets, A26/B8/C9/D7, 12 countries; B, C, D each have one country | D | cell 54 asserts + crosstab | — |
| D6 | Class A has no natural country (convention: country 7); choose a treatment (see step 2) | D | cell 63 table ("Class A gets none"); WU §2 | Sound (no natural country; parsimony). The two country-7 C/D assets also get zero country macro; not stated (NIT) |
| D7 | x1–x5 are cross-sectional characteristics; "one of the five is a risk measure" | D | cell 61: x5 = SD36 in 99.67% of rows; x2 = R12 in 99.63% | — |
| D8 | x6–x164: "Working out what a variable does — from its persistence, its scale, its cross-sectional behaviour and its relation to returns — is part of the project" | **P** | Characteristics: done (Tables 3.4–3.5). Macro: only x10 (same-year mean of x86), x11 (≈ x86 − x12) and x86 (monthly short rate) are identified. Table 3.18b gives the cross-country variance share and Table 3.20 the persistence of the *transformed* inputs against shifted copies | **MAJOR (cheap fix).** No cell prints scale, persistence or update frequency for x6–x9 and x12–x16, and no hypothesis of what each measures. The write-up never says what the macro variables are, so the Q2 answer is purely statistical ("macro adds nothing"). The pre-modelling profile already has the hypotheses (x6 VIX-type; x7 smoothed stress index; x8 activity/sentiment z-score; x9 yield/growth-type; x12 CPI inflation YoY; x13 effective-FX YoY; x14 EPU-type; x15 geopolitical-risk-type; x16 ICRG-type rating), but none of this is in the notebook |
| D9 | Lags: x2, x5 already lagged, use as is; x1, x3, x4 and **every macro variable** contemporaneous, shift one month within asset/country | D | cell 64 `build_features`; cell 66 (20 lag spot checks, truncation test on all 64 columns); identities checked in cell 61 | Verified OK |
| D10 | Macro released with a delay; "a one-month lag is the minimum, and you may argue for more" | D | 2-month lag, with 1 month as robustness (cells 51, 63; Table 3.21: lag 1 is −0.37% worse); WU §4.2 | The argument is brief ("1 month minimum plus release delay"). Fine |
| D11 | Leading months of x2, x3, x5 (2000–02) back-filled with the first observed value | D | cell 55: 8 (variable, asset) pairs in 7 assets; set to NaN; the last fill (2002-07) comes before the first target (2003-01) | Verified OK |
| D12 | "x11 stops being updated before the end of the sample for three countries" | D | cell 56 table (countries 12, 9, 11; no interior holes) + Fig 3.1b | — |
| D13 | "In the extended file, seven columns have gaps" | D | cell 57 lists exactly 7 | None is used, so no treatment is needed |
| D14 | "You are free to bring in more data … document its source and its release timing" | S (silent) | — | MINOR: not done, and the notebook and write-up never say why. The obvious sound reason is that anonymised assets and countries make external data unmappable, except for global series |
| D15 | "Scales differ enormously by class … Think about this before pooling assets in one regression or one equal-weighted portfolio" | D / P | Regression: vol-scaled target plus within-class ranks (cells 50, 63; WU §2–3). Portfolio: Table 3.9b (gross shares), class-risk-budget variants (Table 3.24) | NIT: the write-up never says that the exam's EW benchmark is dominated by class-A risk (52% of gross exposure in the 30%-vol class) |
| D16 | Implied trap: macro stored with period-average stamping (x10) | D (bonus) | cell 56 Fig 3.1a (corr 0.998 same-year vs 0.839 prior-year), Table 3.22 leak demonstration | Found without being told. A strong point |
| D17 | Implied traps: stale x1 for 5 currencies (2024-10..12); x5 floored for asset_16 | D | cell 55; evaluation-only cut in Table 3.23 | — |
| D18 | Implied: are the *used* curated/global series also low-frequency stamped like x10? | P | the frequency audit (cell 57) covers only the extended file | MINOR: my spot check found that x6–x9, x12–x15 and x86 change in every calendar month, and x16 in 63% of months (a stepped rating), so none follows x10's January-only pattern and there is no leak. The notebook does not print this check for the series it actually uses. Add one line to cell 57 |

### A.3 "Two rules that apply throughout" (cell 47)

| # | Rule | Status | Where | Notes |
|---|---|---|---|---|
| R1a | Benchmarks before models: write down the rules and compute their OOS performance before fitting anything | D | cell 50 (written down); Section 3.3 cells 67–69 (Tables 3.8, 3.9), with no model code before cell 71 | — |
| R1b | Forecast benchmark: the trailing mean (or zero) | D | pooled (primary), per-class, per-asset, zero (Table 3.8); raw per-asset mean (Table 3.13) | — |
| R1c | Portfolio benchmarks: EW; RP ∝ 1/σ̂ (trailing vol from past returns); TSMOM ∝ sign(R12)/σ̂; unit gross; monthly rebalancing | D | cell 69 (unit gross asserted; truncation causality test on the weights; Table 3.9) | — |
| R1d | "All three can be built from excess_return alone" | D | built from `R` only | — |
| R2a | Fix the OOS window and initial training block in advance, before fitting anything | D | cells 50–51 (asserted dates) | The timing caveat is disclosed (§3.10 item 6) |
| R2b | "… and say why you chose them" | **P** | cell 50 gives the reason for the 2003-01 start (σ̂ exists, back-fills excluded) and says the block includes 2008 | MINOR: no reason for splitting at 2010/2011 rather than elsewhere appears in the notebook. METHODOLOGY_LOG P3-01 has it (180 vs 168 OOS months cuts the SE by only about 3.5% but shrinks class D's training rows from 672 to 504). WU §3 states the windows with no reason at all |
| R2c | Every predictor known at the end of t−1 | D | cell 66 | — |
| R2d | Every scaler, penalty and hyper-parameter chosen on training data only | D | cell 71 (scaler fitted per fit window, asserted; forward folds asserted); forest untuned | — |
| R2e | Trailing-mean benchmark through t−1, updated monthly (the HW5 1.1 harness) | D | cell 68 (unit tests against brute force); the per-asset version (Table 3.10c) is the literal HW5 harness | — |

### A.4 "A suggested workflow" (cell 48): "a complete project touches each step"

**Step 1: Know your data**

| # | Sub-item | Status | Where | Notes |
|---|---|---|---|---|
| W1.1 | Cumulative excess returns by asset class | D | Fig 3.2 (cell 58); WU §2 | — |
| W1.2 | Per-asset table of annualised mean, vol, Sharpe, worst month | D | Table 3.1b (+3.1a by class) | — |
| W1.3 | Correlation matrix ordered by class | D | Fig 3.3, Table 3.2 (full sample and training block) | — |
| W1.4 | "How strong is the block structure, and what does it imply for pooling?" | D | Table 3.2; cell 52 (effective N ≈ 8 for class-level signals); month-block bootstrap (cell 70); WU §1 and §5 | NIT: the implication is spread over three places, and WU §2 links pooling only to volatility, not to correlation |
| W1.5 | Rolling 12-month Sharpe by class; "how persistent is performance?" | D | Fig 3.4, Table 3.3 | NIT: Fig 3.4 is never referenced in the write-up |
| W1.6 | Distribution, scale and persistence of each characteristic within each class | D | Table 3.4 (mean, sd, p1, p99, AC1, AC12, cross-sectional sd) | No figure; acceptable |
| W1.7 | First look at predictive content: next-month return by tercile sorted each month | D | Tables 3.6/3.6b, **training block only**, within class, in y units | Restricting the sort to the training block is deliberate and sound (anti-snooping, L4 p.43, L5 p.41). The OOS version was never run as an *evaluation* after the design was fixed (LOT #5) |
| W1.8 | A hypothesis for each of x1–x5, "you will need it to interpret your results" | D / P | cell 61, Table 3.5; WU §2; used in WU §5 point 2 | MINOR: for x1, only class-B evidence is printed ("no comparable evidence for the other classes"), although the profile found x1 in class D correlates 0.87 with the term spread (x85 − x86) |

**Step 2: Feature engineering**

| # | Sub-item | Status | Where | Notes |
|---|---|---|---|---|
| W2.1 | Apply the lag rule | D | cells 64, 66 | — |
| W2.2 | Standardise characteristics cross-sectionally within month (all or within class; z or rank) "and say why" | D | within-class rank (cell 63 table; reason: units, bounded, no fitted parameters); z-score robustness (Table 3.21) | — |
| W2.3 | Missing macro: diagnose the pattern per series and country (stops early or holes?) | D | cell 56 (x11: last non-missing, missing months, interior holes per country) | The extended gap columns were not diagnosed; they are unused, so the item doesn't apply |
| W2.4 | Treatment uses only information available at the time (ffill or drop OK; bfill, interpolation, full-mean not) | D | x11 dropped; back-fills set to NaN; nothing imputed (cell 91) | — |
| W2.5 | Trap: a series that stops during a regime change; stale ffill is not a small error | D | x11: forward-fill error up to 10.1 pp (country 12), Fig 3.1b; WU §2 item 3 | — |
| W2.6 | Can the series be rebuilt from other complete columns? Report how close | D | x86 − x12: pooled corr 0.988, RMSE 0.354; per-country table (cell 56) | MINOR imprecision: "exact combination of two included series" holds for the raw levels, but the inputs are per-series differentials each z-scored on its own trailing SD. The z-scored rebuild is therefore *not* spanned exactly by the included z-scores. The rebuilt series was never tried as an input |
| W2.7 | "Add your own data if you have a hypothesis for it" | S (silent) | — | MINOR (see D14) |
| W2.8 | Standardise global macro with trailing information only | D | trailing 60m z (min 24), then lag 2 | — |
| W2.9 | Map country macro through `country`; a differential against base country 7 | D | cell 64 | — |
| W2.10 | Class A: country 7 / cross-country average / dispersion / none; "say which and why" | D | none; reason in cell 63 and WU §2 | Sound. The alternative (a cross-country average or dispersion for class A, 26 of 50 assets) was not run as a sensitivity (LOT #9) |
| W2.11 | "Consider cross-sectional macro transforms": rank across the 12 countries, deviation from the cross-country mean, widened panel | S | WU §5 "What we did not try": "not in the pre-registered list" | MINOR. The reason is procedural, not substantive: other post-hoc diagnostics were added and labelled, so a labelled post-hoc check was possible (LOT #7) |
| W2.12 | "Consider interactions between characteristics and macro states" | S | WU §5, same reason | MINOR. The write-up misses its own best argument: the forest on M3 (18 inputs incl. ranks and macro) can express these interactions and loses (−1.47% expanding; Table 3.10) (LOT #6) |
| W2.13 | … class dummies | D | unpenalised class intercepts; macro × class slopes; per-class ridge; forest class dummies | — |
| W2.14 | … "and the extended file if you have a hypothesis for it" | S | WU §2 item 4, Limitations | Sound (no hypothesis; unknown release timing of the low-frequency columns; multiple testing, L4 p.25–28). It could be strengthened: the main extended-file hypotheses (rate differential = carry, term spread, value) are already inside x1/x3/x4 |
| W2.15 | Choose the target (vol-scaled recommended), explain why and how forecasts become positions | D | cell 50; WU §3; P1 = ŷ/σ̂ (collapses to RP when the forecast is constant, asserted in cell 69) | — |
| W2.16 | "End with one table listing every predictor and the count" | D / P | Table 3.7 (blocks, transforms, lags, counts) + per-set column counts | MINOR: the forest's input sets (`RF_SETS`: M2 = 5 ranks + 3 dummies; M3 = + 4 un-interacted global + 6 country series) appear in neither Table 3.7 nor the write-up |

**Step 3: Models and evaluation schemes**

| # | Sub-item | Status | Where | Notes |
|---|---|---|---|---|
| W3.1 | Fit promising models: OLS, ridge, lasso, RF, **boosted trees**, NN, … | P / S | see M0 | Boosting skipped for a weak reason (MINOR); NN not taught (sound). Neither is explained in the write-up |
| W3.2 | Evaluate under static / expanding / rolling | D | Table 3.10 | — |
| W3.3 | R²_OOS against the trailing mean **and** against zero, model × scheme | D | Table 3.10 (two pivots), Table 3.10b (all 38 with SEs) | — |
| W3.4 | … and by asset class for the models you care about | D / P | Table 3.11 (8 models, vs pooled mean); Table 3.13 (raw units, vs asset mean and zero) | NIT: y-units by class **vs zero** is not tabulated (it can be inferred from Table 3.8) |
| W3.5 | "Which scheme wins and why" | D | Table 3.12b (paired); WU §4.1 ("static goes stale"), linked to drifting class means in §4.3 | — |
| W3.6 | "Which predictors carry the signal and whether that is stable across windows" | D / P | Table 3.15 + Fig 3.5a (expanding coefficients), Fig 3.5b (block sizes), Table 3.16 (forest OOS permutation importance by test year), lasso selection counts | MINOR: stability is shown only across *expanding* windows, which overlap heavily (the write-up admits this). The rolling-window coefficients are already stored (`LOGS['ridge|M2|rolling']`) but never shown (LOT #8) |
| W3.7 | "Whether macro adds anything once you refit without it" | D | Table 3.18 (14 paired comparisons) | Design note (not a coverage gap): ridge M3 shares one penalty across 5 characteristic and 34 macro slopes, so adding macro also pushes the characteristic slopes to the no-signal edge (93% of refits). Lasso, OLS and RF corroborate the Q2 answer, so the conclusion stands |
| W3.8 | Placebo: every macro series circularly shifted by 36 months; "try more than one shift" | D | 8 shifts (36..120), Table 3.19, wrap disclosure Table 3.19b, persistence Table 3.20, Fig 3.7 | Run for the primary ridge M3 only; acceptable |
| W3.9 | "Whether per-class models beat the pooled one" | D | Table 3.17 (ridge M2 and M3) | — |

**Step 4: From forecasts to portfolios**

| # | Sub-item | Status | Where | Notes |
|---|---|---|---|---|
| W4.1 | Form portfolios from forecasts (∝ forecast/vol, sign of forecast, top-minus-bottom sort, own rule) | D | P1 ∝ ŷ/σ̂, P3 within-class long–short, class-risk-budget versions, diagnostics on M1, static and rolling | The sign rule and the top-minus-bottom sort were not run. They are optional ("whatever way you find sensible"), and the value is low (LOT #10) |
| W4.2 | State clearly how the portfolio is formed and rebalanced | D | cell 85; WU §3 | — |
| W4.3 | Sharpe at a minimum; return, vol, drawdown, turnover, net of costs, cumulative-return plots, sub-periods | D | Tables 3.24–3.26, Fig 3.8a–c (cumulative, drawdown, cost curve), Table 3.25 | All listed metrics are present |
| W4.4 | "Where you find outperformance, ask how much of it is exposure to the benchmark rules and how much is timing" | D | Tables 3.27, 3.27b, 3.28, 3.28b; P1 on M1 (weight corr 0.9994); frozen 2010 tilt (0.27 < RP) | The strongest analytical part of Q3 |

### A.5 "What to submit" (cell 49)

| # | Requirement | Status | Where | Notes |
|---|---|---|---|---|
| S1 | The analysis runs top to bottom on the class server, or with `_DATA_DIR` pointed at a local copy | D (not re-run by me) | cell 46: the exam's server line is kept, with a fallback to `./`; Problem 3 re-declares its own style and seed; runtime 9.0 min (Table 3.31) | The course environment (pandas 3.0.5, sklearn 1.9.0; log §A.2) matches the version-sensitive calls (`freq='ME'`, `include_groups=`) |
| S2 | "with every figure and table the write-up refers to" | D | All 26 tables and all figures cited in cell 96 exist above (checked) | — |
| S3 | A write-up in markdown at the end of the notebook | D | cell 96 | — |
| S4 | "in the form of a research paper … the way an empirical finance paper is written" | **P** | all six required sections are present, in order | MINOR–MAJOR (presentation; high leverage). The paper is mostly nested bullet lists, not prose. It contains no main-results table (model × scheme R²_OOS ± SE) and **no figure**: all 8 figures and ~30 tables are references back into §3.0–3.10. Details in part C |
| S5 | Introduction: question; why it matters to a cross-country/asset allocator; what the literature leads you to expect; a one-paragraph preview | D / P | WU §1 | MINOR: "why it matters" is one sentence. The literature covers Q1 only (AMP 2013, MOP 2012, GKX 2020). Nothing on the carry literature, although x1 is identified as carry. Nothing for Q2/Q3 (the classic OOS failure of macro predictors; 1/N-type results). A technical power paragraph sits in the Introduction; it belongs in Methodology. The preview is bullets, not a paragraph |
| S6 | Data: contents; treatment (missing values, lags, standardisation, country mapping); summary statistics and "the plots that convey the structure of the data" | D / P | WU §2 (class table, trap list, features) | The macro variables are never described (D8, MAJOR). Plots are referenced (Figs 3.1–3.3), not shown |
| S7 | Methodology: benchmarks and why; models; schemes and OOS window; tuning; positions; performance measures; "reproduce your design from this section alone" | D / P | WU §3 | MINOR reproducibility gaps: the forest's input set; the z-score minimum of 24 months; the zero country macro for the country-7 C/D assets; the reason for the window choice (R2b) |
| S8 | Results: tables and figures, in the order Q1, Q2, Q3 | D / P | WU §4.1–4.3 in order | Only one small table (net Sharpe ratios) is in the text. **Fig 3.6a** (cumulative SSE gain against the trailing mean, the canonical R²_OOS figure) is never referenced |
| S9 | Interpretation: economics; why; strong vs fragile; "what you tried that did not work"; next steps | D / P | WU §5 (plus a Limitations list) | MINOR: "tried and rejected" omits boosting, KNN, NN, covariance-optimised weights and external data, although METHODOLOGY_LOG §F has the list (the AI guide p.3: "Every write-up should be able to say what you tried and rejected"). The null is not related to the documented post-2010 weakness of value, momentum and TSMOM premia |
| S10 | Conclusion a PM could read on their own | D | WU §6 | Readable and actionable |
| S11 | "No length requirement … length should follow from what you have to say" | D / P | ~5,000 words (cell 96) | MINOR: the length itself is fine, but there is a lot of repetition: the "about 0.5%" detectability point appears 4 times, "pre-registered" 12 times, "not a result" caveats several times. The notebook-engineering items in Limitations (warnings silenced, forest runtime) are not research limitations |

---

## B. MISSING and PARTIAL items by severity

No FATAL items. Nothing the exam *requires* is missing outright. All the gaps are partial deliveries or silent skips.

**MAJOR**
1. **D8 / S6: the macro variables are never characterised.** The exam states that working out what each variable does (persistence, scale, cross-sectional behaviour) "is part of the project". The notebook does this for x1–x5 but not for x6–x9 and x12–x16, and the write-up never says what the macro variables are. Fix: one descriptive cell (predictors only, so no leakage) and a hypothesis column, plus two sentences in WU §2.
2. **S4 / S8: the write-up is not yet a research paper in form.** It has no in-paper results table and no in-paper figure; the numbers sit in bullet lists that point to ~30 notebook tables. This is presentation, but the write-up is what gets graded, so it matters. (I'd call it MINOR by the strict definition, but its grading leverage makes it MAJOR in practice.)

**MINOR**
3. M0 / W3.1: boosting (taught, L8 p.50–55) is skipped for a weak reason, and neither boosting, KNN nor NN is mentioned in the write-up.
4. S9: the "tried and rejected" paragraph is incomplete (boosting, KNN, NN, covariance-optimised weights, external data, extended-file hypotheses). The reasons are in METHODOLOGY_LOG §F.
5. R2b / S7: no stated reason for the 2003/2010/2011 windows in the write-up, and only a partial reason in cell 50.
6. W2.11 / W2.12: cross-country macro transforms and characteristic × macro interactions are skipped for a procedural reason. The forest-on-M3 argument is not made.
7. D14 / W2.7: bringing in outside data is skipped silently.
8. W3.6: coefficient stability is shown only across overlapping expanding windows.
9. W2.16 / S7: the forest's input sets are not in Table 3.7 or the write-up.
10. D18: the frequency and stamping audit is not printed for the curated and global series actually used. My spot check says they are clean.
11. W2.6: "x11's rebuild is an exact combination of included series" is only approximately true after the trailing-z transform.
12. S5: thin "why it matters"; no literature for Q2 or Q3; no carry citation; the power paragraph is misplaced in the Introduction.
13. S11: repetition; notebook-engineering items listed as limitations.
14. W1.8: x1 is identified as carry for class B only, although the profile has evidence for class D.

**NIT**: G1 ("20-fold" not printed); W3.4 (y-units by class against zero); W1.5 (Fig 3.4 not referenced) and S8 (Fig 3.6a not referenced); D6 (the country-7 C/D assets get no country macro, unstated); D15 (EW's risk is concentrated in class A, not discussed); W1.4 (the pooling implication is spread over three places).

---

## C. Write-up (cell 96) against a grader's research-paper expectations

| Expectation | Present? | Comment |
|---|---|---|
| The six required sections in order | Yes | Introduction, Data, Methodology, Results (4.1 Q1, 4.2 Q2, 4.3 Q3), Interpretation and discussion, Conclusion |
| Title | Yes | "Characteristics, macro and cross-country asset returns: a pre-registered out-of-sample study" |
| Abstract | No | Not required; the Preview works as one. A five-line abstract before §1 would help |
| A table of main results in the paper | **No** | The single most expected element. Only the class summary (§2) and the net-Sharpe list (§4.3) are in the text. Suggest three in-paper tables: (1) R²_OOS ± SE vs the pooled mean and vs zero for ~8 key specs × 3 schemes; (2) macro gain vs placebo range; (3) portfolio net Sharpe, α(t), turnover and drawdown |
| Figures in the paper | **No** | All are references into §3.x. Suggest embedding 3–4: Fig 3.6a (cumulative SSE gain), Fig 3.7 (placebo), Fig 3.8a (net cumulative returns), plus a new class-share-of-gross figure for the Q3 tilt story. Options: (a) save PNGs in the analysis cells and reference them from markdown (needs a writable directory; the images show only if the files travel with the notebook); (b) Jupyter markdown attachments (static copies); (c) short display-only code cells between write-up markdown cells (`display(fig)` on stored figure handles), which keeps "runs top to bottom" exact |
| Figures and tables referenced | Mostly | Fig 3.4 and **Fig 3.6a** are never referenced |
| Limitations section | Yes | Inside §5. Two items are notebook housekeeping (warnings, runtime) and should move to §3.10 |
| "What we tried that did not work" / "did not try" | Partly | See B.4 |
| Prose readability | Weak | Nested bullets throughout §3–§6. Each results subsection needs a short topic paragraph that states the finding, then the table. The repeated caveats should be cut |
| Length | Acceptable | ~5,000 words. With repetition removed and bullets turned into paragraphs, ~3,500 words would say the same thing more clearly |
| Every number traceable to a cell | Yes | Table 3.31 (one NIT: "20-fold") |

---

## D. Deliberately skipped items: is the reason sound?

| Item | Stated where | Reason | Sound? |
|---|---|---|---|
| Full-sample tercile sorts | cell 53, 62 | OOS returns would shape the design | Yes |
| x11 forward fill / as a separate column | cell 56, WU §2 | Stale through the regime change; its rebuild is (approximately) spanned | Yes (the "exact" wording is loose, W2.6) |
| x10 at a 1-month lag | cell 56, Table 3.22 | Same-year-average look-ahead | Yes |
| Extended-file screening | cell 57, WU §2, Limitations | No hypothesis; multiple testing; unknown timing | Yes |
| Class A country macro | cell 63, WU §2 | No natural country; parsimony | Yes |
| KNN regression | log §F only | Not taught (L6 teaches the classifier); curse of dimensionality | Yes, but it should be said in the write-up |
| Neural networks | nowhere | Only named, L8 p.62 | Yes, but it should be said |
| Boosted trees | log §F / P3-12 only | "A tuning budget is a leakage risk" (L8 p.55); runtime | **Weak.** L8 p.55's remedy is a validation split, which the harness already has. Runtime is a real but moderate cost (below) |
| Covariance-optimised weights | log §F; WU next steps only | Not taught; 50×50 covariance from 36–96 months is ill-conditioned (L1 p.63) | Yes; should be said in §5 |
| Forest leaf tuning | log §F | Runtime; the forest is nearly tuning-free (L8 p.55); a deep forest is run as a contrast | Yes |
| Cross-country macro transforms, characteristic × macro interactions, widened panel | WU §5 | "Not in the pre-registered list" | Procedural only. Fine for the headline, but not a barrier to a labelled post-hoc check |
| Outside data | nowhere | — | A sound reason exists (anonymised identifiers) but is not stated |

---

## E. Left on the table, ranked by likely grading value

Value = expected effect on the grade (the exam grades evaluation discipline and the research-paper write-up). Every new model run is post hoc: it must be labelled as such, kept out of the headline verdicts, and reported against a Bonferroni value recomputed for 38 + k tests.

| Rank | Item | Why the exam invites it | Cost | Leakage / multiple-testing risk | Scope (8 lectures) | Recommendation |
|---|---|---|---|---|---|---|
| 1 | **Make the write-up a paper**: 3 in-paper tables (main R²_OOS ± SE; macro vs placebo; portfolios), 3–4 embedded figures (3.6a, 3.7, 3.8a, class shares), topic paragraphs instead of nested bullets, repetition cut to ~3,500 words, the power paragraph moved into Methodology, housekeeping moved out of Limitations | cell 49: "write it the way an empirical finance paper is written"; "Results. Tables and figures with the numbers" | Low–moderate (text; one display mechanism) | None, if every number stays printed by a cell (they all are, in Table 3.10b/3.31) | n/a | **Do** |
| 2 | **Complete the design defence in the text**: why these windows (P3-01 trade-off); why no boosting, KNN, NN, covariance optimisation or outside data; why the extended file adds little (carry, term spread and value already in x1/x3/x4); the forest's input set; the forest on M3 as an implicit test of characteristic × macro interactions; the country-7 C/D assets get zero country macro | cell 44 "the write-up must defend it"; AI guide p.3 "say what you tried and rejected"; cell 47 "say why you chose them" | Trivial (sentences; the reasons already exist in METHODOLOGY_LOG §E–F) | None | n/a | **Do** |
| 3 | **Macro variable characterisation table** for x6–x9, x12–x16, x86: scale, AC(1), AC(12), update frequency (the D18 check), share of cross-country variance, correlation with x86/x6, and a hypothesis column (VIX-type, stress, activity, yield, CPI YoY, FX YoY, EPU, GPR, ICRG). Add the class-D x1 vs term-spread check (W1.8) | cell 45: "Working out what a variable does … is part of the project"; gives Q2 an economic reading | Low (one cell; the numbers exist in the profile) | None (predictors only, never returns) | L1 (descriptive), L3 (transforms) | **Do** |
| 4 | **Boosted trees** (GradientBoostingRegressor, L8 p.53) on M2 and M3, expanding. ν fixed (e.g. 0.05), depth 1–2, subsample 0.8; B chosen with `staged_predict` on the same 3 annual folds; refit on the window | exam lists boosting twice (cells 44, 48); L8's "if you have a day, boost" lesson; the only taught family in the list that was not run | Moderate. Timed on synthetic data here: about 3–12 s per 500-round fit at 8,000 rows; 14 refits × 4 fits ≈ 3–11 min per spec, so +5–15 min on top of the current 9 min | Never use `n_iter_no_change` / `validation_fraction` (a shuffled internal `train_test_split`: FATAL, AI guide §4a, log §A.2). +2 specs → Bonferroni ≈ 3.29. The result is very likely negative (the forest is) | Taught (L8 p.50–55) | **Do if runtime allows** (M2 only, or depth 1, halves the cost). Otherwise state the reason honestly in the write-up (item 2) |
| 5 | **OOS univariate characteristic evidence**: within-class tercile spreads (or rank-weighted, vol-scaled long–short portfolios) for x1–x5 on 2011–2024, next to the training-block Table 3.6 | cell 48 step 1 "first look"; step 3 "which predictors carry the signal and whether that is stable across windows"; cell 44's literature claim; a direct, model-free Q1 answer per characteristic | Trivial (reuse the cell-62 code on OOS months) | Evaluation only, since the design is fixed. 25 tests: the Bonferroni 3.18 is already printed. Must not feed back into any spec | Sorts are descriptive; multiple testing L4 p.25–28 | **Do** (cheap; strengthens the "characteristics were given a fair chance" claim in WU §5) |
| 6 | **Characteristic × macro-state interactions**: e.g. the 5 ranks × lagged trailing-z x6 (VIX-type), one ridge spec, plus the same 8 placebos | cell 48 step 2 "Consider interactions …"; the literature's main macro channel is conditional (momentum and carry crash in high-volatility states); WU §5 admits it wasn't tried | Low (5 extra columns in `build_features`; about 1 s per ridge run) | +1 post-hoc spec; the placebo guards against luck | L3 p.53–54 (interactions) | **Do, labelled post hoc**, or at minimum make the forest-on-M3 argument in the text |
| 7 | **Cross-country macro transforms**: each country-macro input as its rank across the 12 countries, or its deviation from the cross-country mean, in place of the differential vs country 7 | cell 48 step 2 "Consider cross-sectional macro transforms" | Low (single-date transforms in `build_features`) | No leakage (single-date). +1 post-hoc spec | L3 (transforms) | Optional; low expected change (Q2 is negative across 14 comparisons) |
| 8 | **Coefficient stability across rolling windows**: a figure of ridge/OLS M2 rolling coefficients by refit (OLS with ±2 SE, L2) | cell 48 step 3 "stable across windows"; the write-up admits the expanding windows overlap | Trivial (`LOGS` already hold the rolling coefficients) | None | L2 (SEs), L5 | Optional; cheap depth |
| 9 | **Class-A country macro sensitivity** (cross-country average or dispersion, cell 48 step 2) and/or a widened panel | cell 45 / cell 48 name these explicitly; class A is 26 of the 50 assets | Low | +1 post-hoc spec | L3 | Low priority; one sentence of defence (item 2) may suffice |
| 10 | Extra portfolio rules: sign of the forecast; top-minus-bottom tercile | cell 48 step 4 lists them | Trivial | More portfolio rules means more multiple testing on Q3; with R²_OOS ≈ 0 they add little | EXAM-DEFINED options | Skip, or report one as a labelled diagnostic |
| 11 | Breadth across lectures: logistic sign classifier (L6; use `C=np.inf`, not plain `LogisticRegression()`), with hit rate, sensitivity and specificity; hierarchical clustering of the return-correlation matrix (L7) to confirm the class blocks | shows use of L6/L7 | Low | Classifier: +1 spec. Clustering: descriptive only | Taught (L6, L7) | Low value; the clustering is a nice single figure for W1.4 if there is time |
| 12 | Summary figure: all 38 R²_OOS ± 2 SE (dot plot with the Bonferroni band) | presentation of Table 3.10b | Trivial | None | L2 p.85, L4 | Fold into item 1 |
| 13 | NITs: y-units by class vs zero (Table 3.11 companion); EW risk share by class; print the "~21-fold" ratio | W3.4, D15, G1 | Trivial | None | — | Fold into items 1–3 |

---

## F. Verified OK (no action)
- All three questions are answered with pre-registered rules, and the verdicts are applied exactly as written (cells 75, 83, 89).
- All four data traps the exam alludes to were found (back-fills, x11 stop and regime change with a quantified rebuild, extended-file duplicates, gap columns), plus three it does not mention (x10 look-ahead, frozen x1, x5 floor).
- Both of the exam's "two rules" are followed in letter and spirit, including the exact benchmark definitions with unit-gross asserts.
- Every sub-item of workflow steps 1, 3 and 4 is delivered. In step 2, every non-optional sub-item is delivered, and the optional "consider" items are skipped openly.
- Every figure and table the write-up cites exists in the notebook, and every number in the write-up comes from a cell (one NIT).
