# Review of the Financial Analytics final reference solution

**Scope.** `Final_project_claude.ipynb` and `DECISION_LOG.md` as of `financial_analytics` @ `e63ed7c`, re-checked against:
- the exam (`Final_Autumn_2026-1.ipynb`);
- the data files;
- the eight lecture PDFs and `AI_Coding_Guide.pdf`.

**Method.**
1. Re-executed the notebook: every printed number reproduces.
2. Re-derived each number and claim with independent code.
3. Six parallel auditors (one per area), each followed by an independent verifier whose job was to refute every finding, plus a completeness pass against every exam requirement.
4. A final independent check of the corrected notebook, this review and the decision log: numbers against outputs, code and look-ahead, exam coverage and citations, again with verifiers. Its confirmed issues were fixed before this commit.

**Outcome.** The corrected notebook replaces the old one in this commit and runs top to bottom (about 20 minutes). `DECISION_LOG.md` is rewritten to match it.

**Bottom line.** Problems 1 and 2 were numerically right but several written answers were wrong, incomplete or unsupported. Problem 3 had real code bugs and look-ahead:
- a placebo that shifted rows instead of months;
- tercile sorts on same-month characteristics;
- `x10` (an annual average) lagged only one month;
- backfilled characteristics left in;
- unscaled inputs to Ridge/Lasso;
- a Ridge penalty stuck at the edge of its grid;
- a portfolio rule that multiplied by volatility instead of dividing.

Several write-up claims contradicted the notebook's own printed output. The headline conclusions (no out-of-sample predictability, macro adds nothing, forecast portfolio does not beat risk parity) **survive the corrections**, but most of the specific numbers, rankings and explanations behind them changed.

Severity: 🔴 invalidates a stated result or is look-ahead · 🟠 wrong number, claim or method a grader would mark down · 🟡 minor.

---

## Problem 1 — trees and ensembles

All printed numbers reproduce (baseline 64.25%, unpruned 85.00%, sweep, RF 88.75%, GBRT 89.00%, importances).

| | Finding | Evidence | Fix |
|---|---|---|---|
| 🟠 | **1.3 explanation was wrong.** It said the forest averages *pruned* trees and that averaging "adds noise back". | sklearn RF grows full-depth trees (about 50 leaves each, 99.75% in-sample). Lecture 8 p.44: "Each tree is grown deep and not pruned — the averaging does the regularizing". | Rewrote the explanation. The boosting part was wrong in direction too: accuracy *falls* as rounds are added (10 rounds 90.75%, 1,000 rounds 85.75%, a sweep now printed), so it over-fits here, the opposite of Lecture 8's California case. |
| 🟠 | **1.3 never tested the gap against noise.** | The tree's lead is 8 people (RF) and 7 (GBRT) out of 400. Paired fold t-statistics are 1.24 and 0.98, and the tree loses to RF in 2 of 5 folds. | Per-fold table and paired tests added. The answer now reads "ensembles do not improve on the tree", not "the tree is better". |
| 🟡 | 1.1 did not explain the exact 3/4-leaf tie, and still described the old salary-then-age split order. | The 3- and 4-leaf trees give identical out-of-fold predictions for all 400 people. | Check added in code; text corrected. |
| 🟡 | 1.2 answer ran to four sentences; the exam asks for one or two. | — | Now two sentences: the rule (age first, then salary) and a note that Gender is unused. |
| 🟡 | 1.4 said salary has "hundreds of unique values". | It has 117 (Age has 43), and no cell computed it. | Printed in code; text corrected; citation to Lecture 8 p.58 added. |
| 🟡 | 1.5 rested on "monthly returns are autocorrelated". | Median lag-1 autocorrelation in the panel is 0.02. The bias comes from persistent predictors and regimes, not raw autocorrelation. The Lecture 5 "0.036" figure measures mis-tuning cost, not score inflation. | Mechanism rewritten; Lecture 5 and AI-guide citations made precise. |

## Problem 2 — training on your own output

The simulation code matches the pipeline manual exactly, and every printed number reproduces.

| | Finding | Evidence | Fix |
|---|---|---|---|
| 🟠 | **2.2 answer reported none of the required numbers.** It said "see printed value above" and "in most runs". It claimed a "15–20x" shrinkage for the story (actually 11.7x) and missed the key fact that the story's implied σ̂² = (0.24/2.8)² = 0.0073 is the **median** desk (48.8th percentile). | cell 25 outputs | Answer rewritten with the printed numbers; story percentile computed in code. |
| 🟠 | **2.3 did not explain why the 1,000-desk mean (0.476) is "neither" 1 nor the median.** It also said "a shrinking majority collapse" (the majority *grows*). | log σ̂² at night 2,500 is ≈ N(−5.01, 3.17²). Half of E[σ̂²] comes from desks above e^{μ+s²} ≈ 152, which occur with probability 0.08%: fewer than one per 1,000 desks. | Theory cell added (exact drift log2 + ψ((n−1)/2) − log(n−1), lognormal check). Reconcile paragraph rewritten. |
| 🟠 | **2.4 explained "accumulation of noise" with the no-anchor collapse of 2.2**, not experiment (a). It said a synthetic sample "contains that one moment and nothing else", missing that it contains *noise*. | In (a) the real half alone gives σ̂² = 1.000 on every desk. Essentially all of the 0.944–1.059 band is synthetic sampling noise, and the real half puts a hard floor of 499/999 under σ̂². | Answer rewritten; (b) now reports the sample vol (1.195%) and kurtosis excluding COVID (5.3). |
| 🟡 | 2.1 "commit before you compute" contained hindsight ("Part 2.3 will show…"). | — | Replaced with a genuine prior, which 2.3 now reconciles. |
| 🟡 | 2.4(a) night-1 library was the real days duplicated (σ̂² = 0.999, not the "exactly 1" the rules require). | — | Night 1 now uses the 500 real days only. Night-2,500 numbers are identical. |
| 🟡 | Simulation checks had no Monte Carlo standard errors; question cells were truncated paraphrases of the exam. | — | SEs added; exam text copied verbatim. |

## Problem 3 — research project

### Code bugs and look-ahead

| | Finding | Evidence | Fix |
|---|---|---|---|
| 🔴 | **Placebo shifted rows, not months.** `np.roll` inside `groupby('country')` on a date × asset frame. With 28 assets in `country_7`, a "12-month" placebo gave ~96% of class-A rows their *own month's* macro; multi-asset countries got shifts of a few months. All placebo numbers and the conclusion drawn from them ("24/36-month placebos beat real macro") were invalid. | effective-shift diagnostics | Roll on the (country, month) grid over the months where macro exists, with a check line confirming every row is shifted exactly *s* months. Shifts 12–60 across four models. |
| 🔴 | **Tercile sorts used same-month `x1`, `x3`, `x4` against same-month returns** (look-ahead; the exam asks for next-month returns). This made `x1` and `x4` look strongly predictive. | Pooled spreads: `x1` 11.3 → 0.5 pp/yr, `x4` 9.3 → 1.8 pp/yr when lagged. | Lagged sorts, pooled and within class, plus within-class t-stats on the target. |
| 🔴 | **`x10` is an annual calendar-year figure stamped on all 12 months** (changes only in January; corr 0.998 with the same-year mean of `x86`). A 1-month lag gives Feb–Dec forecasts the average of months not yet observed (Lecture 5 p.58: "a merge keyed on a period-end rather than a publication date"). | change-month count and correlation check | Lagged 13 months. |
| 🟠 | **Backfilled leading values of `x2`/`x3`/`x5`** (warned about in the exam) were never treated. | `x3` for 5 assets (30–31 months), `x2` for 2, `x5` for 1. | Detected as leading constant runs and blanked. |
| 🟠 | **Ridge/Lasso saw unstandardized country macro** (`x14` s.d. 43, `x16` mean 80; Lecture 5 p.23; AI guide §3). The Lasso intercept was 3.4 and it kept 8 macro columns and 1 characteristic. | — | `StandardScaler` inside a pipeline, fit on each training window. |
| 🟠 | **The "validated" Ridge penalty sat at the top edge of its grid** (validation MSE still falling at α = 100; Lecture 5 p.27 says the optimum must be interior). It was chosen once, on the chars+macro sample, and reused for chars-only and every refit. | — | Penalties now chosen **at every refit** on the last 36 months of the training window (Lecture 5 pp.49–55), on wide grids. Boosting rounds chosen the same way (Lecture 8). |
| 🟠 | **Benchmarks were not fixed across runs.** The trailing-mean benchmark was computed on the feature-set-specific sample, so chars-only and chars+macro were scored against different benchmarks. Per-class runs used class means, so pooled vs per-class compared different denominators. | — | One benchmark computed once from all returns; class-level mean reported separately. |
| 🟠 | **Forecast portfolio used w ∝ r̂ = ŷ·σ̂**, i.e. multiplied by volatility (exam: "the forecast scaled by volatility"). With 90% positive forecasts this was a long-only book with 71% of gross exposure in class A (risk parity: 26%), and its "worst of four" result mostly measured that tilt. | — | w ∝ ŷ/σ̂ (= r̂/σ̂², the diagonal mean-variance weight; collapses to risk parity when forecasts are constant). The old rule is kept as a diagnostic row. |
| 🟠 | **TS momentum used the sum of 12 monthly returns**, not the trailing 12-month return. The sign flips on 3.5% of rows. | — | Compounded return (the definition `x2` matches on almost every row). |
| 🟡 | Latent off-by-one in the rolling window (calendar `DateOffset` on month-ends; harmless only because refits fall in January). | — | Window selected by month position. |
| 🟡 | Turnover ignored weight drift (equal-weight turnover was exactly 0 after month 1). | — | Drift-adjusted. |

### Wrong or unsupported claims

| | Finding | Evidence | Fix |
|---|---|---|---|
| 🟠 | "`x11` cannot be rebuilt (best proxy ≈ 0.61)". | 0.61 is the best *single* column. `x86 − x12` correlates 0.988 (a real rate). A chain-link rebuild cuts forward-fill error by 7–19x in a 36-month pseudo-out-of-sample test. Forward-fill would have frozen `country_12`'s value through the 2021–23 inflation surge. | Chain-link rebuild (last value + change in `x86 − x12`). |
| 🟠 | Characteristic hypotheses contradicted the printed table and the data. | `x2` *is* 12-month momentum (corr 0.993), not value. `x3` ≈ −(5-year return), long-term reversal (−0.94), not size. `x5` is 36-month vol (1.000). `x4` is not momentum. The text also said `x1` had "no monotonic pattern" when the printed table showed it had the largest one. | Identification cell added; hypotheses rewritten. |
| 🟠 | Never looked within class, although the exam asks. | `x1` and `x4` differ in cross-sectional scale by 100–1000x across classes, so an all-asset z-score mostly encodes class membership. | Within-class z-scores are now primary; all-asset kept as a robustness check (it is clearly worse). |
| 🟠 | "Class A shows the strongest within-block correlation". | Printed values: A 0.21 (weakest), C 0.73. | Corrected; full block matrix printed. |
| 🟠 | Class B "bright spot" (+0.026). | It is a benchmark artifact: the pooled mean forecasts about +0.4%/month for a class earning −0.2%. It is −0.015 vs zero and negative vs a class mean. "The only positive cell" was also false. | Three-benchmark by-class table. |
| 🟠 | "Best cell in the grid" (Lasso chars-only). | It kept at most 2 slopes, and an intercept-only forecast scored as well. | Intercept-only reference row added; search size reported (Lecture 5 p.57). |
| 🟠 | Write-up said macro "was lagged a further month". | The code lags one month. | Corrected (and `x10` is now lagged 13). |
| 🟠 | Trees run only under the expanding window while the text claimed "two schemes". | "Which predictors carry the signal and is it stable" was never answered. | Full model × scheme grid; Lasso coefficient paths. |
| 🟠 | Per-class sample-size arithmetic used out-of-sample counts. | "180 months", "9,000 rows" are OOS figures, not training data. | Actual training rows printed. |
| 🟠 | Risk parity "shifts weight toward C and D". | It moves weight to **D and B**; class C's weight falls. | Class-weight table printed. |
| 🟠 | Halved-turnover note claimed +0.05–0.1 Sharpe. | Actual: +0.011 (forecast) and +0.046 (momentum). The convention itself was fine: cost per dollar traded × Σ\|Δw\|. | Replaced by a cost-sensitivity table. |
| 🟠 | Undisclosed choices. | The portfolio model was chosen from out-of-sample results; "what we tried and rejected" described work not in the notebook; unused `x11_stale` flag was described as letting models "discount" staleness. | Disclosed or removed. |
| 🟠 | **Q2 asks about "global or country-level" macro; the two were never separated.** | Split on a common sample: country macro drives the OLS damage, and country macro alone is the only positive linear evidence (rolling Ridge/Lasso). | Split table, t-stats and a country-only placebo added. |
| 🟠 | "Macro hurts because of ~19 extra parameters" was asserted without a control. | Noise built exactly like the macro block (4 global series, 7 country levels, 7 differentials against `country_7`) hurts about as much as real macro: OLS −0.052 to −0.021 vs −0.042; Lasso −0.017 to +0.009 vs −0.013. So the damage is structural (11 persistent regressors shared by all assets), and one noise draw even beats the country-only "pocket". | Structured-noise control added (5 draws); explanation corrected. |
| 🟠 | **Portfolio model picked from the out-of-sample grid; forest settings never validated; Sharpe rankings stated without tests.** | — | Primary model and forest settings are now chosen on 2007–09 validation (training data only). Jobson–Korkie–Memmel tests show no pairwise difference is significant. |
| 🟡 | Incomplete duplicate scan; minor wording. | The extended file also holds exact copies of all four global series and a sign-flipped pair. Counts were off ("~19"/"~25"/"150" → 18/23/148). Per-asset table and predictor list were never printed; "simulated for the exercise" was unsupported. | Fixed. |

### What the corrected analysis finds

The conclusions hold, but on corrected numbers and for partly different reasons.

| | original notebook | corrected notebook |
|---|---|---|
| Best R²_OOS, characteristics only | −0.0018 (Lasso α = 0.05; really a near-constant forecast) | −0.0001 (Lasso tuned per refit, t = −0.03). Intercept-only is −0.0019; on the vol-scaled target every chars-only linear model equals a constant forecast. |
| Macro | "hurts; 24/36-month placebos beat real macro" (invalid placebo; global and country never separated) | The full block hurts OLS, Ridge, Lasso and the forest. Real macro ranks 3rd / 6th / 5th / 2nd of six against correctly shifted placebos (OLS / Ridge / Lasso / boosting). It is indistinguishable from persistent noise built with the same structure. **Two weak pockets survive the placebo test** but not a significance test: country macro alone in rolling Ridge/Lasso (+0.007 / +0.005, t ≤ 0.75, best 2 of 21 split cells) and class B. Both are hypotheses, not findings. |
| Class B | "the one bright spot" | Benchmark artifact: negative vs zero and vs its own class mean. The intercept-only forecast shows the same pattern. |
| Pooled vs per-class | pooled wins "in every class" (different benchmarks) | Pooled wins overall on a common benchmark (−0.0001 vs −0.0044; −0.013 vs −0.025), but not in every class. |
| Portfolios (net Sharpe, 10 bps) | RP 0.42, EW 0.37, TSMOM 0.15, forecast 0.11 (rule ŷσ̂, model picked from OOS results) | RP 0.41, forecast 0.39, EW 0.36, TSMOM 0.09 (compounded 12m return). The forecast is the random forest with macro, picked on 2007–09 validation and run with rule ŷ/σ̂. It has the best Sharpe before costs (0.48) but 5× risk parity's turnover; its alpha is +1.0%/yr (t = 0.78). No pairwise Sharpe difference is significant (p ≥ 0.39), and none of the 10 forecast portfolios has a significant alpha. |
| Characteristics | x5 = vol (right); x3 = size, x2 = value, x4 = momentum (wrong) | x2 = 12m momentum, x3 = −(5y return) long-term reversal, x5 = 36m vol, x1/x4 = class-specific fundamentals. Only x3 and x5 have within-class spreads beyond 2 s.e. |

Results are more fragile than the original text implied:
- Fixing only `x10`'s lag moved the Lasso-with-macro portfolio's Sharpe from about 0 to 0.22 while barely changing its R² (measured by running the corrected notebook with and without that fix).
- The validation-chosen forest is best on 2007–09 and before costs, but has one of the worst out-of-sample R²s.
- Every positive pooled (all-asset) R² cell is within +0.007 of zero, and those with a t-statistic have |t| < 1.

## Items checked and found correct (kept)

- Exam-mandated settings in Problem 1.
- The −1/n drift formula.
- The unbiasedness design.
- The DJ30 reduction and kurtosis convention.
- The Problem 2 night counting.
- The lag of `x1`/`x3`/`x4` and of all other macro.
- The trailing-vol target (`shift(1).rolling(12)`).
- The trailing 60-month global z-score.
- The six curated duplicates.
- The benchmark weight formulas (EW, 1/σ, sign/σ, unit gross).
- The R²_OOS formula against a benchmark fixed in advance, as opposed to `r2_score`.

Lecture citations verified as accurate:
- Lecture 5 p.56 "0.036 of out-of-sample R²";
- Lecture 5 p.57 "0.3–0.5%";
- Lecture 5 p.51 `step = 12`;
- Lecture 8 p.45 "flat after about fifty trees" (cited for the forest's tree count);
- Lecture 8 p.54 "one hundred rounds at the default rate is not enough" (contrasted in 1.3);
- Lecture 8 p.44 "each tree is grown deep and not pruned" and p.58 "MDI rewards a column for merely offering many places to split";
- AI guide §4(d) on `r2_score` (cited in the R² function).

## Caveats that remain

- `SEED = 0` (Problem 2) is still a placeholder for your student ID; all Problem 2 numbers are for seed 0.
- One 15-year out-of-sample window only.
- Forest settings tuned once on 2007–09, not at every refit. The 2007–09 validation block is a crisis period and favoured the macro-driven forest, which then did poorly out of sample.
- Macro lagged the minimum one month (13 for `x10`).
- The extended file is not used as features.
