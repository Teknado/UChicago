# Decision Log — `Final_project_claude.ipynb`

**Purpose of this document.** Every choice, assumption, and judgment call made while building the
reference solution, with the reasoning behind it, the alternatives considered, and — where
relevant — what was *not* done and why. This is written so you can independently challenge any
individual decision without having to reverse-engineer it from the code. Where I made a real
simplification or a debatable call, I say so explicitly rather than presenting it as the only
correct choice; a summary of the highest-priority items to scrutinize is at the very end.

Structure follows the exam: Problem 1, Problem 2, Problem 3 (in the same sub-question order as
the notebook), then a consolidated "what to challenge first" list.

---

## Problem 1: Trees and Ensembles

### 1.1 — baseline, unpruned tree, leaf-size sweep
- **CV setup exactly as specified**: `StratifiedKFold(5, shuffle=True, random_state=7034)`, and
  `random_state=7034` passed to every tree/forest/boosting model. This is not a choice — the exam
  mandates it verbatim so every model's number is comparable to every other. `shuffle=True` here
  is **not** the leakage trap discussed in 1.5 and Lecture 5, because this dataset's rows are
  cross-sectional (people), not time-ordered — shuffling is the *correct* thing to do for i.i.d.
  cross-sectional CV, and the exam is deliberately testing whether you know the difference (1.5
  asks exactly this).
- **Baseline** = `1 - ya.mean()` (accuracy of predicting the majority class, "nobody purchases").
  This is the textbook zero-information baseline; no alternative considered.
- **Leaf grid** `{2,3,4,6,8,12}` — given verbatim by the exam, not a choice.
- **Tie-break rule** ("if two sizes tie, take the smaller"): implemented as
  `max(leaf_acc, key=lambda k: (leaf_acc[k], -k))`. This works because Python's `max` breaks ties
  on the *first* comparison key that differs; since `-k` is strictly decreasing in `k`, among tied
  accuracies the entry with the larger `-k` (i.e., smaller `k`) wins. I chose this one-liner over an
  explicit loop for compactness; a more verbose but equally correct alternative would sort by
  `(-accuracy, size)` and take the first. Both are correct; I picked the shorter one.
- A plot of accuracy vs. `max_leaf_nodes` was added beyond what the question strictly requires
  (it only asks you to "report" the accuracies) — a judgment call that a plot makes the
  U-shape easier to see, not required for credit.

### 1.2 — plot and plain-English description
- `plot_tree(..., feature_names=Xa.columns, class_names=['No','Yes'], filled=True)` — the exam
  specifies `feature_names`; `class_names` and `filled=True` are my additions purely for
  readability, not required.
- I additionally printed the chosen tree's `feature_importances_` (MDI) as extra context before
  1.4 asks for it formally. This is scope creep beyond what 1.2 asks — harmless, but flag it if
  your own solution doesn't do this and you're comparing cell-by-cell.
- The plain-English description ("rich people buy it regardless of age; among people who aren't
  rich, only the older ones do") is a direct read of the fitted 3-leaf tree's actual splits
  (EstimatedSalary ≤ 90,500, then Age ≤ 42.5) — this is descriptive, not a modeling choice, but
  it's worth checking against your own fitted tree, since a different `random_state` or a
  different tie-broken leaf count would change the exact thresholds.

### 1.3 — RF(300) and GBRT(100) vs. the pruned tree
- **Deliberate non-tuning.** The exam specifies only `n_estimators` (300 for RF, 100 for GBRT) and
  `random_state=7034`; I left every other hyperparameter (`max_depth`, `learning_rate`,
  `min_samples_leaf`, etc.) at scikit-learn's defaults. This was a conscious choice, not an
  oversight: the question asks you to compare the *given* configuration against the tree and
  "report what you find whichever way it comes out" — tuning the ensembles until they win would
  defeat the point of the question, which is explicitly testing whether you'll report an honest
  negative result (ensembles losing to a single well-pruned tree) rather than quietly finding
  hyperparameters that flatter the ensemble. If you tuned RF/GBRT further in your own solution,
  your numbers will legitimately differ — that's a design choice worth discussing, not an error.
- The explanation offered (small n, low dimension, genuinely simple boundary → ensembles have
  little variance to kill by averaging) is an interpretation of the *result*, not a claim that
  could be independently "wrong" in the way a formula could be — but it's worth checking whether
  your own explanation for the same (or a different) outcome agrees.

### 1.4 — MDI vs. permutation importance
- **70/30 split**, `random_state=7034`, `stratify=ya` — given verbatim by the exam.
- **`n_repeats=30`** for `permutation_importance` — **my choice, not specified by the exam.**
  Lecture 8 itself warns "repeat the shuffle several times and average, since a single
  permutation is itself a random draw"; 30 is an arbitrary but reasonable number balancing
  stability against the (here, trivial) compute cost. A different `n_repeats` (10, 50, 100) would
  very likely shift the exact permutation-importance numbers slightly but is very unlikely to flip
  the qualitative finding (Age > EstimatedSalary out-of-sample, opposite of MDI's in-sample
  ranking), since the effect size in the printed table is large relative to what such a small
  change in repeat count would move.
- I passed `random_state=7034` to `permutation_importance` too, for full reproducibility — not
  explicitly requested, but consistent with the exam's own stated intent ("so the numbers are
  comparable").

### 1.5 — shuffle=True on time-ordered data (open-ended)
- Pure written answer, no code, since the exam says "(if any)" for the code cell. The claimed
  direction of bias (shuffled CV **overstates** accuracy, i.e., is optimistically biased) is
  taken directly from Lecture 5's own quantified example (random 5-fold cost 0.036 of
  out-of-sample R² relative to an expanding window on a real macro panel) — this is a documented
  course result, not something I derived from scratch, so if you disagree with the direction it's
  worth re-reading that lecture slide specifically before concluding the notebook is wrong.

---

## Problem 2: Training on Your Own Output

### Global setup
- **`SEED = 0`, left at the template default rather than your actual student ID.** This is the
  single most important caveat in the entire notebook: I do not know your student ID, so every
  specific number reported in Problem 2 (0.00443, the 41.6th percentile, the 56.1% share, etc.)
  is valid *only* for `SEED=0`. The exam explicitly says results will differ by seed and that
  this is the point — so if you compare your own (real-seed) numbers against this notebook's
  numbers and they don't match, **that is expected and correct**, not a bug. What should match
  qualitatively across any seed: the direction and rough order of magnitude of the VaR collapse,
  the fact that the median collapses while the mean is dominated by a few extreme desks, and the
  `-1/n` drift formula (which doesn't depend on the seed at all, only on `n`).
- **`np.random.default_rng(SEED)`**, the modern NumPy `Generator` API, rather than the legacy
  `np.random.seed()` / `np.random.randn()` interface — chosen for reproducibility guarantees and
  because it's the interface scikit-learn/scipy documentation now recommends; purely a style
  choice with no effect on the statistical content.
- **A fresh `Generator` instance per sub-experiment** (`rng`, `rng_m`, `rng_v`, `rng_n`, `rng_a`,
  `rng_b` — one each for the one-desk run, the 1000-desk run, the unbiasedness check, the drift-
  formula check, experiment 2.4(a), and experiment 2.4(b)), **all seeded with the same `SEED`**
  rather than one global generator threaded through the whole problem. This means, for instance,
  the "one desk" in 2.2 is a *separate* simulation from any single desk inside the 1000-desk
  cohort — it is not literally desk #1 of that cohort, even though both start from the same seed
  and the same statistical setup. **Alternative not chosen:** thread one `rng` through the whole
  problem, so the one-desk run and the 1000-desk run share a continuous random stream (making the
  "one desk" literally traceable as a specific member of later cohorts). I judged independent,
  per-experiment reproducibility more valuable than that traceability, but it's a legitimate
  design choice to make differently.

### 2.2 — one desk, then a thousand
- **Night-1 variance forced to exactly 1** via `library = library / library.std(ddof=1)` — this is
  not my choice, it's a hard requirement from the exam ("every desk starts night 1 from a library
  whose sample variance is exactly 1"). Rescaling (rather than resampling until sample variance is
  merely *close* to 1) is the only way to satisfy "exactly."
- **Vectorization**: at each of the 2,500 nights, I draw a `(1000, 500)` matrix of standard normals
  and scale each desk's row by that desk's current `sqrt(σ̂²)`, rather than looping over 1,000
  individual desks in Python. This is purely a performance choice (≈20 seconds vectorized vs. what
  would likely be minutes in a naive Python loop) with no effect on the statistics — the
  per-desk math is identical either way.
- The **percentile-rank comparison** ("my one desk sits at the 41.6th percentile of the 1,000-desk
  distribution") is descriptive of whatever `SEED=0` happens to produce; with your real seed, your
  one desk could land anywhere in the distribution, including well above the median. Don't expect
  this specific percentile to reproduce.

### 2.3 — unbiasedness and the log-drift formula
- **Unbiasedness check** used two starting levels (`σ̂²=1` and `σ̂²=4`) and 5,000 independent
  one-step draws at each, purely to make the check convincing at more than one point (showing the
  property isn't an artifact of starting exactly at 1). Using only one starting level would have
  been sufficient for a strict reading of the question but felt like a weaker demonstration.
- **`avg_log_change(n, n_desks=500, n_nights=300)`** — `n_desks=500` and `n_nights=300` are my
  choice within the exam's own suggested range ("a few hundred desks and a few hundred nights are
  enough"); I did not search for a minimal sufficient number, just picked round numbers inside the
  suggested band. The function resets every desk to `σ̂²=1` and tracks the *average* log-change at
  each night, then averages across nights — this works because the one-step log-drift is
  scale-invariant (it depends on `n`, not on the current level of `σ̂²`), so starting everything at
  1 rather than at a mix of levels doesn't bias the estimate.
- **The `-1/n` formula** is not just an empirical pattern-match to the simulation — it follows from
  the exact distributional fact that `(n-1)·S²/σ² ~ χ²_{n-1}` for i.i.d. Gaussian draws, so
  `E[log(S²/σ²)] = log(2) + ψ((n-1)/2) − log(n-1)`, which is ≈ `−1/(n−1) ≈ −1/n` for large `n`
  by the standard digamma asymptotic `ψ(x) ≈ log(x) − 1/(2x)`. This is real, derivable math, not
  just curve-fitting to the numbers the simulation happened to produce.
- The **histogram at night 2,500 reuses the same 1,000-desk cohort from 2.2** rather than
  re-simulating a fresh cohort — a deliberate choice so that the "share held by the top 10"
  statistic and the earlier mean/median/percentile numbers in 2.2 are all describing the *same*
  underlying draws, keeping the write-up's cross-references internally consistent.

### 2.4(a) — permanent real anchor
- **A genuine ambiguity in the exam's wording, resolved by assumption.** The prompt says: "add
  each night's 500 scenarios to them instead of replacing them, so the library holds 1,000
  returns, half real and half synthetic" — this describes the *steady state* (1,000 total, 500
  real + 500 synthetic) but doesn't fully specify what the synthetic half *is* before night 1 has
  happened. **My resolution:** I initialized the synthetic half as a copy of the real half (so
  night 1's fit sees library = [500 real, 500 real] = trivially `σ̂²=1`), then let the synthetic
  half evolve from there. An equally defensible alternative reading is that night 1's fit uses
  only the 500 real returns (library size 500, not 1,000, for the very first fit), reaching the
  full 1,000-observation steady state only from night 2 onward. I do not believe this changes the
  qualitative conclusion (that a permanent real anchor prevents collapse/explosion) since both
  readings converge to the same steady-state behavior quickly, but the specific night-2500 numbers
  could differ slightly between the two readings, and this is worth checking against how you
  resolved the same ambiguity in your own solution.

### 2.4(b) — real dj30 data
- `dj.groupby('date').MrkRet.first()` — exactly the reduction the exam specifies, since `MrkRet`
  is repeated identically across all 30 stocks' rows for a given date.
- `kurtosis(..., fisher=True)` — explicit, even though `fisher=True` is scipy's own default —
  added for clarity/defensiveness against a reader assuming Pearson's convention (kurtosis=3 for
  normal) rather than excess kurtosis (0 for normal), since the exam's language ("excess
  kurtosis") specifically means the Fisher convention.
- A fresh `rng_b` (seeded from the same `SEED`) draws the 500 night-1 synthetic scenarios — not
  reused from any earlier block, for the same "independent reproducibility per experiment" reason
  given above.

---

## Problem 3: Research Project — this is where most of the real judgment calls live

### Target variable
- **Vol-scaled excess return** $y_{i,t} = r_{i,t}/\hat\sigma_{i,t-1}$, exactly the target the exam
  *recommends* (not mandates) — I adopted the recommendation rather than deviating from it.
  **Alternatives explicitly not used:** raw `excess_return` (rejected — a pooled loss function
  over raw returns would be dominated by class-A's ~30%-vol assets, effectively ignoring class D's
  ~6%-vol assets, exactly the scale problem Section 1 documents); log returns (rejected as
  unnecessary complexity for monthly-frequency data of this kind); a cross-sectional rank target
  (rejected — complicates inverting a forecast back into a return-scale number for portfolio
  construction).
- **Trailing volatility**: 12-month rolling standard deviation of `excess_return`, `shift(1)`
  applied before the rolling window (so the vol estimate at date $t$ uses only returns through
  $t-1$), `min_periods=12` (strict — no partial windows). This costs the first 12 observations per
  asset (600 rows total, 50 assets × 12 months) as `NaN`/dropped. **Alternative not used:** a
  shorter `min_periods` (e.g. 6) would retain more early-2000s data at the cost of noisier early
  vol estimates; I chose strictness over sample size for this specific piece.

### Cross-sectional standardization: all-assets vs. within-class
- I standardized characteristics **across all 50 assets each month** (not separately within each
  class), even though the exam explicitly offers both as valid ("z-score or rank," "across all
  assets or within class"). **Rationale:** the primary specification in this notebook is the
  *pooled* model (Section 6 explicitly tests pooled vs. per-class and finds pooled wins), so
  standardizing on the same cross-section the pooled model actually sees is the internally
  consistent choice. **Not done:** a within-class standardization was not implemented or tested as
  a robustness check — this is a real gap if you want to know whether within-class standardization
  would have changed the pooled-vs-per-class conclusion.

### Country-macro feature set: curated only (14 columns), extended file NOT used as features
- **A significant, deliberate scope-narrowing decision.** The final model feature set uses only
  the 7 curated country columns (`x10`–`x16`, lagged) plus their 7 differentials against
  `country_7` — 14 country-macro columns total. **The 150-column extended file was cleaned for
  missing values (forward-filled, duplicates against the curated file dropped) but the cleaned
  result is never fed into any model.** This is explicit in the write-up's Interpretation section
  ("we considered using the full 150-column extended file... given it's explicitly flagged as
  exploratory material of uneven quality... adding 150 more largely-uncurated columns would only
  have multiplied the overfitting risk"), but it means: **if your own solution incorporated any of
  the extended file's variables as features, your model results are not directly comparable to
  this notebook's**, and the missing-value cleanup work you'll see for the extended file's 7 gappy
  columns in the notebook is, in a real sense, inert — performed for completeness/transparency but
  not actually load-bearing for any downstream result.

### The `country_7` base/differential convention — a real structural artifact, not flagged in-notebook
- **Verified fact, not previously stated in the notebook's own write-up:** because every
  differential feature is computed as `country_value - country_7_value`, and this is computed
  for *every* country including `country_7` itself, all 7 differential columns are **identically
  zero, for every month, for every asset physically mapped to `country_7`** — that's all 26
  class-A assets plus the one class-C and one class-D asset that also happen to sit in
  `country_7` (28 of 50 assets, 56% of the panel). For those 28 assets, the "differential" half of
  the country-macro feature set carries no information beyond a constant zero; only the level
  columns (`x10`–`x16`) vary for them. This is a direct, unavoidable consequence of using
  `country_7` as *both* the class-A convention and the differencing base — not a coding bug, but a
  design consequence worth weighing when you look at how much the differential features
  contribute to any model's fit.

### Global macro standardization: 60-month trailing window, `min_periods=24`
- Both numbers are my choice, not specified by the exam (which only says "a rolling z-score, for
  instance"). 60 months (5 years) was chosen as long enough to give slow-moving macro series a
  stable mean/std estimate, short enough to adapt across a 25-year sample that includes very
  different rate/inflation regimes. `min_periods=24` (2 years) trades off starting the
  standardization earlier against using a noisier early estimate. **Alternative not tested:** an
  *expanding* (not fixed-window rolling) standardization — using all data from the start up to
  $t-1$ rather than only the trailing 60 months — is equally defensible and would give
  different (likely smoother, since it uses more data) results, especially in the early part of
  the sample.

### Missing-value treatment, in more depth than the notebook states
- **`x11`**: forward-filled within country, plus an `x11_stale` flag — I checked whether it could
  instead be *rebuilt* from another complete series (best proxy correlation in the extended file
  ≈0.61, judged too weak to substitute) before choosing to forward-fill. **What I did not do:** a
  *country-specific* treatment — e.g., dropping `x11` (or zeroing its influence) specifically for
  `country_12` after October 2020, where the staleness coincides with the COVID regime shift the
  exam explicitly calls out as the dangerous case. The notebook forward-fills uniformly across all
  three affected countries and relies on the flag column to let a model *potentially* learn to
  discount stale periods — but none of the linear/tree models in Section 4 actually use
  `x11_stale` as a feature (it was carried through the pipeline for transparency, not fed into any
  model), so in practice nothing currently protects the models from `country_12`'s stale carry-
  forward during exactly the regime shift the exam warns about most.
- **Extended file's 7 gappy columns**: forward-filled, but — as noted above — never used as
  features at all, so this treatment is likewise inert for the reported results.
- **Duplicate detection** between curated and extended files used an exact-match tolerance of
  `atol=rtol=1e-6` with a `mask.sum() >= 100` overlap requirement before testing. The tight
  tolerance was deliberate — the exam says the extended file duplicates curated series *exactly*,
  so I wanted to catch only true duplicates, not merely highly-correlated-but-distinct series; a
  looser tolerance risked false positives, a much tighter one risked missing genuine duplicates
  with trivial floating-point differences.

### Out-of-sample split: December 2009 / January 2010 (120mo train, 180mo OOS)
- This date is a **free design choice** the exam explicitly hands to you ("fix the out-of-sample
  window... in advance, and say why you chose them") — it is not dictated by the exam, and a
  materially different cutoff (2008, 2012, even a 60/40 split) would be equally legitimate if
  justified. My stated rationale (10 years is enough history against a ~23-column feature set；15
  years of OOS spans the 2010s recovery, the 2015–16 selloff, the 2018 vol spike, COVID, and the
  2022 hiking cycle) is a real justification, but **no robustness check across alternative cutoff
  dates was performed** — I did not verify that shifting the split by a year or two either
  direction preserves the "everything is negative" conclusion, though given how uniformly negative
  every model/scheme/feature-set combination came out, I'd be surprised if a nearby cutoff choice
  overturned it.

### Refit cadence (`refit_every=12`) and rolling window length (`window=120`)
- **Annual refit** or both expanding and rolling schemes — not specified by the exam; chosen to
  match the cadence Lecture 5's own worked macro-forecasting example uses (`step=12`), and as a
  practical middle ground between refitting every month (expensive, likely unnecessary given how
  slowly macro/characteristics move) and refitting only once (too stale for a 15-year OOS window).
- **Rolling window = 120 months**, matching the initial training block length, chosen so the
  expanding and rolling schemes are directly comparable (rolling starts at the same size expanding
  does, then stays fixed while expanding grows). **Not tested:** a shorter rolling window (e.g. 60
  months) — plausibly would show *worse* rolling-window performance still, given the already-
  visible pattern that less training data per refit hurts (especially macro-inclusive) models, but
  this was not verified.

### Model hyperparameters — the single biggest simplification in Problem 3
- **Ridge `alpha=10.0`, Lasso `alpha=0.05`, RF (`n_estimators=200, max_depth=4,
  min_samples_leaf=20`), GBRT (`n_estimators=100, max_depth=2, learning_rate=0.05`) are all fixed
  values, chosen once from a quick, informal check on the training block alone, and then reused
  unchanged across every refit date in every scheme.** This is explicitly *not* the rigorous
  approach Lecture 5 itself demonstrates (time-ordered cross-validation to choose the penalty
  *at each refit*, e.g. via `LassoCV` inside an expanding-window loop) — I judged that adding a
  full nested-CV hyperparameter search inside an already-large notebook was outside a reasonable
  scope/time budget for this exercise, and chose fixed, reasonable-looking values instead. **This
  is a real limitation**: the reported Ridge/Lasso/RF/GBRT numbers reflect one point in
  hyperparameter space each, not the best each model family could achieve with proper per-refit
  tuning. If your own solution tunes hyperparameters via nested time-ordered CV, expect your
  numbers to differ from this notebook's, and don't assume this notebook's fixed choices are
  optimal — they are reasonable, not optimized.
- RF's tree count (200) is lower than Lecture 8's own California-housing example (300) — chosen
  for speed given how many total refits the full grid requires (5 models × 2 feature sets × 2
  schemes × 15 annual refits), leaning on Lecture 8's own finding that forest performance is "flat
  after about fifty trees" to justify that the difference between 200 and 300 is unlikely to
  matter much.
- GBRT's `max_depth=2` (shallower than sklearn's default of 3) and `learning_rate=0.05` (slower
  than the default 0.1) were chosen deliberately in the spirit of Lecture 8's "shallow trees, high
  bias, low variance" boosting recipe — but, per the point above, were not tuned against this
  specific data via a validation search; they are an out-of-the-box-but-thoughtful choice, not an
  optimized one.

### Placebo test: shifts of 12, 24, 36 months
- These three shift values are exactly the ones the exam's own text suggests as an example ("try
  more than one shift"); I did not search a wider or finer grid of shift values (e.g., every shift
  from 1 to 60 months) to map out a fuller "spurious correlation profile" — a reasonable extension
  not performed here.
- **Implementation**: `np.roll` (a *circular* shift) applied within each country for country-level
  macro, and across the date-ordered unique global-macro series for global macro. A circular shift
  preserves each column's full empirical distribution and its autocorrelation structure exactly —
  this was deliberate, because the point of the placebo is specifically to show that a
  *persistent-but-wrongly-dated* series can look almost as good as the real one; a plain random
  shuffle would destroy the persistence and make for a much weaker (too-easy-to-beat) placebo.
- **The actual result surprised the initial draft of the write-up** — an earlier pass (before a
  final correctness check) mis-stated which shift value did best; the corrected version in the
  saved notebook states, accurately, that the 24- and 36-month placebos both outperform the real,
  correctly-dated macro, which is if anything *stronger* evidence against macro carrying genuine
  signal than a milder "one placebo comes close" finding would have been.

### Portfolio construction: weight rule, cost assumption
- **Forecast portfolio weight rule**: $w_i \propto \hat r_i$ (the model's return-scale forecast,
  i.e., its vol-scaled $\hat y_i$ multiplied back by $\hat\sigma_{i,t-1}$), normalized to unit
  gross exposure. This is one of several rules the exam explicitly allows ("positions proportional
  to the forecast scaled by volatility, the sign of the forecast,... or your own rule"). **Worth
  scrutinizing:** because $\hat r_i = \hat y_i \cdot \hat\sigma_{i,t-1}$ already re-introduces the
  asset's volatility on the way out, this weight rule does *not* further divide by volatility at
  the position-sizing stage — a Sharpe-style alternative, $w_i \propto \hat r_i/\hat\sigma_i$
  (tilting toward the best forecast *per unit of risk*, rather than the largest raw forecast
  return), was **not implemented or tested**, and might plausibly have produced a different (though
  I'd guess still unimpressive, given how weak the underlying forecasts are) portfolio result.
- **Transaction cost: 10 bps one-way per unit of monthly gross turnover** — an assumption, not
  given or derived from the data. No sensitivity check against other cost levels (5bps, 20bps) was
  performed; the qualitative ranking (forecast portfolio worst, risk parity best) is driven mostly
  by *gross* Sharpe differences large enough that a different cost assumption is very unlikely to
  reorder the ranking, but this was not explicitly verified.
- **Benchmark weight formulas** (equal weight, risk parity $\propto 1/\hat\sigma_i$, momentum
  $\propto \mathrm{sign}(\text{trailing-12mo return})/\hat\sigma_i$) are implemented exactly as the
  exam's own formulas state, normalized to unit gross exposure by dividing by the sum of absolute
  weights — the only sensible way to hit "unit gross exposure," so there's little discretion here.

### Per-class vs. pooled: tested for one specification only
- Section 6 compares pooled vs. per-class using **only** Lasso on chars+macro (the
  best-behaved specification from Section 4) — not repeated across all 5 models × 2 feature sets.
  This was a scope-management choice to keep the notebook's total runtime and length reasonable; a
  full cross-product comparison was not performed, so "pooling wins" is demonstrated for one
  representative case, not exhaustively for every model/feature combination.

### R²_OOS reporting convention
- Used **Lecture 4's exact formula**, $R^2_{OOS} = 1 - \mathrm{SSE}_{model}/\mathrm{SSE}_{benchmark}$,
  with the benchmark being the trailing mean of `excess_return` computed from data strictly before
  each forecast date and recomputed every month — not scikit-learn's `r2_score` (which benchmarks
  against the *test-set* mean, a quantity that could not have been known in advance, and which the
  AI Coding Guide explicitly flags as a common but wrong substitution). Both benchmarks the exam
  asks for (trailing mean, and zero) are reported side by side throughout.

---

## Summary: highest-priority items to challenge first

If you only have time to scrutinize a handful of decisions, these are the ones most likely to
matter or most likely to differ from your own solution:

1. **`SEED=0`** in Problem 2 — every specific number there is seed-dependent by design; compare
   only the qualitative pattern against your own (real-seed) results, not the exact figures.
2. **The extended 150-column macro file was cleaned but never used as model features** in
   Problem 3 — a real scope-narrowing choice, not a bug. If your solution used it, your numbers
   will differ and neither approach is "more correct" per the exam's own text.
3. **The `country_7` differential features are identically zero for 28 of 50 assets** (all of
   class A, plus one class-C and one class-D asset) — a structural consequence of using
   `country_7` as both the class-A convention and the differencing base, not previously called out
   in the notebook's own write-up.
4. **No rigorous, per-refit hyperparameter tuning** for Ridge/Lasso/RF/GBRT in Problem 3 — fixed
   values chosen once, informally, and reused throughout. This is the single biggest methodological
   simplification in the whole project.
5. **The forecast-portfolio weight rule doesn't re-divide by volatility** at the position-sizing
   stage (it uses the return-scale forecast directly); a Sharpe-style alternative was not tested.
6. **2.4(a)'s pre-night-1 synthetic-library initialization is a resolved ambiguity**, not a fact
   given by the exam — check how you resolved the same ambiguity.
7. **The OOS split date (2010), rolling window length (120mo), and refit cadence (12mo)** in
   Problem 3 are all defensible-but-arbitrary design choices, not derived from the data or dictated
   by the exam, and were not stress-tested against nearby alternatives.
