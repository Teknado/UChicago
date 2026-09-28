# Plan: Problem 2, "Training on Your Own Output" (parts 2.1 to 2.4), BUSN 41210 Final, Autumn 2026

Status: **plan only.** I did not run the pipeline, simulate any desk, or compute any number the exam asks for. What I did run:
- API and default checks in the installed environment (numpy 2.4.6, pandas 3.0.6, scipy 1.17.1, sklearn 1.9.1, statsmodels 0.15.0).
- Timing of a single pipeline "night" on throwaway random matrices, to size the budget.
- A 50-night toy check, on a throwaway seed, that the literal one-desk loop and the vectorised function agree bit for bit.
- An analytic sample-size calculation (Appendix A.6).

The dj30 facts come from `notes/data_profile.md` §B.

Citation keys:
- "L<n> p.<page>" means the PDF page, as used in `notes/Lecture_<n>.md`. The key pages were checked against the PDF text: L2 p.56, 57, 63, 64, 78, 80; L3 p.49; L4 p.26; L5 p.41.
- "Exam cell N" is the 0-based cell index in `Final_Autumn_2026-1.ipynb`:
  - cell 20: P2 story and rules
  - cell 21: setup code (`SEED`, `N_SCEN=500`, `NIGHTS=2500`, `Z01=norm.ppf(0.99)`; imports `norm`, `kurtosis`, `time`)
  - cells 22, 24, 27, 30: questions 2.1–2.4
  - cells 25, 28, 31: code cells
  - cells 23, 26, 29, 32: answer cells

---

## 0. READ THIS FIRST: the 2.1 firewall

2.1 is "commit before you compute" (exam cell 22). **The student should write the 2.1 answer (cell 23) before reading past §1 of this plan.** From §2 onward the plan contains the design and theory, and those would contaminate the commitment. §1 contains no prediction.

---

## 1. Part 2.1: Commit before you compute (2 points)

**(i) Goal.** Before any code, answer in 2–3 sentences:
- The step-(1) estimator is unbiased, so each night's estimate equals the previous night's on average. Does that mean the reported VaR should stay near 2.326 for ten years?
- What do you expect on night 2,500, and why?

No marks depend on being right. 2.3 must reconcile the answer with the results.

**(ii) Steps.**
1. The student reads exam cells 20–22 only.
2. The student writes the answer in cell 23 **in their own words**. It is their prior, and the AI does not draft it.
3. Encouraged: include a number or range for the night-2,500 VaR (or σ̂²). This follows the lecture device "Write down a number".
4. Save the notebook before any Problem 2 code cell is run.
5. Never edit cell 23 afterwards. The 2.3 answer quotes it and says what was right and what was wrong.

**(iii) Grounding.**
- L2 p.78: "Before the next slide: how wide will that histogram be? Write down a number."
- L4 p.26: "Say a number before I show you."

**(iv) Assumptions.** Cell 23 is final once written. Reconciliation happens only in cell 29.

**(v) Guards.**
- No Problem 2 code runs before cell 23 is filled.
- The student does not read §2 onward first.
- Optional evidence: a notebook save or `git commit` right after writing. Git actions happen only if the student asks.

**(vi) Outputs.** 2–3 sentences in cell 23 and no code.

**(vii) Answer checklist.**
- [ ] Says whether unbiasedness implies the VaR stays near 2.326 (yes/no).
- [ ] States what is expected on night 2,500, ideally as a number or range.
- [ ] Gives the reason.
- [ ] Is 2–3 sentences long and was written before any code.

**(viii) Open question.** Q2.1: will the student write it themselves now? **Default: yes.** We then reconcile in 2.3 by quoting cell 23 verbatim.

---

## 2. Scope and grounding map

### 2.1 Methods used, and where the lectures teach them

| Method / concept used | Where taught | Used in |
|---|---|---|
| Unbiased sample variance, n−1 denominator; `np.var(..., ddof=1)` idiom | L2 p.56 (unbiased s² = SSE/(n−p)), p.57 (s_y² = SST/(n−1)), p.17 and p.31 (ddof=1 code) | pipeline step (1), everywhere |
| Degrees of freedom = "the number of times you get to observe useful information about the variance" | L2 p.57 | 2.3 explanation, 2.4 "what synthetic data lacks" |
| Monte-Carlo sampling distribution: many alternative histories from a fitted model, refit, study the spread | L2 p.59–62, p.65–70, p.78–79; L4 p.26–27 (a known truth: "we know this because we made them up"; 4,000 repetitions) | 2.2 (1,000 desks), 2.3 (martingale, drift), 2.4(a), 2.4(b) kurtosis reference band |
| Mean of iid draws: E(X̄)=µ, var(X̄)=σ²/n, hence Monte-Carlo SE = sd/√M | L2 p.63 | every MC standard error, including SE of a fraction √(p(1−p)/M) |
| CLT: sample averages ≈ normal; convergence is slow for skewed data | L2 p.64; p.65–70 (Exp(1), n=2 still skewed) | bell shape of the log σ̂² histogram (a sum of nightly increments); why the 1,000-desk mean of σ̂² is unreliable |
| ≈2-SE rule, CI with t critical value `stats.t.ppf`, t-stat against a non-zero null computed by hand | L2 p.84–85, p.87, p.91, p.97 | martingale check, drift vs −1/n and −1/(n−1) |
| Unbiased ≠ precise; bias–variance | L2 p.71–72; L5 p.6–7 | 2.3 "why unbiased does not protect" |
| Log scale for multiplicative / percentage change; natural log | L3 p.48–49 (and p.50 log-log slope = elasticity, optional) | 2.2 path plot, 2.3 drift and histogram |
| Histogram with a normal curve overlaid | L2 p.79 (left figure), p.66–70 | 2.3 histogram, 2.4(b) real vs normal |
| Nonparametric bootstrap SE (resample the actual days with replacement; iid) | L2 p.80 | 2.4(b) SE of empirical vs normal VaR (optional, default on) |
| "The gap between them is the size of the assumption"; normal-error simulation reproduces only its own assumptions | L2 p.79–80 | 2.4(b) interpretation, 2.4 paragraph |
| Normal tail masses; t is a fat-tailed normal; inference "relies on this model being true" | L2 p.38, p.81; L3 p.2–3 | 2.4(b) interpretation |
| Sums-of-squares decomposition (ANOVA) and group dummies | L2 p.26–28; L3 p.45, p.51–52 | 2.4(a) explanation of the pooled real + synthetic variance |
| Plot the raw return series; crisis cluster (Apple daily returns 2016–2021, COVID) | L1 p.38 | 2.4(b) time-series figure |
| Validation requires knowing where your data came from; "compared to what?"; confidently wrong with no tell; too-good-to-be-true simulated track record | L1 p.80, p.81, p.82, p.39 | 2.3 and 2.4 paragraphs |
| "Never use the same data twice" (analogy only) | L5 p.41 | 2.4 implication sentence |
| Single Gaussian vs mixture of Gaussians (**extension; L7 never discusses kurtosis**) | L7 p.10–12 | optional one clause in 2.4 |

### 2.2 Required by the exam but NOT taught in any lecture

For each of these the answer cites the exam definition and says it is not from a lecture.

| Item | Status | Grounding used instead |
|---|---|---|
| VaR = 2.326 σ̂ (one-day 1% VaR under N(0, σ̂²)) | not in any lecture (L2/L3 boundaries) | defined by the exam, cell 20; `Z01 = norm.ppf(0.99)` in cell 21 |
| Empirical 1% VaR = first percentile of real returns, sign flipped | not taught | defined by the exam, cell 30 |
| Excess kurtosis | not taught | asked by the exam, cell 30; `scipy.stats.kurtosis` imported by the exam, cell 21 (defaults `fisher=True, bias=True`, verified) |
| Martingale property E[σ̂²_{t+1} \| σ̂²_t] = σ̂²_t | not named in the lectures; it follows from unbiasedness (L2 p.56–57) | stated by the exam, cell 27, and verified by simulation |
| χ² law of s², Var(s²) = 2σ⁴/(n−1), delta method / Jensen, digamma/trigamma, lognormal, law of iterated expectations, martingale convergence | **explicitly NOT covered** (L2 notes §21 boundaries) | **Supplementary derivation only**, labelled as beyond the lectures (Appendix A). The formula in 2.3 is established **empirically**, as the exam asks ("measure … and propose the formula"). |

### 2.3 Dropped or flagged

See §9 for details.
- QQ plots, formal normality tests, VaR back-testing tests, block bootstrap, and annualised volatility are not taught and are dropped.
- The "model collapse" literature is **OUTSIDE SCOPE: proposed only if the user approves.**

---

## 3. Global design (applies to every part)

### 3.1 Data, paths and constants

- Keep exam cell 21 as given, **except `SEED`**. Add constants at the top of cell 25:
  ```python
  N_DESKS = 1000
  TRUE_S2 = 1.0                              # sigma = 1: correct VaR = Z01 every night (exam cell 20 "Rules")
  STORY_VAR_START, STORY_VAR_END = 2.8, 0.24 # % of NAV, from the story in exam cell 20
  ```
- Data path: `/classes/41210_MiF_fall2026/Data/` does not exist in this environment (verified). In cell 31, use:
  ```python
  _path = _DATA_DIR if os.path.isdir(_DATA_DIR) else '/home/user/UChicago/'
  ```
  This mirrors the commented local line in exam cell 4. Only `dj30.csv` is used in Problem 2.
- Units: 2.2–2.4(a) are in units of the true σ (σ = 1). 2.4(b) is in real decimal returns, reported in % of NAV.

### 3.2 Night indexing (assumption A1, stated in the 2.2 answer)

- Arrays are indexed `k = 0..2499`, and index k is **night k+1**.
- Night t runs the manual in order:
  1. fit σ̂²_t = var(ddof=1) of the library L_t;
  2. draw S_t: 500 iid N(0, σ̂²_t);
  3. set L_{t+1} := S_t;
  4. report VaR_t = Z01·√σ̂²_t.
- L_1 is the real library, whose sample variance is exactly 1. So **σ̂²_1 = 1 deterministically**, and VaR_1 = 2.326 is correct.
- For t ≥ 2, σ̂²_t is the fit to the scenarios drawn on night t−1.
- "2,500 refits" means the fits on nights 1..2,500, and the first of them is on real data. **Night 2,500's σ̂² is separated from night 1 by 2,499 random draw-and-refit steps.** The scenarios drawn on night 2,500 are never refit.
- Alternative reading (2,500 random steps): it changes one step out of ~2,500, which is well inside Monte-Carlo noise. Mention it in one clause.

### 3.3 RNG design: one master seed, named independent streams

```python
STREAM_NAMES = ['one_desk', 'desks_1000',                  # 2.2
                'mart_s0_1', 'mart_s0_one',                # 2.3 martingale check
                'drift_50', 'drift_500', 'drift_5000',     # 2.3 drift (required)
                'drift_50_precise', 'drift_small',         # 2.3 optional runs
                'mixed_real', 'mixed_synth',               # 2.4(a)
                'dj_scen', 'kurt_null', 'dj_boot']         # 2.4(b)
_SS = dict(zip(STREAM_NAMES, np.random.SeedSequence(SEED).spawn(len(STREAM_NAMES))))
def rng_for(name):
    """A fresh Generator on a named stream: the same numbers every time a cell is re-run."""
    return np.random.default_rng(_SS[name])
```

- `np.random.default_rng(SEED)` is itself built on `SeedSequence(SEED)`, so everything is driven by the one exam-mandated `SEED`. Changing SEED changes every stream ("your numbers will differ from your classmates'", exam cell 20).
- Child i depends only on (SEED, i). This was verified: `SeedSequence(1234).spawn(3)[1]` and `.spawn(8)[1]` give identical draws. So:
  - re-running any single cell reproduces its numbers;
  - cells can be run in any order;
  - **appending** a new stream name never changes existing results. **Never reorder or insert.**
- The children are statistically independent streams, so:
  - the one desk is independent of the 1,000 desks;
  - each 2.3 experiment is independent of 2.2;
  - 2.4(a) is independent of 2.2.
- The only cross-part link is a *value*: 2.3's second martingale start `s0` is the one desk's night-2,500 σ̂² from 2.2.
- 2.4(a) "demeaned" robustness run: it re-creates `rng_for('mixed_synth')`, so both variants use **the same standard-normal draws** (common random numbers). Their difference then isolates the demeaning choice.
- Never use `np.random.seed`, `np.random.normal` (legacy global state) or Python's `random`.

| Stream | Part | Draws |
|---|---|---|
| one_desk | 2.2 | 2,499 × 500 |
| desks_1000 | 2.2 | 2,499 × (1000 × 500) |
| mart_s0_1, mart_s0_one | 2.3 | 200,000 × 500 each, in chunks of 10,000 |
| drift_50 / drift_500 / drift_5000 | 2.3 | 500 steps × (500 × n) |
| drift_50_precise (optional, default ON) | 2.3 | 2,000 steps × (2,000 × 50) |
| drift_small (optional, default OFF) | 2.3 | n ∈ {10, 20}: 1,000 × (1,000 × n) |
| mixed_real | 2.4(a) | (1000, 500) real days |
| mixed_synth | 2.4(a) | 2,499 × (1000 × 500) |
| dj_scen | 2.4(b) | 500 |
| kurt_null | 2.4(b) | (10,000, 500) |
| dj_boot | 2.4(b) | 10,000 × 1,511 integer indices, in chunks of 500 |

### 3.4 One implementation of the pipeline (defined in cell 25, reused in 2.3)

```python
def run_desks(n_desks, n_scen, nights, rng, s2_night1=1.0):
    """Replacement pipeline of exam cell 20, vectorised over desks.
    Returns log_s2 with shape (nights, n_desks); row k = night k+1; row 0 = the night-1 fit."""
    log_s2 = np.empty((nights, n_desks))
    s2 = np.full(n_desks, float(s2_night1))
    log_s2[0] = np.log(s2)
    for k in range(1, nights):                                   # nights-1 random steps
        scen = rng.standard_normal((n_desks, n_scen)) * np.sqrt(s2)[:, None]  # (2) N(0, s2): scale = SD
        s2 = scen.var(axis=1, ddof=1)                            # (3) replace, then (1) next night's fit
        log_s2[k] = np.log(s2)
    return log_s2
```

- Only the current (n_desks, n_scen) matrix is held in memory. Nothing is stored except the log-σ̂² path.
- The time-major (nights, desks) layout makes each row write contiguous.
- The same function serves 2.2 (1,000 desks) and every 2.3 drift run, so 2.2 and 2.3 measure the same pipeline.
- Unit test (in cell 25): the literal one-desk loop (§4) must equal `run_desks(1, N_SCEN, NIGHTS, rng_for('one_desk'))[:, 0]` with `np.array_equal`. This is verified to hold in numpy 2.4.6, because `rng.normal(0, s, 500)` and `s * rng.standard_normal((1, 500))` are bit-identical. If a future numpy breaks it, fall back to `np.allclose(rtol=1e-12)`.

### 3.5 Compute and memory budget

Per-night times were measured on this machine with throwaway matrices.

| Run | Draws | Time | Peak array | Stored |
|---|---|---|---|---|
| 2.2 one desk | 1.25e6 | < 1 s | 4 KB | (2500,) |
| 2.2 1,000 desks | 1.25e9 | ≈ 35 s (0.014 s/night) | 4 MB | (2500, 1000) float64 = 20 MB |
| 2.3 martingale, 2 starts | 2e8 | ≈ 4 s | 40 MB chunk | 2 × (200,000,) |
| 2.3 drift n=50 / 500 / 5000 | 1.25e7 / 1.25e8 / 1.25e9 | < 1 s / ≈ 4 s / ≈ 53 s (0.106 s/night) | 20 MB (n=5000) | (501, 500) each |
| 2.3 n=50 precision (optional ON) | 2e8 | ≈ 3–5 s | 0.8 MB | (2001, 2000) = 32 MB |
| 2.4(a) anchored, and again demeaned | 2 × 1.25e9 | ≈ 2 × 43 s (0.017 s/night incl. concat) | 8 MB | 2 × 20 MB |
| 2.4(b) | ~2e7 | < 2 s | 6 MB chunk | small |

Total ≈ 3–4 minutes. Print `time.perf_counter()` durations, since the exam imports `time`. The exam's "takes seconds" means tens of seconds here.

### 3.6 Where each AI-guide guard lives

| Guard | Where it lives in P2 |
|---|---|
| (a) `train_test_split` / shuffled folds on time-ordered data | **Structurally absent.** Problem 2 fits no predictive model and has no train/test split. The only real time series, dj30 `MrkRet`, is used for descriptive statistics over its full sample, which the exam asks for (cell 30). The one place time order matters is the optional iid bootstrap of days. It ignores volatility clustering, and the answer says so (L2 p.80 note: Route 3 still assumes independence; L1 p.38 shows the cluster). |
| (b) scaler / imputer / PCA before the split | **None used.** The rescaling of each desk's simulated real days to sample variance exactly 1 is part of the exam's data-generating definition (cell 20: "a library whose sample variance is exactly 1"). It is applied within a desk to that desk's own data, and no evaluation split exists that it could leak across. |
| (c) `LogisticRegression()` is L2-penalised | Not used in P2. |
| (d) `r2_score` / `.score` against the test-set mean | Not used. Every simulated σ̂² and VaR is compared with the truth **fixed before any simulation** by the exam: σ² = 1 and VaR = 2.326 (cells 20–21). In 2.4(b), the pipeline VaR is compared with the empirical 1% quantile of the same real data, which is the comparison the exam defines. |
| pandas 3 / sklearn 1.9 pitfalls | Tables use `pd.DataFrame(list_of_dicts)` or `pd.concat`, never `df.append`. Use `.items()`, never `.iteritems()`. No `fillna(method=)`. **No `sns.set()`**: use matplotlib only (`fig, ax = plt.subplots(...)`). |
| Workflow (AI guide §6) | Describe dj30 first (shape, dtypes, head, unique dates) before computing. Work one step per cell block. Run the known-answer checks in §7 before writing. Verify each headline number with a second, independent route (§7). |

### 3.7 Coding pitfalls specific to P2

1. `rng.normal(loc, scale)` takes the **SD**. Pass `np.sqrt(s2)`, never `s2`.
2. `np.var` / `np.std` / `ndarray.var` default to **ddof=0**. Always pass `ddof=1`, which is the manual's estimator (L2 p.56–57). pandas `Series.std/var` default to ddof=1. Pass it explicitly anyway.
3. `np.percentile(x, 1)` takes **percent** (q=1 means 1%); `np.quantile(x, 0.01)` is the equivalent. The default `method='linear'` is stated in the answer.
4. `scipy.stats.kurtosis` defaults to `fisher=True` (excess) and `bias=True`, and it reduces along `axis=0`. Pass `axis=1` for the (10000, 500) reference matrix. pandas `.kurt()` is the bias-corrected version, and the two differ (24.07 vs 24.16 on dj30, data profile §B).
5. Natural log only (`np.log`), labelled "log (natural)" (L3 p.49).
6. dj30: `MrkRet` repeats on every stock row. Reduce with `groupby('date').MrkRet.first()`. Never average `RET`, which is rounded to 0.01 (data profile §B). The `date` column is an int YYYYMMDD. Integer sort order equals chronological order.
7. "VaR below one-tenth of the truth" means `Z01*np.sqrt(s2) < 0.1*Z01`, i.e. `s2 < 0.01` (strict).
8. Guard against a silent collapse to 0: `assert np.isfinite(log_s2).all()` after each run. At the planned sizes the smallest σ̂² stays far above float64 underflow.

---

## 4. Part 2.2: One desk, then a thousand (6 points)

**(i) Goal.**
- Simulate one desk for 2,500 nights exactly as the manual says. Plot log σ̂² against the night. Report σ̂² and the VaR on night 2,500.
- Run 1,000 desks. On night 2,500 report the following for σ̂²: mean, median, 5th and 95th percentiles, and maximum. Also report the fraction of desks with VaR < 0.1 × truth.
- Where does the one desk sit in that distribution? Is the story's fall from 2.8% to 0.24% typical?

**(ii) Method steps** (cell 25, split into a few small cells).
1. **Setup:** streams (§3.3), constants (§3.1), `run_desks` (§3.4). Print a warning if `SEED == 0`.
2. **One desk, literal loop** (manual steps (1)–(4)):
   ```python
   rng = rng_for('one_desk')
   s2_one = np.empty(NIGHTS); s2_one[0] = 1.0      # night 1: real library, sample variance exactly 1
   for k in range(1, NIGHTS):
       library = rng.normal(loc=0.0, scale=np.sqrt(s2_one[k-1]), size=N_SCEN)  # (2) draw, (3) replace
       s2_one[k] = library.var(ddof=1)                                         # (1) next night's fit
   var_one = Z01 * np.sqrt(s2_one)                                             # (4) report, sigma units
   assert np.array_equal(np.log(s2_one), run_desks(1, N_SCEN, NIGHTS, rng_for('one_desk'))[:, 0])
   ```
3. **Figure F2.2:**
   - x = `np.arange(1, NIGHTS+1)`, y = `np.log(s2_one)`.
   - Horizontal lines: y = 0 (truth, log 1) and y = log 0.01 (the "VaR one-tenth of truth" threshold).
   - Optional right axis: `ax.secondary_yaxis('right', functions=(lambda y: Z01*np.exp(y/2), lambda v: 2*np.log(v/Z01)))`, labelled "reported VaR (σ units)".
   - Title, axis labels, legend. The exam asks for exactly this plot.
4. **Report the one desk:**
   - `s2_one[-1]`;
   - `var_one[-1]` in σ units;
   - on the story's scale, % of NAV = `STORY_VAR_START * np.sqrt(s2_one[-1])`, because night-1 VaR 2.326 corresponds to 2.8%.
5. **1,000 desks:** `logS2 = run_desks(N_DESKS, N_SCEN, NIGHTS, rng_for('desks_1000'))` with shape (2500, 1000). Time it. Then `S2_T = np.exp(logS2[-1])` with shape (1000,).
6. **Table T2.2** (night 2,500, 1,000 desks):
   - `S2_T.mean()`, `np.median(S2_T)`, `np.percentile(S2_T, [5, 95])`, `S2_T.max()`;
   - the same quantities as VaR in σ units;
   - `frac_low = (S2_T < 0.01).mean()` with MC SE `sqrt(frac_low*(1-frac_low)/N_DESKS)` (L2 p.63).
7. **Position of the one desk:**
   - `pct_one = 100*(S2_T < s2_one[-1]).mean()` (percentile rank among the 1,000 independent desks);
   - optionally a z-score on the log scale, `(log s2_one[-1] - logS2[-1].mean())/logS2[-1].std(ddof=1)`.
8. **Is the story typical?**
   - The story's VaR ratio is 0.24/2.8. The σ̂² equivalent is `story_s2 = (STORY_VAR_END/STORY_VAR_START)**2`, because VaR ∝ σ̂.
   - Report `pct_story = 100*(S2_T < story_s2).mean()`.
   - Rounding band: 2.75–2.85% and 0.235–0.245% give `story_s2` in [(0.235/2.85)², (0.245/2.75)²]. Report the percentile range over that band as well.
   - Also report the median desk's VaR on the story scale, `2.8*np.sqrt(np.median(S2_T))` (% of NAV), for a like-for-like comparison with 0.24%.

**(iii) Grounding per step.**
- Step (1) estimator: L2 p.56–57, code idiom L2 p.17/p.31.
- "Many desks" is the Monte-Carlo sampling-distribution design: L2 p.59–62, p.65–70, and p.78–79 ("treat that fitted line as if it were the truth, generate 10,000 alternative histories"). A known truth, as in L4 p.26–27.
- Log scale: L3 p.48–49.
- Report the distribution, not one path: L2 p.59 ("if the estimates do vary a lot, then it matters which sample you happen to observe").
- SE of a fraction: L2 p.63.
- VaR definition: exam cell 20 (not taught).

**(iv) Assumptions.**
- **A1** night indexing (§3.2).
- **A2** σ̂²_1 = 1 exactly. The real library is not needed in 2.2, because only its sample variance enters the night-1 fit. Real days are generated only in 2.4(a), where they are needed.
- **A3** the one desk is **independent** of the 1,000. It is not desk #0, so its percentile rank is a clean comparison.
- **A4** the story's percentages are converted with night-1 VaR 2.8% ↔ σ̂² = 1. The dj30 value is 2.78% (2.4(b)), which is consistent.
- **A5** percentiles use the `np.percentile` linear default.

**(v) Guards.**
- `ddof=1` everywhere.
- scale = SD.
- `assert logS2.shape == (NIGHTS, N_DESKS)`.
- `assert np.all(logS2[0] == 0.0)` (night 1 = log 1).
- `assert np.isfinite(logS2).all()`.
- The unit test from step 2.
- The truth (1, 2.326) is fixed by the exam, and nothing is benchmarked on simulated output.

**(vi) Outputs.**
- F2.2.
- A printed one-desk line: σ̂²_2500, VaR_2500 in σ units and in % of NAV.
- T2.2: mean / median / p5 / p95 / max of σ̂², the same as VaR, `frac_low` ± SE, `pct_one`, `story_s2`, `pct_story` with its rounding range, and median VaR in % of NAV.
- Runtime.

**(vii) Answer checklist (cell 26).**
- [ ] Plot of log σ̂² vs night (F2.2), described in one sentence.
- [ ] One desk: σ̂² and VaR on night 2,500, in σ units and on the story's % scale.
- [ ] 1,000 desks: mean, median, 5th and 95th percentiles, and max of σ̂² on night 2,500.
- [ ] Fraction with VaR < one-tenth of truth, with its MC SE.
- [ ] Where the one desk sits (percentile rank).
- [ ] Whether 2.8% → 0.24% is typical: its percentile, compared with the median desk.
- [ ] One clause on A1 (2,499 random steps).

**(viii) Open questions.**
- Q2.2a: A1 indexing. **Default: 2,499 random steps.**
- Q2.2b: one desk independent of the 1,000. **Default: yes.**
- Q2.2c: criterion for "typical". **Default: report the percentile. Call it typical if it lies near the middle of the distribution, say inside the interquartile range, and say so explicitly.**
- Q2.2d: story scale. **Default: 2.8%, from the story.**

---

## 5. Part 2.3: Unbiased every night, wrong in the end (6 points)

**(i) Goal.**
- Verify E[σ̂²_{t+1} | σ̂²_t] = σ̂²_t by one-step simulation. The expected value on night 2,500 is therefore exactly 1.
- Measure the average nightly change in log σ̂² for n = 50, 500 and 5,000, and propose the formula in n.
- Plot the histogram of log σ̂² across the 1,000 desks on night 2,500.
- In one paragraph, reconcile three numbers: the expectation (exactly 1), the median (≪ 1), and the 1,000-desk mean, including the top-10 share of the sum.
- Explain why "unbiased at every step" does not protect a self-trained pipeline.
- Reconcile the result with the 2.1 commitment.

**(ii) Method steps** (cell 28).

*A. Martingale check*
```python
def one_step(s0, n_scen, m, rng, chunk=10_000):
    """m desks all at sigma2_t = s0; one pipeline step; returns sigma2_{t+1}, shape (m,)."""
    out = np.empty(m)
    for i in range(0, m, chunk):
        c = min(chunk, m - i)
        out[i:i+c] = (rng.standard_normal((c, n_scen)) * np.sqrt(s0)).var(axis=1, ddof=1)
    return out
M = 200_000
```
- Starting values:
  - `s0 = 1.0` (stream `mart_s0_1`);
  - `s0 = s2_one[-1]`, the one desk's night-2,500 value, which is "the same σ̂²_t" taken from an actual desk (stream `mart_s0_one`).
- For each start, compute `ratio = out/s0` and **Table T2.3a**:
  - `ratio.mean()`;
  - SE = `ratio.std(ddof=1)/sqrt(M)` (L2 p.63);
  - 95% CI with `st.t.ppf(0.975, M-1)` (L2 p.84, p.87);
  - t = (mean − 1)/SE (L2 p.91/p.97). "Equals" means |t| < 2 (L2 p.85, p.91).
- Diagnostics from the **same** draws (no extra cost). These are the seed of the paradox:
  - `np.median(ratio)`;
  - share of desks that go **down**, `(ratio < 1).mean()` ± √(p(1−p)/M);
  - `np.log(ratio).mean()` with SE and t-stat against 0.
- The two starting values should give the same ratio distribution within MC error. That is the empirical demonstration of **scale invariance**: the nightly factor does not depend on the level. This is what makes log σ̂² a sum of iid nightly changes.
- Planned precision: relative SE ≈ 0.0634/√200,000 ≈ 1.4e-4 (Appendix A.6).
- "Expected value on night 2,500 is exactly 1": chain the verified one-step property over 2,499 nights (the tower property, elementary and **not on the slides**; the exam cell 27 states the conclusion).
- Visual check: the cross-desk mean of σ̂² in the fan chart (F2.3b) stays near 1 for early nights, within ±2 SE.

*B. Drift vs n*
```python
DRIFT_DESIGN = {50: (500, 500), 500: (500, 500), 5000: (500, 500)}   # n: (desks D, steps T)
rows = []
for n, (D, T) in DRIFT_DESIGN.items():
    L = run_desks(D, n, T + 1, rng_for(f'drift_{n}'))   # (T+1, D); row 0 = log 1 = 0
    inc = np.diff(L, axis=0)                           # (T, D): nightly change in log sigma2
    per_desk = inc.mean(axis=0)                        # (D,)  = (L[T] - L[0]) / T
    drift = per_desk.mean()
    se = per_desk.std(ddof=1) / np.sqrt(D)             # desk-level SE (L2 p.63)
    se_iid = inc.std(ddof=1) / np.sqrt(D * T)          # check: should agree with se
    ...
```
- **Table T2.3b** columns:
  - n, D, T;
  - mean Δlog σ̂² and SE, with `se_iid` alongside;
  - 95% CI (t_{D−1});
  - **n × drift ± n × SE**;
  - candidates −1/n and −1/(n−1);
  - t-stats against each candidate, computed by hand (L2 p.97);
  - sd(Δ), and sd(Δ)·√(n/2) (supporting: the spread per night).
- **Proposed formula:** state the one the data support. The design expects n × drift ≈ −1 at all three n, i.e. **drift ≈ −1/n**.
  - The planned SEs of n × drift are ≈ 0.02, 0.06 and 0.20 for n = 50, 500 and 5,000 (Appendix A.6).
  - That is ample to establish the 1/n scaling across two decades of n, and to reject zero drift even at n = 5,000 (≈ 5 SE).
  - It is **not** enough to separate −1/n from −1/(n−1), which differ by 1 SE at n = 50 and by nothing at larger n. The answer says so.
- **Optional run A, `drift_50_precise` (default ON, cheap):**
  - n = 50, D = T = 2,000, which gives SE ≈ 1.0e-4.
  - This separates −1/n from −1/(n−1) by ≈ 4 SE. It cannot separate −1/(n−1) from the exact value, which is ≈ 1.4 SE away.
  - If it favours −1/(n−1), the answer states the formula as **"≈ −1/(n−1), which is ≈ −1/n for these n"**.
  - It links the denominator to **degrees of freedom**: each night's fit has only n−1 "real observations of variability" (L2 p.57).
- **Optional run B, `drift_small` (default OFF):** n ∈ {10, 20}, D = T = 1,000. This shows where the simple formula bends: at n = 10 even −1/(n−1) is off by ≈ 8 SE. It needs the digamma expression (Appendix A.2, **beyond the lectures**).
- **Optional F2.3c:** |drift| vs n on log-log axes, with error bars ±1.96 SE and the line 1/n (L3 p.48–50). An OLS of log|drift| on log n (`smf.ols`, L2/L3 house style) is only worth it if run B adds points, because 3 points are too few. **Default: plot only, no OLS.**
- Cross-check against 2.2 (no new simulation):
  - The 2.2 run is n = 500, 1,000 desks × 2,499 steps. Its increments `np.diff(logS2, axis=0)` give a second, more precise n = 500 drift (SE ≈ 4e-5). It must agree with T2.3b's n = 500 row within the combined SE.

*C. Histogram and the three numbers*
- **F2.3a:** `ax.hist(logS2[-1], bins=40, density=True)`. Overlay the normal pdf with the **sample** mean and sd of `logS2[-1]` (L2 p.79 style; CLT L2 p.64). Vertical lines:
  - 0 = log of the expectation 1;
  - `np.log(S2_T.mean())` (log of the 1,000-desk mean);
  - `np.log(np.median(S2_T))`;
  - `np.log(0.01)` (one-tenth-VaR threshold);
  - the one desk `np.log(s2_one[-1])`;
  - the story `np.log(story_s2)`.
- **F2.3b (fan chart):** uses the 2.2 matrix and no new draws.
  - `q = np.percentile(logS2, [5, 50, 95], axis=1)` with shape (3, 2500), plotted as a band plus the median line;
  - `np.log(np.exp(logS2).mean(axis=1))`, the log of the cross-desk mean, with shape (2500,);
  - the 0 line; optionally the one desk's path overlaid.
  - It shows the median sliding down a straight line on the log scale, the band widening, and the mean hugging 1 early and then becoming erratic.
- **Table T2.3c (reconciliation):**
  - expectation = 1 (exact, from A);
  - `np.median(S2_T)`;
  - `S2_T.mean()` and its naive SE `S2_T.std(ddof=1)/sqrt(1000)`, with a note that it is unreliable (below);
  - **top-10 share** `np.sort(S2_T)[-10:].sum()/S2_T.sum()`;
  - top-1 share `S2_T.max()/S2_T.sum()`;
  - share of desks above 1, `(S2_T > 1).mean()`;
  - consistency checks: `exp(2499 × drift_500)` against the median (from T2.3b), and `sqrt(2499) × sd(Δ)` against `logS2[-1].std(ddof=1)`.

**(iii) Grounding per step.**
- Unbiasedness of the fit gives the martingale: L2 p.56–57, and exam cell 27 states it.
- One-step "many desks, average" is MC: L2 p.59–70, p.78–79.
- SE, CI, t-stat by hand: L2 p.63, p.84–85, p.87, p.91, p.97.
- Log scale / multiplicative change: L3 p.48–49.
- Sum of iid nightly changes looks bell-shaped: L2 p.64 (CLT).
- The average nightly change concentrates at its mean as T grows: var(X̄) = σ²/n, L2 p.63.
- Mean of σ̂² across desks unreliable under extreme skew: L2 p.65–70 (the CLT is slow for skewed data).
- Unbiased ≠ precise: L2 p.71–72, L5 p.6–7.
- DoF: L2 p.57.
- χ², digamma and lognormal are **not taught**; see Appendix A, clearly labelled.

**(iv) Assumptions.**
- **A1** as in §3.2.
- **A6** drift runs start at σ̂² = 1 on night 1. The start is irrelevant, because the nightly factor is scale-free (shown empirically in A).
- **A7** "average nightly change" is the mean over desks and nights of log σ̂²_{t+1} − log σ̂²_t. This equals the per-desk (log σ̂²_{T+1} − log σ̂²_1)/T averaged over desks.
- **A8** D = T = 500 for each n. This is "a few hundred desks and a few hundred nights" (exam cell 27), and the precision is quantified in Appendix A.6.
- **A9** M = 200,000 for the martingale check.

**(v) Guards.**
- Both martingale starts use independent streams.
- `se` vs `se_iid` agree. If they do not, the nightly increments are not independent and the desk-level SE is the one to trust.
- `np.isfinite` asserts.
- The n = 5,000 run is chunk-free but 20 MB per night, which is fine.
- Nothing is compared with a quantity estimated from the same simulated sample, except the explicitly labelled consistency checks.
- The histogram uses the natural log.

**(vi) Outputs.**
- T2.3a (martingale), T2.3b (drift; plus optional rows A and B), T2.3c (reconciliation).
- F2.3a (histogram, required), F2.3b (fan chart), optional F2.3c.

**(vii) Answer checklist (cell 29).** One paragraph for the reconciliation, plus short statements for the measured items.
- [ ] Martingale verified: mean ratio, SE, |t| < 2 for both starts.
- [ ] Hence E[σ̂²_2500] = 1 exactly (chained one-step property).
- [ ] Measured average nightly change in log σ̂² for n = 50, 500, 5,000, with SEs.
- [ ] Proposed formula (≈ −1/n; plus the n−1 nuance if run A is used), with the precision caveat.
- [ ] Histogram of log σ̂² on night 2,500 (F2.3a).
- [ ] Reconciliation paragraph covering all three numbers and the top-10 share.
- [ ] Why unbiased-per-step does not protect a self-trained pipeline.
- [ ] Explicit reconciliation with the cell-23 (2.1) prediction: what was right and what was wrong.

**Reasoning skeleton for the paragraph.** The student writes it with their own numbers.
1. **Expectation = 1.** It is exact, and the one-step check confirms it. But it is an average over *hypothetical re-runs* of the whole ten years (L2 p.78: "if the world were run again"). It is not a statement about the path any one desk lives on.
2. **Median ≪ 1.**
   - Each night multiplies σ̂² by a random factor with mean 1 that is skewed: its median is below 1, more than half the desks go down each night, and its log has a negative mean. All three are measured in T2.3a.
   - On the log scale (L3 p.49) the path is a sum of 2,499 independent nightly changes, each averaging ≈ −1/n (T2.3b).
   - So log σ̂² drifts down by ≈ 2,499/n while spreading like √(2,499) × sd(Δ).
   - The CLT (L2 p.64) makes the cross-section bell-shaped on the log scale (F2.3a), and its centre, the median, sits near exp(−2,499/n).
3. **The 1,000-desk mean is neither.**
   - The expectation is held up by a vanishingly small set of desks in the far right tail, too rare to show up reliably among 1,000 draws.
   - The ten largest desks hold {top-10 share} of the sum. The sample mean is therefore dominated by a handful of desks. It usually lands below 1 but far above the median, and it jumps whenever one giant desk appears.
   - Its naive SE is itself unreliable, because the CLT is slow for data this skewed (L2 p.65–70).
   - *Branch:* if the student's mean happens to exceed 1, the same sentence holds, because one or two giant desks carry it. Check the top-1 share.
4. **Why unbiasedness does not protect.**
   - Unbiasedness controls the mean, not the precision or the typical outcome (L2 p.71–72; L5 p.6–7).
   - Each night's estimation error is **carried forward as the next night's truth** and never corrected. After night 1 no new information about σ enters, and the only "real observations of variability" are the ones used on night 1 (L2 p.57).
   - So errors compound multiplicatively. The variance of σ̂² grows without bound while its mean is pinned at 1, and "unbiased" ends up describing a vanishing minority of desks while almost every desk's VaR decays toward 0.
   - A VaR that keeps falling in an unchanged market is the too-good-to-be-true pattern (L1 p.39): confidently wrong, with no tell (L1 p.80).
5. **Reconcile with 2.1.** Quote cell 23, then say which part held (e.g. "the average stays at 1") and which failed (e.g. "the typical desk does not").

**(viii) Open questions.**
- Q2.3a: martingale starts. **Default: s0 = 1 and the one desk's night-2,500 σ̂²; M = 200,000.**
- Q2.3b: drift design. **Default: D = T = 500 for each n.** Upgrade n = 5,000 to T = 1,000 (≈ +53 s, SE of n·drift 0.14) only if wanted.
- Q2.3c: optional run A (n = 50 precision). **Default: ON.** Optional run B (n = 10, 20). **Default: OFF.**
- Q2.3d: add a short, labelled "supplementary derivation (beyond the lectures)" note after the paragraph (Appendix A.1–A.4 in 3–4 lines). **Default: yes, clearly labelled; any number it cites must be computed in a cell (e.g. `scipy.special.digamma`).**

---

## 6. Part 2.4: What synthetic data can and cannot carry (6 points)

### 6.1 Part (a): keep the real days permanently

**(i) Goal.** The library holds each desk's own 500 real days permanently plus that night's 500 scenarios (1,000 returns). Report the median, 5th and 95th percentiles of σ̂² on night 2,500 across 1,000 desks.

**(ii) Method steps** (cell 31, block a).
```python
Z = rng_for('mixed_real').standard_normal((N_DESKS, N_SCEN))
real = Z / Z.std(axis=1, ddof=1, keepdims=True)          # DEFAULT: rescale only; each desk keeps its own sample mean
assert np.allclose(real.var(axis=1, ddof=1), 1.0, rtol=0, atol=1e-12)

def run_mixed(real, nights, rng):
    """Anchored pipeline: library = the desk's fixed real days + tonight's N_SCEN scenarios (1,000 returns)."""
    D = real.shape[0]
    log_s2 = np.empty((nights, D))
    s2 = real.var(axis=1, ddof=1)                        # night 1: library = 500 real days only -> exactly 1
    log_s2[0] = np.log(s2)
    for k in range(1, nights):
        scen = rng.standard_normal((D, N_SCEN)) * np.sqrt(s2)[:, None]    # N(0, s2): fitted mean set to zero
        s2 = np.concatenate([real, scen], axis=1).var(axis=1, ddof=1)     # half real, half synthetic
        log_s2[k] = np.log(s2)
    return log_s2

logS2_mix = run_mixed(real, NIGHTS, rng_for('mixed_synth'))                      # (2500, 1000)
real_dm = (Z - Z.mean(axis=1, keepdims=True)) / Z.std(axis=1, ddof=1, keepdims=True)
logS2_mix_dm = run_mixed(real_dm, NIGHTS, rng_for('mixed_synth'))                # robustness, same synthetic draws
```
- The literal `np.concatenate` is kept because it is the manual as written. A pooled-sums shortcut is possible but unnecessary at ≈ 43 s.
- **Table T2.4a:** for the anchored run, `np.percentile(np.exp(logS2_mix[-1]), [5, 50, 95])`, plus the mean and the fraction with VaR < 0.1 × truth. Place it **side by side** with the 2.2 replacement row from T2.2 and the demeaned-robustness row.
- Stability rows: the same percentiles at nights 2, 3, 5, 10, 100, 1,000 and 2,500 for the anchored run. These show the band widening over the first few nights and then **not** growing further.
- **F2.4a:** 5–95% bands and medians of log σ̂² vs night for the replacement run (2.2) and the anchored run (2.4(a)) on one axis, with the 0 line. This is the picture of "anchor stops the random walk".

**(iii) Grounding.**
- MC design: L2 p.59–70, p.78–79.
- Unbiased fit: L2 p.56–57.
- Pooled library variance as a sums-of-squares decomposition, SS_total = SS_real + SS_synth + between-halves term. This is the ANOVA identity (L2 p.26–28) for a regression on a real/synthetic dummy (L3 p.45, p.51–52).
- Taking conditional expectations with E[SS_synth | σ̂²_t] = 499 σ̂²_t (unbiasedness, L2 p.56) and var(x̄_synth) = σ̂²_t/500 (L2 p.63) shows the real half is a fixed weight ≈ ½ pulling σ̂² back to ≈ 1 every night. This short explanation is optional in the answer.
- DoF as information: L2 p.57. The real half is the only source of information about σ.

**(iv) Assumptions.**
- **A10:** library = 500 permanent real days + **tonight's** 500 scenarios, replacing yesterday's scenarios. The exam says "the library holds 1,000 returns", so scenarios do not accumulate.
- **A11:** night 1 fits on the 500 real days alone (σ̂²_1 = 1), because nothing synthetic exists yet. From night 2 the fit is on 1,000 returns.
- **A12 (demeaning decision):**
  - **Default: do not demean.** Rescale only, so each desk's sample variance is exactly 1 as the exam requires, and its real days keep their own small sample mean x̄_R ~ N(0, 1/500).
  - This mirrors real data. dj30 has mean 0.000645 ≠ 0 while the pipeline sets its mean to zero.
  - Implication: the synthetic half is centred at 0 and the real half at x̄_R, so the pooled variance picks up a between-halves term 250(x̄_R − x̄_S)²/999. This is a small upward push of order x̄_R², ≈ 0.1% on average, that does not accumulate.
  - The demeaned robustness run on common random numbers measures it directly. The answer reports it in one clause.

**(v) Guards.**
- Each desk has its **own** real days (a (1000, 500) matrix, not one shared vector).
- The real matrix is never modified inside the loop.
- `assert logS2_mix.shape == (NIGHTS, N_DESKS)` and `np.allclose(np.exp(logS2_mix[0]), 1)`.
- `ddof=1` on the 1,000-return library.

**(vi) Outputs.** T2.4a (with the stability rows), F2.4a.

### 6.2 Part (b): seed the library with real dj30 returns

**(i) Goal.**
- Reduce dj30 to 1,511 daily market returns. Report their sample volatility and excess kurtosis.
- On night 1, before any self-generated data, compare:
  - the empirical 1% VaR with the pipeline's reported VaR;
  - the real excess kurtosis with that of the 500 night-1 scenarios.

**(ii) Method steps** (cell 31, block b).
1. **Describe the data first** (AI guide §6.1):
   - `dj = pd.read_csv(_path + 'dj30.csv')`;
   - print `dj.shape` (expect 46,445 × 9), `dj.dtypes` (`date` int64) and `dj.head()`;
   - `dj.date.nunique()` (expect 1,511);
   - `assert dj.groupby('date').MrkRet.nunique().max() == 1`;
   - `dj.MrkRet.isna().sum() == 0`.
2. `r = dj.groupby('date').MrkRet.first()`. Then:
   - `assert len(r) == 1511 and r.index.is_monotonic_increasing`;
   - `r.index = pd.to_datetime(r.index.astype(str), format='%Y%m%d')` (for plotting only).
3. `sd_real = r.std(ddof=1)`, the daily sample volatility, reported in %. The profile value is 0.011951, and a ddof=0 value of 0.011947 is immaterial.
4. `kurt_real = kurtosis(r, fisher=True, bias=True)`, the exam-imported scipy function with its defaults written out explicitly. Footnote: `kurtosis(r, fisher=True, bias=False)` = `r.kurt()`. Profile values: 24.07 and 24.16.
5. Night-1 comparisons (default library = all 1,511 days, A13):
   - `var_pipe = Z01 * sd_real`. This is manual steps (1)+(4): σ̂ from `var(ddof=1)` with the mean set to zero. It should reproduce the story's "2.8%", since the profile gives 2.326 × 0.011951 = 2.78%.
   - `var_emp = -np.percentile(r, 1)` (linear, default). Also report `method='lower'` and `'higher'` to show that the interpolation choice is immaterial (profile: 3.325%, range 3.169–3.342%).
   - In σ̂ units: `var_emp/sd_real` against `Z01`. The ratio is "the size of the assumption" (L2 p.80).
   - Optional supporting number (default on): `exceed = (r < -var_pipe).mean()` against the 1% the normal model promises, with SE √(p(1−p)/1511) (L2 p.63). It restates the same comparison through the exam's VaR definition. No formal back-test is used.
6. Night-1 scenarios: `scen1 = rng_for('dj_scen').normal(0.0, sd_real, size=N_SCEN)` (manual step (2)); `kurt_scen = kurtosis(scen1, fisher=True, bias=True)`.
7. **Reference band for kurtosis under normality** (parametric simulation, L2 p.78–79):
   - `K0 = kurtosis(rng_for('kurt_null').standard_normal((10_000, N_SCEN)), axis=1, fisher=True, bias=True)`;
   - report `np.percentile(K0, [2.5, 50, 97.5])` and `K0.max()`;
   - report where `kurt_scen` falls (percentile) and `(K0 >= kurt_real).mean()` (expected to be 0 out of 10,000).
   - This replaces the untaught formula SE ≈ √(24/n).
8. **Bootstrap SEs** (L2 p.80; default ON). Resample the 1,511 days with replacement, B = 10,000, in chunks of 500 rows:
   - `idx = rb.integers(0, 1511, size=(500, 1511))`, then `rv = r.to_numpy()[idx]`;
   - `-np.percentile(rv, 1, axis=1)`, `Z01*rv.std(axis=1, ddof=1)`, and their paired difference;
   - report the bootstrap SE of `var_emp`, of `var_pipe` and of the difference.
   - **Caveat in the answer:** iid resampling ignores volatility clustering (March 2020), so these SEs are optimistic (L2 p.80: Route 3 still assumes independence; L1 p.38).
9. **Figures.**
   - **F2.4b** (L1 p.38 style): `r` vs date, with horizontal lines at −`var_pipe` ("pipeline night-1 VaR, normal") and −`var_emp` ("empirical 1%"), and March 2020 shaded with `axvspan`.
   - **F2.4c:** density histograms of `r` (bins=100) and `scen1` on shared bins, the N(0, sd_real²) pdf, and a **log y-axis** to make the tails visible (L3 p.48 "think about scale"; L2 p.79 overlay idiom).
10. **Table T2.4b:**
    - sd_real;
    - kurt_real (and the bias-corrected value);
    - var_pipe and var_emp in % and in σ̂ units, with bootstrap SEs;
    - exceedance fraction;
    - kurt_scen;
    - K0 band.

**(iii) Grounding.**
- Normal fit vs "use the actual data": L2 p.79–80.
- Normal tail masses and fat tails: L2 p.38, p.81.
- "Inference and prediction rely on this model being true": L3 p.3.
- One big outlier inflates s: L3 p.14 (March 2020 days inflate σ̂ while the normal shape still understates the extreme quantile).
- Plot the raw series: L1 p.38.
- Parametric reference band: L2 p.78–79.
- Bootstrap SE: L2 p.80.
- **VaR, empirical VaR and kurtosis are defined by the exam (cells 20, 30), not by a lecture.**

**(iv) Assumptions.**
- **A13 (library-size ambiguity).** The library "holds 500" but there are 1,511 real days. **Default: all 1,511 days form the night-1 library.** Reasons:
  - the exam says the reduction "gives 1,511 trading days for 2016–2021 including March 2020" and asks for the volatility and kurtosis "of the real returns";
  - 2.326 × sd(1,511 days) = 2.78% reproduces the story's "2.8%";
  - any 500-day window would be an arbitrary choice.
  - The pipeline still draws N_SCEN = 500 night-1 scenarios, as the exam's "500 night-1 scenarios" says.
  - Only night 1 is needed, so the size change after night 1 is irrelevant.
  - Optional robustness: the same three quantities for the most recent 500 days (which include March 2020) and the first 500 days (2016–17, calm).
- **A14:** the empirical VaR uses `np.percentile` linear interpolation (stated), and the lower/higher range is shown.
- **A15:** kurtosis is the scipy default (excess, biased), stated. The bias-corrected value is a footnote.
- **A16:** scenarios are drawn around mean 0 (manual step (1)). The real mean of 0.000645 is ignored by the pipeline, which is part of "what synthetic data lacks" but immaterial for VaR.

**(v) Guards.**
- One value per date (asserted).
- No use of `RET`.
- Chronological index (asserted).
- `ddof=1`.
- q in percent.
- `axis=1` on the kurtosis matrix.
- The dj30 series is used only descriptively (no split exists).
- The bootstrap caveat is stated.

**(vi) Outputs.** T2.4b, F2.4b, F2.4c, and the robustness rows if Q2.4b is approved.

### 6.3 Answer checklist and reasoning skeleton (cell 32)

**Answer checklist.**
- [ ] (a) Median, 5th and 95th percentiles of σ̂² on night 2,500 (1,000 desks), set against the 2.2 replacement numbers.
- [ ] (b) Sample volatility and excess kurtosis of the 1,511 real returns, with the estimator stated.
- [ ] (b) Night 1: empirical 1% VaR vs the pipeline's VaR, in % and in σ̂ units.
- [ ] (b) Night 1: real excess kurtosis vs that of the 500 scenarios, with the normal reference band.
- [ ] Paragraph: what a synthetic sample contains that the real one did not.
- [ ] Paragraph: what it lacks.
- [ ] Paragraph: which experiment shows loss of information and which shows accumulation of noise.
- [ ] Paragraph: the implication for any self-retrained pipeline (risk model, return forecast, language model).
- [ ] Paragraph: the only thing that stops it.

**Reasoning skeleton for the one paragraph.** Keep the analogies honest and labelled.
- **Contains (that the real sample did not):**
  - exactly the fitted model and nothing else: its assumptions imposed perfectly (normal shape, zero mean, independent days, one constant variance);
  - its own estimation error, now treated as truth: σ̂ in place of σ (L2 p.78, "treat that fitted line as if it were the truth");
  - fresh random noise.
  - The 2.4(b) night-1 scenarios have excess kurtosis inside the normal reference band, near 0, while the real returns have ≈ 24.
- **Lacks:**
  - any new information about the world. It is a function of σ̂ and a random-number generator, so it adds no "real observations of variability" (L2 p.57);
  - everything the model threw away: fat tails, the March-2020 days, calm-vs-panic regimes (L2 p.80; L1 p.38), the non-zero mean, and the true 1% quantile. The empirical VaR exceeds the normal VaR by {x σ̂ units}.
  - Optional flagged extension: a single normal cannot mimic a mixture of calm and panic regimes (L7 p.10–12; L7 does not discuss kurtosis).
- **Loss of information = experiment (b).** On night 1, before any feedback, the synthetic sample has already lost the tails. Every later generation is drawn from a normal, so they never come back. Parametric simulation "only reproduces the model's own assumptions" (L2 p.79–80; "the gap between them is the size of the assumption").
- **Accumulation of noise = experiment (a)**, with 2.2 as the uncapped case.
  - The real half contains all the information, and on its own it gives σ̂² = 1 exactly.
  - The synthetic half adds nothing but estimation noise, fed back night after night. The band widens over the first few nights and then stops (stability rows, F2.4a), because the permanent real half anchors it.
  - In 2.2 nothing anchors it, so the noise compounds into an unbounded random walk on the log scale.
- **Implication for any pipeline retrained on its previous version's output:**
  - the first generation discards whatever the model cannot represent (the tails first);
  - after that, each generation's estimation error becomes the next generation's truth, so noise compounds;
  - per-step unbiasedness does not prevent collapse of the typical run (2.3). For the risk model, VaR → 0 and the desk becomes over-confident.
  - Return forecast: it is refit to its own forecasts and so learns nothing about returns. Language model: the output distribution narrows and rare content disappears. The last two applications are the exam's framing, argued by analogy only.
  - The pipeline never asks where its data came from, which is the validation failure of L1 p.81–82. "Never use the same data twice" (L5 p.41) is the cousin: here the pipeline reuses its own output as if it were new evidence.
- **The only thing that stops it:** fresh real data, i.e. information from outside the model, kept in the training set every generation (the anchor in (a)), with provenance tracked so synthetic data never silently replaces real data. Note the limit: (a) stops the noise but cannot restore the tails lost in (b). Only real observations can carry them.

### 6.4 Open questions for 2.4

- Q2.4a: demean the simulated real days? **Default: no (rescale only)**, with the demeaned common-random-numbers robustness row reported.
- Q2.4b: dj30 night-1 library. **Default: all 1,511 days.** Optional robustness with the most recent and first 500-day windows. **Default: off.**
- Q2.4c: kurtosis estimator. **Default: `scipy.stats.kurtosis` defaults (fisher=True, bias=True)**, with the bias-corrected value as a footnote.
- Q2.4d: percentile method. **Default: linear**, with the lower/higher range shown.
- Q2.4e: bootstrap SEs and the exceedance fraction. **Default: on**, with the iid caveat.
- Q2.4f: L7 mixture clause. **Default: omit**, or include one flagged clause if the student wants it.

---

## 7. Verification checklist (known-answer and two-route checks)

Run these before writing any answer (AI guide §6.4–6.5).

1. **Data (2.4b):**
   - 46,445 rows and 1,511 unique dates;
   - `MrkRet` constant within each date;
   - sd 0.011951;
   - kurtosis 24.07 (bias=True) and 24.16 (pandas);
   - 1st percentile −0.03325;
   - 2.326 × sd = 2.78% ≈ the story's 2.8%. All from data profile §B.
2. **Night 1:** σ̂²_1 = 1 for the one desk, for all 1,000 desks, and for all 2.4(a) desks (asserted).
3. **Unit test:** literal one-desk loop = vectorised pipeline (`np.array_equal`).
4. **Martingale:** |t| < 2 for the mean ratio at both starts, and the ratio distributions agree across starts (scale invariance).
5. **Two routes to the n = 500 drift:** T2.3b's row vs the 2.2 run's increments agree within the combined SE.
6. **Two routes to the night-2,500 median:**
   - `np.median(S2_T)` vs `exp(2,499 × drift_500)`;
   - `logS2[-1].std(ddof=1)` vs `sqrt(2,499) × sd(Δ)`.
7. **SE consistency:** `se` ≈ `se_iid` in every T2.3b row.
8. **Sanity bands** (Appendix A.5; bug-catchers, not answers): for n = 500 after 2,499 steps, the mean of log σ̂² should be near −5 and the sd near 3.2. A value near −10 signals the `ddof=0` bug (A.7). A value near 0 signals that scale was passed as the variance or that the library was not replaced.
9. **2.4(a):** the percentiles at night 100 and night 2,500 are similar (stationary), and the demeaned row differs only slightly.
10. **2.4(b):** `kurt_scen` lies inside the K0 band, and `kurt_real` lies far outside it.
11. **Reproducibility:** re-running any cell gives identical numbers, and every number quoted in cells 26, 29 and 32 is printed by a cell.

---

## 8. Cell-by-cell layout

| Cell | Content |
|---|---|
| 21 (given) | Unchanged except `SEED = <last four digits of student ID>`. |
| 23 | 2.1 answer, written before anything below. |
| 25 (+ optional extra cells) | Constants; streams; `run_desks`; literal one desk and unit test; F2.2; 1,000 desks (timed); T2.2. |
| 26 | 2.2 answer. |
| 28 (+ extra) | `one_step`; T2.3a; drift runs; T2.3b (plus optional runs A and B); 2.2 cross-check; F2.3a; F2.3b; T2.3c; optional F2.3c. |
| 29 | 2.3 answer: measured items plus the one reconciliation paragraph, including the 2.1 reconciliation. Optional labelled supplementary note. |
| 31 (+ extra) | (a) real days, `run_mixed`, anchored and demeaned runs, T2.4a, F2.4a. (b) describe dj30, reduce, sd, kurtosis, VaRs, scenarios, K0 band, bootstrap, T2.4b, F2.4b, F2.4c. |
| 32 | 2.4 answer: reported numbers plus the one paragraph. |

---

## 9. Consolidated open questions (defaults in bold)

1. **SEED:** the last four digits of the student's ID, still unknown. The notebook placeholder is `SEED = 0`. **It must be set before any P2 cell runs.** Every number depends on it.
2. Q2.1: **the student writes 2.1 now, unaided, before reading §2+; reconcile in 2.3 by quoting it.** Git commit as evidence: **only if the student asks.**
3. Q2.2a: night indexing. **2,499 random steps between night 1 and night 2,500.**
4. Q2.2b: **one desk independent of the 1,000.** Q2.2c: **report the percentile, judged against the interquartile range.** Q2.2d: **2.8% story scale.**
5. Q2.3a: **s0 ∈ {1, one desk's night-2,500 σ̂²}, M = 200,000.** Q2.3b: **D = T = 500 per n.** Q2.3c: **run A on, run B off.** Q2.3d: **short labelled supplementary note: yes.**
6. Q2.4a: **no demeaning, plus a demeaned robustness row.** Q2.4b: **all 1,511 days; window robustness off.** Q2.4c: **scipy kurtosis defaults.** Q2.4d: **linear percentile.** Q2.4e: **bootstrap and exceedance on.** Q2.4f: **L7 clause omitted.**
7. Data path: **use the local fallback for development and keep the exam's `/classes/...` path first in the lookup, so the submitted notebook runs on JupyterHub.**

---

## 10. Scope notes (deliberately NOT used)

- **Not taught, so not used:**
  - QQ plots and formal normality tests (L3 boundaries). A histogram with a normal overlay is used instead (L2 p.79).
  - Block or stationary bootstrap (L2 boundary). The iid bootstrap is used with a caveat.
  - HAC or robust SEs.
  - Annualised volatility (√252).
  - The analytic kurtosis SE √(24/n). The simulated band is used instead.
  - VaR back-testing tests (Kupiec, Christoffersen).
  - GARCH or any volatility model.
  - Expected shortfall.
- **Exam-defined, not lecture-defined:** VaR = 2.326 σ̂ and the whole pipeline (cell 20); empirical 1% VaR and excess kurtosis (cell 30); the martingale statement (cell 27); σ = 1 units and "real days = 500 draws from N(0,1)" (cell 20).
- **Beyond the lectures, labelled supplementary:** χ² law, digamma/trigamma, delta method / Jensen, lognormal, tower property, martingale convergence (Appendix A).
- **OUTSIDE SCOPE, proposed only if the user approves:** citing the "model collapse" literature on generative models trained on their own output; any formal VaR back-test.

---

## Appendix A. Supplementary derivations (beyond the lectures) and sample-size design

**Label every use in the notebook "supplementary, not from the lectures".** The exam asks for the formula to be *measured*, and these derivations only support it. Do not read this before writing 2.1.

- **A.1 One step.** Conditional on σ̂²_t, the 500 scenarios are iid N(0, σ̂²_t), so (n−1)σ̂²_{t+1}/σ̂²_t ~ χ²_{n−1}. That is, σ̂²_{t+1} = σ̂²_t · χ²_k/k with k = n−1. The factor has mean 1, which gives the martingale, and variance 2/k. Iterating expectations gives E[σ̂²_2500] = 1.
- **A.2 Log increment.** E[log(χ²_k/k)] = ψ(k/2) − log(k/2) = −1/k − 1/(3k²) + …, and Var = ψ′(k/2) ≈ 2/k + 2/k². The delta-method heuristic: E log X ≈ log E X − Var X/2 = −1/k. So the drift is ≈ −1/(n−1) ≈ −1/n.
- **A.3 Random walk.** log σ̂²_t = Σ_{j<t} Δ_j with iid Δ, so log σ̂²_t ≈ N(−(t−1)/k, 2(t−1)/k). The CLT is L2 p.64. By var(X̄) = σ²/n (L2 p.63), (1/t) log σ̂²_t → −1/k, so **every** desk's σ̂² → 0 even though the mean stays 1.
- **A.4 Mean vs median.** In the lognormal approximation, median ≈ e^{−(t−1)/k}, and E = e^{µ+s²/2} = 1 because µ = −s²/2 exactly at leading order. The mean is carried by desks near log σ̂² ≈ µ + s², which lies s ≈ 3.2 log-sds above the median. With 1,000 desks that region is sampled only a handful of times, which explains the top-10 share.
- **A.5 Bug-catching bands** (n = 500, 2,499 steps): E[log σ̂²_2500] ≈ −5.0 and sd ≈ 3.2. These are for §7 check 8 only. Do not quote them as results.
- **A.6 Sample-size design for the drift** (analytic, from ψ′):

  | n | sd(Δ) per night | exact drift | SE at D×T = 500×500 | SE of n·drift | (1/(n−1) − 1/n)/SE |
  |---|---|---|---|---|---|
  | 50 | 0.2041 | −0.020547 | 4.1e-4 | 0.020 | 1.0 |
  | 500 | 0.0634 | −0.002005 | 1.27e-4 | 0.063 | 0.03 |
  | 5,000 | 0.0200 | −0.000200 | 4.0e-5 | 0.20 | 0.00 |
  | 50 (run A: 2,000×2,000) | 0.2041 | −0.020547 | 1.0e-4 | 0.005 | 4.0 (exact vs −1/(n−1): 1.4) |
  | 10 (run B: 1,000×1,000) | 0.4987 | −0.115206 | 5.0e-4 | 0.005 | 22 (exact vs −1/(n−1): 8.2) |

  Relative SE ≈ √(2n/(D·T)). At fixed compute (∝ n·D·T) it grows ∝ n, so n = 5,000 is the expensive row.
  - The martingale check at n = 500 has relative SE of the mean ratio 0.0634/√200,000 ≈ 1.4e-4.
  - The same draws give SE(mean log ratio) ≈ 1.4e-4 against an expected ≈ −0.002 (≈ 14 SE).
  - They also give the one-step median ratio ≈ 1 − 2/(3k) ≈ 0.9987, about 7 SE below 1.
- **A.7 The ddof=0 bug.** The ddof=0 estimator is the Gaussian MLE, not on the slides as such (cf. L5 p.3–5). It multiplies every step by (n−1)/n and adds log(1−1/n) ≈ −1/n of drift. The signature is n·drift ≈ −2.
- **A.8 Why 2.4(a) is anchored.** Using §6.1(iii):
  - E[σ̂²_{t+1} | σ̂²_t] = [499 + 499.5 σ̂²_t + 250 x̄_R²]/999.
  - This is a contraction with coefficient ≈ ½ toward ≈ 1 + ½x̄_R².
  - The noise injected per night has relative sd ≈ √(2·499)/999 ≈ 3%, so the stationary band stays narrow and does not grow.
