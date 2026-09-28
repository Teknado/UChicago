# Gemini vs Claude: BUSN 41210 final project, a critical comparison

*Branch `financial_analytics_review`, written 2026-09-28. Nothing on `financial_analytics_opus` was changed.
Gemini's files are stored unmodified in `gemini/`. Every number below is printed by a script or a notebook
cell in this branch; the evidence index (§10) says where.*

**How to read this.** §1 is the verdict in one page. §2 lists what must be fixed in Gemini's notebook before
anyone reads it. §3–§5 go problem by problem: what each solution did, where they differ, which one is right
and why. §6 checks Gemini's supporting documents, including the asset-identity decoding. §7 collects every
place where **my (Claude's) work is wrong or weaker**, with the reasons. §8 gives the answers I believe are correct.
§9 covers the identity decoding and the idea of testing on real 2025–26 data.

**Labels used.** "Gemini" means `Final_Autumn_submission_improved.ipynb` unless "Gemini original" is written.
"Claude" means `Final_Autumn_2026-1.ipynb` on `financial_analytics_opus` (commit `955e6be`).
Severity: **Serious** = invalidates a result or breaks a binding exam rule; **Error** = a wrong statement,
number or method that a grader would mark down; **Weakness** = defensible but incomplete or overstated;
**Style** = presentation only.

---

## 0. What was done

1. **Both Gemini notebooks were re-executed** from copies in this repository's environment (Python 3.11,
   pandas 3.0.6, scikit-learn 1.9.1). The original ran in 166 s and the improved one in 197 s, both with no errors. The original
   reproduces its saved outputs exactly. The improved one does **not** (see §2.1).
2. **A blind review.** The two notebooks were rendered as text (with HTML tables converted), labelled only
   "Solution A" (Gemini improved, with re-run outputs) and "Solution B" (Claude), and handed to seven
   independent reviewer agents. The reviewers had no access to the repository, logs or any file naming an author.
   Each one covered one area: Problem 1, Problem 2, Problem 3 data and leakage, Problem 3 models and inference, Problem 3
   portfolios, the write-up, and an overall grade. Each re-ran code to check claims. Their reports are in
   `review/blind_reviews/` and the key is in `review/blind_key.md`.
3. **My own targeted checks** in `review/checks/`, each a script with its printed output. They cover the placebo, a
   working placebo, where the portfolio's gain comes from, bootstrap SEs, the forest comparison, penalty
   dependence, the original notebook's benchmark, and the identity document.
4. I then weighed the reviewers' findings. Where a reviewer made a claim I rely on, I re-ran it myself. Two
   reviewer findings are against my own work and I confirmed both (§7).

You told me Gemini worked independently of my notebook. That makes the overlap in design striking, and it
counts as mutual corroboration of those choices. Both solutions:
- drop `x10` for `x86` and lag macro 2 months;
- rank characteristics within class;
- use country differentials against country 7 with trailing z-scores;
- use no country macro for class A;
- train on 2003–2010 and test on 2011–2024 with 14 December refits;
- use unpenalised class intercepts, P1 ∝ ŷ/σ̂ and 10 bp costs.

---

## 1. The verdict in one page

**Scores from the blind review** (the reviewers did not know which solution was which):

| area (points) | Gemini | Claude | reviewer |
|---|---|---|---|
| Problem 1 (20) | 14.0 | 19.5 | R1 |
| Problem 2 (20) | 12.5 | 19.0 | R2 |
| P3 data, features, leakage (15) | 6 | 14 | R3 |
| P3 models, Q1, Q2 (20) | 5 | 19 | R4 |
| P3 portfolios, Q3 (15) | 6 | 14 | R5 |
| P3 write-up (10) | 2.5 | 8.5 | R6 |
| **Sum of the specialist scores (100)** | **46** | **94** | |
| Overall grader, end to end (100) | 60.5 | 98.5 | R7 |

The overall grader was more lenient on Gemini's Problem 1 (18.5 against R1's 14). R1 checked Gemini's explanations
by experiment and found them wrong, so I weight R1's score more.

**Neither solution has a fatal error in the AI Coding Guide's sense.** There is no shuffled split on time
series and no transformation fitted before the split, in either one. Gemini's core pipeline is leak-free. Its problems
are execution, reporting and interpretation.

**The findings that matter most:**

1. **Gemini's improved notebook was never run after its Problem 3 rewrite** (Serious, §2.1). Its saved Problem 3
   outputs are byte-for-byte the *original* pipeline's: 180 test months, PCR, Elastic Net, 12/24/36-month
   placebos. None of those match its own code or write-up. A grader opening the file sees tables that contradict the paper.
2. **Gemini's placebo test cannot work** (Serious, §5.4). The shifted macro never reaches the model, so all three
   "placebos" print **−2.972%**, identical to real macro. The write-up nevertheless reports −2.715%, −3.084% and
   −2.890%, which no code produces, and concludes the placebo "proves" macro is noise.
3. **Gemini's Q3 answer ("decisively beats the benchmarks after costs") is not supported by its own pipeline** (Serious, §5.5).
   On Gemini's own forecasts, a portfolio built from **class means alone** earns a net Sharpe of **0.441**,
   against **0.426** for its forecast portfolio. With that class-means tilt as a regressor, the alpha is **0.00% (t = 0.01)**. The
   Sharpe gain over risk parity is **+0.12 with SE 0.11**. This is exactly my notebook's conclusion (Q3 = NO):
   the edge is a class-premium tilt concentrated in bonds, not characteristic timing.
4. **Gemini's Q1 answer ("resilient predictability in equities and bonds") contradicts its own tables** (Error, §5.3).
   Bonds score −0.306%. Characteristics do *worse* than class means alone (−0.089%, SE 0.070%).
5. **Gemini's Problem 2.3 answer misquotes its own measurements** (Serious under exam cell 1, §4). The cell prints
   −0.018645 / −0.003788 / −0.000233. The answer says −0.0204 / −0.0020 / −0.0002 (the formula values), and says they "match
   within Monte Carlo error". At n = 500 the measured value is almost twice the formula.
6. **Gemini's write-ups (both versions) are corrupted** (Error, §2.2). Control characters and deleted `$…`
   fragments make the LaTeX unreadable ("egin{pmatrix}", "	ext", ".97\%", "^2_{OOS}$"). The names of the
   characteristics are missing from the list that identifies them.
7. **Gemini original's headline R²_OOS of +0.369% is a benchmark artifact** (Error, §5.7). Against a pooled
   trailing mean, the same forecasts score **−0.085%**. It also fed `x10` with a 1-month lag into its macro models,
   which is the look-ahead trap the exam warns about.
8. **Where I (Claude) was wrong** (§7). One sentence in my 1.3 answer claims the forest "tries one of the three
   features at each split, so a split can be forced onto a weak input". That is refuted: with all three features
   allowed, the forest scores the same 88.75%. Beyond that, the weaknesses in my work are length (several answers
   exceed the exam's sentence/paragraph limits, and the paper is ~6,100 words), wording ("pre-registered" is stronger
   than version control can prove), and several small, disclosed approximations. None changes an answer.

**Where Gemini is genuinely good.**
- The improved core pipeline is correctly timed. It found the `x10`, `x11`, back-fill and `x5`-floor traps.
- It is short and runs in about 3 minutes.
- Its Problem 1 numbers are all right.
- Its 2.4 classification (loss of information vs accumulation of noise) is right.
- Its decoded identities match the panel exactly on every date it cites.

Where the two solutions compute the same thing, they agree to the digit. Examples: the Problem 1 accuracies; the
dj30 statistics; class-means R²_OOS of +0.087%; the benchmark Sharpes EW 0.282, RP 0.308, TSMOM 0.199 gross; about 60% of
P1's gross in bonds; asset_16 at about 22–24% on average and 42% at most.

---

## 2. Gemini's submission as a file: fix these first

### 2.1 Stale outputs in the improved notebook (Serious)

Cells 35, 39 and 40 differ between Gemini's two notebooks. Cell 39's source was rewritten, but its **outputs are
identical, byte for byte, to the original notebook's**. The saved output begins "STEP 1: EXPLORATORY DATA
ANALYSIS & ASSET CLASS DIAGNOSTICS", which is the original's heading, and it prints the original's models. The
current code prints "… & DATA TRAP AUDIT" and different models. So, as submitted:
- the code, the outputs and the paper disagree;
- the exam's rule "every number must come from a cell in this notebook" fails for essentially every Problem 3 number;
- Gemini's methodology log says "Every number, table, and beta matches Cell 39 execution exactly". That is true of a
  run that is not in the file, and false of the file.

Re-executing the notebook fixes most of this (my re-run is `review/runs/Final_Autumn_submission_improved__rerun.ipynb`),
but **not the placebo numbers**, which no version of the code produces (§5.4).

### 2.2 Corrupted write-up markdown (Error, both versions)

The write-up cells contain literal control characters (improved: 30 tabs, 8 backspaces, 2 form feeds and 7 carriage
returns; original: similar). `$name` fragments are also missing, the pattern of a shell expanding `$R`, `$x_2` and so on as
variables. Examples from the improved cell 40:
- "Characteristics … produce positive … predictability in country equities (^2_{OOS} = +0.054\%$ …), driven primarily
  by intermediate momentum ($), carry fundamentals ($), and long-horizon reversals ($)." The names of the
  characteristics are gone.
- "compressing annualized volatility to .97\%$" (3.97%); "Class A … (.97\%$)" (30.97%); "egin{pmatrix} … \mathbf{A}	ext{ (Commodities)}".
- 12 occurrences of "^2_{OOS}" with the "$R" lost, and 26 lines with unbalanced `$` (R6).

### 2.3 Real-world names stated as fact (Weakness)

The improved write-up says "Class A: Global Commodity Futures", "Nikkei 225 during the October 2008 Lehman
collapse", "UK Gilt during the September 2022 mini-budget crisis", "Japanese 10Y JGB", "BoJ YCC" and
"Sweden, Japan, Switzerland", all as fact, for anonymised data. You already plan to remove these. §9 says
what can and cannot be claimed.

---

## 3. Problem 1: trees and ensembles

**The numbers agree exactly** (both re-run by R1): baseline 64.25%; unpruned tree 85.00%; the sweep 83.50 / 90.75 /
90.75 / 89.50 / 88.25 / 87.50; forest 88.75%; boosting 89.00%; MDI Salary 0.519, Age 0.470. Gemini's permutation
importances (Age 0.2433, Salary 0.1817) use 10 shuffles and mine (25.5 pp, 17.1 pp) use 50. Both rankings are the same.

| | Gemini | Claude | verdict |
|---|---|---|---|
| 1.1 choice of size | 3 leaves, tie with 4 broken by the rule | Same; also proves the tie is exact (identical out-of-fold predictions) and shows under/over-fitting with training-fold accuracy | Both correct; Claude's is better evidenced. Gemini says "strictly prefers" while also saying 3 and 4 tie (Style), and quotes "0.3575" and "+26.50%", which no cell prints (Style) |
| 1.2 plain English | Correct two sentences | Correct two sentences with purchase rates, plus a labelled "technical note" | Both full credit; my extra note runs past the brief (Style) |
| 1.3 why ensembles don't win | "⌊√3⌋ = 1 feature per split means about 1/3 of splits are forced onto Gender"; "Gender provides zero signal"; "boosting over-iterates" | Main reason: the pattern is one rectangle the small tree already draws, so there is little variance or bias left to fix (L8 p.51, p.54); boosting fits its training folds to 97.4%. **Secondary sentence: "one feature per split, a split can be forced onto a weak input"** | **Both are partly wrong on the forest mechanism.** R1 and I checked: `max_features=None` gives the same 88.75%, and only 6.0% of the forest's splits are on Gender. The real cause is fully grown trees fitting noise: 99.8% training-fold accuracy, and `min_samples_leaf=10` lifts the forest to 91.0%. Gemini builds its whole explanation on the wrong mechanism (Error) and contradicts its own 1.4 output on Gender (Error). Mine uses it as a secondary point (Error in my work, §7). Gemini is also far over the 2–3 sentence limit (Error) |
| 1.4 MDI vs permutation | Salary first by MDI, Age first by permutation; blames cardinality, "≈116 vs ≈43 distinct values" | Same ranking. Prints the distinct values (108 vs 42 on the training rows; 117 vs 43 overall), split counts (5,623 vs 4,751) and permutation on training rows as a diagnostic | Same conclusion and both right (L8 p.58). Gemini's counts are printed by no cell, and 116 is wrong (Error). "Scrambling Age destroys the root split" confuses the pruned tree with the forest (Weakness) |
| 1.5 time series | Leakage inflates accuracy; use walk-forward | Same, precise, cited; notes that unshuffled KFold and stratification don't fix it | Both correct. Gemini says accuracy could "collapse to near-zero or negative" (accuracy cannot be negative, Error), has an off-by-one ("prior to month t−1"), and uses "purging/embargoing" without labelling it as untaught (Style) |

**Correct answers:** 64.25%; unpruned 85.00%; CV prefers 3 leaves (tied with 4, the tie goes to the smaller tree); neither
ensemble beats the 3-leaf tree, because the signal is a two-split rectangle and the defaults grow deep trees that fit
noise; MDI and permutation disagree on the top variable, and the held-out permutation ranking is the one to show;
shuffled CV on time series leaks the future and biases accuracy upward.

---

## 4. Problem 2: training on your own output

The two solutions use different random-number APIs (`np.random.seed` + legacy draws versus `SeedSequence` named
streams). Different draws are expected and legitimate. What matters is whether each is computed to the spec and
reported faithfully.

| | Gemini | Claude | verdict |
|---|---|---|---|
| Nights | 2,500 draw-and-refit steps after night 1 | 2,499 (night 1 fits the real library) | Claude matches the rules ("every desk starts on night 1 from a library whose sample variance is exactly 1"). Gemini is off by one night (Error; the numerical effect is tiny) |
| 2.2 one desk | σ̂² = 0.096634, VaR 0.7232, 79.3rd percentile | σ̂² = 1.405 × 10⁻⁵, VaR 0.0087σ, 3.2nd percentile | Both valid draws |
| 2.2 1,000 desks | mean 0.7811, median 0.007512, 5% 0.000018, 95% 1.2844, max 159.2; 54.10% below a tenth of the truth | mean 2.344, median 0.00585, 5% 0.000029, 95% 1.225, max 1,566; 57.8% (MC SE 1.6%); mean VaR 0.775σ | Both valid. Only Claude computes where the story sits (52.7th percentile) and gives an MC SE. Gemini asserts "typical" and quotes "69%" and "over 90%", which no cell prints (Error) |
| 2.3 one-step check | 100,000 desks from σ̂² = 1: mean 1.000002 | 200,000 desks from two starts, with CIs; reports that one CI misses 1 (p = 0.0088) and the chance of that | Both fine; Claude's is more careful and honest |
| 2.3 drift measurement | **One** step with 5,000 desks per n. Prints −0.018645 / −0.003788 / −0.000233, **then quotes −0.0204 / −0.0020 / −0.0002 and says they match within MC error** | 500 desks × 500 nights per n, with SEs (−0.01979, −0.00200, −0.000173), plus a precise n = 50 run (−0.02050) that tells −1/(n−1) apart from −1/n | **Gemini: Serious.** The quoted numbers are not the measured ones, and at n = 500 the measurement is about 2 SE from the formula. The design ignores "a few hundred desks and a few hundred nights" and is too noisy to establish anything (MC SE ≈ 45% of the drift at n = 500, 141% at n = 5,000; R2). Claude's is correct |
| 2.3 reconciliation | Log-normal argument; top-10 share 68.04% | Top-10 share 88.8%, single largest desk 66.8%; point-by-point reconciliation with the 2.1 prediction | Both mechanically right. Gemini never reconciles with its 2.1 prediction, which the question asks for (Error), and says a single desk is "mathematically guaranteed to collapse" (overclaim) |
| 2.4a keep the real days | Real days = raw N(0,1) draws, **not rescaled to variance 1**: median 0.9992, band 0.894–1.121 | Rescaled to exactly 1: median 0.9955, band 0.9406–1.059; demeaned robustness run | The rules require variance exactly 1. Most of Gemini's wider band is desk-to-desk variation in the real days themselves (correlation 0.86, R2), not accumulated noise (Error). With rescaling the band is about 0.94–1.07 |
| 2.4b dj30 | vol 1.195%, excess kurtosis 24.07, empirical VaR 3.33% vs 2.78%; night-1 scenario kurtosis 0.398 | Same numbers; scenario kurtosis −0.22 compared against 10,000 normal samples; bootstrap SE on the VaR gap (1.7 SE) | Agree. Both use all 1,511 days on night 1 (Claude says so; Gemini does not) |
| 2.4 classification | (b) loss of information, (a) accumulation of noise | Same | **Both correct** |
| Length | Bulleted lists, not one paragraph | One real paragraph each, but 2.3 and 2.4 run to ~840 and ~740 words in total | Both break the paragraph limit in different ways (Weakness each) |

**Correct answers.**
- The VaR does not stay near 2.326. Log σ̂² is a random walk with drift ≈ −1/(n−1) per night.
- The median desk ends near e^(−2499/499) ≈ 0.0067, i.e. a VaR of about 0.19σ. The story's 2.8% → 0.24% is roughly the median outcome.
- The expectation stays exactly 1 because a vanishing number of desks explode, so a 1,000-desk sample mean is unreliable.
- Keeping the real days anchors σ̂² in a narrow band around 1: accumulation of noise, contained.
- A normal model of dj30 loses the fat tails on night 1: loss of information.
- Only real data kept in the training set stops the collapse.

---

## 5. Problem 3: the research project

### 5.1 Design side by side

| element | Gemini (improved) | Claude | comment |
|---|---|---|---|
| Target | r / σ̂(36m, from returns, t−1) | Same | Identical |
| Characteristics | Within-class monthly ranks in [−0.5, 0.5] | Same (a within-class z-score as a pre-listed variant) | Identical |
| Global macro | log x6, x7, x8, x9; trailing 60m z; lag 2; × class | Same | Identical |
| Country macro | x86, x12, x13, log x14, log x15, x16; differential vs c7; trailing 60m z; lag 2; × class B, C, D; A = 0 | Same | Identical |
| Traps | x10 dropped, x11 dropped, back-fill period skipped, x5 floor avoided | Same, plus: 8 back-filled series set to missing; 6 duplicates and 26 cross-country-constant series found in the extended file; stale x1 in 2024 | Gemini's audit lines are partly **hard-coded strings** (asset list missing asset_22; "34 annual" should be 35). Its "97% redundancy" reason for dropping the extended file comes from a PCA on *unstandardised* columns. Standardised, the top five components explain **43.1%**, not 97% (checked) |
| Windows | Train 2003–2010, OOS 2011–2024, 14 December refits | Same | Identical |
| Schemes | Expanding only | Static, expanding (primary), rolling 96m | The exam asks for model × scheme. Gemini is missing two schemes (Weakness) |
| Models | M1 class means; ridge M2 (α = 50); ridge M3 (α = 200); RF (100 trees, leaf 150) | OLS, ridge (primary), lasso, PCR, RF (leaf 200), deep RF, per-class ridge: 38 pre-listed specs, Bonferroni 3.27 | Gemini's penalties are **hard-coded and never chosen on training data** ("every … penalty and hyper-parameter is chosen on training data only") (Error). Claude tunes inside each window on 3 forward folds |
| Uncertainty | None | Month bootstrap (B = 10,000) on every R²_OOS and difference; power analysis; block-bootstrap check (×1.19) | Gemini has no SE anywhere (Error) |
| Placebo | Intended: 36/48/60m shifts of global macro only. **Actually: no shift reaches the model** | Every raw global and country series shifted 36–120m (8 shifts), pipeline rebuilt | §5.4 |
| Portfolios | EW, RP, TSMOM, P1 ∝ ŷ/σ̂; 10 bp × Σ\|Δw\|; α regression | Same, plus P3 long–short, class-means P1, frozen P1, drift-adjusted turnover, a 0–50 bp grid, break-evens, sub-periods, concentration | §5.5 |

### 5.2 Leakage

Both pipelines are correctly timed. The reviewers (R3, R4) re-ran and confirmed:
- x1, x3 and x4 are lagged; x2 and x5 are used as stored; σ̂ uses returns to t−1;
- macro enters at t−2;
- the benchmark uses months before t only;
- training rows always precede the test month.

Gemini's full-sample PCA on the extended file is an audit that only led to dropping the file, so it is not a leak. Claude
additionally proves causality with a truncation test (all 64 feature columns are unchanged when data after 2010-12 is
deleted) and 20 lag spot checks. **No fatal error in either.**

### 5.3 Q1: how much is forecastable from lagged characteristics?

| R²_OOS vs pooled trailing mean (SE) | Gemini | Claude |
|---|---|---|
| Class means alone (M1) | +0.087% (0.236%)\* | +0.087% (0.236%) |
| Ridge on characteristics (M2) | −0.001% (0.248%)\* | +0.042% (0.241%) |
| Characteristics beyond class means | −0.089% (0.070%)\* | −0.045% (0.027%) |
| vs zero, ridge M2 | −0.322% | −0.279% (0.570%) |
| Stated answer | "Characteristics deliver resilient predictability within equities and bonds" | "NO: not detectable" (a true R²_OOS of ~0.5% would be needed to detect it half the time) |

\* SEs computed by me on Gemini's forecasts (`review/checks/gemini_p3_checks.out`, C4). Gemini reports none.

**Who is right: Claude.** The two pipelines agree on the numbers: both show that characteristics add nothing detectable beyond class
intercepts. Gemini's headline contradicts its own tables:
- bonds are −0.306%;
- equities are +0.054% with SE ≈ 0.21% and are below Gemini's own class-means model in that class;
- it calls M2 "optimal" although M1 beats it.

It also claims "Linear shrinkage dominates non-linear trees" from a comparison in which the **forest is given no class
information** while ridge gets class intercepts. With class dummies the forest rises from −0.115% to −0.082% (my check),
and the gap to ridge is within noise (Error).

### 5.4 Q2: does macro add anything? (the placebo bug)

**What Gemini's code does.**
1. It builds the interaction columns (`g_* × class`, `c_* × class`) from the *real* macro.
2. It builds placebo panels by overwriting the raw `g_*` columns with circularly shifted values.
3. It fits M3 on `m3_features`, the *interaction* columns, which were never rebuilt. Country macro is never shifted.

My check (`gemini_p3_checks.out`, C1): the rows are aligned, the shifted `g_*` columns change by up to 6.6, and the 39 inputs
M3 actually uses change by **exactly 0**. The placebo predictions are identical to the real ones at every shift.

**Gemini's evidence for Q2 is therefore void.** Its Table 5 prints −2.972% for real macro and for all three placebos. The
write-up's −2.715 / −3.084 / −2.890% (and −3.045 / −3.415 / −3.220% vs zero) appear in no output of either notebook.
The methodology section also says raw series are shifted "before feature engineering". The code shifts the transformed series.

**A working placebo in Gemini's own harness** (I rebuilt the interactions from shifted global *and* country macro):

| shift (months) | 36 | 48 | 60 | 72 | 84 | 96 | 108 | 120 | real |
|---|---|---|---|---|---|---|---|---|---|
| R²_OOS vs pooled mean | −1.640% | −1.751% | −2.021% | −3.547% | −0.818% | −2.435% | −3.273% | −2.020% | **−2.972%** |

Real macro ranks **7th of 9**, inside the placebo distribution, so the conclusion "macro adds nothing" survives. But the
size of the damage is not a finding. It is the hard-coded penalty. On a test-period sweep (a diagnostic only, not a
selection), the macro "gain" is −4.01 pp at α = 50, −3.00 at α = 200, −1.43 at 1,000, −0.28 at 10,000 and −0.04 at 100,000.
Gemini's M3 − M2 difference has a bootstrap SE of about 1.7%. "Macro conditioning destroys out-of-sample predictability"
and "the placebo proves macro is pure noise" are both overclaims. The honest statement is Claude's: with a tuned
penalty, macro changes R²_OOS by −0.123% (SE 0.085%), and real macro ranks 4th of 9 against the placebos.

### 5.5 Q3: does a forecast-based portfolio beat the simple rules after costs?

| 2011–2024, 10 bp | Gemini | Claude |
|---|---|---|
| Net Sharpe P1 / EW / RP / TSMOM | 0.426 / 0.281 / 0.302 / 0.123 | 0.44 / 0.28 / 0.30 / 0.12 |
| α on EW + RP + TSMOM | +0.52%/yr, t = 1.56 | +0.61%/yr, t = 1.74 |
| P1 from class means only | **0.441** (my check) | 0.44 |
| Weight correlation, P1 vs class means | 0.984 (my check) | 0.9994 |
| α with the class-means tilt as a regressor | **0.00%, t = 0.01** (my check) | t = 0.82 |
| Net Sharpe P1 − RP (bootstrap SE) | +0.12 (0.11) (my check) | +0.14 (0.11) |
| Share of gross in class D; asset_16 mean / max \|w\| | 59%; 24% / 42% (my check) | 60%; 22% / 42% |
| Stated answer | "Forecast-based allocations **decisively** beat simple heuristics … positive trend-following alpha" | "NO: the advantage is not significant and does not come from the characteristics" |

**Who is right: Claude.** Gemini's own pipeline, analysed as the exam asks ("how much of it is exposure to the benchmark
rules and how much is timing"), gives the same answer as mine. The forecast portfolio is:
- risk parity, plus a class-premium tilt re-estimated every year;
- concentrated in bonds and one very-low-volatility bond;
- matched by class means alone;
- not significantly better than risk parity.

"Decisively" and "positive trend-following alpha" (t = 1.56, p = 0.12) are not supported. Gemini also:
- does not report the concentration;
- quotes SEs, p-values and F = 508.2 that no cell prints (F recomputes to 507.9);
- measures turnover without price drift, so equal weight shows 0.000 turnover (a small effect).

### 5.6 The write-up

| | Gemini (cell 40, ~3,100 words) | Claude (cell 106, ~6,100 words) |
|---|---|---|
| Numbers printed by a cell | **More than 25 not printed.** The placebo results are fabricated. The class Sharpes (0.194, −0.023, 0.282, 0.405) contradict the printed Table 1 (0.412, −0.030, 0.327, 0.508), because the paper uses pooled asset-month statistics the notebook never prints. The attribution SEs, p-values and F are unprinted | Clean: every non-trivial number matched a printed output (about 90 hand-checked in context), and Table 3.31 prints the headline numbers |
| Rendering | Corrupted (§2.2) | Clean |
| Claims | Overclaims throughout ("proves", "decisively", "resilient", "Grader Defense Against Overfitting") | Calibrated, with SEs, power and Bonferroni. Weaknesses in §7 |
| Structure | All sections present; no figures referenced; no "what did not work"/next steps | All sections; every referenced table and figure exists |
| Readability | Short, but formulaic and partly unreadable | Long and dense, with internal process jargon |

### 5.7 Gemini original vs Gemini improved

**What the improvement fixed** (all correct fixes):
- `x10` with a 1-month lag in the macro models (a look-ahead trap) → `x86` with a 2-month lag;
- `x11` forward-filled → dropped;
- back-filled 2000–2002 rows in training → start 2003;
- pooled z-scores → within-class ranks;
- penalised single intercept → unpenalised class intercepts;
- raw macro levels → trailing z-scores × class;
- TSMOM sign(forecast) → sign(12m return), which was fixed in the original already.

**What the original also got wrong** (`review/checks/gemini_original_benchmark_check.out`):
- Its headline **+0.369%** is against a *per-asset* trailing mean. Against the pooled mean the same forecasts score
  **−0.085%**, and against zero **−0.035%**. The per-asset mean is itself 0.456% worse than the pooled mean, so the
  "positive predictability" was the weak benchmark.
- Its "robustness" α sweep (0.1 to 1,000) is computed on the test period.
- Elastic Net and "purging/embargoing" are not taught and are not labelled as the author's own additions.

**What the improvement broke or left:**
- it was never executed (§2.1);
- the placebo (§5.4);
- hard-coded penalties;
- the unfair forest comparison;
- no uncertainty anywhere;
- only one scheme;
- the Q1 and Q3 overclaims.

The vault (`final_project_rationale_and_verification_vault.md`) still reports the *original's* Problem 3 (Class A "Equities",
Class C "REITs", +0.369%, 180 months), and it says `RidgeCV` was added "in code". Neither notebook contains `RidgeCV`.

---

## 6. Gemini's supporting documents

**`asset_and_country_identities.md`** (checked in `review/checks/identity_claims_check.out`):
- **All 23 event returns it quotes match the panel exactly.** Examples: asset_24 −16.78% in 2008-10, asset_34 +7.94% in
  2015-01, asset_31 −8.18% in 2016-06, asset_25 −12.08% in 2022-09, asset_33 −54.65% in 2020-03.
- The country composition matches `asset_info.csv`.
- The trap facts hold: x10 constant in 300/300 country-years with correlation 0.9981 with the same-year x86 mean;
  corr(x11, x86 − x12) = 0.9882 with 99 gaps; the x5 floor at 0.004 on asset_16 for 43 months (2019-09 to 2023-03);
  x6 at 61.18 in 2008-10 and 57.74 in 2020-03; x8 at −2.42 in 2009-03.
- **The class table is pooled asset-month statistics.** For example, A has a vol of 30.97% and a Sharpe of 0.194. It is not the
  equal-weight class portfolio Gemini's notebook prints (14.61%, 0.412), and it doesn't say so. The write-up then quotes
  these numbers, which no cell prints.
- **"Independent historical benchmark" column:** no source is given for any of these figures (e.g. "S&P 500 TR − Rf −16.80%").
  They cannot be checked from the files, and matching a panel to itself proves nothing about identity. The identities are
  *consistent* with the data and with my own hypotheses (commodity-like, currencies with country 7 as base, equity
  indices, government bonds). They remain hypotheses.
- "x11 ≡ x86 − x12 … exact linear combination" is wrong: the correlation is 0.988, and only 0.87 for country 12.

**`methodology_and_decision_log.md`:**
- Its change register is accurate about what changed.
- Its compliance claims are not: "every number matches Cell 39 execution exactly" is false as submitted, and the placebo
  numbers match nothing.
- Its D-04 reason for annual refits ("runs in under 20 seconds") is fine.

**`final_project_rationale_and_verification_vault.md`:**
- The Problems 1–2 sections mirror the notebook.
- It repeats the 2.3 misquote as "Empirical measurements match theoretical formulas within Monte Carlo sampling error".
- Its Problem 3 section describes the original notebook, not the improved one.

---

## 7. Where I (Claude) was wrong or weaker, and why

Found by the blind reviewers and confirmed by me unless noted. Each has a note on whether it matters.

| # | where | issue | severity | rationale / does it matter? |
|---|---|---|---|---|
| C1 | 1.3 | "the forest tries one of the three features at each split, so a split can be forced onto a weak input" is **wrong** as an explanation. `max_features=None` gives the same 88.75%, and only 6% of splits are on Gender (I re-ran it) | Error | I reasoned from the setting rather than testing it. The main explanation in the same answer is right, and so is my printed evidence (99.8% training accuracy for deep trees). The correct secondary point is "the forest's trees are grown fully and fit noise"; a minimum leaf of 10 would lift it to 91.0% |
| C2 | 1.2, 1.3, 1.5, 2.3, 2.4 | Length beyond the exam's limits. 1.2 adds a labelled technical note. 1.3's first "why" sentence is very long. 2.3 and 2.4 contain the required paragraph but total ~840 and ~740 words | Weakness | A strict grader may deduct. R7 took 0.5 in each problem. The content is correct; trimming is your call |
| C3 | P3 write-up | ~6,100 words, with process jargon ("ledger", "development scripts", "version control", "AI guide §4a") | Weakness | The reviewers would cut about a third. It does not affect correctness (a possible next step in CLAUDE.md) |
| C4 | P3 write-up | "Pre-registered" in the title is stronger than can be shown: the design appears in version control only in a commit after the development ledger ran (disclosed in the text) | Weakness | Fair criticism. "Design fixed before evaluation (see §3.0)" would be safer wording |
| C5 | P3 write-up §5 | KNN and neural nets are excluded "under the scope rule" although the exam explicitly allows any method | Weakness | The exclusion is *your* standing rule, and it is legitimate. The wording should say it is our choice of scope, not something the exam imposes |
| C6 | P3 conclusion | "Adding macro lowers R² in every paired comparison" is true of the 14 pre-listed pairs, but the post-hoc interaction spec has +0.012% | Style | The abstract has the qualifier; the conclusion needs it too |
| C7 | P3 inference | Month bootstrap assumes independent months; block SEs are about 1.19× larger | Weakness (disclosed) | Verdicts are already "not significant", so larger SEs only strengthen them |
| C8 | P3 placebo | Shifts of ≥72 months wrap end-of-sample macro into early test months (1–49 months) | Weakness (disclosed) | Doesn't change the rank or the verdict |
| C9 | P3 by-class table | Table 3.11 uses only the pooled mean, so class figures are mostly intercept effects | Weakness | Read correctly in the text; a by-class table against each class's own mean would be cleaner |
| C10 | P3 costs | The first test month's build cost is not charged (to every strategy, disclosed); the design table says "one-way turnover" but the code charges Σ\|Δw\| (conservative, clarified in the appendix) | Style | Uniform across strategies; no ranking changes |
| C11 | P3 data | Descriptives use 2000–2024 including the test period (disclosed; a training-only correlation table is also shown; no design choice depends on it) | Style | Acceptable as description |
| C12 | P3 features | The exam's optional cross-country macro transforms and widened panel are not tried (Gemini does not try them either) | Weakness | Optional ("consider") |
| C13 | 1.1 | Small derived numbers not printed ("1.25 pp of a fold, 0.25 pp pooled") | Style | Arithmetic from printed numbers, but strictly exam cell 1 applies |
| C14 | 2.1 | The prediction is qualitative (no magnitude) | Style | Your own committed text; not to be edited, and 2.3 reconciles it |

**Where my conclusions could be challenged, and why I keep them.** On every research question, the two independent
pipelines agree on the numbers. They differ only on interpretation, and in each case the evidence (Gemini's own forecasts
with SEs and the class-means decomposition) supports my interpretation. I found no place where Gemini's *answer* is right
and mine wrong. Gemini's simpler design is not worse in principle. It is worse only where it skipped tuning,
uncertainty and the decomposition the exam asks for.

---

## 8. The answers I believe are correct

- **P1.** 64.25% baseline; unpruned 85.00%; 3 leaves (90.75%, tied with 4, which the tie rule settles); forest 88.75% and
  boosting 89.00% do not beat it, because the pattern is a two-split rectangle and deep default trees fit noise; MDI favours
  Salary, held-out permutation favours Age, and show permutation; shuffled CV on time series leaks the future upward, so use
  walk-forward.
- **P2.** VaR collapses. Drift ≈ −1/(n−1) per night. The median desk ends near 0.006–0.007 σ̂² and the story is roughly the
  median. The expectation is 1 only through rare exploding desks. Anchoring on real days stops the noise. The normal
  model loses the tails on night 1. Only real data in every generation stops the collapse.
- **P3 Q1.** Not detectably forecastable. R²_OOS ≈ 0.0–0.1% against the pooled mean (SE ≈ 0.24%), negative against
  zero, and characteristics add nothing beyond class intercepts. The design can only detect an R²_OOS of about 0.5%, so the answer is
  "not detectable", not "zero".
- **P3 Q2.** Macro adds nothing. With a tuned penalty the change is −0.12% (SE 0.085%), and real macro sits inside the
  placebo distribution. Lightly penalised macro models lose heavily.
- **P3 Q3.** No forecast-driven outperformance is demonstrated. The point estimate (net Sharpe ≈ 0.43–0.44 against
  RP 0.30) is about 1.1–1.3 SEs. It comes from a class-premium tilt concentrated in bonds (reproduced by class means alone), and
  the characteristic bet loses after costs.

---

## 9. The identity decoding and a 2025–26 test

- **What it gives you.** Knowing the classes are (almost certainly) commodities, G10 FX against the dollar, developed
  equity indices and 10-year government bonds makes the results easy to *interpret*. The P1 portfolio is a
  bond-heavy risk-parity tilt; asset_16 behaves like a yield-capped bond (the x5 floor during 2019-09 to 2023-03); and
  the characteristics look like momentum, volatility, carry, value/reversal and change in carry. My notebook already
  states these as hypotheses (Section 2 of the write-up), which is the right register.
- **What it must not do.** Drive design choices. That would be information from outside the sample, and for the test period
  it is hindsight. Nor should names be stated as facts in the paper. The "independent benchmark" figures are unsourced.
- **Testing on real 2025–26 data.** Useful only as a *post-hoc sanity check*, labelled as such:
  - vendor series (futures rolls, total-return vs price indices, excess-return conventions) will not match the panel's
    construction exactly;
  - the characteristics' exact definitions are only partly known (x2 and x5 exactly; x1, x3 and x4 approximately);
  - 12–24 new months cannot detect an R²_OOS of 0.5%, since the power analysis needs far more;
  - if your professor tests on a modified or extended panel, the pipeline should simply be re-run on it. Both pipelines
    read the CSVs and have no hard-coded dates beyond the design windows.

  If you want this done, I'd propose it as a separate, post-hoc analysis in its own section, for your approval first.

---

## 10. Evidence index

| claim | evidence |
|---|---|
| Gemini files, unmodified | `gemini/` (commit `8e83f00`) |
| Re-runs of both Gemini notebooks | `review/runs/*__rerun.ipynb`, `review/runs/run_nb.py` |
| Improved notebook's cell-39 outputs identical to the original's | byte comparison, reproduced by `review/runs/run_nb.py` + a cell diff (§2.1) |
| Placebo bug, working placebo, class-means decomposition, bootstrap SEs, forest with class dummies, coefficient paths never shown | `review/checks/gemini_p3_checks.py` / `.out` (C1–C6) |
| Extended-file PCA 97.0% → 43.1% standardised; macro loss vs penalty | `review/checks/gemini_p3_extra_checks.py` / `.out` |
| Original's +0.369% vs per-asset mean = −0.085% vs pooled; x10 lag-1 in macro models | `review/checks/gemini_original_benchmark_check.py` / `.out` |
| Identity document vs data | `review/checks/identity_claims_check.py` / `.out` |
| Blind reviews (anonymised, A = Gemini improved, B = Claude) | `review/blind_reviews/R1…R7*.md` and their `*_work/` scripts; key in `review/blind_key.md` |
| My notebook's numbers | `Final_Autumn_2026-1.ipynb` on `financial_analytics_opus`, Table 3.31 (cell 104) |
| Tool used to render notebooks as text | `review/tools/nb2text.py` |
