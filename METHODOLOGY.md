# BUSN 41210 Financial Analytics — Final Exam, Autumn 2026: Methodology

**What this document is.** A complete account of what was done in the submitted notebook, `Final_Autumn_2026-1.ipynb`, and why. For every problem it covers:
- the method used and the lecture material it rests on;
- the alternatives that were considered;
- why each was used or set aside.

Every result quoted here is printed by a cell of the notebook, and Problem 3's headline numbers are collected in its Table 3.31; design constants (windows, grids, seeds, settings) are those set in its code. Lecture citations are PDF page numbers (e.g. "L5 p.57"). In Lectures 1, 5 and 8 the number printed on a slide can be lower, because of overlays.

**Contents.**
0. The rules the whole notebook follows
1. Problem 1: trees and ensembles
2. Problem 2: training on your own output
3. Problem 3: cross-country asset return prediction
4. The answers at a glance
5. How the notebook is organised and how to run it

---

## 0. The rules the whole notebook follows

### 0.1 Binding rules

| rule | what it means in practice |
|---|---|
| **The course's AI Coding Guide is binding** | No shuffled split or fold on time-ordered data (§4a) and no scaler, imputer, PCA or other transformation fitted before the split (§4b); the guide grades both as fatal. `LogisticRegression()` is L2-penalised by default (§4c). `r2_score` scores against the test-set mean, so it is not the course's out-of-sample R² (§4d) |
| **Scope is the eight lectures** | Every model, formula and test is one the lectures teach. A method a lecture only names (neural networks, SVM, elastic net, HAC standard errors, t-SNE, Benjamini–Hochberg) is treated as not taught. Anything we built from taught parts is labelled *our construction* or *our choice*; quantities the exam defines (VaR, kurtosis, the benchmark portfolios, the placebo) are implemented exactly as the exam defines them |
| **Every number comes from a cell** | The exam's opening instruction. Every number in an answer or in the write-up is printed by a code cell, and Problem 3 ends with a table (Table 3.31) that prints every number the write-up quotes |
| **Honest reporting** | Results are reported whichever way they come out, and a negative result is a result. The benchmark is fixed before any result is seen and never switched. The size of the search sits next to every headline. Anything added after the out-of-sample results were seen is labelled *post hoc* |
| **Exam limits are literal** | 1.2 in one or two sentences, 1.3 in two or three, 2.1 in two or three, and 2.3 and 2.4 in one paragraph each. Numbers the questions ask for are reported in tables beside the paragraph |

### 0.2 AI Coding Guide compliance

| AI Coding Guide item | how the notebook satisfies it |
|---|---|
| §4a no shuffled split on time-ordered data | Problem 3 has one splitter: a date mask that only moves forward, with `train < validation < test` asserted at every fold and refit. No `train_test_split`, `KFold`, `*CV` estimator or `GridSearchCV` touches Problem 3. Problem 1 uses shuffled stratified folds because the exam prescribes them and its 400 rows are unrelated people, so the rows are exchangeable (L5 p.28; L8 p.45). Answer 1.5 explains why the same folds would be wrong for time series |
| §4b nothing fitted before the split | Characteristics are ranked within each month, which uses no information from other months. Macro series are standardised with trailing 60-month windows. Every `StandardScaler` and PCA is fitted on the training rows of each window and of each validation fold. A truncation test rebuilds all features from data cut at 2010-12 and finds all 64 columns identical up to that date |
| §4c `LogisticRegression` penalised by default | Not used anywhere (no classification problem arises) |
| §4d `r2_score` is not R²_OOS | Never used. R²_OOS = 1 − SSE(model)/SSE(benchmark), with the benchmark a trailing mean through t−1, updated monthly, and zero as a second benchmark |
| §3 pandas 3 / sklearn traps | No `append`, `fillna(method=)`, `iteritems`, `normalize=`; `sns.set()` is never called. `ddof=1` is passed explicitly. `scoring` is always set explicitly, never left at `None`. `RandomForestRegressor` is given `max_features` explicitly (its default is bagging). `GradientBoosting*(n_iter_no_change=…)`, which uses a shuffled internal split, is not used |
| §6.1 describe the data before modelling | Each problem opens with shapes, counts and dtypes, asserted where the answer is known |
| §6.4 check against something known | In-sample fit exceeds out-of-sample fit for every Problem 3 specification (38 of 38, printed). The vectorised pipeline of Problem 2 reproduces the literal one-desk loop exactly (asserted). Unit tests check the harness against sklearn on real training rows |
| §7 say what was tried and rejected | §1–§3 below, and Section 5 of the write-up ("Considered and not used") |

### 0.3 Fixed conventions
- **Seeds.** Problem 1 passes `random_state=7034` to every split and model, as the exam requires. Problem 2 sets `SEED = 2694` (the last four digits of the student ID) and draws from named, independent streams (`SeedSequence(SEED).spawn`), so re-running any cell reproduces its numbers and adding a stream never changes the others. Problem 3 uses `P3_SEED = 7034` and never rebinds `SEED`.
- **Variance.** Always `ddof=1` (L2 p.56–57), the estimator the Problem 2 manual specifies.
- **Data path.** Each setup cell keeps the exam's server path and falls back to the notebook's own folder, so the notebook runs on the class server or next to a local copy of the data.

---

## 1. Problem 1: trees and ensembles

**Setting.** 400 people, three inputs (Gender, Age, EstimatedSalary), a binary outcome. Every accuracy is 5-fold stratified cross-validation with the exam's `cv5` (`random_state=7034`). All models are scored on the same five folds, so they can be compared fold by fold.

### 1.1 Baseline, unpruned tree and the size sweep
- **Baseline.** "Nobody purchases" is right for every non-buyer: 257 of 400 = **64.25%** (63.75% to 65.00% by fold, because 257 cannot be split evenly into five stratified folds). It is the bar every model must beat (L6 p.57: report accuracy "with the majority-class rate next to it").
- **Unpruned tree: 85.00%** (340/400). Grown on all 400 people it has 61 leaves and depth 14 and gets 399 of 400 right in sample; the one miss is a pair of identical people with opposite outcomes. The gap between 399/400 in sample and 85% out of sample is memorised noise (L8 p.30, p.38).
- **Sweep of `max_leaf_nodes` over {2, 3, 4, 6, 8, 12}:** 83.50%, **90.75%**, **90.75%**, 89.50%, 88.25%, 87.50%.
- **Choice: 3 leaves.** 3 and 4 leaves tie exactly (363/400 correct), and the exam's rule gives a tie to the smaller tree. The tie is structural: the fourth split divides a leaf into two leaves with the same predicted class, so the two trees make identical out-of-fold predictions for all 400 people (checked in code).
- **Either side of the optimum.**
  - With 2 leaves, CV and training-fold accuracy are both low (83.50% and 84.0%): the model is too simple.
  - From 6 to 12 leaves, CV accuracy falls while training-fold accuracy rises (92.6% to 94.4%): the trees are overfitting (the hump of L8 p.32).
- **How much to read into the gaps.** The comparison is fold by fold on shared folds: the 3-leaf tree beats the 6-leaf tree in 4 of 5 folds, by 1 to 3 people each. The 90.75% also won a search over six sizes, so it is slightly optimistic as an estimate of future accuracy (L4 p.43; L5 p.57).

**Alternatives set aside.**
- **A one-standard-error rule, or any "within noise" tolerance.** Rejected because the exam's tie rule is literal, and a one-standard-error rule is not taught as a method.
- **A paired per-person test for the model differences.** This is McNemar's test, which is not taught. The fold spread, and the count of folds won or lost, are what the course uses (L8 p.45).
- **`cv=5` as an integer.** It gives unshuffled folds, which are different folds from the exam's.

### 1.2 The chosen tree in words
- The tree is refit on all 400 people at the size cross-validation chose, because CV chooses a size, not a particular tree (L5 p.36, as in L8 p.33). It is L8 p.33's tree, split by split: Age ≤ 42.5, then EstimatedSalary ≤ 90,500.
- It is stable. The five trees fitted on the CV training folds all split on Age and then Salary, with age thresholds between 42.5 and 44.5.
- Its honest accuracy is the CV figure, 90.75%, not the 91.5% it scores on the people it was fitted to.
- **The answer** (two sentences, as the exam asks): people over 42 mostly bought (97 of 115), as did people aged 42 or younger earning more than \$90,000 (37 of 44). Younger people earning \$90,000 or less almost never bought (9 of 241). Gender plays no part.
- The thresholds are translated for a lay reader: sklearn thresholds are midpoints, ages are whole years, and salaries are multiples of \$1,000, so ≤ 42.5 becomes "42 or younger". Both buyer regions are hedged ("mostly", "almost never") because no region is pure.

### 1.3 Random forest and boosting
- **Results.** Random forest (300 trees) **88.75%** (355/400) and gradient boosting (100 rounds) **89.00%** (356/400): **neither beats the 3-leaf tree** (363/400). Both are far above the baseline and the unpruned tree.
- **Fold by fold.** The forest beats the tree in 2 folds and loses in 3; boosting beats it in 1, ties 2 and loses 2. Most of the shortfall is one fold, where the tree gets 74 people right against 68 and 67. The fair reading is "no better than the small tree", not "worse".
- **Why.** The buying pattern is close to one rectangle in the Age × Salary plane, which the 3-leaf tree already draws. So there is little variance for a forest to average away and little bias for boosting to remove (L8 p.51: bagging and boosting attack opposite errors). At their untuned defaults both ensembles fit noise instead:
  - the forest's trees are grown until their leaves are pure (50.2 leaves per tree on average) and score 99.81% on their own training folds;
  - boosting (learning rate 0.1, depth 3) scores 97.44% on its training folds, against 91.63% for the small tree.

  On the lecture's much larger housing data, averaging beats one tree (L8 p.54); on 400 people and one rectangle it does not.
- **The explanation rests on the printed training-fold accuracies.** The forest's default of trying one of the three features at each split is not offered as a reason for its shortfall; the evidence the notebook prints is that its fully grown trees fit the training folds almost perfectly.
- **Tuning the ensembles** was not done. The exam fixes 300 trees and 100 rounds with default settings, and the question is whether the defaults beat the tree.

### 1.4 Two importance rankings
- **Setup.** A 70/30 stratified split (`random_state=7034`): 280 people for training (100 buyers) and 120 held out (43 buyers). A 300-tree forest is fitted on the 280.
  - MDI (`feature_importances_`) is computed in sample, as the exam asks.
  - Permutation importance uses the held-out 30%, `scoring='accuracy'`, and **50 repeats**. The default of 5 is too few for a 120-person test set, where one person is a visible step in accuracy.
- **Result.** The rankings disagree on the top variable:
  - MDI puts Salary first (0.519 against 0.470 for Age);
  - held-out permutation puts Age first (25.5 against 17.1 points of accuracy lost);
  - both put Gender last.
- **Why they disagree** (L8 p.47, p.56–58):
  - MDI adds up in-sample impurity reductions. These trees are grown deep, so many splits fit noise, and each such split is credited to some variable.
  - Salary offers the most places to split: 108 distinct values on the training rows against 42 for Age, and the forest splits on it 5,623 times against 4,751.
  - Permutation importance on the *training* rows also ranks Salary first (28.8 against 27.0 points), consistent with that mechanism.
  - The MDI margin is small, so the exact order could flip on another split.
  - Correlation between the inputs is not the main story: Age and Salary correlate only 0.155.
- **What to show a decision maker: the held-out permutation ranking.** It measures how much accuracy on new people is lost when a variable's information is destroyed, which is the question being asked (L8 p.57). The answer gives three caveats:
  - it rests on 120 people;
  - importance has no sign (L8 p.60);
  - it describes the model, not a causal effect.

### 1.5 Why shuffled cross-validation is wrong for time series
- **What goes wrong.** Shuffled K-fold assumes exchangeable rows. With months of one stock:
  - the future enters the training folds (L5 p.56; L1 p.82);
  - neighbouring months are near-duplicates, because trailing-window predictors barely change and regimes persist (our reasoning; the course's rule is that "the order of data is respected", L5 p.27);
  - the model is then tuned to the leak (L5 p.56 measures this).
- **Two tempting fixes do not work.**
  - Dropping `shuffle` still trains early folds on later data.
  - Stratifying uses the full-period share of up months, which is only known at the end.
- **Direction of the error.** Reported accuracy is biased upward: every leak adds information a real forecaster lacks. The course grades this as fatal (L5 p.58; AI Coding Guide §4a).
- **What to do instead.** Walk-forward evaluation (L5 p.48–52; L6 p.30):
  - build predictors from information known at the end of month t−1;
  - fit any scaler inside each training window;
  - choose the tree size on the following validation block, predict the next block, and roll forward;
  - keep a final test period untouched and touch it once (L4 p.43; L8 p.55);
  - compare with a majority-direction benchmark learned from the training months only (L6 p.57).
- The answer is conceptual and needs no code (the exam asks for code "if any"). A purging gap is unnecessary for a one-month target.

---

## 2. Problem 2: training on your own output

**Setting.** A desk refits σ̂² each night on 500 scenarios drawn from its own previous fit. The work is in units of the true volatility (σ = 1, true VaR 2.326), with `SEED = 2694`.

### Implementation choices that apply to all parts
- **Night indexing.** Night 1 fits the real library, whose sample variance is exactly 1 (the exam's rule). Each later night fits the previous night's 500 scenarios, so night 2,500 is 2,499 draw-and-refit steps after night 1.
- **One implementation.** A vectorised `run_desks` function simulates many desks at once. It is used in 2.2 and 2.3 and is asserted to reproduce the literal one-desk loop exactly. Scenarios are drawn as `sd × standard_normal`, because the standard deviation, not the variance, is the scale.
- **Named random streams** (§0.3). The one desk is independent of the 1,000 desks, and each experiment has its own stream. The global `np.random.seed` was set aside: it makes results depend on the order in which cells run.
- **Monte-Carlo uncertainty.** A fraction's SE is the sample SD of the 0/1 indicator over √M, and a mean's CI uses the t distribution (L2 p.63, p.81–86). A result that does not reject is reported as "consistent with", never "equal to" (L2 p.91, p.93).

### 2.1 The prediction (written before any code)
The answer is the student's own prediction, written before any Problem 2 code, as the exam's "commit before you compute" requires. It predicts:
- that the average across desks stays near 1;
- that by Jensen's inequality the median and a single desk drift down at a rate proportional to −1/(n−1);
- that the result is a collapse of the reported VaR.

2.3 reconciles each claim with the measurements, as the exam asks.

### 2.2 One desk, then a thousand
- **One desk:** σ̂² on night 2,500 = **1.405 × 10⁻⁵**, reported VaR **0.0087σ** against 2.326σ. That is the 3.2nd percentile of the 1,000 desks: an unlucky desk, not a typical one.
- **1,000 desks, night 2,500:**
  - σ̂²: mean 2.344, median 0.00585, 5th percentile 0.000029, 95th 1.225, maximum 1,566;
  - **57.8%** (MC SE 1.6%) report a VaR below one-tenth of the truth.
- **The mean VaR** is the average VaR across desks, 0.775σ. It is not the VaR evaluated at the mean σ̂² (3.56σ), which a few huge desks dominate. Both are printed so the difference is visible.
- **Is the story typical? Yes.** A fall from 2.8% to 0.24% of NAV means σ̂² = (0.24/2.8)² = 0.0073, which sits at the 52.7th percentile of the desks (51.5% to 53.3% allowing for rounding in the story's numbers). The median desk reports 0.214% of NAV.

### 2.3 Unbiased every night, wrong in the end
- **The one-step property.** 200,000 desks start at the same σ̂²_t and take one step. The reporting rule, fixed before running, was "consistent with 1 if the 95% CI contains 1".
  - At σ̂²_t = 1 the mean ratio is 0.99963 (CI 0.99935–0.99991), a narrow miss (p = 0.0088).
  - At σ̂²_t = 1.405 × 10⁻⁵ it is 1.00009, which passes.
  - A miss this large at one of two starts has a 1.7% chance if the property holds exactly. The draws were not repeated to make it pass. Unbiasedness of the `ddof=1` variance is exact (L2 p.56), so this is a rare Monte-Carlo miss.
- **The drift.** The exam's design is used: 500 desks × 500 nights for n = 50, 500 and 5,000, with SEs across desks.
  - n × (mean nightly change in log σ̂²) is −0.99, −1.00 and −0.87 (SE 0.20).
  - A precise n = 50 run (2,000 × 2,000), planned in advance, gives −1.025 (SE 0.005). That is 4.9 SE from −1/n but 0.9 SE from −1/(n−1).
  - **Formula: −1/(n−1), which is approximately −1/n for any realistic n.**
  - The n = 500 value is cross-checked on the 2.2 run's increments (−0.002059 per night).
- **The three numbers:**
  - the expectation of σ̂² on night 2,500 is exactly 1;
  - the median is 0.0059;
  - the 1,000-desk mean is 2.34. The 10 largest desks hold 88.8% of the sum, the other 999 average 0.78, and only 5.9% of desks end above 1.
- **The reconciliation.**
  - Each night multiplies σ̂² by a factor with mean exactly 1 but median below 1. On the log scale this is a random walk with drift ≈ −1/n.
  - The median therefore falls along a straight line: exp(−2,499 × 0.00206) = 0.0058 against the measured 0.0059.
  - The spread grows like √nights (predicted sd 3.17, observed 3.29).
  - The expectation stays at 1 only because a vanishing minority of desks explodes, so a sample of 1,000 is neither 1 nor the median.
  - Unbiasedness controls only the mean of tomorrow's estimate, not its typical value (L2 p.71–72). Each night's error becomes the next night's truth, and after night 1 no new information about σ enters the library (L2 p.57).
- **A supplementary cell, labelled as beyond the lectures,** checks the formula against theory: the χ² law of s² and the digamma function, which are not taught. It only corroborates; the formula rests on the measurements.

### 2.4 What synthetic data can and cannot carry
- **(a) Keeping the real days.**
  - Each desk keeps its own 500 N(0, 1) draws, rescaled to sample variance exactly 1 (the exam's rule) but not demeaned. The library is those 500 plus tonight's 500 scenarios.
  - Night 2,500: median **0.9955**, 5th–95th percentile **0.9406–1.059**, and no desk below a tenth of the truth. The band is in place by night 3 and stays there.
  - Demeaning the real days as well (a robustness row on the same draws) changes nothing that matters.
  - One shared real sample for all desks was set aside, because the desks would not be independent.
- **(b) Seeding with real returns.** The 1,511 daily market returns of `dj30.csv` (2016–2021).
  - Volatility **1.195%** a day (`std(ddof=1)`). Excess kurtosis **24.1**, using `scipy.stats.kurtosis` with its defaults, the function the exam imports; 24.2 bias-corrected.
  - The night-1 library is all 1,511 days, so 2.326 × sd = **2.78%** reproduces the story's 2.8%.
  - The empirical 1% VaR is **3.33%** (3.17% to 3.34% under other percentile rules). The normal model understates the 1-in-100 loss by 0.54 points. Resampling the days gives that gap an SE of 0.32 points (1.7 SE), with the caveat that i.i.d. resampling ignores volatility clustering (our caveat; L2 p.80 describes the resampling).
  - The stronger evidence: losses beyond the pipeline's VaR occur on 1.65% of days rather than 1%. The 500 night-1 scenarios have excess kurtosis −0.22, well inside the range of normal samples of 500 (95% of 10,000 simulated samples fall between −0.38 and 0.46). The real data's 24.1 is far above anything in that simulation.
- **Classification.**
  - **(b) shows a loss of information.** The first synthetic sample has already lost the tails, before any feedback, and every later generation is normal.
  - **(a) shows an accumulation of noise.** The model is right and the information sits in the real days; the synthetic half adds only estimation noise, which the real half contains in a stable band. Without the anchor (2.2) the same noise compounds into collapse.
- **What stops it:** real data, information from outside the model, kept in the training set at every generation. In (a), even the same 500 original days suffice. It is not phrased as "fresh data": the anchor in (a) is not fresh, and the market never changes in this exercise. Getting the tails back in (b) needs a better model, not more of its own output.

---

## 3. Problem 3: cross-country asset return prediction

**Setting.**
- 50 assets in four classes and 12 countries, monthly, 2000–2024.
- Five characteristics (x1–x5), four global macro series (x6–x9), a curated country file (x10–x16) and an uncurated extended file (x17–x164).
- Three questions:
  - **Q1** how much is forecastable from lagged characteristics;
  - **Q2** whether macro adds anything;
  - **Q3** whether a forecast-based portfolio beats equal weight (EW), risk parity (RP) and time-series momentum (TSMOM) after costs.

**Order of work.** The design (Section 3.0 of the notebook) was written before any model was fitted: windows, target, benchmarks, models, tuning, the list of candidate specifications, and the rule that decides each question. Every later departure is listed in Section 3.10, and every analysis added after the out-of-sample results were seen is labelled post hoc. The notebook cannot itself prove the order in which things were written, which is why these lists exist.

### 3.1 Data: traps found and how each was treated

Each item is recomputed and printed in Section 3.1 of the notebook. The rule for every treatment: use only information available at the time.

| issue | evidence (printed) | treatment | why this is legitimate |
|---|---|---|---|
| **`x10` is a same-year average** of the monthly short rate, stamped from January | constant within each calendar year; correlation 0.998 with the *same-year* mean of `x86`, against 0.839 with the prior-year mean | **Not used.** The monthly short rate `x86` from the extended file is used instead, at the same 2-month lag as every macro series | A one-month lag of `x10` would leave up to 11 months of look-ahead. The fix is to use the monthly series at its release lag (L5 p.58: merge on the publication date) |
| **`x11` stops early** for countries 12, 9 and 11 (2020-10, 2021-08, 2024-03) | no interior holes. Two of the three stops (countries 12 and 9) fall just before or during the 2021–23 inflation shock, where a stale forward fill would be wrong by up to 10.1 percentage points | **Not used.** Rebuilt as `x86 − x12` to measure the redundancy (pooled correlation 0.988, RMSE 0.354; country 12 weakest) | Its gaps, and its near-redundancy with `x86` and `x12`, which are both in the model. Forward-filling was rejected (regime change). A fitted rebuild would itself be a transformation fitted on data |
| **Back-filled leading values** | 8 (variable, asset) pairs in 7 assets carry copies of their first genuine value; the last is dated 2002-07 | Set to missing; the first target month is 2003-01, so none enters any design matrix (asserted) | A backward fill carries the future into the past (the exam's workflow) |
| **Duplicates and hidden global series in the extended file** | 6 exact copies of curated series (e.g. `x73` = `x10`); 26 columns identical across all 12 countries, 4 of them copies of the global file | The extended file is never concatenated or screened; only `x86` is taken from it | Exact collinearity (L3 p.40); the exam's own warning to find the duplicates |
| **Low-frequency columns of unknown timing** | 35 annual (changing only in January) and 26 quarterly columns | Not used | Their release dates are unknown, so any lag would be a guess (L5 p.58) |
| **`x1` frozen** for five class-B assets at the end of 2024 | 10 target rows use a stale `x1` | Kept as stored (a desk would have seen them); an evaluation-only cut excludes them (0.042% → 0.026%) | A stale value is not look-ahead |
| **`x5` floored at 0.004** for asset_16 (2019-09 to 2023-03) | its own 36-month SD falls to 0.00298 there | σ̂ is computed from returns, never from `x5`; `x5`'s within-class rank is unaffected | Both quantities are trailing |
| **Scales differ enormously** | monthly volatility from 0.82% (asset_16) to 17.40%; `x1` is in percent for one class and a monthly decimal for another | Within-class ranks for characteristics; a volatility-scaled target | Penalties depend on scale (L5 p.23); constant variance (L3 p.2–3) |

**What the characteristics are** (inferred from past returns and macro only, never from the target):
- `x2` is the compounded return over months t−12..t−1 (99.63% of asset-months), i.e. 12-month momentum.
- `x5` is the 36-month volatility (99.67%), the risk measure.
- For the currencies, `x1` behaves like carry (correlation 0.93 with the short-rate differential against country 7) and `x4` like its 12-month change (1.00).
- `x3` behaves like value or five-year reversal (correlation −0.93 to −0.98 with the 60-month past return in classes A, B and D).

Two implications were stated before modelling:
- `x2` and `x5` are exactly the inputs of the TSMOM and RP benchmarks;
- for the currencies, `x1` and `x4` already contain country interest-rate information.

**What the classes look like** (hypotheses, used only for interpretation): A is commodity-like (no country; a 54.7% loss in 2020-03), B currencies (country 7, which has no B asset, is the base), C equity indices, D government bonds (negatively correlated with A and C). Three classes are tight blocks (within-class correlations 0.55, 0.73 and 0.58 for B, C and D); A is not (0.21, below its correlation with B).

### 3.2 Design decisions

| decision | what was chosen | alternatives and why they were set aside | basis |
|---|---|---|---|
| **Windows** | First target month 2003-01 (the 36-month volatility first exists then, and every back-fill is out). Initial training block 2003-01 to 2010-12 (96 months, 4,800 asset-months, including the 2008 crisis). Out-of-sample window 2011-01 to 2024-12 (168 months, 8,400 asset-months), touched once, with halves 2011–17 and 2018–24 | A 2010 start would add 12 test months but cut the training block to 84 months and the thinnest class's rows sharply; a 2013 start wastes test months. Fixed in advance, as the exam's second rule requires | The exam's "Two rules"; L4 p.43; L5 p.48–52 |
| **Target** | $y_{i,t} = r_{i,t}/\hat\sigma_{i,t-1}$, with σ̂ the 36-month SD (`ddof=1`) of the asset's own past excess returns, computed by us. Forecasts become positions as w ∝ ŷ/σ̂ | Raw returns: a pooled squared-error fit would be dominated by the most volatile class. `x5` as σ̂: it carries the asset_16 floor. A shorter σ̂ window: noisier, and it changes the target itself | The exam recommends the vol-scaled target; constant variance (L3 p.2–3). The scaling equalises classes (sd of y 1.05–1.09 across classes) but not years (0.76 in 2017 to 1.74 in 2008) |
| **Forecast benchmark** | The **pooled trailing mean** of y through t−1, updated monthly (primary). Zero, the per-class and the per-asset trailing means are always reported too | The per-asset mean as primary: noisier and easier to beat (it scores −0.27% against the pooled mean). Fixing one primary before any result prevents benchmark switching (L5 p.58) | The exam's second rule; L4 p.45–46; L5 p.55–57 |
| **Characteristic standardisation** | Rank within class and month, scaled to [−0.5, 0.5]. The within-class z-score is a pre-listed variant | Ranks across all 50 assets would mostly sort the classes, which the class intercepts already carry. A z-score as primary is sensitive to extremes (the 1st percentile of `x3` in class A is −4.13) | The exam's workflow; L5 p.23; L3 p.11–12. Ranks have no parameter estimated across time, so they cannot leak |
| **Class intercepts** | Ridge and lasso penalise only the characteristic and macro slopes. Implemented by demeaning y and X within class on the fit rows and adding the class means back, which equals unpenalised class dummies (unit-tested) | Penalised dummies would shrink the class means with the same penalty, confusing "characteristics add beyond class premia" with dummy shrinkage | The ridge objective penalises the slopes it contains (L5 p.12); common slopes with class intercepts (L3 p.50). Keeping the intercepts out of the penalty is our choice |
| **Global macro** | log `x6`, `x7`, `x8`, `x9`, each as a trailing 60-month z-score (at least 24 observations), lagged 2 months, × class (16 columns) | A full-sample z-score is forbidden: it uses the future | The exam's workflow ("trailing information only"); logs for skewed positive series (L3 p.48–49); interactions (L3 p.53–54) |
| **Country macro** | `x86`, `x12`, `x13`, log `x14`, log `x15`, `x16`. Each as the differential against country 7, then a trailing 60-month z-score per country, then lagged 2 months, × class for B, C and D (18 columns). Class A gets none | Raw differentials: static country differences would pose as "macro" (country means explain much of their variance). For class A: country 7's macro (identically zero as a differential), a cross-country average, or none. None was chosen because A has no natural country, and for parsimony (L4 p.23) | The exam's workflow (country 7 is "the natural base") |
| **Macro lag** | 2 months for every macro series; 1 month as a pre-listed variant | Series-specific lags: the names are anonymised, so market series cannot be told apart from statistical releases; one conservative rule is stated | The exam: "a one-month lag is the minimum, and you may argue for more" |
| **Feature sets** | M1 = class intercepts only; M2 = M1 + the five characteristic ranks; M3 = M2 + 34 macro columns (Table 3.7) | — | — |

### 3.3 The evaluation harness
- **Schemes** (L5 p.48–52):
  - *static*: fit once at 2010-12, as in the exam's suggested workflow;
  - ***expanding* (primary)**: a fixed start, refit every December, so 14 refits;
  - *rolling*: the 96 months ending each December.

  Monthly refits were set aside: twelve times the computation, for premia that move slowly.
- **Tuning, inside each training window only.**
  - Three annual validation folds: fit to year b−j, validate on year b−j+1, for j = 3, 2, 1.
  - The lowest mean validation MSE wins, and a tie goes to the more heavily penalised model (L5 p.25). The model is then refit on the whole window.
  - The ridge penalty is parametrised as α/n on the summed squared error, so the validated shrinkage is the shrinkage applied at refit.
  - Grids: ridge on 36 log-spaced values from 10³ to 10⁻⁴; lasso on 33 values from 10^−0.5 to 10^−4.5; PCR over K ∈ {1, 2, 3, 4, 5, 6, 8, 10}.
  - The top of the ridge and lasso grids is the class-means model ("no signal"), so an edge choice has a meaning, and edge choices are logged.
  - `RidgeCV`, `LassoCV(cv=int)` and `GridSearchCV` were set aside. They use row-wise or random folds, which are wrong on time-ordered data (AI Coding Guide §4a).
- **One splitter.** A date mask that only moves forward, with order asserted at every fold and refit. Unit tests on real training rows check that:
  - demeaning equals class dummies;
  - the ridge and PCR paths match sklearn;
  - lasso converges;
  - the scorer and splitter behave.
- **R²_OOS** = 1 − SSE(model)/SSE(benchmark), pooled over the 8,400 test asset-months, against the pooled trailing mean and against zero.

### 3.4 Models, and the list of candidate specifications
- **OLS**, as the unshrunk reference.
- **Ridge (primary)** (L5 p.12–16, p.60: "suitable when features are weak and should still contribute").
- **Lasso** (L5 p.17–24).
- **PCR**, i.e. PCA (L1 p.45–74) followed by OLS on the first K components. Combining the two taught tools is our construction. The inputs are standardised first (correlation-matrix PCA, our choice), because their columns differ in scale.
- **Random forest**: 300 trees, a third of the inputs per split, minimum leaf 200 observations, untuned (L8 p.41–45).
  - The large leaf is our choice for a weak signal. It departs from L8 p.44, where a forest's trees are grown deep.
  - A deep forest (leaf 5) tests that choice.
  - The forest's inputs include the class dummies, so its comparison with ridge is fair.
- **Per-class ridge**, which fits each class separately against the pooled model.

**The ledger: 38 specifications, listed before any fit.**
- **29 core:**
  - class means (expanding);
  - {OLS, ridge, forest} × {M2, M3} × {static, expanding, rolling};
  - {lasso, PCR} × M3 × the three schemes;
  - ridge with global-only and with country-only macro;
  - per-class ridge on M2 and on M3.
- **9 pre-listed extras:**
  - PCR with K fixed at 3 and at 5;
  - deep forests on M2 and M3;
  - a 1-month macro lag;
  - the within-class z-score;
  - characteristic levels instead of ranks (M2 and M3);
  - curated-only macro with `x10` at a 14-month lag, so every month uses a completed prior-year average.
- **Not candidates:** the 8 placebo runs, and a demonstration of the `x10` leak at the naive 1-month lag (reported in the leakage audit only).
- **Multiple testing.** A "best of 38" claim must clear a Bonferroni critical |t| of 3.27, against 1.97 for a single test (L4 p.25–28). One post-hoc interaction specification makes it 39, and the bar 3.28.

### 3.5 Inference
- **Month bootstrap** (L2 p.80: resample the actual observations). Each resample keeps a whole month's cross-section, because all 50 assets share each month's shocks. B = 10,000, and the forecasts are held fixed (our construction).
  - Every R²_OOS and every paired difference between models gets an SE, and results are read as estimate ± 2 SE (L2 p.85).
  - A 12-month moving-block bootstrap (post hoc) shows serial dependence makes the SEs a median of 1.19 times larger. Every verdict is already "not significant", so this does not change any.
- **Power, stated before any result** (our construction, from var(X̄) = σ²/n, L2 p.63).
  - A ranking signal of 0.3% within class would be detectable. A class-level (macro) signal of the same size would not: all assets in a class share it, so the effective cross-section is small (8.077).
  - Q2 is therefore read from magnitudes, bootstrap SEs and the placebo, not from a single test.
  - The detectable effect of the actual Q1 statistic (post hoc): the rule needs an R²_OOS of 0.48% to pass half the time, and 0.68% to pass with 80% probability.
- **Set aside:**
  - a Diebold–Mariano test (in no lecture; its i.i.d. form is the paired t-test the bootstrap already gives);
  - percentile bootstrap intervals and HAC or robust SEs (not taught);
  - F-tests and information criteria (they treat asset-months as independent rows, and the rows are strongly correlated within a month).

### 3.6 The decision rules, fixed before any result

| question | primary test | "yes" only if |
|---|---|---|
| Q1 | R²_OOS of ridge M2 (expanding) against the pooled trailing mean | R²_OOS > 0 and more than 2 bootstrap SEs above 0 |
| Q2 | gain in R²_OOS of ridge M3 over ridge M2 (expanding) | the gain is more than 2 SEs above 0 **and** larger than every placebo gain |
| Q3 | net Sharpe of P1 (on ridge M2 forecasts, 10 bp) against EW, RP and TSMOM, and α of P1 regressed on the three net rules | higher net Sharpe than all three **and** α > 0 with t > 2 |

### 3.7 Results

**Q1: not detectably forecastable. NO.**
- **The primary test.**
  - Ridge M2 scores **0.042% (SE 0.241%)** against the pooled trailing mean and −0.279% (SE 0.570%) against zero.
  - Class means alone score 0.087% (SE 0.236%).
  - The characteristics add **−0.045% (SE 0.027%)** beyond class means (±2 SE: −0.099% to 0.009%).
  - This is "not detectable", not "zero".
- **Across the ledger.**
  - The best of the 38 specifications is the rolling forest on characteristics, 0.21% (SE 0.36%), far from the Bonferroni bar.
  - No specification beats a zero forecast. The mean of y fell from 0.137 in training to 0.036 out of sample, so zero beats the pooled mean by 0.32%.
- **Signs of no signal.**
  - Ridge chose the no-signal end of its grid in 64% of refits (93% once macro is added).
  - Even the best fixed penalty chosen with hindsight (post hoc) does not beat the class means.
  - Static fits are worst (paired differences, post hoc): static minus expanding is −0.727% for ridge M2 (t = −1.89).
- **Flexible models overfit.**
  - OLS on M3 scores −20.4% static and −4.7% expanding.
  - The deep forests score −2.2% and −6.2%, with in-sample R² of 27% and 58%.
  - Forest minus ridge on characteristics is −0.083% (SE 0.130%, post hoc): no detectable nonlinearity.
- **Against each asset's own trailing mean** (post hoc). Class means score +0.36% (t = 2.18) and ridge M2 +0.32% (t = 1.92). That is shrinkage of noisy asset means toward class means, not characteristic signal.

**Q2: macro adds nothing. NO.**
- **The primary test.** Ridge M3 minus ridge M2 is **−0.123% (SE 0.085%)**. Macro lowers R²_OOS in every pre-listed paired comparison (Table 3.18); global macro alone costs −0.414%, country macro alone −0.106%.
- **The placebo.**
  - Every raw macro series, global and country, is circularly shifted by s ∈ {36, 48, …, 120} months before any transformation, and the whole pipeline is rebuilt and re-tuned.
  - Real macro ranks **4th of 9**; the placebo gains span −0.563% to 0.125%.
  - Shifts of 72 months or more place end-of-sample macro at early test dates (1 to 49 target months), which is disclosed. Such values are real data at the wrong date, which is what a placebo is meant to be.
- **The one positive macro comparison** is the curated `x10` (14-month lag) against the `x86` version: +0.146% (SE 0.069%). It is one of nine robustness checks, and far from the Bonferroni bar.
- **The naive 1-month `x10` leak** scores −0.050%: the look-ahead would not have produced a spurious "yes" here, but it had to be excluded anyway.
- **Post hoc.** Characteristic × macro interactions (a 39th specification) add +0.012% (SE 0.089%) and rank 6th of 9 against their own placebos.

**Q3: no forecast-driven outperformance. NO.**
- **The portfolios.**
  - All are at unit gross exposure, rebalanced monthly, with trailing σ̂ from past returns.
  - EW = 1/50.
  - RP ∝ 1/σ̂.
  - TSMOM ∝ sign(compounded 12-month return)/σ̂.
  - **P1 ∝ ŷ/σ̂** (primary). A constant forecast makes P1 exactly RP, so RP is its natural bar.
  - **P3** is within-class long–short on the forecasts. It isolates the cross-sectional bet.
- **Costs.**
  - 10 bp per unit of turnover Σ|Δw| (buys plus sells), measured against drifted weights and charged to every strategy.
  - Also shown: a 0–50 bp grid and break-even costs, so readers can substitute their own cost.
  - Class-specific costs were set aside: they would need a sourced number per class.
- **Result.**
  - Net Sharpe: P1 **0.44**; EW 0.28; RP 0.30; TSMOM 0.12. P1 beats all three on the point estimate.
  - α on the three net rules is **0.61% a year, t = 1.74**. The rule needs t > 2, so NO.
  - The Sharpe gain over RP is 0.14 (SE 0.11).
- **Where the gain comes from.** Not from the characteristics:
  - P1's weights correlate 0.9994 with those of the same portfolio built on class means alone, which earns the same 0.44.
  - With that class-means portfolio as a regressor, α's t falls to 0.82.
  - P3, the pure characteristic bet, earns a gross Sharpe of 0.08 and −0.11 net, at 34% monthly turnover.
  - The gain depends on re-estimating the class means every year (post hoc): the same model frozen at 2010 earns 0.27, below RP (α −0.16%, t = −0.91).
- **Fragility.**
  - α is −0.16% (t = −0.37) in 2011–17 and +1.36% (t = 2.47) in 2018–24 (post hoc), and EW beats P1 in 2018–24.
  - The same model on rolling windows (not selected; post hoc) earns a net Sharpe of 0.60 at 8.2% monthly turnover, so the portfolio result depends on the scheme.
  - Six months have studentized residuals beyond ±2.5. Without them t rises to 2.20: reported, not used, because L3 p.17 allows deleting a point only for a good reason.
- **Concentration.** 60% of P1's gross exposure is in class D (37% for RP). asset_16, the least volatile asset, averages 22% and reaches 42%. A class-risk-budget variant is shown as a diagnostic (our choice: each class sleeve at unit gross, scaled by the inverse trailing volatility of that class's RP sleeve). It cuts class D to 46% and the net Sharpe ratio to 0.41.
- **Break-even costs.** 260 bp against EW and 363 bp against RP, so costs are not what decides Q3.
- **Set aside:** mean–variance or minimum-variance weights. They are not taught, and a 50 × 50 covariance matrix from 36–96 months is rank-deficient or badly conditioned (L1 p.63). Correlation is handled through the class risk budget instead.

### 3.8 Leakage audit (Section 3.8 of the notebook)
The five questions of L5 p.58 and the traps of AI Coding Guide §4 are each answered with a check in code:

| question | answer |
|---|---|
| scaler, imputer or PCA fitted on the full sample? | No. Every `StandardScaler` and PCA is fitted on training rows of each window and validation fold. Ranks are cross-sections of one month. Macro z-scores are trailing |
| predictors known at the end of t−1? | Yes. The characteristics are lagged (`x2` and `x5` are stored already lagged). Every macro series is lagged 2 months. `x10` is replaced by `x86`. Checked by 20 lag spot checks and by the truncation test |
| a random fold on time-ordered data? | No. There is one forward-only splitter, and order is asserted at every model refit |
| hyper-parameters chosen on test data? | No. Validation folds lie inside the training window. The whole-grid test-period curve is post hoc, labelled as an upper bound chosen with hindsight, and chooses nothing |
| benchmark switched after seeing results? | No. The pooled trailing mean was fixed in Section 3.0; every other benchmark is reported beside it |
| too good to be true? | An alarm fires on any R²_OOS above 2% (L5 p.57). It is not triggered |

### 3.9 Deviations from the design, and post-hoc analyses
The notebook's Section 3.10 lists these in full; in summary:
- **Deviations from the design** (none changes a verdict):
  - the leak alarm is applied to the largest R²_OOS, since the design means a suspiciously high value (not triggered);
  - in the curated-`x10` variant, the 50 rows of 2003-01 get the neutral value 0, because a 14-month lag leaves that month without a trailing z-score;
  - "in-sample fit exceeds out-of-sample fit" is reported as a count (38 of 38), not asserted;
  - the design text says "one-way turnover"; the cost is charged on Σ|Δw|, twice one-way turnover under the usual convention, so it is conservative.
- **Post-hoc analyses** (added after the out-of-sample results were seen; each is labelled post hoc in the notebook; they re-use the test window and none changes a verdict):
  - descriptive: the tercile averages (Table 3.6b), the macro inputs described (Table 3.5b), the spread of the target by class and year (Tables 3.6c–d), the forecast deciles (Table 3.17b);
  - inference: the detectable effect of the Q1 statistic, the per-asset benchmark summary (Table 3.10c), paired scheme differences (Table 3.12b), serial dependence and the moving-block bootstrap (Tables 3.12c–d), forest minus ridge (Table 3.12e), R²_OOS at every fixed penalty (Table 3.14c), PCR's loadings (Table 3.14d), PCR with K = 0 allowed;
  - macro: the cross-country share of country-macro variance (Table 3.18b), where the placebo wraps (Table 3.19b), and characteristic × macro interactions (Table 3.19c; the 39th specification);
  - portfolios: P1 on the static and rolling forecasts (Tables 3.24–3.26, 3.29), the α regression by sub-period (Table 3.27b), the class tilt year by year (Table 3.28b), and the α diagnostics and bootstrap SE (Tables 3.27c–e).

### 3.10 Methods considered and not used

| method | why not |
|---|---|
| **Gradient boosting** | Rounds, depth and learning rate would have to be tuned on the same forward folds. That is possible without leakage, but it multiplies the runtime, and L8 p.55 recommends the forest when time is short. No flexible learner beat ridge (forest minus ridge within about one SE in five of six pairs), so the expected value was low |
| **KNN** | The lectures teach the KNN classifier (L6), not KNN regression. Distances over 42 columns meet the curse of dimensionality, and the small-leaf forest, which plays the local role, lost |
| **Neural networks, SVM, elastic net** | Only named in the lectures; kept out by our choice of scope (the exam itself allows any method) |
| **A classification framing** (logistic regression, a sign classifier) | Q1 asks how much of a continuous return is forecastable, and the exam prescribes R²_OOS; a sign answers a different question |
| **Clustering** | The classes are given; clustering would be exploratory. Class A is a loose block, which is noted for interpretation |
| **Information criteria, F-tests** | They assume independent rows and a parameter count; asset-months are correlated within a month, and no count is taught for a penalised model |
| **Screening the extended file** (≈150 columns) | No hypothesis, unknown release timing for the low-frequency columns, duplicates, and a much larger search (L4 p.25–28; L5 p.41) |
| **Outside data** | Allowed by the exam, but each series would need a documented source and release date; the assets are anonymised, and there was no hypothesis to test |
| **Cross-country macro transforms, a widened panel** | Suggested by the exam as options. Not pre-listed; country macro already varies across countries (20% to 80% of its variance lies across countries within a month) and loses |
| **Tuning the forest's leaf size in every fold** | Runtime; the fixed deep forest shows the direction instead |
| **`x10` at a 1-month lag; `x11`; forward-filled `x11`** | Look-ahead; gaps and a stale fill through a regime change (§3.1) |
| **Shuffled K-fold, `*CV` estimators, a random hold-out** | Fatal on time-ordered data (AI Coding Guide §4a) |

### 3.11 Limitations
- The data are anonymised, so the class and characteristic identities are inferences.
- The test period is 168 months. Only an R²_OOS of about 0.5–0.7% could have been reliably detected, so the answer to Q1 is "not detectable" rather than "absent".
- The month bootstrap treats months as independent (the block bootstrap shows the SEs are somewhat too small, which only strengthens the "no" verdicts).
- The post-hoc diagnostics re-use the test window that the design meant to score once; none changes a verdict.
- Credit, valuation and external-balance macro series (in the extended file) were not tested.
- P1's apparent edge depends on one period, one asset class and one very low-volatility asset.

---

## 4. The answers at a glance

| question | answer |
|---|---|
| 1.1 | Baseline 64.25%; unpruned tree 85.00%; CV prefers **3 leaves (90.75%)**, tied with 4 (identical predictions), the smaller chosen; 2 leaves underfit, 6+ leaves overfit |
| 1.2 | Over 42: mostly buy; 42 or younger: buy only with salary above $90,000; Gender unused |
| 1.3 | Forest 88.75%, boosting 89.00%: **neither beats the tree**; one-rectangle signal, and the untuned ensembles fit noise |
| 1.4 | **MDI: Salary first; held-out permutation: Age first.** MDI rewards many split points and in-sample noise; show permutation |
| 1.5 | Shuffled CV leaks the future and biases accuracy **upward**; use walk-forward evaluation with a training-period benchmark |
| 2.1 | The student's own prediction, written before any code |
| 2.2 | One desk: σ̂² = 1.405 × 10⁻⁵ (3.2nd percentile); median 0.00585; 57.8% below a tenth of the truth; the story is typical (52.7th percentile) |
| 2.3 | One-step mean ≈ 1; drift **−1/(n−1)**; expectation 1, median 0.0059, mean 2.34 (top 10 hold 88.8%); errors compound multiplicatively |
| 2.4 | (a) band 0.94–1.06: **accumulation of noise**, contained; (b) kurtosis 24.1 lost on night 1: **loss of information**; only real data in every generation stops it |
| Q1 | **No detectable predictability**: 0.042% (SE 0.241%); characteristics add −0.045% (SE 0.027%) to class means |
| Q2 | **Macro adds nothing**: −0.123% (SE 0.085%); real macro ranks 4 of 9 against placebos |
| Q3 | **No**: P1 net Sharpe 0.44 vs 0.28 / 0.30 / 0.12, α t = 1.74; the gain is a re-estimated class-premium tilt, concentrated in bonds |

---

## 5. How the notebook is organised and how to run it

- **41 cells in the exam's order.** Every exam question is followed by its code cell and its answer cell (2.1 has no code cell, because it is answered before any code; 1.5's code cell only records that no code is needed). Problem 3's analysis is one code cell, printing a banner where each section (3.0–3.9) starts. The write-up is the final cell, followed by the notes of Sections 3.0–3.10 as appendices; table and figure numbers (Table 3.x, Figure 3.x) are shared between the code output and the write-up.
- **Running it.** On the class server as it stands, or with the data files next to the notebook. It runs top to bottom with no errors, in about 14 minutes on four cores (Problem 3 in about 12).
- **Environment.** Python 3.11, pandas 3.0.6, numpy 2.4.6, scikit-learn 1.9.1, statsmodels 0.15.0 (the class server has Python 3.14, pandas 3.0.5 and scikit-learn 1.9.0; no known dependence on the difference).
