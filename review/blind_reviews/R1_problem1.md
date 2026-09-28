# R1: Blind grading of Problem 1 (Trees and Ensembles, 20 points)

Sources used: `exam.txt`, `solution_A_P1.txt`, `solution_B_P1.txt` (notebook JSON checked for cell structure and execution order), `lectures/*.txt`, `data/Social_Network_Ads.csv`. Verification scripts: `reports/p1_work/verify.py`, `reports/p1_work/rf_check.py` (scikit-learn 1.9.1).

## 0. Independent verification

I re-ran every required model with `StratifiedKFold(5, shuffle=True, random_state=7034)` and `random_state=7034` on each estimator. **Every accuracy and importance number that either solution prints reproduces exactly.**

| quantity | my run | A prints | B prints |
|---|---|---|---|
| "nobody purchases" | 0.6425 | 0.6425 | 257/400 = 64.25% |
| unpruned tree | 0.8500 | 0.8500 | 85.00% |
| leaves 2/3/4/6/8/12 | .835/.9075/.9075/.895/.8825/.875 | same | same |
| people correct per fold, 3 leaves | 68 72 73 74 76 | n/a | same |
| training-fold acc. 2/6/12 leaves | .840/.926/.944 | n/a | same |
| RF 300 / GB 100 | 0.8875 / 0.8900 | same | same |
| RF / GB training-fold acc. | 0.998 / 0.974 | n/a | 99.81% / 97.44% |
| 70/30 split MDI (G, A, S) | .0111/.4699/.5191 | same | .011/.470/.519 |
| permutation, held out, n_repeats=10 | G .0192, A .2433, S .1817 | same | n/a |
| permutation, held out, n_repeats=50 | G .0218, A .2550, S .1713 | n/a | 2.18/25.50/17.13 pp |
| permutation, **training** rows, 50 | G .0503, A .2696, S .2883 | n/a | same |
| distinct values, full data | G 2, A 43, S 117 | "≈43", "≈116" (**no cell prints these**) | 2/43/117 (printed, cell 6) |
| distinct values, 280 training rows | G 2, A 42, S 108 | n/a | 2/42/108 (printed, cell 19) |
| splits in 1.4 forest (G/A/S) | 776 / 4751 / 5623 | n/a | same |

Extra checks I ran to test the 1.3 explanations:

| check | result | what it shows |
|---|---|---|
| share of forest splits on Gender (full-data RF, default `max_features='sqrt'`=1) | **6.0%** (A 43.3%, S 50.8%) | Gender is **not** split on "roughly one-third" of the time. Once a node is split on Gender its children are constant in Gender, and sklearn draws again when the drawn feature is constant. |
| RF with `max_features` = 1 / 2 / 3 | 0.8875 / 0.8975 / 0.8875 | Considering all 3 features (plain bagging) gives **exactly** the same 88.75%, so feature subsampling does not explain the shortfall. |
| RF with `min_samples_leaf` = 5 / 10 / 20 | 0.9025 / 0.9100 / 0.9125 | The shortfall comes from deep, fully grown trees fitting noise (training-fold accuracy 99.8%). |
| RF without Gender | 0.8975 | Gender costs about 1 pp through noise-fitting, not through forced splits. |
| GB with 10 / 20 / 50 / 100 rounds | 0.9075 / 0.9025 / 0.8925 / 0.8900 | Supports "100 rounds starts fitting noise" (A and B both say this; only B prints any evidence for it). |
| purchase rate by Gender | F 37.7%, M 33.7% | "Gender has zero predictive signal" (A) is not established by anything printed and is not literally true. |
| 4-leaf tree on all 400 | 4th split is Age <= 46.5, both children class 1 | Confirms B's explanation of the 3/4 tie. |
| data granularity | ages are integers; salaries are multiples of 1,000 | B's "over 42" / "more than $90,000" are exactly equivalent to the tree's thresholds 42.5 / 90,500. |

Neither solution has a FATAL error. Problem 1 is a cross-section, so shuffled stratified CV is what the exam requires. Both use the exam's `cv5` and `random_state=7034` on every tree, forest, boosting model and split, and the 70/30 split uses `stratify=ya`. The default `permutation_importance` scoring is the estimator's `.score`, which is accuracy, so no sklearn default quietly changes the method.

---

## 1.1 Baseline, unpruned tree, leaf sweep

**A.** Reports 64.25%, 85.00% and the full sweep. Picks 3 leaves by the tie rule. Says size 2 underfits and 6/8/12/unpruned overfit, "monotonically degrading ... from 90.75% down to 85.00%". The numbers and the choice are correct.
- It says CV "**strictly** prefers" 3 leaves in the same sentence as "ties size 4". The two claims contradict each other.
- "(1 − 0.3575)": 0.3575 is not printed anywhere. Cell 4 prints "35.8%".
- "+26.50% above the naive baseline": no cell computes this. It is also a difference in percentage points, not per cent.
- "Achieves the global bias-variance optimum": an overclaim. The search covered a six-point grid.
- The overfitting claim rests only on falling CV accuracy. No training-accuracy diagnostic is shown. There are no lecture citations.

**B.** Same numbers, plus people correct out of 400, fold spread, and training-fold accuracy as an explicit diagnostic. Every number in the answer is printed by cell 6 or 7.
- It handles the tie rule in code (`k_star = min(tied)`). It also proves the tie is structural: "3- and 4-leaf trees make identical out-of-fold predictions for all 400 people: True". I confirmed the reason it gives, that the 4th split separates two class-1 leaves.
- It shows the bias/variance shape with training-fold numbers: 2 leaves 84.0% train and 83.5% CV; 6→12 leaves CV falls while training rises 92.6%→94.4%; unpruned 99.8%/85.0%. It cites L8 p.32–33 and p.38, and L8 p.33 is indeed this exact tree.
- It is honest about selection optimism (L4 p.43, L5 p.57) and compares fold by fold: 3 leaves beat 6 leaves in 4 of 5 folds, which I verified.
- Minor: "1.25 pp of a fold / 0.25 pp of the pooled figure" is simple arithmetic that no cell prints. The claim about the 4-leaf tree's 4th split is correct but no cell prints it; the OOF-identity check stands in for it.

**Verdict: B is better.** A's answer is correct but thinner, and it has the contradictory "strictly" plus two unprinted numbers.

## 1.2 Plot and one-to-two-sentence description

**A.** Uses `plot_tree(... feature_names=list(Xa.columns))` on the 3-leaf tree refit on all 400. The answer is two sentences: over 42.5 buy; 42.5 or younger buy only if salary is above $90,500. This is correct, plain and within the limit. The thresholds appear only in the figure (precision=2), which is acceptable.

**B.** Same tree. It also prints `export_text` with class weights, a leaf table (115/97, 241/9, 44/37), and a check that the tree uses Gender (False). The answer is two sentences with counts and "about 84 in every 100". That is well suited to a lay reader and every count is printed. "Over 42" and "more than $90,000" are exactly equivalent to the tree on this data. B adds a "technical note (outside the two sentences)". It is clearly labelled, but it goes beyond what was asked (STYLE). The fold-stability check (thresholds 42.5–44.5 and 89,500–91,500) is printed and is a nice touch.

**Verdict: both earn full credit.** B's version is more informative.

## 1.3 Random forest and gradient boosting vs the tree

**A.** Reports 88.75% and 89.00% and says neither beats the tree. This is correct.
- Cell 12 **hard-codes** the tree's result: `print(f"Optimal Pruned Tree (3 leaves) 5-Fold CV Accuracy: 0.9075 (90.75%)")` is a string literal, not a computed value. The number is right, but the cell does not produce it.
- "+1.75%" and "+2.00%": no cell prints these, and they are percentage points.
- **The RF mechanism is wrong.** A writes: "roughly one-third of all split evaluations are forced to consider only `Gender` ... injects pure subsampling noise". In fact only 6.0% of forest splits are on Gender, because Gender can split at most once per path. Also, `max_features=3`, which removes subsampling entirely, gives the identical 88.75%. The real driver is fully grown trees: RF training-fold accuracy is 99.8%, and `min_samples_leaf=10` gives 91.0%.
- "`Gender` providing zero predictive signal": no printed output supports this. A's own 1.4 output shows a held-out permutation drop of 0.0192 for Gender, and purchase rates differ (37.7% vs 33.7%).
- "The true data-generating boundary is an elementary orthogonal step function": an unsupported overclaim.
- The GB "over-iteration" story points the right way (my check: 10 rounds gives 90.75%), but A prints no evidence for it.
- **The sentence limit is exceeded.** The exam asks for two or three sentences explaining why. A gives an intro sentence plus two bulleted paragraphs of about five sentences, plus a verdict/conclusion block.

**B.** Reports a table with all four models and the baseline, people correct, and fold-by-fold head-to-head. All figures are printed in cell 15. The fold analysis is verified: the shortfall is concentrated in fold 4 (tree 74, RF 68, GB 67), and the other folds are within 2 people. It concludes "no better than", not "worse". That is an honest reading.
- The "Why" paragraph is three (long) sentences, so it is within the limit. Main mechanism: the pattern is roughly one rectangle that a 3-leaf tree already draws. A forest removes variance and boosting removes bias, and the small tree has little of either (L8 p.51). Contrast with L8 p.54. This is correct and grounded in the lectures.
- GB evidence is printed: training folds 97.4% vs the tree's 91.6%.
- WEAKNESS: "the forest tries one of the three features at each split, so a split can be forced onto a weak input". This is the same mechanism A uses, though B hedges it as a secondary point, and it does not survive the `max_features=3` check. B printed the forest's 99.81% training-fold accuracy but did not use it. Deep, unrestricted trees are the better explanation.

**Verdict: B is clearly better.** It has a correct main mechanism tied to the lectures, keeps to the sentence limit, and every number is printed. A's headline RF mechanism is mechanically wrong, and A breaks the limit and hard-codes a number.

## 1.4 MDI (in sample) vs permutation (held out)

**A.** The code is correct: 70/30, `random_state=7034`, `stratify=ya`, 300-tree RF on train, MDI, and permutation on the 30% with n_repeats=10. The side-by-side table is printed. Conclusions: the rankings disagree (MDI: Salary; permutation: Age), Gender is negligible, and permutation is what to show a decision-maker. All correct.
- "`EstimatedSalary` has ≈116 distinct values compared to `Age`'s ≈43": **no cell prints these.** 116 is wrong: it is 117 on the full data and 108 on the training rows where MDI is computed. This breaks the every-number-printed rule.
- "Scrambling `Age` destroys the root split (Age > 42.5)" confuses the single 3-leaf tree with the forest. Forest trees are grown on bootstrap samples with one random candidate feature, so they do not share that root.
- "24.33% drop in test accuracy" is 24.33 **pp**.
- "Directly isolates true out-of-sample economic generalizability" is mild overclaiming. The mechanism itself (high cardinality rewarded in sample) is right and matches L8 p.58, but A does not cite it.

**B.** Same design with n_repeats=50. The table adds splits per feature, distinct values on the training rows, permutation on the training rows as a diagnostic, and the Age–Salary correlation. Every quoted number is printed in cell 19.
- The explanation is correct and backed by evidence. MDI is in sample (L8 p.47, p.56). The trees are deep (38.2 leaves, 99.6% training accuracy). Salary has 108 vs 42 distinct values and gets 5,623 vs 4,751 splits, exactly L8 p.58's "rewards a column for merely offering many places to split".
- The training-row permutation also ranks Salary first (28.8 vs 27.0 pp), which is consistent with the mechanism. B honestly notes that the MDI margin is small and could flip, and it rules out the correlated-inputs mechanism (r = 0.155).
- It recommends permutation (L8 p.57) with sensible caveats: n=120, no sign (L8 p.60), not causal.

**Verdict: B is better.** A's core answer is right, but it rests on unprinted and partly wrong cardinality numbers and a loose "root split" argument.

## 1.5 Time-ordered data

**A.** Says shuffling causes look-ahead leakage, accuracy is biased **upward**, and the fix is expanding or rolling walk-forward evaluation. These points are correct and match L5 p.48 and Guide 4(a).
- "Collapses completely to near-zero or negative performance in live deployment" is wrong on the mechanics. Accuracy cannot be negative, and the honest collapse is to about the base rate, not near zero.
- "Trained strictly on data prior to month t−1 to predict month t" has an off-by-one: it should be data through t−1.
- "Purging and embargoing" is not taught in the 8 lectures (grep finds nothing) and is not labelled as the author's own addition.
- The cell 18 "code" only prints two slogan strings (STYLE).

**B.** Gives four mechanisms: look-ahead (L5 p.56, L1 p.82); near-duplicate neighbouring months, labelled "our own reasoning"; model selection tuned to the leak, using the lecture's 23-vs-15 variables and 0.036 R² cost (L5 p.56); and "why KFold(shuffle=False) and stratification don't fix it", the latter labelled as its own reasoning.
- Direction: biased upward, with the fatal-error note (L5 p.58, Guide §4a) and the base-rate sanity check (L1 p.33, L5 p.57).
- Fix: an expanding window with a separate validation block for `max_leaf_nodes`, an untouched final test period (L4 p.43), and a benchmark learned from training months only (L6 p.57, L4 p.46).
- Every lecture citation I checked says what B claims it says.

**Verdict: B is better.** It is more precise, correctly scoped and labelled, and makes no mechanical errors. A is acceptable but has a mechanical error and untaught content that is not labelled.

---

## 2. Findings

| id | solution | sub-q | severity | finding | evidence |
|---|---|---|---|---|---|
| A1 | A | 1.3 | ERROR | The RF mechanism is wrong: "roughly one-third of all split evaluations are forced to consider only Gender ... injects pure subsampling noise". | Cell 13. My check: Gender is 6.0% of splits; `max_features=3` gives the same 0.8875; `min_samples_leaf=10` gives 0.9100. |
| A2 | A | 1.3 | ERROR | "Gender providing zero predictive signal" is unsupported by any printed output and contradicted by A's own 1.4 output. | Cell 15 prints Gender permutation 0.0192 (sd 0.0140). Purchase rates F 37.7% vs M 33.7%. |
| A3 | A | 1.3 | ERROR | The explanation breaks the "two or three sentences" limit: intro plus two bulleted multi-sentence paragraphs plus verdict/conclusion blocks. | Cell 13 vs exam cell 11. |
| A4 | A | 1.4 | ERROR | "≈116 distinct values" for Salary and "≈43" for Age are not printed by any cell. 116 is wrong. | No `nunique` in A's P1 code. True values: 117 full / 108 training rows (B cell 6 and 19; my run). |
| A5 | A | 1.3 | WEAKNESS | The tree's 0.9075 is hard-coded as a string in a print statement, not computed in the cell. | Cell 12, last `print`. |
| A6 | A | 1.1, 1.3 | WEAKNESS | Derived numbers no cell prints: "0.3575", "+26.50%", "+1.75%", "+2.00%". These are also percentage-point differences written with a % sign. | Cells 7 and 13. Cell 4 prints 35.8% only. |
| A7 | A | 1.3 | WEAKNESS | "The true data-generating boundary is an elementary orthogonal step function" is an overclaim about an unknowable DGP. | Cell 13. |
| A8 | A | 1.4 | WEAKNESS | "Scrambling Age destroys the root split (Age > 42.5)" confuses the pruned single tree with the 300 bootstrap / `max_features=1` forest trees. | Cell 16. |
| A9 | A | 1.5 | ERROR | "Collapses ... to near-zero or negative performance": accuracy cannot be negative. The honest collapse is to about the base rate. | Cell 19, item 3. |
| A10 | A | 1.5 | WEAKNESS | Off-by-one, "data prior to month t−1 to predict month t". Also "purging and embargoing" is not in the 8 lectures and is not labelled as the author's own. | Cell 19, item 4. grep of lectures finds no "embargo" or relevant "purg". |
| A11 | A | 1.1 | STYLE | "Strictly prefers" contradicts "ties size 4". "Global bias-variance optimum" overstates a six-point grid search. | Cell 7. |
| A12 | A | all | STYLE | No lecture citations in Problem 1. Templated headings ("Direct Verdict / Mechanism / Actionable Conclusion"). The 1.5 cell prints slogans. | Cells 7, 13, 16, 18, 19. |
| B1 | B | 1.3 | WEAKNESS | Secondary mechanism "the forest tries one of the three features at each split, so a split can be forced onto a weak input" does not survive a check. B's own printed 99.81% RF training-fold accuracy (deep trees) is the better explanation and is unused. | Cell 15/17. My check: `max_features=3` gives 0.8875, the same as the default. |
| B2 | B | 1.1 | STYLE | Small derived numbers not printed ("1.25 pp of a fold, 0.25 pp pooled"). The 4-leaf tree's extra split is described but not printed; it is correct per my run, and the OOF-identity check is printed. | Cell 9; cell 7 output. |
| B3 | B | 1.2 | STYLE | A "technical note (outside the two sentences)" follows the two-sentence answer. It is labelled, but the answer ends up longer than the brief. | Cell 13. |
| B4 | B | 1.1, 1.3, 1.5 | STYLE | Long answers. The 1.3 "why" is three sentences, but the first is a very long colon-joined sentence. | Cells 9, 17, 24. |

No FATAL findings for either solution.

## 3. Scores (out of 20)

Weights I used: 1.1 = 5, 1.2 = 3, 1.3 = 4, 1.4 = 4, 1.5 = 4.

| sub-q | A | B | note |
|---|---|---|---|
| 1.1 | 4.0 | 5.0 | A: correct but unprinted derived numbers and "strictly"/tie contradiction. B: complete, with a structural proof of the tie. |
| 1.2 | 3.0 | 3.0 | Both correct, within the limit and lay-readable. |
| 1.3 | 1.5 | 3.5 | A: wrong RF mechanism, unsupported "zero signal", sentence limit broken, hard-coded number. B: correct main mechanism, one weak secondary claim. |
| 1.4 | 2.5 | 4.0 | A: right conclusion, but unprinted and partly wrong cardinality numbers plus the root-split conflation. B: evidence-backed, cited, honest. |
| 1.5 | 3.0 | 4.0 | A: right direction and fix, but "negative accuracy", off-by-one and unlabelled untaught content. B: precise and cited. |
| **Total** | **14.0 / 20** | **19.5 / 20** | |
