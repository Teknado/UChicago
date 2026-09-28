# Plan: Problem 1, "Trees and Ensembles" (parts 1.1 to 1.5), BUSN 41210 Final, Autumn 2026

Status: **plan only.** No model was fitted on the exam data and no exam number was computed. The only things run were API and default checks in the installed environment (Python 3.11, pandas 3.0.6, sklearn 1.9.1). Some of these used synthetic data with the same class counts, including a dry run of the `evaluate` helper on synthetic data. Data facts come from `notes/data_profile.md` §A.

Citation keys: "L<n> p.<page>" = PDF page, as used in `notes/Lecture_<n>.md`. "Exam cell N" = 0-based cell index in `Final_Autumn_2026-1.ipynb`: cell 3 is the P1 intro, 4 the setup code, 5/8/11/14/17 the questions 1.1–1.5, 34 and 36 the P3 lag and benchmark rules.

---

## 0. Global design (applies to every part)

### 0.1 Fixed conventions

| Item | Value | Grounding |
|---|---|---|
| Data | `Xa = ads[['Gender','Age','EstimatedSalary']]`, shape (400, 3); `ya = ads['Purchased']`, shape (400,). Gender is already coded Male=1, Female=0 by exam cell 4. `User ID` is excluded because it is unique per row. | exam cell 4; dummy coding L3 p.45; L8 p.30 (`Gender <= 0.5` means 0/1 coding) |
| Folds | **Reuse** the notebook's `cv5 = StratifiedKFold(5, shuffle=True, random_state=7034)` and `cv_acc()`. Never pass `cv=5` as an integer: that gives *unshuffled* StratifiedKFold, so different folds. Never redefine `cv5`. The folds are deterministic across calls (verified). They are 5 × 80 test rows (verified), with about 28–29 buyers each. | exam cells 3–4; shared split L8 p.53 ("two numbers computed on two different splits are two different questions"); K-fold L5 p.28–35 |
| Seeds | `random_state=7034` on every `DecisionTreeClassifier`, `RandomForestClassifier`, `GradientBoostingClassifier`, `train_test_split`, `permutation_importance`. | exam cell 3; L6 p.18 ("since the sample is random, you might get different results") |
| Metric | Accuracy = 1 − misclassification rate, measured out of sample. It is always reported next to the majority-class ("nobody purchases") rate. | L6 p.19, p.57; L8 p.32 |
| Why shuffled folds and a random 70/30 split are legitimate *here* | The rows are 400 unrelated people (a cross-section), so a random fold imitates drawing new people. This is exactly the reason 1.5 says it fails for months. | L8 p.45 ("a cross-section, so a random split is legitimate"); L6 p.17 |
| No preprocessing | Trees split on `x_j <= c`, so they need no scaling. There are no NaN values (data profile). So P1 has **no scaler, imputer or PCA at all**, and nothing can leak through preprocessing. | L8 p.5 (split definition); data profile §A |

### 0.2 Where each AI-guide guard lives

| Guard | Where it is enforced in P1 |
|---|---|
| (a) shuffled folds / `train_test_split` on time-ordered data | Not violated: P1 data are a cross-section (L8 p.45). The shuffle is used knowingly, and the 1.1/1.4 answers say why it is fine here. **1.5 is the place where this exact error is explained** (L5 p.56, p.58 #3, p.59). |
| (b) scaler/imputer/PCA before split | None is used anywhere in P1 (0.1). If anyone adds a linear comparison model, it must be `Pipeline([('sc', StandardScaler()), ('m', ...)])` passed whole to `cross_val_score(..., cv=cv5)`, so it is refit inside each training fold (L5 p.47, p.58 #1). Default: no such model (not asked). |
| (c) `LogisticRegression()` is L2-penalised | Not used in P1. If a logit benchmark were ever added, it must be `LogisticRegression(penalty=None)` and described as the MLE (L6 p.38). Default: not added. |
| (d) `r2_score` / `.score` benchmark against the test mean | P1 uses accuracy, not R². The benchmark is the majority-class rule, **fixed before modelling** (L4 p.46 logic; L6 p.57). In 1.5 the analogue matters: the "always up/down" rule must be learned from the *training* months. `clf.score` on a classifier returns plain accuracy, which is fine. |
| pandas 3 / sklearn 1.9 pitfalls | Build tables with `pd.DataFrame(list_of_dicts)` or `pd.concat` (no `df.append`). Use `.items()`. Don't call `sns.set()`. Do **not** pass `criterion=` to `GradientBoostingClassifier`: it is deprecated in 1.9 and has no effect (verified docstring). Predict with DataFrames that keep the column names, which avoids feature-name warnings. |

### 0.3 Step 0: describe the data first (AI guide §6.1; L1 p.81–82)

This is printed at the top of cell 6. It describes the data only, and every value is checked against the data profile:
- `ads.shape` → (400, 5); `ads.dtypes`; `ads.head()`.
- `ya.value_counts()` → 0: 257, 1: 143 (35.75%).
- `Xa.describe()`: Age 18–60; salary 15,000–150,000, all multiples of 1,000.
- `Xa.nunique()` → Gender 2, Age 43, EstimatedSalary 117. This feeds 1.4 (MDI favours features with many possible split points).
- `Xa[['Age','EstimatedSalary']].corr()` (used in the 1.4 argument about correlated features splitting credit, L8 p.58).
- `ads.groupby('Gender').Purchased.mean()`.
- Conflicting-label cells: `ads.groupby(['Gender','Age','EstimatedSalary']).Purchased.agg(['size','sum'])`, keeping cells where 0 < sum < size. The profile says there is 1 such cell and 20 duplicated feature rows. This gives a known ceiling: the unpruned tree's in-sample accuracy must equal 1 − (minority rows in conflicting cells)/400, so it is not exactly 1.0. That is a check.

### 0.4 Helper (defined once in cell 6, reused by 1.1–1.3)

```python
from sklearn.model_selection import cross_validate
from sklearn.tree import export_text

N = len(ya)                                    # 400

def evaluate(model, name):
    """All numbers on the shared cv5 folds: per-fold test/train accuracy and per-person OOF hits."""
    r = cross_validate(model, Xa, ya, cv=cv5, scoring='accuracy',
                       return_train_score=True, return_estimator=True, return_indices=True)
    hits = np.empty(N, dtype=int)
    for est, te in zip(r['estimator'], r['indices']['test']):
        hits[te] = (est.predict(Xa.iloc[te]).astype(int) == ya.iloc[te].to_numpy())
    acc = r['test_score'].mean()
    assert np.isclose(acc, cv_acc(model))       # identical folds and fits as the notebook helper
    assert np.isclose(acc, hits.mean())         # equal 80-row folds, so pooled = mean of folds
    return {'model': name, 'cv_acc': acc, 'n_correct': int(hits.sum()),
            'fold_sd': r['test_score'].std(ddof=1),
            'fold_min': r['test_score'].min(), 'fold_max': r['test_score'].max(),
            'se': hits.std(ddof=1) / np.sqrt(N),               # L2 p.56-57 (ddof=1), p.63
            'train_acc': r['train_score'].mean(),
            'folds': r['test_score'], 'hits': hits, 'estimators': r['estimator']}

def paired(h_a, h_b):
    """Difference in accuracy between two models on the SAME 400 people, and its rough SE."""
    d = h_a - h_b                                # values in {-1, 0, 1}
    return d.mean(), d.std(ddof=1) / np.sqrt(N)  # L2 p.63; t-stat logic L2 p.91
```

**How to judge whether an accuracy difference is meaningful (used in 1.1 and 1.3).** Grounding: L2 p.63–64, p.85, p.91.
- Resolution. One person is 1/80 = **1.25 pp** of a fold's accuracy and 1/400 = **0.25 pp** of the pooled CV accuracy.
- SE of one accuracy. Accuracy is a sample mean of 0/1 hits, so var(mean) = σ²/n (L2 p.63). Estimate σ with the ddof=1 sd of the hits (L2 p.56–57); the closed form is ≈ √(acc(1−acc)/n). For an accuracy near 0.9, the level L8 p.32 reports, this gives about 1.5 pp at n = 400 and about 3.4 pp per 80-person fold.
- SE of a difference, paired. Every model is scored on the same people (L8 p.53), so the right yardstick is the sd of the per-person difference h_A − h_B, divided by √400. It is much smaller than √2 × 1.5 pp when two models disagree on few people: with m disagreements it is ≈ √m/400. A gap of less than about 2 SE is treated as indistinguishable (L2 p.85, p.91 "about 2 standard errors").
- Fold-level view. Also show the 5 per-fold differences and the count of folds where A beats B.
- Caveat to state in the answers. Hits within a fold share one fitted model, and folds share training rows, so the iid premise of L2 p.63 holds only roughly. Treat the SE as a yardstick, not a formal test (L5 p.41 "CV ... unstable"). McNemar's test and nested CV are **not taught** and are not used.
- Parsimony when within noise: prefer the simpler model (L5 p.25; L4 p.11, p.23).

---

## 1.1 Baseline, unpruned tree, `max_leaf_nodes` sweep

**(i) Goal.** (a) Accuracy of "nobody purchases", the bar every model must clear. (b) 5-fold CV accuracy of an unpruned tree. (c) CV accuracy for `max_leaf_nodes` ∈ {2, 3, 4, 6, 8, 12}. (d) The preferred size (ties go to the smaller). (e) What happens on either side of it.

**(ii) Method steps.**
1. Baseline, two equivalent computations, asserted equal:
   - `base_full = (ya == 0).mean()`. Expected 257/400 = 0.6425 (L8 p.30/p.33 root `value=[257,143]`).
   - `base_cv = np.mean([(ya.iloc[te] == 0).mean() for _, te in cv5.split(Xa, ya)])`. The rule needs no training, so this is the same rule scored on the same folds. With equal 80-row folds it must equal 0.6425 exactly; this was verified with synthetic labels of identical counts.
   - No `DummyClassifier`: it is not taught and not needed.
2. Unpruned tree: `DecisionTreeClassifier(random_state=7034)`, with all defaults (verified): `criterion='gini', splitter='best', max_depth=None, min_samples_leaf=1, max_leaf_nodes=None, ccp_alpha=0.0`. Run `res['unpruned'] = evaluate(...)`. Also fit it once on all 400 rows and record `get_n_leaves()`, `get_depth()` and in-sample accuracy.
   - Check against L8 p.31: about 93 nodes, depth about 12.
   - Check against step 0: the in-sample accuracy ceiling set by the conflicting cell.
3. Sweep: `for k in [2, 3, 4, 6, 8, 12]: res[k] = evaluate(DecisionTreeClassifier(max_leaf_nodes=k, random_state=7034), f'{k} leaves')`. Also record the full-sample `get_n_leaves()` (confirm it equals k) and `get_depth()`.
4. Choose the size with an **exact integer tie rule**, which avoids float equality:
   ```python
   sizes = [2, 3, 4, 6, 8, 12]
   best = max(res[k]['n_correct'] for k in sizes)
   k_star = min(k for k in sizes if res[k]['n_correct'] == best)   # exam: tie -> smaller
   tied = [k for k in sizes if res[k]['n_correct'] == best]
   ```
   `n_correct` is the number of people correctly classified out of 400 (= acc × 400, an integer because the folds are equal).
5. Meaningfulness: for each size, `paired(res[k]['hits'], res[k_star]['hits'])` against the chosen size, per 0.4.

**(iii) Lecture grounding per step.**
- Baseline as the bar: L6 p.57 (accuracy "with the majority-class rate next to it"); L6 p.36 (accuracy can merely equal the majority rate); L1 p.81 ("Compared to what?"); L4 p.46 (R²_OOS as a comparison of two predictors, same logic).
- Default tree = unpruned and overfit: L8 p.24, p.30 (pure 1–3-observation leaves on this very dataset), p.38 (grown-out tree interpolates).
- Size chosen out of sample, not in sample: L8 p.29 ("rely on out-of-sample prediction"), p.32 (test accuracy hump-shaped while training accuracy falls monotonically with pruning), p.33 (house style: `max_leaf_nodes` as the size knob; L8 chose 3).
- Averaging fold scores and picking the best: L5 p.28, p.35.
- Coarse grid is enough: L8 p.39. The optimum should be interior to the grid: L5 p.27. If k* = 12, say so, because the grid would then be edge-bound.
- Tie → smaller: exam cell 5; parsimony L5 p.25, L4 p.11/p.23.
- Either side = under- vs over-fitting: L5 p.25, L4 p.9, L8 p.40 (variance dominates for big trees).

**(iv) Assumptions.**
- A1. The choice is made among the six sweep sizes only. The unpruned tree is reported as the far-right reference, not as a candidate (the exam says "sweep ... which size does CV prefer").
- A2. A "tie" means an identical `n_correct` (the 1/400 resolution). The exam's rule is applied literally. The SE is reported next to it so the reader can see which sizes are statistically indistinguishable, but the SE does not override the rule. The 1-SE rule appears only in an L5 p.46 figure legend and is never taught, so it is not used.
- A3. `max_leaf_nodes` grows best-first by impurity decrease (sklearn behaviour). That is not literally cost-complexity pruning (L8 p.29), but it is the course's own knob (L8 p.33).
- A4. "Training accuracy" is the mean training-fold accuracy from the same fits (`return_train_score=True`). This mirrors L8 p.32's train curve.

**(v) Pitfall guards.**
- Same `cv5` for every number, enforced by the assert inside `evaluate`.
- `random_state=7034` on every tree: tie-breaking between equal splits uses the RNG.
- No integer `cv=5`.
- Selection optimism: the CV score of k* won a search over 6 sizes, so it is slightly optimistic (L4 p.43 "gray zone"; L5 p.57 "how many things did you try?"; L8 p.55). Disclose the search size in the answer. Nested CV is not taught, so it is not used.

**(vi) Outputs.**
- **Table 1.1**, one row each for: nobody purchases | unpruned | 2 | 3 | 4 | 6 | 8 | 12. Columns: `leaves (full-sample fit)`, `depth`, `CV acc`, `n_correct/400`, `fold sd`, `fold min–max`, `SE`, `diff vs k* (pp) ± paired SE`, `mean train-fold acc`, `in-sample acc (full fit)`. The chosen row is marked, and tied sizes are listed if any.
- **Figure 1.1**, in the style of L8 p.32:
  - x = number of leaves on a log scale (2, 3, 4, 6, 8, 12, then the unpruned leaf count as a separate marker labelled "unpruned");
  - y = accuracy;
  - lines: CV accuracy with ±1 SE error bars, and mean training-fold accuracy;
  - a dashed horizontal line at 0.6425 labelled "nobody purchases";
  - a vertical marker at k*;
  - title "Accuracy vs tree size, 5-fold stratified CV (random_state=7034)".
- Printed: `k_star`, `tied`, baseline value(s).

**(vii) The written answer must cover:**
- [ ] The "nobody purchases" accuracy (257/400 = 64.25%), stated as the bar. Note it is identical on the 5 folds.
- [ ] Unpruned tree: CV accuracy, its size (leaves/depth), and the in-sample vs CV gap. Say whether it beats the baseline, and by how much.
- [ ] CV accuracy at each of 2, 3, 4, 6, 8, 12 (from Table 1.1).
- [ ] The size CV prefers. State whether a tie occurred and that the smaller size was taken.
- [ ] Left side: fewer leaves underfit. For example, 2 leaves is one split, probably on Age: bias, and training and CV accuracy both lower.
- [ ] Right side: more leaves leave CV accuracy flat or falling while training accuracy keeps rising. That is fitting noise, and the unpruned tree is the extreme case.
- [ ] How big the gaps are against the ~1.5 pp SE and the paired SE (1 person = 0.25 pp pooled). Neighbouring sizes may be statistically indistinguishable.
- [ ] One sentence relating the result to L8 p.32–33. L8 chose 3 leaves from one unspecified train/test split. If our CV choice differs, attribute it to a different validation design and noise (L6 p.18).

**(viii) Open questions (with defaults).**
- Q1.1a: should "tie" allow a tolerance, such as within 1 SE? **Default: no.** Use exact equality of `n_correct`, and report the SE alongside.
- Q1.1b: is the unpruned tree a candidate for "preferred size"? **Default: no**; it is a reference only.

---

## 1.2 Plot the chosen tree and say what it does in plain English

**(i) Goal.** Plot the k*-leaf tree with `plot_tree(..., feature_names=Xa.columns)`. Describe it in 1–2 non-technical sentences.

**(ii) Method steps.**
1. **Fit on all 400 rows:** `tree_star = DecisionTreeClassifier(max_leaf_nodes=k_star, random_state=7034).fit(Xa, ya)`.
   - Why: CV chose a *size* (a procedure), not a particular tree. Each of the 5 fold-trees saw only 320 people. Once the tuning value is chosen, the model is refit on all the data (L5 p.36 step 4). L8 p.33 does exactly this on this dataset: the root shows `samples = 400`.
   - The honest accuracy to quote is the CV number from 1.1, not the in-sample accuracy implied by the leaf counts (L4 p.43–46; L6 p.54).
2. Plot:
   ```python
   fig, ax = plt.subplots(figsize=(12, 6))
   plot_tree(tree_star, feature_names=Xa.columns, class_names=['No purchase', 'Purchase'],
             filled=True, rounded=True, impurity=True, proportion=False, precision=3, fontsize=10, ax=ax)
   ```
   Passing `feature_names=Xa.columns` (a pandas Index) works in sklearn 1.9.1; this was verified.
3. Text version with the exact thresholds: `print(export_text(tree_star, feature_names=list(Xa.columns), show_weights=True))` (L8 p.25).
4. Leaf table: `leaf = tree_star.apply(Xa)`, then group `ya` by leaf to get n, number purchased, purchase rate and predicted class (majority). Add a hand-written rule string per leaf. The leaf class is the majority vote and the leaf probability is the class share (L8 p.17; Bayes rule p̂ > 0.5, L6 p.20/p.43).
5. Data-plane picture, as in L8 p.34:
   - scatter of Age (x) vs EstimatedSalary (y), coloured by `Purchased`;
   - predicted regions shaded by predicting `tree_star` on a grid (`Age` 18–60 step 0.25 × `EstimatedSalary` 15,000–150,000 step 500), passed as a DataFrame with the `Xa` column order;
   - if the tree splits on Gender, draw one panel per Gender value (0 = Female, 1 = Male); otherwise hold Gender fixed, since it does not affect the prediction.
6. Stability check (optional, default ON because it is cheap): from `res[k_star]['estimators']`, print each fold-tree's root split (`est.tree_.feature[0]`, `est.tree_.threshold[0]`) and its `export_text`. This shows whether the 320-row trees agree with the 400-row tree (L8 p.40: the greedy first split can flip under resampling).

**(iii) Lecture grounding.** Reading `plot_tree` output (split rule, gini, samples, value=[n_no, n_yes]): L8 p.25, p.33. Regions and rectangles: L8 p.8, p.34. Leaf probabilities: L8 p.17. Refit at the chosen tuning value: L5 p.36. Dummy reading of `Gender <= 0.5` = Female: L3 p.45 and exam cell 4. Audience framing: L1 p.81 (framing), L6 p.57 ("the decision rule"). Readable tree vs accurate tree: L8 p.39.

**(iv) Assumptions.**
- A1. Thresholds are sklearn midpoints between observed values, and the data are integers or multiples of 1,000. So `Age <= 42.5` reads as "42 or younger", and `EstimatedSalary <= 90500` reads as "$90,000 or less" (data profile §A: Age is an integer, salaries are multiples of 1,000).
- A2. If k* = 3, the tree should reproduce L8 p.33 exactly: root `Age <= 42.5` [257,143]; left `EstimatedSalary <= 90500` [239,46] → leaves [232,9] and [7,37]; right leaf [18,97]. That is a check against something known. Any difference must be explained. It should not happen, because the data, `max_leaf_nodes` and Gender coding are the same.

**(v) Pitfall guards.**
- Don't plot a fold-tree or the unpruned tree as "the chosen tree".
- Don't quote the leaf-count accuracy as the model's accuracy.
- Use the same `random_state=7034`.
- Keep feature names on grid predictions.

**(vi) Outputs.**
- **Figure 1.2a:** the dendrogram.
- The `export_text` print.
- **Table 1.2:** leaf, rule in words, n, buyers, purchase rate, predicted class.
- **Figure 1.2b:** the Age–Salary scatter with regions.
- Optional print of the 5 fold-tree root splits.

**(vii) The written answer must cover:**
- [ ] The figure exists and was made with `plot_tree(feature_names=Xa.columns)`.
- [ ] One clause on which data it is fitted on (all 400) and why (size from CV, then refit).
- [ ] **One or two English sentences, no jargon** (no "node", "gini" or "leaf"), with thresholds translated into everyday units and each group's purchase rate. Template if k* = 3, with numbers filled from Table 1.2: "People over 42 mostly bought (about 84 in 100). People 42 or younger bought only if their estimated salary was above $90,000 (about 84 in 100 did); younger people earning $90,000 or less almost never bought (about 4 in 100)." If Gender is absent from the tree, add: "Gender makes no difference to the rule."
- [ ] Optionally one clause on how often the rule is right on people it has not seen (the CV accuracy from 1.1), next to the 64% of "nobody buys".

**(viii) Open questions.**
- Q1.2a: plot the tree fitted on all 400 rows (default) or one CV fold's tree? **Default: all 400**, following L5 p.36 and L8 p.33.
- Q1.2b: include Figure 1.2b (the L8 p.34-style region plot)? **Default: yes**, since it makes the English sentence easy to verify.

---

## 1.3 Random forest (300 trees) and gradient boosting (100 rounds) vs the tree

**(i) Goal.** 5-fold CV accuracy of `RandomForestClassifier(n_estimators=300, random_state=7034)` and `GradientBoostingClassifier(n_estimators=100, random_state=7034)`. Do they beat the 1.1 tree? Report the result whichever way it comes out, and explain why in 2–3 sentences given the size and shape of the data.

**(ii) Method steps.**
1. `res['rf'] = evaluate(RandomForestClassifier(n_estimators=300, random_state=7034), 'RF(300)')`.
2. `res['gb'] = evaluate(GradientBoostingClassifier(n_estimators=100, random_state=7034), 'GB(100)')`.
3. Reuse `res[k_star]`, `res['unpruned']` and the baseline from 1.1: same folds, no recomputation with other folds.
4. For each ensemble, compute `paired(res[m]['hits'], res[k_star]['hits'])`, the fold-by-fold differences `res[m]['folds'] - res[k_star]['folds']`, and the count of folds won, tied and lost.
5. Print the *actually used* hyper-parameters with `model.get_params()`, restricted to the keys in the table below, so the explanation cites real settings.

**Verified sklearn 1.9.1 defaults, which the explanation needs:**

| Model | Setting that matters | Value | Consequence here | Grounding |
|---|---|---|---|---|
| RF | `max_features` | `'sqrt'` → int(√3) = **1 feature per split** (verified `max_features_ == 1`). If the drawn feature cannot split, sklearn inspects others. | Each split's variable is essentially random among 3, including Gender, which strongly decorrelates the trees but costs bias per tree. | L8 p.43 (√p for classification; "price: a little bias per tree") |
| RF | `max_depth=None`, `min_samples_leaf=1`, `bootstrap=True`, `max_samples=None` (bootstrap of n = 320 per fold) | Deep, unpruned trees | Each tree memorises noise: duplicates and conflicting labels. Only averaging regularises. | L8 p.41–44 ("grown deep and not pruned — the averaging does the regularizing") |
| RF | `n_estimators` | Default is 100; the exam sets **300** | B is a budget, not a tuning parameter | L8 p.45, p.51 |
| RF | prediction | Averages tree probabilities (soft vote; sklearn doc) | L8 p.42 says "vote". Mention it only as an implementation detail. | L8 p.42 |
| GB | `learning_rate` | **0.1** | Untuned ν; L8 used 0.5/0.1/0.05/0.01 and chose by validation | L8 p.50, p.52 |
| GB | `max_depth` | **3** (up to 8 leaves per tree) | Deeper than L8's "shallow, e.g. L = 1" | L8 p.50 |
| GB | `n_estimators` | **100** (exam) | "One hundred rounds at the default rate" lost to RF out of the box in L8 | L8 p.53–54 |
| GB | `subsample=1.0`, `n_iter_no_change=None`, `loss='log_loss'`, `init` = prior log-odds | No stochastic boosting, no early stopping | Each tree fits the residual y − p̂ on the log-odds scale | L8 p.50 teaches the residual-fitting recipe for regression. The classification version (log-loss pseudo-residuals) is **not on the slides**: it is the exam-mandated estimator (exam cell 11), explained by analogy only. |

**(iii) Lecture grounding.**
- Shared folds: L8 p.53.
- RF mechanics and variance reduction: L8 p.40–45.
- Boosting mechanics and tuning need: L8 p.50–52.
- Out-of-box GB can lose; RF nearly tuning-free: L8 p.54–55, p.61.
- A simple boundary suffices when the structure is simple; flexible methods pay off only when the boundary is complex and data are plentiful: L6 p.37–40.
- ML earns its keep with many correlated variables or unknown nonlinear form: L1 p.34. Here n = 400 and p = 3.
- Report it either way: L5 p.57 ("a negative R² is a result"; same spirit) and L8 p.55 ("state it precisely").
- Parsimony if within noise: L5 p.25.

**(iv) Assumptions.**
- A1. "The tree from 1.1" means the CV-chosen k*-leaf tree. The unpruned tree is shown as a second reference.
- A2. The ensembles are **not tuned**: the exam fixes 300 trees and 100 rounds. Tuning ν, B and depth on the same folds and then reporting on them would be optimistic (L8 p.55 "a tuning budget is a leakage risk").
- A3. Comparison fairness is asymmetric. The tree's number won a 6-way search on these folds; the ensembles' numbers did not. Mention this in the answer (L4 p.43).

**(v) Pitfall guards.**
- Same `cv5`, enforced by the `evaluate` assert.
- Seeds on both ensembles.
- Don't pass `criterion` to GB (deprecated in 1.9).
- Don't read ensemble training accuracy, which will be near 1.0 for RF, as evidence. It serves only as an in-sample > out-of-sample check (L8 p.47; L6 p.54).
- `n_jobs` left at default: it does not change results with a fixed seed. Runtime is small (300 trees × 5 folds × 320 rows).

**(vi) Outputs.**
- **Table 1.3**, rows: nobody purchases | unpruned tree | k*-leaf tree | RF(300) | GB(100). Columns: `CV acc`, `n_correct/400`, `fold sd`, `fold min–max`, the 5 fold scores, `SE`, `diff vs k*-tree (pp) ± paired SE`, `folds won/tied/lost vs tree`, `mean train-fold acc`.
- **Figure 1.3:** bar chart of CV accuracy with ±1 SE error bars, a dashed baseline line at 0.6425, and the 5 fold scores overlaid as dots so the fold spread is visible.

**(vii) The written answer must cover:**
- [ ] RF and GB 5-fold accuracies (same folds as 1.1).
- [ ] Beat, tie or lose vs the k*-tree, in pp, against the paired SE and fold count. Also vs the baseline. Reported **whichever way it comes out**.
- [ ] Two or three sentences of *why*, built from these facts, and not claiming more than the numbers show:
  - Size: 400 people, 80 per fold, so one person is 1.25 pp.
  - Shape: 3 inputs, of which effectively 2 matter (L8 p.33–34). The boundary is close to one axis-aligned rectangle, which a 3-leaf tree already draws. So the small tree has little bias left, and its variance is small too.
  - Ensembles mainly remove variance (L8 p.40, p.43). The RF's decorrelation lever (1 of 3 features per split) mostly forces splits on the noise variable (Gender) and on tiny salary bands, with deep unpruned trees.
  - GB runs at untuned defaults (ν = 0.1, depth 3, 100 rounds; L8 p.52–54).
  - Where ensembles won in L8 (20,640 × 9 California; L8 p.54), the signal was complex and data plentiful. Neither holds here.
- [ ] If the differences are within about 2 paired SE: prefer the tree for parsimony and readability (L5 p.25; L8 p.39).

**(viii) Open questions.**
- Q1.3a: also show a tuned GB (for example ν and B chosen on inner folds)? **Default: no**, because it is outside the question and risks optimism (L8 p.55).
- Q1.3b: report the paired SE (this plan) or only fold spreads? **Default: both.**

---

## 1.4 MDI (in sample) vs permutation importance (held out)

**(i) Goal.**
- Split the data 70/30 (`random_state=7034`, `stratify=ya`) and fit RF(300) on the 70%.
- Rank features by `feature_importances_` (MDI, in sample) and by `permutation_importance` on the held-out 30%, and show them side by side.
- Do the rankings agree on the top variable? Where they disagree, why? Which would you show a decision maker?

**(ii) Method steps.**
1. Split and fit:
   ```python
   X_tr, X_te, y_tr, y_te = train_test_split(Xa, ya, test_size=0.3, random_state=7034, stratify=ya)
   assert X_tr.shape == (280, 3) and X_te.shape == (120, 3)
   assert y_tr.value_counts().to_dict() == {0: 180, 1: 100}   # allocation verified with identical class counts
   assert y_te.value_counts().to_dict() == {0: 77, 1: 43}
   rf14 = RandomForestClassifier(n_estimators=300, random_state=7034).fit(X_tr, y_tr)
   ```
2. Sanity: training accuracy `rf14.score(X_tr, y_tr)` vs held-out `rf14.score(X_te, y_te)` vs the held-out rate of the training-majority rule `(y_te == 0).mean()` (77/120). Expect in-sample > held-out (AI guide §6.4; L6 p.54).
3. MDI: `mdi = pd.Series(rf14.feature_importances_, index=Xa.columns)`, and assert that it sums to 1. This is the in-sample normalised mean decrease in Gini impurity, computed on each tree's bootstrap of the 280 training rows (L8 p.47, p.56).
4. Permutation on the held-out rows:
   ```python
   pi_te = permutation_importance(rf14, X_te, y_te, scoring='accuracy', n_repeats=50, random_state=7034)
   ```
   `importances` has shape (3, 50); `importances_mean` and `importances_std` have shape (3,). Each value is the drop in held-out accuracy when that one column is shuffled, averaged over 50 shuffles (L8 p.57: "repeat the shuffle several times and average").
   - Why 50 rather than the default 5: 120 rows make accuracy move in 0.83 pp steps, so single shuffles are noisy, and 50 repeats cost seconds.
5. Diagnostic (default ON, labelled "diagnostic, not for decisions"): the same call on `X_tr, y_tr`, giving `pi_tr`, an in-sample permutation. It separates two reasons for disagreement: in-sample vs held-out rows, and impurity-based vs accuracy-based. Grounding: L8 p.56–57 method; L4 p.44–46 IS vs OOS.
6. Mechanism diagnostics, taken from the fitted forest (descriptive):
   - splits per feature across the 300 trees: `sum((t.tree_.feature == j).sum() for t in rf14.estimators_)` for j = 0, 1, 2;
   - mean leaves per tree: `np.mean([t.get_n_leaves() for t in rf14.estimators_])`;
   - `Xa.nunique()` and `corr(Age, Salary)` from step 0.
   These test the L8 p.58 explanations (many places to split; correlated columns share credit) instead of asserting them.
7. Side-by-side table, built with a single `pd.DataFrame({...}, index=Xa.columns)`; ranks use `.rank(ascending=False, method='min').astype(int)`:
   - `MDI share (in-sample)`, `MDI rank`
   - `Perm. drop in held-out accuracy (pp)` = 100·mean, `sd over 50 shuffles (pp)`, `Perm. rank`
   - `Perm. share` = max(mean, 0)/Σ max(mean, 0). This is normalised like L8 p.58; negatives are clipped to 0 and the clipping is stated.
   - `[diagnostic] Perm. drop on training rows (pp)`
   - `splits in forest`, `distinct values`
8. Figure 1.4a, like L8 p.58: horizontal grouped bars of "share of total importance", MDI (in-sample) vs permutation (held-out), with features ordered by permutation. Figure 1.4b: boxplot of the 50 held-out permutation drops per feature (pp), with a vertical line at 0.

**(iii) Lecture grounding.**
- `feature_importances_` is in-sample MDI and "not an out-of-sample statistic": L8 p.47, p.56.
- Permutation on held-out data, repeated and averaged; "answers the question you actually asked": L8 p.57.
- Rankings disagree; MDI "rewards a column for merely offering many places to split" and splits credit across correlated columns: L8 p.58.
- Importance has no sign; partial dependence gives direction; not causal: L8 p.59–60.
- The 70/30 split is a "traditional split": L5 p.27. A random split is fine for a cross-section: L8 p.45.
- The held-out metric is the relevant one: L6 p.54; L4 p.43–46.
- Correlated or confounded inputs are not marginal effects: L3 p.57.

**(iv) Assumptions.**
- A1. `scoring='accuracy'`, consistent with the problem's metric. Accuracy is coarse, because a feature can sharpen probabilities without flipping many 0.5-threshold decisions. That is itself one reason MDI (a probability- or impurity-level measure) and permutation (a decision-level measure) can disagree.
- A2. The permutation sd measures shuffle noise only, not the sampling noise of *which* 120 people were held out. There is one split; L6 p.18 says results would differ with another.
- A3. "In sample" MDI means on the rows the forest was fit to: each tree's bootstrap of the 280 training rows.

**(v) Pitfall guards.**
- The split happens before any fitting, and nothing is fitted on `X_te`.
- `permutation_importance` receives the held-out data, not `Xa` and not `X_tr` (except the labelled diagnostic).
- `stratify=ya` and `random_state=7034` are exactly as the exam says.
- Rule (a) is not violated because the data are a cross-section (L8 p.45); say so in one clause.
- Seeds are on both the forest and the permutations.

**(vi) Outputs.**
- **Table 1.4** (as above).
- **Figures 1.4a and 1.4b.**
- The printed train, held-out and baseline accuracies.
- The printed split counts per feature.

**(vii) The written answer must cover:**
- [ ] The split (280/120, stratified, seed) and the held-out accuracy vs the 64% majority rate. This checks the forest is worth interpreting.
- [ ] Side-by-side rankings (Table 1.4 and Figure 1.4a).
- [ ] **Agree on the top variable? yes/no**, stated explicitly.
- [ ] Where they disagree, why. These are hypotheses to be confirmed with the diagnostics, not asserted in advance:
  - MDI is computed on training rows by deep trees that also fit noise, and every noise split's impurity drop is credited to some feature (L8 p.47, p.56).
  - EstimatedSalary has 117 distinct values, Age 43 and Gender 2. So Salary offers the most places to split and absorbs most noise-fitting credit (L8 p.58); check the splits-per-feature column.
  - With `max_features=1`, trees are forced to split on Gender whenever it is the only feature drawn, so Gender gets MDI mass that the held-out permutation shows is worth about 0.
  - If |corr(Age, Salary)| is small, credit-splitting between correlated columns is *not* the main mechanism; say so.
  - Impurity vs accuracy scale (A1).
- [ ] Which one to show a decision maker: **held-out permutation importance**. It measures what the model loses on people it has not seen, "the question you actually asked" (L8 p.57); MDI is an in-sample statistic (L8 p.47). Three caveats:
  - it is noisy with 120 people (show the spread);
  - it says which variable matters, not which way (pair it with a partial-dependence plot if direction is needed, L8 p.60);
  - it is what the model uses, not a causal effect (L8 p.59).

**(viii) Open questions.**
- Q1.4a: `n_repeats`? **Default: 50** (the sklearn default of 5 is too few for 120 rows).
- Q1.4b: add `scoring='roc_auc'` as a robustness column? AUC is taught in L6 p.52–53. **Default: no**; mention it only if the accuracy-based ranking is ambiguous.
- Q1.4c: keep the in-sample permutation diagnostic column? **Default: yes**, clearly labelled.
- Q1.4d: add partial-dependence plots for Age and Salary with the L8 p.59 brute-force function? That means averaging `predict_proba[:, 1]` over a 40-point grid from the 5th to the 95th percentile. **Default: no**; mention it in words only.

---

## 1.5 (Open-ended) Why shuffled CV is wrong for monthly data on one stock

No code by default; cell 18 gets `# No code: see the answer below.` Below is the answer outline, with every claim grounded.

**(i) Goal.** Each row becomes a month of one stock and the target is 1{next month's return > 0}, a binary "greater or less than" target (L6 p.3). Explain precisely what goes wrong with `StratifiedKFold(shuffle=True)`, what to do instead, and which way the reported accuracy is biased.

**(ii)–(iii) Answer outline, point by point with its grounding.**

1. **What CV is supposed to estimate.** It should estimate the accuracy of a model *used as it would be deployed*: trained on the past, predicting months it has not lived through (L6 p.30: "as it would be if the model were put to use"; L6 p.54). K-fold as taught is for **i.i.d. data** (L5 p.28), and it is legitimate for the ads data only because the rows are unrelated people (L8 p.45).
2. **What goes wrong: look-ahead.** Shuffled folds put months *after* each test month into its training set. The model is scored on "predicting" 2015 after training on 2021 (AI guide §4(a); L1 p.82 "watch whether the future ends up in the training set. It usually does"; L5 p.56 "the random fold sees the future"). The course grades this as **fatal** (L5 p.58 #3; L5 p.59 on `train_test_split`'s `shuffle=True` default).
3. **What goes wrong: dependence between neighbouring rows.** Monthly rows are not exchangeable:
   - Predictors built from trailing windows are highly persistent: a 12-month past return shares 11 of its 12 months with the next row's.
   - Volatility and up/down regimes cluster ("calm months and panics do not have the same variance", L2 p.80).
   - Nonstationarity changes the base rate over time (L6 p.30: 19.6% → 17.0%; L1 p.34).

   Under shuffling, each test month has near-identical neighbours, months t−1 and t+1 with similar features and a correlated outcome, sitting in the training folds. A flexible tree can simply look up its neighbour's answer, which is interpolation within a history it has mostly seen, not forecasting. The iid premise behind the CV average and its fold spread (L2 p.63) fails, so even the reported uncertainty is too small.
4. **Removing `shuffle` alone does not fix it.** `KFold(shuffle=False)` still trains folds 1…K−1 on *later* blocks. Stratifying by up/down months is irrelevant to the time problem. The requirement is that training always precedes validation and test (L5 p.49).
5. **Consequence for tuning.** Validation error looks too good for the most complex settings, so CV *under-regularises*. For a tree that means too many leaves. The model it picks is worse on the true future. The course's own measurement on the same LASSO and panel: random 5-fold picked 23 variables vs 15 for the expanding window and lost 0.036 of R²_OOS (L5 p.56).
6. **Direction: the reported accuracy is biased upward (optimistic).** It overstates what the rule would achieve going forward, because the leak and the neighbour look-up only ever *add* information that a real forecaster would not have. Be precise: the size of the bias depends on how persistent the predictors and regimes are. With truly iid months and no persistent features, the shuffled estimate would be roughly unbiased; monthly stock data are not like that. Sanity anchor: return signs have low signal-to-noise (L1 p.33), so an honest forward accuracy near the base rate is normal, and a much larger number on financial data is "a prompt to go looking for the leak" (L5 p.57).
7. **What to do instead.** Time-ordered, forward-only evaluation (L5 p.48–52):
   - **Expanding window:** fixed start, growing training block. The course iterator is `expanding_window_iterator(data, initial_train_size, validation_size, test_size, step)` (L5 p.50), with, e.g., 180/84/12/12 months (L5 p.51) or smaller for a shorter single-stock history. Alternatively a **rolling window** of fixed length if the stock's behaviour drifts (L5 p.48). Default: expanding, because L5 p.50 calls it "more reliable".
   - Order strictly **train → validation → test**. Choose `max_leaf_nodes` (or any hyper-parameter) on the validation block only, predict the next test block, roll forward, and pool the test predictions (L5 p.49, p.52). Keep a final test period untouched until the end (L4 p.43; L8 p.55 "report on a test split you touched once").
   - **Predictors known at the end of month t−1**: lag every feature (exam cell 36, rule 2; exam cell 34 lag conventions; merge on publication date, L5 p.58 #2).
   - Any scaler or imputer is fitted inside each training window only, as a `Pipeline` refit per window (L5 p.58 #1; L6 p.30; AI guide §4(b)). A tree needs none.
   - **Benchmark fixed in advance.** Use the majority direction ("always up" or "always down") learned from the *training* months, re-learned as the window expands, and evaluated on the test months. Report it next to the model's accuracy (L6 p.57; L4 p.46 "why not out-of-sample Ȳ?"; this mirrors the exam's trailing-mean rule, cell 36).
   - Report how many configurations were tried (L5 p.57).
8. **Gap / embargo. OUTSIDE SCOPE: proposed only if the user approves.** No lecture teaches it. If targets overlap, for example predicting a 12-month-ahead return with monthly rows, or if a feature window contains the target month, leave a gap of (overlap − 1) months between each training block and its validation or test block. Otherwise neighbouring rows share the same future returns across the boundary. sklearn's `TimeSeriesSplit(n_splits, test_size=..., gap=...)` implements this; it is a library convenience, not on the slides. For the question as posed (a one-month-ahead sign, with predictors dated ≤ t−1), consecutive targets do not overlap and **no gap is needed**. Say this in one sentence at most, flagged as an extension.

**(iv) Assumptions.** A single stock gives a few hundred months at most (L1 p.34: "100s of months"), so the validation and test blocks must be sized to leave enough training data (L5 p.27: small datasets, don't over-allocate validation).

**(v) Pitfall guards.** Don't propose `train_test_split(..., shuffle=True)`, `KFold(shuffle=False)` or `StratifiedKFold` for time data. Don't benchmark against the test-period majority rate. Don't claim a gap is course material.

**(vi) Outputs.** Text only. Optionally, a 5-row schematic table of expanding-window folds (train months / validation months / test months). No computed numbers.

**(vii) The written answer must cover:**
- [ ] *Precisely* what goes wrong: look-ahead (future in training); leakage through persistent, overlapping neighbouring months and regimes (non-iid); CV then under-regularises; removing the shuffle alone is not enough.
- [ ] What to do instead: an expanding (or rolling) walk-forward window; train → validation → test; hyper-parameters on validation only; test touched once; predictors lagged to t−1; per-window preprocessing; a training-period majority benchmark.
- [ ] Which way: **upward / optimistic**, and why (information a real forecaster lacks). A one-line magnitude anchor (L5 p.56; L5 p.57 sanity check).
- [ ] (Optional, flagged) the gap when targets overlap.

**(viii) Open questions.**
- Q1.5a: include a small simulated demonstration in cell 18 (e.g. a persistent AR feature, shuffled vs expanding CV)? **Default: no.** The exam says "if any", and it adds risk without adding marks.
- Q1.5b: mention the gap/embargo at all? **Default: one flagged sentence**, conditional on overlapping targets.

---

## 2. Cell-by-cell layout

| Cell | Content |
|---|---|
| 4 (given) | Unchanged. If running locally, switch to the commented `pd.read_csv('Social_Network_Ads.csv')` line or set `_DATA_DIR` to the local folder, because `/classes/41210_MiF_fall2026/Data/` does not exist here (verified). Restore the class-server path before submission. |
| 6 | Extra imports (`cross_validate`, `export_text`); step-0 description; `evaluate`/`paired` helpers; baseline; unpruned tree; sweep; `k_star`; Table 1.1; Figure 1.1. |
| 7 | Answer 1.1 (checklist above). |
| 9 | `tree_star` fit on 400; Figure 1.2a; `export_text`; Table 1.2; Figure 1.2b; fold-tree root splits. |
| 10 | Answer 1.2: 1–2 plain sentences plus one clause on the fitting data. |
| 12 | RF and GB via `evaluate`; defaults printout; Table 1.3; Figure 1.3. |
| 13 | Answer 1.3. |
| 15 | Split; `rf14`; accuracies; MDI; `pi_te` (and `pi_tr` diagnostic); forest split counts; Table 1.4; Figures 1.4a/b. |
| 16 | Answer 1.4. |
| 18 | `# No code: see the answer below.` |
| 19 | Answer 1.5. |

Order of checks before writing any answer:
1. Row and class counts match the profile.
2. Baseline = 0.6425 both ways.
3. `evaluate` asserts pass.
4. The unpruned in-sample accuracy equals the conflicting-cell ceiling.
5. If k* = 3, the tree matches L8 p.33.
6. In-sample > CV for every tree size and for the ensembles.
7. MDI sums to 1.
8. `pi_te.importances` has shape (3, 50).
9. The top-ranked feature is consistent with the tree in 1.2: a feature absent from the chosen tree should not top the held-out permutation ranking. If it does, investigate before writing.

## 3. Consolidated open questions (defaults in bold)

1. Data path for local runs: **use the local CSV only for development; submit with the class path.**
2. Q1.1a tie tolerance: **exact `n_correct` equality.** Q1.1b unpruned as a candidate: **no.**
3. Q1.2a tree fitted on: **all 400 rows.** Q1.2b region plot: **yes.**
4. Q1.3a tuned GB: **no.** Q1.3b paired SE plus fold spread: **both.**
5. Q1.4a `n_repeats`: **50.** Q1.4b AUC column: **no.** Q1.4c in-sample permutation diagnostic: **yes.** Q1.4d PD plots: **no.**
6. Q1.5a simulation demo: **no.** Q1.5b gap sentence: **one flagged sentence.**

## 4. Scope notes (what is deliberately NOT used)

- **Not taught, so not used:** `DummyClassifier`; McNemar's test; nested CV; the 1-SE rule (it appears only in the L5 p.46 legend); OOB error; SHAP; XGBoost/LightGBM; `class_weight`; entropy criterion; calibration.
- **Exam-defined rather than lecture-defined:** stratified 5-fold with seed 7034 (exam cells 3–4); `GradientBoostingClassifier`, whose classification loss is not on the slides (L8 p.50 shows only the regression recipe; exam cell 11 mandates it); the 70/30 stratified split (exam cell 14).
- **Outside scope, only with the user's approval:** the gap/embargo in 1.5, and `TimeSeriesSplit` as a convenience wrapper. The course tool is L5 p.50's iterator.
