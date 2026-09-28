# Lecture 8 — Trees, Random Forests, and Boosted Trees (BUSN 41210, Dacheng Xiu, Sep 23 2026)

Source: `/home/user/UChicago/Lecture_8.pdf` (62 PDF pages; beamer frame counter runs "1/58" to "58/58").
All 62 pages were viewed rendered (not only text-extracted). Formulas, tables, tree dendrograms and charts below are transcribed from the rendered slides.

**Citation convention:** "L8 p.X" = PDF page X. Mapping PDF page -> printed frame number:
p.1–8 = frames 1–8; p.9–13 = frame 9 (five overlay builds of one illustration); p.k = frame k−4 for k ≥ 14 (e.g. p.33 = frame 29, p.62 = frame 58).

---

## 0. One-paragraph summary (what L8 actually teaches)

L8 teaches (Part I) a single CART tree — classification and regression — grown greedily and recursively by minimizing deviance/impurity, stopped by a minimum leaf size, and pruned by **minimal cost-complexity** $R_\alpha(T)=R(T)+\alpha|T|$ with α chosen **out of sample** (test sample / 5-fold CV). It then argues (Part II) the problem with one tree is **variance, not bias**, and fixes it with **bagging → random forest** (bootstrap rows + random column subset at each split; average/vote; B is a budget, not a tuning parameter) and **gradient boosting** (shallow trees fit sequentially to residuals, shrunk by ν; L, ν, B tuned on validation; more rounds eventually hurt). Model comparison is done on a **shared 5-fold split with R² benchmarked against the training-fold mean**. Interpretation is recovered with **permutation importance on held-out data** (preferred) vs sklearn's **MDI `feature_importances_` (in-sample)**, and with **partial dependence** (brute-force definition, gives the sign/shape). Worked datasets: umbrella toy, 500-point 2-D toy, NBC TV pilots, **Social Network Ads (the exact Problem-1 dataset)**, motorcycle crash-test, California housing.

---

## 1. Slide-by-slide notes

### L8 p.1–2 — Title and roadmap
- p.1: "Lecture 8: Trees, Random Forests, and Boosted Trees", Dacheng Xiu, Chicago Booth, September 23, 2026.
- p.2 roadmap:
  - **Part I — one tree:** Classification And Regression Trees (CART): using tree-logic to make predictions; Growing a tree: deviance, impurity, and the greedy split; **Pruning: cost complexity, and choosing α out of sample**; Nonlinearity and interaction, for free.
  - **Part II — many trees:** Bagging, and the one extra trick that makes it a random forest; Boosting: fit the residual, shrink, repeat; Which model actually wins, and opening the black box.

### L8 p.3–4 — What a decision tree is
- p.3 (umbrella tree): Wake Up → ">70% Rain" → Umbrella; "<70% Rain" → ">30% Rain" → {Cloudy → Umbrella, Sunny → No Umbrella}; "<30% Rain" → No Umbrella. "Tree-logic uses a series of steps to come to a conclusion. The trick is to have mini-decisions combine for good choices. Each decision is a node, and the final prediction is a **leaf node**."
- p.4 "Decision Trees are Like a Game of Mousetrap": drop covariates **x** in at the top, each decision node bounces you left or right; you end in a leaf node = data subset defined by the splits. Diagram: split at $x_i=0$: right child $\{x: x_i>0\}$; left child split at $x_j=2$ into $\{x: x_i\le 0, x_j\le 2\}$ and $\{x: x_i\le 0, x_j>2\}$.

### L8 p.5 — Tree-based statistical learning
- Predict y from $x=(x_1,\dots,x_p)'$ by **dividing the feature space into small regions where outcomes are more similar**; within each region fit a very simple model locally.
- Works for categorical y (classification) and continuous y (regression).
- Regions come from **successive binary splits**: choose a variable $x_j$, split into $x_j\le c$ and $x_j>c$; repeat on each half.

### L8 p.6 — Estimation of decision trees
- "As usual, we'll **maximize data likelihood (minimize deviance)**."
- Two likelihood types: classification and regression trees. A covariate **x** dictates the path to a leaf.
- **Classification trees have class probabilities at the leaves** (e.g. P(heavy rain)=0.9).
- **Regression trees have a mean response at the leaves** (e.g. expected rain = 2in).

### L8 p.7 — Tree deviances (formulas)
- Regression deviance: $\sum_{i=1}^n (y_i-\hat y_i)^2$
- Classification deviance: $-\sum_{i=1}^n \log(\hat p_{y_i})$, where $\hat p_{y_i}=\hat p(y_i\mid x_i)$ (i.e. multinomial negative log-likelihood / cross-entropy).
- "It is also common to use **Gini Deviance**, $-\sum_{i=1}^n \hat p_{y_i}(1-\hat p_{y_i})$, a measure of multinomial variance." (Transcribed exactly as on slide, including the leading minus sign. Note: the node-level Gini **impurity** sklearn prints in `plot_tree` is $G=\sum_k \hat p_k(1-\hat p_k) = 1-\sum_k \hat p_k^2$ — see p.25/p.30/p.33 where e.g. root [257,143] has gini 0.459 = $1-(257/400)^2-(143/400)^2$.)
- Instead of $x'\beta$, $\hat p$ and $\hat y$ are functions of x passed through decision nodes.
- Need to estimate the sequence of decisions (how many? what order?) — a **huge** set of possible tree configurations.

### L8 p.8 — Classification trees
- Popular because interpretable (mimic how decisions are made).
- $y_i\in\{1,\dots,K\}$ class labels, $x_i\in\mathbb R^p$.
- Tree defines m regions (rectangles) $R_1,\dots,R_m$, one per leaf.
- Assign each $R_j$ a class label $c_j\in\{1,\dots,K\}$ (typically the most dominant class within the region); classify a new x as $c_j$ if $x\in R_j$.

### L8 p.9–13 — Classification Trees: An Illustration (one frame, 5 overlay builds)
- Progressive partition of $(X_1,X_2)$ plane: split $X_1\le c_1$?; then on the "No" side $X_2\le c_2$?; then $X_2\le c_3$?; then on the "Yes" side $X_1\le c_4$?. Final tree (p.13): root $X_1\le c_1$; left: $X_1\le c_4$ (Yes/No leaves); right: $X_2\le c_2$ → (Yes → $X_2\le c_3$ → Yes/No leaves; No leaf). Shows each split is axis-parallel and applies only within the current region.

### L8 p.14–15 — Toy example
- p.14: n = 500 points in p = 2 dims ($X_1,X_2\in[0,1]$), classes 0 (black) and 1 (red). Red points cluster in top-left block and a right-middle block. "Does dividing up the feature space into rectangles look like it would work here?"
- p.15: fitted partition drawn: vertical splits at $X_1\approx0.4$ and $0.6$, horizontal splits at $X_2\approx0.6$ (left part), $X_2\approx0.5$ (right part), $X_2\approx0.1$ (bottom) → rectangles isolate the red blocks (top-left $X_1<0.4, X_2>0.6$; right $X_1>0.6$, $0.1<X_2<0.5$).

### L8 p.16 — "Other Regions Not Allowed"
- Picture of a rectangular partition of $(X_1,X_2)$ with overlapping/pinwheel rectangles that **cannot be produced by recursive binary splitting**. Trees only produce nested, recursive axis-aligned partitions.

### L8 p.17 — Predicted class probabilities (formula)
- Region $R_j$ contains $n_j$ training points. For class k:
  $$\hat P(C=k\mid X\in R_j)=\frac{\#\{y_i: x_i\in R_j \text{ and } y_i=k\}}{\#\{y_i: x_i\in R_j\}}$$
  (slide writes "$\#y_i: y_i\in R_j$ and $y_i=k$" over "$\#y_i: y_i\in R_j$") = the proportion of points in the region of class k.
- Predicted class: $\hat c_j=\arg\max_{k=1,\dots,K}\hat P(C=k\mid X\in R_j)$ — the most commonly occurring class.
- (This is what `predict_proba` / `predict` return for a sklearn classification tree.)

### L8 p.18 — Classification trees and competitors (table)
| | Model assumptions? | Estimated probabilities? | Interpretable? | Flexible? |
|---|---|---|---|---|
| LR (logistic regression) | Yes | Yes | Yes | No |
| k-NN | No | No | No | Yes |
| Trees | No | Yes | Yes | Somewhat |

| | Predicts well? |
|---|---|
| LR | Depends on X |
| k-NN | If properly tuned |
| Trees | ? |

### L8 p.19–20 — Regression trees (formulas)
- p.19: continuous outcome — fit a mean rather than a proportion in each rectangle. Figure (ESL-style): tree $X_1\le t_1$; left $X_2\le t_2$ → $R_1,R_2$; right $X_1\le t_3$ → $R_3$ and $X_2\le t_4$ → $R_4,R_5$; plus 3-D piecewise-constant surface over $(X_1,X_2)$.
- p.20: 
  $$E[y\mid x]=\sum_{j=1}^m c_j\cdot \mathbb 1\{x\in R_j\} = c_j \text{ such that } x\in R_j,$$
  $c_j$ now real numbers, chosen as the region average $$c_j=\frac1{n_j}\sum_{x_i\in R_j} y_i.$$
- "The main difference in building the tree is that we use **sums of squares** instead of misclassification error (or Gini index or deviance) to decide which region to split."

### L8 p.21 — How to build trees?
- Two issues: **how to choose the splits?** and **how big to grow the tree?**
- Depth tradeoff: big tree = more complex; "What tradeoff is at play here? How might we eventually consider choosing the depth?"
- For depth d, about $2^d$ nodes, each could use any of p variables → number of possibilities $$p\cdot 2^d$$ — "huge even for moderate d! And we haven't even counted the actual split points themselves." (Motivates greedy search.)

### L8 p.22 — CART algorithm (recursive and greedy)
- "We estimate decision trees by being **recursive and greedy**."
- Given any node, find the **optimal (error-minimizing) split** and divide into two child sets; repeat on each child; children become parents...
- **Stopping:** stop splitting when leaf size hits a minimum threshold ("e.g., say **no less than 10 obsv per leaf**"). "Often there are also **minimum deviance improvement thresholds**." (sklearn equivalents, not shown on slide: `min_samples_leaf`, `min_impurity_decrease`.)

### L8 p.23 — NBC example (dataset)
- NBC TV pilots: **6241 views and 20 questions for 40 shows**. Goal: predict **engagement**.
- Ratings: **GRP** = gross ratings points (estimated total viewership).
- **Projected Engagement (PE):** viewer quizzed on order/detail after watching → engagement with show (and ads).
- Viewer demographics: percent of viewership per show in categories by region, race, how the household consumes TV.

### L8 p.24 — CODE: DecisionTreeClassifier
```python
from sklearn import tree
clf = tree.DecisionTreeClassifier()
clf = clf.fit(X, Y)
```
"As usual, you can plot, and predict the tree." Classification tree to predict **genre from demographics**:
```python
X = demos.iloc[:,1:]
y = nbc.Genre
clf = tree.DecisionTreeClassifier()
clf = clf.fit(X, Y)      # (slide mixes y / Y)
```
Note: default call = **unpruned** tree (no max_depth / max_leaf_nodes / ccp_alpha).

### L8 p.25 — Dendrogram of the NBC genre tree (plot_tree output)
- Root: `WIRED.CABLE.W.O.PAY <= 28.665`, gini 0.584, samples 40, value [19, 17, 4] (3 genres).
  - Left: `VCR.OWNER <= 83.749`, gini 0.43, samples 22, value [16, 2, 4]
    - `TERRITORY.PACIFIC <= 18.871`, gini 0.48, 5, [0,2,3] → leaves [0,0,3] (3 obs) and [0,2,0] (2 obs), gini 0
    - `X2.PERSONS <= 28.802`, gini 0.111, 17, [16,0,1] → leaves [16,0,0] and [0,0,1], gini 0
  - Right: `BLACK <= 17.202`, gini 0.278, 18, [3,15,0] → leaves [0,15,0] and [3,0,0], gini 0
- All leaves pure (gini 0.0), some with 1 observation → classic unpruned overfit.
- CODE: "To get the dendrogram, do `tree.export_text(clf)` then `plot_tree(clf)`." (slide typo "expor_text").

### L8 p.26 — CODE: regression tree with dummies
- Predict engagement from ratings and genre; split on genre by turning it into numeric dummy variables:
```python
X = pd.concat([nbc.GRP, pd.get_dummies(nbc.Genre)], axis=1)
y = nbc.PE
clf = tree.DecisionTreeRegressor().fit(X, y)
```
- "Instead of genre, leaf predictions are expected engagement."
- House style: categorical → `pd.get_dummies`, concatenate with `axis=1`.

### L8 p.27 — NBC show engagement tree (regression dendrogram + fit plot)
- Root `GRP <= 223.05`, mse 141.161, samples 40, value 72.683.
  - Left `GRP <= 12.4` (mse 216.103, 7, 56.637) → leaf (1 obs, 30.0) and `Reality <= 0.5` (mse 114.16, 6, 61.076) → leaves (1, 81.662), (5, 56.959, mse 35.281).
  - Right `Reality <= 0.5` (mse 59.06, 33, 76.087) → `Drama/Adventure <= 0.5` (mse 28.295, 22, 78.848) → leaves (3, 84.174, mse 3.369), (19, 78.007, mse 27.043); and `GRP <= 433.85` (mse 74.858, 11, 70.565) → leaves (3, 63.13, mse 48.34), (8, 73.354, mse 56.294).
- Right panel: PE vs GRP (0–2500) coloured by genre (Reality, Drama/Adventure, Situation Comedy), step-function fits per genre.
- "**Nonlinear: PE increases with GRP, but in jumps.** Follow how the tree translates into changing E[PE]."
- (Node label "mse" = older sklearn; current sklearn prints "squared_error".)

### L8 p.28 — Automatic Interaction Detection
- Different genres are more/less dependent on GRP → interaction.
- AID was an original motivation for trees; older algorithms carry it in their name: **CHAID**, ... (mention only).
- "**nonlinearity and interaction without having to specify it in advance. Moreover, nonconstant variance is no problem.**"
- Such methods are **nonparametric**: no assumed parametric model (e.g. $y=x\beta+\varepsilon$, $\varepsilon\sim N(0,\sigma^2)$).

### L8 p.29 — Pruning via minimal cost-complexity (formula)
- "Biggest challenge with such flexible models is avoiding overfit. For CART, the usual solution is to **rely on out-of-sample prediction**."
- Minimal cost-complexity pruning, "like LASSO", parametrized by complexity parameter α:
  $$R_\alpha(T)=R(T)+\alpha|T|$$
  $R(T)$ = tree's (training) risk/deviance/impurity; $|T|$ = number of terminal nodes (leaves).
- "**Pruning yields candidate trees, and we use a testing sample to choose.**"
- (sklearn implementation, not shown on slide: `DecisionTreeClassifier(ccp_alpha=...)`, `clf.cost_complexity_pruning_path(X, y)` → `ccp_alphas`, `impurities`.)

### L8 p.30 — Example: Social Network Ads (THE PROBLEM-1 DATASET), unpruned tree
- "A categorical dataset to determine whether a user purchased a particular product. **Gender, age and estimated salary** are available to help in predicting purchase behavior. An **overly complex tree fit to 400 clients**." "Do we need all the splits? Is the tree just fitting noise?"
- Top of the fully grown tree (from rendered dendrogram + text):
  - Root `Age <= 42.5`, gini 0.459, samples 400, value [257, 143]  → **majority class = 0 (not purchased), 257/400 = 64.25%**.
  - Left `EstimatedSalary <= 90500.0`, gini 0.271, 285, [239, 46]; right `Age <= 46.5`, gini 0.264, 115, [18, 97].
  - Deeper: splits on Age, EstimatedSalary and `Gender <= 0.5` (Gender coded 0/1) repeatedly, down to many pure leaves of 1–3 obs (depth ≈ 12, ~90+ nodes). Large pure leaves include [162, 0] (young, low salary), [40,0], [0,29], [0,20], [0,18].
- Gender appears only deep in the tree (small nodes).

### L8 p.31 — Complexity as a function of α (charts)
- Left: number of nodes vs α: ~93 nodes at α=0, falling steeply to ~5 nodes by α≈0.007, stays 5 until α≈0.138 (→3 nodes), ≈0.171 (→1 node).
- Right: depth vs α: 12 at α=0 → 10 → 5 → 4 → **2** (α≈0.007 to 0.138) → 1 → 0 (α≈0.171).

### L8 p.32 — Out-of-sample tree pruning accuracy (chart)
- "Accuracy vs alpha for training and testing sets":
  - train: 1.00 at α=0, falling through 0.99, 0.97, 0.96 ... to **0.91** on the plateau (α≈0.007–0.138), 0.82 at α≈0.138, 0.63 at α≈0.171 (root only).
  - test: **0.90 at α=0** (unpruned), 0.92, 0.93, rising to **0.94** on the plateau (5 nodes, depth 2, i.e. 3 leaves), 0.89 at α≈0.138 (1 split), 0.68 at α≈0.171 (root only = majority-class rule on the test set).
- Lesson: training accuracy monotonically falls with α; test accuracy is **hump-shaped** — pruning improves OOS accuracy; too much pruning underfits.
- (Inference, not stated: test accuracies move in 0.01 steps → likely a 100-obs test set, i.e. a 300/100 split. The split used is not specified on the slide.)

### L8 p.33 — CODE: the chosen tree (depth 2 = 3 leaves)
```python
clf = tree.DecisionTreeClassifier(max_leaf_nodes=3).fit(X, Y)
```
- "We thus fit a tree with depth 2". Tree:
  - Root `Age <= 42.5`, gini 0.459, samples 400, value [257,143]
    - True: `EstimatedSalary <= 90500.0`, gini 0.271, 285, [239,46]
      - leaf gini 0.072, 241, [232, 9] → predict 0 (no purchase)
      - leaf gini 0.268, 44, [7, 37] → predict 1 (purchase)
    - False: leaf gini 0.264, 115, [18, 97] → predict 1 (purchase)
- "**CV chooses age and estimated salary as deciding variables.**" (Gender is dropped.)
- Derived: in-sample accuracy = (232+37+97)/400 = 366/400 = **0.915**; plain English: "People 42 or younger with estimated salary ≤ \$90,500 are predicted not to buy; everyone older than 42.5, or younger but earning > \$90,500, is predicted to buy."
- House style: **`max_leaf_nodes` used as the tree-size knob**; α-path → leaf count mapped to `max_leaf_nodes`.

### L8 p.34 — Social Network Ads tree in the data plane
- Scatter of age (≈18–60) vs estimated salary (≈15k–150k), blue = not purchased, orange = purchased; the tree's rectangle {age ≤ 42.5, salary ≤ 90,500} shaded as the "no purchase" region; the rest predicts purchase. "With only 2 relevant inputs, we can plot the data and tree fit."
- Visible: some orange points inside the no-purchase rectangle near age 35–42, and blue points in the purchase region (age 43–50, lower salaries) → irreducible misclassification.

### L8 p.35 — Trees detect nonlinearity (motorcycle data)
- Motorcycle crash-test dummy data: x = time from impact (0–60 ms), y = acceleration on the helmet (≈ −130 to 75). A single regression tree fits a step function capturing the dip (~15–25) and rebound (~30–35).
- "They automatically learn non-linear response functions and will discover interactions between variables."

### L8 p.36 — California housing data (dataset)
- **20,640 census tracts**: latitude/longitude of tract centers; population totals, household counts, median income; average room/bedroom numbers, home age. (Nine predictors; p.53.)
- Goal: predict **log(MedVal)**. "Difficult regression: covariate effects change with location, and how they change is certainly not linear."

### L8 p.37 — Trees detect interaction automatically (numbers)
- Left map: data log(MedVal) over (longitude, latitude); right: a **depth-6 tree, 64 leaves**, using **longitude and latitude only**, "no instruction that the two interact".
- **Five-fold out-of-sample R²: depth-6 tree 0.5580, standardized LASSO 0.3164.**
- "A plane in (lat, long) is not a map of California. The tree finds the interaction because **recursive splitting *is* interaction**."

### L8 p.38 — How big should the tree be? (cost-complexity path, table)
- Left: leaves $|T_\alpha|$ vs α (log–log), "the path: **18,660 distinct α**"; roughly a straight (log-linear) decline from ~10⁴ leaves at α=10⁻⁶ to 1 leaf at α=10⁻¹.
- Right: 5-fold OOS R² vs leaves: rises to a peak then falls; marked "**12 leaves: R² = 0.561**" and "**CV: 211 leaves, R² = 0.722**".
- "Grown out, the tree has **19,826 leaves for 20,640 tracts: it interpolates, in-sample R² = 1.0000**."
- Table:

| α | 0 | 10⁻⁴ | 3×10⁻⁴ | 10⁻³ | 3×10⁻³ | 10⁻² | 10⁻¹ |
|---|---|---|---|---|---|---|---|
| leaves | 19,826 | **211** | 81 | 31 | 13 | 4 | 1 |
| OOS R² | 0.6352 | **0.7220** | 0.6985 | 0.6334 | 0.5571 | 0.4318 | 0.1124 |

- Note the unpruned tree still has OOS R² 0.6352 (well below 0.7220). The table lists 0.1124 for α = 10⁻¹ / "1 leaf"; a constant (1-leaf) predictor should score ≈0 against the training-fold mean, so the leaf count there is probably the full-sample tree's count while per-fold trees keep a split — the slide does not explain it; do not rely on that cell.

### L8 p.39 — What the path is telling you
- **CV picks α = 10⁻⁴: 211 leaves, OOS R² = 0.7220.** "Not the biggest tree, and not a small one either."
- A **twelve-leaf tree** (readable on a slide) scores **0.5610** — legible, but **0.16 of R² worse** than the CV tree.
- "**The readable tree and the accurate tree are not the same tree.** You cannot have both, and which one you want depends on whether the deliverable is a **forecast or an explanation**."
- Tuning: leaves fall roughly **log-linearly in α**, so **a coarse geometric grid is enough**; no need to visit all 18,660 knots.

### L8 p.40 — The problem is not bias, it is variance
- The 211-leaf CV tree (0.7220) "is the best single tree there is, and it is still not good enough."
- A deep tree can approximate almost any function (ours interpolated all 20,640 tracts).
- "What it cannot do is give you the **same function twice**. The greedy first split is a choice between close competitors; if a resample flips it, **every split below it changes**."
- **Prediction error = (bias)² + variance + noise**; for a deep tree the **variance** term dominates.
- "Cross-validation picks the depth that minimizes **average** out-of-sample error. It does nothing about the dispersion *at* that depth — and **with one history you only get one draw**."

### L8 p.41 — From CART to averaging bootstrapped trees
- "Unfortunately, it is tough to avoid overfit with CART: Deep tree structure is so unstable that optimal depth is not easily chosen via cross validation, and there's no theory to fall back on."
- Average over a **bootstrapped** sample of trees: repeatedly re-sample the data **with replacement** to get a 'jittered' dataset of n obs; fit a CART tree to each; predict by the **average** prediction of the forest.
- "Real structure that persists across datasets shows up in the average. Noisy useless signals will average out to have no effect." — "This is a Random Forest" (loosely; p.43 distinguishes bagging).

### L8 p.42 — Random forest algorithm sketch (formulas)
1. Sample B subsets of the data + variables (e.g. observations 1, 5, 20, … and inputs 2, 10, 17, …).
2. Fit a tree to each subset → B fitted trees $\mathcal T_b$. **At each split, sample a subset of candidate variables for splitting.**
3. Average predictions:
   - regression: $$\mathbb E[y\mid \mathbf x]=\frac1B\sum_{b=1}^B \mathcal T_b(\mathbf x)$$
   - classification: let $\{\mathcal T_b(\mathbf x)\}_{b=1}^B$ **vote** on $\hat y$.
- Observation resample usually with replacement → average of bootstrapped trees = **'bagging'**.

### L8 p.43 — Step 2 is the whole difference (bagging vs RF)
- Drop step 2 → **bagging** (bootstrap, fit, average). Keep it → **random forest**.
- **Bagged trees are correlated**: they see nearly the same data, find nearly the same strong first split. "Averaging B estimators cuts variance by about B *only if they are independent*; averaging correlated things buys much less."
- **Starving each split of columns decorrelates them**: offer a random subset of predictors at every split — **√p for classification, p/3 or p for regression** — trees forced to disagree.
- **Price: a little bias per tree** (sometimes the best split was the one you hid). "Bias is small and shared, variance is large and cancels."
- (sklearn mapping, not on slide: `max_features='sqrt'` is RandomForestClassifier default; RandomForestRegressor default `max_features=1.0` = all p, i.e. column-wise bagging. With p = 3, `'sqrt'` gives int(√3) = **1 candidate feature per split**.)

### L8 p.44 — Understanding random forests
- CART in practice: split to lower deviance until leaves hit min size → create candidate trees by pruning back → choose best by CV.
- "**Random Forests avoid the need for CV.**" Each tree not overly complicated (limited variables); predictions not 'optimized to noise' because they are averages over different subsets.
- "Each tree is **grown deep and *not* pruned** — the averaging does the regularizing. In practice you **set B as large as your patience allows, set a minimum leaf size, and stop.**"

### L8 p.45 — What averaging actually buys (California, numbers)
- Chart: held-out R² vs B (1 to 300, log x-axis): forest curve rises from ≈0.69 at B=1 to ≈0.83 and flattens; dashed line "one CV-pruned CART" at ≈0.734; grey band ±1 sd of single trees (≈0.67–0.71).
- "California panel, **70/30 split (14,447 fit, 6,193 held out)** — **a cross-section, so a random split is legitimate**."
- one tree R² = 0.6875 → ten 0.8084 → fifty 0.8225 → **three hundred 0.8268**.
- CV-pruned single CART on the same split: **0.7341**. Averaging buys **+0.0926**.
- The 300 individual trees have held-out R² with **sd 0.0148** about their mean — and each is a legitimate CART fit.
- "**Flat after about fifty trees: B is a budget, not a tuning parameter.**"

### L8 p.46 — Model averaging
- Central to many nonparametric algorithms: ensemble learning, mixture of experts, Bayesian averages, … (mentions only).
- Works best with **flexible but simple** models.
- Recall **lasso as a stabilized version of stepwise regression** (jitter the data, estimates stay constant).
- "Model averaging is a way to take arbitrary *unstable* methods, and make them stable. This makes training easier."
- P(rain) on a new day = average P(rain) across trees splitting on forecast vs sky — "We don't get tied to one way of deciding about umbrellas."

### L8 p.47 — CODE: random forests in python + MDI warning
- sklearn `RandomForestRegressor` and `RandomForestClassifier` "work essentially the same as tree":
```python
rf = RandomForestRegressor(n_estimators=300).fit(X, y)
```
- "Unfortunately, you **lose the interpretability** of a single tree. You have traded one readable rule set for three hundred of them."
- **Warning (red on slide):** "One warning about `feature_importances_`, because the name invites the wrong reading. It is the **in-sample mean decrease in impurity**, summed over splits and normalized. **It is not an out-of-sample statistic.** If you want importance measured out of sample, **permute** — we do that at the end of the lecture, and the two rankings disagree."

### L8 p.48–49 — Motorcycle data: random trees vs averaged forest
- p.48: several trees fit to random subsets (coloured step functions) — "you get a slightly different tree each time."
- p.49: averaging many trees → a single (smoother) response surface.

### L8 p.50 — Boosted trees algorithm sketch
- "Boosting": ensemble aggregating many "**weak learners**" (models that don't forecast well) into a single "**strong learner**". Cf. **Schapire (1990, ML)**.
- Algorithm:
  - Fix tree-depth **L (shallow!, e.g. L = 1)**.
  1. Fit 1st simple tree — forces trees to be weak learners with large bias.
  2. Fit 2nd shallow tree to **residuals from the first tree**.
  - Forecasts from the trees are **summed** to form the ensemble prediction.
  - **Shrink the forecast component from the 2nd tree by a factor ν ∈ (0,1)** — helps prevent overfitting the residuals.
  - Iterate until a total of **B** trees.
- "**L, ν, and B are tuning parameters selected via validation.**"
- Implied formula (standard form of the sketch): $\hat f_0=$ first tree; $r_i \leftarrow y_i-\hat f_{b-1}(x_i)$; fit depth-L tree $\mathcal T_b$ to $r$; $\hat f_b=\hat f_{b-1}+\nu\,\mathcal T_b$; final $\hat f(x)=\sum_b \nu\,\mathcal T_b(x)$ (first tree unshrunk per the slide's wording).

### L8 p.51 — Bagging and boosting are opposites (table)
| Bagging / random forest | Boosting |
|---|---|
| Trees fit **in parallel**, each to a resample of the same data. | Trees fit **in sequence**, each to what the previous ones got wrong. |
| Trees are **deep**: low bias, high variance. Averaging kills the variance. | Trees are **shallow**: high bias, low variance. Adding kills the bias. |
| Trees should be **as independent as possible** — hence column subsampling. | Trees are **deliberately dependent** — tree b exists only because of trees 1..b−1. |
| **More trees never hurt. B is a budget.** | **More trees eventually hurt. B is a tuning parameter.** |
- "Same ingredient, opposite recipe. That last row is why boosting needs more care than a forest — and why it wins when you give it that care."

### L8 p.52 — Shrinkage, rounds, and when to stop (numbers)
- Held-out R² vs boosting rounds B (log scale, 10 to ~4000); California panel, **depth-4 trees, subsample 0.8** (stochastic gradient boosting); horizontal line "one CART" ≈ 0.734.
  - **ν = 0.5**: peaks at **125 rounds, 0.8262**, then **decays to 0.8045 by 4,000**. "Big steps overfit the residual."
  - **ν = 0.1**: peaks at **2,803, 0.8586**.
  - **ν = 0.05**: **0.8632, still creeping up at 4,000**.
  - **ν = 0.01**: **0.8533, not converged**.
- "**Halve the learning rate and you need roughly twice the rounds.** Pick ν small enough to be safe, then **let the held-out curve tell you B**." (Early stopping on a validation curve.)

### L8 p.53 — Back to California housing: the model horse race (conventions)
- Same 20,640 tracts, same **nine predictors**, same target log(MedVal).
- "Five models, **all scored on the same 5-fold split, against the same training-fold mean**":
  - standardized LASSO — the linear benchmark
  - cross-validated CART — one tree, 211 leaves
  - `GradientBoostingRegressor(100)` — boosting straight out of the box
  - `RandomForestRegressor(300)` — the forest
  - boosting with ν and B chosen on held-out data
- "**The comparison is only meaningful because the split is shared. Two numbers computed on two different splits are two different questions.**"
- Implied OOS R² definition (benchmark = training-fold mean): $$R^2_{OOS}=1-\frac{\sum_{i\in\text{test}}(y_i-\hat y_i)^2}{\sum_{i\in\text{test}}(y_i-\bar y_{\text{train fold}})^2}$$

### L8 p.54 — CA housing: out-of-sample prediction (bar chart)
| LASSO | CART | GBRT (default) | Random Forest | GBRT (tuned) |
|---|---|---|---|---|
| 0.6415 | 0.7220 | 0.7997 | 0.8327 | 0.8600 |
(5-fold OOS R²)
- **One tree beats the linear model**: 0.7220 vs 0.6415 — gap = nonlinearity + interaction (the geographic map of Part I).
- **Averaging beats one tree**: 0.8327 vs 0.7220, for almost no tuning.
- **Boosting out of the box *loses* to the forest**, 0.7997. "One hundred rounds at the default rate is not enough."
- **Boosting tuned wins, 0.8600** — bought with **3,998 rounds** and a held-out search, "where the forest cost one line."

### L8 p.55 — The honest version of "boosting wins"
- Boosted trees are the standard workhorse on tabular data — but state it precisely:
  - **A random forest is nearly tuning-free.** Set B large, set a min leaf size, walk away → 0.8327.
  - **Boosting is not.** With ν, B, depth and subsampling all live, the out-of-the-box call sits **0.03 below** the forest. The win of **+0.027** is real, bought with a validation budget.
  - **"And a tuning budget is a leakage risk. Every round of the search looks at held-out data. Tune on a validation split, report on a test split you touched once."**
- "If you have a day, boost. If you have an hour, or a deadline, or a committee, the forest is the better answer and it is not close."

### L8 p.56 — Variable importance: MDI (algorithm)
Mean decrease in impurity (MDI) importance of a feature:
1. For each tree: the decrease in variance (impurity) is calculated every time the feature is used to split a node — how much the split improves node homogeneity (reduces target impurity); this decrease is averaged across all splits in the tree that use the feature.
2. For the forest: average across all trees → importance score per feature.
3. **Normalization**: scores normalized to sum to 1.
- Formula form (standard; weights by node size in sklearn): $\text{MDI}_j \propto \frac1B\sum_b \sum_{t\in \mathcal T_b: v(t)=j} \frac{n_t}{n}\big[i(t) - \tfrac{n_{tL}}{n_t}i(t_L)-\tfrac{n_{tR}}{n_t}i(t_R)\big]$, then $\sum_j \text{MDI}_j=1$.
- **Red warning:** "Note what this is not: **every quantity above is computed on the rows the tree was fit to. MDI is an in-sample statistic.**"

### L8 p.57 — Permutation variable importance (algorithm)
- "Model-agnostic and often **more reliable, especially in the presence of correlated features**."
1. Train the model on the original dataset and compute the performance metric of interest **on held-out data**.
2. For each feature: **shuffle** its values across all samples (breaking its relationship with the target); recompute the metric on this modified data; importance = **how much the metric decreases** (larger decrease = more important).
3. **Repeat the shuffle several times and average**, "since a single permutation is itself a random draw."
- Formula: $\text{PI}_j=\text{score}(\hat f; X_{\text{test}},y_{\text{test}})-\frac1R\sum_{r=1}^R\text{score}(\hat f; X_{\text{test}}^{(j,\pi_r)},y_{\text{test}})$.
- "More computationally intensive than MDI ... and it is worth it, because **it answers the question you actually asked**."
- (sklearn: `sklearn.inspection.permutation_importance(model, X_test, y_test, n_repeats=..., random_state=...)` — the API is not shown on the slide; the exam notebook imports it.)

### L8 p.58 — The two rankings disagree (California; chart + numbers)
- Horizontal bars, share of total importance, **MDI in sample (normalised)** vs **permutation, held out (normalised)**; "Same forest, same data; MDI in sample, permutation on the **held-out third**."
- Approximate bar values read from chart:

| feature | MDI (in-sample) | Permutation (held-out, normalised) |
|---|---|---|
| medianIncome | ≈0.49 | ≈0.34 |
| latitude | 0.1177 (3rd) | ≈0.33 (2nd) |
| longitude | ≈0.12 (2nd) | ≈0.23 |
| AveOccupancy | ≈0.10 | ≈0.07 |
| housingMedianAge | ≈0.04 | ≈0.02 |
| AveRooms | ≈0.07 | ≈0.01 |
| AveBedrms | ≈0.02 | ≈0.00 |
| households | 0.0201 | 0.0035 |
| population | 0.0185 | 0.0015 |

- "**Latitude: MDI third at 0.1177, permutation second at 0.6101.** MDI **splits the credit for location across two correlated columns and undercounts both**." (0.6101 is evidently the raw, un-normalised held-out drop; the chart shows normalised shares.)
- "**Population and households: real MDI mass (0.0185, 0.0201), worth essentially nothing out of sample (0.0015, 0.0035). MDI rewards a column for merely offering many places to split.**"

### L8 p.59 — Partial dependence, defined (formula + CODE)
- Take one variable $x_j$ and a value v; set $x_j=v$ for **every** row, leave other columns as observed, predict, average:
  $$\bar f_j(v)=\frac1n\sum_{i=1}^n \hat f\big(v,\ \mathbf x_{i,-j}\big).$$
- Repeat over a grid of v — **forty points from the 5th to the 95th percentile of $x_j$** — and plot $\bar f_j(v)$ vs v.
```python
def partial_dependence(model, X, feature, grid):
    out = []
    for v in grid:
        Xv = X.copy()
        Xv[feature] = v          # every row gets x_j = v; the rest as observed
        out.append(model.predict(Xv).mean())
    return np.array(out)
```
- "It averages over the other columns *as observed*, so it reports **what the model says, not a causal effect**."
- "sklearn's `partial_dependence` defaults to a faster tree-walk approximation for forests; **`method='brute'` is this definition exactly**."
- Grid idiom (implied): `grid = np.linspace(X[f].quantile(0.05), X[f].quantile(0.95), 40)`.

### L8 p.60 — Importance says which; partial dependence says which way
- Three PD panels (computed **on the fitting sample** with the seven-line function):
  - **medianIncome** (≈1.5–7.5): PD rises from ≈11.78 to ≈12.79 — "spans **1.01** in log MedVal — a factor of 2.7 in levels — and rises throughout. A sentence a committee can act on."
  - **AveOccupancy** (≈2–4.5): PD falls from ≈12.32 to ≈11.93 — "spans **0.39**, steeply decreasing at the low end then nearly flat. **One linear coefficient would average those two regimes into a misleading number.**"
  - **housingMedianAge** (≈8–52): PD ≈12.04→12.115 — "spans **0.08**: present, economically small."
- "**Importance is a magnitude with no sign. Partial dependence has the sign.**"

### L8 p.61 — Roundup on tree-based learning
- **CART**: recursive partitions, pruned back by CV. Readable. Unstable. California R² = 0.7220.
- **Random Forest**: average many deep CART trees, decorrelated by column subsampling. Nearly tuning-free, R² = 0.8327. **"The default answer."**
- **Boosted Trees**: repeatedly fit shallow trees to residuals, shrinking each by ν. Best in class at 0.8600 — *after* ν and B are chosen honestly. Out of the box it loses to the forest.
- "**Trees are poor in high dimension, but fitting them to low-dimensional factors (principal components, Lecture 1) is a good option.**"
- "Whichever you pick, you have bought accuracy with interpretability. **Permutation importance and partial dependence are how you buy some of it back.**"

### L8 p.62 — Roundup on nonlinear regression and classification (passing mentions)
- Other nonparametric learners: **Neural Networks** (many recursive logistic regressions); **Support Vector Machines** (project to HD, then classify); **Gaussian Processes, splines, wavelets** (sums of curvy functions). "Some of these are great, but **all take a ton of tuning**."
- "**On tabular financial data of the size you will actually meet, nothing out of the box beats a tree ensemble. When the linear model does win, it is usually because the signal really is weak and linear — which, in asset pricing, is more often than you would like.**"

---

## 2. Consolidated formula sheet (L8)

| Object | Formula | Page |
|---|---|---|
| Binary split | $x_j\le c$ vs $x_j>c$ | p.5 |
| Regression deviance | $\sum_i (y_i-\hat y_i)^2$ | p.7 |
| Classification deviance | $-\sum_i \log \hat p_{y_i}$, $\hat p_{y_i}=\hat p(y_i\mid x_i)$ | p.7 |
| Gini deviance (as written) | $-\sum_i \hat p_{y_i}(1-\hat p_{y_i})$; node Gini impurity printed by sklearn $=1-\sum_k\hat p_k^2$ | p.7, p.25, p.33 |
| Leaf class prob. | $\hat P(C=k\mid X\in R_j)=\#\{x_i\in R_j, y_i=k\}/\#\{x_i\in R_j\}$ | p.17 |
| Leaf class | $\hat c_j=\arg\max_k \hat P(C=k\mid X\in R_j)$ | p.17 |
| Regression tree | $E[y\mid x]=\sum_{j=1}^m c_j\mathbb 1\{x\in R_j\}$, $c_j=\frac1{n_j}\sum_{x_i\in R_j}y_i$ | p.20 |
| Search-space size | $p\cdot 2^d$ (ignoring split points) | p.21 |
| Cost-complexity | $R_\alpha(T)=R(T)+\alpha|T|$ | p.29 |
| Error decomposition | error = bias² + variance + noise | p.40 |
| RF regression | $\mathbb E[y\mid x]=\frac1B\sum_{b=1}^B\mathcal T_b(x)$; classification: majority vote | p.42 |
| RF column subset | √p (classification), p/3 or p (regression) per split | p.43 |
| Variance of average | cut by ≈B only if trees independent | p.43 |
| Boosting | depth L, fit residuals, shrink by ν∈(0,1), sum B trees; L, ν, B via validation | p.50 |
| ν–B tradeoff | halve ν ⇒ ≈ double B | p.52 |
| OOS R² benchmark | training-fold mean | p.53 |
| MDI | per-tree average impurity decrease over splits on feature → average over trees → normalize to sum 1 (in-sample) | p.56 |
| Permutation importance | held-out metric drop after shuffling feature, averaged over repeats | p.57 |
| Partial dependence | $\bar f_j(v)=\frac1n\sum_i\hat f(v,\mathbf x_{i,-j})$, 40-point grid 5th–95th pct | p.59 |

## 3. Code idioms (course "house style")

| Idiom | Page |
|---|---|
| `from sklearn import tree`; `clf = tree.DecisionTreeClassifier(); clf = clf.fit(X, Y)` | p.24 |
| `X = demos.iloc[:,1:]`; `y = nbc.Genre` | p.24 |
| `tree.export_text(clf)`; `plot_tree(clf)` (dendrogram) | p.25 |
| `X = pd.concat([nbc.GRP, pd.get_dummies(nbc.Genre)], axis=1)` (categoricals → dummies) | p.26 |
| `clf = tree.DecisionTreeRegressor().fit(X, y)` | p.26 |
| `tree.DecisionTreeClassifier(max_leaf_nodes=3).fit(X, Y)` (size control via max_leaf_nodes) | p.33 |
| `Gender <= 0.5` in dendrogram ⇒ Gender coded 0/1 | p.30 |
| `rf = RandomForestRegressor(n_estimators=300).fit(X, y)` | p.47 |
| `rf.feature_importances_` = in-sample MDI | p.47, p.56 |
| `GradientBoostingRegressor(100)` (out-of-box, 100 rounds, default rate), `RandomForestRegressor(300)` | p.53 |
| Boosting config explored: depth-4 trees, `subsample=0.8`, learning rates 0.5/0.1/0.05/0.01, up to 4,000 rounds | p.52 |
| Custom `partial_dependence(model, X, feature, grid)` function; sklearn `partial_dependence(..., method='brute')` | p.59 |
| 70/30 random split for cross-sectional data (14,447 / 6,193) | p.45 |
| 5-fold CV OOS R² as the comparison metric | p.37, p.38, p.53–54 |
Not shown in code on the slides (but implied/used by exam notebook): `cost_complexity_pruning_path`/`ccp_alpha`, `permutation_importance`, `StratifiedKFold`, `cross_val_score`, `train_test_split`, `RandomForestClassifier`, `GradientBoostingClassifier`, `min_samples_leaf`, `max_features`.

## 4. Course conventions insisted on in L8
1. **Tree size / α chosen out of sample** (test sample or 5-fold CV), never by in-sample fit (p.29, p.32, p.38–39).
2. **Coarse geometric grid** for α suffices (p.39).
3. **All models compared on the same split / same folds**, R² computed **against the same training-fold mean** (p.53); "two numbers computed on two different splits are two different questions."
4. **Random split is legitimate only because the data are a cross-section** (p.45) — implication: not for time series.
5. **Tune on a validation split, report on a test split touched once** (p.55).
6. **RF: B is a budget, not a tuning parameter**; grow deep unpruned trees, set min leaf size, B large (p.44–45, p.51).
7. **Boosting: L, ν, B are tuning parameters chosen via validation**; pick ν small, let the held-out curve choose B (p.50, p.52).
8. **MDI `feature_importances_` is in-sample; permutation importance on held-out data is the out-of-sample importance** and the one that "answers the question you actually asked" (p.47, p.56–58).
9. **Permutation importance repeated and averaged** (p.57).
10. **Partial dependence: 40-point grid from 5th to 95th percentile; brute-force definition** (p.59); PD gives sign/shape, importance only magnitude (p.60).
11. Standardized LASSO is the linear benchmark (p.37, p.53).
12. In high dimension, fit trees to **principal components** (Lecture 1) (p.61).
13. Categorical predictors → dummy variables (p.26); binary coded 0/1 (p.30).
14. Deliverable matters: forecast → CV-optimal tree; explanation → readable small tree (p.39).

## 5. Warnings / pitfalls emphasised
- Unpruned trees overfit: pure 1-obs leaves (p.25, p.30); California grown-out tree interpolates, in-sample R² = 1.0000 but OOS 0.6352 (p.38); Social Network Ads unpruned test accuracy 0.90 vs pruned 0.94 (p.32).
- Exhaustive tree search infeasible ($p\cdot2^d$) → greedy; greedy first split is fragile (p.21, p.40).
- Readable tree ≠ accurate tree (p.39).
- Variance not bias; CV addresses the average, not dispersion; "with one history you only get one draw" (p.40).
- Deep tree structure too unstable for CV to reliably choose depth; "no theory to fall back on" (p.41).
- Bagged trees are correlated; averaging correlated estimators buys much less (p.43).
- Boosting: more trees eventually hurt; big ν overfits the residual (ν=0.5 decays 0.8262→0.8045) (p.51–52).
- Out-of-box boosting (100 rounds, default rate) can lose to the forest (p.54).
- **Tuning budget is a leakage risk** — every search round looks at held-out data (p.55).
- Comparisons on different splits are meaningless (p.53).
- `feature_importances_` "name invites the wrong reading" — in-sample MDI (p.47, p.56).
- MDI splits credit among correlated columns and rewards columns with many split points (p.58).
- A single permutation is a random draw — repeat (p.57).
- PD is not causal; sklearn's default for forests is an approximation (p.59).
- One linear coefficient can average two regimes into a misleading number (p.60).
- Trees poor in high dimension (p.61).
- Losing interpretability with ensembles (p.47, p.61).
- In asset pricing the signal is often weak and linear, so linear models can win (p.62).

## 6. Datasets / worked examples
- Umbrella/rain toy tree (p.3, p.6, p.46).
- 500-point 2-D two-class toy (p.14–15).
- NBC TV pilots: 6241 views, 20 questions, 40 shows; GRP, PE, genre, demographics (p.23–28).
- **Social Network Ads**: 400 clients; Gender, Age, EstimatedSalary → Purchased; root [257, 143]; pruned tree `max_leaf_nodes=3`: Age ≤ 42.5, EstimatedSalary ≤ 90,500 (p.30–34).
- Motorcycle crash-test dummy data: times vs accels (p.35, p.48–49).
- California housing: 20,640 tracts, nine predictors, target log(MedVal) (p.36–39, p.45, p.52–60).

## 7. Boundaries (mentioned only in passing / not taught in L8)
- Neural networks, SVMs, Gaussian processes, splines, wavelets — named only; "all take a ton of tuning" (p.62).
- CHAID / AID — historical mention (p.28).
- Mixture of experts, Bayesian averaging — named only (p.46).
- Schapire (1990) boosting — citation only; **AdaBoost is not taught**; boosting is taught as residual-fitting with shrinkage (gradient boosting for squared loss).
- XGBoost / LightGBM / CatBoost, out-of-bag (OOB) error, `class_weight`, entropy vs Gini `criterion` argument, Shapley/SHAP, ICE curves — **not mentioned**.
- LR and k-NN appear only in the comparison table (p.18) (taught elsewhere).
- Time-series cross-validation / expanding windows — **not taught in L8**; L8 only says a random split is legitimate because California is a cross-section (p.45). Time-ordered evaluation comes from other lectures (e.g. Lecture 5, per exam notebook).
- The cost-complexity pruning algorithm's weakest-link details and the sklearn `ccp_alpha` API are not spelled out; only $R_\alpha(T)$ and the α-path plots/table.
- Only regression boosting code is named (`GradientBoostingRegressor`); classification boosting is not shown.

## 8. Mapping to the final exam

### Problem 1 (Trees & Ensembles, Social_Network_Ads.csv) — L8 is the primary source; the notebook itself says "Lecture 8 built a single tree, then pruned it, then averaged many of them."
- **1.1 baseline:** p.30/p.33 root node value [257, 143] ⇒ "nobody purchases" accuracy = 257/400 = **64.25%**; p.32 shows the root-only tree (α≈0.171) test accuracy 0.68 / train 0.63 = majority-class rule on those subsets. (In CV, baseline is the fraction of 0s in each test fold; with stratification ≈ 0.6425.)
- **1.1 unpruned tree:** p.24 `DecisionTreeClassifier()` default = unpruned; p.30 shows it overfits (pure tiny leaves, depth ≈12); p.32 unpruned test acc ≈0.90 vs pruned 0.94.
- **1.1 max_leaf_nodes sweep:** p.29 (choose complexity out of sample), p.31–32 (complexity vs α; hump-shaped test accuracy), p.33 (lecture's own choice `max_leaf_nodes=3`, depth 2), p.38–39 (coarse grid suffices; left side of optimum underfits, right side overfits with OOS falling). Tie → smaller tree consistent with p.39 "readable tree" logic. Describe either side: too few leaves = bias (e.g. 2 leaves = one Age split, ≈0.89 on p.32 at 1 split), too many = variance.
- **1.2 plot & describe:** p.25 (`plot_tree` dendrogram reading: split rule, gini, samples, value=[n0,n1]); p.33 (the tree: Age ≤ 42.5 then EstimatedSalary ≤ 90,500); p.34 (rectangle in age–salary plane); p.8/p.17 (leaf class = majority; leaf probability = proportion).
- **1.3 RF(300) & GB(100):** p.41–45 (RF = bootstrap + column subsampling, deep trees, averaging reduces variance; B=300 as budget); p.50–52 (GB: shallow trees, ν, B; default 100 rounds); p.54–55 (out-of-box GB may lose to RF; ensembles won on 20,640 × 9 California data). Explanation "given size and shape": 400 rows, 3 features (only 2 relevant per p.33–34), true boundary nearly a 2-split rectangle ⇒ the pruned tree has little bias to remove; ensembles mainly cut variance, and with p=3 the √p rule gives 1 candidate feature per split (sklearn default) so the decorrelation lever is weak; small n makes 5-fold accuracy noisy (each fold = 80 people, 1 person = 1.25pp). Report whichever way it comes out (p.55 "state it precisely"; p.53 same split for all).
- **1.4 MDI vs permutation:** p.47 (`feature_importances_` is in-sample MDI, "not an out-of-sample statistic"), p.56 (MDI algorithm, normalized), p.57 (permutation on held-out, repeat & average, "answers the question you actually asked"), p.58 (rankings disagree; MDI rewards many split points — continuous EstimatedSalary/Age vs binary Gender; MDI splits credit among correlated columns). Show the decision maker the **held-out permutation** ranking (p.57, p.61), optionally with a PD plot for direction (p.59–60: importance has no sign).
- **1.5 shuffled CV on monthly stock data:** p.45 is the key hook — "a cross-section, so a random split is legitimate" (monthly rows of one stock are a time series, so it is not); p.40 "with one history you only get one draw"; p.55 leakage from looking at held-out data. What goes wrong: shuffling puts future months into the training folds used to predict earlier months (look-ahead) and neighbouring autocorrelated/overlapping-feature months straddle folds (leakage) ⇒ reported accuracy **biased upward (optimistic)**. Remedy (from other lectures): time-ordered split / expanding-window (walk-forward) evaluation, possibly with a gap; tune on an earlier validation block, report on a later test block touched once (p.55).

### Problem 2 (VaR self-training pipeline)
- L8 contains **no** content on variance estimation, VaR, martingales or feedback loops. Only loose analogies: bootstrap resampling with replacement (p.41) and "with one history you only get one draw" (p.40). Do **not** cite L8 as the method source for Problem 2.

### Problem 3 (return forecasting research project)
- **3-models:** RF and boosted trees are allowed models (notebook). L8 guidance: RF "the default answer", nearly tuning-free (p.44, p.55, p.61); GBRT needs ν/B tuned on validation, small ν, early stopping on held-out curve (p.50–52); depth/min-leaf controls (p.22, p.44).
- **3-features:** "Trees are poor in high dimension, but fitting them to low-dimensional factors (principal components, Lecture 1) is a good option" (p.61) — relevant for x17..x164 extended set. Trees handle nonlinearity/interactions automatically (p.28, p.37) — e.g. characteristic × macro interactions; categorical asset class/country → dummies (p.26).
- **3-evaluation:** same split for every model, R² vs the training-fold mean (p.53) ↔ exam's R²_OOS vs trailing mean (mean through t−1) and vs zero; tune on validation, report on test touched once (p.55); random split legitimate only for cross-sections (p.45) ⇒ must use time-ordered expanding/rolling windows (from other lectures); variance of results / one history (p.40); expect linear or near-zero predictability — "When the linear model does win, it is usually because the signal really is weak and linear — which, in asset pricing, is more often than you would like" (p.62).
- **3-interpretation (which predictors carry signal; does macro add; placebo):** permutation importance on the out-of-sample window, not MDI (p.47, p.56–58); MDI would reward noisy continuous columns and split credit across duplicated/correlated columns — directly relevant to the duplicates in x17..x164 (p.58); PD for direction/nonlinearity (p.59–60).
- **3-portfolio:** L8 has no portfolio-construction content; only supplies the forecast models.

---

## 9. Quick-reference numbers (for sanity checks)
- Social Network Ads: n=400, [257 no, 143 yes] → baseline 0.6425; pruned tree 3 leaves: [232,9], [7,37], [18,97] → in-sample acc 0.915; test acc on p.32 split: unpruned ≈0.90, pruned ≈0.94, one split ≈0.89, root ≈0.68.
- California: depth-6 lat/long tree 0.5580 vs LASSO 0.3164 (p.37); CV tree 211 leaves 0.7220; 12-leaf 0.5610; grown-out 19,826 leaves, in-sample 1.0, OOS 0.6352 (p.38–39); 70/30 split: tree 0.6875, forest B=10 0.8084, B=50 0.8225, B=300 0.8268, CV-CART 0.7341, sd of single trees 0.0148 (p.45); boosting ν=0.5 peak 0.8262@125 → 0.8045@4000; ν=0.1 0.8586@2803; ν=0.05 0.8632@4000; ν=0.01 0.8533 (p.52); 5-fold horse race LASSO 0.6415, CART 0.7220, GBRT default 0.7997, RF 0.8327, GBRT tuned 0.8600 (3,998 rounds) (p.54).
