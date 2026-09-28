# Lecture 6: Classification (BUSN 41210, Dacheng Xiu, Sep 18 2026)

Source: `/home/user/UChicago/Lecture_6.pdf` (57 slides). I viewed every slide as a rendered image (80 dpi, and 160 dpi
for p.34, 48, 49, 50, 55, 56) and also read the extracted text. Citations use the form "L6 p.X"; X is both the PDF
page and the slide number.

**Scope in one sentence.** The lecture covers binary and multi-class **classification**. It teaches the logit link
and logistic regression (statsmodels `glm` with a Binomial family), MLE and deviance, deviance-based R^2 in sample
and on a random left-out sample, gradient descent, the Bayes classifier, K-nearest neighbours (hand-coded and with
sklearn), a Lending Club case with a **time-ordered train/test split**, turning probabilities into decisions
(the 0.5 Bayes rule, then a cost-matrix threshold), the confusion matrix, FPR/FNR, sensitivity/specificity,
ROC/AUC, precision-recall under rare events, precision-at-k and lift, and a checklist for reporting a classifier.

**What L6 does NOT cover:** it never mentions decision trees, random forests, boosting, impurity/Gini, pruning or
variable importance (the exam says those come from Lecture 8). It shows no K-fold, stratified or time-series
cross-validation code. It never covers class weights, resampling, calibration, LDA/QDA, naive Bayes, SVMs, or
multinomial logistic regression.

---

## 0. Outline (L6 p.1–2)

- p.1: Title "Lecture 6: Classification", Dacheng Xiu, Chicago Booth, September 18, 2026.
- p.2 Outline:
  1. Logistic regression
  2. K-nearest neighbors and group membership
  3. Binary classification: from probabilities to decisions
  4. Misclassification, sensitivity and specificity

---

## 1. Logistic regression: model and link (L6 p.3–6)

### p.3 Motivation
- "Linear regression is just one type of linear model."
- "Logistic regression: when y is true or false (1/0)."
- Examples of binary response targets: **Profit or Loss, greater or less than**, Pay or Default; thumbs up/down,
  buy or not buy, potential customer?; Win/Lose, Sick/Healthy, Republican/Democrat.
  - Exam link: "greater or less than" / "profit or loss" is exactly the framing of **1.5** (predicting whether next
    month's return is positive).

### p.4 Building a linear model for binary response
- Original specification: $\mathbb{E}[y\mid \mathbf{x}] = f(\mathbf{x}'\beta)$.
- With $y\in\{0,1\}$:
  $$\mathbb{E}[y\mid\mathbf{x}] = p(y=1\mid\mathbf{x})\times 1 + p(y=0\mid\mathbf{x})\times 0 = p(y=1\mid\mathbf{x}).$$
  "⇒ The expectation is a probability."
- "We'll choose $f(\mathbf{x}'\beta)$ to give values between zero and one."

### p.5 Binary choice model
- $p = P(y=1\mid\mathbf{x}) = f(\beta_0 + \beta_1x_1 + \dots + \beta_p x_p)$, "where f is a function that increases in
  value from zero to one."
- Figure: an S-shaped curve $f(XB)$ plotted for $XB\in[-4,4]$, crossing 0.5 at $XB=0$ (dashed guides at 0 and 0.5).

### p.6 Logit link
$$P(y=1\mid\mathbf{x}) = \frac{e^{\mathbf{x}'\beta}}{1+e^{\mathbf{x}'\beta}} = \frac{\exp[\beta_0+\beta_1x_1+\dots+\beta_dx_d]}{1+\exp[\beta_0+\beta_1x_1+\dots+\beta_dx_d]}$$
- "The 'logit' link is common, for a couple of good reasons." One big reason:
  $$\log\left[\frac{p}{1-p}\right] = \beta_0+\beta_1x_1+\dots+\beta_dx_d,$$
  "so that it is a linear model for **log-odds**."

---

## 2. Credit-card default example: estimation, interpretation, deviance (L6 p.7–14)

### p.7 Data: `Default.csv`
- "Credit Card Clients does binary regression: Default versus not Default." $y=1$ for Default, else 0.
- `Default.csv` holds 30,000 credit card clients (about 6,636 defaults). Link given: "Default of Credit Card Clients
  Data Set" (the UCI dataset).
- "Logistic regression fits p(y = 1) as a function of related information."

### p.8 Code: logistic regression via statsmodels GLM (**house style**)
```python
smf.glm(formula=my_formula, data=default,
        family=sm.families.Binomial()).fit()
```
- "glm stands for Generalized Linear Model." `family=sm.families.Binomial()` "indicates y is binary."
- The response may be numeric (1,1,0,…), logical (TRUE/FALSE,…) or a factor ('win','win','lose',…). The wording is R-like.
- "Everything else is the same as for linear regression." The formula-API conventions carry over from earlier
  lectures (`smf` = `statsmodels.formula.api`, `sm` = `statsmodels.api`).

### p.9 Interpreting coefficients: odds multipliers
- Model: $\dfrac{p}{1-p} = \exp[\beta_0 + x_1\beta_1 + \dots + x_p\beta_p]$.
- "So $\exp(\beta_j)$ is the **odds multiplier** for a unit increase in $x_j$."
- `b['AGE']=0.007`: one unit increase in AGE multiplies the odds of default by $\exp(0.007)\approx 1.007$.
- `b['PAY_0']=0.577`: one unit increase in last month's payment (status) multiplies the odds of default by
  $\exp(0.577)\approx 1.781$.
- Question posed: "What is the odds multiplier for a covariate coefficient of zero?" (Answer: $e^0 = 1$, no change in
  odds. The slide does not state it.)

### p.10 MLE for logistic regression
$$L(\beta)=\prod_{i=1}^n P(y_i\mid\mathbf{x}_i)=\prod_{i=1}^n p_i^{y_i}(1-p_i)^{1-y_i}
=\prod_{i=1}^n\left(\frac{\exp[\mathbf{x}_i'\beta]}{1+\exp[\mathbf{x}_i'\beta]}\right)^{y_i}\left(\frac{1}{1+\exp[\mathbf{x}_i'\beta]}\right)^{1-y_i}$$
"This is maximized by minimizing the **deviance**":
$$Dev = -2\sum_{i=1}^n\big(y_i\log(p_i) + (1-y_i)\log(1-p_i)\big)\ \propto\ \sum_{i=1}^n\Big[\log\big(1+e^{\mathbf{x}_i'\beta}\big) - y_i\mathbf{x}_i'\beta\Big]$$
- This is the binary cross-entropy / log-loss (×2). The lecture never uses the words "log-loss" or "cross-entropy".

### p.11 Output check: dispersion = 1
- "We have the same output as for a linear/gaussian model. But the 'dispersion parameter' here is always set to one.
  **Check this to make sure you've actually run logistic regression.**"
```text
proba.summary()
(Dispersion parameter for binomial family taken to be 1)
Residual deviance: 27877  on 29976  degrees of freedom
```
- "'degrees of freedom' is actually 'number of observations - df', where df is the number of coefficients estimated
  in the model." So **df(deviance) = nobs − df(regression)**.
- Question: "From the Python output, how many observations do we have?" (Answer: n = 30,000 from p.7, so
  30,000 − 29,976 = 24 estimated coefficients. This arithmetic is mine, not the slide's.)
- `proba` is the name of the fitted logistic model object throughout.

### p.12 Gaussian analogue (recycled from the linear-regression lectures)
- "Sum of Squares (Deviance) is the bit we need to minimize": $Dev \propto \sum_{i=1}^n (y_i - \mathbf{x}_i\beta)^2$.
  "This makes the observed data as likely as possible."
- Error variance: $\sigma^2 = \mathrm{var}(\varepsilon)$, $\varepsilon_i = y_i - \mathbf{x}_i\beta$ ("residuals"),
  $$\hat\sigma^2 = \frac{1}{n-p-1}\sum\hat\epsilon_i^2.$$
- "Python estimates $\sigma^2$ and calls it the **Scale**." "Even if we know β, we only predict log sales with
  uncertainty," e.g. a 95% probability that log sales lie in $\mathbf{x}'\beta \pm 2\sqrt{0.48}$. The OJ/sales example
  comes from earlier lectures. Point: in the Gaussian GLM the Scale is estimated; in the Binomial GLM it is fixed at 1.
- The $n-p-1$ degrees-of-freedom correction is the same "unbiased" logic as `ddof=1` for sample variance. That is
  relevant only by analogy to Problem 2, which uses ddof=1.

### p.13 Null deviance and deviance R^2
- **Residual deviance** $D$ is what we minimised, using $\mathbf{x}'\beta$. **Null deviance** $D_0$ is for the model
  without $\mathbf{x}$, i.e. $\hat y_i = \bar y$:
  - $D_0 = \sum (y_i-\bar y)^2$ in linear regression;
  - $D_0 = -2\sum[y_i\log(\bar y) + (1-y_i)\log(1-\bar y)]$ in logistic regression.
- "The difference between D and D0 is due to info in x." "Proportion of deviance explained by x is called $R^2$":
  $$R^2 = \frac{D_0 - D}{D_0} = 1 - \frac{D}{D_0}.$$
  "This measures how much variability you are able to model."
- In `proba`: $R^2 = 1 - 27877/31705 = 0.12$.
- Course pattern: the benchmark is the **constant (intercept-only) prediction $\bar y$**. This parallels $R^2_{OOS}$
  with a mean benchmark (L4), and the majority-class rule is its classification analogue.

### p.14 Fit plot for logistic regression
- "We can plot $\hat y$ vs $y$ in logistic regression using a boxplot." Figure: "fit plot for default logistic
  regression". x-axis is default category 0/1; y-axis is fitted probability of default.
  - Category 0: median ≈ 0.17, box ≈ 0.12–0.23, many high outliers up to 1.0.
  - Category 1: median ≈ 0.27, box ≈ 0.19–0.50, whisker to ≈ 0.97.
- "The estimation pushes each distribution away from the middle." "Where would you choose for a classification
  cut-off?" This sets up p.43–47.

---

## 3. Implementation, prediction and out-of-sample evaluation (L6 p.15–18)

### p.15 Implementation: gradient descent
- Linear regression has a closed form: $\hat\beta = (X'X)^{-1}X'y$.
- "Logistic regression requires a numerical procedure. For instance, we can adopt **gradient descent** of the loss
  function." Gradient descent is "applicable in the search of local minimum of a differentiable function. Extremely
  useful (with some twists) in deep learning."
- Algorithm:
  1. Start with an initial value $\beta_0$.
  2. Iterate $\beta_{t+1} = \beta_t - \gamma_t\nabla loss(\beta_t)$.
  3. Stop when converged.
- "$\gamma_t$ is called **learning rate**, the most important parameter to tune."
  - Exam link: in gradient boosting (1.3) `learning_rate` is the shrinkage. L6 teaches only generic gradient descent;
    boosting belongs to L8.

### p.16 Prediction
- "In logistic regression, we use `reg.predict(mynewdata)` to get probabilities $e^{\mathbf{x}'\hat\beta}/(1+e^{\mathbf{x}'\hat\beta})$":
```text
proba.predict(default[0:4])
      1         2         3         4
 0.505544  0.150024  0.200426  0.242773
```
- "`newdata` **must** match the format of original data."
- statsmodels GLM `predict` returns **probabilities** on the response scale, not classes.

### p.17 Out-of-sample prediction: random validation sample (code)
- "You care about how your model predicts out-of-sample (OOS). One way to test this is to use a validation sample.
  Fit your model to the remaining training data, and see how well it predicts the left-out data."
```python
# Sample 1000 random indices
leaveout = sample(range(len(default)),1000)
# train the model WITHOUT these observations
probatrain = smf.glm(formula=my_formula ,
    data=default.drop(leaveout), family=sm.families.Binomial()).fit()

# predicted probability of default on the left out data
pdefault = probatrain.predict(default.iloc[leaveout])
```
- Idioms: `random.sample(range(n), k)` draws a hold-out of 1000 indices with **no seed**. The model is fit on
  `df.drop(idx)` and predicted on `df.iloc[idx]`. This matches `drop` against `iloc` because `default` has a
  RangeIndex.
- A **random** hold-out is used here because rows are unrelated credit-card clients (cross-sectional). Contrast the
  Lending Club time split on p.30.

### p.18 OOS fit plot and OOS deviance R^2
- Boxplots of fitted probability on the 1000 left-out observations look like p.14: class 0 median ≈ 0.17, class 1
  median ≈ 0.27, box ≈ 0.20–0.47.
- "For the left-out data, we get $D_0 = 1063.897$, $D = 934.8424$, $R^2 = 0.1213$." (Check: $1-934.8424/1063.897 = 0.1213$.)
- "**Since the sample is random, you might get different results.**" This is why seeds must be set (the exam uses
  `random_state=7034`).
- Note: the slide does not say whether the OOS $D_0$ uses the training $\bar y$ or the left-out $\bar y$. L4's
  convention for $R^2_{OOS}$ is the **training/in-sample mean** benchmark, so follow that when in doubt.
- Here OOS R^2 (0.1213) is about equal to in-sample (0.12). With 29,000 observations and ~24 coefficients there is
  little overfitting. Contrast p.54, where the time split gives a real drop.

---

## 4. Classification framework and the Bayes classifier (L6 p.19–22)

### p.19 Classification problem
- Training observations $(x_1,y_1),\dots,(x_n,y_n)$, with $y_i$ qualitative: membership in a category $\{1,2,\dots,m\}$.
- "The classification problem: given new $x_i^{new}$ what is the class label $f(x_i^{new})$?"
- Quality is assessed by the **misclassification risk**, the probability of falsely classifying a new observation:
  $$P(Y^{new}\neq f(x_i^{new})).$$
- "This quantity is unknown but can be estimated by a **proportion of wrong labels in a validation dataset**. Good
  classifiers yield small risk." So accuracy = 1 − misclassification rate, measured **out of sample**.

### p.20 Bayes classifier
- "There is actually a theoretically optimal classifier, the **Bayes classifier**, which minimizes the misclassification
  risk." It assigns each observation to the most likely class given its predictors, i.e. it chooses the
  $j\in\{1,\dots,m\}$ for which $P(Y=j\mid X=x)$ is largest.
- "Unfortunately $P(Y=j\mid X=x)$ is not known. Bayes classifier is **unattainable gold standard**. But! We can estimate it!"

### p.21 Analogy to regression (table)

| | Classification | Regression |
|---|---|---|
| Risk | classification risk $P(Y\neq f(X))$ | squared error risk $E(Y-f(X))^2$ |
| Minimizer | Bayes classifier $f(x)=\arg\max_j P(Y=j\mid X=x)$ | conditional expectation $f(x)=E(Y\mid X=x)$ |

### p.22 Classifiers: parametric vs non-parametric
- There are many ways to estimate $P(Y=j\mid X=x)$ from training data:
  - **Parametric:** assume $P(Y=j\mid X=x,\beta)$ is a specific function of unknown $\beta$ and learn those.
    "Sounds familiar? Logistic regression."
  - **Non-parametric:** estimate $P(Y=j\mid X=x)$ directly "without estimating any parameters." Example: **K-nearest
    Neighbors (KNN)**.
  - (Trees and forests, taught in L8, are also non-parametric estimates of $P(Y=j\mid X=x)$ that predict the majority
    class in a region. L6 does not say this.)

---

## 5. K-nearest neighbours (L6 p.23–28)

### p.23 KNN algorithm
- Idea: estimate $P(Y=j\mid X=x_{new})$ locally from the labels of similar observations already seen.
  "KNN: what is the most common class around x?"
  1. Take the K nearest neighbours $\mathbf{x}_{i_1},\dots,\mathbf{x}_{i_K}$ of $x_{new}$ in the training data.
     Nearness is Euclidean distance: $\sqrt{\sum_{j=1}^p (x_j - x_{ij})^2}$.
  2. Estimate $\widehat P(Y=j\mid X=x_{new}) = \frac{1}{K}\sum_{k=1}^K \mathbf{1}_{\{y_{i_k}=j\}}$.
  3. Select the class with the highest $\widehat P(Y=j\mid X=x_{new})$ (Bayes classifier).
- "Since we're calculating distances on X, **scale Matters!** We'll use Python's `StandardScaler` function in sklearn
  to divide each $x_j$ by $\mathrm{sd}(x_j)$. The new units of distance are in standard deviations."
  - `StandardScaler` also centres, which does not change distances. By the course's train/test discipline (p.30), fit
    the scaler on training data only.

### p.24 KNN voting illustration
- Figure: a 2-D toy plot with red, green and cyan points and one grey query point; dashed lines go to its nearest
  neighbours.
- "K-NN's collaborative estimation: Each neighbor votes. Neighborhood is by shortest distance (shown as the dashed
  lines). The relative vote counts provide a **very crude** estimate of probability."
- "For 3-nn, p(blue) = 2/3, but for 4-nn or 2-nn, it's only 1/2." "**Sensitive to neighborhood size** (think about
  extremes: 1 or n)." With K = n, every point gets the overall majority class, which is exactly the majority-class
  baseline.

### p.25 Decision boundaries: K = 3 vs K = 1
- Figure: two panels (K=3, K=1) over a grid on $[0,1]^2$ with about 10 labelled points (black/red). K=1 carves small
  islands around each point.
- "Larger K leads to higher **training error** (proportion of in-sample misclassification rate)."
- "Smaller K leads to higher **flexibility** (overfitting and poor out-of-sample misclassification rate)."
  - This is the bias-variance / complexity trade-off in classification. Exam 1.1 asks what happens on either side of
    the chosen `max_leaf_nodes`: fewer leaves behave like large K (underfit), more leaves like small K (overfit).

### p.26 KNN from scratch (code)
```python
def euc_distance(arr1, arr2):
    distance = np.sqrt(sum((arr1-arr2)**2))
    return distance

def knn_classifier(X, y, testVector, k):
    distance_list = [euc_distance(testVector, x) for x in X]
    neighbors = np.argsort(distance_list)[:k]
    count = Counter(y[neighbors][0])
    return count.most_common()[0][0]
```
- Idioms: `np.argsort(...)[:k]` and `collections.Counter(...).most_common()[0][0]`. "Sort by distance, find the index
  of the closest K neighbors and select the class with highest possibility."
- Transcription caveat: `Counter(y[neighbors][0])` is copied as printed. For a 1-D `y` the intended call is
  `Counter(y[neighbors])`. The `[0]` works only for particular array shapes, so do not copy it blindly.

### p.27 KNN with sklearn (code, **house style**)
```python
from sklearn.neighbors import KNeighborsClassifier

nn1 = KNeighborsClassifier(n_neighbors=1, n_jobs=-1).fit(X_train, y_train)
nn5 = KNeighborsClassifier(n_neighbors=5, n_jobs=-1).fit(X_train, y_train)
y_pred_1 = nn1.predict(X_test)
y_pred_5 = nn5.predict(X_test)
```
- "Alternatively, you may use some existing package, which is less transparent but offers more features."
- "You set `n_neighbors` to specify how many neighbors get to vote." "You set `n_jobs` to use parallel computing."
- Pattern: construct, `.fit(X_train, y_train)` chained, `.predict(X_test)`. Names are `X_train, y_train, X_test, y_test`.

### p.28 KNN with simulated data (figure only)
- Two panels titled "2-Class classification (k = 100)": concentric **circles** and interleaved **moons** (they look
  like sklearn `make_circles` / `make_moons`, though the slide does not name them). KNN with k=100 recovers the
  non-linear boundaries: a disc for the circles and a curved boundary for the moons.

---

## 6. Lending Club case study: data design (L6 p.29–34)

### p.29 Data (revisits HW 1)
- LendingClub (LC) is a US P2P lender. It was the first to register its offerings as securities with the SEC and to
  offer loan trading on a secondary market.
- Goal: "understand how data is used to minimize the risk of losing money while lending to customers in P2P business."
- Two loan statuses: **Fully paid (0)** and **Charged-off (1)**, i.e. defaulted. The positive class is the bad and
  rarer outcome.
- 27 explanatory variables, 15 of them categorical.
- "We use loans issued in 2012–2018 as the training sample and loans issued in 2019–2020 as the test sample."

### p.30 Training and test samples (**key course convention**)

| | training sample | test sample |
|---|---|---|
| loans issued | Mar 2012 – Dec 2018 | Jan 2019 – Sep 2020 |
| number of loans | 1,199,593 | 79,391 |
| default rate | 19.6% | 17.0% |

- "The split is by **issue date**: every loan in the training sample was issued **before** every loan in the test
  sample, **as it would be if the model were put to use**."
- "Missing values are filled with **training-sample medians**, and the dummy variables are defined on the training
  sample; the same transformation is then applied to the test sample." So every preprocessing step is fit on the
  training data only and then applied to the test data unchanged. This is the course's anti-leakage rule.
- "We leave out LC's own `grade`, `sub_grade` and `int_rate` when we build our models, and use them to compare our
  predictions with LC's assessment." These are outputs of LC's own risk model, and int_rate is set by LC from the
  grade. Using them as predictors would embed another model's forecast.
- The base rate shifts between samples (19.6% to 17.0%), so class proportions are not stationary over time.

### p.31 grade (figure)
- Stacked bar chart of the proportion Charged Off vs Fully Paid by loan grade A–G. The charged-off share rises
  roughly from 0.05 (A), 0.12 (B), 0.22 (C), 0.30 (D), 0.38 (E), 0.45 (F) to 0.50 (G).
- "As grade varies from A to G, the default probability increases."

### p.32 int_rate vs grade (figure)
- Boxplots of `int_rate` by grade rise monotonically from about 7% (A) to about 27% (G).
- "As default probability increases, interest rate increases. Both are determined by LC."

### p.33 int_rate vs sub_grade (figure)
- The same monotone pattern holds across sub-grades A1…G5. "We will use these variables to compare with LC's model."

### p.34 Correlations with default (figure)
- Horizontal bar chart titled "correlation with loan status". Approximate values: mort_acc ≈ −0.075,
  annual_inc ≈ −0.04, revol_bal ≈ −0.02, total_acc ≈ −0.01, pub_rec_bankruptcies ≈ +0.02, pub_rec ≈ +0.02,
  open_acc ≈ +0.03, installment ≈ +0.055, revol_util ≈ +0.058, loan_amnt ≈ +0.07, dti ≈ +0.10, int_rate ≈ +0.26.
- "Some covariates are better **discriminators** (dti, mort_acc)." mort_acc is the number of mortgage accounts; dti is
  the borrower's debt-to-income ratio. int_rate has the largest correlation but is LC's own output, so it is excluded
  (p.30).
- This is the classification version of the correlation screening in L4.

---

## 7. KNN results on Lending Club and pros/cons (L6 p.35–37)

### p.35 Predicted number of defaults by grade, test sample (figure)
- Grouped bars for True, 1-nn, 5-nn and 20-nn by grade. Approximate values:
  - True: A≈1,700, B≈3,000, C≈4,200, D≈4,050, E≈450, F≈G≈0. The test sample holds almost no F/G loans.
  - 1-nn: A≈3,450, B≈3,450, C≈3,250, D≈2,600, E≈200. It over-predicts defaults in good grades.
  - 5-nn: about 800–900 in each of A–D. 20-nn: almost zero everywhere.

### p.36 By subgrade, and accuracy
- Same comparison by subgrade. "5-nn and 20-nn clearly **underestimate** # of default, but in terms of accuracy, they
  **dominate** 1-nn!"
- **Accuracy $P(\hat Y = Y)$: 1-nn 0.72, 5-nn 0.80, 20-nn 0.83.**
- Warning: higher accuracy came from predicting "no default" almost always. The 20-nn accuracy of 0.83 equals the
  majority-class rate (83% non-defaulters, p.51). So accuracy alone rewards ignoring the minority class.

### p.37 KNN pros and cons
- Pros: simple; naturally handles multiple categories (m > 2); "will outperform linear classifiers when the decision
  boundary is non-linear."
- Cons:
  - "Computing neighbors can be costly for large n and p."
  - "KNN's do not perform variable selection. How did we avoid the **curse of dimensionality**?"
  - "Choosing K can be tricky. **Cross-validation works, but is unstable.**"
  - "And the classification is very sensitive to K."
  - "All you get is a classification, with only rough local probabilities. Without good probabilities we cannot assess
    uncertainty."
  - "There is no natural or simple way to handle categorical variables."
- CV for choosing K is only mentioned here. L6 shows no CV procedure or code.

---

## 8. Logistic regression as a classifier (L6 p.38–42)

### p.38 Logistic vs KNN
- "Many decisions can be reduced to binary classification. $y_i\in\{0,1\}$." KNN is non-parametric; "a useful
  parametric alternative for two categories is the logistic regression."
- Compared to KNN:
  - "Logistic regression yields **parametric decision boundaries** (linear, quadratic depending on our regression
    equation) ⟹ it is principled but it can be flexible."
  - "Logistic regression is a '**global**' method, i.e. it uses all the training data to estimate probabilities, not
    just neighbors ⟹ probability estimates are **more stable**."
  - "Logistic regression can do **variable selection**! (yay!)" (mentioned only; the slide gives no L1 details)
  - "Note that Scikit-learn's `LogisticRegression` function utilizes **Ridge penalty by default**; we set
    **`penalty=None`** to obtain the **MLE**." (**house style / pitfall**)
  - Code implied: `from sklearn.linear_model import LogisticRegression; LogisticRegression(penalty=None).fit(X_train, y_train)`.
    The slide shows only the kwarg, not the full call.

### p.39 Logistic regression on simulated data, degree 1 (figure)
- Panels "2-Class classification (degree = 1)" on the circles and moons data: a **linear** boundary cuts straight
  through both and fails badly on the circles.

### p.40 Degree 2 (circles) and degree 3 (moons) (figure)
- With polynomial terms of degree 2, the logit boundary becomes an ellipse that separates the circles. With degree 3
  it becomes a curved boundary that separates the moons. Flexibility comes from the regression equation, i.e. feature
  expansion. The slide gives no code (e.g. no `PolynomialFeatures`).

### p.41 Logistic regression by grade, test sample (figure)
- "Logistic Regression (grade)": True vs Predict number of defaults. Predicted ≈ 20 (A), 70 (B), 200 (C), 1,130 (D),
  170 (E), against true ≈ 1,700 / 3,000 / 4,200 / 4,050 / 450. At the 0.5 cut-off it heavily under-predicts defaults.

### p.42 By subgrade, and accuracy
- Predicted defaults concentrate in the D/E subgrades.
- **Accuracy $P(\hat Y=Y)$: Logit 0.82.** This is below the 83% majority-class rate (p.51).

---

## 9. From probabilities to decisions (L6 p.43–47)

### p.43 Bayes decision rule
- Logistic regression estimates $P(Y=1\mid X=x,\beta)$. The Bayes decision rule classifies as a defaulter, $\hat Y=1$, when
  $$P(Y=1\mid X=x,\hat\beta) > 0.5.$$
- "Recall that the Bayes rule is optimal in terms of misclassification error $P(\hat Y\neq Y)$."
- sklearn `.predict()` on a classifier uses exactly this rule: majority vote, or probability > 0.5 for binary.

### p.44 Two ways to be wrong
$$P(\hat Y\neq Y) = P(\hat Y=1\mid Y=0)\,P(Y=0) + P(\hat Y=0\mid Y=1)\,P(Y=1).$$
- **False positive (Type I error):** predict $\hat Y=1$ when $Y=0$, i.e. classify as defaulters when they are not.
- **False negative (Type II error):** predict $\hat Y=0$ when $Y=1$, i.e. classify as non-defaulters when they in fact
  default.
- "Misclassification error weights different errors by the relative weights of different labels in the population."

### p.45 Asymmetric error importance and imbalanced classes
- "Both false positives and false negatives are bad, but sometimes one of them can be much worse ⟹ the cost can be
  **asymmetric**!" Examples: investment, medical diagnosis.
- "**Maximizing accuracy does not appear to be the most relevant criterion in practice.**"
- "In many scenarios in practice, the class label for worse outcome is also **rare**!" Examples: fraud detection,
  default prediction, rare disease.
- "Maximizing accuracy puts more weight on the larger class. $P(Y=0)/P(Y=1)$ is large ⟹ far more false negatives!"

### p.46 Minimizing cost: action-cost matrix
- For every $1 loaned, lenders make 25¢ in interest if it is repaid and lose $1 on default.

| | payer | defaulter |
|---|---|---|
| loan | −0.25 | 1 |
| no loan | 0 | 0 |

- With estimated default probability $p$, expected profit from lending is positive iff
  $$(1-p)\tfrac14 - p > 0 \iff \tfrac14 > \tfrac54 p \iff p < 1/5.$$
- "From this simple matrix you should lend whenever probability of default is less than 0.2."
- General form (my derivation, not on the slide): lend iff $p < \dfrac{g}{g+\ell}$, with gain $g$ on a payer and loss
  $\ell$ on a defaulter. Here $0.25/1.25 = 0.2$. **The threshold comes from the costs, not from 0.5.**

### p.47 Minimizing cost: Lending Club (figure + table)
- Figure: profit per dollar of applications vs the threshold on predicted default probability ("lend when
  $\hat p <$ threshold"). It peaks at about 0.064 near 0.2 and flattens to about 0.037 ("lend to everyone") as the
  threshold rises to 1. A dashed red line marks threshold = 0.2.

| rule | profit per $ | funded |
|---|---|---|
| lend to everyone | 0.037 | 100% |
| lend when $\hat p<0.5$ | 0.042 | 98% |
| lend when $\hat p<0.2$ | **0.064** | 67% |

- "Lend when the predicted probability of default is below a threshold. Profit is 25¢ per dollar repaid and −$1 per
  dollar defaulted, **computed in the test sample**."
- "The best threshold in the test sample is 0.195, essentially the 1/5 implied by the cost matrix."
  - Reading: the threshold is fixed **ex ante** from the cost matrix. The test-sample optimum (0.195) serves only as a
    check. Choosing the threshold by maximising test-sample profit would be tuning on the test set.
  - Check: lend-to-everyone profit ≈ $0.25(1-0.17) - 0.17 = 0.0375$, which matches 0.037 at the 17% test default rate.
  - The "lend to everyone" rule is the **benchmark**; the model's value is the gain over it (0.064 vs 0.037).

---

## 10. Confusion matrix, error rates, sensitivity/specificity (L6 p.48–51)

### p.48 Confusion matrix layout (figure, transcribed)

| | Predicted Negative (N) − | Predicted Positive (P) + |
|---|---|---|
| **Actual Negative −** | True Negatives (TN) | False Positives (FP), Type I error |
| **Actual Positive +** | False Negatives (FN), Type II error | True Positives (TP) |

- "An alternative approach is to think about False/True Positive Rates in tandem when evaluating a given classification rule."

### p.49 Confusion matrix in Python (code, **house style**)
```python
from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(y_test, y_pred)
```
- "`y_test` and `y_pred` are true values and predicted values." sklearn ordering is rows = actual, columns = predicted,
  giving `[[TN, FP], [FN, TP]]` for labels (0,1).
- Pitfall: the slide assigns the result to the name `confusion_matrix`, which shadows the function. A second call would
  fail, so use a different variable name.
- Lending Club test sample (n = 79,391):
```text
5-nn   array([[63116,  2769],
              [12791,   715]])
20-nn  array([[65792,    93],
              [13473,    33]])
logit  array([[64846,  1039],
              [12889,   617]])
```
- Derived (my arithmetic): actual negatives = 65,885, actual positives = 13,506 (17.0%), so the **majority-class
  ("nobody defaults") accuracy = 0.830**.

| | accuracy | FPR | FNR | TPR (sens.) | TNR (spec.) | precision | # predicted 1 |
|---|---|---|---|---|---|---|---|
| majority rule | 0.830 | 0 | 1 | 0 | 1 | n/a | 0 |
| 5-nn | 0.804 | 0.042 | 0.947 | 0.053 | 0.958 | 0.205 | 3,484 |
| 20-nn | 0.829 | 0.001 | 0.998 | 0.002 | 0.999 | 0.262 | 126 |
| logit | 0.825 | 0.016 | 0.954 | 0.046 | 0.984 | 0.373 | 1,656 |

  **At the 0.5 cut-off, no model beats the majority-class accuracy.** This is why p.57 insists on reporting accuracy
  "with the majority-class rate next to it".

### p.50 FP and FN rates (formulas + code comments)
- False Positive Rate: # misclassified as positive / # actual negatives, $\dfrac{FP}{FP+TN}$.
- False Negative Rate: # misclassified as negative / # actual positives, $\dfrac{FN}{FN+TP}$.
```text
# False Positive Rate
FPR = FP/(FP+TN)   5-nn: 0.042   20-nn: 0.001   logit: 0.016
# False Negative Rate
FNR = FN/(FN+TP)   5-nn: 0.947   20-nn: 0.998   logit: 0.954
```
- "We can choose a threshold to control false positive rate, while minimizing false negative rate."
  This is the Neyman-Pearson-style use of the threshold.

### p.51 Sensitivity and specificity
- **Sensitivity**: the proportion of true y = 1 classified as such (True Positive Rate) $= TP/(TP+FN) = 1-FNR$.
- **Specificity**: the proportion of true y = 0 classified as such (True Negative Rate) $= TN/(TN+FP) = 1-FPR$.
- "A rule is sensitive if it predicts 1 for most y = 1 observations, and specific if it predicts 0 for most y = 0
  observations."
- "All our rules are similarly specific, though not sensitive at all. This has to do with the fact that the **majority
  class (83% of the test sample) is non-defaulters**."

---

## 11. ROC, AUC, precision-recall, lift (L6 p.52–56)

### p.52 ROC curve: sensitivity vs 1 − specificity
- Figure "ROC (Logit)": TPR vs FPR, with the logit curve above the diagonal and the legend "Logit (area = 0.66)".
- "**R**eceiver **O**perating **C**haracteristic: performance measure of a classifier as cutoff varies. A tight fit has
  the curve forced into the top-left corner."

### p.53 AUC
- "The Area Under the Curve (AUC) is the measure of the ability of a classifier to distinguish between classes and is
  used as a summary of the ROC curve."
  - AUC = 1: perfect separation.
  - The higher the AUC, the better the classifier.
  - 0.5 < AUC < 1: acceptable.
  - AUC = 0.5: either random guessing or constant for all. **A constant (majority-class) predictor has AUC 0.5.**
  - AUC = 0: all predictions are off.
- AUC is threshold-free and uses the ranking of predicted probabilities. The slide shows no code (no `roc_curve` /
  `roc_auc_score`).

### p.54 AUC for Lending Club: in-sample vs test (**course convention**)
- "The logistic regression is estimated on loans issued in 2012–2018 and evaluated on loans issued in 2019–2020."

| | training sample | test sample |
|---|---|---|
| AUC | 0.70 | 0.66 |

- "**The test-sample AUC is the relevant one**: the model is used on loans issued **after** the ones it was estimated on."
- "Performance is lower out of sample than in sample, as we have seen with regression."
- "For comparison, a logistic regression on `mort_acc` and `dti` alone has a test-sample AUC of 0.58." So a simple
  model serves as a **benchmark**.

### p.55 ROC and precision-recall with rare events (figure)
- Left, ROC: default rate 17% (AUC = 0.66) and default rate 1% (AUC = 0.65 in the legend). The two curves nearly
  overlap.
- Right, Precision-Recall: default rate 17% (PR-AUC = 0.27), with precision ≈ 0.38 at low recall declining to ≈ 0.17
  at recall = 1; default rate 1% (PR-AUC = 0.02), with precision ≈ 0.02 flat.
- **Precision**: the proportion of predicted $\hat Y=1$ that are true $y=1$, $TP/(TP+FP)$. "The PR curve plots
  precision against sensitivity (recall) as the cutoff varies."
- Experiment: "Same model, same non-defaulters, but keep only enough defaulters for a 1% default rate":
  - "AUC is unchanged (0.66): **sensitivity and specificity do not depend on the class proportions**." The text says
    0.66 and the legend says 0.65, a minor inconsistency on the slide.
  - "The area under the PR curve falls from 0.27 to 0.02: **precision does**."
- "**With rare events (fraud, default), report precision and recall in addition to the ROC curve.**"
- Implicit baseline: a random classifier's precision equals the prevalence (the PR curve ends at ≈ 0.17 at recall 1).

### p.56 Precision at k and lift (table)
- "Rank the test-sample loans by predicted probability of default and look at the riskiest k%."

| riskiest k% | loans | default rate | lift |
|---|---|---|---|
| 1% | 793 | 0.39 | 2.3 |
| 5% | 3,969 | 0.34 | 2.0 |
| 10% | 7,939 | 0.32 | 1.9 |
| 20% | 15,878 | 0.29 | 1.7 |
| 50% | 39,695 | 0.24 | 1.4 |

- **Lift** = default rate among the selected loans / default rate in the whole test sample (0.17):
  $$\text{lift}(k) = \frac{\text{precision@}k}{\bar y_{test}}.$$
  "It is a useful summary when only a fixed number of cases can be reviewed."
- This is the classification cousin of a portfolio sort: rank on the model score and inspect the top bucket.
  Monotone decline in the default rate across k shows the ranking has content.

---

## 12. Reporting a classifier (L6 p.57), the course's reporting template

"When you report a classifier, include:"
1. **The question and the target:** what decision the model informs, and what exactly is predicted.
2. **The data and the split:** the training and test samples, with dates and sample sizes.
3. **The model:** the estimator and how any tuning parameter was chosen.
4. **Performance in the test sample against a benchmark:** accuracy with the **majority-class rate next to it**; AUC,
   and precision and recall when the event is rare.
5. **The decision rule:** the threshold, and the costs that justify it.

"The final project asks for a one-page summary of this kind for the model you submit; the template arrives with the
project."
- Note: the provided final notebook (`Final_Autumn_2026-1.ipynb`) contains no separate one-page classifier template.
  Problem 3 instead asks for a research-paper write-up whose Data and Methodology sections (split, dates, tuning,
  benchmarks, decision rule) cover the same checklist. Apply the checklist to Problem 1's reporting and to the
  Problem 3 write-up.

---

## 13. Formula sheet (all from L6)

| Quantity | Formula | Slide |
|---|---|---|
| Conditional mean of binary y | $E[y\mid x]=P(y=1\mid x)$ | p.4 |
| Logit model | $P(y=1\mid x)=e^{x'\beta}/(1+e^{x'\beta})$ | p.6 |
| Log-odds | $\log[p/(1-p)]=x'\beta$ | p.6 |
| Odds multiplier | $\exp(\beta_j)$ per unit increase of $x_j$ | p.9 |
| Likelihood | $L(\beta)=\prod p_i^{y_i}(1-p_i)^{1-y_i}$ | p.10 |
| Deviance (binomial) | $-2\sum[y_i\log p_i+(1-y_i)\log(1-p_i)] \propto \sum[\log(1+e^{x_i'\beta})-y_ix_i'\beta]$ | p.10 |
| Deviance (Gaussian) | $\propto\sum(y_i-x_i\beta)^2$; $\hat\sigma^2=\frac{1}{n-p-1}\sum\hat\epsilon_i^2$ ("Scale") | p.12 |
| df of deviance | nobs − df(regression) | p.11 |
| Null deviance (logit) | $D_0=-2\sum[y_i\log\bar y+(1-y_i)\log(1-\bar y)]$ | p.13 |
| Deviance R^2 | $R^2=1-D/D_0$ | p.13, p.18 |
| Gradient descent | $\beta_{t+1}=\beta_t-\gamma_t\nabla loss(\beta_t)$ | p.15 |
| Misclassification risk | $P(Y^{new}\neq f(x^{new}))$, estimated by the validation error rate | p.19 |
| Bayes classifier | $f(x)=\arg\max_j P(Y=j\mid X=x)$ | p.20–21 |
| KNN distance | $\sqrt{\sum_j (x_j-x_{ij})^2}$ | p.23 |
| KNN probability | $\hat P(Y=j\mid x)=\frac1K\sum_k 1\{y_{i_k}=j\}$ | p.23 |
| Accuracy | $P(\hat Y=Y)$ | p.36, 42 |
| Bayes decision rule (binary) | $\hat Y=1$ iff $\hat P(Y=1\mid x)>0.5$ | p.43 |
| Error decomposition | $P(\hat Y\ne Y)=P(\hat Y=1\mid Y=0)P(Y=0)+P(\hat Y=0\mid Y=1)P(Y=1)$ | p.44 |
| Cost threshold | lend iff $(1-p)\frac14-p>0 \iff p<1/5$ | p.46 |
| FPR, FNR | $FP/(FP+TN)$, $FN/(FN+TP)$ | p.50 |
| Sensitivity (TPR, recall) | $TP/(TP+FN)$ | p.51, 55 |
| Specificity (TNR) | $TN/(TN+FP)$ | p.51 |
| ROC | TPR vs FPR = sensitivity vs 1 − specificity, as the cutoff varies | p.52 |
| Precision | $TP/(TP+FP)$ | p.55 |
| Lift at k | default rate in top k% / overall test default rate | p.56 |

---

## 14. Code idioms (the course's house style from L6)

| Purpose | Code | Slide |
|---|---|---|
| Logistic regression (MLE) | `smf.glm(formula=my_formula, data=df, family=sm.families.Binomial()).fit()` | p.8 |
| Summary / dispersion check | `proba.summary()` → "(Dispersion parameter for binomial family taken to be 1)" | p.11 |
| Coefficients by name | `b['AGE']`, `b['PAY_0']` (params Series) | p.9 |
| Predicted probabilities | `proba.predict(default[0:4])`; `probatrain.predict(default.iloc[leaveout])` | p.16–17 |
| Random hold-out | `leaveout = sample(range(len(default)),1000)`; fit on `default.drop(leaveout)` | p.17 |
| Standardize for KNN | sklearn `StandardScaler` (divide each $x_j$ by sd) | p.23 |
| KNN by hand | `np.sqrt(sum((a-b)**2))`, `np.argsort(d)[:k]`, `Counter(...).most_common()[0][0]` | p.26 |
| KNN sklearn | `KNeighborsClassifier(n_neighbors=5, n_jobs=-1).fit(X_train, y_train)`; `.predict(X_test)` | p.27 |
| sklearn logistic MLE | `LogisticRegression(penalty=None)` (default is ridge/L2) | p.38 |
| Confusion matrix | `from sklearn.metrics import confusion_matrix; confusion_matrix(y_test, y_pred)` | p.49 |
| Rates | `FPR = FP/(FP+TN)`, `FNR = FN/(FN+TP)` | p.50 |

The slides mention or plot these but show no code: ROC/AUC (`roc_curve`, `roc_auc_score`), PR curves
(`precision_recall_curve`), `predict_proba`, polynomial features, and CV for K. Using the standard sklearn functions
for them is consistent with the lecture's content.

---

## 15. Pitfalls and warnings emphasised (with slides)

1. **Check the dispersion parameter is 1** to confirm you actually ran logistic rather than Gaussian regression (p.11).
2. **newdata must match the format** of the original data when predicting (p.16).
3. **Random splits give different results** ("Since the sample is random, you might get different results", p.18),
   so fix seeds.
4. **Scale matters for distance-based methods** (KNN): standardize with StandardScaler (p.23).
5. **KNN is sensitive to K.** Extremes are 1 or n (p.24). Small K overfits and gives poor OOS error; large K raises
   training error (p.25). The classification is very sensitive to K, and CV for K "works, but is unstable" (p.37).
6. **KNN probabilities are crude** (p.24, p.37). KNN does no variable selection, so it faces the curse of
   dimensionality (p.37). It has no natural way to handle categorical variables (p.37).
7. **sklearn `LogisticRegression` is penalized (ridge) by default.** Set `penalty=None` for the MLE (p.38).
8. **A linear logit boundary fails on non-linear structure** (circles/moons, p.39). Flexibility requires terms in the
   regression equation (p.40).
9. **Accuracy is misleading with imbalanced classes.** 5-nn/20-nn "clearly underestimate # of default, but in terms
   of accuracy, they dominate 1-nn" (p.36). Maximizing accuracy puts weight on the larger class and yields far more
   false negatives (p.45). The rules are "specific, though not sensitive at all" because 83% are non-defaulters
   (p.51). Every model's accuracy was at or below the 83% majority rate (p.36, p.42, p.49, p.51).
10. **Costs are asymmetric.** Pick the threshold from the cost matrix (p = 0.2), not 0.5 (p.45–47).
11. **ROC/AUC ignore class proportions.** With rare events also report precision/recall (PR-AUC falls from 0.27 to
    0.02 while AUC is unchanged) (p.55).
12. **Out-of-sample is lower than in-sample** (AUC 0.70 to 0.66). The **test-sample** number is the relevant one
    (p.54).
13. **Split time-ordered data by date**, "as it would be if the model were put to use" (p.30, p.54).
14. **Fit preprocessing on the training sample only.** Impute medians and define dummies on train, then apply them to
    test (p.30).
15. **Exclude variables that are another model's output** (LC's grade, sub_grade, int_rate) and use them only as a
    comparison (p.30–33).
16. **Base rates drift over time** (19.6% train vs 17.0% test, p.30). This affects precision and the majority-class
    baseline.
17. The learning rate is "the most important parameter to tune" in gradient descent (p.15).
18. Code gotchas: the slide assigns `confusion_matrix = confusion_matrix(...)`, which shadows the function (p.49).
    `Counter(y[neighbors][0])` works only for particular array shapes (p.26).

---

## 16. Datasets and worked examples

- **Default.csv** (UCI Default of Credit Card Clients): 30,000 clients, about 6,636 defaults. Logistic regression via
  statsmodels GLM gives coefficients AGE 0.007 (odds ×1.007) and PAY_0 0.577 (odds ×1.781). Residual deviance is
  27,877 on 29,976 df and null deviance 31,705, so $R^2=0.12$. A random 1000-observation hold-out gives
  $D_0=1063.897$, $D=934.8424$, $R^2=0.1213$. The first four predicted probabilities are 0.5055, 0.1500, 0.2004,
  0.2428 (p.7–18).
- **Lending Club** (revisits HW1): 27 predictors, 15 of them categorical. Train is Mar 2012–Dec 2018 (1,199,593 loans,
  19.6% default); test is Jan 2019–Sep 2020 (79,391 loans, 17.0% default).
  - Models: 1-nn, 5-nn, 20-nn and logit. Accuracies are 0.72, 0.80, 0.83 and 0.82. Confusion matrices and FPR/FNR are
    on p.49–50.
  - Cost-based lending: profit 0.064/$ at threshold 0.2 vs 0.037 lending to all. The test-optimal threshold is 0.195.
  - AUC is 0.70 train and 0.66 test, against 0.58 for a mort_acc + dti benchmark. PR-AUC is 0.27 at a 17% default
    rate and 0.02 at 1%. Lift table on p.56 (p.29–56).
- **Simulated 2-D data:** a toy KNN voting example (p.24), K = 1 vs K = 3 boundaries (p.25), KNN with k = 100 on
  circles/moons (p.28), and logistic regression with degree 1/2/3 on circles/moons (p.39–40).

---

## 17. Boundaries: what L6 does NOT teach, or only mentions in passing

- **Not in L6 at all:** decision trees, `max_leaf_nodes`, pruning, Gini/entropy impurity, random forests, bagging,
  gradient boosting, `feature_importances_`, permutation importance, `plot_tree`. All of these come from L8.
- **No CV machinery:** no K-fold, `StratifiedKFold`, `cross_val_score` or time-series CV. The only mention is
  "Cross-validation works, but is unstable" for choosing K (p.37). The OOS tools actually shown are a random 1000-obs
  hold-out (p.17) and a date-based train/test split (p.30).
- **Mentioned in passing only:**
  - Variable selection with logistic regression, i.e. a penalized logit (p.38); no L1 or `C` details.
  - Ridge as the sklearn default (p.38).
  - Gradient descent in deep learning (p.15).
  - Polynomial logistic regression (figures only, p.39–40).
  - Multi-class (m > 2) handled naturally by KNN (p.37). Multinomial logit is not covered.
  - Controlling FPR while minimizing FNR via the threshold (p.50); no procedure given.
- **Not covered:** standard errors, z-tests or likelihood-ratio tests for logit coefficients (the summary is shown only
  for the dispersion/deviance lines). Also not covered: class weights, resampling/SMOTE, probability calibration,
  LDA/QDA, naive Bayes, SVM, and threshold selection by CV.
- **No code** for ROC/AUC/PR/lift. The concepts, definitions and interpretations are taught.
- The **one-page summary template** "arrives with the project" (p.57). It does not appear as a separate item in the
  final notebook.

---

## 18. Mapping to the final exam

### Problem 1 (Trees & Ensembles, Social_Network_Ads, 5-fold StratifiedKFold, random_state=7034)
- **1.1 majority-class baseline.**
  - p.57 says report "accuracy with the majority-class rate next to it".
  - p.51, 36, 42 and 49 show that 20-nn (0.83) and logit (0.82) only matched or trailed the 83% majority rate.
  - p.53 gives a constant predictor AUC 0.5. p.24 notes that with K = n everyone gets the majority class.
  - So "nobody purchases" accuracy = 1 − mean(Purchased), and it is the number to beat.
  - Accuracy = $P(\hat Y=Y)$ is estimated out of sample as 1 − the error rate on held-out data (p.19). CV folds play the
    validation role.
  - The tree-size sweep reads like p.25: too few leaves means high bias / high training error (like large K); too many
    means overfitting and worse OOS error (like small K, or the unpruned tree).
  - The tie rule "take the smaller" matches the course's preference for the simpler model under equal OOS performance.
    L6 does not state this rule; it is consistent with L4.
- **1.2 plain-English description.** The p.57 checklist ("the question and the target", "the decision rule") and the
  plain-language interpretation style of p.9 and p.46 apply. A tree leaf predicts the majority class, the Bayes rule
  with $\hat p>0.5$ (p.20, p.43).
- **1.3 RF/boosting vs tree.** p.37–38 argue: non-parametric local methods win when the boundary is non-linear, but
  their probabilities are less stable, and global/parametric methods are more stable. p.39–40 show that a simple
  boundary suffices when the structure is simple. p.36 shows small accuracy differences can hide very different
  behaviour. With 400 rows, 3 features and 2 dominant ones (Age, Salary), the tree already captures most of the
  structure and CV accuracy is noisy. Report the result whichever way it comes out and compare it with the majority
  baseline. Learning rate = shrinkage (p.15) only by analogy.
- **1.4 in-sample impurity vs held-out permutation importance.**
  - L6 teaches no importance measures (L8).
  - It supplies the principle: "the test-sample AUC is the relevant one" and "performance is lower out of sample than
    in sample" (p.54). The OOS evaluation on left-out data is on p.17–18.
  - So the held-out permutation importance is the one to show a decision maker, because in-sample measures reward
    what the model memorized.
  - The 70/30 stratified split mirrors the hold-out design (p.17) with class proportions preserved. The base-rate
    concern is on p.30 and p.55.
- **1.5 shuffled CV on monthly data of one stock.**
  - Predicting the sign of next month's return is binary classification ("greater or less than", "Profit or Loss",
    p.3).
  - L6's prescription for time-ordered data is to **split by date**, "every loan in the training sample was issued
    before every loan in the test sample, as it would be if the model were put to use" (p.30), and to evaluate on
    later data: "the test-sample AUC is the relevant one: the model is used on loans issued after the ones it was
    estimated on" (p.54).
  - Preprocessing (medians, dummies, scalers) must be fit on the training period only (p.30).
  - Direction of bias: shuffled CV trains on the future and tests on the past, and serial dependence/overlap leaks
    information. So the reported accuracy is **biased upward**, i.e. optimistic. Compare the IS vs OOS drop of
    0.70 to 0.66 (p.54).
  - The base rate (share of up-months) can drift across time (19.6% to 17.0%, p.30). So also report the majority-class
    ("always up") accuracy measured on the test period (p.57).
  - What to do instead: chronological train/test, or an expanding/rolling window (L5/HW5), with no shuffle.

### Problem 2 (VaR pipeline trained on its own output)
- L6 has **little direct content**: no VaR, quantiles, martingales or simulation.
- Tangential links:
  - The $\frac{1}{n-p-1}$ unbiased variance correction and the "Scale" (p.12) parallel `ddof=1`.
  - Rare-event evaluation (p.45, p.55) matters for a 1% tail event: accuracy-type summaries are uninformative when the
    event is rare.
  - Model outputs should not be fed back as inputs, as with the exclusion of LC's own grade/int_rate (p.30). This
    loosely echoes "training on your own output".
- Do not claim L6 supports specific Problem 2 methods.

### Problem 3 (research project: panel forecasting and portfolios)
- **3-features:**
  - Fit every transformation on training data only (medians, dummies; p.30). This carries over to scalers,
    imputation and standardization in the P3 harness; trailing-only standardization for macro is required by the
    notebook.
  - Standardize before distance-based methods like KNN (p.23). KNN is explicitly allowed in P3.
  - KNN does no variable selection and suffers the curse of dimensionality (p.37). This argues against KNN on the
    wide macro file x17..x164.
  - Leave out variables that are another model's output (p.30).
  - Screen predictors by correlation with the target (p.34).
- **3-evaluation:**
  - Time-ordered split with test data after training data (p.30, p.54). Report the test-period metric, not in-sample
    (p.54).
  - Benchmark against simple rules: majority rate (p.57) or a small benchmark model (the mort_acc + dti logit,
    p.54), analogous to $R^2_{OOS}$ vs the trailing mean and vs zero.
  - Deviance R^2 = $1-D/D_0$ with $D_0$ from the constant prediction (p.13, p.18) has the same structure as
    $R^2_{OOS}$.
  - If sign-classification of returns is used (e.g. logit on $1\{r>0\}$): use sklearn `LogisticRegression` with
    `penalty=None` for the MLE, or a deliberately tuned penalty (p.38). Report accuracy vs the majority rate, and AUC
    (p.53, 57).
- **3-portfolio:**
  - Forecasts become positions through a decision rule with a threshold justified by costs (p.46–47): "the threshold,
    and the costs that justify it" (p.57).
  - Compare against the naive rule ("lend to everyone" is to the lending model as equal weight is to a forecast
    portfolio). Compute profit in the test sample (p.47). Fix rule parameters ex ante rather than optimizing them on
    the test sample (0.195 vs 1/5, p.47).
  - Precision-at-k and lift from ranking on predicted scores (p.56) is the classification analogue of a
    long-top/short-bottom sort or tercile analysis.
- **3-write-up:** the p.57 checklist maps onto the research-paper sections.
  - Question/target: the Introduction.
  - Data and split with dates and sizes: the Data and Methodology sections.
  - Estimator and how tuning was chosen: Methodology.
  - Test-sample performance against a benchmark: Results.
  - Decision rule and costs: the portfolio section.
