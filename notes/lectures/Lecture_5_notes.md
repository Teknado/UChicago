# Lecture 5 — Regularization for Linear Models: Ridge & Lasso (BUSN 41210, Dacheng Xiu, Sept 16 2026)

Source: `/home/user/UChicago/Lecture_5.pdf` (60 PDF pages). All 60 pages were viewed as rendered images; the text extraction was used for wording.

**Citation convention.** "L5 p.X" means **PDF page X**. The slide-footer numbers differ after p.28, because PDF pp.29–35 are the build-up animation of one K-fold slide (footer 29). Mapping:

| PDF page | footer # |
|---|---|
| 1–28 | 1–28 |
| 29–35 | 29 (animation builds; only p.35 shows the footer) |
| 36–60 | 30–54 (footer = PDF page − 6) |

**One-line summary.** Regularized linear regression (penalized MLE): ridge (L2), lasso (L1), and the elastic-net penalty (formula only). How to pick λ: K-fold CV for i.i.d. data, AIC/BIC as alternatives, and expanding/rolling windows for time series. Then how to design an out-of-sample experiment and report R²_OOS honestly: benchmark choice, leakage checklist, and warnings about AI-written code. The running example is a macro forecasting dataset (743 × 119).

---

## 0. Outline (L5 p.2)
- Maximum Likelihood Estimation
- Regularization: Ridge, Lasso, Elastic Net
- Cross Validation
- Design an OOS Experiment: Splitting Sample; Covariates Standardization; Choice of Parameter Grid

---

## 1. MLE and deviance (L5 pp.3–5)

**p.3 Maximum Likelihood Estimation**
- Likelihood is a function of the unknown parameters given the data: $\mathrm{LHD}(\beta) = P(\text{data}\mid\beta)$.
- ML estimation: $\hat\beta = \arg\max_\beta \mathrm{LHD}(\beta)$. "ML estimates are those parameter values β that are most likely to have generated our data."

**p.4 Minimizing Deviance**
- Deviance is the distance between data and fit, and you want it as small as possible:
  $$Dev(\beta) = -2\log \mathrm{LHD}(\beta) + C$$
  ("C is a constant you can mostly ignore.")
- Deviance is useful for comparing models. It is a goodness-of-fit (GOF) measure that "plays the role of residual sums of squares for a broader class of models (logistic regression etc.)".
- "We'll think of deviance as a **cost to be minimized**." Minimize deviance ⇔ maximize likelihood.

**p.5 Least-squares and deviance in linear regression**
- Model: $y\mid \mathbf{x} \sim N(\mathbf{x}'\beta, \sigma^2)$. Density of $N(\mu,\sigma^2)$: $\exp[-(y-\mu)^2/2\sigma^2]/\sqrt{2\pi\sigma^2}$.
- For n independent observations: $\prod_{i=1}^n \mathrm{pr}(y_i\mid \mathbf{x}_i) \propto \exp\left[-\tfrac12\sum_{i=1}^n (y_i-\mathbf{x}_i'\beta)^2/\sigma^2\right]$
- Hence $Dev \propto \frac{1}{\sigma^2}\sum_{i=1}^n (y_i - \mathbf{x}_i'\beta)^2$.
- "Minimizing deviance is the same as least squares! And thus the MLE minimizes our sum of squared errors."
- House convention: the **Gaussian deviance is the sum of squared errors**. The code later uses `deviance(..., family='gaussian')` as SSE, and MSE = dev / n (p.52).

## 2. Why regularize: variance of OLS, bias–variance (L5 pp.6–10)

**p.6 Variance of OLS**
- OLS is unbiased: $E(\hat\beta_{OLS}) = \beta$.
- OLS can have large variance: $\mathrm{Var}(\hat\beta_{OLS}) = \sigma^2 (X'X)^{-1}$. As p approaches or exceeds n, $X'X$ gets close to singular or poorly conditioned.

**p.7 Bias–Variance Tradeoff.** Figure only: the classic 2×2 dartboard (low/high bias × low/high variance).

**p.8 Regularization, the general idea.** "Impose restrictions on the MLE estimates such that they are suitably disciplined (stable and purposeful)." We would like $\hat\beta$ to (i) be less variable, (ii) predict better, and (iii) have exact zeros. "We will **penalize** solutions β that are not desirable. Ultimately, we are departing from optimality to stabilize the system (bias-variance tradeoff)."

**p.9 MLE vs penalized MLE**
- $\hat\beta_{MLE} = \arg\min_\beta -\frac{2}{n}\log \mathrm{LHD}(\beta)$. "$\hat\beta_{MLE}$ can only be obtained when p < n, it can be unstable and it cannot achieve exact zeros."
- Penalized MLE: $\hat\beta_{PMLE} = \arg\min_\beta -\frac{2}{n}\log\mathrm{LHD}(\beta) + pen(\beta)$. "Penalty incurs **cost** when choosing solutions that are far from our preferences."

**p.10 Decision theory: cost in estimation**
- Estimation: deviance is the cost of the distance between data and model.
- Testing: "Since $\hat\beta_j = 0$ is *safe*, it should cost us to decide otherwise."
- ⇒ "The cost of $\hat\beta$ is deviance plus a penalty away from zero."

## 3. Regularized regression, general form (L5 p.11)
$$\hat\beta_{PMLE} = \arg\min_\beta \left\{-\frac{2}{n}\log\mathrm{LHD}(\beta) + \lambda\sum_j pen(\beta_j)\right\}$$
- λ > 0 is the penalty weight and *pen* is a cost function. pen(β) is lowest at β = 0 and "we pay more for |β| > 0".
- Three panels plot pen(β_j) on [−1, 1]: LASSO (V shape), RIDGE (parabola), ELASTIC NET (in between: rounded near 0, kinked at 0).
- **Options: ridge $\beta^2$, lasso $|\beta|$, elastic net $\alpha\beta^2 + (1-\alpha)|\beta|$, ...**
  - CAUTION (implementation note, not on the slide): in the slide's elastic-net formula **α weights the L2 part**. sklearn's `ElasticNet(alpha, l1_ratio)` minimizes $\frac{1}{2n}\|y-Xw\|^2 + \alpha\,\text{l1\_ratio}\|w\|_1 + \frac{\alpha}{2}(1-\text{l1\_ratio})\|w\|_2^2$. There `alpha` is the overall λ and `l1_ratio` weights the **L1** part. Do not mix up the two α's.

## 4. Ridge (L5 pp.12–16)

**p.12 Ridge Regression**
$$\hat\beta_{Ridge}(\lambda) = \arg\min_\beta \sum_{i=1}^n\Big(y_i - \sum_{j=1}^p \beta_j x_{ij}\Big)^2 + \lambda\sum_{j=1}^p\beta_j^2$$
- λ ≥ 0 is the complexity parameter that controls shrinkage: **higher λ ↔ more shrinkage**.
- Equivalent constrained form: $\hat\beta_{Ridge}(\tau) = \arg\min_\beta \sum_i (y_i - \sum_j\beta_j x_{ij})^2$ s.t. $\sum_{j}\beta_j^2 \le \tau$.
- **Closed form:** $\hat\beta_{Ridge}(\lambda) = (X'X + \lambda I)^{-1}X'Y$.
- Other names: Tikhonov regularization, weight decay, L² regularization.
- Implementation note (not on slide): sklearn `Ridge(alpha)` minimizes $\|y-Xw\|_2^2 + \alpha\|w\|_2^2$, so `alpha` = the slide's λ directly. The intercept is not penalized.

**p.13 Ridge as a constrained optimization program.** Figure: SSE level-curve ellipses around the OLS estimate (cross at about (3, 1.8)) and circular L2 constraint contours centred at 0. The red dots trace the ridge solutions as λ varies, running smoothly from OLS toward the origin.

**p.14–15 Why ridge "works": OLS on an augmented dataset**
- $X_\lambda = \begin{pmatrix} X \\ \sqrt{\lambda} I_p\end{pmatrix}$ (n rows of x's, then p rows with √λ on the diagonal), and $Y_\lambda = \begin{pmatrix} Y \\ 0_p\end{pmatrix}$.
- $\hat\beta_{Ridge}(\lambda) = (X_\lambda' X_\lambda)^{-1}X_\lambda' Y_\lambda = \left[(X', \sqrt\lambda I)\binom{X}{\sqrt\lambda I}\right]^{-1}(X',\sqrt\lambda I)\binom{Y}{0} = (X'X+\lambda I)^{-1}X'Y$. The slide's last line has the typo "XY"; it should be X'Y.
- "$X_\lambda'X_\lambda$ will not be singular even if $X'X$ is singular."

**p.16 Ridge path.** Simulated linear regression with p = 10 covariates (x1..x10) and T = 100 observations. The figure plots ridge estimates against λ/T from 0 to 10. All coefficients shrink smoothly toward 0: positives from above, negatives (x4, x10) from below.
- "FACT: The ridge path is continuous, and the estimates go to zero only as λ → ∞" (no exact zeros).

## 5. Lasso (L5 pp.17–24)

**p.17 Penalization and automatic variable selection**
- "The minimum of a smooth + spiky function can be at the spike." Three panels: deviance (minimized at β̂ = 1), penalty |β| (spike at 0), and deviance + penalty (minimized at 0).
- "The PMLE estimator is **shrunk to zero** ⇒ automatic variable selection." LASSO has this property. "You can think of the LASSO as the *modern least squares*."

**p.18 LASSO ("least absolute shrinkage and selection operator")**
$$\hat\beta_{LASSO}(\lambda) = \arg\min_\beta \sum_{i=1}^n\Big(y_i-\sum_{j=1}^p\beta_j x_{ij}\Big)^2 + \lambda\sum_{j=1}^p|\beta_j| \quad(\text{or min MSE s.t. } \textstyle\sum_j|\beta_j|\le\tau)$$
Unlike ridge:
- There is no closed-form solution in general, "but efficient algorithms place it at roughly same computational cost".
- For large enough λ, **some coefficient estimates will be exact zeros**.
- So lasso is a **model selection** alternative to stepwise regression.
- Other names: L¹ regularization, basis pursuit.
- Implementation note (not on slide): sklearn `Lasso(alpha)`, `LassoCV`, and `lasso_path` minimize $\frac{1}{2n}\|y-Xw\|_2^2 + \alpha\|w\|_1$. That is the slide's λ divided by 2n, so grid values are not directly comparable to the slide's λ.

**p.19 LASSO as a constrained program.** Figure: SSE ellipses around OLS and diamond-shaped L1 contours. The red dotted lasso solution path reaches the axis (β₂ = 0) before the origin, so the solution sits at a corner and gives exact zeros.

**p.20 LASSO, Ridge, Subset under orthonormal design ($X'X = I$)**

| Estimator | Formula |
|---|---|
| Best subset (size K) | $\hat\beta_j\,\mathbb{1}\{|\hat\beta_j^{ols}| \ge |\hat\beta^{ols}_{(K)}|\}$ |
| Ridge | $\hat\beta_j^{ols}/(1+\lambda)$ |
| Lasso | $\mathrm{sign}(\hat\beta_j^{ols})(|\hat\beta_j^{ols}| - \lambda)^+$ |

- Best subset drops every variable whose coefficient is smaller than the Kth largest ("**hard thresholding**").
- Ridge does **proportional shrinkage**.
- Lasso "translates each coefficient by constant factor λ, truncating at zero" ("**soft thresholding**").
- The figure has three panels (Best Subset, Ridge, Lasso) showing each estimator as a function of β̂_ols against the 45° line.

**p.21 LASSO paths**
- The lasso minimizes $-\frac{2}{n}\log\mathrm{LHD}(\beta) + \lambda\sum_j|\beta_j|$. How do we pick λ? Solve for a **sequence** $\lambda_1 > \lambda_2 > \dots > \lambda_T$, then apply model selection tools to choose the best $\hat\lambda$.
- **Path estimation algorithm:** start with λ₁ so big that $\hat\beta = 0$. For t = 2…T, update $\hat\beta$ to be optimal under $\lambda_t < \lambda_{t-1}$ (warm starts).
- Since $\hat\beta$ changes smoothly along the path, it is **fast** (each update is easy) and **stable**: "optimal λ_t may change a bit from sample to sample, but that won't affect the model much."
- "It's a better version of forward stepwise selection."

**p.22 Path plots.** Figure "Lasso coefficients as a function of alpha": standardized coefficients (y) against alpha on a log scale (about 10^-3.5 to 1). This is apparently the macro dataset, with many lines. All coefficients are 0 for α ≳ 0.3. "The algorithm moves *right to left*. The y-axis is standardized β̂ (each line a different β̂_j) as a function of λ_t."

**p.23 Scale matters** (key convention for standardization)
- Penalization is sensitive to the scale of x₁…x_p. For example, xβ has the same effect as (2x)β/2, but |β| is twice as much penalty as |β/2|.
- "It would not be 'fair' to penalize β's equally if x's were not on the same scale."
- You can multiply β_j by sd(x_j) in the cost function to standardize, i.e. minimize $-\frac{2}{n}\log\mathrm{LHD}(\beta) + \lambda\sum_j \mathrm{sd}(x_j)|\beta_j|$.
- ⇒ "β_j's penalty is calculated per effect of 1SD change in x_j." In practice (code later), X is standardized with `StandardScaler` fitted on training data.

**p.24 How much regularization?**
- The LASSO's "sparse regularization" auto-selects variables. "Sound too good to be true? You need to choose λ."
- "Think of λ > 0 as a signal-to-noise filter: like squelch on a radio."
- "We'll use **cross validation** or **information criteria** to choose."
- Path algorithms are key: they let us quickly enumerate candidate models, and "This set is stable, so selected 'best' is probably pretty good."

## 6. Model selection philosophy (L5 pp.25–26)

**p.25 Prediction vs interpretability**
- Model selection serves two tasks: (i) learning about the data-generating mechanism (which variables contribute, and can we interpret them?), and (ii) prediction.
- "None of your models will be 'true' for complicated HD systems. Instead, just try to get as close to the truth as possible."
- **Parsimony principle:** "If two models do similarly well in predicting the unseen data, choose the simpler one." Overly simple models **underfit**; complicated models **overfit** and make noisy predictors. "The goal is to find the sweet spot in the middle."

**p.26 Prediction-driven model selection ("It is All About Prediction")**
- Recipe: (1) find a manageable set of candidate models (fast to fit all of them), which is what the LASSO path provides. (2) Choose the candidate with the best predictive performance **on unseen data**.
- "We need to *estimate* the prediction accuracy associated with each model when applied on future unseen data. Recall that predictive performance can be measured with 'deviance'."

## 7. Designing out-of-sample experiments (L5 p.27) — four considerations
- **Sample splitting**
  - Large datasets: a traditional split (e.g. **80/20 or 70/30**) is feasible and generalizes well.
  - Small datasets: "Splitting too much data into validation can hinder effective training."
- **Temporal or sequential data:** "Ensure the order of data is respected to prevent leakage and maintain sequence integrity."
- **Covariate scaling:** "Be mindful of information leakage by **scaling after splitting** the data."
- **Grid selection for tuning parameters:** "Ensure the optimal tuning parameters fall **within the middle of the chosen grid**."

## 8. K-fold cross-validation for i.i.d. data (L5 pp.28–36)

**p.28 Sample splitting for i.i.d. data**
- K-fold CV "guarantees each observation is left-out for validation, and lowers the sampling variance of CV model selection."
- Leave-one-out CV (K = n) "is nice but takes a long time. **K = 5 to 10 is fine** in most applications."

**pp.29–35 K-fold diagram (animation).** Take the dataset, divide it into K folds (K = 6 in the picture). In run k, fold k (orange) is the validation fold and the rest are training folds. Compute MSE_k, the mean squared error over the validation fold. Then
$$\text{MSE} = \frac{1}{6}\sum_{i=1}^6 \text{MSE}_i \quad\Rightarrow\quad \text{"Select } \lambda \text{ which minimizes MSE"} \quad(\text{p.35}).$$
Folds are drawn as contiguous blocks in the picture. General form: $\text{CV}(\lambda) = \frac1K\sum_{k=1}^K \text{MSE}_k(\lambda)$.

**p.36 CV LASSO (algorithm)**
- The lasso path minimizes $-\frac2n\log\mathrm{LHD}(\beta)+\lambda_t\sum_j|\beta_j|$ over $\lambda_1>\lambda_2>\dots>\lambda_T$. This gives T fitted coefficient vectors $\hat\beta_1\ldots\hat\beta_T$, each defining a deviance for new data: $-\log p(\mathbf{y}^{new}\mid \mathbf{X}^{new}\hat\beta_t)$.
- Algorithm:
  1. Set a sequence of penalties λ₁…λ_T.
  2. For each fold k = 1…K, fit the path $\hat\beta^k_1\ldots\hat\beta^k_T$ on **all data except fold k**, and get the fitted deviance **on left-out data**: $-\log p(\mathbf{y}^k\mid \mathbf{X}^k\hat\beta_t)$.
  3. This gives K draws of OOS deviance for each λ_t.
  4. Choose the "best" $\hat\lambda$, then **re-fit the model to all of the data** by minimizing $-\frac2n\log\mathrm{LHD}(\beta)+\hat\lambda\sum_j|\beta_j|$.

## 9. Macro forecasting example: CV LASSO and the grid (L5 pp.37–40)

Dataset: a macroeconomic dataset with **743 observations and 119 variables** (`X1.shape → (743, 119)`, p.45). The target appears to be a monthly macro growth series: p.57 says "forecast no growth surprise at all", and predictions are about 0.002 (p.54). The dataset is not named on the slides. Train/test split with `X_train_scaled, y_train_scaled` (both X and y standardized).

**p.37 Base grid**
```python
alphas = 2**(np.linspace(-12, 1, 50))
K = 10
model_CV = LassoCV(alphas=alphas, cv=K)
model_CV.fit(X_train_scaled, y_train_scaled)
```
Output: Optimal Alpha 0.06075; Number of Variables 22; LassoCV IS R² 0.35; **LassoCV OOS R² −0.03**.
- "Selecting the proper range for the tuning parameter is important."
- If the tuning parameter is too large, the model "might effectively shrink most coefficients to nearly zero, making the variables insignificant".
- If it is too small, "the model will include all variables with little to no penalty".

**p.38 Large-λ grid**
```python
alphas_large = 10**(np.linspace(0, 10, 50))
model_CV_large = LassoCV(alphas=alphas_large, cv=K)
model_CV_large.fit(X_train_scaled, y_train_scaled)
alpha_CV_large = model_CV_large.alpha_
```
alpha_CV_large = 1e10 (the grid edge); 0 variables; IS R² 0.0; OOS R² 0.0 (the null model).

**p.39 Small-λ grid**
```python
alphas_small = 10**(np.linspace(-20, -10, 50))
model_CV_small = LassoCV(alphas=alphas_small, cv=K)
...
alpha_CV_small = model_CV_small.alpha_
```
alpha_CV_small = 1e-10 (again the grid edge); **119 variables**; IS R² 0.52; **OOS R² −0.4**. This is OLS-like overfitting: IS R² goes up and OOS R² falls apart.

**p.40 Optimal λ, CV curve.** Figure: CV mean squared error (y from about 0.74 to 1.01) against log(α) (natural log, −6.5 to +0.7). It is U-shaped with a local dip near −6.5, a global minimum of about 0.74 at log α ≈ −2.8 (= ln 0.06075, the green vertical line), and a flat line at about 1.01 for large α (all-zero model; y is standardized so null MSE ≈ 1). "Again, the routine is most easily understood visually."

## 10. Problems with CV (L5 p.41)
- **Time consuming:** when estimation is not instant, fitting K times can become infeasible even for K in 5–10.
- **Unstable:** "imagine doing CV on many different samples. There can be large variability on the model chosen."
- "Still, some form of CV is used in most DM applications."
- **Don't cheat:** "with the screening cut model we've already used the full n observations to select the strongest variables. It is not surprising they do well 'OOS'."
- **"The rules: never use the same data twice (for selection and validation)."**

## 11. Information criteria as alternatives to CV (L5 pp.42–46)

**p.42 Information criteria (IC)**
- IC "measure how much information is lost when we use our model to represent our data. They approximate distance between a model and 'the ideal'."
- IC allow **in-sample model comparisons** by trading off in-sample deviance against model complexity. Choose the model with minimum IC.
- **AIC = IS Deviance + 2 df.** df = degrees of freedom used in the fit. "For lasso and MLE, this is just the # of nonzero $\hat\beta_j$."

**p.43 AIC overfits in high dimensions**
- AIC estimates OOS deviance: "what your deviance would be on another *independent* sample of size n".
- IS deviance is too small because the model is tuned to this data. "Some deep theory shows that OOS − IS deviance ≈ 2df" ⇒ AIC ≈ OOS deviance.
- "It's common to claim this approx (i.e., AIC) is good for 'big n'. Actually, its only good for **big n/df**."
- "In Big Data, df (# parameters) can be huge. Often df ≈ n. In this case the AIC will be a bad approximation: **it overfits!**"

**p.44 Bayes IC**
- **BIC = IS Deviance + log(n) × df.** It looks like AIC but comes from a very different place.
- $BIC \approx -\log p(M_b\mid\text{data})$, the "probability that model b is true", with $p(M_b\mid\text{data}) = \frac{p(\text{data},M_b)}{p(\text{data})} \propto \underbrace{p(\text{data}\mid M_b)}_{\text{LHD}}\underbrace{p(M_b)}_{\text{prior}}$.
- The prior is your probability that a model is true before seeing data. BIC uses a "unit-info" prior $N[\hat\beta, \frac{2}{n}\mathrm{var}(\hat\beta)^{-1}]$.
- "AIC tries to approx OOS deviance. BIC is trying to get at the 'truth'."

**p.45 Macro example, AIC vs BIC**
- "Our macroeconomic data has 743 observations and 119 variables. It is difficult to claim that we have a big n/df."
- Figure "Information-criterion for model selection": AIC and BIC against α (log scale 1e-5 to about 1). The AIC curve is flat and low, with its minimum near α ≈ 0.015 (blue line). The BIC curve starts high (~2010) and falls to its minimum near α ≈ 0.08 (red line).
- Table:

| | AIC | BIC |
|---|---|---|
| #variables | 57 | 17 |
| R² (IS) | 0.44 | 0.32 |
| OOS R² | −0.22 | −0.01 |

**p.46 IC and CV on the macro data**
```python
model_aic = LassoLarsIC(criterion='aic').fit(X_train_scaled, y_train_scaled)
model_bic = LassoLarsIC(criterion='bic').fit(X_train_scaled, y_train_scaled)
```
- Figure: CV mean squared error with **error bars** against log(α). Vertical dashed lines mark Alpha CV estimate (purple, ≈ −2.8), **Alpha 1se estimate** (grey, ≈ −1.7), Alpha BIC estimate (blue, ≈ −2.5), and Alpha AIC estimate (green, ≈ −4.3).
- Table:

| | AIC | BIC | CV |
|---|---|---|---|
| #variables | 57 | 17 | 22 |
| R² (IS) | 0.44 | 0.32 | 0.35 |
| OOS R² | −0.22 | −0.01 | −0.03 |

- Lesson: AIC under-penalizes when n/df is small (57 vars, worst OOS). BIC is the most parsimonious and has the best OOS here. All three have negative OOS R².
- The **1se rule** appears only in this figure legend. It is not explained anywhere in L5.

## 12. Scaling leakage in LassoCV (L5 p.47)
- "LassoCV() shown above has an issue when it comes to scaling."
- "The correct way to perform cross-validation (CV) is to **scale using only the training data**."
- "In the LassoCV() example above, both the training and validation sets were scaled, which is not ideal." `X_train_scaled` was scaled once on the whole training block, so every internal CV validation fold was scaled with statistics that included it.
- "To resolve this issue, you should implement LassoCV using **Lasso Path**." That means fitting the scaler on the fold's training part only, running `lasso_path`, and scoring deviance on the validation part. A per-fold scaler inside a sklearn `Pipeline` is the standard equivalent; that is an implementation remark and is not on the slides.

## 13. Cross-validation for time-series data (L5 pp.48–55)

**p.48**
- "**Usually, CV is not valid with time-series data.**" Several solutions can be valid.
- **Expanding window CV:** "the training dataset starts with a small subset of the data and incrementally incorporates more data points over time. … the training dataset [grows] to encompass a larger portion of the historical data, while the validation set advances along the time axis."
- **Rolling window CV:** "advancing both the training and validation sets forward in time by a fixed interval … maintaining a constant window size for both training and testing periods."

**p.49 Recursive time-ordered validation and testing.** Diagram with Training in blue, Validation in green, Testing in red, and unused future in grey. In each row the blue training block starts at the **same first observation** and grows by one step. A fixed-length green validation block follows immediately, then the red test point(s). The whole triple moves forward in time row by row. So the design is **train → validation → test, strictly in time order**.

**p.50 Expanding window algorithm (code, verbatim)**
```python
# generate expanding window iterator
def expanding_window_iterator(data, initial_train_size, validation_size, test_size, step):
    tscv_expanding = []
    train_size = initial_train_size
    for i in range(0, len(data) - train_size - validation_size - test_size + 1, step):
        train_index = data[:(i+train_size)].index.values.astype(int)
        tscv_expanding.append(train_index)
    return tscv_expanding
```
- The algorithm "starts with **a fixed starting point** and expands the training set by adding new data points in each iteration."
- "It is important to maintain the same starting point in each expanding window to ensure that the model is trained consistently over time."
- "the model learns incrementally from all available past data, leading to more reliable predictions."
- Only training indices are returned. For window i, validation is the next `validation_size` observations after the training block and test is the next `test_size` after that. This is done inside `lasso_solver`, which is not shown.

**p.51 Parameters used in the macro example**
```python
initial_train_size = 180   # 15 years
validation_size   = 84     # 7 years
test_size         = 12     # 1 year
step              = 12     # window size that expands training data: 1 year
```
Monthly data. The first training set is 15 years, validation is 7 years, and the test block is 1 year. Each iteration adds 12 months to the training set. With n = 743 this gives range(0, 468, 12), i.e. **39 windows** (indexed 0..38) and 39 × 12 = **468 test predictions** (matching the 0..467 rows shown on p.54).

**p.52 Per-window lasso fit and alpha selection (code, verbatim)**
```python
alphas, coefs, _ = lasso_path(X_train_scaled, y_train_scaled, alphas=alphas)
coefs = coefs[0]     # y_train_scaled is 2-D (n,1) -> coefs shape (1, p, n_alphas)

def predict(Xs, coef):             # scale in, y units out
    return yscaler.inverse_transform(
        (Xs @ coef).reshape(-1, 1)).reshape(-1)

for idx, alpha in enumerate(alphas):
    coef = coefs[:, idx]
    dev  = deviance(y_valid.values, predict(X_valid_scaled, coef),
                    family='gaussian')
    dev_ser.loc[alpha] = dev
    mse.loc[alpha]     = dev / len(y_valid)
    best_pred[alpha]   = predict(X_test_scaled, coef)

alpha_best = alphas[np.argmin(mse.values)]
```
- "For each window, we obtain the optimal alpha that minimizes MSE (with validation dataset)."
- "Note: 'y_pred' from the testing dataset has **nothing to do with obtaining optimal alpha**."
- House-style details:
  - X **and** y are standardized with scalers fitted on the window's training data (`yscaler`). Predictions are mapped back to y units with `inverse_transform`.
  - `lasso_path` has no intercept, so centered/scaled data is needed. This is an inference, not stated on the slide.
  - The deviance is Gaussian (SSE) and MSE = dev/len(y_valid).
  - The test prediction for the chosen α uses the coefficients fitted on the **training block only**; as coded, there is no refit on train+validation before predicting the test year.
  - `deviance` is a course helper function (not shown); with family='gaussian' it is the sum of squared errors.

**p.53 Running the solver (code, verbatim)**
```python
expanding_list = expanding_window_iterator(y_full, initial_train_size, validation_size, test_size, step)
alphas = 2**(np.linspace(-10, -1.3, 50))
table_mse, table_alpha, table_model, table_n_features, table_r2, table_pred = lasso_solver(
    X_full, y_full, expanding_list, validation_size, test_size, params=alphas)
```
- The tables show, per window 0..38:
  - `Lasso_mse`: validation MSE of about 0.0018–0.0075.
  - `Alpha_best`: 0.406126, 0.359098, or 0.317515.
  - `n_features`: 1–4 selected variables.
- "Then, we get the smallest MSE with an optimal alpha for every window."
- Observation (mine, not a slide claim): 0.406126 = 2^−1.3 is the **upper edge** of this grid, and 0.359098 and 0.317515 are the next two grid points. So the chosen α sits at or next to the grid boundary in every window shown. By p.27's rule ("optimum within the middle of the grid") a wider grid would be warranted. Worth avoiding in the exam: always check where the chosen hyperparameter falls in its grid.

**p.54 Collecting test predictions (code, verbatim)**
```python
LassoExp_pred_data = []
for i in range(table_pred.shape[1]):
    values_to_insert = table_pred.iloc[:, i].values[0]
    for value in values_to_insert:
        LassoExp_pred_data.append([value])
LassoExp_pred_df = pd.DataFrame(LassoExp_pred_data)
```
Output: 468 rows (0..467) of predictions of about 0.0016–0.0022.

**p.55 Computing OOS R² (code, verbatim)**
```python
test = y_full[initial_train_size + validation_size :
              last_train_data + validation_size + test_size + 1]
trailing_mean = y_full[:last_train_data + validation_size + 1].mean()
dev0 = deviance(test, trailing_mean, family='gaussian')
dev  = deviance(test, LassoExp_pred_df.values, family='gaussian')
LassoExp_R2 = 1 - dev/dev0
```
Output: LassoExp_R2 = 0.01. "Again, you **must not use test data** when you construct the model."
- **Formula (course definition):**
  $$R^2_{OOS} = 1 - \frac{\sum_{t\in\text{test}}(y_t-\hat y_t)^2}{\sum_{t\in\text{test}}(y_t-\bar y^{bench})^2} = 1-\frac{Dev(\text{model})}{Dev(\text{benchmark})}$$
  with benchmark = a **trailing mean** (here) or **zero** (p.56–57).
- Subtlety (my observation): as coded on p.55, `trailing_mean` is a **single** mean of y through the last window's validation block, so for early test years it includes later data. The final-exam notebook (Problem 3, "Two rules") requires the stricter version: "the trailing-mean benchmark for month t is the mean of returns through month t−1 only, updated each month (the expanding-window harness of HW5 Problem 1.1)". Use the per-month recursive trailing mean in the exam.

## 14. What the wrong fold costs (L5 p.56) — key empirical slide
- Left figure, "what each fold validates on": five folds over a training block of about 600 months. Bars show expanding-window folds: fit in blue from month 0 to t, validate on the next block in red, with each fold extending further. Grey ticks show random 5-fold validation points scattered over the whole block, future included.
- Right figure: test-set OOS R² "against the ȳ = 0 benchmark":

| scheme for cutting the training block into validation folds | test OOS R² |
|---|---|
| random 5-fold | −0.041 |
| expanding window | −0.005 |
| rolling window | −0.019 |
| training mean (forecast) | +0.001 |

- "Same panel, same LASSO, same α grid. The *only* change is how the training block is cut into validation folds."
  - Random 5-fold: α = 4.4 × 10⁻⁴, **23 variables**, test score −0.0405.
  - Expanding window: α = 7.0 × 10⁻⁴, **15 variables**, −0.0046. Rolling: −0.0186.
- "**The random fold sees the future, so validation error looks too good at small α and it under-regularizes** — costing 0.036 of out-of-sample R² on a target whose entire signal is smaller than that."
- Direction of bias: random-fold CV on time-ordered data gives an **optimistic** validation score, especially for complex or low-penalty models. That leads to picking an overly complex model, which does worse on the true future test block.

## 15. Reporting an out-of-sample R² (L5 p.57) — three conventions, "you must say which you used"
1. **What is the benchmark?** "Above we used ȳ = 0 — forecast no growth surprise at all. A *test-block* mean is not knowable in advance, so it is **not a legitimate benchmark** for a forecast you claim to have made." Implementation note (not on slide): `sklearn.metrics.r2_score` and `model.score` use the test-block mean as benchmark, so don't report them as R²_OOS. Compute 1 − SSE_model/SSE_bench with a zero or trailing-mean benchmark by hand.
2. **"A negative R² is a result, not a bug."** It says the model is worse than the benchmark, and every model on the previous slide is negative. "**Report it. Do not quietly switch benchmarks until it turns positive.**"
3. **"How many things did you try?"** "If the reported model won a search, the size of the search belongs next to the number."
- **Sanity check:** "published *monthly* stock-level out-of-sample R² runs about **0.3–0.5%**. Any far larger number on financial data is a prompt to go looking for the leak before celebrating."

## 16. The leakage checklist (L5 p.58) — "One Thing to Keep Next to the Keyboard"
Nearly every inflated OOS number comes from one of these recurring mistakes:
1. **A transformation fitted before the split**: a scaler, an imputation, a PCA, a winsorisation cut-off. **FATAL.**
2. **A merge keyed on a period-end rather than a publication date.**
3. **A random fold on time-ordered data.** **FATAL.**
4. **A threshold or hyperparameter chosen on the test block.**
5. **A benchmark quietly changed after the fact.**

"Keep this list. The five patterns above are the leakage questions that matter — ask them of every notebook, yours or a machine's, before you submit any homework or the final project. **The first and the third are graded as fatal.** This is not a marking rubric. It is the list of things that will otherwise make you believe a result that is not there."

## 17. Warning about borrowed (AI) code (L5 p.59)
- AI tools are allowed and encouraged. "But the two lines they most reliably write are the two this lecture just spent twenty minutes on."
  - **`train_test_split(X, y)`**: "its default is **shuffle=True**. On a time-ordered panel that is the random fold from three slides ago, and it cost 0.036 of out-of-sample R². The third pattern on the last slide." Use `shuffle=False` or explicit date-based slicing.
  - **`StandardScaler().fit_transform(X)` before the split**: "the scaler learns the mean and variance of rows you are about to hold out. The first pattern on the last slide. Lecture 7 measures this one." Fit on train, then `transform` the validation/test data.
- "**Neither raises a warning. Both produce a number.** That is why the last slide's list exists, and why you read code before you run it — whoever, or whatever, wrote it." See AI_Coding_Guide.pdf on Canvas.

## 18. Ridge or LASSO? (L5 p.60)
- Both are widely used ML techniques for linear models, and "Both can be easily extended to non-linear models" (not shown).
- The choice depends on your assumptions about the DGP:
  - **LASSO:** best when you believe only a few features matter; useful for feature selection and interpretability.
  - **Ridge:** ideal when most features are relevant but you need to handle multicollinearity or prevent overfitting; "Suitable when features are weak and should still contribute to the model."

---

## A. Course conventions (house style) distilled
1. **Loss = deviance** (−2 log LHD). Gaussian deviance = SSE, and MSE = SSE/n (pp.4–5, 52). CV and model comparison are done in deviance/MSE (pp.26, 35–36).
2. **Penalized objective** is written as $-\frac2n\log\mathrm{LHD}(\beta)+\lambda\sum_j pen(\beta_j)$ (pp.9, 11). Ridge and lasso are also shown in SSE + λ·penalty form (pp.12, 18).
3. **Standardize covariates** so the penalty is fair (p.23). **Scale after splitting**, fitting the scaler on training data only (pp.27, 47, 58–59). In the course code y is standardized too and predictions are inverse-transformed (p.52).
4. **Grid:** log-spaced, 50 points (`2**np.linspace(a,b,50)`, `10**np.linspace(a,b,50)`). The chosen value must fall in the **middle** of the grid (pp.27, 37–39).
5. **i.i.d. data:** K-fold CV with K = 5–10 (p.28). Pick λ minimizing the average fold MSE (p.35), then refit on all data at λ̂ (p.36).
6. **Time series:** no random K-fold (pp.48, 56, 58). Use expanding window (fixed start) or rolling window, ordered **train → validation → test** (pp.48–51). Choose α per window on the validation block only and predict the next test block (p.52). Example sizes: 180/84/12, step 12 (p.51).
7. **R²_OOS = 1 − Dev(model)/Dev(benchmark)** on the test predictions (p.55). The benchmark is a **trailing mean** (p.55) or **zero** (pp.56–57), stated explicitly. Never the test-block mean (p.57).
8. Report **negative R²** honestly, report the **size of the search**, and sanity-check against 0.3–0.5% monthly (p.57).
9. **Parsimony:** when models predict about equally well on unseen data, choose the simpler one (p.25).
10. **IC:** AIC = IS Dev + 2df and BIC = IS Dev + log(n)·df, with df = # nonzero coefficients. AIC is unreliable when n/df is small (pp.42–45).
11. **Never use the same data twice** (for selection and validation) (p.41). Test data is never used to build the model or pick α (pp.52, 55).
12. Five leakage patterns; #1 (transform before split) and #3 (random fold on time data) are **graded as fatal** (p.58).

## B. Pitfalls and warnings (with pages)
- OLS variance $\sigma^2(X'X)^{-1}$ explodes as p → n (p.6). MLE is unavailable for p ≥ n, is unstable, and gives no exact zeros (p.9).
- Penalization is scale-dependent, so standardize (p.23).
- Small datasets: don't over-allocate to validation (p.27).
- Grid mis-specification: too-large grid → null model (0 vars, R² 0). Too-small grid → all 119 vars, IS R² 0.52, OOS R² −0.4 (pp.37–39). A boundary optimum signals a bad grid (p.27). Note that the slide's own expanding-window α sits at the grid edge (p.53).
- CV is time-consuming and unstable (p.41). Screening on the full sample and then "validating" is cheating (p.41).
- AIC overfits when df is close to n; it is only good for big n/df (p.43). With 743/119, AIC picks 57 vars and gets OOS R² −0.22 (pp.45–46).
- LassoCV on pre-scaled data leaks scaling into the validation folds; implement it via lasso_path (p.47).
- CV is usually invalid for time series (p.48). Keep the same starting point in the expanding window (p.50).
- Test predictions play no part in choosing α (p.52), and the test data must not be used to construct the model (p.55).
- A random fold sees the future → optimistic validation → under-regularization → −0.036 of R²_OOS (p.56).
- Test-block mean is an illegitimate benchmark. Don't switch benchmarks. Report the search size. Treat large R² as a leak signal (p.57).
- Five leakage patterns (p.58). `train_test_split` defaults to shuffle=True, and `StandardScaler().fit_transform` before the split leaks. Neither warns (p.59).

## C. Datasets and worked examples
- **Simulated linear regression**, p = 10, T = 100: the ridge path (p.16).
- **2-D toy geometry**: SSE ellipses with L2/L1 constraint sets (pp.13, 19). The deviance + |β| spike illustration (p.17).
- **Orthonormal design** thresholding comparison (p.20).
- **Macro forecasting dataset**, 743 × 119 (p.45), target a monthly growth series (pp.54, 57). It is used for:
  - lasso path plot (p.22)
  - LassoCV with three grids (pp.37–40)
  - AIC/BIC/CV comparison (pp.45–46)
  - expanding-window lasso, 39 windows × 12 months, R²_OOS = 0.01 vs trailing mean (pp.51–55)
  - random vs expanding vs rolling fold comparison vs ȳ = 0, on a ~600-month training block (p.56)

## D. Boundaries: in passing or out of scope in L5
- **Elastic net**: only the outline entry and the penalty formula αβ² + (1−α)|β| (pp.2, 11). No estimator slide, no code, no tuning of α.
- **Best subset / hard thresholding**: only in the orthonormal comparison (p.20). **Stepwise / forward stepwise regression**: mentioned only as what lasso improves on (pp.18, 21).
- **Leave-one-out CV**: mentioned, not recommended (p.28).
- **Screening cut model**: referenced from elsewhere as a cheating example (p.41).
- **BIC derivation (unit-info prior)** and the "deep theory" OOS − IS ≈ 2df: stated, not derived (pp.43–44).
- **1se rule**: appears only as a figure legend entry (p.46). It is not taught.
- **Rolling-window CV**: described (p.48) with results shown (p.56), but no code.
- **Fold-wise LassoCV via lasso_path**: recommended (p.47), but only the single-window lasso_path code is shown (p.52). The `lasso_solver` and `deviance` helpers are not shown.
- **Logistic regression / GLM deviance**: mentioned only as a context where deviance applies (p.4).
- **Extensions to non-linear models**: asserted, not shown (p.60).
- **Lecture 7** will measure the scaler-leakage cost (p.59), a forward reference.
- PCA, imputation, winsorisation appear **only** as examples of transformations that must be fitted after the split (p.58). They are not taught here.
- **Not in L5 at all:** trees, CART, pruning, random forests, boosting, variable importance, stratified K-fold, classification accuracy, VaR, portfolio construction, Sharpe ratios, transaction costs, volatility scaling, cross-sectional standardization, placebo tests.

## E. Mapping to the final exam

**Problem 1 (Trees & Ensembles, Social_Network_Ads, i.i.d. cross-section)**
- **1.1 majority-class baseline, then tree size sweep:**
  - "Benchmarks before models" spirit, and a model must be compared to a stated benchmark (p.57).
  - K-fold CV for i.i.d. data, K = 5 fine (p.28). Average the fold scores and pick the tuning value that optimizes them (p.35).
  - Tie → smaller tree follows the **parsimony principle** (p.25).
  - "Either side of the optimum": underfit on the small side and overfit on the large side (p.25, bias–variance p.7–8, U-shaped CV curve p.40).
  - The grid {2,…,12} should contain the optimum in its interior (p.27).
  - Classification loss: deviance generalizes beyond regression (p.4). Accuracy itself is not defined in L5.
- **1.3 RF/GB vs tree, "given size/shape of data":**
  - Bias–variance tradeoff (pp.7–8).
  - Small datasets (p.27: splitting off too much hurts training).
  - Low-dimensional p (3 features) vs n = 400. Regularization pays most when p is large relative to n (pp.6, 9, 43).
  - L5 has no ensemble content.
- **1.4 70/30 stratified split; impurity (in-sample) vs permutation (held-out) importance:**
  - A 70/30 split is the "traditional split" (p.27).
  - In-sample criteria are optimistic because "IS deviance is too small, since the model is tuned to this data" (p.43).
  - "Never use the same data twice (for selection and validation)" (p.41).
  - Prediction vs interpretability (p.25) frames "which to show a decision maker".
- **1.5 rows as months of one stock, shuffled CV wrong:**
  - Core support: "Usually, CV is not valid with time-series data" (p.48).
  - Use expanding or rolling windows with train → validation → test order (pp.48–51).
  - A random fold on time-ordered data is **fatal** (p.58). `train_test_split` defaults to shuffle=True (p.59).
  - **Direction of bias (p.56): the random fold sees the future, so validation error looks too good (optimistic accuracy) and the procedure under-regularizes (picks too complex a tree). The true forward-looking performance is worse.**
  - Alternatives: expanding-window (fixed start, p.50) or rolling-window CV, with validation strictly after training and a final untouched test block (pp.49, 52, 55).

**Problem 2 (self-retrained VaR pipeline)**
- L5 covers no VaR or simulation. Loosely relevant background:
  - Gaussian likelihood/MLE and deviance (pp.3–5). The Gaussian MLE of σ² divides by n; the exam uses ddof=1, so note the difference. This remark is mine.
  - Bias–variance (pp.7–8), relevant to "loss of information vs accumulation of noise".
  - "Never use the same data twice" and "don't cheat" (p.41) as the conceptual cousin of training on your own output.
  - Honest reporting of unwelcome results (p.57).
- No direct methods for 2.1–2.4 come from L5.

**Problem 3 (research project: cross-country return prediction)**
- **3-models:**
  - Ridge (pp.12–16) and lasso (pp.17–24) are the core linear regularized models the notebook lists.
  - Elastic net is allowed but only sketched (p.11). Watch the α/l1_ratio convention.
  - Choose ridge vs lasso by DGP belief (p.60): many weak characteristics/macro → ridge; few strong ones → lasso.
  - Tuning via CV or IC (pp.24, 36, 42–46). Prefer BIC over AIC when n/df is small (pp.43–46).
- **3-features:**
  - Standardize predictors before penalizing (p.23). The notebook's cross-sectional within-month z-scoring uses only same-month info, so it is not a cross-time leak.
  - Standardize macro with **trailing** info only, and fit any scaler/imputation/PCA/winsorization within the training window (pp.27, 47, 58 #1, 59).
  - Lag macro and characteristics: "a merge keyed on a period-end rather than a publication date" (p.58 #2) is the lag/publication-delay issue.
  - Missing values: no full-sample imputation, since imputation is listed among the transforms that must not be fitted before the split (p.58).
- **3-evaluation:**
  - The notebook explicitly cites "the expanding-window scheme of Lecture 5". Implement per pp.48–55: fixed start, train → validation → test, α picked on validation, refit each step. Also compare rolling and static schemes as on p.56.
  - Report **R²_OOS = 1 − SSE_model/SSE_bench** (p.55) against the **recursive trailing mean** (through t−1, updated monthly, per the notebook) and against **zero** (pp.56–57).
  - Never use the test-block mean, and never `sklearn.r2_score` on the test block (p.57).
  - Report negative R² as a result and report the number of specifications tried (p.57). Sanity-check against 0.3–0.5% monthly (p.57).
  - Run the five-item leakage checklist on the notebook; items 1 and 3 are fatal (p.58).
  - Make sure the chosen α is not at the grid edge (pp.27, 37–39, 53).
  - Macro-adds and placebo question: parsimony (p.25) and search-size disclosure (p.57). The placebo is not in L5.
- **3-portfolio:** L5 has no portfolio content (no Sharpe, costs, risk parity, TSMOM). Only the general OOS discipline carries over: benchmarks stated in advance and not switched (p.57).
