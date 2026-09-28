# BUSN 41210 Financial Analytics — Lecture 2: Linear Regression Models

Study notes with slide citations. Source: `/home/user/UChicago/Lecture_2.pdf` (106 pages; PDF page = slide number, cited "L2 p.X").
Instructor: Dacheng Xiu, Chicago Booth. Date on title slide: September 9, 2026 (L2 p.1).
All 106 pages were checked against the rendered slides, not just the extracted text. Formulas and code are transcribed from the rendered images.
The house-data numbers (b0, b1, SSR, SSE, R^2, s, standard errors) were re-computed from the 15 data points on p.5 and match the slides exactly.

---

## 0. One-paragraph summary / what this lecture is for

L2 covers **simple linear regression (SLR)** from start to finish:
- the least-squares (LS) line and its closed form;
- the fitted-value/residual decomposition and ANOVA;
- R^2;
- the SLR probability model `Y = b0 + b1 X + eps`, `eps ~ N(0, sigma^2)`;
- the single-factor/CAPM application (AAPL on S&P 500, weekly 2010–2011);
- the error-variance estimator `s^2 = SSE/(n-p)` and degrees of freedom;
- the sampling distributions of b0, b1 and the sample mean, plus the CLT;
- **what a standard error really is**, shown by Monte-Carlo refits and bootstrap resampling (p.78–80, new-style slides);
- Student-t, confidence intervals, hypothesis tests and p-values;
- **prediction intervals**, including the statsmodels `get_prediction` idiom.

The house style is **statsmodels formula API** (`smf.ols`) plus **numpy with `ddof=1`** and **scipy.stats.t**. sklearn `LinearRegression` is named once.

The lecture does **not** cover:
- multiple-regression inference;
- out-of-sample R^2;
- cross-validation, regularization or time-series dependence corrections.

"Robust" standard errors are only mentioned in passing (p.80).

---

## 1. Outline (L2 p.2)
1. Simple Linear Regressions
2. Prediction and Linear Regression Model
3. Example: Single Factor Model
4. Sampling Distributions, Central Limit Theorem
5. Confidence Interval, Hypothesis Testing
6. Prediction Interval

---

## 2. Slides 3–10: Setup, house-price example, eyeball line

- **p.3 — General regression setup.**
  - Y = response/outcome variable. X1,...,Xd = explanatory/input variables.
  - General relationship: `Y = f(X1, X2, ..., Xd) + e`.
  - Linear case: `Y = b0 + b1 X1 + b2 X2 + ... + bd Xd + e`.
  - This is the only place multiple regressors appear. The rest of the lecture is single-X.
- **p.4 — Problem framing.**
  - Problem: predict market price from observed characteristics.
  - Solution: use sales data with known prices and build "a decision rule that predicts price as a function of the observed characteristics".
  - Y = price (thousands of $). X = size (thousands of sq ft).
  - Terms used: dependent (output) variable, explanatory (input) variable.
- **p.5 — House data table (n = 15):**

  | Size | 0.80 | 0.90 | 1.00 | 1.10 | 1.40 | 1.40 | 1.50 | 1.60 | 1.80 | 2.00 | 2.40 | 2.50 | 2.70 | 3.20 | 3.50 |
  |---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
  | Price | 70 | 83 | 74 | 93 | 89 | 58 | 85 | 114 | 95 | 100 | 138 | 111 | 124 | 161 | 172 |

- **p.6 — Scatterplot.** "It is much more useful to look at a scatterplot". View the data as points in the X × Y plane.
- **p.7 — Eyeball line.** "Appears to be a linear relationship ... As size goes up, price goes up." The line is fit by the "eyeball" method.
- **p.8 — Line equation `Y = b0 + b1 X`.**
  - b0 = intercept, in units of Y ($1,000).
  - b1 = slope, in units of Y per unit of X ($1,000 per 1,000 sq ft).
  - **Units matter.** This is relevant to standardization later.
- **p.9 — Eyeball line values.** Diagram of intercept and slope. Eyeball line has b0 = 35, b1 = 40.
- **p.10 — Prediction by reading off the line.** `Ŷ = 35 + 40(2.2 * 1,000 sq ft) = $123,000`. "Conversion from 1,000 sq ft to $1,000 is done for us by the slope coefficient (b1)."

## 3. Slides 11–18: Fitted values, residuals, least squares

- **p.11** — "Can we do better than the eyeball method?" Fit a line by minimizing how far the **fitted value** is from the realized value. That gap is the **residual**.
- **p.12** — Fitted value: `Ŷ_i = b0 + b1 X_i`. The slide typesets it as `b0 + b1 X1`, a typo.
- **p.13** — Residual: `e_i = Y_i − Ŷ_i`. So `Y_i = Ŷ_i + (Y_i − Ŷ_i) = Ŷ_i + e_i`.
- **p.14 — Least-squares idea.**
  - We want all residuals small. All zeros would be a perfect line. There is a "trade-off between moving closer to some points".
  - Fitting process: give weights to all residuals, then minimize the "total".
  - **LS chooses b0, b1 to minimize** `sum_{i=1}^N e_i^2`.
- **p.15 — LS objective (figure shows positive and negative residuals):**
  `sum_{i=1}^n e_i^2 = sum_{i=1}^n (Y_i − Ŷ_i)^2 = sum_{i=1}^n (Y_i − [b0 + b1 X_i])^2`
- **p.16 — "LS chooses a different line from ours".**
  - Code: `model = smf.ols('Price ~ Size', data=house).fit()`, then `model.params`.
  - Output: Intercept 38.884683, Size 35.385963.
  - Figure compares "Our line" (orange) with the "LS line" (green dashed).
- **p.17 — Closed form from correlations and means (code):**
  ```python
  b1 = np.corrcoef(price, size)[0, 1] * np.std(price, ddof=1) / np.std(size, ddof=1)
  b0 = np.mean(price) - np.mean(size) * b1
  result = pd.DataFrame({'b0': [b0], 'b1': [b1]})
  print(result)        # b0 = 38.884683, b1 = 35.385963
  ```
  "Our scaled correlation and 'means on the line' approach is the same as the least squares estimates."
- **p.18 — Summary.**
  - Python's `LinearRegression().fit(X,Y)` from `sklearn.linear_model`, or `smf.ols('Y ~ X', data).fit()` from `statsmodels.formula.api`, fits the LS line `Ŷ = b0 + b1 X` minimizing `sum (Y_i − Ŷ_i)^2 = sum e_i^2`.
  - **LS formulas:** `b1 = r_xy * s_y / s_x` and `b0 = Ȳ − b1 X̄`.

## 4. Slides 19–25: Why LS works — orthogonality of residuals

- **p.19** — Plot of fitted values against x. It is a perfectly straight line, annotated `corr(y_hat, x) = 1.0`.
- **p.20** — Residuals against x, annotated `corr(e, x) = 0.0` and `mean(e) = 0.0`.
- **p.21 — "Crazy" alternative line** `10 + 50X`, compared with the LS line `38.9 + 35.4X`.
- **p.22 — Crazy-line residuals.** `corr(e, x) = −0.7`, `mean(e) = 1.8`.
  - "This is a bad fit! We are underestimating the value of small houses and overestimating the value of big houses. **Clearly, we have left some predictive ability on the table!**"
- **p.23 — Key principle.** "As long as the correlation between e and X is non-zero, we could always adjust our prediction rule to do better. We need to exploit all of the predictive power in the X values and put this into Ŷ, leaving no 'Xness' in the residuals."
  - In summary, `Y = Ŷ + e` where:
    - Ŷ is "made from X": `corr(X, Ŷ) = 1 or −1`;
    - e is uncorrelated with X: `corr(X, e) = 0`;
    - `E(e) = 0`.
- **p.24 — Deriving the intercept from mean-zero residuals:**
  `(1/n) sum e_i = 0 ⇒ (1/n) sum (Y_i − b0 − b1 X_i) = 0 ⇒ Ȳ − b0 − b1 X̄ = 0 ⇒ b0 = Ȳ − b1 X̄`
- **p.25 — Deriving the slope from `corr(e, X) = 0`:**
  `0 = sum e_i (X_i − X̄) = sum (Y_i − b0 − b1 X_i)(X_i − X̄) = sum (Y_i − Ȳ − b1 (X_i − X̄))(X_i − X̄)`
  `⇒ b1 = sum (X_i − X̄)(Y_i − Ȳ) / sum (X_i − X̄)^2 = r_xy s_y / s_x`

## 5. Slides 26–32: ANOVA decomposition and R^2

- **p.26 — Variance decomposition.** Because `cov(Ŷ, e) = 0`, `var(Y) = var(Ŷ + e) = var(Ŷ) + var(e)`. This gives the **ANOVA for regression**:
  `sum (Y_i − Ȳ)^2 = sum (Ŷ_i − Ȳ)^2 + sum e_i^2`
- **p.27** — Figure splitting `(Y_i − Ȳ)` into `(Ŷ − Ȳ)` (explained) plus `(Y_i − Ŷ)` (residual).
- **p.28 — The three sums of squares.** **SST** (Total SS) = **SSR** (Regression SS) + **SSE** (Error SS).
  - SSR is the variation in Y explained by the regression line. SSE is the variation left unexplained. SSR = SST means a perfect fit.
  - **Warning:** "Be careful of similar acronyms; e.g. SSR for 'residual' SS."
  - Practical gotcha (from statsmodels, not the slides): `results.ssr` in statsmodels is the **residual** sum of squares, while `results.ess` is the explained SS.
- **p.29 — R^2, the coefficient of determination:** `R^2 = SSR/SST = 1 − SSE/SST`.
  - Slide states `0 < R^2 ≤ 1`; the closer to 1, the better the fit.
  - This is **in-sample** R^2 with the in-sample mean Ȳ as the implicit benchmark.
- **p.30 — R^2 equals the squared correlation in SLR:**
  `R^2 = sum (Ŷ_i − Ȳ)^2 / sum (Y_i − Ȳ)^2 = sum (b0 + b1 X_i − b0 − b1 X̄)^2 / sum (Y_i − Ȳ)^2 = b1^2 sum (X_i − X̄)^2 / sum (Y_i − Ȳ)^2 = b1^2 s_x^2 / s_y^2 = r_xy^2`
  "No surprise: the higher the sample correlation between X and Y, the better you are doing in your regression."
- **p.31 — ANOVA table in Python:**
  ```python
  anova_results = anova_lm(model)      # from statsmodels.stats.anova import anova_lm (import not shown)
  anova_results
  #            df      sum_sq       mean_sq          F     PR(>F)
  # Size      1.0  12393.107710  12393.107710  61.998311  0.000003
  # Residual 13.0   2598.625623    199.894279        NaN       NaN
  n = len(model.fittedvalues)
  var_fitted = np.var(model.fittedvalues, ddof=1) * (n - 1)   # 12393.11 = SSR
  var_resid  = np.var(model.resid, ddof=1) * (n - 1)          # 2598.63  = SSE
  ```
  "The Residuals SS is our Error SS (SSE). The size SS is our Regression SS (SSR)."
- **p.32 — `model.summary()` for the house data:**
  - Dep. Variable Price; OLS; No. Observations 15; Df Residuals 13; Df Model 1; Covariance Type **nonrobust**.
  - R-squared 0.827; Adj. R-squared 0.813; F 62.00, Prob(F) 2.66e-06; Log-Likelihood −59.944; AIC 123.9; BIC 125.3.
  - Intercept: 38.8847, se 9.094, t 4.276, P 0.001, CI [19.238, 58.531].
  - Size: 35.3860, se 4.494, t 7.874, P 0.000, CI [25.677, 45.095].
  - R^2 computed by hand:
  ```python
  var_price = np.var(house['Price'], ddof=1)
  np.var(model.fittedvalues, ddof=1) / var_price    # 0.826662763718887 = R^2
  ```
  - Adjusted R^2, AIC and BIC appear in the output but are **not explained** in L2.

## 6. Slides 33–44: Prediction goal and the SLR probability model

- **p.33** — "A prediction rule is any function where you input X and it outputs Ŷ as a predicted response at X." The LS line `Ŷ = f(X) = b0 + b1 X` is one such rule.
- **p.34** — Ŷ will not be a perfect prediction, so we need a notion of **prediction accuracy**.
- **p.35** — We want to know two things: what Y to expect for a given X, and how sure we are about that. Hence the **prediction interval**, the "probable range for Y-values given X".
- **p.36 — Key insight.** To build a PI we must assess the likely range of residuals for a Y **not yet observed**. That requires a **probability model** (e.g., normal).
  - "with 95% probability the residuals will be no less than −$28,000 or larger than $28,000." This is ±2s, with s = 14.14 for the house data.
  - "We must also acknowledge that the 'fitted' line may be fooled by particular realizations of the residuals."
- **p.37 — Simple Linear Regression Model:** `Y = β0 + β1 X + ε`, `ε ~ N(0, σ^2)`.
  - "The error term ε is independent 'idiosyncratic noise'."
  - "The power of statistical inference comes from the ability to make precise statements about the accuracy of the prediction. In order to do this we must invest in a probability model."
- **p.38 — Normal 2σ rule (figure).** ±1σ covers 68.2%, ±2σ covers 95.4%, ±3σ covers 99.7%. Band masses are 34.1%, 13.6%, 2.1% and 0.1%.
- **p.39 — Why `ε ~ N(0, σ^2)`?**
  - `E[ε] = 0 ⇔ E[Y|X] = β0 + β1 X`, the conditional expectation.
  - "Many things are close to Normal (central limit theorem)."
  - "It works! This is a very robust model for the world."
  - `β0 + β1 X` is the "true" regression line.
- **p.40** — E[Y|X] is the expected price of houses with size X. Some houses sell above it and some below. ε represents the influence of factors other than X.
- **p.41 — Conditional distribution.** `Y|X ~ N(β0 + β1 X, σ^2)`. σ controls the dispersion; the figure contrasts small σ with large σ.
- **p.42 — Conditional vs marginal distribution.** `Y|X ~ N(E[Y|X], Var(Y|X))`.
  - Mean: `E[Y|X] = E[β0 + β1 X + ε | X] = β0 + β1 X`.
  - Variance: `Var(Y|X) = Var(β0 + β1 X + ε | X) = Var(ε) = σ^2`.
  - `σ^2 < Var(Y)` if X and Y are correlated.
  - ANOVA view: "The bigger `[1 − σ^2/Var(Y)]`, the more X matters!" This is the population analogue of R^2.
- **p.43 — PI with the true model known.** Given `β0 = 40, β1 = 45, σ = 10`, predict a 1500 sq ft house.
  - `Y = 40 + 45(1.5) + ε = 107.5 + ε`, so `Y ~ N(107.5, 10^2)`.
- **p.44** — The mean value is $107,500 and the deviation is within ≈ $20,000. We are 95% sure that `−20 < ε < 20`, i.e. `$87,500 < Y < $127,500`.
  - **In general, the 95% PI with the true model is `β0 + β1 X ± 2σ`.**
  - "Next, we will learn about how to incorporate another source of risk: uncertainty in the β0, β1 and σ."

## 7. Slides 45–46: Estimation for SLR; population vs estimate

- **p.45** — SLR assumes every observation was generated by `Y_i = β0 + β1 X_i + ε_i`. It is a model of the conditional distribution of Y given X. LS estimates the parameters:
  `β̂1 = b1 = sum (X_i − X̄)(Y_i − Ȳ) / sum (X_i − X̄)^2`,  `β̂0 = b0 = Ȳ − b1 X̄`.
- **p.46 — "NOTE!!: β0 is not b0, β1 is not b1 and ε_i is not e_i".**
  - Figure contrasts the LS line `b0 + b1X` (red dotted) with the true line `β0 + β1X` (blue). The residual e_i differs from the true error ε_i.

## 8. Slides 47–54: Single-factor (index) model, CAPM, AAPL example

- **p.47 — Single Factor (Index) Model (statistical model).** Relates asset return r_i to the market return r_M through the regression
  `r_i − r_f = α + β (r_M − r_f) + ε`, observed at t = 1...T, with `[α, β] ≡ [β0, β1]`.
  - Returns enter as **excess returns** over the risk-free rate.
- **p.48 — Covariances and risk decomposition.**
  - Covariance between assets i and j: `cov(r_i, r_j) = β_i β_j σ_M^2`.
  - Number of estimates needed for all covariances: n estimates of β_i plus 1 estimate of σ_M^2.
  - Total risk = Systematic risk + Firm-specific risk: `σ_i^2 = β_i^2 σ_M^2 + σ^2(ε)`.
- **p.49 — CAPM (economic model).**
  - Implies `E(r_i) − r_f = β_i (E(r_M) − r_f)`, so it predicts `α_i = 0` for every asset i.
  - "When asset i is a mutual fund, [α_i, β_i] can be used as a **performance benchmark** for fund managers."
- **p.50 — CAPM example data.**
  - Weekly stock prices, 1/1/2010–12/31/2011, for AAPL, AIG, C, GE, GOOG, GS and MMM.
  - Weekly **annualized** US 1-month T-bill rates in **percent**.
  - Problem: calculate alphas and betas.
- **p.51 — Loading the data:**
  ```python
  capm = pd.read_csv('./beta.csv', index_col='Date')
  Tbill = capm['TBILL'].copy()
  capm.head(10)
  ```
  - The columns hold prices: AAPL 210.732, AIG 25.1146, C 33.0721, GE 14.2195, GOOG 619.980, GS 165.3277, MMM 78.5626, SPX 1115.10, TBILL 0.03 (on 1/1/10).
  - **The price-to-(excess)-return conversion code is not shown.** The regression on p.53 runs on weekly returns in **percent**, n = 104.
- **p.52** — Scatter of AAPL returns against SPX returns (percent, roughly −4 to +7) with the fitted line.
- **p.53 — Fit and summary:**
  ```python
  model_capm = smf.ols('AAPL ~ SPX', data=capm).fit()
  model_capm.summary()
  ```
  - Dep. Variable AAPL; n = 104; Df Residuals 102; Covariance Type **nonrobust**.
  - R^2 0.482; Adj R^2 0.477; F 94.96, Prob 2.99e-16; LL −195.12; AIC 394.2; BIC 399.5.
  - **Intercept 0.2943**, se 0.156, t 1.882, P 0.063, CI [−0.016, 0.605].
  - **SPX 1.0893**, se 0.112, t 9.745, P 0.000, CI [0.868, 1.311].
- **p.54** — Scatter of α against β for the 7 firms (approximate readings):

  | Firm | β | α |
  |---|---|---|
  | AAPL | 1.09 | +0.30 |
  | GE | 1.33 | +0.09 |
  | MMM | 0.98 | ≈0 |
  | GOOG | 1.13 | ≈0 |
  | AIG | 1.66 | ≈0 |
  | C | 1.83 | −0.09 |
  | GS | 1.08 | −0.32 |

## 9. Slides 55–58: Estimating the error variance; degrees of freedom

- **p.55** — Given `ε_i iid~ N(0, σ^2)`, σ drives the width of the PIs. `σ^2 = var(ε_i) = E[(ε_i − E[ε_i])^2] = E[ε_i^2]`.
  - A sensible first estimator is the sample average squared residual, `s^2 = (1/N) sum e_i^2`, with N still to be chosen.
- **p.56 — Unbiased estimator.** "In order to obtain an **unbiased** estimator of σ^2, what is the right value for N in the denominator":
  `s^2 = (1/(n − p)) sum e_i^2 = SSE/(n − 2)`
  - p is the number of regression coefficients (2 for β0 + β1).
  - "We have n − p degrees of freedom because p (= 2) have been 'used up' in the estimation of b0 and b1."
  - "We usually use `s = sqrt(SSE/(n − p))`, in the same units as Y."
- **p.57 — Degrees of Freedom.** "the number of times you get to observe useful information about the variance you're trying to estimate."
  - Example with `SST = sum (Y_i − Ȳ)^2`: if n = 1, then Ȳ = Y_1 and SST = 0, because "Y_1 is 'used up' estimating the mean, we haven't observed any variability!"
  - For n > 1 there are only n − 1 chances for deviation from the mean, so `s_y^2 = SST/(n − 1)`.
  - "In regression with p coefficients ... you only get n − p real observations of variability ⇒ **DoF = n − p**."
- **p.58 — Three ways to get s in Python** (CAPM example):
  ```python
  anova_lm(model_capm)
  #            df     sum_sq     mean_sq          F        PR(>F)
  # SPX       1.0  241.601109  241.601109  94.964514  2.988360e-16
  # Residual 102.0 259.500228    2.544120        NaN          NaN
  np.sqrt(anova_result.mean_sq[1])                               # 1.5950
  np.sqrt(model_capm.scale)                                      # 1.5950
  np.sqrt(sum(model_capm.resid ** 2) / model_capm.df_resid)      # 1.5950
  ```
  - `model.scale` = s^2 = SSE/(n − p). `model.df_resid` = n − p.

## 10. Slides 59–70: Sampling distributions and the CLT

- **p.59 — Thought experiment.** How much do our estimates depend on the particular sample? Imagine drawing many samples of the same size and computing b0, b1 and s for each.
  - "If the estimates don't vary much from sample to sample, then it doesn't matter which sample you happen to observe. If the estimates do vary a lot, then it matters which sample you happen to observe."
- **p.60 — Four simulated samples with N = 5** (true β0 = 1, β1 = 2; X in [−3, 3]):
  - (b0, b1) = (1.72, 2.14), (0.68, 1.95), (0.93, 2.25), (1.78, 1.58).
- **p.61 — Four simulated samples with N = 50:**
  - (b0, b1) = (1.26, 2.11), (0.93, 1.95), (1.22, 2.03), (1.13, 2.04).
- **p.62** — "LS lines are much closer to the true line when n = 50. For n = 5, some lines are close, others aren't: **we need to get 'lucky'**."
- **p.63 — Sampling distribution of the sample mean.** For an iid sample with `E(X_i) = µ` and `var(X_i) = σ^2`:
  - `E(X̄) = (1/n) sum E(X_i) = µ`.
  - `var(X̄) = var((1/n) sum X_i) = (1/n^2) sum var(X_i) = σ^2/n`.
  - If X is normal, then `X̄ ~ N(µ, σ^2/n)`. "If X is not normal, we have the central limit theorem!"
- **p.64 — Simple CLT.** For iid X with mean µ and variance σ^2, the distribution of the sample mean becomes normal as n grows: `X̄ →_n N(µ, σ^2/n)`. "Sample averages tend to be normally distributed in large samples."
- **p.65** — Exponential density with `E[X] = 1`, `var(X) = 1`. "Exponential random variables don't look very normal".
- **p.66–70** — Histograms of 1000 sample means for n = 2, 5, 10, 100 and 1000.
  - At n = 2 the histogram is right-skewed, spanning 0 to 4.
  - At n = 5 and 10 the skew shrinks.
  - At n = 100 it is bell-shaped, roughly 0.7–1.35 (sd ≈ 0.1).
  - At n = 1000 it spans roughly 0.92–1.1 (sd ≈ 0.03).
  - The spread shrinks like 1/sqrt(n).
- The procedure itself (simulate many samples, compute the statistic, histogram it) is the **Monte-Carlo sampling-distribution idiom**. It is reused on p.78–80.

## 11. Slides 71–77: Sampling distributions of b1 and b0; bias vs variance

- **p.71 — Sampling distribution of b1.** It describes how `b1 = β̂1` varies over samples **with the X values fixed**.
  - `b1 ~ N(β1, σ_{b1}^2)`. b1 is **unbiased**: `E[b1] = β1`. The sampling sd `σ_{b1}` determines precision.
  - "The variance term determines how close the estimate will be to the true value. Remember: large σ is bad!"
- **p.72 — Bias and Variance.** A 2×2 bullseye figure: low/high bias against low/high variance.
- **p.73 — Formula for σ_{b1}:**
  `σ_{b1}^2 = var(b1) = σ^2 / sum (X_i − X̄)^2 = σ^2 / ((n − 1) s_x^2)`
  - **Three factors:** sample size n, error variance σ^2 = σ_ε^2, and X-spread s_x.
- **p.74 — Optional derivation of var(b1).** Because `sum (X_i − X̄) = 0`:
  `b1 = sum (X_i − X̄) Y_i / sum (X_i − X̄)^2 = sum w_i Y_i`, with `w_i = (X_i − X̄)/D` and `D = sum (X_i − X̄)^2`.
  `var(b1) = sum w_i^2 var(Y_i | X_i) = σ^2 sum w_i^2 = σ^2 D/D^2 = σ^2/D`
  - "Note: b1 is more heavily weighted by X_i that are far from X̄." This is the leverage idea: extreme X points move the slope.
- **p.75 — Sampling distribution of b0.** b0 is normal and unbiased: `b0 ~ N(β0, σ_{b0}^2)` with
  `σ_{b0}^2 = var(b0) = σ^2 (1/n + X̄^2 / ((n − 1) s_x^2))`
  - Intuition: `var(Ȳ − X̄ b1) = var(Ȳ) + X̄^2 var(b1) + 2cov(Ȳ, b1)`. Ȳ and b1 are uncorrelated "because the slope (b1) is invariant if you shift the data up or down (Ȳ)."
- **p.76 — Joint distribution of b0 and b1.** `cov(b0, b1) = −σ^2 ( X̄ / ((n − 1) s_x^2) )`.
  - Usually X̄ > 0, so if the slope estimate is too high the intercept estimate is too low (negative correlation).
  - The correlation decreases with more X spread s_x^2.
- **p.77 — Estimated coefficient standard deviations (standard errors):**
  `s_{b1} = sqrt( s^2 / ((n − 1) s_x^2) )`,  `s_{b0} = sqrt( s^2 (1/n + X̄^2 / ((n − 1) s_x^2)) )`
  - Here `s = sqrt(sum e_i^2 /(n − p))` estimates σ = σ_ε. So `s_{b1} = σ̂_{b1}` and `s_{b0} = σ̂_{b0}`.
  - "A high level of info/precision/accuracy means small s_b values."

## 12. Slides 78–80 (new-style slides): "What Is a Standard Error, Really?" Simulation vs formula vs resampling

These three slides are the pedagogical heart of the lecture. They are the closest template for Problem 2.

- **p.78 — The question.** "We just derived `var(b1) = σ^2/sum (x_i − x̄)^2` and called its square root the standard error. **Where does that number actually come from?**"
  - The definition is a thought experiment: "if the world were run again — same market returns, different luck — how different would your b1 be?"
  - "**We can stop imagining and just do it.**" AAPL on the S&P 500, n = 104 weeks: **b1 = 1.0923, se(b1) = 0.1116**.
  - Plan: "treat that fitted line as if it were the truth, generate 10,000 alternative histories from it, refit all 10,000, and look at the spread."
  - "**Before the next slide: how wide will that histogram be? Write down a number.** Is the spread of ten thousand refits going to be near 0.11, or nowhere near it?" This is the **commit-before-compute** device.
  - Numbers flag: p.78 reports b1 = 1.0923, but p.53, p.87 and p.95 report 1.0893 (se 0.112 in both places). The two slide sets probably used slightly different return definitions, e.g. raw vs excess. Cite whichever you use and note the discrepancy.
- **p.79 — "Two Routes to the Same Number".**
  - Left figure: histogram of 10,000 refits overlaid with `N(b1, se(b1)^2)`.
  - Right figure: "normal errors: 0.112" against "resampled weeks: 0.119".
  - **Route 1, the formula:** one regression and one line of algebra. se(b1) = 0.1116.
  - **Route 2, brute force:** 10,000 refits on simulated histories (parametric simulation from the fitted model with normal errors). The sd of the 10,000 slopes is **0.1120**.
  - "They agree to 0.3%, and nothing was rigged — the two routes share no arithmetic at all."
  - **Coverage check:** "the textbook 95% confidence interval contained the true slope in **95.0%** of the ten thousand replications."
  - "That is the whole content of a standard error — not a decoration on a coefficient, but a statement about how much your answer would move if you had been dealt a different hand."
- **p.80 — "What the Assumption Costs".**
  - "Route 2 assumed what the formula assumes: errors that are **normal, independent, and the same size in every week**. Weekly equity returns are not like that — calm months and panics do not have the same variance."
  - **Route 3 (nonparametric bootstrap):** "Resample the 104 actual weeks with replacement, refit, ten thousand times."

    | Route | se(b1) |
    |---|---|
    | formula, textbook | 0.1116 |
    | simulation, normal errors | 0.1120 |
    | resampling the actual weeks | **0.1190** |

  - "**The textbook standard error is about 7% too small here.** Neither number is wrong; they answer the question under different assumptions, and **the gap between them is the size of the assumption**."
  - "It is why practitioners report 'robust' standard errors — and it is the first time in this course that a formula and the data have disagreed. It will not be the last."
- **No code is shown on p.78–80.** The procedure for re-implementing it:
  - **Route 2 (parametric):** Fix the X's (SPX returns). Set `Y* = b0 + b1 X + ε*` with `ε* ~ N(0, s^2)`. Refit and store b1*. Repeat 10,000 times. Report `std(b1*)`. For the coverage check, count how often `b1* ± t_{n−2,.025} s_{b1*}` contains the "true" b1.
  - **Route 3 (pairs bootstrap):** Resample (X_i, Y_i) rows with replacement, n = 104. Refit and store b1*. Repeat 10,000 times. Report `std(b1*)`.
  - Note that Route 3 resamples weeks **iid**. It relaxes normality and constant variance, but it still assumes independence across weeks.

## 13. Slides 81–87: Student-t and confidence intervals

- **p.81 — What Student discovered.** "If θ ~ N(µ, σ^2), but you estimate σ^2 ≈ s^2 based on n − p degrees of freedom, then θ ~ t_{n−p}(µ, s^2)."
  - Examples: `Ȳ ~ t_{n−1}(µ, s_y^2/n)`; `b0 ~ t_{n−2}(β0, s_{b0}^2)`; `b1 ~ t_{n−2}(β1, s_{b1}^2)`.
  - "The t distribution is just a **fat-tailed** version of the normal. As n − p → ∞, our tails get skinny and the t becomes normal."
- **p.82 — Standardization.** `(b_j − β_j)/σ_{b_j} ~ N(0,1)`, which becomes `(b_j − β_j)/s_{b_j} ~ t_{n−2}(0,1)` once σ is estimated.
  - Notation: `Z ~ N(0,1)` and `Z_{n−p} ~ t_{n−p}(0,1)`.
  - The t and normal distributions "depend upon assumed values for β_j: this forms the basis for confidence intervals, hypothesis testing, and p-values."
- **p.83 — Centered interval.** `Pr(−t_{n−p,1−α/2} < Z_{n−p} < t_{n−p,1−α/2}) = 1 − α`. The figure is a t with df = 10. Total tail area is α, with α/2 on each side.
- **p.84 — Confidence interval.** Since `b_j ~ t_{n−p}(β_j, s_{b_j})`:
  `1 − α = P(−t_{n−p,α/2} < (b_j − β_j)/s_{b_j} < t_{n−p,α/2}) = P(b_j − t_{n−p,α/2} s_{b_j} < β_j < b_j + t_{n−p,α/2} s_{b_j})`
  - So (1 − α)·100% of the time β_j lies in the **CI `b_j ± t_{n−p,α/2} s_{b_j}`**.
  - Notation note: p.83 writes the upper critical value as `t_{n−p,1−α/2}` and p.84 writes it as `t_{n−p,α/2}`. Both mean the upper-α/2 critical value.
- **p.85 — Coverage probability.** `P[β1 ∈ (b1 ± 2 s_{b1})] = 95%`. The figure shows many simulated CIs around the true β1, with a few (red) missing it.
- **p.86 — Why CIs matter.** The CI "captures the amount of information in the data about the parameter". The center is your estimate. The length tells you how sure you are.
- **p.87 — CI in Python:**
  ```python
  model_capm.conf_int(alpha=0.05)
  #                  0          1
  # Intercept -0.015936   0.604524
  # SPX        0.867574   1.311002
  df_resid = n - p
  t025 = stats.t.ppf(0.975, df_resid)
  params = model_capm.params
  stderr = model_capm.bse
  ci_intercept = [params[0] - t025 * stderr[0], params[0] + t025 * stderr[0]]   # [-0.015936, 0.604524]
  ci_slope     = [params[1] - t025 * stderr[1], params[1] + t025 * stderr[1]]   # [0.867574, 1.311002]
  ```
  - "`scipy.stats.t.ppf` is the Student's t 'quantile function', and `scipy.stats.t.ppf(prob, df)` returns t_df such that prob = P(Z_df < t_df)."
  - Modern-pandas note (not on the slide): `params[0]` positional indexing on a Series raises a FutureWarning in pandas ≥ 2.1. Prefer `params.iloc[0]` or labels (`params['Intercept']`), as p.95 does.

## 14. Slides 88–97: Hypothesis testing, t-statistics, p-values; CAPM tests

- **p.88 — Hypotheses.** Is there evidence of a relationship between X and Y?
  - `H0: β1 = 0`: the null/safe hypothesis. It implies "no effect" and we ignore X.
  - `H1: β1 ≠ 0`: the alternative. It leads to our best guess β1 = b1.
- **p.89** — Assuming H0 is true, the decision rule sets plausible and implausible ranges for the test statistic. We either reject or do not reject H0, "the default and usually simpler claim". Rejection gives statistical support for H1.
- **p.90** — We reject H0 when b_j is far from β_j^0 (usually 0) and keep it when b_j is close. The raw difference b_j − β_j^0 ignores estimation uncertainty. "What we really care about is how many standard deviations b_j is away from β_j^0."
- **p.91 — t-statistic.** `z_{b_j} = (b_j − β_j^0)/s_{b_j}`, which equals `b_j/s_{b_j}` for β_j^0 = 0.
  - Under H0, `z_{b_j} ~ t_{n−p}(0,1)`. Small |z| leaves us happy with the null. **Large |z| (> about 2) should worry us.**
- **p.92 — p-value.** The probability, under H0, of a test statistic at least as extreme as the one observed:
  `φ = P(|Z_{n−p}| > |z_{b_j}|) = 2 P(Z_{n−p} > |z_{b_j}|)`
  The figure shows p-value = 0.05 with 8 df.
- **p.93 — Formal two-step approach.**
  1. Pick the significance level α (often 1/20 = 0.05). This is the acceptable risk of rejecting a true null, a **type 1 error**. "This α plays the same role as α in CI's."
  2. Compute the p-value and reject H0 if φ < α, in favor of the best alternative guess β_j = b_j. If φ > α, continue under the null.
  - This is equivalent to the **rejection region `|z_{b_j}| > t_{n−p,α/2}`**.
- **p.94 — CAPM test of the intercept.** Does AAPL have a non-zero intercept?
  - `H0: β0 = 0`: no expected return in excess of (or below) the fair CAPM return.
  - `H1: β0 ≠ 0`: AAPL is under- or overpriced.
- **p.95 — Intercept test in Python:**
  ```python
  b0 = model_capm.params['Intercept']          # 0.2943
  n = len(model_capm.fittedvalues)
  sb0 = model_capm.bse['Intercept']            # 0.156
  t_stat = b0 / sb0                            # 1.882
  p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df=n-2))   # 0.063
  ```
  - "The t 'distribution function' `scipy.stats.t.cdf(z, df)` returns P(Z_df < z)." We **cannot reject** the null at α = .05.
- **p.96 — CAPM test of the slope.**
  - `H0: β1 = 1`: AAPL is as aggressive as the market.
  - `H1: β1 ≠ 1`: AAPL softens or exaggerates market moves.
- **p.97 — Non-zero null.** "This time, Python's output t/p values are not what we want (why?)." The default summary tests β = 0, not β = 1.
  ```python
  b1 = model_capm.params['SPX']
  n = len(model_capm.fittedvalues)
  sb1 = model_capm.bse['SPX']
  zb1 = (b1 - 1) / sb1
  p_value = 2 * stats.t.cdf(-abs(zb1), df=n-2)     # 0.4262693
  ```
  - We cannot reject at α = .05 (φ = .426).

## 15. Slides 98–106: Prediction intervals

- **p.98 — Conditional prediction problem.** Given a covariate X_f and sample data {X_i, Y_i}, predict the "future" y_f.
  - Solution: use the LS fitted value `Ŷ_f = b0 + b1 X_f`. "This is the easy bit. The hard (and very important!) part of predict is assessing uncertainty about our predictions."
- **p.99 — Prediction error.** `e_f = Y_f − Ŷ_f = Y_f − b0 − b1 X_f`. The figure separates ε from "sampling error" (the gap between the true line and b0 + b1X at X_f).
- **p.100 — Two sources of prediction error.** (i) inherent idiosyncratic randomness ε; (ii) estimation error in intercept and slope.
  `e_f = Y_f − Ŷ_f = (Y_f − E[Y_f|X_f]) + E[Y_f|X_f] − Ŷ_f = ε_f + (E[Y_f|X_f] − Ŷ_f) = ε_f + (β0 − b0) + (β1 − b1) X_f`
- **p.101 — Variance of the fitted value and of the prediction error.**
  `var(Ŷ_f) = var(b0 + b1 X_f) = var(b0) + X_f^2 var(b1) + 2 X_f cov(b0, b1) = σ^2 [1/n + (X_f − X̄)^2 / ((n − 1) s_x^2)]`
  `var(e_f) = σ^2 [1 + 1/n + (X_f − X̄)^2 / ((n − 1) s_x^2)]`
  The "1" is the ε part and the rest is estimation error.
- **p.102 — Predictive distribution and interval.**
  `Y_f ~ N(Ŷ_f, σ^2 [1 + 1/n + (X_f − X̄)^2/((n − 1)s_x^2)])` (sums of normals are normal). With σ estimated:
  `Y_f ~ t_{n−p}(Ŷ_f, s^2 [1 + 1/n + (X_f − X̄)^2/((n − 1)s_x^2)])`
  - **(1 − α)100% prediction interval:** `b0 + b1 X_f ± t_{n−2,α/2} ( s sqrt(1 + 1/n + (X_f − X̄)^2/((n − 1) s_x^2)) )`
- **p.103 — What makes s_pred large.** `s_pred = s sqrt(1 + 1/n + (X_f − X̄)^2/((n − 1)s_x^2))`. Large predictive uncertainty comes from:
  - large s (large ε's);
  - small n (not enough data);
  - small s_x (not enough spread in covariates);
  - **large (X_f − X̄)**.
- **p.104** — "For X_f far from our X̄, the space between lines is magnified..." The figure shows the true and estimated lines crossing near the point of means (X̄, Ȳ) and diverging away from it.
- **p.105** — "The prediction interval needs to **widen away from X̄**." The figure shows a 95% PI band (curved) around `Ŷ_f = b0 + b1 X_f`, a narrower "Std Dev Fit Area", a 2s width, and the "Smallest Standard Error" at X̄.
- **p.106 — Prediction in Python:**
  ```python
  Xf = pd.DataFrame({'SPX': [0.75]})          # slide omits the braces: pd.DataFrame('SPX': [0.75])
  predictions = model_capm.get_prediction(Xf)
  pred_summary = predictions.summary_frame(alpha=0.05)
  pred_summary
  #   mean      mean_se   mean_ci_lower  mean_ci_upper  obs_ci_lower  obs_ci_upper
  #   1.11126   0.177342  0.759502       1.463018       -2.071969     4.294489
  s = np.sqrt(model_capm.scale)
  mu = np.mean(capm['SPX'])
  s_fit = s * np.sqrt(1/n + (Xf['SPX'] - mu)**2 / ((n-1) * np.var(capm['SPX'], ddof=1)))   # 0.177
  fit = model_capm.params['Intercept'] + model_capm.params['SPX'] * Xf['SPX']              # 1.11126
  s_pred = np.sqrt(s**2 + s_fit**2)
  lower_bound = fit - t_critical * s_pred      # t_critical not defined on slide; = stats.t.ppf(0.975, n-2)
  upper_bound = fit + t_critical * s_pred
  # 95% Prediction Interval: [-2.07197, 4.29449]
  ```
  - "Notice that `s_pred = sqrt(s^2 + s_fit^2)`; **you need to square before summing**."
  - `mean_ci_*` is the CI for E[Y_f|X_f] and uses s_fit only. `obs_ci_*` is the prediction interval for Y_f and uses s_pred.

---

## 16. Consolidated formula sheet (L2)

| Quantity | Formula | Slide |
|---|---|---|
| Linear model (general) | Y = b0 + b1 X1 + ... + bd Xd + e | p.3 |
| LS objective | min_{b0,b1} Σ (Y_i − b0 − b1 X_i)^2 | p.14–15 |
| Slope | b1 = Σ(X_i−X̄)(Y_i−Ȳ)/Σ(X_i−X̄)^2 = r_xy s_y/s_x | p.18, 25, 45 |
| Intercept | b0 = Ȳ − b1 X̄ | p.18, 24, 45 |
| Residual properties | mean(e)=0; corr(e,X)=0; corr(Ŷ,X)=±1 | p.23 |
| ANOVA | SST = SSR + SSE: Σ(Y−Ȳ)^2 = Σ(Ŷ−Ȳ)^2 + Σe^2 | p.26–28 |
| R^2 | SSR/SST = 1 − SSE/SST = r_xy^2 (SLR) | p.29–30 |
| Population "R^2" | 1 − σ^2/Var(Y) | p.42 |
| SLR model | Y = β0 + β1X + ε, ε ~ N(0,σ^2) iid; Y\|X ~ N(β0+β1X, σ^2) | p.37, 41 |
| PI, true model | β0 + β1X ± 2σ | p.44 |
| Single-factor model | r_i − r_f = α + β(r_M − r_f) + ε | p.47 |
| Covariance via factor | cov(r_i,r_j) = β_iβ_jσ_M^2 | p.48 |
| Total risk | σ_i^2 = β_i^2σ_M^2 + σ^2(ε) | p.48 |
| CAPM | E(r_i) − r_f = β_i(E(r_M) − r_f); α_i = 0 | p.49 |
| Error variance | s^2 = SSE/(n−p), s = sqrt(SSE/(n−p)) | p.56 |
| Sample variance | s_y^2 = SST/(n−1) | p.57 |
| DoF | n − p | p.57 |
| Mean of X̄ | E(X̄)=µ, var(X̄)=σ^2/n | p.63 |
| CLT | X̄ →_n N(µ, σ^2/n) | p.64 |
| var(b1) | σ^2/Σ(X_i−X̄)^2 = σ^2/((n−1)s_x^2) | p.73–74 |
| var(b0) | σ^2(1/n + X̄^2/((n−1)s_x^2)) | p.75 |
| cov(b0,b1) | −σ^2 X̄/((n−1)s_x^2) | p.76 |
| se(b1), se(b0) | sqrt(s^2/((n−1)s_x^2)), sqrt(s^2(1/n + X̄^2/((n−1)s_x^2))) | p.77 |
| t result | (b_j − β_j)/s_{b_j} ~ t_{n−p} | p.81–82 |
| CI | b_j ± t_{n−p,α/2} s_{b_j} | p.84 |
| t-stat | z = (b_j − β_j^0)/s_{b_j} | p.91 |
| p-value | 2P(Z_{n−p} > \|z\|) | p.92 |
| Rejection region | \|z\| > t_{n−p,α/2}; reject if φ < α | p.93 |
| var(Ŷ_f) | σ^2[1/n + (X_f−X̄)^2/((n−1)s_x^2)] | p.101 |
| var(e_f) | σ^2[1 + 1/n + (X_f−X̄)^2/((n−1)s_x^2)] | p.101 |
| PI | b0 + b1X_f ± t_{n−2,α/2} s sqrt(1 + 1/n + (X_f−X̄)^2/((n−1)s_x^2)) | p.102 |
| s_pred | sqrt(s^2 + s_fit^2) | p.103, 106 |

## 17. Code idioms (house style)

Imports implied by the slides (the import lines themselves are never shown):
```python
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.stats.anova import anova_lm
from scipy import stats
from sklearn.linear_model import LinearRegression   # mentioned p.18 only
```

| Idiom | Slide |
|---|---|
| `smf.ols('Price ~ Size', data=house).fit()`; `model.params` | p.16 |
| `np.corrcoef(y, x)[0, 1] * np.std(y, ddof=1) / np.std(x, ddof=1)`; `np.mean`; `pd.DataFrame({'b0':[b0],'b1':[b1]})` | p.17 |
| `LinearRegression().fit(X, Y)` (sklearn) | p.18 |
| `anova_lm(model)`; `len(model.fittedvalues)`; `np.var(model.fittedvalues, ddof=1)*(n-1)`; `np.var(model.resid, ddof=1)*(n-1)` | p.31 |
| `model.summary()`; `np.var(house['Price'], ddof=1)`; R^2 = var(fitted)/var(y) | p.32 |
| `pd.read_csv('./beta.csv', index_col='Date')`; `capm['TBILL'].copy()`; `capm.head(10)` | p.51 |
| `smf.ols('AAPL ~ SPX', data=capm).fit()` | p.53 |
| `anova_result.mean_sq[1]`; `model.scale`; `model.resid`; `model.df_resid` | p.58 |
| `model.conf_int(alpha=0.05)`; `stats.t.ppf(0.975, df_resid)`; `model.params`; `model.bse` | p.87 |
| `model.params['Intercept']`; `model.bse['Intercept']`; `2*(1 - stats.t.cdf(abs(t), df=n-2))` | p.95 |
| `(b1 - 1)/sb1`; `2*stats.t.cdf(-abs(z), df=n-2)` | p.97 |
| `model.get_prediction(Xf)`; `.summary_frame(alpha=0.05)` with columns mean / mean_se / mean_ci_lower / mean_ci_upper / obs_ci_lower / obs_ci_upper | p.106 |

**Consistent convention:** every variance/std in the slides uses **`ddof=1`** (p.17, 31, 32, 106).

## 18. Course conventions emphasised in L2
1. **Unbiased variance with n − 1 or n − p denominators** (p.56–57). Code uses `ddof=1` throughout (p.17, 31, 32, 106). `model.scale` = SSE/(n−p) (p.58).
2. **Inference uses t_{n−p}, not z, when σ is estimated** (p.81–84, 87, 95, 97). The rule of thumb |t| > about 2 and ±2s approximates 95% (p.44, 85, 91).
3. **Returns in the factor model are excess returns** `r_i − r_f` against `r_M − r_f` (p.47). α is a performance benchmark (p.49). The T-bill is quoted annualized in percent and must be converted (p.50).
4. **Population parameters (β, ε) and estimates (b, e) are kept distinct** (p.46).
5. **R^2 is in-sample** (SSR/SST) with Ȳ as the benchmark (p.29). Adjusted R^2, AIC and BIC appear in output but are not taught (p.32, 53).
6. **A standard error is the sd of the estimator across hypothetical re-runs.** It can be verified by (a) formula, (b) parametric Monte-Carlo from the fitted model, or (c) resampling the data with replacement. The gap between these is "the size of the assumption" (p.78–80).
7. **Commit-before-compute.** "Write down a number" before seeing a simulation result (p.78).
8. **statsmodels' default output tests β = 0**, with nonrobust covariance (p.32, 53). Other nulls must be computed by hand (p.97).
9. **Prediction uncertainty = noise + estimation error.** PIs use s_pred, not s_fit (p.100–106).

## 19. Pitfalls / warnings emphasised
- **Residuals correlated with X** mean predictive ability has been left on the table (p.22–23).
- **Acronym trap:** SSR (regression) vs SSR (residual) (p.28). statsmodels `.ssr` is the residual SS.
- **β ≠ b, ε ≠ e** (p.46).
- **Small n: "we need to get lucky"** (p.62). Estimates vary a lot from sample to sample (p.59).
- **Large σ is bad** for precision (p.71). var(b1) depends on n, σ^2 and s_x (p.73).
- **Leverage:** b1 is weighted more heavily by X_i far from X̄ (p.74).
- **b0 and b1 errors are negatively correlated** when X̄ > 0 (p.76).
- **Textbook SEs assume normal, independent, equal-variance errors.** Weekly equity returns violate this ("calm months and panics do not have the same variance"). The textbook SE is ~7% too small against resampling. Hence robust SEs. "It will not be the last" time formula and data disagree (p.80).
- **The t distribution is fat-tailed relative to the normal**; don't use z critical values with small DoF (p.81).
- **Default p-values are for H0: β = 0**, so they are the wrong numbers for testing β1 = 1 (p.97).
- **α = type I error rate** (p.93).
- **The fitted line "may be fooled by particular realizations of the residuals"** (p.36).
- **Extrapolation:** the PI widens with (X_f − X̄)^2. Predictions far from the data centre are much less certain (p.103–105).
- **Square before summing:** `s_pred = sqrt(s^2 + s_fit^2)`, not s + s_fit (p.106).
- **mean_ci vs obs_ci:** CI for the mean is not the PI for an observation (p.106).
- **Slide typos/inconsistencies to be aware of:**
  - p.12: `b0 + b1 X1` should read `b0 + b1 X_i`.
  - p.29 says `0 < R^2`; in-sample R^2 is ≥ 0 with an intercept.
  - p.78 b1 = 1.0923 vs p.53 b1 = 1.0893.
  - p.106 `pd.DataFrame('SPX': [0.75])` is missing braces, and `t_critical` is undefined.
  - p.87 `params[0]` uses deprecated positional Series indexing.

## 20. Datasets and worked examples
1. **House prices** (n = 15; size in 1000 sq ft, price in $1000) (p.5–32, 36).
   - Eyeball line b0 = 35, b1 = 40 (p.9–10).
   - LS: b0 = 38.8847 (se 9.094), b1 = 35.3860 (se 4.494) (p.16, 32).
   - SSR = 12393.11, SSE = 2598.63, MSE = 199.894, s = 14.14, R^2 = 0.8267 (p.31–32).
2. **Hypothetical true model** β0 = 40, β1 = 45, σ = 10, for a 1500 sq ft house. Y ~ N(107.5, 100); 95% PI is $87.5k–$127.5k (p.43–44).
3. **CAPM / single-factor**, `beta.csv`: weekly prices 2010–2011 for AAPL, AIG, C, GE, GOOG, GS, MMM, SPX, plus TBILL (annualized %) (p.50–54, 58, 87, 95, 97, 106).
   - AAPL on SPX, n = 104: α = 0.2943 (se 0.156, t 1.882, p 0.063); β = 1.0893 (se 0.112, t 9.745).
   - R^2 = 0.482; s = 1.5950.
   - Test of β = 1: p = 0.426.
   - Prediction at SPX = 0.75: fit 1.11126, s_fit 0.177, 95% PI [−2.072, 4.294], mean CI [0.760, 1.463].
4. **Simulated SLR** with β0 = 1, β1 = 2 at N = 5 vs N = 50 (p.60–62).
5. **CLT demo:** Exp(1) population, 1000 sample means at n = 2, 5, 10, 100, 1000 (p.65–70).
6. **SE verification** (AAPL, n = 104): formula 0.1116; 10,000 parametric refits 0.1120; 10,000 bootstrap resamples of weeks 0.1190; 95% CI coverage 95.0% (p.78–80).

## 21. Boundaries: what L2 does NOT teach (or only mentions)
- **Multiple regression** appears only as the general equation (p.3). No multi-X inference, no matrix OLS, no multicollinearity.
- **sklearn `LinearRegression`** is named once (p.18). No sklearn workflow (pipelines, scaling, CV) is shown.
- **Robust standard errors** are mentioned only by name (p.80). No `cov_type='HC*'` or HAC/Newey-West code or formula.
- **Bootstrap:** only a verbal description of resampling weeks with replacement (p.80). No block bootstrap and no code.
- **Monte-Carlo simulation of sampling distributions:** figures and descriptions only (p.60–70, 78–79). No code.
- **Adjusted R^2, F-statistic, log-likelihood, AIC and BIC** appear in `summary()` output (p.32, 53) but are not explained.
- **Out-of-sample R^2, train/validation/test splits, cross-validation, time-series CV, expanding/rolling windows, look-ahead bias and standardization procedures** are not in L2.
- **Chi-square distribution of s^2, Var(s^2) = 2σ^4/(n−1), delta method, Jensen's inequality, lognormal** are not in L2.
- **Kurtosis, VaR, Sharpe ratio, portfolio construction, transaction costs** are not in L2. "Fat-tailed" appears only for the t distribution (p.81).
- **Classification, accuracy, trees, ensembles, variable importance** are not in L2.
- **Heteroskedasticity** is named only informally ("calm months and panics do not have the same variance", p.80). No WLS or tests.
- **Converting prices to returns and the annualized T-bill to a weekly rate** is implied but not shown (p.50–53).

## 22. Mapping to the final exam

### Problem 1 (Trees & Ensembles): weak direct relevance; conceptual support only
- **1.1 / 1.3, noise in CV accuracy.** The sample-mean sampling distribution (p.63: var(X̄) = σ^2/n) and the CLT (p.64) justify an approximate standard error for an accuracy estimate: accuracy is a mean of 0/1 hits, so se ≈ sqrt(acc(1 − acc)/n).
  - With n = 400 the se is ≈ 2 percentage points. Use this, or the spread of fold scores, to judge whether RF/GB really "beat" the tree.
  - This is the p.59–62 message: small samples, "we need to get lucky".
  - Hypothesis-testing logic (p.88–93) gives the language for "is the difference distinguishable from zero?".
- **1.1, baseline comparison.** L2's in-sample R^2 is defined against the naive Ȳ benchmark (p.29). Measuring a model against a naive predictor is the same idea as the majority-class baseline. The majority-class baseline itself is not in L2.
- **1.5, time-series rows.** p.80 says the textbook SE (and iid resampling) assumes errors that are independent and of constant size, and that equity returns violate this.
  - The CLT and var(X̄) = σ^2/n (p.63–64) require **iid** data. Shuffled K-fold CV implicitly treats rows as exchangeable/iid.
  - L2 supports the "iid assumption fails" argument. The time-ordered CV remedy and the direction of the bias must come from other lectures.
- **1.4:** there is no L2 content on importance measures.

### Problem 2 (Training on your own output / VaR): strong relevance
- **Step (1) estimator: unbiased sample variance with ddof = 1.**
  - p.56: "In order to obtain an unbiased estimator of σ^2 ... divide by n − p".
  - p.57: DoF, "n − 1 chances for deviation from the mean", `s_y^2 = SST/(n−1)`.
  - Code idiom `np.var(x, ddof=1)` (p.17, 31, 32, 106).
  - The notebook's `σ̂^2 = 1/(n−1) Σ (x_i − x̄)^2` is exactly `s_y^2` from p.57.
- **2.1, commit before compute.** This is the p.78 device: "Before the next slide: how wide will that histogram be? **Write down a number.**" Cite it.
- **2.2, one desk then 1000 desks.** Same design as p.78–79: treat the fitted model as the truth, generate many alternative histories, refit each, and study the spread. Also p.59–62 ("randomly draw different samples ... compute the estimates") and p.65–70 (1000 simulated means, histogram).
- **2.3, unbiased yet wrong in the end.**
  - Unbiasedness `E[s^2] = σ^2` (p.56) is what gives the martingale `E[σ̂^2_{t+1} | σ̂^2_t] = σ̂^2_t`. Each night's variance is estimated from draws of N(0, σ̂^2_t).
  - p.71–72: "b1 is unbiased ... sampling sd determines precision", plus the bias–variance bullseye. An unbiased estimator can still have large variance.
  - Unbiasedness says nothing about the **median/typical** outcome.
  - CLT (p.63–64): log σ̂^2_T is a **sum of ~iid nightly increments** log(σ̂^2_t / σ̂^2_{t−1}). After 2,500 nights it is approximately normal, which is why the histogram of log σ̂^2 looks bell-shaped.
  - The mean nightly drift is **negative** (≈ −1/(n−1)), so the median collapses while the mean stays 1, held up by a few huge desks.
  - The ≈ −1/(n−1) formula needs `Var(s^2) = 2σ^4/(n−1)` plus a second-order (delta-method/Jensen) argument. **These are not in L2** (see §21). Present it as an empirically fitted formula checked across n = 50, 500, 5000, with the derivation flagged as beyond L2.
  - var(X̄) = σ^2/n intuition (p.63): larger n per night means a smaller nightly log-variance step, so slower drift.
- **2.4(a), keep 500 real days.** p.57: DoF is "the number of times you get to observe useful information about the variance you're trying to estimate".
  - Synthetic draws from σ̂^2 add no new information about the true σ. Only the real observations do.
  - Anchoring the library on the real days stops the random walk; the variance of σ̂^2 stays bounded.
- **2.4(b), real dj30 returns: normal VaR vs empirical VaR, kurtosis.**
  - p.80 is the direct template. The formula/normal-simulation answer vs the resample-the-actual-data answer; "Weekly equity returns are not like that — calm months and panics do not have the same variance"; "the gap between them is the size of the assumption"; "the first time ... a formula and the data have disagreed. It will not be the last".
  - Drawing night-1 scenarios from a fitted N(0, σ̂^2) is Route 2 (parametric, normal errors). The empirical 1% quantile of real returns is the Route 3 analogue (use the data themselves).
  - p.81: the normal has thinner tails than fat-tailed alternatives ("t is just a fat-tailed version of the normal").
  - p.38: normal tail areas; the 2σ rule.
  - Quantile functions: p.87 uses `stats.t.ppf`. The notebook's `norm.ppf(0.99) = 2.326` is the same idiom for the normal.
  - Loss of information: the fitted normal keeps only σ̂ and discards tail shape (kurtosis) and volatility clustering.
  - Kurtosis itself is not defined in L2; use scipy's `kurtosis`, which reports excess (Fisher) kurtosis by default.
- **Implication for self-retrained pipelines.** p.78–80: an estimator's reliability comes from real independent information. Simulating from your own fit can only reproduce your assumptions (p.79: simulation agreed with the formula because both assumed the same thing; only real data revealed the 7% gap).

### Problem 3 (Research project): foundational relevance
- **3-features / modeling, linear predictive regression.**
  - OLS as a prediction rule `Ŷ = f(X)` (p.33).
  - E[Y|X] as the target (p.39–40); the general linear model with d regressors (p.3).
  - House-style fitting via `smf.ols('y ~ x1 + ...', data).fit()` or `LinearRegression().fit(X, Y)` (p.18).
  - Residuals must carry no remaining "Xness" (p.23).
- **3-features, standardization.** Slopes are unit-dependent (p.8, 10) and `b1 = r_xy s_y/s_x` (p.18). Standardizing X and Y makes slopes comparable across characteristics (the standardized slope equals the correlation in SLR). The cross-sectional standardization procedure itself is from other lectures.
- **3-features, vol-scaled target r/σ_{t−1}.**
  - The SLR model assumes constant σ^2 (p.37, 41).
  - p.80 warns that returns have time-varying variance ("calm months and panics"), which makes textbook SEs too small.
  - p.71–73: large σ hurts precision (var(b1) ∝ σ^2).
  - Scaling returns by lagged volatility brings the data closer to the equal-variance assumption, which motivates the vol-scaled target. Use only σ_{t−1}; the no-look-ahead rule is from other lectures.
- **3-features, per-class vs pooled models.**
  - var(b1) = σ^2/((n − 1)s_x^2) (p.73) with the three factors n, σ^2, s_x.
  - Pooling across 50 assets raises n and s_x, so the slope variance falls. Per-class models allow different slopes but use fewer observations: the bias–variance trade-off (p.72) and "we need to get lucky" with small n (p.62).
- **3-evaluation, forecastability (Q1).**
  - p.100–101: every forecast error is `ε_f + estimation error`, with `var(e_f) = σ^2[1 + 1/n + (X_f − X̄)^2/((n−1)s_x^2)]`.
  - With return data σ^2 dominates. Estimation error can make an estimated model predict worse OOS than a naive forecast, so R2_OOS can be negative. In-sample R^2 is always ≥ 0 (p.29) because it is benchmarked against the in-sample Ȳ.
  - **The R2_OOS definition (vs trailing mean, vs zero) is not in L2; cite the later lecture that defines it.** L2 supplies the structure `1 − SSE/SST` (p.29) with SST's Ȳ replaced by the benchmark forecast.
  - Using the full-sample Ȳ would be look-ahead in an OOS setting; that argument comes from other lectures.
- **3-evaluation, coefficient inference.**
  - t-stats, p-values and CIs for predictive slopes (p.84–95).
  - Tests of non-zero nulls done by hand (p.97).
  - Default output is "nonrobust" (p.32, 53); returns violate its assumptions, hence robust SEs (p.80). L2 names them but shows no code.
  - The bootstrap (resampling periods with replacement) is an L2-grounded way to get SEs for R2_OOS or Sharpe differences (p.80). Plain iid resampling ignores serial dependence; flag this in the write-up.
- **3-evaluation, placebo (macro circularly shifted 36 months).** L2's null-hypothesis logic (p.88–93) is "what would we see if there were no effect?". The placebo builds an empirical distribution under the null, like the p.78–79 simulated histories. The placebo design itself comes from the exam/other lectures.
- **3-evaluation, extrapolation.** p.103–105: PIs widen as X_f moves away from X̄. Forecasts in periods where characteristics or macro variables sit outside the training range (e.g., 2008, 2020) are less reliable. This supports rolling or expanding refits and caution in the write-up.
- **3-portfolio, benchmarks and risk parity.**
  - Single-factor model and CAPM (p.47–49): excess returns `r_i − r_f`; `σ_i^2 = β_i^2 σ_M^2 + σ^2(ε)`; `cov(r_i, r_j) = β_iβ_jσ_M^2`.
  - α "can be used as a performance benchmark". Regress the forecast portfolio's excess return on a benchmark (e.g., equal weight) and test **H0: α = 0** with the t-stat and p-value (p.94–95). This is the L2-grounded way to ask "does it beat EW?".
  - A β ≠ 1 test (p.96–97) tells you whether the strategy just levers the benchmark.
  - Risk-parity weights 1/σ_i use volatility estimated with `ddof=1` (p.56–57 convention).
- **3-write-up.** "Neither number is wrong ... the gap between them is the size of the assumption" (p.80). This supports reporting results under several estimation schemes (static/expanding/rolling) and SE methods, and defending each assumption. It fits the exam's "negative result, honestly established" ethos.
