# BUSN 41210 Financial Analytics — Lecture 3: Diagnostics and Transformations, Categorical Variables, Interactions, Multiple Linear Regressions

Study notes with slide citations.
- Source: `/home/user/UChicago/Lecture_3.pdf`, 58 pages. PDF page = slide number, cited "L3 p.X".
- Instructor: Dacheng Xiu, Chicago Booth. Date on the title slide: September 11, 2026 (L3 p.1).
- All 58 pages were checked against the rendered slides, not only the extracted text. The extracted text garbles every formula, table and chart. Formulas, code and output tables below are transcribed from the rendered images.
- Verification: I re-ran the Anscombe regressions with statsmodels (Anscombe data typed in).
  - The studentized residuals printed on p.15 match `OLSInfluence(reg1).resid_studentized_external` for **Anscombe dataset 1** exactly (0.031345, -0.040845, -2.081099, ...).
  - For dataset 3, the outlier's studentized residual is about 1203.5, which matches the "~1200" point on p.16.
  - Both the coefficients (3.0, 0.5) and R^2 = 0.6665 (shown rounded to 0.7 on p.7) match.
- Tags used below:
  - **[slide]** = what the slide says.
  - **[my note]** = my own derivation, observation or inference. Treat these as not taught.

---

## 0. One-paragraph summary / what this lecture is for

L3 has two halves.

**First half (p.2–18): regression diagnostics for simple linear regression (SLR).**
- Restates the model assumptions: linear mean; errors that are normal, independent and constant-variance.
- Anscombe's quartet (identical summary statistics and fits, very different data) shows why plots matter.
- Residuals vs fitted values is the "#1 tool".
- Leverage h_i; standardized residuals; externally studentized residuals ~ t_{n-p-1}, with the [-2.5, 2.5] rule.
- How to treat outliers: delete only with a really good reason, run with and without, document.
- Warning: high-leverage outliers hide from r_i.

**Second half (p.19–58): multiple linear regression (MLR).**
- The model, least squares, `b = (X'X)^{-1}X'Y`, s^2 with p = d+1, and `S_b = s^2 (X'X)^{-1}`.
- t-intervals and tests (one-at-a-time); R^2 = cor^2(Ŷ, Y).
- Forecasting and prediction intervals, both through statsmodels `get_prediction` and by hand with numpy.
- Worked example: Welch–Goyal (2007) equity-premium regression, 16 predictors, 74 annual observations 1948–2021.
- **Multicollinearity** (infl/lty/tbl): individually insignificant but jointly significant by the F-test; individually significant in univariate regressions.
- **Categorical variables and dummies** (R−1 dummies, reference level absorbed by the intercept, `dmatrix`).
- **Log transforms and elasticities**: the orange-juice (OJ) log-log demand model.
- **Interactions** (`*` in the formula means a separate intercept and slope per brand).
- Advertising (feat) interactions, **confounding**, and the ŷ-vs-y fit plot.

The house style is the **statsmodels formula API** (`smf.ols(...).fit()`), `statsmodels.stats.outliers_influence.OLSInfluence`, `sm.add_constant`, numpy linear algebra, and `scipy.stats.t.ppf`.

Everything in L3 is **in-sample**. There is no out-of-sample R^2, no train/test split, no cross-validation, no regularization, no trees, no robust/HAC standard errors, and no time-series treatment.

---

## 1. Slide-by-slide notes

### p.1 — Title
"Lecture 3: Diagnostics and Transformations, Categorical Variables, Interactions, Multiple Linear Regressions". Dacheng Xiu, Chicago Booth, September 11, 2026.

### p.2–3 — SLR model assumptions (recap)
- **p.2 [slide]:** `Y_i | X_i ~ind N(β0 + β1 X_i, σ^2)`.
- The key assumptions:
  - (i) The conditional mean of Y is **linear** in X.
  - (ii) The additive errors (deviations from the line) are **normally distributed**, **independent** from each other, and **identically distributed** (i.e., they have **constant variance**).
  - The words "linear", "independent" and "constant variance" are in red on the slide.
- **p.3 [slide]:** "**Inference and prediction relies on this model being true!**" (red). If the model assumptions do not hold, "all bets are off":
  - prediction can be **systematically biased**;
  - standard errors, intervals and t-tests are **wrong**.
  - "We will focus on using **graphical methods (plots!)** to detect violations of the model assumptions."

### p.4–8 — Anscombe's quartet: why plots matter
- **p.4 [slide]:** "Anscombe's quartet comprises four datasets that have similar statistical properties..."
  - Code:
    ```python
    for c in ['1', '2', '3', '4']:
        x, y = anscombe['x'+c], anscombe['y'+c]
        print(x.mean(), y.mean(), x.std(), y.std(),
              np.corrcoef(x, y)[0, 1])
    ```
  - Output table:

    | set | mean(x) | mean(y) | sd(x) | sd(y) | corr |
    |---|---|---|---|---|---|
    | 1 | 9.0 | 7.5 | 3.317 | 2.032 | 0.816 |
    | 2 | 9.0 | 7.5 | 3.317 | 2.032 | 0.816 |
    | 3 | 9.0 | 7.5 | 3.317 | 2.030 | 0.816 |
    | 4 | 9.0 | 7.5 | 3.317 | 2.031 | 0.817 |

  - [my note] pandas `.std()` defaults to ddof=1. The sd values are sample sds.
- **p.5 [slide]:** "...but vary considerably when graphed". Four scatter plots:
  - Dataset 1: noisy linear.
  - Dataset 2: a smooth concave curve.
  - Dataset 3: an exact line plus one large outlier at x=13.
  - Dataset 4: all x=8 except one high-leverage point at x=19.
- **p.6 [slide]:** "Similarly, let's consider linear regression for each dataset". The same four panels with the fitted OLS line overlaid. The lines look identical.
- **p.7 [slide]:** "The regression lines and R^2 values are the same..."
  ```python
  # Store models in a list
  ansreg = [smf.ols(formula=f'y{i} ~ x{i}', data=anscombe).fit() for i in range(1, 5)]
  # Get the coefficients for each regression model
  coefs = np.column_stack([reg.params for reg in ansreg])
  coefs_rounded = np.round(coefs, 1)      # [[3.0 3.0 3.0 3.0], [0.5 0.5 0.5 0.5]]
  # Apply summary to each model and get R-squared values
  smry = [reg.rsquared for reg in ansreg]
  r_squared_rounded = np.round(smry, 1)   # [0.7, 0.7, 0.7, 0.7]
  ```
  So b0 = 3.0, b1 = 0.5 and R^2 = 0.7 for all four datasets.
- **p.8 [slide]:** "...but the residuals (**plotted against Ŷ**) look totally different." Residual-vs-fitted panels:
  - Set 1: patternless.
  - Set 2: inverted U, a sign of nonlinearity.
  - Set 3: a downward-sloping line plus one large positive outlier.
  - Set 4: a vertical stack plus one isolated point.
  - Key line (blue): "**Plotting e vs Ŷ is your #1 tool for finding fit problems.**"

### p.9–12 — Residuals, their distribution, and leverage
- **p.9 [slide]:** Model `Y_i = β0 + β1 X_i + ε_i`, with `ε_i ~iid N(0, σ^2)`.
  - The goal is to determine whether the "true" residuals are iid normal and unrelated to X.
  - If the SLR assumptions hold, the residuals must be "white noise":
    - (i) each ε_i has the same variance (σ^2);
    - (ii) each ε_i has the same mean (0);
    - (iii) all ε_i have the same normal distribution.
- **p.10 [slide]:** The true ε_i are unknown, so we look at the least-squares estimated residuals.
  - Fit `Y_i = b0 + b1 X_i + e_i`, with residuals `e_i = Y_i − Ŷ_i`.
  - "What should the e_i residuals look like if the SLR model is true?"
- **p.11 [slide]:** If the SLR model is true,
  - `e_i ~ N(0, σ^2 [1 − h_i])`, with `h_i = 1/n + (X_i − X̄)^2 / Σ_{j=1}^n (X_j − X̄)^2`.
  - h_i is the i-th observation's **leverage**: the point's share of the data (1/n) plus its proportional contribution to variability in X.
  - As n → ∞, h_i → 0 and the residuals become ε_i ~ N(0, σ^2).
  - [my note] In MLR, h_i is the i-th diagonal element of the hat matrix `X(X'X)^{-1}X'`, and Σ h_i = p. The MLR form is **not** on the slides. p.37 does show the related quantity `x_f'(X'X)^{-1}x_f`.
- **p.12 [slide]:** "Understanding Leverage".
  - h_i measures the sensitivity of the estimated LS line to changes in Y_i.
  - Mechanical intuition: the farther you are from a pivot joint, the more torque you have pulling on a lever.
  - "**Outliers do more damage if they have high leverage!**" (red)

### p.13–16 — Standardized and studentized residuals
- **p.13 [slide]:** Since `e_i ~ N(0, σ^2[1 − h_i])`, we have `e_i / (σ √(1 − h_i)) ~ N(0, 1)`.
  - These are the **Standardized Residuals**.
  - They all have the same distribution if the SLR model is true.
  - They are "almost (close enough) independent (~iid N(0,1))".
- **p.14 [slide]:** We don't know sd(ε) = σ.
  - Usually we estimate `σ^2 ≈ s^2 = (1/(n−p)) Σ_{j=1}^n e_j^2`, with **p = 2 for SLR** (in red).
  - Here we want to see whether any particular e_i is "too big", and "you don't want a single outlier to make s artificially large."
  - Figure: Anscombe dataset 3 with its fitted line. Caption: "**One big outlier can make s overestimate σ**".
- **p.15 [slide]:** The **Studentized Residual** is
  - `r_i = e_i / (s_{−i} √(1 − h_i)) ~ t_{n−p−1}(0, 1)`
  - where `s_{−i}^2 = (1/(n−p−1)) Σ_{j≠i} e_j^2` is "σ̂ calculated **without** e_i".
  - "These are easy to get in Python with the `OLSInfluence().resid_studentized_external` function."
    ```python
    OLSInfluence(reg1).resid_studentized_external
    # 0: 0.031345  1: -0.040845  2: -2.081099  3: 1.126800  4: -0.139801
    # 5: -0.038196 6: 1.116959   7: -0.704581  8: 1.838330  9: -1.568460  10: 0.156809
    ```
  - [my note, verified] `reg1` is Anscombe set 1 (`y1 ~ x1`, n=11); the numbers match exactly.
  - [my note, verified] The literal slide formula (drop e_i^2 from the full-fit SSE) is an approximation. statsmodels uses the exact leave-one-out variance, `s_{−i}^2 = [(n−p)s^2 − e_i^2/(1−h_i)]/(n−p−1)`. For obs 2 the slide formula gives −1.96, while statsmodels (and the slide's printed output) gives −2.081.
  - [my note] Import path: `from statsmodels.stats.outliers_influence import OLSInfluence`. It is not shown on the slide.
- **p.16 [slide]:** "Since the studentized residuals are distributed t_{n−p−1}(0,1), we should be **concerned about any r_i outside of about [−2.5, 2.5]**."
  - Example: the third Anscombe dataset.
  - Left plot: residuals vs fitted values. The outlier is at about +3.2; the other points lie on a downward-sloping line.
  - Right plot: studentized residuals vs fitted values. The outlier is at **about 1200**; all others are about 0.
  - [my note] The value is so large because, once the outlier is removed, the other 10 points are almost exactly collinear, so s_{−i} ≈ 0.

### p.17–18 — Dealing with outliers; leverage hides outliers
- **p.17 [slide]:** "When should you delete outliers? **Only when you have a really good reason!**" (red)
  - "There is nothing wrong with running regression **with and without** potential outliers to see whether results are significantly impacted."
  - "**Any time outliers are dropped, the reasons for doing so should be clearly noted.**" (blue)
- **p.18 [slide]:** **Warning:** "Unfortunately, outliers with **high leverage are hard to catch through r_i** (since the line is pulled towards them)."
  - Example: house Rent vs SqFt.
    - Left: scatter with the fitted line. Most points have SqFt < 20; a handful sit at SqFt about 30–85 with moderate rents, and they pull the line.
    - Right: studentized residuals vs SqFt, with dashed reference lines at about ±2.5. The far-right high-leverage points have r_i of only about −1.5 to −2, inside the band.
  - "**Plots of r_i or e_i against Ŷ_i or X_i are still your best diagnostic!**"

### p.19–26 — The MLR model and least squares
- **p.19 [slide]:** Beyond SLR. Examples:
  - multi-factor asset-pricing models (beyond CAPM);
  - demand for a product given prices of competing brands, advertising, household attributes;
  - more than size to predict house price.
  - MLR extends SLR to more than one independent variable.
- **p.20 [slide]:** `Y | X_1 ... X_d ~ind N(β0 + β1 X_1 + ... + βd X_d, σ^2)`.
  - The same assumptions: (i) the conditional mean is linear in the X_j; (ii) the errors are normal, independent and identically distributed (constant variance).
- **p.21 [slide]:** `β_j = ∂E[Y | X_1, ..., X_d] / ∂X_j`.
  - "**Holding all other variables constant**, β_j is the average change in Y per unit change in X_j."
- **p.22 [slide]:** Data are Y_i and `x_i = [X_{1i}, X_{2i}, ..., X_{di}]`, for i = 1..n, stored as a data array (**DataFrame**) with rows `[Y_i X_{1i} X_{2i} ... X_{di}]`.
- **p.23 [slide]:** `Y = β0 + β1 X1 + ... + βd Xd + ε`, `ε ~ N(0, σ^2)`.
  - Least squares works exactly as before: define the fitted values, then find the best-fitting **plane** by minimizing the sum of squared residuals.
- **p.24 [slide]:**
  - Fitted values: `Ŷ_i = b0 + b1 X_{1i} + b2 X_{2i} + ... + bd X_{di}`.
  - Residuals: `e_i = Y_i − Ŷ_i`.
  - Standard error: `s = sqrt( Σ_{i=1}^n e_i^2 / (n − p) )`, where **p = d + 1**.
  - Least squares: find b0, ..., bd to minimize s^2.
- **p.25 [slide]:** `Y = [Y_1 ... Y_n]'`, and X̂ is the n×(d+1) design matrix with a leading column of 1s.
  - `b = [b0 ... bd]' = (X̂'X̂)^{-1} X̂'Y`.
  - Intuition: b captures the covariance between X_j and Y (X̂'Y), normalized by the input sum of squares (X̂'X̂).
- **p.26 [slide] "Regression in Python — You need only one command":**
  ```python
  reg = smf.ols(y ~ var1 + ... + varP, data=mydata).fit()   # formula as a string in practice
  ```
  - "ols stands for Ordinary Least Squares."
  - `y ~ a + b` is the "formula" that defines the regression.
  - `reg` is "a list of useful things":
    - `reg.summary()` prints a bunch of information;
    - `reg.params` gives the coefficients;
    - `reg.predict(mynewdata)` predicts.
  - "**mynewdata must be a data frame with exactly the same format as mydata (same variable names, same factor levels).**"

### p.27–28 — Worked example: predicting the market equity premium (Welch & Goyal 2007)
- **p.27 [slide]:** Predict market equity returns from financial and macroeconomic indicators compiled by **Welch and Goyal (2007)**.
  - Data: **16 predictors, 74 annual observations, 1948 to 2021**.
  - There is a link, "Market Equity Premium Data Set".
  - `F1.head(10)` is shown. Columns: `dfy, infl, svar, d_e, lty, tms, tbl, dfr, d_p, d_y, ltr, e_p, b_m, ik, ntis, eqis, ret`. The target `ret` is the last column.
  - Row 0 values: dfy 0.0066, infl 0.088372, svar 0.019339, d_e −0.650588, lty 0.0243, tms 0.0148, tbl 0.0095, dfr 0.002911, d_p −2.902206, d_y −2.902206, ltr −0.026275, e_p −2.251619, b_m 0.725326, ik 0.034505, ntis 0.025922, eqis 0.239630, ret 0.033241.
  - [my note] The slides do not say whether the predictors in row t are lagged relative to `ret` in row t.
- **p.28 [slide] "Summary: Market Equity Premium":**
  ```python
  rest_full = F1.columns[0:-1].tolist()
  rest_full = '+'.join(rest_full)
  reg_full = smf.ols(formula='ret ~ {}'.format(rest_full), data=F1).fit()
  reg_full.summary()
  ```
  - Output header:
    - Dep. Variable: ret. Model: OLS. Method: Least Squares.
    - **R-squared 0.419; Adj. R-squared 0.281; F-statistic 3.040; Prob (F-statistic) 0.00142.**
    - Log-Likelihood 45.512; AIC −61.02; BIC −26.46.
    - No. Observations 74; Df Residuals 59; **Df Model 14**; Covariance Type: **nonrobust**.
  - Coefficient table:

    | var | coef | std err | t | P>\|t\| | [0.025 | 0.975] |
    |---|---|---|---|---|---|---|
    | Intercept | 1.1315 | 0.456 | 2.481 | 0.016 | 0.219 | 2.044 |
    | dfy | 9.4915 | 7.093 | 1.338 | 0.186 | −4.701 | 23.684 |
    | infl | −2.0196 | 1.094 | −1.846 | 0.070 | −4.209 | 0.170 |
    | svar | 1.5749 | 0.960 | 1.640 | 0.106 | −0.347 | 3.496 |
    | d_e | −0.0248 | 0.069 | −0.360 | 0.720 | −0.163 | 0.113 |
    | lty | −0.8330 | 0.719 | −1.158 | 0.251 | −2.272 | 0.606 |
    | tms | −0.1880 | 1.067 | −0.176 | 0.861 | −2.324 | 1.948 |
    | tbl | −0.6451 | 0.792 | −0.815 | 0.418 | −2.229 | 0.939 |
    | dfr | 0.3791 | 0.562 | 0.675 | 0.502 | −0.745 | 1.503 |
    | d_p | 0.1297 | 0.119 | 1.088 | 0.281 | −0.109 | 0.368 |
    | d_y | −0.0127 | 0.150 | −0.084 | 0.933 | −0.314 | 0.288 |
    | ltr | 0.1221 | 0.227 | 0.538 | 0.592 | −0.332 | 0.576 |
    | e_p | 0.1545 | 0.086 | 1.800 | 0.077 | −0.017 | 0.326 |
    | b_m | −0.0574 | 0.229 | −0.250 | 0.803 | −0.516 | 0.401 |
    | ik | −2.8554 | 9.049 | −0.316 | 0.753 | −20.963 | 15.252 |
    | ntis | 2.1022 | 1.319 | 1.594 | 0.116 | −0.536 | 4.741 |
    | eqis | −0.9558 | 0.337 | −2.833 | 0.006 | −1.631 | −0.281 |

  - Only the Intercept and eqis are significant at 5%. The large in-sample R^2 (0.419) shrinks to 0.281 after the degrees-of-freedom adjustment.
  - **[my note, verified from the p.27 head() values] Exact collinearity is hidden in this output.**
    - `d_e = d_p − e_p` exactly (row 0: −2.902206 − (−2.251619) = −0.650587). This is the log dividend/earnings identity.
    - `tms = lty − tbl` exactly (row 0: 0.0243 − 0.0095 = 0.0148).
    - So the 17-column design has rank 15. That is why **Df Model = 14, not 16**, and Df Resid = 74 − 15 = 59.
    - statsmodels silently uses a pseudo-inverse, so the individual coefficients of d_e/d_p/e_p and tms/lty/tbl are not identified.
    - The slide does not point this out. It is directly relevant to the exam's "duplicate columns" warning (Problem 3, extended macro file).

### p.29–36 — MLR inference
- **p.29 [slide]:** Concepts translate directly from SLR:
  - individual parameter inference and estimation are the same, "**conditional on the rest of the model**";
  - **ANOVA is exactly the same, and the F-test still applies**;
  - our diagnostics and transformations apply directly.
  - The hardest part is the matrix algebra: "Luckily, Python does all that for you."
- **p.30 [slide]:** Residual standard error.
  - `s^2 = var(e) = Σ_{i=1}^n (Y_i − Ŷ_i)^2 / (n − p)`, with `Ŷ_i = b0 + Σ b_j X_{ji}` and **p = d + 1**.
  - The residual standard error is `σ̂ = s = √s^2`.
  - [my note] In statsmodels this is `reg.scale` (used on p.38), and `s = np.sqrt(reg.scale)`.
- **p.31 [slide]:** MLR residuals are purged of any relationship to the independent variables: `Y = Ŷ + e`, `corr(X_j, e) = 0`, `corr(Ŷ, e) = 0`.
- **p.32 [slide]:** The LS coefficients are random (they differ across samples) and correlated with each other.
  - **Unbiased:** `E[b_j] = β_j` for j = 0..d.
  - Sampling distribution: `b ~ N(β, S_b)` (multivariate normal), with β = [β0 ... βd]'.
- **p.33 [slide]:** var(b) is the p×p covariance matrix S_b, with var(b_j) on the diagonal and cov(b_j, b_k) off it.
  - "⇒ **Standard errors are the square root of the diagonal of S_b.**"
  - `S_b = s^2 (X̂'X̂)^{-1}`.
- **p.34 [slide]:** "You can get Covariance Matrix with `cov_params()`":
  ```python
  cov_matrix = reg_full.cov_params()
  cov_matrix
  ```
  - A 17×17 table (Intercept plus the 16 predictors) is shown. Examples:
    - var(Intercept) = 0.208015, so sqrt = 0.456, which matches its std err on p.28.
    - var(dfy) = 50.304174 (sqrt = 7.093).
    - var(ik) = 81.888835 (sqrt = 9.049).
    - cov(lty, tms) = 0.514876; cov(tms, tbl) = −0.624236.
  - [my note] The large off-diagonals among lty/tms/tbl are the numerical footprint of the exact collinearity noted above.
- **p.35 [slide]:** Intervals and t-statistics are "**exactly the same** as in SLR":
  - a (1−α)100% CI for β_j is `b_j ± t_{α/2, n−p} s_{b_j}`;
  - `z_{b_j} = (b_j − β_j^0)/s_{b_j} ~ t_{n−p}(0,1)` is the number of standard errors between the LS estimate and the null value.
  - "Intervals and testing via b_j & s_{b_j} are **one-at-a-time procedures**: You are evaluating the j-th coefficient conditional on the other X's being in the model, but regardless of the values you've estimated for the other b's."
- **p.36 [slide]:** R^2 for multiple regression.
  - `R^2 = SSR/SST = Σ(Ŷ_i − Ȳ)^2 / Σ(Y_i − Ȳ)^2`.
  - Correlation interpretation: `R^2 = cor^2(Ŷ, Y) = r_{ŷy}^2` (in SLR, r_{ŷy} = r_{xy} since cor(X, Ŷ) = 1).
  - [my note] In this course's notation (from L2), **SSR is the regression (explained) sum of squares**, not the residual sum of squares. SSE is the residual sum of squares. Caution: statsmodels `reg.ssr` is the *residual* SS, and `reg.ess` is the explained SS. R^2 here is in-sample. Adjusted R^2 appears in the output but is not defined in L3.

### p.37–38 — Forecasting and prediction intervals in MLR
- **p.37 [slide]:** For new data `x_f = [X_{1,f} ... X_{d,f}]'`:
  - `E[Y_f | x_f] = Ŷ_f = b0 + b1 X_{1f} + ... + bd X_{df}`;
  - `var[Y_f | x_f] = var(Ŷ_f) + var(e_f) = s_fit^2 + s^2 = s_pred^2`;
  - with X̂ the design matrix and `x̂_f = [1, X_{1,f} ... X_{d,f}]'`, `s_fit^2 = s^2 · x̂_f' (X̂'X̂)^{-1} x̂_f`;
  - a (1−α) level **prediction interval** is `Ŷ_f ± t_{α/2, n−p} s_pred`.
- **p.38 [slide] "Prediction in MLR" code:**
  ```python
  reg_reduced = smf.ols(formula='ret ~ infl + lty + tbl', data=F1).fit()
  Xnew = pd.DataFrame([[0.03, 0.02, 0.01]], columns=['infl', 'lty', 'tbl'])
  reg_reduced.get_prediction(Xnew).summary_frame(alpha=0.05)
  #  mean 0.091831 | mean_se 0.033244 | mean_ci_lower 0.025529 | mean_ci_upper 0.158134
  #  obs_ci_lower -0.236458 | obs_ci_upper 0.42012

  X = sm.add_constant(X_reduced)          # add 1
  Xf = np.array([1, 0.03, 0.02, 0.01]).T
  y_hat_f = Xf @ reg_reduced.params       # 0.0918
  n = len(X)
  p = len(reg_reduced.params)
  s2 = reg_reduced.scale
  s2_fit = s2 * (Xf.T @ np.linalg.inv(X.T @ X) @ Xf)
  s_pred = np.sqrt(s2 + s2_fit)
  t_critical = t.ppf(1 - 0.05 / 2, df=reg_reduced.df_resid)
  lower_bound = y_hat_f - t_critical * s_pred
  upper_bound = y_hat_f + t_critical * s_pred
  # Prediction Interval: [-0.2365, 0.4201]
  ```
  - In `summary_frame`, `mean_ci_*` is the CI for the conditional mean (uses s_fit only).
  - `obs_ci_*` is the **prediction interval** (uses s_pred = sqrt(s^2 + s_fit^2)). The hand computation reproduces `obs_ci_*`.
  - [my note] `X_reduced` is not defined on the slide; presumably `F1[['infl','lty','tbl']]`. `t` is `scipy.stats.t`. `mean_se` (0.033244) equals s_fit.

### p.39–44 — Multicollinearity
- **p.39 [slide]:** Multicollinearity is strong linear dependence between some of the covariates in an MLR.
  - "**The usual marginal effect interpretation is lost**": a change in one X leads to change in others.
  - Coefficient standard errors will be large, so multicollinearity leads to **large uncertainty about the b_j's**.
  - "Regression coefficient estimates are **unstable**. Dropping or adding variables may cause significant changes in the rest coefficient estimates."
- **p.40 [slide]:** Regress Y on X1 and X2 = 10·X1.
  - Then `E[Y] = β0 + β1 X1 + β2 (10 X1)` and `∂E[Y|X1,X2]/∂X1 = β1 + 10 β2`.
  - "**X1 and X2 do not act independently!**"
- **p.41 [slide]:** Three covariates: `infl` (inflation), `lty` (long-term yield), `tbl` (Treasury bill; spelled "Tresuary" on the slide).
  - Pairwise scatter plots show correlations: **infl–lty r = 0.6, infl–tbl r = 0.7, lty–tbl r = 0.9.**
  - "Sure enough, they are all correlated with each other."
- **p.42 [slide]:** "In the 3 covariate regression, **none of the effects are significant**":
  ```python
  reg_reduced = smf.ols(formula='ret ~ infl + lty + tbl', data=F1).fit()
  reg_reduced.summary()
  ```
  - R^2 0.165; Adj R^2 0.130; **F 4.627, Prob(F) 0.00520**; LL 32.108; AIC −56.22; BIC −47.00; n 74; Df Resid 70; Df Model 3; nonrobust.

    | var | coef | se | t | p | 95% CI |
    |---|---|---|---|---|---|
    | Intercept | 0.1083 | 0.047 | 2.325 | 0.023 | [0.015, 0.201] |
    | infl | −0.0007 | 0.910 | −0.001 | 0.999 | [−1.816, 1.815] |
    | lty | 0.5056 | 1.428 | 0.354 | 0.724 | [−2.343, 3.354] |
    | tbl | −2.6552 | 1.459 | −1.820 | 0.073 | [−5.565, 0.255] |

  - "**But the F-stat is significant with p-value = 0.005!**" (red)
- **p.43 [slide]:** "If you look at individual regression effects, **all 3 are significant**":
  ```python
  reg_infl = smf.ols(formula='ret ~ infl', data=F1).fit();  print(reg_infl.summary())
  reg_lty  = smf.ols(formula='ret ~ lty',  data=F1).fit();  print(reg_lty.summary())
  reg_tbl  = smf.ols(formula='ret ~ tbl',  data=F1).fit();  print(reg_tbl.summary())
  ```
  - infl: Intercept 0.0879 (se 0.030, t 2.902, p 0.005); **infl −1.6956 (se 0.671, t −2.526, p 0.014, CI [−3.034, −0.358])**.
  - lty: Intercept 0.1449 (0.042, 3.411, 0.001); **lty −2.0887 (0.685, −3.049, 0.003, [−3.454, −0.723])**.
  - tbl: Intercept 0.1201 (0.030, 3.942, 0.000); **tbl −2.2545 (0.600, −3.758, 0.000, [−3.450, −1.059])**.
  - Note that the sign of lty flips from −2.09 (alone) to +0.51 (jointly). That is the instability described on p.39.
- **p.44 [slide]:** "**Multicollinearity is not a big problem in and of itself, you just need to know that it is there.**"
  - Fitting could remain good. **No effect on prediction.**
  - If you recognize multicollinearity:
    - understand that the β_j are **not true marginal effects**;
    - consider **dropping variables** to get a simpler model ("**use the partial F-test!**");
    - expect big standard errors on your coefficients (the coefficient estimates are unstable).
  - [my note] "No effect on prediction" is an in-sample and interpolation statement. The slide says nothing about out-of-sample forecast variance. The partial F-test is named but its formula is not given in L3.

### p.45–52 — Categorical effects, dummy variables, logs, elasticity, design matrix
- **p.45 [slide]:** Regress Y on a categorical X with R levels:
  - `E[Y | X = 0] = β0`; `E[Y | X = r] = β0 + β_r`, for r = 1, ..., R−1.
  - Use **dummy, binary, or indicator variables**. "Dummy variables allow the mean (intercept) to shift by taking on the value 0 or 1."
  - A factor with **R levels is represented through R − 1 dummy variables**: `E[Y|X] = β0 + β1 1[X=1] + β2 1[X=2] + ... + β_{R−1} 1[X=R−1]`, where `1[X=r] = 1` if X = r and 0 otherwise.
  - "What is E[Y|X] if X = 1?" (Answer: β0 + β1.)
- **p.46 [slide] Orange Juice dataset:**
  - Three brands (b): Tropicana, Minute Maid, Dominicks; 83 Chicagoland stores, with demographic info for each.
  - Variables: price, sales (log of # units sold), and whether advertised (`feat`).
  - Data in `oj.csv`. Source: bayesm & Montgomery, 1987.
- **p.47 [slide] "The Juice: price, brand, and sales":**
  - Left: conditional box plots of log price by brand, split by feat (0/1). Featured weeks have lower prices within each brand.
  - Right: scatter of log price vs log sales, colored by brand.
  - "**Each brand occupies a well defined price range. Sales decrease with price.**" Tropicana is the most expensive, then Minute Maid, then Dominicks.
- **p.48 [slide] "Thinking About Scale":** "When making a linear point (this goes up, that goes down) think about the **scale on which you expect to find linearity**."
  - Figure: GDP vs IMPORTS by country on raw scale (everything squashed near 0, with the US far away) vs log(GDP) vs log(IMPORTS) (a roughly linear cloud).
  - "**If your scatterplots look like the left panel, consider using log. Outliers?**"
- **p.49 [slide] "Log Linear":** We often model the mean of log(y) instead of y. Why? **Multiplicative** (rather than additive) change.
  - `log(y) = log(a) + xβ ⇔ y = a e^{xβ}`. "Predicted y is multiplied by e^β after a unit increase in x."
  - Recall `log(y) = z ⇔ e^z = y`, e ≈ 2.7; `log(ab) = log a + log b`; `log(a^b) = b log a`.
  - "**I use log = ln, natural log.** Anything else will be noted, e.g., log2."
  - "**Whenever y changes on a percentage scale, use log(y).**" Examples: prices ("Foreclosed homes sell at a 20% to 30% discount"); sales ("our y.o.y. sales are up 20% across models").
- **p.50 [slide] "Price Elasticity":** A simple OJ elasticity model for sales y: `E[log y] = γ log(price) + x'β`.
  - "Elasticities and log-log regression: for small values we can interpret **γ as % change in y per 1% increase in price**."
  ```python
  smf.ols(formula='log_sales ~ log_price + brand', data=oj).fit()
  # (Intercept) 10.8288 | log(price) -3.1387 | brand minute.maid 0.8702 | brand tropicana 1.5299
  ```
  - "...and see sales drop by about **3.1% for every 1% price hike**."
  - (The printed labels such as "branBDinute.maid" are garbled R-style names carried over on the slide. In Python the names are `brand[T.minute.maid]` and `brand[T.tropicana]`; see p.51.)
- **p.51 [slide] "The Design Matrix":** "What happened to `branddominicks`?"
  - brand is not a number, so you can't do brand×β. The first step is to create a numeric **design matrix**, via a call to the **`dmatrix`** function (patsy).
  - Table: columns `Intercept | brand[T.minute.maid] | brand[T.tropicana] | log_price`, with first rows (1, 0, 1, 1.160021), (1, 1, 0, 1.026042), (1, 0, 0, 0.329304), (1, 0, 1, 1.078410), (1, 1, 0, 0.524729).
  - "The dummy variable is on the left, and on the right we have numeric x that we can multiply against β coefficients."
- **p.52 [slide] "Intercepts":**
  - `dmatrix(' ~ log_price + brand', data = oj)` builds this 4-column design.
  - "Each factor's **reference level is absorbed by the intercept**. Coefficients are '**change relative to reference**' (dominicks here)."
  - "**Why not include all three dummies?**" [my note] With an intercept, three brand dummies sum to the intercept column, which gives perfect collinearity (the dummy-variable trap; compare p.40). Dominicks is the reference because patsy's default Treatment coding uses the first level in sorted order.

### p.53–57 — Interactions, advertising, confounding
- **p.53 [slide] "Interaction":** "Beyond additive effects: variables change how others act on y."
  - An interaction term is the **product of two covariates**: `E[y | x] = ... + β_j x_j + x_j x_k β_jk`.
  - So the effect on E[y|x] of a unit increase in x_j is **β_j + x_k β_jk**: "**It depends upon x_k!**"
  - "**Interactions play a massive role in statistical learning**, and they are often central to social science and business questions." Examples:
    - Does gender change the effect of education on wages?
    - Do patients recover faster when taking drug A?
    - How does advertisement affect price sensitivity?
- **p.54 [slide] "Fitting interactions in Python: use * in your formula."**
  ```python
  reg_interact = smf.ols(formula='log_sales ~ log_price * brand', data=oj).fit()
  # (Intercept) 10.95468 | log(price) -3.37753 | minute.maid 0.88825 | tropicana 0.96239
  # log(price):minute.maid 0.05679 | log(price):tropicana 0.66576
  ```
  - "This is the model `E[log(v)] = α_b + β_b log(price)`: **a separate intercept and slope for each brand 'b'**."
  - "**Why not separately estimate three models?**"
  - "Elasticities are dominicks: −3.4, minute maid: −3.3, tropicana: −2.7. Where do these numbers come from? Do they make sense?"
    - [my note, arithmetic] Dominicks = −3.37753 (reference slope). Minute Maid = −3.37753 + 0.05679 = −3.32. Tropicana = −3.37753 + 0.66576 = −2.71.
    - [my note] Why pool instead of three separate regressions? A fully interacted pooled model gives the same point estimates as three separate fits, but one common σ is estimated from all the data, and brand differences can be tested directly (interaction t-stats and F-tests). Partial pooling (e.g., common slope, brand intercepts as on p.50) is also possible. The slide poses the question but does not answer it.
- **p.55 [slide] "Advertisements":** A key question: what changes when we feature a brand (in-store display promo or flier ad)? Three models, in increasing richness:
  1. Additive effect on log sales: `E[log(v)] = α_b + 1[feat] α_feat + β_b log(p)`.
  2. That plus its effect on elasticity: `E[log(v)] = α_b + β_b log(p) + 1[feat] (α_feat + β_feat log(p))`.
  3. Brand-specific effect on elasticity: `E[log(v)] = α_b + β_b log(p) + 1[feat] (α_{b,feat} + β_{b,feat} log(p))`.
  - "See the Python code for runs of all three models. Connect the regression formula and output to these equations."
  - [my note] The code is not on the slides. The natural formulas are:
    - (1) `'log_sales ~ log_price*brand + feat'`;
    - (2) `'log_sales ~ log_price*brand + log_price*feat'`;
    - (3) `'log_sales ~ log_price*brand*feat'`.
- **p.56 [slide] "Brand-specific Elasticities":**

  | | Dominicks | Minute Maid | Tropicana |
  |---|---|---|---|
  | Not Featured | −2.8 | −2.0 | −2.0 |
  | Featured | −3.2 | −3.6 | −3.5 |

  - "**Ads always decrease elasticity**" (i.e., make it more negative, meaning more price sensitive).
  - "Minute Maid and Tropicana elasticities drop 1.5% with ads, moving them from less to more price sensitive than Dominicks."
  - Discussion questions: why does marketing increase price sensitivity? How does this influence pricing and marketing strategy?
- **p.57 [slide] "Confounding":** "Before including feat, Minute Maid behaved like Dominicks. With feat, Minute Maid looks more like Tropicana. Why?"
  - Figure: mosaic plot of the amount of advertising by brand. Minute Maid has the largest "ads" share.
  - "Because Minute Maid was more heavily promoted, and promotions have a negative effect on elasticity, we were **confounding** the two effects in the brand average elasticity."

### p.58 — Fit plot
- **[slide]** "Fit Plot: ŷ vs y": scatter of predicted sales vs log sales for the OJ regression, colored by brand, with a 45° line.
- "**It's good practice to plot ŷ vs y as a check for misspecification.** (e.g., non-constant variance, nonlinearity in residuals, ...)"

---

## 2. Formula sheet (all L3)

| Concept | Formula | Slide |
|---|---|---|
| SLR model | `Y_i \| X_i ~ind N(β0 + β1 X_i, σ^2)`; `Y_i = β0 + β1 X_i + ε_i`, `ε_i ~iid N(0, σ^2)` | p.2, p.9 |
| Sample residual | `e_i = Y_i − Ŷ_i` | p.10, p.24 |
| Residual distribution | `e_i ~ N(0, σ^2 [1 − h_i])` | p.11 |
| Leverage (SLR) | `h_i = 1/n + (X_i − X̄)^2 / Σ_j (X_j − X̄)^2`; h_i → 0 as n → ∞ | p.11 |
| Standardized residual | `e_i / (σ √(1 − h_i)) ~ N(0,1)` | p.13 |
| Error variance estimate | `s^2 = (1/(n−p)) Σ e_j^2`, p = 2 in SLR, p = d+1 in MLR | p.14, p.24, p.30 |
| Studentized residual | `r_i = e_i / (s_{−i} √(1 − h_i)) ~ t_{n−p−1}(0,1)`, `s_{−i}^2 = (1/(n−p−1)) Σ_{j≠i} e_j^2` | p.15 |
| Outlier flag | \|r_i\| > about 2.5 | p.16 |
| MLR model | `Y \| X_1..X_d ~ind N(β0 + Σ β_j X_j, σ^2)` | p.20, p.23 |
| Marginal effect | `β_j = ∂E[Y \| X_1..X_d]/∂X_j`, holding all others constant | p.21 |
| Fitted values | `Ŷ_i = b0 + b1 X_{1i} + ... + bd X_{di}` | p.24 |
| Residual SE | `s = sqrt(Σ e_i^2 / (n − p))`, p = d + 1; `σ̂ = s` | p.24, p.30 |
| OLS estimator | `b = (X̂'X̂)^{-1} X̂'Y` | p.25 |
| Orthogonality | `corr(X_j, e) = 0`, `corr(Ŷ, e) = 0`, `Y = Ŷ + e` | p.31 |
| Unbiasedness and sampling distribution | `E[b_j] = β_j`; `b ~ N(β, S_b)` | p.32 |
| Coefficient covariance | `S_b = s^2 (X̂'X̂)^{-1}`; SE = sqrt(diag S_b) | p.33 |
| CI for β_j | `b_j ± t_{α/2, n−p} s_{b_j}` | p.35 |
| t-stat | `z_{b_j} = (b_j − β_j^0)/s_{b_j} ~ t_{n−p}` | p.35 |
| R^2 | `SSR/SST = Σ(Ŷ_i − Ȳ)^2 / Σ(Y_i − Ȳ)^2 = cor^2(Ŷ, Y)` | p.36 |
| Point forecast | `Ŷ_f = b0 + Σ b_j X_{jf}` | p.37 |
| Forecast variance | `s_pred^2 = s^2 + s_fit^2`, `s_fit^2 = s^2 x̂_f'(X̂'X̂)^{-1} x̂_f` | p.37 |
| Prediction interval | `Ŷ_f ± t_{α/2, n−p} s_pred` | p.37, p.38 |
| Collinear marginal effect | X2 = 10 X1 ⇒ `∂E[Y]/∂X1 = β1 + 10β2` | p.40 |
| Categorical regression | `E[Y\|X=0] = β0`; `E[Y\|X=r] = β0 + β_r`; R−1 dummies `E[Y\|X] = β0 + Σ_{r=1}^{R−1} β_r 1[X=r]` | p.45 |
| Log-linear | `log y = log a + xβ ⇔ y = a e^{xβ}`; unit increase in x multiplies y by e^β | p.49 |
| Elasticity | `E[log y] = γ log(price) + x'β`; γ = % change in y per 1% change in price | p.50 |
| Interaction | `E[y\|x] = ... + β_j x_j + x_j x_k β_jk`; effect of x_j is `β_j + x_k β_jk` | p.53 |
| Brand-specific model | `E[log v] = α_b + β_b log(price)` | p.54 |
| Feat models | see the three equations on p.55 | p.55 |

---

## 3. Code idioms (house style)

Imports used on the slides (implied): `import numpy as np`, `import pandas as pd`, `import statsmodels.formula.api as smf`, `import statsmodels.api as sm`, `from statsmodels.stats.outliers_influence import OLSInfluence`, `from scipy.stats import t`, `from patsy import dmatrix`.

- **Summary statistics loop** (p.4): `x.mean(), y.mean(), x.std(), y.std(), np.corrcoef(x, y)[0, 1]`.
- **Fit many OLS models with an f-string formula** (p.7): `[smf.ols(formula=f'y{i} ~ x{i}', data=anscombe).fit() for i in range(1,5)]`.
- **Stack parameters** (p.7): `np.column_stack([reg.params for reg in ansreg])`, then `np.round(..., 1)`.
- **R^2 from a fit** (p.7): `reg.rsquared`.
- **Studentized residuals** (p.15): `OLSInfluence(reg1).resid_studentized_external`.
- **One-command regression** (p.26): `reg = smf.ols('y ~ var1 + ... + varP', data=mydata).fit()`. Then `reg.summary()`, `reg.params`, `reg.predict(mynewdata)`. The new data must have the same variable names and factor levels.
- **Build a formula from all columns except the target** (p.28):
  ```python
  rest_full = '+'.join(F1.columns[0:-1].tolist())
  smf.ols(formula='ret ~ {}'.format(rest_full), data=F1).fit()
  ```
- **Coefficient covariance** (p.34): `reg_full.cov_params()`.
- **Prediction with intervals** (p.38): `reg.get_prediction(Xnew).summary_frame(alpha=0.05)`. The output columns are `mean, mean_se, mean_ci_lower, mean_ci_upper, obs_ci_lower, obs_ci_upper`.
- **Manual OLS prediction interval** (p.38):
  - `sm.add_constant(X)`, `Xf @ reg.params`, `reg.scale` (= s^2), `np.linalg.inv(X.T @ X)`;
  - `t.ppf(1 - 0.05/2, df=reg.df_resid)`;
  - `len(reg.params)` (= p).
- **Univariate regression and print** (p.43): `print(smf.ols(formula='ret ~ infl', data=F1).fit().summary())`.
- **Categorical in a formula** (p.50): string column `brand` in `'log_sales ~ log_price + brand'`. patsy creates the R−1 treatment dummies automatically. Printed names look like `brand[T.minute.maid]`.
- **Design matrix** (p.51–52): `dmatrix(' ~ log_price + brand', data=oj)`.
- **Interaction** (p.54): `'log_sales ~ log_price * brand'`. The `*` gives main effects plus product terms; the product terms are printed as `log_price:brand[T.x]`.
- **Plots used**:
  - scatter plots with fitted lines (p.5–6);
  - residual vs fitted (p.8, p.16);
  - studentized residual vs X with ±2.5 guide lines (p.18);
  - pairwise scatter with r in the title (p.41);
  - conditional box plots by group and hue (p.47);
  - scatter colored by category (p.47);
  - raw vs log scatter (p.48);
  - mosaic plot (p.57);
  - ŷ vs y fit plot with a 45° line (p.58).
  - The figure style looks like seaborn/matplotlib, with a white grid.

---

## 4. Course conventions established in L3

1. **log means the natural log**. Anything else is labelled, e.g., log2 (p.49).
2. **p counts the intercept**: p = d + 1 in MLR, p = 2 in SLR. The residual-variance denominator is **n − p** (p.14, p.24, p.30). Studentized residuals use n − p − 1 degrees of freedom (p.15). t-intervals and tests use t_{n−p} (p.35, p.37).
3. **The primary diagnostic is a residual plot:** e (or r) vs Ŷ, or vs X (p.8, p.18), supplemented by a ŷ-vs-y fit plot (p.58). Diagnosis is graphical, not by formal tests (p.3).
4. **Outlier flag:** externally studentized residual outside about [−2.5, 2.5] (p.16).
5. **Outlier policy:** delete only for a really good reason; run with and without; always document why (p.17).
6. **Coefficient interpretation:** "holding all other variables constant", i.e., the partial effect (p.21). Inference is "conditional on the rest of the model" (p.29) and one-at-a-time (p.35).
7. **Categorical coding:** R − 1 dummies; the reference level is absorbed by the intercept; coefficients are changes relative to the reference (p.45, p.52). Default reference = first level (dominicks) (p.52).
8. **Interactions via `*`** in the formula, which includes the main effects (p.54). An interaction effect is read as β_j + x_k β_jk (p.53).
9. **Use log(y) when y changes on a percentage scale**. Log-log slopes are elasticities (p.49–50).
10. **New data for `predict` must match the training frame's names and factor levels** (p.26).
11. **Prediction intervals include parameter uncertainty** (s_fit^2) plus noise (s^2) (p.37–38).
12. Standard errors in all L3 output are the default **"nonrobust"** OLS standard errors (p.28, p.42).

---

## 5. Pitfalls and warnings the professor emphasises

- "**Inference and prediction relies on this model being true!**" If the assumptions fail, prediction can be systematically biased and SEs, intervals and t-tests are wrong (p.3).
- **Identical summary statistics, fitted lines and R^2 can hide completely different data.** Always plot (Anscombe, p.4–8).
- **One big outlier inflates s**, so it can hide itself in standardized residuals. Use studentized (leave-one-out) residuals (p.14–15).
- **High-leverage outliers are hard to catch with r_i** because the line is pulled toward them. Plot r_i or e_i vs Ŷ or X (p.18). Outliers do more damage with high leverage (p.12).
- **Don't delete outliers without a really good reason. Document every drop** (p.17).
- **Multicollinearity** (p.39–44):
  - the marginal-effect interpretation is lost;
  - standard errors are large and coefficients unstable;
  - adding or dropping variables changes the other coefficients (the sign of lty flips, p.42 vs p.43);
  - individual t-tests can all be insignificant while the joint F-test is significant (p.42);
  - it does not hurt fit or prediction per se (p.44).
- **Exact collinearity (X2 = 10·X1)** means the effects cannot be separated (p.40). [my note] The same thing happens silently in the Welch–Goyal full regression (Df Model 14 < 16, p.28).
- **Dummy-variable trap:** do not include all R dummies alongside an intercept (the p.52 question).
- **Scale matters for linearity:** consider logs when a scatter looks like the raw GDP/IMPORTS panel (p.48).
- **Confounding:** omitting a correlated driver (feat) distorts brand elasticities (p.57). The estimated "brand effect" partly reflects promotion intensity.
- **Fit plot ŷ vs y** to catch non-constant variance and nonlinearity (p.58).

---

## 6. Datasets and worked examples

| Dataset | Description | Slides | Key numbers |
|---|---|---|---|
| Anscombe's quartet | 4 datasets, n = 11 each; columns x1..x4, y1..y4 | p.4–8, 14–16 | mean x = 9, mean y = 7.5, sd x = 3.317, sd y ≈ 2.03, corr ≈ 0.816; b0 = 3.0, b1 = 0.5, R^2 ≈ 0.7 for all 4; set 3 outlier r_i ≈ 1200 |
| House Rent vs SqFt | rents vs square footage (units unstated), a few high-leverage large units | p.18 | high-leverage points have r_i within ±2.5 |
| Welch & Goyal (2007) market equity premium (`F1`) | 16 predictors (dfy, infl, svar, d_e, lty, tms, tbl, dfr, d_p, d_y, ltr, e_p, b_m, ik, ntis, eqis), target `ret`; 74 annual observations 1948–2021 | p.27–28, 34, 38, 41–43 | full: R^2 0.419, adj 0.281, F 3.04 (p 0.00142); reduced (infl, lty, tbl): R^2 0.165, F 4.627 (p 0.0052); forecast at (0.03, 0.02, 0.01) = 0.0918, PI [−0.2365, 0.4201] |
| Orange juice (`oj.csv`) | Dominick's scanner data: 3 brands, 83 Chicagoland stores, price, log units sold, feat, store demographics (bayesm; Montgomery 1987) | p.46–58 | additive elasticity −3.14; per-brand interacted elasticities −3.4/−3.3/−2.7; with feat: not featured −2.8/−2.0/−2.0, featured −3.2/−3.6/−3.5 |
| GDP vs imports by country | illustration of the log scale | p.48 | figure only |

---

## 7. Boundaries — what L3 does **not** teach (or mentions only in passing)

**Not in L3 at all:**
- Out-of-sample R^2, benchmarks such as the trailing mean or zero, train/validation/test splits.
- Cross-validation (k-fold, stratified, time-series), expanding or rolling windows.
- Regularization (ridge, lasso), model selection criteria as a method, PCR.
- Trees and ensembles, feature importance, classification or accuracy.
- Time-series dependence (autocorrelation, HAC/Newey–West), robust/heteroskedasticity-consistent SEs.
- Formal normality or heteroskedasticity tests (no QQ plot, no Breusch–Pagan).
- Cook's distance, VIF, winsorization, standardization/z-scoring of predictors.
- Look-ahead/leakage discussion, portfolio construction, Sharpe ratio, VaR, kurtosis.

**Mentioned only in passing (named, not taught):**
- ANOVA and the F-test for MLR: "exactly the same" (p.29). The F-stat is used on p.42 to show joint significance, but no MLR F formula is given.
- The **partial F-test** for dropping variables: named on p.44, formula not given.
- **AIC, BIC, Log-Likelihood, Adj. R-squared**: appear only in `summary()` output (p.28, p.42). Not defined or used.
- "Covariance Type: nonrobust" appears in the output; alternatives are not discussed.
- "Interactions play a massive role in statistical learning" (p.53): motivational only.
- The three advertising models (p.55) are written out, but their code is deferred to "the Python code" (not on the slides). Only the resulting elasticity table (p.56) is shown.
- MLR leverage/hat matrix: only the SLR leverage formula is given (p.11).
- Welch & Goyal (2007) is used only as an in-sample MLR/multicollinearity example. Its famous out-of-sample finding is **not** discussed in L3.

---

## 8. Mapping to the final exam (BUSN 41210 Final, Autumn 2026)

Be honest about coverage: L3 is an in-sample linear-regression lecture. It is the **primary** source for OLS mechanics, dummies, interactions, collinearity, outliers, transformations and residual diagnostics. It is **not** a source for CV, OOS evaluation, trees or VaR, which must come from other lectures.

### Problem 1 (Trees & Ensembles, Social_Network_Ads)
- **Gender encoding (setup cell).** The notebook does `ads['Gender'] = (ads.Gender == 'Male').astype(int)`. That is exactly the **R − 1 = 1 dummy** convention of L3 p.45 (Female is the reference level, coded 0). When describing the tree in plain English (1.2), a split on Gender ≤ 0.5 means "Female vs Male" (p.45, p.51–52 interpretation of dummies).
- **1.1 baseline / 1.3 why ensembles or trees can beat a linear view.** L3 p.53 says an interaction makes the effect of x_j depend on x_k. A linear model captures that only with explicit product terms (p.54 `*`). Purchase depends jointly on Age and EstimatedSalary, and trees capture interactions automatically. Supporting argument only; the tree and ensemble content comes from other lectures.
- **1.4 impurity vs permutation importance.**
  - L3 p.39–44 (multicollinearity) supports the caution that correlated predictors make "importance" and marginal attributions unstable and non-separable (p.40: "X1 and X2 do not act independently!"; p.44: the β_j are not true marginal effects).
  - p.57 (confounding) supports the caution that importance is not causation.
  - Use these as interpretation caveats when advising a decision maker.
- **1.5 time-series rows.** L3 p.2/p.9/p.20 assume **independent** errors, and p.3 says that when assumptions fail "all bets are off: prediction can be systematically biased, standard errors ... wrong". Monthly returns of one stock violate independence and ordering. The actual remedy (time-ordered splits, no shuffle) is taught elsewhere, not in L3.

### Problem 2 (training on your own output / VaR pipeline)
- **2.1–2.2 ddof = 1 / "unbiased" variance.** L3 p.14, p.24, p.30: `s^2 = Σ e^2/(n − p)`, where p is the number of estimated mean parameters. For a plain sample variance p = 1, so the n − 1 denominator (`ddof=1`) is the course's unbiased estimator. p.32 states the unbiasedness concept (E[b] = β). [my note] Unbiasedness of s^2 does not make log s^2 or s unbiased. That is the core of 2.3; the slides don't say it.
- **2.3 work on the log scale.** L3 p.49: "Whenever y changes on a percentage scale, use log(y)". The model `y = a e^{xβ}` is multiplicative. The nightly refit multiplies σ̂^2 by a random factor (χ^2_{n−1}/(n−1)), so log σ̂^2 is the natural scale for the path plot, the average nightly change and the histogram. p.48 "Think about scale" supports a histogram of log s^2 rather than s^2 (the raw scale is dominated by a few extreme desks, which is the "top-10 desks" point).
- **2.4(b) real dj30 returns vs the normal model.**
  - L3 p.2/p.20 assume **normal** errors, and p.3 says that if the assumptions fail, prediction is biased and intervals are wrong. That supports why normal VaR = 2.326σ can misstate the empirical 1% quantile under fat tails.
  - p.14: "One big outlier can make s overestimate σ", i.e., the variance estimate is sensitive to extreme observations.
  - p.16: the studentized-residual cutoff reflects heavier-than-normal t tails.
  - p.37 prediction intervals include parameter uncertainty (s_fit^2), whereas the pipeline's VaR is a plug-in σ̂. That can be a discussion point.
  - Excess kurtosis and VaR themselves are not defined in L3.

### Problem 3 (60-pt research project: cross-country return prediction)
- **3-models (OLS baseline).**
  - The MLR model, OLS estimator `b = (X'X)^{-1}X'Y` (p.25), and `smf.ols` / `sm.add_constant` idioms (p.26, p.38) are the course basis for a pooled OLS forecast of next-month (vol-scaled) excess return on lagged x1..x5.
  - p.26: prediction frames must have the same variable names and factor levels. Relevant when predicting each OOS month with class dummies (every class level must be present or declared).
  - The numpy form (p.38: `np.linalg.inv(X.T@X)`) supports a fast hand-rolled expanding-window refit. [my note] `np.linalg.lstsq` is numerically safer when predictors are near-collinear.
- **3-features (duplicates in `macro_country_extended.csv`, collinear macro).**
  - L3 p.39–44 (multicollinearity) and p.40 (exact linear dependence) support finding and dropping duplicated columns before concatenating.
  - [my note] The Welch–Goyal full regression on p.28 already shows the silent failure mode: Df Model 14 < 16 regressors because d_e = d_p − e_p and tms = lty − tbl, and statsmodels quietly uses a pseudo-inverse. Check the rank (`np.linalg.matrix_rank`) or the pairwise |corr| = 1 before fitting.
  - p.44: coefficients of collinear predictors are unstable, so "which predictors carry the signal and whether that is stable across windows" must be interpreted with care. Use the partial F-test idea (p.44) or joint tests (p.42) to judge a **block** of macro variables rather than individual t-stats.
- **3-features (class dummies, per-class vs pooled, interactions).**
  - p.45/p.51–52: asset-class (A/B/C/D) or country effects as R − 1 dummies with a reference level (`C(asset_class)` or a string column in the formula).
  - p.53–54: characteristic × macro-state or characteristic × class interactions via `*`. `y ~ x1 * asset_class` gives a separate intercept and slope per class, which is the pooled-regression equivalent of per-class models (p.54, "Why not separately estimate three models?"). p.50 (common slope, class intercepts) and p.55 (a hierarchy from additive to fully interacted) give the menu between "pooled" and "per-class".
- **3-evaluation / diagnostics.**
  - p.2/p.20 constant-variance assumption and p.58 fit plot (checks non-constant variance). The exam notes that monthly volatility ranges from <1% to >17% across assets and differs by class. The vol-scaled target `r/σ̂_{t−1}` is the remedy that makes pooled errors closer to identically distributed, and L3 is the justification for why unequal variances matter (p.3: SEs and intervals wrong; pooled fits dominated by high-variance assets).
  - p.8/p.18: residual-vs-fitted plots as a diagnostic of the fitted forecast model.
  - p.36: the in-sample R^2 definition. [my note] Contrast it with R^2_OOS, whose benchmark (trailing mean / zero) is **not** in L3 and comes from later lectures and HW5. The p.28 example (in-sample R^2 0.419 but adjusted 0.281 with 16 predictors and 74 observations) is a useful cautionary citation that in-sample fit overstates forecastability.
- **3-features (outliers and transforms).**
  - p.17: outlier policy. Do not drop extreme returns without a good reason; if you winsorize or trim characteristics or returns, run with and without and document it.
  - p.48–49: log-transform macro variables that change on percentage scales (levels such as prices or GDP-type series).
- **3-interpretation (confounding).** p.57: an apparent macro effect can be confounded with class or country composition, e.g., a macro series correlated with which assets dominate the pooled sample. Relevant to "does macro add anything" and to interpreting the placebo test.
- **3-portfolio.** Not covered by L3.

---

## 9. Verification log
- All 58 pages were rendered and read, in 2×2 montages at 150 dpi. Tables and code on pages 27, 28, 34, 38, 41, 42, 43, 51, 52 and 54 were re-rendered at 250 dpi.
- Anscombe set 1 studentized residuals (p.15) reproduced exactly with statsmodels. Set 3's outlier r_i ≈ 1203.5 (p.16). b0 = 3.0001, b1 = 0.5001, R^2 = 0.6665 (p.7).
- The slide's literal s_{−i} formula (p.15) gives r_3 = −1.96 vs the printed −2.081. The printed value uses the exact deletion variance `[(n−p)s^2 − e_i^2/(1−h_i)]/(n−p−1)`.
- Welch–Goyal rank deficiency inferred from p.27 values (d_e = d_p − e_p; tms = lty − tbl) and p.28 Df Model = 14.
- OJ interaction elasticities (p.54): −3.378, −3.321, −2.712, which round to −3.4, −3.3, −2.7.
