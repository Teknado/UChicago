# Lecture 4 — Model Evaluation and Selection (BUSN 41210, Dacheng Xiu, Sep 14 2026)

Source: `/home/user/UChicago/Lecture_4.pdf` (50 slides). Every slide was viewed as a rendered image (160 dpi) as well as
read from the extracted text. Citations use the form "L4 p.X", where X is both the PDF page and the slide number.

**Scope in one sentence.** The lecture covers *linear-regression* model evaluation and selection: the overall F-test,
the partial F-test (nested models) and its equivalence to the t-test, why adjusted R^2 fails, multiple testing (a live
noise-predictor experiment plus Bonferroni), information criteria (AIC, BIC, BIC model probabilities), subset selection,
correlation screening, forward stepwise selection with AIC, the train/validation/test design, and **in-sample vs.
out-of-sample R^2**, including the course's benchmark convention for R^2_OOS. The code is statsmodels formula-API OLS.
**The lecture never mentions cross-validation (K-fold or otherwise), regularisation (ridge/lasso), trees,
classification, or expanding/rolling windows.**

---

## 0. Outline (L4 p.2) and the evaluation-vs-selection distinction (L4 p.3)

Outline (p.2): 1. F-test/Partial F-test; 2. Principles of Model Selection; 3. Information Criteria (AIC; BIC and
Bayesian Model Selection); 4. Screening; 5. Stepwise Strategies; 6. Training, Validation, and Test; 7. In-sample (IS)
vs Out-of-sample (OOS) R^2s.

Table on p.3 (transcribed from the rendered slide):

| Aspect | Model Evaluation | Model Selection |
|---|---|---|
| Motivation | Assess the performance of a trained model on unseen data | Choose the best model or configuration from a set of candidate models |
| Goal | Measure generalization performance and compute evaluation metrics | Select the model that performs best on future data |
| Process | Use validation or test sets to evaluate model accuracy, error rates, etc. | Compare different models or hyper-parameter settings based on evaluation metrics |
| Outcome | Understand how well the model might perform in real-world situations | Identify the best-performing model for deployment |

The course keeps two jobs separate: *selecting* a model or hyper-parameter, and *evaluating* the chosen model. That
separation comes back on p.43, where the validation set is called a "gray zone".

---

## 1. F-test for overall significance (L4 p.4–10)

### p.4 Model and hypotheses
- MLR model: $Y = \beta_0 + \beta_1 X_1 + \dots + \beta_d X_d + \varepsilon,\ \varepsilon \sim N(0,\sigma^2)$.
- The big-picture question: "Is this regression worthwhile? Does it help us predict Y?"
- $H_0: \beta_1 = \beta_2 = \dots = \beta_d = 0$ vs. $H_1$: at least one $\beta_j \neq 0$.
- "Assess if the overall model is statistically significant in explaining the data."

### p.5 F statistic (from the regression ANOVA)
$$ f = \frac{SSR/(p-1)}{SSE/(n-p)} = \frac{R^2/(p-1)}{(1-R^2)/(n-p)} $$
- Here $p = d+1$ is the number of parameters including the intercept (p.29 uses the same convention), SSR is the
  *regression* (explained) sum of squares, and SSE is the residual sum of squares.
- A big f means the regression is "working": big SSR relative to SSE, and R^2 close to one.
- Under $H_0$, $f \sim F_{p-1,\,n-p}$.
- The slide's rule of thumb: "Generally, f > 4 is very significant (reject the null)."

### p.6 The F distribution
- Plot of an F density with (4, 50) d.f. It is a right-skewed, positive-valued family indexed by two df parameters.
- p-value: $\varphi = \Pr(F_{p-1,n-p} > f)$.

### p.7 Market Equity Premium example: full model
```python
rest_full = F1.columns[0:-1].tolist()          # all columns except the last ('ret')
rest_full = '+'.join(rest_full)
reg_full = smf.ols(formula='ret ~ {}'.format(rest_full), data = F1).fit()
reg_full.summary()
```
Summary output (rendered slide): Dep. Variable `ret`; R-squared 0.419; Adj. R-squared 0.281; F-statistic 3.040;
Prob (F-statistic) 0.00142; Log-Likelihood 45.512; No. Observations 74; Df Residuals 59; **Df Model 14**; AIC −61.02;
BIC −26.46; covariance type nonrobust.

Coefficients (coef, se, t, p):

| var | coef | se | t | p |
|---|---|---|---|---|
| Intercept | 1.1315 | 0.456 | 2.481 | 0.016 |
| dfy | 9.4915 | 7.093 | 1.338 | 0.186 |
| infl | −2.0196 | 1.094 | −1.846 | 0.070 |
| svar | 1.5749 | 0.960 | 1.640 | 0.106 |
| d_e | −0.0248 | 0.069 | −0.360 | 0.720 |
| lty | −0.8330 | 0.719 | −1.158 | 0.251 |
| tms | −0.1880 | 1.067 | −0.176 | 0.861 |
| tbl | −0.6451 | 0.792 | −0.815 | 0.418 |
| dfr | 0.3791 | 0.562 | 0.675 | 0.502 |
| d_p | 0.1297 | 0.119 | 1.088 | 0.281 |
| d_y | −0.0127 | 0.150 | −0.084 | 0.933 |
| ltr | 0.1221 | 0.227 | 0.538 | 0.592 |
| e_p | 0.1545 | 0.086 | 1.800 | 0.077 |
| b_m | −0.0574 | 0.229 | −0.250 | 0.803 |
| ik | −2.8554 | 9.049 | −0.316 | 0.753 |
| ntis | 2.1022 | 1.319 | 1.594 | 0.116 |
| eqis | −0.9558 | 0.337 | −2.833 | 0.006 |

- The slide says "F value of 3.040 is statistically significant (p-value = 0.00142)." Only one individual t-stat
  (eqis) is significant at 5%, yet the joint F-test rejects.
- **Dataset:** "Market Equity Premium", DataFrame `F1`. It is annual (n = 74) with Welch–Goyal-style predictors: dfy,
  infl, svar, d_e, lty, tms, tbl, dfr, d_p, d_y, ltr, e_p, b_m, ik, ntis, eqis (16 covariates). The target `ret` is the
  last column.
- **My inference, not stated on the slide:** Df Model is 14 although there are 16 covariates. The likely reason is
  exact linear dependencies (d_e = d_p − e_p and tms = lty − tbl by construction), so statsmodels reports the rank.
  This matters for the p.14 discrepancy below.

### p.8 Reduced model (7 covariates)
```python
reg_reduced = smf.ols(formula='ret ~ infl + ik + eqis + d_p + ntis + e_p + svar', data=F1).fit()
reg_reduced.summary()
```
Output: R^2 0.387; Adj. R^2 0.322; F 5.942; Prob(F) 2.22e−05; Log-Lik 43.498; n = 74; Df Resid 66; Df Model 7;
AIC −71.00; BIC −52.56.
Coefficients: Intercept 1.2639 (t 5.152); infl −2.1795 (t −2.799, p 0.007); ik −10.5915 (t −1.620, p 0.110);
eqis −0.9050 (t −3.141, p 0.003); d_p 0.0843 (p 0.305); ntis 1.2801 (p 0.235); e_p 0.1411 (p 0.099);
svar 2.2006 (t 2.773, p 0.007).

### p.9 R^2 always increases (overfitting)
- $R^2_{full}=0.419$ (16 covariates) vs. $R^2_{base}=0.387$ (7 covariates). "Is this difference worth 9 extra covariates?"
- "The R^2 will **always** increase as more variables are added. If you have more b's to tune, you can get a smaller
  SSE. Least squares fit 'noise' in the data. This is known as **overfitting**."
- "More parameters will always result in a better fit to the sample data, but will not necessarily lead to better
  predictions."

### p.10 Adjusted R^2
$$ R^2_a = 1 - s^2/s_y^2 $$
- $s^2/s_y^2$ is a ratio of variance estimates (residual variance over the variance of y), so $R^2_a$ does not
  necessarily increase when variables are added.
- "Unfortunately, the adjusted r-square is **practically useless**!" The stated problem: "There is no theory for
  inference about $R^2_a$, so we will not be able to tell 'how big is big.'"
- p.28 shows empirically that $R^2_a$ does not protect against a specification search.

---

## 2. Partial F-test (nested models) (L4 p.11–15)

### p.11 Purpose
- The previous question was "Is this regression worthwhile?" The question now is "Is it useful to add these extra
  covariates to the regression?"
- The partial F-test compares two **nested** models and decides whether the additional predictors or complexity in
  the larger model are justified.
- "You **always** want to use the simplest model possible. Only add covariates if they are truly informative or
  compulsory." (This is a course principle: parsimony.)

### p.12 Setup
$$ Y = \beta_0 + \beta_1X_1 + \dots + \beta_{d_{base}}X_{d_{base}} + \beta_{d_{base}+1}X_{d_{base}+1} + \dots + \beta_{d_{full}}X_{d_{full}} + \varepsilon $$
- $d_{base}$ is the number of covariates in the base (small) model and $d_{full} > d_{base}$ is the number in the full model.
- $H_0: \beta_{d_{base}+1} = \dots = \beta_{d_{full}} = 0$; $H_1$: at least one $\beta_j \neq 0$ for $j > d_{base}$.

### p.13 Statistic (under $H_0$, i.e. the base model is true)
$$ f = \frac{(R^2_{full}-R^2_{base})/(d_{full}-d_{base})}{(1-R^2_{full})/(n-d_{full}-1)}
     = \frac{(SSE_{full}-SSE_{base})/(p_{full}-p_{base})}{SSE_{full}/(n-p_{full})} \sim F_{p_{full}-p_{base},\ n-p_{full}} $$
- $1-R^2_{full}$ is the full model's $SSE/SST$. Degrees of freedom: $d_{full}-d_{base}$ and $n-d_{full}-1$.
- A big f means $R^2_{full}-R^2_{base}$ is statistically significant, i.e. at least one added X is useful.
- **Sign error on the slide:** the SSE form is written as $(SSE_{full}-SSE_{base})$. The correct numerator, which is
  positive, is $(SSE_{base}-SSE_{full})$. p.16 has the same typo. Use $SSE_{base}-SSE_{full}$ in code.
- Notation: d counts covariates and p = d + 1 counts parameters, so $n-p_{full} = n-d_{full}-1$.

### p.14 Partial F-test in Python (by hand)
- Hypotheses: $H_0: \beta_j = 0$ for all $j \in \{dfy, d\_e, lty, tms, tbl, dfr, d\_y, ltr, b\_m\}$ (the 9 dropped vars).
- The displayed formula evaluates to "= 0.36". The code below prints `F_stat: 0.47`.
```python
n = reg_full.nobs
R2_full = reg_full.rsquared
R2_base = reg_reduced.rsquared
d_full = reg_full.df_model
d_base = reg_reduced.df_model
F_statistic = ((R2_full - R2_base)/(d_full - d_base)) / ((1-R2_full)/(n - d_full - 1))
# F_stat: 0.47
df1 = d_full - d_base
df2 = n - d_full - 1
p_value = 1 - f.cdf(F_statistic, df1, df2)       # from scipy.stats import f
# pvalue: 0.85
```
- Conclusion on the slide: "A p-value of 0.85 is not at all significant, so we stick with the null hypothesis and
  assume the base (7 covariates) model."
- **Why the slide shows 0.36 and the code shows 0.47 (checked by recomputing):** 0.36 comes from $d_{full}=16$
  (9 extra, df2 = 57), which gives f ≈ 0.35. The code uses `df_model` = 14 (rank), so df1 = 7 and df2 = 59, giving
  f = 0.464 ≈ 0.47 with p = 0.85. **Lesson: take degrees of freedom from the fitted model's rank (`df_model`,
  `df_resid`), not from the number of columns. Collinear columns do not add parameters.**

### p.15 Partial F via ANOVA (the house idiom)
```python
anova_results = anova_lm(reg_reduced, reg_full)   # from statsmodels.stats.anova import anova_lm; small model first
anova_results
```
Output:

| df_resid | ssr | df_diff | ss_diff | F | Pr(>F) |
|---|---|---|---|---|---|
| 66.0 | 1.337192 | 0.0 | NaN | NaN | NaN |
| 59.0 | 1.266348 | 7.0 | 0.070844 | 0.471527 | 0.851311 |

- **Naming trap:** in statsmodels, `ssr` means the *sum of squared residuals*, which the slides call SSE. On p.5 the
  slides use SSR for the *regression* sum of squares. Check by hand: (1.337192 − 1.266348)/7 ÷ (1.266348/59) = 0.4715.

---

## 3. F-test vs t-test (L4 p.16–21)

### p.16 Same hypotheses
- You have d covariates and are considering adding $X_{d+1}$.
- t-test: $H_0:\beta_{d+1}=0$ vs $H_1:\beta_{d+1}\neq 0$. Statistic $z = b_{d+1}/s_{b_{d+1}}$, p-value
  $P(Z_{n-d-2} > |z|)$. Here $Z_{n-d-2}$ denotes a t with n − d − 2 df, and the slide's notation stands for the
  two-sided probability $P(|T_{n-d-2}|>|z|)$.
- Partial F: same hypotheses, $f = (SSE_{full}-SSE_{base})/MSE_{full}$ (same sign typo as p.13; read it as
  base − full), p-value $P(F_{1,n-d-2} > f)$.
- "The test hypotheses are exactly the same!"

### p.17 Example: ik only vs ik + ltr
- 2-covariate regression: Intercept 0.8713 (t 3.987); ik −23.7512 (se 6.057, t −3.921, p 0.000); ltr 0.1158 (se 0.172,
  t 0.675, p 0.502).
- The t-test asks whether $b_{ltr}$ is far enough from zero to be significant. The F-test asks whether the increase in
  R^2 is significantly big.
```python
anova_lm(reg_simple, reg_2cov)   # ltr
```
Output: df_resid 72 → 71; ssr 1.785140 → 1.773761; df_diff 1; ss_diff 0.011379; F 0.455463; Pr(>F) 0.501944.

### p.18 Equivalence
- $f = z^2$ and $P(Z_{n-d-2}>|z|) = P(F_{1,n-d-2} > z^2)$: "the f stat for one extra variable is just a squared t stat."
  Check: 0.675^2 = 0.4556 ≈ 0.4555, and the p-values match (0.502).
- Figures: simulated histograms of a t with 50 df and of $Z^2 \sim F_{1,50}$.

### p.19 Why not just use individual t-stats? Multicollinearity
- "If the X's are highly correlated with each other, then $s_{b_j}$'s will be very big (since you don't know which
  $X_j$ to regress onto), and you will fail to reject $\beta_j = 0$ for all of the $X_j$'s even if they **do** have a
  strong effect on Y."
- Question posed: does F significance imply at least one significant t-test, and vice versa?

### p.20 F significant, no t significant
```python
reg_temp = smf.ols(formula='ret ~ infl + lty + tbl + svar', data=F1).fit()
reg_temp.summary()
```
- R^2 0.177; Adj 0.130; **F 3.718, Prob(F) 0.00847**; LL 32.636; AIC −55.27; BIC −43.75; Df Model 4; Df Resid 69.
- No individual t is significant: infl t −0.122 (p 0.903), lty t 0.140 (p 0.889), tbl t −1.465 (p 0.147), svar t 0.995
  (p 0.323); Intercept t 1.923 (p 0.059).
- Answer: **F significance does NOT imply that any single t is significant.** Multicollinearity is the reason
  (lty and tbl move together).

### p.21 t significant, F not significant
```python
reg_temp = smf.ols(formula='ret ~ svar + d_y', data=F1).fit()
```
- R^2 0.069; Adj 0.043; **F 2.624, Prob(F) 0.0795** (not significant at 5%); LL 28.053; AIC −50.11; BIC −43.19.
- svar t 2.075, **p 0.042** (significant); d_y t 1.623 (p 0.109).
- Answer: **one significant t does not imply a significant joint F.**

---

## 4. Variable selection principles (L4 p.22–24)

- p.22: Observe Y and $(X_1,\dots,X_d)$. "Assuming the model is always linear, so model selection simplifies to
  variable selection." The goal is to find the "best" subset of X's for predicting Y. This is particularly of interest
  when d is large and the X's contain many redundant or irrelevant variables.
- p.23: **Ockham's Razor: "Plurality ought not be posited without necessity."** "Brevity is the soul of wit."
  "**Overly complicated models lead to bad forecasts.**"
- p.24: We need a trade-off between data fit and simplicity. One method we already know is the partial F-test:
  1. regress on the covariates you think should be in the model;
  2. kick out the variables that don't seem significant;
  3. use a partial F-test to see whether the simple model is good enough.
  "This only works for a small number of variables, which you've chosen intelligently."

---

## 5. Multiple testing and the noise-predictor experiment (L4 p.25–28)

### p.25 Arithmetic of false discoveries
- A big problem with using tests (t or F) to compare models is the **false discovery rate** from **multiple testing**.
  If you do 20 tests of a true $H_0$ at $\alpha = .05$, you expect one false rejection.
- Example: 100 predictors, 10% of which are truly influential, and all 10 are found significant. The test also rejects
  for 5% of the 90 useless ones, i.e. 4.5 of them. So $4.5/(4.5+10) \approx 1/3$ of the significant $b_j$'s are false
  positives.
- "In modern big data problems, <1% of variables are influential."

### p.26 The experiment ("note how honest it is")
- Take the real market equity premium series, n = 74.
- Generate **100 predictors from a random number generator**: standard normal, independent of each other and of
  everything else.
- By construction none of them is related to the market. "There is no signal to find. We know this because we made
  them up."
- Regress the market on each one in turn and collect the 100 t-statistics.
- "How many of the 100 come back significant at the 5% level? And how big will the best t-statistic be? **Say a
  number before I show you.**" The professor commits to a prediction before computing, which is the same
  pedagogical device as exam 2.1.

### p.27 Results: "Ten of Them. And the Best Has p = 0.005."
- Left panel: bar chart of the 100 t-stats with ±1.96 lines; 10 bars exceed them (in red).
- Right panel: histogram of the **max |t| of 100 under the null** across repetitions, with markers at 1.96, "our
  winner" at 2.88, and "Bonferroni 3.65".
- 10 of the 100 are significant at 5%, "twice what the arithmetic predicted, which is itself just luck."
- The winner has t = −2.88, p = 0.0052. "Written up on its own, with a story attached, that is a publishable-looking
  finding about a variable that *is definitionally noise*."
- Over **4,000 repetitions** of the whole exercise, P(at least one of 100 noise variables looks significant) =
  **99.6%**. For comparison, the theoretical $1-0.95^{100} = 99.4\%$.
- "You will essentially always find something. The question was never whether you would."

### p.28 "And Adjusted R^2 Does Not Save You"
Keep the best k of the 100 noise columns and fit them together:

| best k noise columns | 1 | 2 | 5 | 10 | 20 |
|---|---|---|---|---|---|
| in-sample R^2 | 0.10 | 0.16 | 0.34 | 0.40 | 0.54 |
| adjusted R^2 | +0.09 | +0.13 | +0.29 | **+0.31** | +0.37 |

- "An adjusted R^2 of 0.31 on a model containing no information whatsoever. The penalty for extra parameters is real,
  but it was never designed to protect you from having *searched*."
- "**The fix is not a better statistic. It is a different critical value.**" Under the null, the largest of 100
  t-stats has a 95th percentile of **3.65**, not 1.96. Bonferroni gives 3.65 for 100 tests, the same number (it is
  $t_{72}$ at $1-0.05/(2\cdot100)$, which I checked equals 3.646). The winning t of 2.88 does not clear it.
- Bonferroni rule (implied): with m tests, use level $\alpha/m$, so the two-sided critical value is
  $t_{df}^{-1}(1-\alpha/(2m))$.
- "**Report the size of the search next to the result.** And then, for the rest of this course, **stop asking
  whether it fits and start asking whether it predicts.**" This is the thesis statement of the course's move to
  out-of-sample evaluation.

---

## 6. Information criteria (L4 p.29–33)

### p.29 BIC
- Information criteria "attempt to quantify how well our model **would** have predicted the data (regardless of
  what you've estimated for the $\beta_j$'s)." They are presented as an alternative to testing.
- "The best of these is the BIC: Bayes Information Criterion," based on a Bayesian philosophy.
$$ BIC = n\log(s^2) + p\log(n),\qquad p = d+1 = \text{'degrees of freedom' used in the fit} $$
- Choose the model with the **minimum** BIC.

### p.30 AIC
$$ AIC = n\log(s^2) + 2p $$
- General form: $n\log(s^2) + kp$, with k = 2 for AIC and k = log(n) for BIC.
```python
reg_full = smf.ols(formula='ret ~ {}'.format(rest_full), data = F1).fit()
full_AIC = reg_full.aic       # -61.02
full_BIC = reg_full.bic       # -26.46
```
- "AIC prefers more complicated models than BIC, and it is not as easily interpretable."
- Implementation note (my check): statsmodels computes $-2\log L + kp$ with p = df_model + 1. For example,
  −2(45.512) + 2(15) = −61.02 and −2(45.512) + 15 ln 74 = −26.46. This equals the slide formula up to a constant that
  depends only on n, so comparisons across models are valid **only on the same sample (same n)**.

### p.31 BIC model probabilities
$$ P(M_i) \approx \frac{e^{-\frac12 BIC(M_i)}}{\sum_{r=1}^R e^{-\frac12 BIC(M_r)}}
 = \frac{e^{-\frac12 [BIC(M_i)-BIC_{min}]}}{\sum_{r=1}^R e^{-\frac12 [BIC(M_r)-BIC_{min}]}} $$
- Subtract $BIC_{min}$ for numerical stability.

### p.32 Market equity premium comparison

| | full | reduced |
|---|---|---|
| AIC | −61.02 | −71.00 |
| BIC | −26.46 | −52.56 |

- "BIC and AIC both agree with our F-testing selection (reduced)." BIC indicates we are practically 100% sure the
  reduced model is better.
```python
prob = np.exp(-0.5*(BIC-min(BIC))).round(4)
print(prob/prob.sum())         # [0. 1.]
```
(ΔBIC = 26.1, so the relative weight is exp(−13.05) ≈ 2e−6.)

### p.33 Why use BIC
- It is an alternative to testing: easy to calculate, gives model probabilities, has no "multiple testing" type
  worries, and generally leads to simpler models than F-tests.
- "As with testing, you need to narrow down your options before comparing models. What if there are too many
  possibilities?" (This leads into subset selection.)

---

## 7. Subset selection, the macro dataset, and screening (L4 p.34–39)

### p.34 Subset selection
- General idea: search across all permutations of models and choose the best by some criterion.
- Challenges: the set of models may be large. You need the best model of size k for k = 1..p, and the number of
  combinations is $\sum_{k\le p}\binom{p}{k} = 2^p$.
- Best-subset search is described only conceptually. No algorithm or code is given.

### p.35 Macro Forecasting dataset
- "Forecast monthly growth rate of US industrial production using **119 macro variables**." The data run from
  February 1960 to December 2021 (**743 monthly observations**). DataFrame `df`: `growth` is the first column, then
  FRED-MD-style mnemonics (RPI, W875RX1, DPCERA3M086SBEA, CMRMTSPLx, RETAILx, IPFPNSS, IPFINAL, IPCONGD, IPDCONGD, …,
  PCEPI), all in growth-rate or transformed units.
- The slide does not show how the predictors are timed relative to `growth` (contemporaneous or lagged). The lecture
  never discusses lag alignment.

### p.36 Full macro regression
```python
rest_full = df.columns[1:].tolist()      # growth is column 0 here
rest_full = '+'.join(rest_full)
full_model = smf.ols(formula='growth ~ {}'.format(rest_full), data = df).fit()
full_model.summary()
```
Output: R^2 0.624; Adj R^2 0.553; F 8.699; Prob(F) 3.55e−76; LL 2750.5; n 743; Df Resid 623; Df Model 119;
AIC −5261; BIC −4708. Coefficients shown (partial list): Intercept 0.1423 (t 2.601); RPI −0.1838 (t −2.753);
W875RX1 0.1266 (t 2.164); DPCERA3M086SBEA 0.1610 (t 2.102); CMRMTSPLx 0.0296; RETAILx −0.0678; IPFPNSS 0.4865;
IPFINAL −0.3209; IPCONGD 0.2672; IPDCONGD −0.0450; IPNCONGD −0.2912.

### p.37–38 Correlation screening
- p.37: histogram of the 119 correlations between each predictor and growth. Most lie between −0.3 and +0.35, with
  one outlier near −0.6.
- "By only considering predictors with abs correlations greater than a certain threshold, we can reduce the total
  number of covariates."
- p.38: horizontal bar chart titled "correlation of growth" listing the surviving predictors. From the plot the
  threshold appears to be |corr| ≈ 0.2, and exactly 33 variables survive, matching "screening #variables 33" on p.49.
  - Positive: HWIURATIO, HWI, IPDMAT, DPCERA3M086SBEA, IPMANSICS, CUMFNS, IPFPNSS, IPDCONGD, RETAILx, USGOOD, IPFINAL,
    IPBUSEQ, MANEMP, CMRMTSPLx, TB6MS, DMANEMP, GS1, IPMAT, IPCONGD, NDMANEMP, TB3MS, W875RX1, T1YFFM, AMDMNOx, S_P_500,
    PERMITMW, TB3SMFFM, TB6SMFFM.
  - Negative: UEMPLT5, M2SL, ISRATIOx, BUSLOANS, CLAIMSx (≈ −0.6).
- No code for screening is shown (the idiom is presumably `df.corr()['growth']` plus a threshold filter).

### p.39 Problems with screening
- "Maybe we grab models with covariates selected based on different correlation thresholds and compare?"
- Problems with screening:
  - it may keep redundant variables and omit variables that are important only when combined;
  - in post-screening regression, multicollinearity may occur and cause problems;
  - "Bringing together variables that are useful individually does not mean that they will be useful collectively."

---

## 8. Forward stepwise regression (L4 p.40–41)

### p.40 Algorithm (with AIC)
- Forward stepwise starts from a simple "null" model (intercept only) and incrementally updates the fit to allow
  slightly more complexity. The `forward_selected()` function (a helper; its code is not shown on the slides) runs:
  1. Fit all univariate models. Choose the one with the lowest AIC and put that variable, $x_{(1)}$, in the model.
  2. Fit all bivariate models that include $x_{(1)}$ ($y \sim \beta_{(1)}x_{(1)} + \beta_j x_j$) and add the $x_j$ from
     the one with the lowest AIC.
  3. Repeat: minimise AIC by adding one variable at a time.
  - **Stopping rule:** stop when the AIC of the current model is lower than that of every model that adds one variable.

### p.41 Pros and cons
- Forward is better than backward methods:
  - the "full" model can be expensive or tough to fit, while the null model is usually available in closed form;
  - jitter the data and the full model can change dramatically (because it overfits), while the null model is always
    the same.
- Stepwise approaches are "**myopic**": they find the best solution at each step without considering the overall
  global path.
- "A related subtle (but massively important) issue is **stability**." Stepwise has high **sampling variability**:
  the selected variables change a lot from one dataset to another.
- Backward elimination is mentioned only as a comparison. No algorithm is given.

---

## 9. Out-of-sample prediction: training / validation / test (L4 p.42–43)

### p.42
- "How do we evaluate a prediction model? **Make predictions!**" Use the model to predict outcomes for observations
  we have not seen before: use the data to create a prediction problem and see how the candidate models perform.
- "We'll split the entire sample into training, validation, and test subsamples."

### p.43 The three phases (a core course convention)
1. **Training phase (model building):** the model learns from the training set by adjusting its parameters (e.g.
   regression coefficients).
2. **Validation phase (model selection):** during training, different models or hyper-parameters are evaluated on
   the validation set to determine the best configuration.
3. **Testing phase (model evaluation):** after model selection, the test set is used to evaluate the final model,
   which gives a realistic assessment of performance on unseen data. "**The test set remains untouched until this
   stage to avoid biased estimates of model performance.**"
- "1: In-sample; 3: Out-of-sample; 2: gray zone. 'In-' for model evaluation purpose, 'Out-of-' for model selection."
  So the validation score of the *selected* configuration is optimistically biased as a performance estimate, because
  it was used to choose. An honest performance number needs untouched data.

---

## 10. In-sample vs out-of-sample R^2 (L4 p.44–48) — the key convention

### p.44 IS R^2
- Data $[X_1,Y_1],\dots,[X_n,Y_n]$ are used to fit $\hat\beta$.
$$ SSE_{IS}(\hat\beta) = \sum_{i=1}^n (Y_i - X_i'\hat\beta)^2,\qquad SSE_{IS}(\hat Y=\bar Y)=\sum_{i=1}^n (Y_i-\bar Y)^2 $$
$$ R^2_{IS} = 1 - SSE_{IS}(\hat\beta)/SSE_{IS}(\hat Y = \bar Y) $$

### p.45 OOS numerator
- "Observations 1…n are still used to fit $\hat\beta$, but the SSE of the numerator is now calculated over **new**
  observations":
$$ SSE_{OOS}(\hat\beta) = \sum_{i\in OOS}(Y_i - X_i'\hat\beta)^2 $$

### p.46 OOS denominator: the benchmark choice
- "The denominator choice for OOS R^2 has certain freedom."
- **Common choice: $\bar Y$ from the in-sample (training) data:**
$$ SSE_{OOS}(\hat Y=\bar Y) = \sum_{i\in OOS}(Y_i - \bar Y)^2,\qquad R^2_{OOS} = 1 - SSE_{OOS}(\hat\beta)/SSE_{OOS}(\hat Y=\bar Y) $$
  - "Why not out of sample $\bar Y$?" (The slide leaves this as a rhetorical question.) The answer: the OOS mean is
    not known at forecast time. Using it is look-ahead and handicaps the benchmark with information no real
    forecaster had.
- **Or use 0**, or other estimators of β based on in-sample data:
$$ R^2_{OOS} = 1 - SSE_{OOS}(\hat\beta)/SSE_{OOS}(0),\qquad SSE_{OOS}(0) = \sum_{i\in OOS} Y_i^2 $$
- "**Ultimately, OOS R^2 is a comparison of SSEs between two predictors. It can be negative!**"
  A negative value means the model forecasts worse than the benchmark.

### p.47 Market equity premium: 80:20 split, code (house idiom)
```python
N = len(F1)
leaveout = range(4*N//5, N)                       # LAST 20% of rows = test (chronological, no shuffle)
yout = F1[['ret']].iloc[leaveout]
yin  = F1.drop(F1.index[leaveout])[['ret']]
variable = '+'.join(F1.iloc[:,:-1].columns)
ISreg = smf.ols(formula='ret ~ {}'.format(variable), data=F1.drop(F1.index[leaveout])).fit()
IS_R2 = ISreg.rsquared                             # IS_R2: 0.47554
y_pred = ISreg.predict(F1.iloc[leaveout])
TSS = sse(yout.ret, yin.ret.mean())                # benchmark = TRAINING mean
SSE = sse(yout.ret, y_pred)
OOS_R2 = 1 - SSE/TSS                               # OOS_R2: -0.11806
```
- `sse` is a helper not defined on the slides; it is presumably `def sse(y, yhat): return ((y - yhat)**2).sum()`.
- The split is **time-ordered**: the last 20% of observations are the test set. The model is fit **once** on the
  first 80% (a static scheme) and never re-estimated.
- Result: IS R^2 = 0.476 but **OOS R^2 = −0.118**. The kitchen-sink model forecasts worse than the historical mean.

### p.48 Macro forecasting: same recipe
```python
N = len(df)
leaveout = range(4*N//5, N)
yout = df[['growth']].iloc[leaveout]
yin  = df.drop(df.index[leaveout])[['growth']]
variable = '+'.join(df.iloc[:,:-1].columns)
ISreg = smf.ols(formula='growth ~ {}'.format(variable), data=df.drop(df.index[leaveout])).fit()
IS_R2 = ISreg.rsquared                              # 0.52849
y_pred = ISreg.predict(df.iloc[leaveout])
TSS = sse(yout.growth, yin.growth.mean())
SSE = sse(yout.growth, y_pred)
OOS_R2 = 1 - SSE/TSS                                # -0.50138
```
- IS R^2 0.528 vs **OOS R^2 −0.501**: 119 predictors overfit badly.
- Caution (my observation): p.36 builds the predictor list as `df.columns[1:]` (growth first) and p.48 uses
  `df.iloc[:,:-1].columns`. These give different sets unless the columns were reordered. Always build the predictor
  list explicitly and exclude the target.

---

## 11. Screening and forward selection OOS, and sample sensitivity (L4 p.49–50)

### p.49 Comparison table

| | full | screening | forward |
|---|---|---|---|
| #variables | 119.00 | 33.00 | 36.00 |
| R2 | 0.62 | 0.47 | 0.58 |
| OOS R2 | **−0.50** | **0.08** | **−0.23** |
| AIC | −5260.94 | −5177.60 | −5336.11 |
| BIC | −4707.65 | −5020.84 | −5165.51 |

- "The results appear to improve when using Correlation Screening or Forward Stepwise Selection. However, both
  methods still present certain challenges…"
- Observations:
  - AIC picks *forward* (lowest AIC −5336) and BIC also picks *forward* (−5166), yet *screening* has the best OOS R^2
    (0.08). The in-sample criteria and OOS performance disagree.
  - The R2/AIC/BIC for "full" equal the p.36 full-sample fit, so those in-sample columns use the full sample.
  - The slides do not state whether screening correlations and forward selection used only the training 80%. For
    exam work, **do all selection (screening, stepwise, thresholds) on training data only**, or the OOS R^2 is
    contaminated by look-ahead.

### p.50 Sensitivity to the sample (instability)
```python
random_start = np.random.choice(df.index[:-11])
random_year = range(random_start, random_start + 12)
df_new = df.drop(index=random_year)        # drop 12 consecutive months
```

| | full | screening | forward | screening w/ new data | forward w/ new data |
|---|---|---|---|---|---|
| #variables | 119.00 | 33.00 | 36.00 | 30.00 | 41.00 |
| R2 | 0.62 | 0.47 | 0.58 | 0.47 | 0.59 |
| AIC | −5260.94 | −5177.60 | −5336.11 | −5089.73 | −5254.95 |
| BIC | −4707.65 | −5020.84 | −5165.51 | −4947.31 | −5061.99 |

- "Our results indicate that both methods (Correlation Screening and Forward Selection) are sensitive to the sample
  data." Dropping one year of 743 months changes the screened set from 33 to 30 variables and the forward set from 36
  to 41. This demonstrates the p.41 stability warning.
- The AIC/BIC values for the "w/ new data" columns come from a different n (731), so they are not comparable to the
  original columns (my note).
- The code has no seed (`np.random.choice` without `np.random.seed`/`default_rng(seed)`), so it is not reproducible.
  Set seeds in exam code.

---

## 12. Consolidated formula sheet (Lecture 4)

| Quantity | Formula | Slide |
|---|---|---|
| Overall F | $f=\frac{SSR/(p-1)}{SSE/(n-p)}=\frac{R^2/(p-1)}{(1-R^2)/(n-p)}\sim F_{p-1,n-p}$ | p.5 |
| F p-value | $\varphi=\Pr(F_{p-1,n-p}>f)$ | p.6 |
| Adjusted R^2 | $R^2_a = 1-s^2/s_y^2$ | p.10 |
| Partial F | $\frac{(R^2_{full}-R^2_{base})/(d_{full}-d_{base})}{(1-R^2_{full})/(n-d_{full}-1)}=\frac{(SSE_{base}-SSE_{full})/(p_{full}-p_{base})}{SSE_{full}/(n-p_{full})}\sim F_{p_{full}-p_{base},n-p_{full}}$ (slide's sign typo corrected) | p.13 |
| t vs F | $z=b_{d+1}/s_{b_{d+1}}$; $f=z^2$; $P(\lvert T_{n-d-2}\rvert>\lvert z\rvert)=P(F_{1,n-d-2}>z^2)$ | p.16, 18 |
| False-discovery share | $\frac{\alpha(1-\pi)m}{\alpha(1-\pi)m + \pi m}$, e.g. $4.5/(4.5+10)\approx 1/3$ | p.25 |
| Bonferroni critical value | for m = 100 tests at 5%: 3.65 (= 95th pct of max of 100 null t's) | p.27–28 |
| BIC | $n\log(s^2)+p\log n$, p = d + 1 | p.29 |
| AIC | $n\log(s^2)+2p$; general form $n\log s^2 + kp$ | p.30 |
| BIC model prob | $P(M_i)\approx \frac{e^{-\frac12[BIC_i-BIC_{min}]}}{\sum_r e^{-\frac12[BIC_r-BIC_{min}]}}$ | p.31 |
| # subsets | $\sum_{k\le p}\binom pk = 2^p$ | p.34 |
| IS R^2 | $1-\sum_{i=1}^n(Y_i-X_i'\hat\beta)^2/\sum_{i=1}^n(Y_i-\bar Y)^2$ | p.44 |
| OOS R^2 (mean benchmark) | $1-\sum_{OOS}(Y_i-X_i'\hat\beta)^2/\sum_{OOS}(Y_i-\bar Y_{train})^2$ | p.45–46 |
| OOS R^2 (zero benchmark) | $1-\sum_{OOS}(Y_i-X_i'\hat\beta)^2/\sum_{OOS}Y_i^2$ | p.46 |

## 13. Code idioms (house style)

- `import statsmodels.formula.api as smf`; `smf.ols(formula='y ~ {}'.format('+'.join(cols)), data=df).fit()`
  (p.7, 8, 20, 21, 36, 47, 48).
- Fitted-result attributes: `.summary()`, `.rsquared`, `.nobs`, `.df_model`, `.aic`, `.bic`, `.predict(newdata)`
  (p.7, 14, 30, 47).
- `from scipy.stats import f`; `p_value = 1 - f.cdf(F_statistic, df1, df2)` (p.14).
- `from statsmodels.stats.anova import anova_lm`; `anova_lm(small_model, big_model)` gives the partial F-test
  (p.15, 17).
- BIC weights: `prob = np.exp(-0.5*(BIC-min(BIC))).round(4); prob/prob.sum()` (p.32).
- Chronological holdout: `leaveout = range(4*N//5, N)`; test = `.iloc[leaveout]`; train = `.drop(df.index[leaveout])`
  (p.47–48).
- OOS R^2: `1 - sse(y_test, y_pred)/sse(y_test, y_train.mean())` (p.47–48).
- `forward_selected()`: a user-defined AIC forward-stepwise helper (p.40); its code is not shown.
- Sample perturbation: `np.random.choice(df.index[:-11])`, then drop 12 consecutive rows (p.50).
- No sklearn appears anywhere in this lecture.

## 14. Warnings and pitfalls emphasised

1. R^2 always rises with more variables, and least squares fits noise. This is overfitting. Better fit does not mean
   better prediction (p.9).
2. Adjusted R^2 is "practically useless": it has no inference theory (p.10) and does not protect against search. An
   adjusted R^2 of 0.31 was obtained from pure noise (p.28).
3. Multicollinearity inflates standard errors, so individual t-tests can all fail while F is significant (p.19–20).
   The converse also happens: a significant t with an insignificant F (p.21).
4. Multiple testing: about 1/3 of discoveries are false in the 10%-true scenario (p.25). With 100 noise predictors you
   find about 10 "significant" and a best t of 2.88 (p.27). P(≥1 false hit) = 99.6%. Use the search-adjusted critical
   value (Bonferroni/max-t 3.65) and **report the size of the search** (p.28).
5. "Stop asking whether it fits and start asking whether it predicts." (p.28)
6. Testing-based selection only works for a small number of intelligently chosen variables (p.24). Information
   criteria also require narrowing the options first (p.33).
7. Screening keeps redundant variables, drops variables that matter only jointly, and causes post-screening
   multicollinearity (p.39).
8. Stepwise is myopic and unstable (high sampling variability) (p.41). Dropping one year changes the selected sets
   (p.50).
9. The test set must stay untouched until final evaluation. Validation is a "gray zone" (p.43).
10. The OOS R^2 benchmark must use the **in-sample (training) mean**, not the OOS mean (p.46, rhetorical "Why not out
    of sample Ȳ?"). OOS R^2 can be negative (p.46–48).
11. AIC prefers more complex models than BIC. BIC generally gives simpler models than F-tests (p.30, 33).
12. Numerical: subtract BIC_min before exponentiating (p.31).
13. (Inferred from slide numbers.) Degrees of freedom must reflect rank (`df_model`), not the column count
    (p.14: 0.36 vs 0.47).
14. (Inferred.) statsmodels `ssr` = residual SS, i.e. the course's SSE (p.15 vs p.5).
15. (Inferred.) AIC/BIC are comparable only across models fit on the same observations (p.50).

## 15. Datasets and worked examples
- **Market Equity Premium** (`F1`): n = 74 annual observations, target `ret`, 16 Welch–Goyal predictors (dfy, infl,
  svar, d_e, lty, tms, tbl, dfr, d_p, d_y, ltr, e_p, b_m, ik, ntis, eqis). Used for the F-test (p.7), the reduced
  7-variable model (p.8), the partial F (p.14–15), F vs t (p.17, 20, 21), the noise experiment (p.26–28), AIC/BIC
  (p.30, 32), and the 80:20 OOS R^2 (p.47).
- **Noise-predictor Monte Carlo:** 100 iid N(0,1) predictors, univariate regressions, and 4,000 repetitions for the
  max-t null distribution (p.26–28).
- **Macro Forecasting** (`df`): US industrial production monthly growth, 119 FRED-MD-style macro predictors,
  Feb 1960 to Dec 2021, 743 observations. Used for the full regression (p.36), screening (p.37–38), forward selection
  (p.40, 49), OOS R^2 (p.48–49), and sample sensitivity (p.50).

## 16. Boundaries: what is NOT taught here, or only mentioned in passing
- **Cross-validation (K-fold, stratified, time-series CV) is not taught in L4.** Only a single train/test split
  (80:20, chronological) and the conceptual train/validation/test design appear.
- **Expanding-window and rolling-window evaluation are not in L4.** Only the static single-split scheme appears. The
  exam notebook attributes expanding windows to Lecture 5 / HW5.
- Regularisation (ridge, lasso, elastic net), PCR/PLS, trees, ensembles, classification metrics (accuracy,
  confusion), feature importance and portfolios are all absent.
- Best-subset selection: combinatorics only (2^p). No algorithm or code.
- Backward elimination: mentioned only as inferior to forward (p.41). No procedure.
- The false discovery rate is named (p.25), but there is no Benjamini–Hochberg or other FDR-control procedure.
- Bonferroni is mentioned only as a critical value (3.65 for 100 tests). There is no formal treatment.
- AIC appears only through its formula and the remark that it is "not as easily interpretable" (p.30). There is no
  derivation. BIC is called "the best of these".
- The adjusted R^2 formula is given in the compact form $1-s^2/s_y^2$ only (p.10).
- The `forward_selected()` and `sse()` helpers are used but not shown.
- The t distribution and F distribution are shown via simulated histograms and a density (p.6, 18). No derivations.
- Robust or HAC standard errors are not covered (summaries say "nonrobust").

## 17. Mapping to the final exam

**Problem 1 (trees, CV accuracy):** L4 supplies the *selection/evaluation logic*, not the tree methods (those are
from Lecture 8, per the notebook).
- 1.1 majority-class baseline: "a comparison between two predictors" (p.46). Every model's score is relative to a
  naive benchmark; the "nobody purchases" rule plays the role of $\bar Y$ / 0 in R^2_OOS.
- 1.1 `max_leaf_nodes` sweep, choosing the best by CV: this is the validation phase, i.e. model selection (p.3, p.43).
  The tie → smaller rule is Ockham's razor (p.23) and "always use the simplest model possible" (p.11). Either side of
  the optimum shows under-fitting (too few leaves) or overfitting (p.9: more parameters fit noise). The CV score of the
  selected size is "gray zone" (p.43), i.e. slightly optimistic.
- 1.3 RF/GB vs tree: model selection by comparing evaluation metrics (p.3). For a small n = 400 with 3 features, extra
  complexity need not help ("overly complicated models lead to bad forecasts", p.23).
- 1.4 impurity importance (in-sample) vs permutation importance (held-out 30%): the IS vs OOS distinction (p.44–46).
  In-sample measures reward fit to noise (p.9, p.28), so show the decision maker the held-out measure. The held-out
  set is the untouched test set (p.43).
- 1.5 time-series rows: L4's time-series examples always hold out the **last** 20% chronologically
  (`range(4*N//5, N)`, p.47–48), and the benchmark must use only in-sample data ("Why not out of sample Ȳ?", p.46).
  Shuffled CV lets the model train on future months, which is look-ahead and biases accuracy **upward** (optimistic).
  The time-ordered split is the course-consistent fix (p.43: the test set is untouched and comes after training).

**Problem 2 (VaR self-training pipeline):** weak direct overlap.
- 2.1 "commit-before-compute" mirrors the p.26 device: "Say a number before I show you."
- 2.2–2.3 1,000-desk simulation: analogous to the p.27 Monte Carlo (4,000 repetitions to get the distribution of a
  statistic under a known truth). The honest design uses a known truth (σ = 1), the same way the noise predictors were
  "made up".
- The sampling variability / stability theme (p.41, 50): an estimate refit on its own noisy output inherits sampling
  noise.
- Unbiased variance estimates: $s^2$ appears as a variance estimate in $R^2_a$ and AIC/BIC (p.10, 29–30), but L4 does
  not discuss ddof or martingales. These come from other lectures.

**Problem 3 (research project):** strong overlap.
- **3-evaluation, R^2_OOS definitions:** p.44–46 give the exact course formulas. Use the benchmark = in-sample mean
  (the trailing mean in an expanding harness, which the notebook attributes to L5/HW5) **and** the benchmark = 0.
  "It can be negative!" A negative result is legitimate and matches the notebook's rule that a negative result
  honestly established earns full credit. Report IS and OOS R^2 side by side to show overfitting (p.47–48 contrast).
- **3-evaluation, split design:** train/validation/test (p.43). Tune hyper-parameters and choose features on
  training/validation only, and keep the OOS window untouched. Fix the OOS window in advance (notebook rule 2, and
  p.43 "remains untouched until this stage"). A chronological holdout without shuffling (p.47–48).
- **3-features, extended x17..x164 (uncurated, 148 columns):** multiple testing (p.25–28). Searching over roughly 150
  macro series will "always find something". Report the size of the search next to results (p.28). Adjusted R^2 won't
  protect you (p.28). Screening (p.37–39) and forward stepwise (p.40–41) are course-sanctioned selection tools, but
  they are unstable (p.50) and must be run inside the training window only. Duplicated columns cause exact
  collinearity and rank deficiency (compare the p.7 Df Model 14 vs 16 covariates); detect and drop exact duplicates.
- **3, question (2), does macro add?** Nested-model logic (partial F, p.11–15; `anova_lm(base, full)`) is the
  in-sample version. The course's thesis (p.28) is to judge by prediction, so compare R^2_OOS of the characteristics-
  only vs characteristics+macro models under the same scheme and the same OOS window. The **placebo** (macro shifted 36
  months) follows the p.26–28 "honest null" design: build predictors that by construction carry no timely signal. If
  real macro does no better than the placebo, its apparent contribution is search or noise.
- **Tuning by information criteria:** the notebook allows "cross-validation or information criteria for tuning". AIC,
  BIC and BIC model probabilities (p.29–32) are course-sanctioned, e.g. choosing the number of predictors or lag
  length. BIC favours simpler models (p.33). Compute them on the training sample only, and compare only on identical
  n (p.50 caveat).
- **Parsimony:** Ockham's razor (p.23). Per-class vs pooled models are a complexity trade-off (p.24: fit vs
  simplicity).
- **3-portfolio:** L4 does not cover portfolios, Sharpe ratios or transaction costs.
