message: Referee report: Problem 3 (cross-country return prediction). Read-only review; nothing under /home/user/UChicago was modified.

I reproduced the development run (dev_prefix plus cells F/G/H) and ran my own checks in `<scratch>/review/` (chk2–chk7.py, repro.log). The executed notebook at git HEAD (ee43a2e) prints the same headline numbers: Q1 0.042% (SE 0.241%), Q2 −0.123% (SE 0.085%), Q3 net Sharpe 0.44 with α t = 1.74. It also shows 58 of 58 code cells executed without error and a Problem 3 runtime of 8.9 minutes.

The code is clean. The problems are in what the write-up claims.

## FATAL
**No finding.** None of the course's fatal patterns is present: no shuffled split, fold or `*CV` estimator; no scaler, imputer or PCA fitted before the split; no `r2_score`; no `LogisticRegression`. I found no look-ahead in any model input.

## MAJOR

**1. The "static class tilt / fixed overweight / not timing" attribution is contradicted by the ledger's own forecasts.**
- Where: writeup.md l.26 (Preview), l.174, l.227 (Conclusion).
- Evidence:
  - A truly static class tilt is P1 on `ridge|M2|static`: one fit at 2010-12, which chose the no-signal end of the grid, so it is class means frozen at 2010. Its net Sharpe is **0.27**, below RP's 0.30 (my chk3.py, using the notebook's own weight, turnover and cost formulas; it reproduces the primary P1 at 0.444).
  - P1 on the class-means model is not static. Its class-D share moves from 45% (2011) to 72–74% (2020–21) and back to 46% (2024). Its class-B forecast falls from 0.122 to −0.011.
  - So the 0.44 comes from the class means being re-estimated each year on an expanding window. That is slow class-level learning or timing, not a fixed tilt.
  - The regression with the class-means portfolio as a fourth regressor (β 0.95, R² 0.998) only shows P1(M2) ≈ P1(M1): average weight correlation 0.9994.
- Fix: add P1 on `ridge|M2|static` to Tables 3.24 and 3.27. Rewrite the attribution: "the characteristics add nothing; the gain comes from annually updated class-mean estimates, and a frozen 2010 tilt does worse than RP; α not significant."

**2. The Conclusion's "it halved in the second half" is unsupported, and wrong if "it" is α.**
- Where: writeup.md l.227; the sub-period story at l.180 and l.189.
- Evidence:
  - No cell computes sub-period α. I computed it: 2011–17 α = −0.16%/yr (t −0.37); 2018–24 α = +1.36%/yr (t 2.47). The outperformance against the three rules is concentrated after 2018.
  - That undercuts the "equity–bond hedge broke, so the bond tilt stopped working" explanation.
  - The Sharpe ratio fell 0.58 → 0.32 (−44%).
  - Also unmentioned: in 2018–24, P1 (0.32) is beaten by EW (0.53) (Table 3.25).
- Fix: print sub-period α in G3 and state these numbers.

**3. The power statement does not apply to the pre-registered Q1 test, and "there was none" / "the strong evidence is negative" overclaim.**
- Where: writeup.md l.21, l.185, l.191–193; cells_a.py A1B.
- Evidence:
  - The power calculation (t = 3.34 for a within-class R² of 0.3%) implies an SE of about 0.09%.
  - The realised SE of the Q1 statistic (ridge M2 vs pooled mean) is 0.241%, because class-intercept errors load on common class shocks. The Q1 rule therefore needed R²_OOS ≳ 0.5%; a true 0.3% gives t ≈ 1.2.
  - The calculation also assumes the signal is known, ignoring estimation error.
  - "No specification beats zero" is 38 highly dependent point estimates, driven by one common mean drop (zero beats the pooled mean by 0.32%). Ridge M2 vs zero is −0.28% with SE 0.57%.
- Fix: report the minimum detectable effect of the actual statistic. Use the paired "characteristics beyond class means" delta (−0.045%, SE 0.027%) as the within-class evidence, with the caveat. Call the negative evidence "no detectable predictability", not "strong".

**4. Three result statements are false, and one of them is printed by no cell.**
- **l.130, "Every difference is inside one SE."** No cell computes paired scheme differences. My paired month-bootstrap:
  - ridge M2 static − expanding = −0.727% (SE 0.385%, t −1.89)
  - OLS M2: t −2.47
  - RF M2: t −2.10
  - pcr M3: t −1.89

  Even with the unpaired SEs, static (−0.68%) vs expanding (+0.04%) is more than one SE apart.
- **l.125, "The shallow forest (leaf 200) and the linear models sit between −0.4% and +0.2%."** Many are outside that range:
  - rf M3: −3.74% (static), −1.47% (expanding), −0.74% (rolling)
  - ridge-per-class M3: −1.17%
  - pcr-K5: −1.13%
  - ols M2 static −0.97%, pcr static −0.93%, ridge/lasso static −0.68%
  - ridge M3-lag1 −0.45%, ridge M3-levels −0.43%
- **l.26, "(0.44 vs 0.28–0.30)".** TSMOM is 0.12, so the range should be 0.12–0.30.
- Fix: add a paired scheme-difference table to cell E3 and correct the text.

**5. The benchmark discussion leaves out the per-asset result.**
- Where: writeup.md l.77–78, which says per-asset means are "always shown".
- Evidence:
  - Against the per-asset trailing mean of y (the literal per-series HW5 harness), ridge M2 expanding scores **+0.32% (SE 0.164%, t 1.92)** and the class-means model **+0.36% (t 2.18)**.
  - 17 of 38 specifications are positive against it, which sits inside the 0.3–0.5% range the write-up cites.
  - This is shrinkage of noisy asset means toward class means, not characteristic predictability. But the write-up only contrasts pooled with zero ("the easier of the two").
- Fix: add one paragraph with this number and its interpretation.

**6. The Discussion's "why there is so little" logic is partly wrong.** (This one is borderline major/minor.)
- **l.186**, "a characteristic model can only improve on the rules through x1, x3 and x4": a within-class cross-sectional bet on x2 or x5 rank is a different bet from TSMOM or RP weighting. u5 is in fact the most sign-stable coefficient, and P3 is exactly that bet.
- **l.187**, "macro enters at the class level … about 8 independent assets": this holds for global macro only. For country macro, 20–46% of the variance of `c_x86` and `c_x12` within B, C and D is cross-country, within the month (chk6.py).
- Fix: correct both statements.

## MINOR

**Placebo and grid design**
1. **Placebo wording.** cells_f.py l.11 says "No shift exceeds 130, so no future macro value lands on an out-of-sample date." That is literally true, but:
   - For s ≥ 72 the trailing-z window at early OOS dates contains wrapped future values. For s = 120, features for 49 OOS target months (2011-01 to 2015-01) change when post-2016 macro is altered.
   - For every shift, about 59 training months carry wrapped later macro.
   - METHODOLOGY_LOG P3-14 and the plan promised this disclosure; it is missing. Add it.
2. **The PCR grid has no K = 0 (no-signal) option** (cells_a.py l.94), while ridge and lasso can reach class-means-only. PCR sits at K = 1 in 86% of refits. With K = 0 allowed, validation picks 0 in 8 of 14 refits and R² improves from −0.17% to −0.09%. "PCR mostly chooses one component" (l.126) is therefore a grid artifact. Add K = 0 or disclose.

**Data claims**
3. **x4 and x1 claims generalised beyond class B** (l.50, l.52). The rate-differential evidence exists for currencies only; for class C, x4 correlates −0.60 with the past 12-month return.
4. **x11** (l.57): country 11 stops in 2024-03, after the 2021–23 shock, not "just before" it.
5. **Stale-x1 cut** (l.59): "without changing anything" is inaccurate; ridge M2 goes from 0.043% to 0.026%. Say "negligibly".
6. **Extended-file stamping** (l.58; cells_b.py B2 print): the 35 annual and 26 quarterly columns are called "stamped at the start of their period" / "look-ahead stamps". The notebook only shows which months they change in; the same-period-average property is verified only for x10/x73.

**Citations and methods presented as taught**
7. Asness, Moskowitz and Pedersen (2013) is cited for "carry, low risk" (l.17); that paper covers value and momentum only.
8. L1 p.36 is cited for the R² magnitude (l.18); that page is a portfolio figure. L5 p.57 is the right source.
9. **Spot-check of 36 citations.** 34 say what they are used for. The weak or mis-attributed ones:
   - leaf 200 is cited to L8 p.44, which says trees are "grown deep"; the log records it as a departure but the notebook does not flag it;
   - L8 p.51, cited for deep-forest overfitting, actually says averaging kills a deep tree's variance;
   - L3 p.12–18 (outlier diagnostics), cited as support for ranks;
   - L5 p.23 (penalty scale), cited for "cannot leak";
   - L7 p.23 (k-means scaling), cited for the z-score variant;
   - PCR listed under "(lecture menu)" (l.88); PCA is taught, the regression on components is not;
   - the month-block bootstrap, cited to L2 p.80 without "our construction" in the notebook (the log does flag it).
10. **Cell references are wrong in the final notebook.** "cell 33/34/36/37" (13 occurrences in cells_*.py and writeup.md) are 0-based indices of the original exam. In the submitted notebook, cell 36 is a Problem 2 code cell. The exam text is at indices 44/45/47/48.

**Descriptions and labels**
11. **Class risk budget description** (cells_g.py l.9): "each class sleeve scaled by … its own trailing volatility … applied to RP and to P1". For P1 the RP sleeve's volatility is used, not the P1 sleeve's own. By construction, both variants then have identical class shares (Table 3.28).
12. **Turnover wording** (cells_c.py l.26): Σ|Δw| is called "one-way"; by the usual convention one-way is ½Σ|Δw|. The costs are therefore 2× under that convention (conservative). Clarify the wording.
13. **Audit table** (cells_h.py l.24): "426 linear refits, each tuned on 3 forward validation folds" includes about 100 untuned refits (OLS, fixed-K PCR).
14. **Unit test** (cells_d.py l.198): `bb < (bb + 1)` is vacuous. The test-window order is really asserted in `run_linear`/`run_rf`, which is fine.

**Weak inference and omissions**
15. **Sign stability** (l.135): "three keep their sign in every refit" is weak evidence. Expanding windows share most of their rows, and in 64% of refits α/n = 1000, where the coefficient sign is essentially the sign of the univariate covariance.
16. **Release-lag inconsistency.** Macro gets a 2-month lag for release delay, but class-B x1 and x4 are same-month functions of x86 used at a 1-month lag. This is legal under the exam rule, but it gives the characteristics a timing edge in the Q2 comparison; say so.
17. **Scheme sensitivity of the portfolio is unreported.** P1 on `ridge|M2|rolling` has a net Sharpe of 0.60 with 8.2% monthly turnover. Mention it as unselected, not as a result.
18. **Pre-registration is not evidenced by a commit.** The Q1 and Q3 decision rules first appear in version control at 19:50:57 (591c8ff), after the dev ledger file was written at 19:40. The only supporting evidence is cells_a.py's modification time (19:07). Word the "written before any OOS number" claim accordingly.

**Exam coverage gaps**
19. Not tried and not discussed as rejected:
    - characteristic × macro-state interactions;
    - cross-country macro transforms (rank across the 12 countries, deviation from the cross-country mean, widened panel).
20. The tercile table shows only the top-minus-bottom spread in y units, not each tercile's average next-month return.
21. The Methodology section is not reproducible "from this section alone". Missing: the grids, the α/n parametrisation, the number of bootstrap draws, the turnover/drift formula, the P3 formula, the class-risk-budget construction, the per-class and levels specifications.
22. The HEAD notebook's final cell is still only "## Write-up" (13 characters); the write-up has not been inserted yet.

**Class-server robustness**
23. No breaking API found. Running with warnings enabled, only a seaborn PendingDeprecationWarning appears. `lasso_path` in sklearn 1.9 has `alphas='warn'` as its default; passing an explicit list, as the code does, is safe.
24. `warnings.filterwarnings('ignore')` would hide any future deprecation.
25. Runtime is 8.9 minutes on 4 cores (`rf|M3|expanding` with permutation importance takes 294 s), so expect longer on fewer cores.
26. The dev environment is Python 3.11, so Python 3.14 itself was not tested.

## Verified correct
- **Lags and timing:**
  - x1/x3/x4 are shifted 1 month; x2 and x5 are used as stored (identities hold for 99.6%).
  - Every macro series is lagged 2 months, after log, differential against country 7 and a backward trailing 60-month z-score (min 24).
  - Ranks use one month only.
  - σ̂ is the 36-month rolling SD shifted 1.
  - x10 at a 14-month lag is safe (13 would suffice).
  - The x12–x16 and x86 series actually used are all monthly; x86 has no gaps; logged series are positive.
  - Back-fills are set to missing.
- **Causality:** truncation tests at 2010-12 (the notebook's) plus my cuts at 2015-06, 2019-12 and 2023-12 all give a max difference of 0.0. b_pool and b_asset reproduce exactly from truncated data. RP/TSMOM weights and the sleeve volatility use data through t−1 only.
- **Harness:**
  - static/expanding/rolling windows, 14 refits, rolling length 96;
  - folds (a..b−12j / next 12 months) moving forward only; no test row ever inside a fit window;
  - ridge α/n = a·n on an unscaled SSE, matching sklearn `Ridge(alpha=a·n)`;
  - class demeaning and the scaler use fit-row means at predict time;
  - lasso path order and PCR = PCA + OLS are correct;
  - ties go to the heavier penalty;
  - the bootstrap and paired-delta formulas are correct;
  - re-running lasso, per-class ridge, pcr-K3 and the static RF reproduces the saved forecasts (max difference ≤ 1e-16).
- **Decision rules** are applied exactly as pre-registered (Q1 NO, Q2 NO, Q3 NO). The Q3 α t is robust to HC1 and HAC(3, 6) errors: 1.72–1.77.
- **Numbers:** every other number in writeup.md matches a printed output (Tables 3.1–3.31).