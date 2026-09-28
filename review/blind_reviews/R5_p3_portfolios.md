# R5 - Problem 3, step 4 ("From forecasts to portfolios") and research question Q3

Scope: benchmark construction, forecast-portfolio construction and timing, turnover and costs, drawdowns, exposure versus timing (the class-means question), statistical significance, concentration, and whether the Q3 conclusion follows from printed evidence. Grading standard: the exam, the AI Coding Guide, the 8 lectures, the numbers each cell prints, and honest reporting.

Work files: `reports/p3port_work/A_p3.py` (A's cells 35 and 39, extracted and re-run with `data/` as the working directory), `A_run.log` (its output, identical to the printed outputs in `solution_A_P3.txt`), `A_checks.py` (my checks on A's own forecasts).

---

## 1. What each solution built

| item | Solution A | Solution B |
|---|---|---|
| sigma-hat | 36-month SD (ddof=1) of the asset's own past returns, `shift(1)` | the same (`SD36`); a causality assert rebuilds RP/TSMOM from data cut at 2010-12 |
| TSMOM signal | compounded return over t-12..t-1 (`shift(1).rolling(12)`) | the same (`R12`); identity x2 = R12 checked |
| EW / RP / TSMOM | 1/50; 1/sigma normalised; sign(R12)/sigma normalised to sum of abs(w) = 1 | the same, with an assert that sum of abs(w) = 1 |
| Forecast portfolio | P1: w proportional to yhat/sigma, yhat = ridge M2 = class mean + characteristic tilt (alpha = 50, not tuned) | P1: the same rule on the pre-registered ridge M2 (tuned on forward folds); also P3 (within-class long-short), P1 on class means (M1), static and rolling P1, and class-risk-budget versions |
| Timing | forecasts refit each January on data through December; sigma and R12 lagged | the same; the train < test order is asserted at every refit |
| OOS window | 2011-01 to 2024-12, 168 months | the same |
| Turnover | sum of abs(w_t - w_{t-1}) using **target** weights, with no drift; first month charged turnover = 1 for every strategy | sum of abs(w_t - w_{t-1} drifted), i.e. traded notional against drifted weights; first-month build excluded for every strategy |
| Cost | 10 bp per unit of turnover | 10 bp per unit of the sum, plus a 0-50 bp grid and break-even costs |
| Drawdown | max DD of net wealth; underwater plot | max DD, worst month; drawdown plot |

**Benchmarks.** Both solutions build the three benchmarks exactly as the exam defines them, and the numbers agree. Gross Sharpe ratios: EW 0.282, RP 0.308, TSMOM 0.199 in both. I also reproduced these from A's re-executed panel with B's drift-based turnover and got EW 0.041, RP 0.036 and TSMOM 0.265 per month, which match B's Table 3.9 exactly. No look-ahead was found in either portfolio construction.

---

## 2. Independent check of Solution A: where the P1 outperformance comes from

A never separates the class-mean part of its forecast from the characteristic part. I ran that check with A's own objects (`oos_results`: `pred_m1` = A's class-means model, `pred_m2_ridge` = A's P1 forecast). Portfolios use A's rule w proportional to yhat/sigma_36m at unit gross. My P1 gross return series matches A's `port_df` to within 1e-17.

| portfolio (A's forecasts) | gross SR | net SR, drift turnover, 10 bp | net SR, A's convention | turnover/month (target) | turnover/month (drift) | max DD (gross) |
|---|---|---|---|---|---|---|
| EW | 0.282 | 0.276 | 0.282 | **0.000** | 0.041 | 0.242 |
| RP | 0.308 | 0.300 | 0.303 | 0.022 | 0.036 | 0.114 |
| TSMOM | 0.199 | 0.121 | 0.125 | 0.251 | 0.265 | 0.125 |
| **P1 (A's headline)** | 0.454 | 0.427 | 0.427 | 0.089 | 0.091 | 0.096 |
| **Class means only (A's M1 / sigma)** | 0.450 | **0.440** | 0.442 | 0.029 | 0.038 | 0.103 |
| Characteristic tilt only (P1 forecast demeaned within class) | 0.124 | **-0.074** | -0.073 | 0.340 | 0.342 | 0.053 |
| Class means frozen at 2010 | 0.275 | 0.267 | 0.270 | 0.022 | 0.036 | 0.110 |

Attribution, monthly net returns, 168 months (A's net series):

| regression | alpha (ann.) | t(alpha) | HAC(6) t | R^2 | key beta |
|---|---|---|---|---|---|
| P1 ~ EW + RP + TSMOM (A's Table 8) | +0.52% | 1.56 | 1.61 | 0.903 | RP 1.185 |
| **P1 ~ EW + RP + TSMOM + class-means portfolio** | **+0.00%** | **0.01** | 0.01 | **0.988** | M1 0.854 |
| class-means portfolio ~ EW + RP + TSMOM | +0.61% | 1.66 | 1.68 | 0.897 | RP 1.208 |
| P1 minus class-means portfolio (net spread) | -0.17%/yr | -1.20 | | | |
| P1 ~ rules, 2011-2017 | -0.27% | -0.65 | | | |
| P1 ~ rules, 2018-2024 | +1.19% | 2.27 | | | |

Paired month bootstrap (10,000 draws) of net Sharpe differences for A's P1:

| P1 minus | difference | SE | 95% interval |
|---|---|---|---|
| EW | 0.145 | 0.187 | [-0.22, 0.52] |
| RP | 0.124 | 0.112 (12-month block: 0.131) | [-0.09, 0.35] |
| TSMOM | 0.303 | 0.420 | [-0.51, 1.14] |
| class-means portfolio | -0.015 | 0.032 | [-0.08, 0.05] |

Concentration in A's P1 (never reported by A):
- Class D (bonds) carries 59.2% of gross exposure, against 37.3% under RP. Class B carries 7.9%, against 21.5%.
- asset_16 has an average abs(weight) of 24.1% and reaches 41.8%. This is the low-volatility bond that A's own trap audit (T7) flags for its x5 floor.
- Net exposure averages 0.98, and P1 is short at least one class in 14.3% of months.
- The average monthly correlation between P1 weights and class-means weights is 0.984. Against RP weights it is 0.942.

**Result.** A's "forecast" portfolio is risk parity plus a class-means tilt: long bonds, light on currencies, with the tilt re-estimated each year. The class-means tilt alone earns a higher net Sharpe ratio than P1 (0.440 against 0.427). Once the class-means portfolio is a regressor, alpha is 0.00% (t = 0.01). The pure characteristic bet loses money after costs (net SR -0.07), and the characteristic tilt triples turnover (0.029 to 0.089). The tilt frozen at 2010 earns only 0.27, below RP. Solution B reports the same facts (Tables 3.24, 3.27, 3.28b), and its numbers agree with mine to about 0.01 (class-means net SR 0.440 in both; frozen 0.267 in both; B's class-mean forecasts by year in Table 3.28b equal A's `pred_m1` by year to three decimals). The finding is therefore robust across two independent pipelines.

---

## 3. Findings table

| id | solution | area | severity | finding | evidence |
|---|---|---|---|---|---|
| A1 | A | exposure vs timing | ERROR | The step-4 instruction "how much is exposure to the benchmark rules and how much is timing" is met only by regressing P1 on EW/RP/TSMOM. A has a class-means model (M1) in hand but never forms a class-means portfolio or uses it as a regressor. The write-up credits the performance to "tactical cross-sectional tilts" and "combining structural risk parity with tactical cross-sectional tilts", and 6.2 says "a forecast model with R^2 ~ 0 can deliver substantial Sharpe gains". My check with A's own code: class means alone give net SR 0.440 > P1's 0.427; alpha goes to 0.00% (t = 0.01) once the class-means portfolio is added; the characteristic-only portfolio has net SR -0.07. The outperformance is a static/slowly updated class-premium tilt, not characteristic timing. | solution_A_P3.txt lines 783-785, 802, 934-938; A_checks.py output (section 2 above) |
| A2 | A | significance / honesty | ERROR | The abstract says "Forecast-Based Allocations **Decisively** Beat Simple Heuristics" and "decisively outperforming". No SE or interval is given for any Sharpe difference. My bootstrap gives P1 - RP = 0.12 (SE 0.11), P1 - EW = 0.15 (SE 0.19) and P1 - TSMOM = 0.30 (SE 0.42), all within about 1.1 SE. "Decisively" is unsupported. | lines 767, 785; bootstrap table above |
| A3 | A | significance / honesty | ERROR | The alpha of +0.52% with t = 1.56 (p = 0.121, quoted by A itself) is described as "positive trend-following alpha" and "Positive net timing alpha". It is not significant, and it is not "trend-following": the TSMOM beta (0.086) is exposure, not alpha. The sub-period split (which A does not run) shows alpha -0.27% (t = -0.65) in 2011-17 and +1.19% (t = 2.27) in 2018-24, so any alpha sits in one half. | lines 767, 916; A_checks.py sub-period output |
| A4 | A | concentration | WEAKNESS | Concentration is never measured or discussed. P1 puts 59% of gross in class D and on average 24% (max 42%) in asset_16, the JGB-like asset that A itself flagged for a volatility floor during YCC. P1's low volatility and small drawdown come largely from this bond tilt, which is not disclosed. | A_checks.py (class shares, asset_16) |
| A5 | A | turnover / costs | WEAKNESS | Turnover is computed on target weights with no drift, so a monthly-rebalanced EW book shows turnover 0.000 and pays no cost after month 1. RP turnover is understated (0.022 against 0.036 with drift). The convention is not stated anywhere. The effect on Sharpe is small (EW net 0.281 against 0.276 with drift) and the ranking does not change, but the printed "0.000" misdescribes trading. First-month cost (turnover = 1) is charged uniformly, which is acceptable. There is only one cost level, with no grid or break-even. | code lines 562-566; Table 7; A_checks.py TO_target vs TO_drift |
| A6 | A | numbers printed | WEAKNESS | The write-up attribution table quotes standard errors (0.0033, 0.0512, 0.0303, 0.0249), p-values (0.121, 0.000, 0.000, 0.001) and "F = 508.2". No cell prints these. Table 8 prints only R^2, alpha, t and betas. Recomputed, the values are right except F, which is 507.9, not 508.2. This breaks "every number comes from a cell". | lines 912-921 vs Table 8 output (lines 736-742) |
| A7 | A | interpretation | WEAKNESS | Drawdown and volatility are compared across portfolios at very different volatility levels ("restricting maximum drawdown to -9.70% versus -24.15% for EW"; "cuts portfolio drawdowns in half"). At unit gross exposure, P1 has 3.97% vol and EW 8.92%, so the drawdown gap is mostly scale, not skill. The claim that the continuous forecast is "cutting turnover to 8.9%" compares only with TSMOM. P1's turnover is 4 times RP's (2.2%) and 3 times the class-means portfolio's. | lines 930-933, 945 |
| A8 | A | Q3 conclusion | ERROR | Section 7 and the abstract state that the forecast portfolio beats the rules and that its value comes from characteristic signals. The printed evidence (alpha t = 1.56, no Sharpe-difference SE) and the check A omitted (class means explain everything) do not support this. This is an unexplained positive result presented as a finding. No sub-period analysis is given, although the exam lists sub-periods. | lines 767, 802, 940-944 |
| A9 | A | style | STYLE | The markdown has many LaTeX characters mangled ("$" stripped; ".1\%$", ".97\%$", "\u0007lpha"), so several quoted numbers in 6.2 are unreadable. | lines 931-933, 916-919 |
| A10 | A | benchmarks / timing | (credit) | The benchmarks are exactly as the exam defines them. sigma-hat and R12 use only past data. The annual refit uses data through December. Unit gross exposure and monthly rebalancing hold. Cumulative net return and underwater plots are present. The attribution regression on the three benchmarks is done. | code lines 290-298, 538-560 |
| B1 | B | exposure vs timing | (credit, exemplary) | B runs every check the step requires. P1 on class means (M1) earns the same net Sharpe (0.44), and the weight correlation of P1 with M1 is 0.9994. With M1 as a fourth regressor, alpha t = 0.82 and R^2 = 0.998. P3 (the within-class characteristic bet) has gross SR 0.08 and net -0.11 at 34% monthly turnover. The static (frozen 2010) P1 earns 0.27 with alpha -0.16% (t = -0.91). Weight correlations are reported for RP (0.935) and TSMOM (0.251). The class tilt is shown year by year. Conclusion: "the characteristics contribute nothing to it: it is risk parity plus a class tilt that follows the running class means." My independent replication on A's pipeline confirms every one of these. | Tables 3.24, 3.27, 3.28, 3.28b; write-up section 4.3 |
| B2 | B | significance | (credit) | Paired month-bootstrap SEs are given for Sharpe differences (P1 - RP 0.144, SE 0.109; - EW 0.168, SE 0.181; - TSMOM 0.324, SE 0.414; - frozen 0.178, SE 0.109). The alpha has textbook and bootstrap SEs (t = 1.74 / 1.74). An influence diagnostic shows the alpha reaches t = 2.20 only after dropping 6 flagged months; this is "reported only". A pre-registered Q3 rule (beat all three rules AND alpha t > 2) returns **NO**. Wording stays within the evidence ("not significant", "not precise", "1.3 SEs"). | Tables 3.27e, 3.29; Q3 verdict print; write-up 4.3, 5, 6 |
| B3 | B | concentration | (credit) | B reports 60% of gross in class D (37% under RP), asset_16 averaging 22% and reaching 42%, net exposure 0.99, and short a class in 14% of months. It adds a class-risk-budget variant that cuts D to 46% and net SR to 0.41, and recommends concentration limits. | Tables 3.9b, 3.28; overlap table; conclusion |
| B4 | B | costs / turnover | (credit) | Turnover is measured against drifted weights (correct traded notional). The 10 bp cost is charged to every strategy. A 0-50 bp grid is given, plus break-even costs (P1 = EW at 260 bp, P1 = RP at 363 bp, net mean = 0 at 374 bp). | Tables 3.9, 3.24, 3.26 |
| B5 | B | costs | WEAKNESS (minor) | The first OOS month's initial build is excluded from turnover and costs for every strategy. This is uniform and disclosed, but it slightly flatters high-turnover strategies relative to charging the build. | cell 71 `perf`; section 3.3 text |
| B6 | B | terminology | STYLE | The design table says "10 bp per unit of one-way turnover", but the code charges per unit of the buys-plus-sells sum. B notices this and clarifies in the appendix: conservative under the half-sum convention. | line 158 vs lines 1123, 3070 |
| B7 | B | inference | WEAKNESS (minor) | The Sharpe-difference bootstrap treats months as i.i.d. B discloses this and reports that a 12-month block bootstrap inflates SEs by a median of about 1.19. My block SE for P1 - RP (0.131 against 0.112 i.i.d.) agrees. This does not affect the conclusion, which is already "not significant". | Limitations paragraph; my block bootstrap |
| B8 | B | benchmark switching | (credit) | The rolling-window P1 (net SR 0.60) is printed but explicitly labelled "not selected; diagnostic ... not a result". The pre-registered expanding P1 remains the Q3 answer. Sub-period regressions and the static/rolling variants are flagged as post-review additions. There is no benchmark or spec switching. | Table C; write-up 4.3 |
| B9 | B | sub-periods | (credit) | Net Sharpe by half is reported, 2011-17 / 2018-24. P1 is 0.58 / 0.32, and B notes that EW (0.53) beats P1 in 2018-24. Alpha by half is -0.16% (t = -0.37) and +1.36% (t = 2.47). The C-D correlation regime change is discussed, with the caveat that its periods do not match the halves. | Tables 3.25, 3.27b, 3.30 |
| B10 | B | numbers printed | (credit) | A dedicated "headline numbers" cell prints every Q3 figure quoted in the write-up. I spot-checked 20 Q3 numbers against the cell outputs and all matched. | cell 104 output, lines 3016-3050 |

---

## 4. Direct comparison where the solutions differ

1. **Class means versus characteristic timing.** B is correct and A is wrong. A attributes P1's Sharpe to characteristic "tactical tilts". On A's own forecasts, the class-means portfolio alone matches or beats P1 and absorbs all of its alpha. The characteristic tilt adds turnover and loses money on its own. B found and reported this. A did not look, although the AI Coding Guide says (section 6.5) "Verify the thing that matters ... ask yourself what would have to be true for it to be believable, then check that thing."
2. **Significance.** B is correct. B gives SEs for every Sharpe difference and for alpha, and states that none clears 2 SE. A gives none for Sharpe differences, and calls a t = 1.56 alpha "positive alpha" and the result "decisive".
3. **Turnover.** B's drift-adjusted definition is the right measure of traded notional. A's target-weight definition gives EW zero turnover. The Sharpe impact is small, but B is more accurate and also reports a cost grid and break-evens.
4. **Concentration.** Only B measures and discusses it (60% in bonds, 22% in one asset), and only B tests a risk-budget remedy.
5. **Q3 answer.** B's answer is "higher point estimate, not significant, not from the characteristics, a class-premium tilt, concentrated, only in 2018-24". It follows from the printed evidence and survives my replication. A's answer ("decisively beat ... after costs") does not.

---

## 5. Verdict on Q3 (the correct answer)

Out of sample (2011-2024, 168 months, 10 bp costs), the vol-scaled forecast portfolio has a higher net Sharpe ratio than the three rules as a point estimate: about 0.43-0.44, against RP 0.30, EW 0.28 and TSMOM 0.12. The advantage is **not statistically significant**. Against RP it is about 0.12-0.14 (SE about 0.11, about 1.1-1.3 SE). Alpha on the three rules is about 0.5-0.6% a year with t of about 1.6-1.7, negative in 2011-17 and positive only in 2018-24.

The advantage **does not come from the characteristic forecasts**. A portfolio built only from trailing class means (w proportional to class mean of y / sigma) earns the same or a higher net Sharpe, holds weights correlated about 0.98-0.999 with P1, and drives alpha to zero when added as a regressor. The within-class characteristic bet loses after costs. The class tilt frozen at 2010 underperforms RP.

The portfolio is risk parity plus a slowly re-estimated class-premium tilt. That tilt puts about 60% of gross exposure in bonds and about 22-24% in one low-volatility bond. The correct answer to Q3 is therefore: **no demonstrated outperformance from forecasting.** At most, there is a fragile, concentrated class-tilt effect that needs fresh data to confirm.

---

## 6. Scores (out of 15 for this part)

| component (max) | A | B |
|---|---|---|
| Benchmark and forecast-portfolio construction, timing (4) | 4 | 4 |
| Turnover, costs, drawdowns, sub-periods (3) | 1.5 | 2.5 |
| Exposure vs timing / class-means decomposition (4) | 1 | 4 |
| Statistical significance of Sharpe differences and alpha (2) | 0.5 | 2 |
| Concentration and risk discussion (1) | 0 | 1 |
| Q3 conclusion follows from printed evidence, all numbers printed (1) | 0 | 1 |
| **Total** | **7 -> 6 after honesty deduction** | **14.5 -> 14** |

**Solution A: 6/15.** The mechanics are correct, and a single 3-factor attribution was run. However, the headline claim is an unexplained, misattributed positive result. The class-means check A never ran shows the "forecast" edge is a class-premium tilt. Sharpe differences have no uncertainty, "decisive" and "positive alpha" are overclaims, concentration is not reported, a few quoted statistics are not printed by any cell, and EW turnover is printed as 0.000. The exam's honest-reporting rule ("an unexplained positive one does not [earn credit]") applies to the Q3 conclusion.

**Solution B: 14/15.** This is an exemplary treatment of step 4. It includes every decomposition the step asks for (class-means portfolio, class-means regressor, weight correlations, within-class P3, frozen and rolling diagnostics), bootstrap SEs for Sharpe differences and alpha, a pre-registered decision rule that returns NO, a full discussion of concentration with a remedy, drift-adjusted turnover with a cost grid and break-evens, and every quoted number printed. The small deductions are the uncharged first-month build, the "one-way" wording inconsistency, and the i.i.d. month bootstrap, which B itself discloses.
