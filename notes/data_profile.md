# Data profile (pre-modelling), Final Autumn 2026

Scripts: `<scratch>/profile/*.py` (a_b, b2, c, d1–d13, e, f, f2, g1–g11). Python 3.11, pandas 3.0.6. Nothing in `/home/user/UChicago` was modified (`git status` is clean).

**Anti-snooping compliance.** No predictor (x1..x164) was ever compared with the return it would forecast. All predictor-vs-return work uses the **forecast-aligned frame**: target month t, usable predictors = x2, x5 as stored and x1, x3, x4 shifted by +1 within asset, and return windows ending at **t-1** (for example R12 = r_{t-12..t-1} and SD36 = sd(r_{t-36..t-1})). For stored-contemporaneous variables (x1, x3, x4, macro), that is the same as "stored row s against returns through s". These returns are all dated before the target month s+1 in which the variable gets used. No tercile sorts, no regressions of returns, no backtests, no strategy Sharpe ratios were computed.

**pandas 3 gotchas (verified).** String columns load as `StringDtype` (`str`), not `object`. `parse_dates` gives `datetime64[us]`, not `[ns]`. `DataFrame.values` from `.corr()` is read-only (copy-on-write).

---

## A. Social_Network_Ads.csv

| item | value |
|---|---|
| shape | 400 × 5: `User ID` int64, `Gender` str, `Age` int64, `EstimatedSalary` int64, `Purchased` int64 |
| NaN | none |
| class balance | Purchased=1: **143 (35.75%)**, 0: 257 |
| "nobody purchases" accuracy | **0.6425** (257/400) |
| Age | 18 – 60, mean 37.66, sd 10.48, median 37 |
| EstimatedSalary | 15,000 – 150,000, mean 69,742.5, sd 34,097, median 70,000; every value is a multiple of 1,000 |
| distinct values | Gender **2**, Age **43**, EstimatedSalary **117**, User ID 400 (unique, so it must not be used as a feature) |
| Gender | Female 204 (purchase rate 37.7%), Male 196 (33.7%) |
| duplicates | 0 full-row duplicates. Excluding User ID there are 20 duplicate rows. 35 duplicated (Age, Salary) pairs. 1 (Gender, Age, Salary) cell and 2 (Age, Salary) cells carry conflicting labels. |

What this means for 1.4: MDI importance is biased toward high-cardinality splitters (Salary 117 values > Age 43 > Gender 2). Gender offers only one possible split. Stratified 5-fold CV gives about 28–29 positives per fold of 80.

## B. dj30.csv

| item | value |
|---|---|
| shape | 46,445 × 9: `Unnamed: 0` (a saved row index equal to 0..n-1), PERMNO, `date` **int64 YYYYMMDD**, TICKER str, COMNAM str, PERMCO, PRC, RET, MrkRet |
| unique dates | **1,511** (confirmed); 2016-01-04 → 2021-12-31 |
| stocks | 32 PERMNOs / 31 tickers / 33 names. `DOW` maps to two PERMNOs: Dow Chemical (20626, 2016-01-04..2017-09-01) and Dow Inc (18428, 2019-04-03..2021-12-31). Walmart changed name. 30 or 31 rows per date. |
| NaN | exactly 1 row: PERMNO 20626 (DOW) on 20170901, where PRC and RET are NaN. MrkRet has no NaN. |
| MrkRet constant within date? | **Yes.** Max nunique per date = 1 and max within-date range = 0.0. |
| RET rounded? | **Yes, to 2 decimals.** Only 48 unique values and 100% are multiples of 0.01. MrkRet has up to 6 decimals. RET vs price pct-change corr 0.966. MrkRet is **not** the EW mean of RET (corr 0.989), so it is probably an index return. |
| MrkRet (one per date) | n=1,511; mean 0.000645; **sd (ddof=1) 0.011951** (ddof=0: 0.011947); min −0.127612 (2020-03-16), max +0.110192 (2020-03-24) |
| excess kurtosis | **24.07** (`scipy.stats.kurtosis`, default fisher=True, bias=True) / 24.16 (bias-corrected, equal to pandas `.kurt()`). State which one you report. |
| 1% quantile | −0.03325 (`np.percentile` linear); 'lower' −0.033423, 'higher' −0.031693. Normal VaR = 2.326 × 0.011951 = 0.0278, which matches the story's "2.8% of NAV". |

Ambiguity for 2.4(b): the library holds 500 returns but dj30 has 1,511 days. Should all 1,511 days be used, or a 500-day window, and which one? This needs a clarifying question.

## C. asset_panel.csv / asset_info.csv / asset_returns_wide.csv

- Shape 15,000 × 10. `date` parses to month-ends 2000-01-31..2024-12-31, which equals `pd.date_range(freq='ME')` with 300 dates. **50 assets × 300 months, complete**: no NaN, no duplicate (asset, date), no duplicate rows. Class and country are constant within each asset, and asset_info matches the panel exactly.
- Class counts: **A 26, B 8, C 9, D 7** (confirmed). All 26 A assets are `country_7` (confirmed).

| country | B | C | D | (A) |
|---|---|---|---|---|
| 1 | asset_17 | – | – | |
| 2 | – | asset_10 | – | |
| 3 | asset_30 | asset_3 | asset_35 | |
| 4 | – | asset_45 | – | |
| 5 | asset_31 | asset_9 | asset_25 | |
| 6 | – | asset_42 | – | |
| 7 | – | asset_24 | asset_27 | 26 A |
| 8 | asset_22 | asset_38 | asset_21 | |
| 9 | asset_8 | asset_12 | asset_16 | |
| 10 | asset_44 | – | – | |
| 11 | asset_34 | asset_50 | asset_41 | |
| 12 | asset_46 | – | asset_49 | |

- Monthly excess-return vol (ddof=1). Overall min **0.82%** (asset_16, D), median 5.75%, max **17.40%** (asset_47, A). This confirms "under 1% to above 17%".

| class | n | vol min | vol median | vol max | median ann. mean | median ann. Sharpe | worst month |
|---|---|---|---|---|---|---|---|
| A | 26 | 4.23% (a19) | 8.57% | 17.40% (a47) | 7.8% | 0.25 | −54.7% (a33, 2020-03) |
| B | 8 | 2.46% (a30) | 2.80% | 3.52% (a17) | −0.1% | −0.01 | −16.3% |
| C | 9 | 3.92% (a9) | 5.08% | 5.78% (a38) | 4.4% | 0.24 | −25.4% |
| D | 7 | 0.82% (a16) | 1.97% | 2.63% (a27) | 2.8% | 0.47 | −12.1% |

- Average pairwise return correlation (diagonal = within class):

| | A | B | C | D |
|---|---|---|---|---|
| A | 0.21 | 0.24 | 0.18 | −0.08 |
| B | | 0.55 | 0.23 | 0.05 |
| C | | | 0.725 | −0.10 |
| D | | | | 0.58 |

  B, C and D show strong blocks. A is weakly correlated internally, and D is negative with A and C. Return AR(1) medians: A −0.02, B 0.035, C 0.045, D 0.04 (max 0.31).
- Hypothesis: A = commodities (no country, 26 contracts, fat tails; a33 −54.7%/+91.5% in 2020-03/05 looks like crude oil). B = currencies vs the base (country_7 has no B asset, so it is the base currency and Sharpe is about 0). C = equity indices. D = government bonds.
- `asset_returns_wide.csv`: 300 × 50, no NaN, same asset set and dates, **max |wide − panel| = 0.0**.
- Units: returns are decimals. 9 exact zeros. 19 |r| > 0.40, all class A (asset_47 11 times; asset_33 max kurtosis 14.5).

## D. Characteristics x1..x5

**Summary by class** (mean / sd / min / max):

| | A | B | C | D |
|---|---|---|---|---|
| x1 | −0.003 / 0.024 / −0.14 / 0.15 | −0.0003 / 0.0017 / −0.006 / 0.005 | 0.012 / 0.054 / −0.24 / 0.41 | **1.80 / 1.52** / −2.38 / 6.57 |
| x2 | 0.069 / 0.363 / −0.84 / 2.67 | 0.000 / 0.106 / −0.31 / 0.45 | 0.058 / 0.191 / −0.56 / 0.68 | 0.026 / 0.072 / −0.26 / 0.24 |
| x3 | −0.25 / 0.95 / **−9.50** / 0.96 | 0.038 / 0.209 / −0.70 / 0.58 | −0.70 / 0.35 / −1.71 / 0.28 | −0.008 / 0.012 / −0.06 / 0.04 |
| x4 | 0.000 / 0.054 / −0.40 / 0.34 | −0.018 / **1.30** / −4.71 / 5.15 | 0.038 / **1.36** / −4.36 / 7.96 | −0.000 / 0.004 / −0.013 / 0.021 |
| x5 | 0.083 / 0.032 / 0.024 / 0.273 | 0.028 / 0.0075 / 0.013 / 0.057 | 0.049 / 0.015 / 0.019 / 0.100 | 0.018 / 0.0066 / **0.004** / 0.039 |

Scale differs by up to 3 orders of magnitude across classes for x1 and x4 (units differ). They must be standardised within class (or ranked) before pooling.

**Persistence** (median within-asset autocorrelation, lags 1 / 6 / 12 / 24):

| | A | B | C | D |
|---|---|---|---|---|
| x1 | .53/.22/.28/.06 | .95/.83/.60/.11 | **.30/.32/.55/.42 (seasonal)** | .98/.84/.66/.22 |
| x2 | .91/.46/−.06/−.23 | .92/.46/−.03/−.11 | .93/.50/−.05/−.15 | .93/.54/.14/−.01 |
| x3 | .97/.82/.61/.30 | .98/.86/.74/.45 | .96/.78/.63/.36 | .98/.83/.67/.36 |
| x4 | .43/.14/**−.38**/−.10 | .98/.63/.08/−.31 | .96/.58/.00/−.28 | .96/.59/−.02/−.23 |
| x5 | .99/.90/.75/.47 | .99/.91/.78/.46 | .99/.88/.68/.20 | .99/.87/.70/.24 |

For x2 and for x4 in B/C/D, the ACF falls linearly to about 0 at lag 12, which is the signature of a 12-month overlapping window.

**Cross-sectional dispersion.** Within-class monthly sd has median A/B/C/D as follows. x1: .021/.0013/.038/.93. x2: .25/.058/.076/.032. x3: .56/.12/.27/.006. x4: .048/.51/.57/.002. x5: .026/.0058/.0096/.0058. Across all 50 assets, dispersion in x1 and x3 trends down over time (corr with time −0.43, −0.48). Share of within-class variance explained by asset means: x5 0.22–0.58 and x3 in C 0.52 (level differences). x2 and x4 ≈ 0.

**Predictor–predictor correlation** (forecast-aligned, average monthly cross-sectional Spearman, all 50 assets): x1–x2 .19, x1–x3 −.17, x1–x4 .20, **x1–x5 −.38** (driven by class mix), **x2–x3 −.38**, x2–x4 .06, x3–x5 −.04. Within class, notable values are A x1–x4 .49 and **C x3–x5 .57**. Within-class x2–x3 is −.14 to −.41.

**Relation to PAST returns** (forecast frame, within-asset time-series corr averaged over assets in class, windows ending t−1):

| | R1 | R12 | R60 | SD36 (own trailing vol) |
|---|---|---|---|---|
| x1 (A/B/C/D) | .11/.10/−.03/.08 | .28/.37/−.04/.39 | .21/.38/−.11/.47 | .05/.28/.07/.20 |
| x2 | .30/.29/.27/.30 | **.98/1.00/.99/1.00** | .44/.42/.28/.60 | .08/.10/−.11/−.06 |
| x3 | .03/−.15/.11/−.18 | −.43/−.47/−.27/−.59 | **−.92/−.97/−.46/−.97** | −.25/−.14/.30/.33 |
| x4 | .13/−.01/−.12/−.06 | .27/.16/**−.63**/−.13 | .03/.14/−.21/−.22 | −.01/−.10/.12/−.02 |
| x5 | .03/.05/−.01/.02 | .11/.08/−.21/−.04 | .39/.18/−.44/−.27 | **1.00/1.00/1.00/1.00** |

**Exact identifications** (predictor vs past returns and predictor vs macro only):
- **x2 = 12-month momentum, pre-lagged (confirmed).** x2[t] = ∏(1+r_{t−12..t−1}) − 1 exactly (<1e-5) in **14,347 / 14,400** rows from 2001-01 on. The 53 exceptions are always the class-month maximum or minimum, pulled toward the centre, so some within-class winsorisation is applied (rule not identified). The 2000 rows use pre-sample returns. The best alternatives (sum instead of compounding, or window t−13..t−2) have corr ≤ 0.97.
- **x5 = risk measure = 36-month vol, pre-lagged (confirmed).** x5[t] = sd(r_{t−36..t−1}, ddof=1) exactly in 13,157 / 13,200 rows from 2003-01. The exceptions are the **43 months of asset_16 (2019-09..2023-03) floored at 0.004**; the true SD36 goes as low as 0.00298 there, and x5 = 0.004 is the global minimum. Before 2003 the values use pre-sample returns. Class means of x5 (A .083, B .028, C .049, D .018) match realised class vol.
- **x1 = carry (contemporaneous).** B: corr 0.96 with (x86 − x86_country7)/≈1190, the short-rate differential, best at lag 0. D: corr 0.87 with term spread x85 − x86 (in percent). C: seasonal (AC12 > AC1), corr −0.40 with own short rate, so dividend-yield-minus-rate type. A: seasonal, which fits a roll yield/basis.
- **x3 = value / long-horizon reversal (contemporaneous).** Corr −0.93 (A), −0.98 (B), −0.97 (D) with the 60-month cumulative log return. For C it is only −0.45: C mean −0.70 is about log(0.5), so x3 is probably B/M-like for C, and it correlates +0.63 with extended x38 (a dividend-yield-like variable).
- **x4 = 12-month change in the carry/rate fundamental ("macro/carry momentum", contemporaneous).** **B: x4 = Δ12(x86 − x86_c7) exactly** (corr 0.9996, 98.9% within 1e-3, slope 0.997, same date). A: corr 0.984 with Δ12 x1. C: ≈ −Δ12 own short rate (corr 0.86, slope −1.01). D: ≈ −0.0024·Δ12 short rate (corr 0.71). Its values are often rounded to 2 dp (a17, a24, a44, a46 ≥ 76%).

So x4 (B) and x1 (B/D) are near-deterministic functions of country macro. For Q2, "macro beyond characteristics" is partly already inside x1/x4. x2 and x5 are functions of past returns only, which is the same information as the TSMOM and risk-parity benchmarks. That matters for Q3 attribution.

**Leading fills.** Each run below is one value repeated from 2000-01. Assuming the run's last month is the first genuine observation, filled = run − 1.

| var | asset (class) | run | value | filled months |
|---|---|---|---|---|
| x3 | asset_1, asset_2, asset_5, asset_26 (A) | 2000-01..2002-07 (31) | −0.109145 / 0.505351 / 0.369127 / 0.634937 | 30 |
| x3 | asset_21 (D) | 2000-01..2002-08 (32) | −0.012300 | 31 |
| x2 | asset_3 (C) | 2000-01..2000-10 (10) | 0.461332 | 9 |
| x2 | asset_22 (B) | 2000-01..2000-02 (2) | −0.159354 | 1 |
| x5 | asset_3 (C) | 2000-01..2001-04 (16) | 0.072654 | 15 |

That is 7 assets in total, all fills ending by 2002-08. **A backward fill is look-ahead.** For example, the filled x2 of asset_3 is the 12-month return realised up to 2000-09, which contains the 2000-01..09 returns it would be "predicting". Treat these cells as missing, or start estimation after 2002-08.

**Other constant runs.**
- x5 asset_16 has the 43-month floor above.
- **x1 frozen 2024-09..2024-12 for 5 of 8 B assets** (a8, a17, a30, a31, a34) while the rate differential kept moving (a17: −0.71 → −0.04). These are stale end-of-sample values that the notebook does not mention.
- x1 C asset_24 has 3-month runs in 2012 and 2017. x4 has short runs of round values in B/C.

## E. macro_global.csv (x6..x9)

300 × 4, no NaN, dates are the same 300 month-ends.

| | mean | sd | min (date) | max (date) | AR1 | AC12 | interpretation (hypothesis) |
|---|---|---|---|---|---|---|---|
| x6 | 19.87 | 8.07 | 10.13 (2017-10) | 62.67 (2008-11) | 0.845 | 0.22 | VIX-type implied vol (monthly avg) |
| x7 | 0.812 | 0.119 | 0.661 (2014-05) | 1.307 (2020-05) | 0.984 | 0.40 | smoothed stress/uncertainty index. Diff-AC1 is 0.77 and the peak lags crises, so it is a trailing-window construct. |
| x8 | 0.232 | 1.007 | −2.92 (2009-02) | 2.59 (2007-05) | 0.961 | 0.47 | z-scored activity/sentiment indicator (2020 mean −1.28) |
| x9 | 0.0668 | 0.0192 | 0.0223 (2009-02) | 0.1294 (2000-08) | 0.978 | 0.73 | pro-cyclical decimal level (yield/growth-type) |

Correlations: x6–x7 .64, x8–x9 .52, x6–x8 −.43. Against trailing 12-month realised vol of class-EW returns (t−11..t, macro stored at t): x6 .60/.37/.62/.27 (A/B/C/D), x9 −.67 with A.

## F. macro_country.csv (x10..x16)

3,600 rows = 12 countries × 300 month-ends, complete keys, no duplicates. Rows come in natural country order (country_1..country_12, contiguous blocks, dates ascending), which is safe for `groupby().shift()`. Note that `sort_values('country')` reorders them lexicographically (country_1, country_10, …), so always merge on keys, never by position.

| | mean | sd | range | AR1 (med) | AC12 | frac. zero changes | XS-variance share | interpretation |
|---|---|---|---|---|---|---|---|---|
| x10 | 1.85 | 1.96 | −0.82..7.23 | .985 | .81 | **.92 (annual, changes only in January)** | .42 | short rate, **calendar-year average** (see look-ahead below) |
| x11 | −0.24 | 2.10 | −14.28..6.47 | .973 | .48 | 0 | .36 | **ex-post real short rate ≈ x86 − x12** |
| x12 | 1.99 | 1.82 | −2.56..14.53 | .968 | .31 | .003 | .46 | CPI inflation YoY % (all peaks in 2022–23) |
| x13 | −0.18 | 5.52 | −24.0..31.0 | .936 | −.10 | 0 | .97 | YoY % change of effective exchange rate. Extremes: c11 +19.3% in 2011-08, c9 −24% in 2013-05, c5 −20.6% in 2008-12. |
| x14 | 142.7 | 92.0 | 16.6..849.2 | .786 | .43 | 0 | .62 | policy-uncertainty index (EPU-type) |
| x15 | 0.46 | 0.82 | 0.004..13.23 | .583 | .18 | 0 | .99 | geopolitical-risk index (c7 13.2, c5 6.0, c3 1.7, c9 1.2, all peaking in 2001-09) |
| x16 | 81.3 | 5.46 | 65.5..93.5 | .959 | .62 | .38 | .92 | country risk rating in 0.25 steps (ICRG-type) |

**Identical series across countries.** x10 is identical for **countries 2, 4, 6, 8** (a shared currency area; x86 is also identical for c2 and c6), and x14 is identical for **countries 10 and 11**. Cross-country ranks will have ties.

**Missing values.** Only **x11**, 99 NaN in total. No leading gaps and **no interior holes**; the series simply stops early:

| country | first | last non-missing | trailing missing |
|---|---|---|---|
| country_12 | 2000-01-31 | **2020-10-31** | 50 |
| country_9 | 2000-01-31 | **2021-08-31** | 40 |
| country_11 | 2000-01-31 | **2024-03-31** | 9 |

**Stop = regime change** (x11 rebuilt as x86 − x12):
- **country_12**: last x11 −0.18. Inflation then rises 0.28 → 12.34% (2022-12) and the rebuilt real rate falls to **−10.32 (2022-12)**. A stale forward-fill would be wrong by 3.67 pp on average and **10.1 pp at worst**.
- **country_9**: last x11 0.37, rebuilt minimum −4.39 (2023-01). Error mean 2.74 pp, max 4.76 pp.
- **country_11**: benign. Error mean 0.21 pp, max 0.39 pp.

All other countries hit their x11 minimum in 2022 (−3.7 to −14.3), so this is the 2021–23 inflation shock.

**x11 rebuild quality** (overlap period, no fitting). x11 vs x86 − x12: pooled corr **0.988**, RMSE 0.354, median |err| 0.128. This beats every lag variant. Country_7 is exact (RMSE 0.010). By country, RMSE / corr:

| country | RMSE | corr |
|---|---|---|
| c12 | 0.78 | 0.871 (weakest) |
| c9 | 0.20 | 0.992 |
| c11 | 0.33 | 0.972 |
| others | 0.22–0.42 | ≥ 0.991 |

A per-country OLS x11 ~ a + b·x86 + c·x12 fitted on pre-stop data gives R² 0.786 (c12, RMSE 0.64), 0.995 (c9, 0.068) and 0.944 (c11, 0.285). An exhaustive search over pairs of complete columns found nothing better. Do **not** use x10 − x12 (corr 0.975) because x10 carries look-ahead.

**⚠ x10 LOOK-AHEAD (curated file).** x10 is constant within each calendar year and equals the **same-year mean of the monthly short rate x86** exactly in 208/300 country-years. It is within 0.10 pp in 84% of country-years, and closer to the same-year mean than to the prior-year mean in 91%. Pooled corr is 0.998 with the same-year mean against 0.839 with the prior-year mean.

Example, country_7 in 2022: x10 = 2.228 in every month, but x86 was 0.08 in January and 4.10 in December.

A one-month lag does not fix this. You need to use the prior year's value (at least a 12-month lag, i.e. each month of year Y uses x10 of Y−1) or use x86 instead. The same applies to its duplicate x73.

## G. macro_country_extended.csv (x17..x164)

- Shape 3,600 × 150: 148 float columns x17..x164 (contiguous) plus country and date. No string-typed numbers. The same 300 dates and 12 countries, with row order identical to the curated file after sorting.
- Magnitudes reach **1e11–1e12** (x40–x48, x80, x81) and 1e7 (x123–x126). These are levels in currency units and must be scaled or transformed.
- **Frequency** (148 columns): 60 monthly, **39 annual (all change only in January)**, **26 quarterly (change in Jan/Apr/Jul/Oct)**, 22 irregular (ratings or step series), and 1 constant (x148).

**The 7 gap columns.** All stop early, with **no interior holes and no leading gaps**:

| col | freq | countries affected: last obs (trailing NaN) | complete for |
|---|---|---|---|
| x55 | quarterly | c2, c5, c11: 2024-10 (2) | other 9 |
| x82, x83 | annual | c7, c9: 2023-04 (20); other 10 countries: 2024-04 (8) | none |
| x113 | monthly | c4, c10, c11, c12: 2022-11 (25); c1, c2, c3, c5, c6, c7, c8, c9: 2024-01 (11) | none |
| x163, x164 | monthly | same as x113 (they are 3- and 6-month diffs of x113) | none |
| x136 | monthly | c2, c5, c8: 2024-07 (5); c3: 2024-06 (6); c4, c6, c10, c12: 2024-04 (8) | c1, c7, c9, c11 |

**Exact duplicates of curated columns** (max |diff| = 0.0 on all 3,600 rows, identical NaN pattern):

| curated | extended copy |
|---|---|
| x10 | **x73** |
| x12 | **x89** |
| x13 | **x114** |
| x14 | **x17** |
| x15 | **x92** |
| x16 | **x18** |

x11 has no duplicate.

**Global series hidden in the extended file.** 26 columns are identical across all 12 countries every month: x88, x91, x93–x109, x137–x140, x143, x148, x160. Of these:
- x88 = x6, x104 = x7, x109 = x8 and x99 = x9, all exact (max |diff| 0).
- **x148 is constant** (0.512821): drop it.
- **x160 = −x143 exactly.**

**Exact pre-computed transforms** (within country, tolerance 1e-4):
- x127, x128, x129, x130 = 1, 3, 6, 12-month changes of **x86** (monthly short rate)
- x131..x134 = the same changes of **x85** (monthly long yield)
- x137..x140 = the same changes of x88 (= x6)
- x135 = Δ12 x12
- x149 = Δ3 x90
- x163 = Δ3 x113, x164 = Δ6 x113
- x116 = Δ12 x112
- x157 = Δ12 x156

**Near-duplicates** (|corr| > 0.999, not exact): x56~x57 (.9992), x56~x58 (.9996), x57~x58 (.9997), x106~x107 (.9991). Other very close families: x103/x104/x105 (≥ .987), x96/x97/x98, x54~x82 .996, x55~x83 .996.

**⚠ Low-frequency look-ahead.** Annual values are stamped from January of the year they describe:
- **x74 = same-year mean of x85** (corr 1.0000, median diff 0; 0.939 with prior year)
- x73 = x10 = same-year mean of x86
- **x71 and x118 ≈ same-year mean of inflation x12** (0.9985 and 0.980, against 0.55 and 0.53 with prior year)
- **x153 ≈ same-year mean of x13** (0.992 vs 0.227)
- x76, x77 and x157 also track same-year means of x135/x136 (0.44–0.63 vs ≈ 0)

Quarterly x90, x147, x149 and x150 track **same-quarter** means of monthly series (0.43–0.73 vs ≈ 0 prior quarter).

Rule of thumb: lag annual series ≥ 12 months (plus release delay) and quarterly series ≥ 3 months. The other annual and quarterly columns cannot be tested this way but share the same stamping, so treat them the same.

## H. Date alignment

asset_panel, asset_returns_wide, macro_global, macro_country and macro_country_extended all have **exactly the same 300 month-end dates**, 2000-01-31..2024-12-31, with no gaps. In the CSVs, dates are ISO strings (`YYYY-MM-DD`) and need `parse_dates`. dj30 is daily int `YYYYMMDD`, a separate universe used only in Problem 2.

## I. Surprises and cautions (summary)

1. **x10/x73 annual averages carry up to 11 months of look-ahead**, and so do other January-stamped annual and quarter-start-stamped columns.
2. x11 stops just before the 2021–23 inflation regime for c12 and c9. It can be rebuilt as x86 − x12 (corr 0.988).
3. x2 and x5 are exact functions of past returns (12-month compounded return; 36-month sd), apart from winsorised extremes and asset_16's 0.004 floor. They overlap with the TSMOM and risk-parity benchmarks.
4. x4 (B) = Δ12 rate differential and x1 (B) ≈ rate differential. These "characteristics" contain country macro.
5. The leading fills are backward fills (look-ahead). 5 of 8 B assets have x1 frozen in 2024-09..12. asset_16's x5 floor affects any 1/x5 weight.
6. Units are mixed: returns and most x's are decimals, macro rates are in percent, x1 for D is in percent while x1 for B is monthly decimal, and extended levels reach 1e12.
7. Shared series across countries: x10 for c2/4/6/8, x14 for c10/c11, x86 for c2 = c6.
8. dj30: RET is rounded to 0.01, the DOW ticker has two PERMNOs, there is one NaN row, and there is an `Unnamed: 0` index column. MrkRet is clean.
9. Ads: 20 duplicate feature rows, and User ID must be excluded.
10. Country files are in natural country order, but a lexicographic sort changes row order. Merge on (country, date) keys, never by position.
