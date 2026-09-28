"""Problem 3, stage 8: leakage audit, headline numbers, deviations."""

MD_AUDIT = r"""## 3.8 Leakage audit

These are the five questions of L5 p.58 and the four traps of AI guide §4, and where each is guarded in this notebook. The first and third L5 items are graded as fatal.

| risk | where it could enter | how it is prevented | executable check |
|---|---|---|---|
| **a transformation fitted before the split** (FATAL; L5 p.58 #1; AI guide §4b) | scalers, imputation, PCA, z-scores, winsorising | `StandardScaler` and PCA are fitted inside every fit window and every validation fold. Ranks use a single date. Global and country z-scores are trailing. Nothing is imputed from data: the only fills are fixed constants (the level features are 0 until 24 months of history exist; one month of the x10 variant is 0). The clip bounds are fixed constants. | the truncation causality test (Section 3.2); fit, validation and test masks asserted at every refit; unit tests (Section 3.4) |
| a merge on period end instead of publication date (L5 p.58 #2) | macro used contemporaneously; `x10`'s same-year averages | 2-month lag on every macro series; `x10` replaced by `x86`; annual/quarterly columns unused | lag spot checks; `x10` audit (Figure 3.1a); leak demonstration (Table 3.22) |
| **a random fold on time-ordered data** (FATAL; L5 p.58 #3; AI guide §4a) | `train_test_split`, `KFold`, `*CV` estimators, shuffled early stopping | one splitter (`window`, `inner_folds`), date masks only | `train.max() < validation.min()` and `window.max() < test.min()` asserted at every fold and refit |
| a threshold or hyper-parameter chosen on the test block (L5 p.58 #4) | penalties, number of components, forest settings, portfolio rules, cost | tuned on validation years inside the training window; forest untuned; rules and costs fixed in Section 3.0 | refit log (Table 3.14); the test rows never enter `run_linear`'s tuning loop |
| a benchmark changed after the fact (L5 p.58 #5) | switching trailing means | pooled trailing mean fixed in Section 3.0; all others always shown | benchmark unit tests (Section 3.3) |
| `r2_score` read as $R^2_{OOS}$ (AI guide §4d) | sklearn's default scoring | own `r2_oos`; explicit `neg_mean_squared_error` in permutation importance | scorer unit tests |
| `LogisticRegression()` called plain (AI guide §4c) | none | no classifier in Problem 3 | none needed |
| look-ahead in characteristics | x2/x5 pre-lagged, x1/x3/x4 contemporaneous, back-fills | lag only x1/x3/x4; back-filled cells set to missing; first target after the last fill | identities x2 = R12, x5 = SD36; fill table (Section 3.1) |
| snooping in the exploratory analysis | predictor–return statistics on 2011–2024 | tercile sorts on 2003–2010 only | assertion in the sort cell |
| too good to be true (L5 p.57) | any leak | alarm on any $R^2_{OOS}$ above 2% | Section 3.5 (not triggered) |"""

H1 = r'''n_refits = sum(len(LOGS[n]) for n in ledger.index)
n_of = lambda models: sum(len(LOGS[n]) for n in ledger.index if ledger.loc[n, 'model'] in models)
audit = pd.Series({
    'model refits, each with train < test asserted': n_refits,
    'linear refits tuned on 3 forward validation folds, order asserted (ridge, lasso, PCR, per-class ridge)': n_of(('ridge', 'lasso', 'pcr', 'ridge-per-class')),
    'linear refits with nothing to tune (OLS, PCR with K fixed)': n_of(('ols', 'pcr-K3', 'pcr-K5')),
    'forest refits (untuned)': n_of(('rf', 'rf-deep')),
    'lag spot checks passed': 20,
    'causality test (rebuild from data truncated at 2010-12): identical features': True,
    'unit tests of the harness passed': True,
    'largest R2_OOS of the 38 specifications (alarm at 2%)': f"{res_tab[['R2 vs pooled mean', 'R2 vs zero']].max().max():.2%}",
    'tercile sorts use months up to': str(DATES[INIT_END_M].date())}, name='check')
display(audit.to_frame())'''

H2 = r'''# ---------- Every number quoted in the write-up, printed from the objects above ----------
best = res_tab['R2 vs pooled mean'].idxmax()
b_tab = res_tab.loc[best]
def edge_share(n):
    return (LOGS[n]['choice'] == GRIDS['ridge'][0]).mean()
headline = pd.Series({
    'specifications in the ledger': len(ledger), 'Bonferroni critical |t| for 38 tests': round(bonf38, 2),
    'OOS months / asset-months': f'{T_OOS} / {len(OOS):,}',
    'mean of y: training block / OOS': f'{F.loc[F.m <= INIT_END_M, "y"].mean():.3f} / {OOS.y.mean():.3f}',
    'R2_OOS of the zero forecast vs the pooled trailing mean': f'{r2_oos(OOS.y, ZERO, OOS.b_pool):.2%}',
    'Q1: R2_OOS ridge M2 expanding vs pooled mean (SE)': f'{q1:.3%} ({q1_se:.3%})',
    'Q1: same, vs zero (SE)': '{:.3%} ({:.3%})'.format(*r2_boot(RES[P], ZERO)),
    'Q1: class means alone (M1) vs pooled mean (SE)': f'{m1:.3%} ({m1_se:.3%})',
    'Q1: characteristics beyond class means (SE)': f'{d_char:.3%} ({d_char_se:.3%})',
    'Q1: pre-registered verdict': 'YES' if q1_verdict else 'NO',
    'Q1: characteristics beyond class means, +/- 2 SE': f'{d_char - 2 * d_char_se:.3%} to {d_char + 2 * d_char_se:.3%}',
    'Q1: true R2_OOS needed to pass the rule at 50% / 80% power': f'{q1_mde:.2%} / {q1_mde80:.2%}',
    'R2_OOS vs the pooled mean, range over the 38 specifications': f"{res_tab['R2 vs pooled mean'].min():.2%} .. {res_tab['R2 vs pooled mean'].max():.2%}",
    'specifications at or above -1% vs the pooled mean, and their lowest value': f"{(res_tab['R2 vs pooled mean'] >= -0.01).sum()}, {res_tab.loc[res_tab['R2 vs pooled mean'] >= -0.01, 'R2 vs pooled mean'].min():.2%}",
    'ridge M2: static minus expanding, paired (SE, t)': f"{sch_tab.loc[('ridge M2', 'static - expanding'), 'difference in R2_OOS']:.3%} "
                                                        f"({sch_tab.loc[('ridge M2', 'static - expanding'), 'paired SE']:.3%}, t = {sch_tab.loc[('ridge M2', 'static - expanding'), 'difference / SE']:.2f})",
    'paired t, static minus expanding: OLS M2 / RF M2 / PCR M3': ' / '.join(f"{sch_tab.loc[(k, 'static - expanding'), 'difference / SE']:.2f}" for k in ['ols M2', 'rf M2', 'pcr M3']),
    'vs the per-asset trailing mean of y: ridge M2 / class means M1 (SE, t)': ' / '.join(f"{pa.loc[n, 'R2 vs per-asset mean']:.2%} ({pa.loc[n, 'SE']:.3%}, t = {pa.loc[n, 'R2 / SE']:.2f})"
                                                                                  for n in ['ridge|M2|expanding', 'ols|M1|expanding']),
    'specifications with R2_OOS > 0 vs the per-asset mean; the per-asset mean vs the pooled mean': f"{n_pa} of {len(res_tab)}; {r2_oos(OOS.y, OOS.b_asset, OOS.b_pool):.2%}",
    'ridge M2 with and without the 10 stale-x1 rows': f"{cuts.loc['ridge|M2|expanding', 'all 168 months']:.3%} / {cuts.loc['ridge|M2|expanding', 'without the 10 stale-x1 rows']:.3%}",
    'share of refits where ridge chose the no-signal end: M2 / M3': f"{edge_share('ridge|M2|expanding'):.0%} / {edge_share('ridge|M3|expanding'):.0%}",
    'best of 38 vs pooled mean (SE)': f"{best}: {b_tab['R2 vs pooled mean']:.2%} ({b_tab['SE']:.2%})",
    'number of specifications with R2_OOS > 0 vs zero': int((res_tab['R2 vs zero'] > 0).sum()),
    'OLS M3 static / expanding': f"{res_tab.loc['ols|M3|static', 'R2 vs pooled mean']:.1%} / {res_tab.loc['ols|M3|expanding', 'R2 vs pooled mean']:.1%}",
    'deep forest M2 / M3: OOS (mean IS R2)': f"{res_tab.loc['rf-deep|M2|expanding', 'R2 vs pooled mean']:.1%} ({res_tab.loc['rf-deep|M2|expanding', 'mean IS R2']:.0%}) / "
                                          f"{res_tab.loc['rf-deep|M3|expanding', 'R2 vs pooled mean']:.1%} ({res_tab.loc['rf-deep|M3|expanding', 'mean IS R2']:.0%})",
    'Q2: macro gain ridge M3 - M2 expanding (SE)': f'{q2_gain:.3%} ({q2_se:.3%})',
    'Q2: rank of real macro among real + 8 placebos': f'{1 + (plac["gain vs ridge M2"] > q2_gain).sum()} of 9',
    'Q2: placebo gains, min / max': f'{plac["gain vs ridge M2"].min():.3%} / {plac["gain vs ridge M2"].max():.3%}',
    'Q2: pre-registered verdict': 'YES' if q2_verdict else 'NO',
    'country macro: share of variance across countries within a month (classes B-D), min / max': f'{xs_share.min().min():.0%} / {xs_share.max().max():.0%}',
    'placebo: OOS target months with wrapped macro, s = 72 / 120': f"{wrap.loc[72, 'OOS target months with wrapped macro']} / {wrap.loc[120, 'OOS target months with wrapped macro']}",
    'PCR M3 with K = 0 allowed: R2_OOS; refits choosing K = 0': f"{r2_oos(OOS.y, PCR0_YH, BP):.3%}; {(PCR0_LOG['choice'] == 0).sum()} of {len(PCR0_LOG)}",
    'x10 at 14-month lag minus x86 (SE)': f"{rob.loc['curated x10 (14-month lag) vs x86', 'difference']:.3%} ({rob.loc['curated x10 (14-month lag) vs x86', 'SE']:.3%})",
    'x10 leak demonstration, R2_OOS vs pooled mean': f'{r2_oos(OOS.y, LEAK_YH, BP):.3%}',
    'Q3: net Sharpe P1 / EW / RP / TSMOM': ' / '.join(f'{net_sharpe(k, HEADLINE_BP):.2f}' for k in ['P1 ridge M2 (primary)', 'EW', 'RP', 'TSMOM']),
    'Q3: alpha (annual) and t on EW + RP + TSMOM': f"{att['P1 on EW + RP + TSMOM']['alpha (annualised)']:.2%}, t = {att['P1 on EW + RP + TSMOM']['t(alpha)']:.2f}",
    'Q3: net Sharpe P1 minus RP (bootstrap SE)': f"{d_tab.loc['RP', 'net Sharpe P1 - benchmark']:.2f} ({d_tab.loc['RP', 'bootstrap SE']:.2f})",
    'Q3: net Sharpe P1 minus P1 static (bootstrap SE)': f"{d_tab.loc['P1 ridge M2, static (one fit, 2010-12)', 'net Sharpe P1 - benchmark']:.2f} "
                                                        f"({d_tab.loc['P1 ridge M2, static (one fit, 2010-12)', 'bootstrap SE']:.2f})",
    'Q3: P1 on class means alone, net Sharpe': f"{net_sharpe('P1 class means (M1)', HEADLINE_BP):.2f}",
    'Q3: with the class-means tilt as a regressor: beta, R2, alpha t': f"{att['P1 on EW + RP + TSMOM + class-means tilt']['beta M1']:.2f}, {att['P1 on EW + RP + TSMOM + class-means tilt']['R2']:.3f}, "
                                                                     f"{att['P1 on EW + RP + TSMOM + class-means tilt']['t(alpha)']:.2f}",
    'Q3: P3 (within-class long-short) gross Sharpe': f"{g['P3 ridge M2']['Sharpe']:.2f}",
    'Q3: P1 static: alpha and t on EW + RP + TSMOM': f"{att['P1 static (M2) on EW + RP + TSMOM']['alpha (annualised)']:.2%}, t = {att['P1 static (M2) on EW + RP + TSMOM']['t(alpha)']:.2f}",
    'Q3: P3 (within-class long-short) net Sharpe and turnover': f"{net_sharpe('P3 ridge M2', HEADLINE_BP):.2f}, {g['P3 ridge M2']['turnover / month']:.0%} / month",
    'Q3: P1 share of gross in class D (RP)': f"{STRATS['P1 ridge M2 (primary)'].abs().T.groupby(CLS_S).sum().T.mean()['D']:.0%} ({STRATS['RP'].abs().T.groupby(CLS_S).sum().T.mean()['D']:.0%})",
    'Q3: P1 asset_16 average |w|, largest |w|': f"{Wp['asset_16'].abs().mean():.0%}, {Wp.abs().max().max():.0%}",
    'Q3: break-even cost vs EW / RP (bp)': f"{cost_tab.loc['P1 ridge M2 (primary)', 'cost (bp) at which P1-M2 = EW']:.0f} / {cost_tab.loc['P1 ridge M2 (primary)', 'cost (bp) at which P1-M2 = RP']:.0f}",
    'Q3: pre-registered verdict': 'YES' if q3_verdict else 'NO',
    'Q3: net Sharpe of P1 static (2010 tilt frozen) / P1 rolling (not selected)': f"{net_sharpe('P1 ridge M2, static (one fit, 2010-12)', HEADLINE_BP):.2f} / "
                                                                                f"{net_sharpe('P1 ridge M2, rolling (not selected)', HEADLINE_BP):.2f}",
    'Q3: P1 rolling turnover / month': f"{g['P1 ridge M2, rolling (not selected)']['turnover / month']:.1%}",
    'Q3: P1 class means (M1) alpha and t on EW + RP + TSMOM': f"{att['P1 class means (M1) on EW + RP + TSMOM']['alpha (annualised)']:.2%}, t = {att['P1 class means (M1) on EW + RP + TSMOM']['t(alpha)']:.2f}",
    'Q3: avg monthly corr of P1 weights, ridge M2 vs class means M1': f"{np.mean([np.corrcoef(Wp.iloc[i], STRATS['P1 class means (M1)'].iloc[i])[0, 1] for i in range(T_OOS)]):.4f}",
    'Q3: alpha of P1 (t), 2011-2017 / 2018-2024': ' / '.join(f"{v['alpha (annualised)']:.2%} ({v['t(alpha)']:.2f})" for v in sub_att.values()),
    'Q3: net Sharpe 2011-2017 / 2018-2024, EW': f"{subp.loc['EW', '2011-2017']:.2f} / {subp.loc['EW', '2018-2024']:.2f}",
    'Q3: class-D share of P1 by year: first / highest (year) / last': f"{tilt_tab.iloc[0, 2]:.0%} / {tilt_tab.iloc[:, 2].max():.0%} ({tilt_tab.iloc[:, 2].idxmax()}) / {tilt_tab.iloc[-1, 2]:.0%}",
    'Q3: class-means forecast of y for class B, 2011 / 2024': f"{tilt_tab.loc[2011, 'class B: forecast of y']:.3f} / {tilt_tab.loc[2024, 'class B: forecast of y']:.3f}",
    'Q3: class-means forecast of y for class D, 2011 / highest (year) / 2024': f"{tilt_tab.loc[2011, 'class D: forecast of y']:.3f} / {tilt_tab['class D: forecast of y'].max():.3f} "
                                                                             f"({tilt_tab['class D: forecast of y'].idxmax()}) / {tilt_tab.loc[2024, 'class D: forecast of y']:.3f}",
    'C-D correlation 2011-2019 / 2020-2024': f"{div['2011-2019']['C-D (equity-like vs bond-like)']:.2f} / {div['2020-2024']['C-D (equity-like vs bond-like)']:.2f}",
    'P1 net Sharpe 2011-2017 / 2018-2024': f"{subp.loc['P1 ridge M2 (primary)', '2011-2017']:.2f} / {subp.loc['P1 ridge M2 (primary)', '2018-2024']:.2f}",
    # ----- the nine post-hoc additions (P-A ... P-M), approved after the final audit
    'P-A: corr(x6, trailing realised vol of class A / class C); x12 peaks in 2022-23 in n of 12 countries':
        f"{x6_vol['A']:.2f} / {x6_vol['C']:.2f}; {int(peak('x12').dt.year.isin([2022, 2023]).sum())}",
    'P-A: largest share of changes made in January (series used)': f'{jan.max():.0%}',
    'P-M: sd of y across classes; across years (lowest, highest)': f'{sd_cls.iloc[:, 1].min():.2f}-{sd_cls.iloc[:, 1].max():.2f}; '
        f'{sd_year.min():.2f} ({sd_year.idxmin()}), {sd_year.max():.2f} ({sd_year.idxmax()})',
    'P-C: lag-1 autocorrelation of the benchmark losses; block/iid SE ratio, median':
        f"{acf_tab.iloc[0, 0]:.2f}; {ratio.median():.2f}",
    'P-C: t with 12-month blocks: x10 variant / characteristics beyond class means / class means vs per-asset mean':
        ' / '.join(f"{dep_tab.loc[k, 't, blocks of 12']:.2f}" for k in ['curated x10 (14-month lag) minus x86', 'characteristics beyond class means', 'class means vs the per-asset mean']),
    'P-H: forest minus ridge, M2 expanding (SE); largest |t| over the 6 pairs':
        f"{fr_tab.loc[('M2', 'expanding'), 'forest minus ridge']:.3%} ({fr_tab.loc[('M2', 'expanding'), 'paired SE']:.3%}); {fr_tab['difference / SE'].abs().max():.2f}",
    'P-D: best fixed penalty with hindsight, R2_OOS: ridge M2 / ridge M3 / lasso M3 / PCR M3':
        ' / '.join(f"{fixed_tab.loc[k, 'best fixed value with hindsight (upper bound)']:.3%}" for k in fixed_tab.index),
    'P-G: largest share of the characteristics in the loadings of PCR PC1-PC3': f"{pcr_tab[char_cols].max().max():.3f}",
    'P-E: interactions, R2_OOS (SE); gain over ridge M2 (SE); rank among 9': f'{int_r2:.3%} ({int_r2_se:.3%}); {int_gain:.3%} ({int_se:.3%}); '
        f'{1 + int((plac_int > int_gain).sum())} of 9',
    'P-B: flagged months (|r| > 2.5); alpha and t without them': f"{int(flag.sum())}; {12 * fit_wo.params['Intercept']:.2%}, t = {fit_wo.tvalues['Intercept']:.2f}",
    'P-L: t(alpha) with bootstrap SE; corr(EW, RP)': f"{q3_boot.loc['alpha (annualised)', 't, bootstrap']:.2f}; {dfp['EW'].corr(dfp['RP']):.2f}",
    'Problem 3 runtime (minutes)': round((time.perf_counter() - P3_T0) / 60, 1)}, name='value')
display(headline.to_frame().style.set_caption('Table 3.31: the numbers the write-up quotes'))'''

MD_DEV = r"""## 3.10 Deviations from the design in Section 3.0

Every change made after the design cell was written is listed here, together with its effect.

1. **The leak-alarm check.** It was first coded on the absolute value of $R^2_{OOS}$, so large *negative* values set it off. The design (and L5 p.57) means a suspiciously *high* value, so the check was corrected to test the maximum. No result changed: the largest $R^2_{OOS}$ is far below 2%.
2. **The curated-`x10` variant (an extra).** With a 14-month lag, `x10`'s trailing z-score first exists for target 2003-02, so the 50 rows of 2003-01 are set to 0, the neutral value. This affects one training month of one extra specification.
3. **"In-sample fit exceeds out-of-sample fit"** was written as an assertion. It is a sanity expectation, not a guarantee, so it became a printed count. It holds for all 38 specifications.

4. **Diagnostics added after an independent review of the first full run.** They are post hoc. None is a candidate specification, and none changes a pre-registered verdict:
   - the average next-month $y$ in each tercile (Table 3.6b);
   - the power of the actual Q1 statistic (Section 3.5) and the per-asset benchmark summary (Table 3.10c);
   - paired scheme differences (Table 3.12b);
   - the share of country-macro variance that lies across countries (Table 3.18b);
   - where the placebo shift wraps (Table 3.19b), and PCR with $K = 0$ allowed (Section 3.6). The pre-registered PCR grid starts at $K = 1$, so it could not reach the class-means model that ridge and lasso can reach;
   - P1 on the static and rolling ridge M2 forecasts, the regression of P1 on the rules by sub-period (Table 3.27b) and the class tilt year by year (Table 3.28b).
5. **Descriptions corrected, computations unchanged.**
   - Turnover is $\sum_i|\Delta w_i|$, buys plus sells. The design cell's "one-way turnover" means this sum, not half of it, so the cost is conservative under the half-sum convention.
   - The class risk budget uses the RP sleeve's volatility for P1 too.
   - The leaf of 200 is flagged as a departure from L8 p.44.
   - Several lecture citations were tightened, and references to exam cells now name the exam's sections.
   - A unit test that could not fail was removed. The property it named is asserted inside the harness.
6. **Timing evidence for the design.** The post-hoc diagnostics of item 4 re-use the out-of-sample window that the design meant to score once; they change no verdict. The design cells were written in the development scripts before the ledger was run. Version control does not prove that order: the design first appears in a commit (591c8ff) made after the development ledger had run, together with the analysis cells.
7. **Corrections from a final audit against the AI Coding Guide and the eight lectures.** No computation changed:
   - The reason for leaving out the stored `x11` is its gaps and near-redundancy with `x86 − x12`, not exact collinearity.
   - Section 3.4 now states why boosted trees were not run (runtime, not leakage), the forest's inputs, that PCR's PCA is taken on standardised inputs, and that the month bootstrap holds the forecasts fixed and treats months as independent.
   - Several citations were corrected (L8 p.54 for "the forest may beat ridge", L8 p.57 for permutation importance, L1 p.43–44 for Table 3.30, L3 p.11–12 for leverage), and the harness cell was split into three, one step per cell (AI guide §6.2).
   - Printed additions that support corrected sentences: the ±2 SE interval of the characteristics-beyond-class-means difference, the Sharpe difference between P1 and its frozen version (Table 3.29), and the library versions (Section 3.0).
8. **Nine post-hoc additions, approved by the student after the final audit.** Each is labelled "post hoc" in its cell. None changes a pre-registered verdict:
   - P-A: the macro inputs described (Table 3.5b); P-M: the spread of the target by class and year (Tables 3.6c–d) and the decile numbers behind Figure 3.6b (Table 3.17b);
   - P-C: serial dependence of the monthly losses and a moving-block bootstrap, our construction (Tables 3.12c–d); P-H: forest minus ridge, paired (Table 3.12e);
   - P-D: R²_OOS at every fixed penalty, an upper bound chosen with hindsight, and the validation curves (Table 3.14c, Figures 3.5c–d); P-G: what PCR's components load on (Table 3.14d, Figure 3.5e);
   - P-E: characteristic × macro interactions with their own placebos (Table 3.19c). This is a 39th specification; the Bonferroni value becomes 3.28;
   - P-B and P-L: diagnostics and a bootstrap SE for the Q3 α regression (Tables 3.27c–e, Figure 3.9).

Nothing else changed: windows, target, benchmarks, primary tests, decision rules, grids, forest settings, placebo shifts and cost are exactly as fixed in Section 3.0."""

MD_HEAD = r"""## 3.9 Headline numbers

Every number the write-up quotes is printed in the table below (cell 1: every number comes from a cell)."""

CELLS_H = [('md', MD_AUDIT), ('code', H1), ('md', MD_HEAD), ('code', H2), ('md', MD_DEV)]
