"""Problem 3, stage 7: from forecasts to portfolios."""

MD_PORT = r"""## 3.7 From forecasts to portfolios

**Forecast portfolios**, all at unit gross exposure and rebalanced monthly. Each uses the month-$t$ forecast, which is built from information through $t-1$:
- **P1** (primary): $w_{i,t}\propto \hat y_{i,t}/\hat\sigma_{i,t-1}$. Since $\hat r = \hat y\,\hat\sigma$, this is the mean–variance tilt $\hat r/\hat\sigma^2$ with a diagonal covariance. With a constant forecast it collapses exactly to risk parity, so RP is the natural bar (identity checked in Section 3.3).
- **P3**: $w_{i,t}\propto(\hat y_{i,t}-\bar{\hat y}_{\text{class},t})/\hat\sigma_{i,t-1}$, a long–short portfolio within each class that isolates the cross-sectional signal.
- **P1 on the class-means model (M1)**: a diagnostic of what class premia alone do.
- **P1 on ridge M2, static** (one fit at 2010-12) and **rolling**: added after the first review, as diagnostics. The static version holds the forecasts of 2010 fixed, so it separates a frozen class tilt from one that is re-estimated every year. The rolling version shows how much the portfolio depends on the scheme. Neither is a candidate for Q3.
- **Class risk budget** (our choice): each class sleeve is set to unit gross within the class and scaled by the inverse of the trailing 36-month volatility of that class's **RP** sleeve, so every class carries roughly equal ex-ante risk. The same class scales are applied to RP and to P1. The two budgeted versions therefore have identical class shares of gross exposure by construction (Table 3.28); only the weights inside each class differ. Because the sleeve's volatility already includes the correlation inside the class, no covariance matrix is inverted. A 50 × 50 sample covariance is rank-deficient when the window is shorter than 50 months (L1 p.63), as a 36-month window is, and covariance-optimised weights are not taught.

**Costs** (not taught; our assumption): 10 bp per unit of turnover $\sum_i|\Delta w_{i}|$ (buys plus sells), charged to every strategy including the benchmarks. Where one-way turnover is defined as half that sum, this is 20 bp per unit, so it is conservative. We also report a grid from 0 to 50 bp and break-even costs.

**Exposure vs timing.** Net P1 returns are regressed on the three net benchmark returns (α as a performance benchmark, L2 p.49; tests of α and β, L2 p.94–97; multiple regression, L3 p.19–21 and p.35). α is what the rules cannot replicate. The β's are not separate exposures: the benchmark returns are highly correlated, so the individual β's are unstable (L3 p.39–44). The joint R² measures how much of P1 the rules explain."""

G1 = r'''OOS_IDX = DATES[OOS_M]
assert list(OOS.asset_id[:50]) == IDS and (OOS.m.to_numpy().reshape(T_OOS, 50) == OOS_M[:, None]).all()
def yh_wide(name):
    """Forecasts of one specification as a (month x asset) table."""
    return pd.DataFrame(np.asarray(RES[name]).reshape(T_OOS, 50), index=OOS_IDX, columns=IDS)

# class risk budget: each class sleeve (unit gross within the class) scaled by 1 / its trailing 36-month volatility
def within_class_unit(W):
    out = W.copy()
    for c in 'ABCD':
        cols = CLS_S.index[CLS_S == c]
        out[cols] = unit_gross(W[cols])
    return out
sleeve_ret = pd.DataFrame({c: port_ret(unit_gross((1.0 / SIG_W.loc[bench_idx])[CLS_S.index[CLS_S == c]])) for c in 'ABCD'})
sleeve_vol = sleeve_ret.rolling(36, min_periods=36).std(ddof=1).shift(1)          # known at the end of t-1
def class_budget(W):
    scale = (1.0 / sleeve_vol.loc[W.index])[CLS_S[W.columns].to_numpy()].to_numpy()
    return unit_gross(within_class_unit(W) * scale)
assert sleeve_vol.loc[OOS_IDX].notna().all().all()

STRATS = {'EW': W_EW.loc[OOS_IDX], 'RP': W_RP.loc[OOS_IDX], 'TSMOM': W_TS.loc[OOS_IDX],
          'RP, class risk budget': class_budget(1.0 / SIG_W.loc[OOS_IDX]),
          'P1 ridge M2 (primary)': weights_from_forecast(yh_wide('ridge|M2|expanding'), 'P1'),
          'P1 ridge M3': weights_from_forecast(yh_wide('ridge|M3|expanding'), 'P1'),
          'P3 ridge M2': weights_from_forecast(yh_wide('ridge|M2|expanding'), 'P3'),
          'P3 ridge M3': weights_from_forecast(yh_wide('ridge|M3|expanding'), 'P3'),
          'P1 class means (M1)': weights_from_forecast(yh_wide('ols|M1|expanding'), 'P1'),
          'P1 ridge M2, static (one fit, 2010-12)': weights_from_forecast(yh_wide('ridge|M2|static'), 'P1'),
          'P1 ridge M2, rolling (not selected)': weights_from_forecast(yh_wide('ridge|M2|rolling'), 'P1'),
          'P1 ridge M2, class risk budget': class_budget(yh_wide('ridge|M2|expanding') / SIG_W.loc[OOS_IDX])}
for k, W in STRATS.items():
    assert np.allclose(W.abs().sum(axis=1), 1), k
S_RET = {k: port_ret(W) for k, W in STRATS.items()}
S_TO = {k: (BENCH_TO[k].loc[OOS_IDX] if k in BENCH_TO else turnover(W)) for k, W in STRATS.items()}

rows = {}
for k in STRATS:
    rows[(k, 'gross')] = perf(S_RET[k], S_TO[k], OOS_M)
    rows[(k, f'net {HEADLINE_BP} bp')] = perf(S_RET[k], S_TO[k], OOS_M, HEADLINE_BP)
port_tab = pd.DataFrame(rows).T
display(port_tab.style.format('{:.3f}').set_caption('Table 3.24: portfolios, 2011-2024 (168 months)'))
subp = pd.DataFrame({lab: {k: perf(S_RET[k], S_TO[k], np.arange(s0, s1 + 1), HEADLINE_BP)['Sharpe'] for k in STRATS}
                     for lab, (s0, s1) in SUBPERIODS.items()})
display(subp.style.format('{:.2f}').set_caption(f'Table 3.25: net Sharpe ratio ({HEADLINE_BP} bp) by sub-period'))'''

G2 = r'''# ---------- Costs: net Sharpe on a grid, and break-even costs ----------
from scipy.optimize import brentq
def net_sharpe(k, bp):
    return perf(S_RET[k], S_TO[k], OOS_M, bp)['Sharpe']
cost_tab = pd.DataFrame({f'{bp} bp': {k: net_sharpe(k, bp) for k in STRATS} for bp in COST_BP})
g = {k: perf(S_RET[k], S_TO[k], OOS_M) for k in STRATS}
cost_tab['break-even cost (bp): net mean = 0'] = [1e4 * g[k]['ann. mean'] / 12 / g[k]['turnover / month'] for k in STRATS]
def breakeven_vs(k, b, hi=500):
    f = lambda bp: net_sharpe(k, bp) - net_sharpe(b, bp)
    return brentq(f, 0, hi) if f(0) * f(hi) < 0 else np.nan
for b in ['EW', 'RP', 'TSMOM']:
    cost_tab[f'cost (bp) at which P1-M2 = {b}'] = [breakeven_vs('P1 ridge M2 (primary)', b) if k == 'P1 ridge M2 (primary)' else np.nan for k in STRATS]
display(cost_tab.style.format('{:.2f}', na_rep='').set_caption('Table 3.26: net Sharpe ratio against the cost per unit of turnover (sum of |change in w|)'))
print('(a blank break-even means the two net Sharpe curves do not cross between 0 and 500 bp)')

show = ['EW', 'RP', 'TSMOM', 'P1 ridge M2 (primary)', 'P1 ridge M3', 'P3 ridge M2']
cols6 = [C_BLUE, C_ORANGE, C_AQUA, C_YELLOW, '#e87ba4', '#008300']
fig, axes = plt.subplots(1, 3, figsize=(15.5, 4))
for k, col in zip(show, cols6):
    r = S_RET[k] - HEADLINE_BP / 1e4 * S_TO[k].where(S_TO[k].index != OOS_IDX[0], 0.0)
    w = (1 + r).cumprod()
    axes[0].plot(OOS_IDX, np.log(w), color=col, label=k)
    axes[1].plot(OOS_IDX, -(1 - w / w.cummax()), color=col, linewidth=1.3, label=k)
    axes[2].plot(COST_BP, [net_sharpe(k, bp) for bp in COST_BP], marker='o', color=col, label=k)
axes[0].set_title(f'Figure 3.8a: cumulative log return, net of {HEADLINE_BP} bp')
axes[0].set_ylabel('log wealth')
axes[1].set_title('Figure 3.8b: drawdown (net)')
axes[1].set_ylabel('drawdown')
axes[2].set_title('Figure 3.8c: net Sharpe ratio against cost')
axes[2].set_xlabel('cost per unit of turnover, sum of |change in w| (bp)')
axes[2].set_ylabel('annualised Sharpe ratio')
for ax in axes:
    ax.legend(fontsize=7)
plt.tight_layout()
plt.show()'''

G3 = r'''# ---------- Exposure vs timing: regress P1 on the benchmark rules (net of costs) ----------
import statsmodels.formula.api as smf
def net_ret(k):
    to = S_TO[k].copy()
    to.iloc[0] = 0.0
    return S_RET[k] - HEADLINE_BP / 1e4 * to
dfp = pd.DataFrame({'P1': net_ret('P1 ridge M2 (primary)'), 'EW': net_ret('EW'), 'RP': net_ret('RP'), 'TS': net_ret('TSMOM'),
                    'M1': net_ret('P1 class means (M1)'), 'P1_M3': net_ret('P1 ridge M3'),
                    'P1_static': net_ret('P1 ridge M2, static (one fit, 2010-12)')})
att = {}
for lab, formula in [('P1 on EW + RP + TSMOM', 'P1 ~ EW + RP + TS'), ('P1 on RP alone', 'P1 ~ RP'),
                     ('P1 on EW + RP + TSMOM + class-means tilt', 'P1 ~ EW + RP + TS + M1'), ('P1 (M3) on EW + RP + TSMOM', 'P1_M3 ~ EW + RP + TS'),
                     ('P1 class means (M1) on EW + RP + TSMOM', 'M1 ~ EW + RP + TS'), ('P1 static (M2) on EW + RP + TSMOM', 'P1_static ~ EW + RP + TS')]:
    fit = smf.ols(formula, data=dfp).fit()
    att[lab] = {'alpha (annualised)': 12 * fit.params['Intercept'], 't(alpha)': fit.tvalues['Intercept'], 'R2': fit.rsquared,
                **{f'beta {v}': fit.params[v] for v in ['EW', 'RP', 'TS', 'M1'] if v in fit.params}}
att_tab = pd.DataFrame(att).T
display(att_tab.style.format('{:.3f}', na_rep='').set_caption(f'Table 3.27: exposure vs timing, monthly net returns 2011-2024 ({T_OOS} months)'))
fit_rp = smf.ols('P1 ~ RP', data=dfp).fit()
t_beta1 = (fit_rp.params['RP'] - 1) / fit_rp.bse['RP']                            # testing beta = 1 by hand (L2 p.97)
print(f'P1 on RP alone: beta = {fit_rp.params["RP"]:.3f} (SE {fit_rp.bse["RP"]:.3f}); t for beta = 1: {t_beta1:.2f}')
# the same regression within each sub-period (added after the first review)
sub_att = {}
for lab, (s0, s1) in SUBPERIODS.items():
    d_ = dfp.loc[DATES[s0]:DATES[s1]]
    f_ = smf.ols('P1 ~ EW + RP + TS', data=d_).fit()
    sub_att[lab] = {'months': len(d_), 'alpha (annualised)': 12 * f_.params['Intercept'], 't(alpha)': f_.tvalues['Intercept'], 'R2': f_.rsquared}
display(pd.DataFrame(sub_att).T.style.format({'months': '{:.0f}', 'alpha (annualised)': '{:.3%}', 't(alpha)': '{:.2f}', 'R2': '{:.3f}'})
        .set_caption('Table 3.27b: P1 (ridge M2) on EW + RP + TSMOM, by sub-period (net of 10 bp)'))

Wp, Wr, Wt = STRATS['P1 ridge M2 (primary)'], STRATS['RP'], STRATS['TSMOM']
overlap = pd.Series({'avg monthly corr(P1 weights, RP weights)': np.mean([np.corrcoef(Wp.iloc[i], Wr.iloc[i])[0, 1] for i in range(T_OOS)]),
                     'avg monthly corr(P1 weights, TSMOM weights)': np.mean([np.corrcoef(Wp.iloc[i], Wt.iloc[i])[0, 1] for i in range(T_OOS)]),
                     'share of months P1 is net short at least one class': np.mean([(Wp.iloc[i].T.groupby(CLS_S).sum() < 0).any() for i in range(T_OOS)]),
                     'P1 average net exposure (sum of w)': Wp.sum(axis=1).mean(),
                     'P1 largest single |weight|': Wp.abs().max().max(), 'P1 asset_16 average |weight|': Wp['asset_16'].abs().mean()}, name='value')
display(overlap.to_frame().style.format('{:.3f}'))
display(pd.DataFrame({k: STRATS[k].abs().T.groupby(CLS_S).sum().T.mean() for k in ['RP', 'P1 ridge M2 (primary)', 'P1 ridge M2, class risk budget', 'RP, class risk budget']}).T
        .style.format('{:.3f}').set_caption('Table 3.28: average share of gross exposure by class, 2011-2024'))

# is the class tilt fixed or re-estimated? class-D share of gross by year, and the class-means forecast of y by year
yr = OOS_IDX.year
d_share = pd.DataFrame({k: STRATS[k].abs().T.groupby(CLS_S).sum().T['D'].groupby(yr).mean()
                        for k in ['RP', 'P1 class means (M1)', 'P1 ridge M2 (primary)', 'P1 ridge M2, static (one fit, 2010-12)']})
cls_fc = yh_wide('ols|M1|expanding').T.groupby(CLS_S).mean().T.groupby(yr).mean()
cls_fc.columns = [f'class {c}: forecast of y' for c in cls_fc.columns]
tilt_tab = pd.concat([d_share.add_prefix('class-D share: '), cls_fc], axis=1)
tilt_tab.index.name = 'year'
display(tilt_tab.style.format('{:.3f}').set_caption('Table 3.28b: the class tilt year by year (class means re-estimated every December)'))'''

G4 = r'''# ---------- How precise are the Sharpe comparisons? Paired month bootstrap (the same resampled months as before) ----------
def sharpe_b(r):
    a = r.to_numpy()[BOOT_IDX]
    return np.sqrt(12) * a.mean(1) / a.std(1, ddof=1)
sb = {k: sharpe_b(net_ret(k)) for k in STRATS}
p1 = 'P1 ridge M2 (primary)'
d_tab = pd.DataFrame({b: {'net Sharpe P1 - benchmark': net_sharpe(p1, HEADLINE_BP) - net_sharpe(b, HEADLINE_BP),
                          'bootstrap SE': (sb[p1] - sb[b]).std(ddof=1)} for b in ['EW', 'RP', 'TSMOM', 'RP, class risk budget',
                                                                  'P1 ridge M2, static (one fit, 2010-12)']}).T
d_tab['difference / SE'] = d_tab.iloc[:, 0] / d_tab.iloc[:, 1]
display(d_tab.style.format('{:.3f}').set_caption('Table 3.29: Sharpe-ratio differences, P1 (ridge M2) minus each rule and minus its frozen version, net of 10 bp'))

# Q3, pre-registered: higher net Sharpe than all three rules AND a positive alpha with t > 2
sr_p1 = net_sharpe(p1, HEADLINE_BP)
beats = {b: sr_p1 > net_sharpe(b, HEADLINE_BP) for b in ['EW', 'RP', 'TSMOM']}
q3_verdict = all(beats.values()) and att['P1 on EW + RP + TSMOM']['alpha (annualised)'] > 0 and att['P1 on EW + RP + TSMOM']['t(alpha)'] > 2
print(f'Q3: net Sharpe P1 = {sr_p1:.2f}; beats ' + ', '.join(f'{b}: {v}' for b, v in beats.items())
      + f"; alpha t = {att['P1 on EW + RP + TSMOM']['t(alpha)']:.2f} -> pre-registered rule: {'YES' if q3_verdict else 'NO'}")'''

G5 = r'''# ---------- Diversification: does the correlation structure change over time? ----------
periods = {'2003-2010 (training)': ('2003-01-31', '2010-12-31'), '2011-2019': ('2011-01-31', '2019-12-31'), '2020-2024': ('2020-01-31', '2024-12-31')}
div = {}
for lab, (a, b) in periods.items():
    C = R.loc[a:b].corr()
    bt = block_table(C)
    ev = np.linalg.eigvalsh(C.to_numpy())
    div[lab] = {**{f'within {c}': bt.loc[c, c] for c in 'ABCD'},
                'A-C (commodity-like vs equity-like)': bt.loc['A', 'C'], 'C-D (equity-like vs bond-like)': bt.loc['C', 'D'],
                'B-C': bt.loc['B', 'C'], 'first principal component, share of variance': ev.max() / ev.sum()}
display(pd.DataFrame(div).style.format('{:.2f}').set_caption('Table 3.30: average return correlations (L1 p.44) and the first-PC share, by period. First-PC share = largest eigenvalue of the return correlation matrix / 50 (our construction; eigenvalues as in L1 p.43)'))'''

from cells_x import G3B
CELLS_G = [('md', MD_PORT), ('code', G1), ('code', G2), ('code', G3), ('code', G3B), ('code', G4), ('code', G5)]
