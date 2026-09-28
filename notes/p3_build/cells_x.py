"""The nine post-hoc additions approved by the student after the final audit (P-A, P-B, P-C, P-D, P-E, P-G, P-H, P-L, P-M).
None is a candidate in the pre-registered ledger except P-E, which is a labelled post-hoc 39th specification."""

# ---------------------------------------------------------------------------------------------------------- §3.1, after A9
A9M = r'''# ---------- The macro inputs used: scale, persistence, update frequency, and what each probably measures ----------
# Post hoc (P-A, added after the final audit at the student's request). Predictors alone, and their relation to PAST returns
# only: no link to the returns they will forecast.
def describe_macro(W):
    """W: date x country (or one column) for one series. Pooled scale, peak and trough of the cross-country mean,
    persistence, and how often and in which calendar months the values change."""
    v = W.stack()
    path = W.mean(axis=1)
    ch = W.diff().abs().iloc[1:] > 1e-12
    months_changed = np.repeat(ch.index.month.to_numpy()[:, None], ch.shape[1], axis=1)[ch.to_numpy()]
    return {'mean': v.mean(), 'sd': v.std(ddof=1), 'min': v.min(), 'max': v.max(),
            'peak (cross-country mean)': path.idxmax().strftime('%Y-%m'), 'trough': path.idxmin().strftime('%Y-%m'),
            'AC(1) of the level': np.nanmean([W[c].autocorr(1) for c in W]),
            'share of months with a change': ch.to_numpy().mean(),
            'share of changes made in January': (months_changed == 1).mean()}
MAC_G_LEVELS = mac_g.set_index('date')
macro_desc = {k: describe_macro(MAC_G_LEVELS[[k]]) for k in ['x6', 'x7', 'x8', 'x9']}           # the global series used
for k, src in [('x86', mac_x), ('x12', mac_c), ('x13', mac_c), ('x14', mac_c), ('x15', mac_c), ('x16', mac_c)]:
    macro_desc[k] = describe_macro(src.pivot(index='date', columns='country', values=k))  # the country series used
MEANING = {'x6': 'implied-volatility index (VIX-type)', 'x7': 'smoothed stress or uncertainty index (a trailing construct)',
           'x8': 'standardised activity or sentiment indicator', 'x9': 'pro-cyclical level in decimals (yield or growth type)',
           'x86': 'short-term interest rate, %', 'x12': 'CPI inflation, year on year, %',
           'x13': 'year-on-year % change of the effective exchange rate', 'x14': 'economic-policy-uncertainty index',
           'x15': 'geopolitical-risk index', 'x16': 'country risk rating (steps of 0.25)'}
mtab = pd.DataFrame(macro_desc).T
mtab['what it probably measures (our reading)'] = pd.Series(MEANING)
display(mtab.style.format({**{c: '{:.3g}' for c in ['mean', 'sd', 'min', 'max']}, 'AC(1) of the level': '{:.3f}',
                           'share of months with a change': '{:.0%}', 'share of changes made in January': '{:.0%}'})
        .set_caption('Table 3.5b: the ten macro inputs used, 2000-2024 (country series pooled over the 12 countries)'))
# the evidence behind the readings (series alone, or against past returns only)
rv12 = class_ret.rolling(12, min_periods=12).std(ddof=1)                # trailing realised volatility, months t-11..t
x6_vol = {c: MAC_G_LEVELS.x6.corr(rv12[c]) for c in 'ABCD'}
print('corr(x6, trailing 12-month realised volatility of each class average): ' + ', '.join(f'{c} {v:.2f}' for c, v in x6_vol.items()))
peak = lambda k: mac_c.pivot(index='date', columns='country', values=k).idxmax()
print(f"x12 peaks in 2022 or 2023 for {int(peak('x12').dt.year.isin([2022, 2023]).sum())} of 12 countries; "
      f"x15 peaks in 2001-09 for {int((peak('x15').dt.strftime('%Y-%m') == '2001-09').sum())} of 12")
print(f"x6 peaks in {MAC_G_LEVELS.x6.idxmax():%Y-%m}; x8 has mean {MAC_G_LEVELS.x8.mean():.2f} and sd {MAC_G_LEVELS.x8.std(ddof=1):.2f} (a z-score)")
jan = mtab['share of changes made in January'].astype(float)
print(f'largest share of changes made in January among the series used: {jan.max():.0%} (an annual series stamped in '
      'January would show 100%), so every series used updates within the year, unlike x10')'''

# ---------------------------------------------------------------------------------------------------------- §3.1, after A10
A10M = r'''# ---------- Does volatility scaling equalise the variance of the target? (P-M, post hoc) ----------
# Returns alone (no predictor), so the full sample is used, as for the other return statistics in this section.
yl = (R / SD36).loc[DATES[FIRST_M]:].stack().rename('y').reset_index()
yl.columns = ['date', 'asset_id', 'y']
yl['r'] = R.loc[DATES[FIRST_M]:].stack().to_numpy()
yl['cls'] = yl.asset_id.map(CLS_S)
assert yl.y.notna().all() and len(yl) == 50 * (len(DATES) - FIRST_M)
sd_cls = yl.groupby('cls')[['r', 'y']].std(ddof=1)
sd_cls.columns = ['sd of the raw monthly excess return', 'sd of y = r / sigma-hat']
display(sd_cls.style.format('{:.3f}').set_caption('Table 3.6c: spread of the raw return and of the target by class, 2003-2024'))
sd_year = yl.groupby(yl.date.dt.year).y.std(ddof=1)
display(sd_year.to_frame('sd of y').T.style.format('{:.2f}').set_caption('Table 3.6d: sd of the target by year'))
print(f'sd of y: {sd_cls.iloc[:, 1].min():.2f}-{sd_cls.iloc[:, 1].max():.2f} across classes, '
      f'{sd_year.min():.2f} ({sd_year.idxmin()}) to {sd_year.max():.2f} ({sd_year.idxmax()}) across years')'''

# ---------------------------------------------------------------------------------------------------------- §3.5, after E3
E3B = r'''# ---------- Are the months independent? Serial dependence behind the month bootstrap (P-C, post hoc) ----------
BPOOL, BASSET = OOS.b_pool.to_numpy(), OOS.b_asset.to_numpy()
M1Y, M2Y, M3Y = RES['ols|M1|expanding'], RES['ridge|M2|expanding'], RES['ridge|M3|expanding']
def sse_m(v):
    return month_sums((OOS.y - v) ** 2)
loss_series = {'benchmark losses (pooled mean)': sse_m(BPOOL),
               'Q1: pooled mean minus ridge M2': sse_m(BPOOL) - sse_m(M2Y),
               'characteristics: class means minus ridge M2': sse_m(M1Y) - sse_m(M2Y),
               'Q2: ridge M2 minus ridge M3': sse_m(M2Y) - sse_m(M3Y)}
acf_tab = pd.DataFrame({k: {f'lag {L}': pd.Series(v).autocorr(L) for L in range(1, 7)} for k, v in loss_series.items()}).T
display(acf_tab.style.format('{:.2f}').set_caption('Table 3.12c: autocorrelation of the monthly loss series (168 months)'))

def block_idx(L, B=BOOT_B, seed=P3_SEED + 1):
    """Circular moving-block bootstrap of the 168 OOS months (our construction; not taught): blocks of L consecutive months."""
    rng = np.random.default_rng(seed)
    nblk = int(np.ceil(T_OOS / L))
    starts = rng.integers(0, T_OOS, size=(B, nblk))
    return ((starts[:, :, None] + np.arange(L)) % T_OOS).reshape(B, -1)[:, :T_OOS]
BLOCKS = {L: block_idx(L) for L in (6, 12)}
def delta_se(idx, yh1, yh2, bench):
    """SE of R2(yh1) - R2(yh2) against bench under resampling scheme idx (yh2 = bench gives the SE of R2(yh1) itself)."""
    e1, e2, eb = sse_m(yh1), sse_m(yh2), sse_m(bench)
    return ((e2[idx].sum(1) - e1[idx].sum(1)) / eb[idx].sum(1)).std(ddof=1)
checks = {'Q1: ridge M2 vs the pooled mean': (M2Y, BPOOL, BPOOL),
          'class means vs the pooled mean': (M1Y, BPOOL, BPOOL),
          'characteristics beyond class means': (M2Y, M1Y, BPOOL),
          'Q2: ridge M3 minus ridge M2': (M3Y, M2Y, BPOOL),
          'class means vs the per-asset mean': (M1Y, BASSET, BASSET),
          'curated x10 (14-month lag) minus x86': (RES['ridge|M3-x10|expanding'], M3Y, BPOOL),
          'OLS M2: static minus expanding': (RES['ols|M2|static'], RES['ols|M2|expanding'], BPOOL),
          'forest M2: static minus expanding': (RES['rf|M2|static'], RES['rf|M2|expanding'], BPOOL)}
dep = {}
for lab, (a_, b_, c_) in checks.items():
    est = delta_boot(a_, b_, c_)
    row = {'estimate': est[0], 'SE, months independent': est[1]}
    for L, idx in BLOCKS.items():
        row[f'SE, blocks of {L} months'] = delta_se(idx, a_, b_, c_)
    row['t, months independent'] = est[0] / est[1]
    row['t, blocks of 12'] = est[0] / row['SE, blocks of 12 months']
    dep[lab] = row
dep_tab = pd.DataFrame(dep).T
display(dep_tab.style.format({c: '{:.3%}' for c in dep_tab.columns[:4]} | {c: '{:.2f}' for c in dep_tab.columns[4:]})
        .set_caption('Table 3.12d: the month bootstrap against a moving-block bootstrap (our construction), same B = 10,000'))
ratio = dep_tab['SE, blocks of 12 months'] / dep_tab['SE, months independent']
print(f'block (12-month) SE / independent-month SE: {ratio.min():.2f} to {ratio.max():.2f} (median {ratio.median():.2f})')'''

E3C = r'''# ---------- Forest minus ridge on the same features, scheme and months: any exploitable nonlinearity? (P-H, post hoc) ----------
fr_rows = {}
for fs in ['M2', 'M3']:
    for sch in SCHEMES:
        d, se = delta_boot(RES[f'rf|{fs}|{sch}'], RES[f'ridge|{fs}|{sch}'], BPOOL)
        fr_rows[(fs, sch)] = {'forest minus ridge': d, 'paired SE': se, 'difference / SE': d / se}
fr_tab = pd.DataFrame(fr_rows).T
fr_tab.index.names = ['features', 'scheme']
display(fr_tab.style.format({'forest minus ridge': '{:.3%}', 'paired SE': '{:.3%}', 'difference / SE': '{:.2f}'})
        .set_caption('Table 3.12e: random forest (leaf 200) minus ridge, R2_OOS vs the pooled trailing mean'))'''

# ---------------------------------------------------------------------------------------------------------- §3.5, after E4
E4B = r'''# ---------- Does the "no" depend on tuning? R2_OOS at every fixed penalty, and the validation curves (P-D, post hoc) ----------
# Every grid value is scored on the test months as if it had been fixed in advance. The best of these is an UPPER BOUND
# chosen with hindsight: it is never used to choose anything (L5 p.58 item 4).
def path_forecasts(kind, cols, grid):
    """Expanding scheme. At every refit, fit the whole grid on the window, forecast the next 12 months for every value,
    and keep the mean validation MSE over the 3 forward folds (the curve the tuning minimised)."""
    X, y, k, m = F[cols].to_numpy(), F.y.to_numpy(), F.k.to_numpy(), F.m.to_numpy()
    pred, curves = np.full((len(F), len(grid)), np.nan), []
    for b in REFIT_ENDS:
        a, b = window('expanding', b)
        mse = np.zeros(len(grid))
        for (f0, f1), (v0, v1) in inner_folds(a, b):
            fit, val = (m >= f0) & (m <= f1), (m >= v0) & (m <= v1)
            assert m[fit].max() < m[val].min()
            mse += ((y[val][:, None] - LinearFE(kind, grid).fit(X[fit], y[fit], k[fit]).predict(X[val], k[val])) ** 2).mean(0) / 3
        curves.append(mse)
        win, test = (m >= a) & (m <= b), (m > b) & (m <= b + 12)
        assert m[win].max() < m[test].min()
        pred[test] = LinearFE(kind, grid).fit(X[win], y[win], k[win]).predict(X[test], k[test])
    return pred[F.m.to_numpy() >= OOS_M[0]], np.array(curves)

yo, om = OOS.y.to_numpy(), OOS.m.to_numpy()
den = ((yo - BPOOL) ** 2).sum()
fixed_rows, PATHS = {}, {}
for lab, kind, fs, grid, tuned in [('ridge M2', 'ridge', 'M2', GRIDS['ridge'], 'ridge|M2|expanding'),
                                   ('ridge M3', 'ridge', 'M3', GRIDS['ridge'], 'ridge|M3|expanding'),
                                   ('lasso M3', 'lasso', 'M3', GRIDS['lasso'], 'lasso|M3|expanding'),
                                   ('PCR M3 with K = 0..10', 'pcr', 'M3', [0] + list(GRIDS['pcr']), None)]:
    P_, C_ = path_forecasts(kind, FSETS[fs], grid)
    r2 = 1 - ((yo[:, None] - P_) ** 2).sum(0) / den
    choice = C_.argmin(1)                                   # the harness's rule: the first minimum = the heavier penalty
    rebuilt = np.empty(len(OOS))
    for i, b in enumerate(REFIT_ENDS):
        sel = (om > b) & (om <= b + 12)
        rebuilt[sel] = P_[sel, choice[i]]
    if tuned:                                               # the curves reproduce the ledger's own choice at every refit
        assert np.allclose(np.asarray(grid, float)[choice], LOGS[tuned]['choice'].astype(float)), lab
        if kind == 'ridge':                                 # closed form: the forecasts match exactly too (lasso's differ
            assert np.allclose(rebuilt, RES[tuned], atol=1e-10), lab   # only at solver tolerance, cold vs warm start)
    PATHS[lab] = (np.asarray(grid, float), r2, C_)
    jb = int(np.argmax(r2))
    fixed_rows[lab] = {'tuned on the validation folds (as run)': 1 - ((yo - rebuilt) ** 2).sum() / den,
                       'best fixed value with hindsight (upper bound)': r2[jb], 'that value': grid[jb],
                       'first grid value (class means only for ridge, lasso, PCR K = 0)': r2[0], 'last grid value': r2[-1],
                       'median validation gain of the best value over the first (%)': np.median(1 - C_.min(1) / C_[:, 0]) * 100}
fixed_tab = pd.DataFrame(fixed_rows).T
display(fixed_tab.style.format({c: '{:.3%}' for c in fixed_tab.columns if c not in ('that value', fixed_tab.columns[-1])}
                               | {'that value': '{:.3g}', fixed_tab.columns[-1]: '{:.3f}'})
        .set_caption('Table 3.14c: R2_OOS vs the pooled trailing mean at every fixed grid value (expanding, 2011-2024)'))

grid_r, r2_m2, cur_m2 = PATHS['ridge M2']
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 3.8))
ax1.plot(grid_r, 100 * r2_m2, color=C_BLUE, label='ridge M2, fixed penalty')
ax1.plot(grid_r, 100 * PATHS['ridge M3'][1], color=C_ORANGE, label='ridge M3, fixed penalty')
ax1.axhline(100 * fixed_tab.loc['ridge M2', 'tuned on the validation folds (as run)'], color=C_BLUE, linestyle='--', linewidth=1,
            label='ridge M2 as tuned')
ax1.axhline(0, color=C_GREY, linewidth=1)
ax1.set_xscale('log')
ax1.set_xlabel('penalty alpha / n (log scale; right = class means only)')
ax1.set_ylabel('R2_OOS vs pooled mean (%)')
ax1.set_title('Figure 3.5c: test R2 at every fixed penalty (hindsight)')
ax1.legend(fontsize=8)
for i in range(len(cur_m2)):
    ax2.plot(grid_r, 100 * (cur_m2[i] / cur_m2[i, 0] - 1), color=C_BLUE, alpha=0.35, linewidth=1)
ax2.axhline(0, color=C_GREY, linewidth=1)
ax2.set_xscale('log')
ax2.set_xlabel('penalty alpha / n (log scale)')
ax2.set_ylabel('validation MSE vs class means only (%)')
ax2.set_title('Figure 3.5d: ridge M2 validation curves, one per refit')
plt.tight_layout()
plt.show()'''

E4C = r'''# ---------- What PCR's components are made of (P-G, post hoc) ----------
# The same preprocessing as the harness (within-class demeaning, then a scaler, on the training window only).
blocks3 = {'characteristics': CHARS, 'global macro x class': GLOB, 'country macro x class': CTRY_INT}
Xp, kp, mp, yp = F[FSETS['M3']].to_numpy(), F.k.to_numpy(), F.m.to_numpy(), F.y.to_numpy()
pcr_rows, scree = [], {}
for b in REFIT_ENDS:
    a, b = window('expanding', b)
    win = (mp >= a) & (mp <= b)
    fe = LinearFE('pcr', [1]).fit(Xp[win], yp[win], kp[win])
    _, s_, Vt = np.linalg.svd(fe.sc.transform(Xp[win] - fe.xm[kp[win]]), full_matrices=False)
    share = s_ ** 2 / (s_ ** 2).sum()
    scree[DATES[b].year] = share
    row = {'refit': DATES[b].year, 'PC1 share of variance': share[0], 'PC1-PC3 share of variance': share[:3].sum()}
    for j in range(3):
        for bn, cols in blocks3.items():
            row[f'PC{j + 1}: {bn}'] = (Vt[j, [FSETS['M3'].index(c) for c in cols]] ** 2).sum()
    pcr_rows.append(row)
pcr_tab = pd.DataFrame(pcr_rows).set_index('refit')
display(pcr_tab.style.format('{:.3f}').set_caption("Table 3.14d: PCR's first three components (M3, expanding): share of "
                                                    "variance and share of squared loadings on each block"))
char_cols = [c for c in pcr_tab if c.endswith('characteristics')]
print(f"largest share of the characteristics in the squared loadings of PC1-PC3, over the 14 refits: {pcr_tab[char_cols].max().max():.3f} "
      f"(they are 5 of the 39 standardised columns)")
fig, ax = plt.subplots(figsize=(7, 3.4))
for yr, col in [(min(scree), C_BLUE), (max(scree), C_ORANGE)]:
    ax.plot(np.arange(1, 16), scree[yr][:15], 'o-', markersize=4, color=col, label=f'window ending {yr}')
ax.set_xlabel('principal component')
ax.set_ylabel('share of variance')
ax.set_title('Figure 3.5e: scree plot of the M3 design (L1 p.43)')
ax.legend(fontsize=8)
plt.show()'''

# ---------------------------------------------------------------------------------------------------------- §3.6, after F2
F2B = r'''# ---------- Characteristic x macro-state interactions (P-E: a post-hoc 39th specification, with its own placebos) ----------
# Each characteristic rank times each lagged global macro z-score: a macro channel that varies WITHIN a class-month.
INTER = [f'i_{u}_{g}' for u in CHARS for g in G_SER]
def add_inter(fr):
    fr = fr.copy()
    for u in CHARS:
        for g in G_SER:
            fr[f'i_{u}_{g}'] = fr[u] * fr['g_' + g]
    return fr
F = add_inter(F)
assert F[INTER].notna().all().all()
YH_INT, LOG_INT = run_linear('ridge', CHARS + INTER, 'expanding')
int_r2, int_r2_se = r2_boot(YH_INT, BP)
int_gain, int_se = delta_boot(YH_INT, RES['ridge|M2|expanding'], BP)
plac_int = {}
for s in PLACEBO_SHIFTS:
    yh_s, _ = run_linear('ridge', CHARS + INTER, 'expanding', frame=add_inter(variant(shift=s)))
    plac_int[s] = delta_boot(yh_s, RES['ridge|M2|expanding'], BP)[0]
plac_int = pd.Series(plac_int, name='placebo gain over ridge M2')
bonf39 = stats.t.ppf(1 - 0.025 / 39, T_OOS - 1)
int_tab = pd.Series({'R2_OOS vs pooled mean (SE)': f'{int_r2:.3%} ({int_r2_se:.3%})',
                     'gain over ridge M2 (SE)': f'{int_gain:.3%} ({int_se:.3%})',
                     'rank among real + 8 placebos (1 = best)': f'{1 + int((plac_int > int_gain).sum())} of 9',
                     'placebo gains, min / max': f'{plac_int.min():.3%} / {plac_int.max():.3%}',
                     'refits at the no-signal end of the ridge grid': f"{(LOG_INT['choice'] == GRIDS['ridge'][0]).sum()} of {len(LOG_INT)}",
                     'Bonferroni critical |t| with this 39th test': f'{bonf39:.2f}'}, name='ridge on M2 + 20 interactions, expanding')
display(int_tab.to_frame().style.set_caption('Table 3.19c: characteristic x global-macro interactions (post hoc)'))
print('Rule applied as for Q2 (gain > 2 SE and above every placebo): '
      + ('YES' if (int_gain > 2 * int_se and int_gain > plac_int.max()) else 'NO'))'''

# ---------------------------------------------------------------------------------------------------------- §3.7, after G3
G3B = r'''# ---------- How fragile is the Q3 alpha? Regression diagnostics (L3) and a bootstrap SE (L2 p.80) (P-B, P-L, post hoc) ----------
from statsmodels.stats.outliers_influence import OLSInfluence
fit_q3 = smf.ols('P1 ~ EW + RP + TS', data=dfp).fit()
infl = OLSInfluence(fit_q3)
q3diag = pd.DataFrame({'studentized residual': infl.resid_studentized_external, 'leverage': infl.hat_matrix_diag}, index=dfp.index)
flag = q3diag['studentized residual'].abs() > 2.5                               # the L3 p.16 flag
hi_lev = q3diag.leverage > 3 * q3diag.leverage.mean()
display(q3diag[flag | hi_lev].style.format('{:.3f}')
        .set_caption('Table 3.27c: months with |studentized residual| > 2.5 or leverage above 3 times its mean'))
fit_wo = smf.ols('P1 ~ EW + RP + TS', data=dfp[~flag]).fit()
print(f'alpha, all {len(dfp)} months: {12 * fit_q3.params["Intercept"]:.2%} a year (t = {fit_q3.tvalues["Intercept"]:.2f}); '
      f'without the {int(flag.sum())} flagged months: {12 * fit_wo.params["Intercept"]:.2%} (t = {fit_wo.tvalues["Intercept"]:.2f}).')
print('   Reported only: L3 p.17 allows deleting a point only for a good reason, and the Q3 verdict is fixed by the '
      'pre-registered rule on all months.')
display(dfp[['EW', 'RP', 'TS']].corr().style.format('{:.2f}')
        .set_caption('Table 3.27d: correlations of the three net benchmark returns (why the individual betas are unstable)'))

# bootstrap: resample the 168 months with replacement and refit (L2 p.80), with the same draws as everywhere else
Xq = np.column_stack([np.ones(len(dfp)), dfp[['EW', 'RP', 'TS']].to_numpy()])
yq = dfp['P1'].to_numpy()
bcoef = np.array([np.linalg.lstsq(Xq[ix], yq[ix], rcond=None)[0] for ix in BOOT_IDX])
names_q3 = ['alpha (annualised)', 'beta EW', 'beta RP', 'beta TS']
scale = np.array([12, 1, 1, 1])
q3_boot = pd.DataFrame({'estimate': fit_q3.params.to_numpy() * scale,
                        'textbook SE': fit_q3.bse.to_numpy() * scale,
                        'bootstrap SE (months resampled, refit)': bcoef.std(0, ddof=1) * scale}, index=names_q3)
q3_boot['t, textbook'] = q3_boot['estimate'] / q3_boot['textbook SE']
q3_boot['t, bootstrap'] = q3_boot['estimate'] / q3_boot['bootstrap SE (months resampled, refit)']
display(q3_boot.style.format('{:.4f}').set_caption('Table 3.27e: P1 on EW + RP + TSMOM, textbook and bootstrap standard errors'))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 3.6))
ax1.scatter(fit_q3.fittedvalues, q3diag['studentized residual'], s=12, color=C_BLUE, alpha=0.7)
ax1.scatter(fit_q3.fittedvalues[flag], q3diag['studentized residual'][flag], s=30, color=C_ORANGE, label='|r| > 2.5')
for lvl in (-2.5, 2.5):
    ax1.axhline(lvl, color=C_GREY, linestyle='--', linewidth=1)
ax1.set_xlabel('fitted P1 return')
ax1.set_ylabel('studentized residual')
ax1.set_title('Figure 3.9a: residuals against fitted values (L3 p.8)')
ax1.legend(fontsize=8)
ax2.plot(dfp.index, q3diag['studentized residual'], color=C_BLUE, linewidth=1)
ax2.scatter(dfp.index[flag], q3diag['studentized residual'][flag], s=30, color=C_ORANGE)
ax2.axhline(0, color=C_GREY, linewidth=1)
ax2.set_ylabel('studentized residual')
ax2.set_title('Figure 3.9b: residuals over time')
plt.tight_layout()
plt.show()'''
