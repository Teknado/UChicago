"""Problem 3, stage 5: run the ledger and report forecast results."""

MD_RUN = r"""## 3.5 Running the ledger

All 38 pre-listed specifications run through the harness above, plus one diagnostic: the `x10` leak demonstration, which is never a candidate. The design was fixed in Section 3.0 and is not edited below."""

E1 = r'''t0 = time.perf_counter()
def variant(**kw):
    """The same feature build with one option changed; rows must line up with the base frame."""
    Fv, _ = build_features(panel, mac_g, mac_c, mac_x, **kw)
    assert (Fv[['m', 'asset_id']].to_numpy() == F[['m', 'asset_id']].to_numpy()).all() and np.allclose(Fv.y, F.y)
    assert Fv[model_cols].notna().all().all(), f'missing predictor in variant {kw}'
    Fv['k'] = F.k.to_numpy()
    return Fv
FRAMES = {'lag1': variant(macro_lag=1), 'zscore': variant(char_std='z'),
          'x10': variant(short_rate='x10', sr_lag=14), 'x10-leak': variant(short_rate='x10', sr_lag=1)}
VARIANT_OF = {'M3-lag1': ('lag1', 'M3'), 'M2-zscore': ('zscore', 'M2'), 'M3-x10': ('x10', 'M3'), 'M3-x10-leak': ('x10-leak', 'M3')}

def class_cols(fs, c):
    """The columns that can be non-zero for class c (other classes' macro interactions are zero by construction)."""
    return [col for col in FSETS[fs] if col in CHARS or col in LEVELS or col.endswith('_' + c)]

IMPORTANCE = {}
def run_spec(model, feats, scheme):
    """Dispatch one ledger row to its runner. Returns forecasts for the 8,400 OOS rows and the refit log."""
    frame, fs = (FRAMES[VARIANT_OF[feats][0]], VARIANT_OF[feats][1]) if feats in VARIANT_OF else (None, feats)
    if model in ('ols', 'ridge', 'lasso', 'pcr'):
        return run_linear(model, FSETS[fs], scheme, frame=frame)
    if model in ('pcr-K3', 'pcr-K5'):
        return run_linear('pcr', FSETS[fs], scheme, fixed=int(model[-1]))
    if model in ('rf', 'rf-deep'):
        params = dict(RF_PARAMS, min_samples_leaf=200 if model == 'rf' else 5)
        yh, log, imp = run_rf(RF_SETS[fs], scheme, params, importance=(model == 'rf' and scheme == 'expanding'))
        if imp is not None:
            IMPORTANCE[f'rf|{fs}|expanding'] = imp
        return yh, log
    if model == 'ridge-per-class':
        yh, logs = np.full(len(OOS), np.nan), []
        for c in 'ABCD':
            p, lg = run_linear('ridge', class_cols(fs, c), scheme, rows=(F.cls == c).to_numpy())
            yh[(OOS.cls == c).to_numpy()] = p
            logs.append(lg.assign(cls=c))
        return yh, pd.concat(logs, ignore_index=True)
    raise ValueError(model)

RES, LOGS, RUNTIME = {}, {}, {}
for name, row in ledger.iterrows():
    t1 = time.perf_counter()
    RES[name], LOGS[name] = run_spec(row.model, row.features, row.scheme)
    RUNTIME[name] = time.perf_counter() - t1
    assert RES[name].shape == (len(OOS),) and np.isfinite(RES[name]).all()
LEAK_YH, LEAK_LOG = run_spec('ridge', 'M3-x10-leak', 'expanding')
print(f'{len(RES)} specifications + 1 diagnostic run in {time.perf_counter() - t0:.0f} s '
      f'(slowest: {max(RUNTIME, key=RUNTIME.get)}, {max(RUNTIME.values()):.0f} s)')'''

E2 = r'''# ---------- Q1 and the model x scheme table ----------
ZERO = np.zeros(len(OOS))
rows = []
for name, row in ledger.iterrows():
    yh = RES[name]
    r2p, sep = r2_boot(yh, OOS.b_pool.to_numpy())
    r2z, sez = r2_boot(yh, ZERO)
    rows.append({'model': row.model, 'features': row.features, 'scheme': row.scheme, 'tier': row.tier,
                 'R2 vs pooled mean': r2p, 'SE': sep, 'R2 vs zero': r2z, 'SE (zero)': sez,
                 'R2 vs per-class mean': r2_oos(OOS.y, yh, OOS.b_class), 'R2 vs per-asset mean': r2_oos(OOS.y, yh, OOS.b_asset),
                 'mean IS R2': LOGS[name]['IS R2'].mean()})
res_tab = pd.DataFrame(rows, index=ledger.index)
core = res_tab[res_tab.tier == 'core']
for bn in ['R2 vs pooled mean', 'R2 vs zero']:
    piv = core.pivot_table(index=['model', 'features'], columns='scheme', values=bn, sort=False)[SCHEMES]
    display(piv.style.format('{:.2%}', na_rep='').set_caption(f'Table 3.10: {bn}, core specifications, 2011-2024 (8,400 asset-months)'))
display(res_tab.drop(columns=['model', 'features', 'scheme']).style.format({c: '{:.2%}' for c in res_tab.columns[4:]})
        .set_caption('Table 3.10b: every specification, with month-bootstrap SEs and in-sample R2'))
print(f'largest R2_OOS vs the pooled mean: {res_tab["R2 vs pooled mean"].max():.2%} ({res_tab["R2 vs pooled mean"].idxmax()}); '
      f'vs zero: {res_tab["R2 vs zero"].max():.2%} ({res_tab["R2 vs zero"].idxmax()})')
alarm = res_tab[['R2 vs pooled mean', 'R2 vs zero']].max().max() > LEAK_ALARM     # a suspiciously HIGH R2 (L5 p.57)
print('LEAK ALARM (an R2_OOS above 2% would mean: audit before interpreting):', 'TRIGGERED' if alarm else 'not triggered')
print(f"in-sample R2 exceeds R2_OOS for {(res_tab['mean IS R2'] > res_tab['R2 vs pooled mean']).sum()} of {len(res_tab)} specifications (AI guide 6.4)")

# Q1, pre-registered: ridge on characteristic ranks, expanding window, against the pooled trailing mean
P = 'ridge|M2|expanding'
q1, q1_se = r2_boot(RES[P], OOS.b_pool.to_numpy())
m1, m1_se = r2_boot(RES['ols|M1|expanding'], OOS.b_pool.to_numpy())
d_char, d_char_se = delta_boot(RES[P], RES['ols|M1|expanding'], OOS.b_pool.to_numpy())
q1_verdict = q1 > 0 and q1 - 2 * q1_se > 0
print(f'\nQ1 primary: R2_OOS(ridge|M2|expanding) = {q1:.3%} (SE {q1_se:.3%}); vs zero {r2_boot(RES[P], ZERO)[0]:.3%}')
print(f'   of which class intercepts alone (M1): {m1:.3%} (SE {m1_se:.3%}); characteristics beyond class means: {d_char:.3%} (SE {d_char_se:.3%})')
print(f'   pre-registered rule (R2 > 0 and more than 2 SE above 0): {"YES" if q1_verdict else "NO"}')
print(f'   characteristics beyond class means, +/- 2 SE: {d_char - 2 * d_char_se:.3%} to {d_char + 2 * d_char_se:.3%}')
q1_mde, q1_mde80 = 2 * q1_se, (2 + stats.norm.ppf(0.8)) * q1_se
print(f'   power of this statistic (added after the result): the rule needs R2_OOS > 2 SE = {q1_mde:.2%}; a true R2_OOS of '
      f'{q1_mde80:.2%} would pass it with 80% probability.\n   The Section 3.0 calculation assumed a pure within-class signal; the '
      'Q1 statistic also carries the class-intercept errors, which load on common class shocks.')

# the per-asset trailing mean of y (the literal per-series harness of HW5) is a much noisier benchmark than the pooled mean
BA = OOS.b_asset.to_numpy()
pa = pd.DataFrame({n: dict(zip(['R2 vs per-asset mean', 'SE'], r2_boot(RES[n], BA))) for n in ['ols|M1|expanding', P, 'ridge|M3|expanding']}).T
pa['R2 / SE'] = pa['R2 vs per-asset mean'] / pa['SE']
display(pa.style.format({'R2 vs per-asset mean': '{:.3%}', 'SE': '{:.3%}', 'R2 / SE': '{:.2f}'})
        .set_caption('Table 3.10c: against the per-asset trailing mean of y (2011-2024)'))
n_pa = int((res_tab['R2 vs per-asset mean'] > 0).sum())
print(f'specifications with R2_OOS > 0 against the per-asset mean: {n_pa} of {len(res_tab)}; '
      f'the per-asset mean itself scores {r2_oos(OOS.y, OOS.b_asset, OOS.b_pool):.2%} against the pooled mean')'''

E3 = r'''# ---------- By class, by sub-period, and in raw-return units ----------
KEY = ['ols|M1|expanding', 'ols|M2|expanding', 'ridge|M2|expanding', 'rf|M2|expanding', 'ridge|M3|expanding',
       'lasso|M3|expanding', 'pcr|M3|expanding', 'rf|M3|expanding']
by_cls = pd.DataFrame({c: {n: r2_oos(OOS.y[OOS.cls == c], RES[n][OOS.cls == c], OOS.b_pool[OOS.cls == c]) for n in KEY} for c in 'ABCD'})
by_cls['all'] = [res_tab.loc[n, 'R2 vs pooled mean'] for n in KEY]
display(by_cls.style.format('{:.2%}').set_caption('Table 3.11: R2_OOS vs the pooled trailing mean, by asset class (2011-2024)'))
sub = {}
for lab, (s0, s1) in SUBPERIODS.items():
    rows_sp = ((OOS.m >= s0) & (OOS.m <= s1)).to_numpy()
    sub[lab] = {n: r2_boot(RES[n], OOS.b_pool.to_numpy(), rows_sp)[0] for n in KEY}
    sub[lab + ' (vs zero)'] = {n: r2_boot(RES[n], ZERO, rows_sp)[0] for n in KEY}
display(pd.DataFrame(sub).style.format('{:.2%}').set_caption('Table 3.12: R2_OOS by sub-period'))
# scheme differences on the same months (paired): static or rolling minus expanding
sch_rows = {}
for mdl, fs in [('ols', 'M2'), ('ols', 'M3'), ('ridge', 'M2'), ('ridge', 'M3'), ('rf', 'M2'), ('rf', 'M3'), ('lasso', 'M3'), ('pcr', 'M3')]:
    for s_ in ['static', 'rolling']:
        d, se = delta_boot(RES[f'{mdl}|{fs}|{s_}'], RES[f'{mdl}|{fs}|expanding'], OOS.b_pool.to_numpy())
        sch_rows[(f'{mdl} {fs}', f'{s_} - expanding')] = {'difference in R2_OOS': d, 'paired SE': se, 'difference / SE': d / se}
sch_tab = pd.DataFrame(sch_rows).T
display(sch_tab.style.format({'difference in R2_OOS': '{:.3%}', 'paired SE': '{:.3%}', 'difference / SE': '{:.2f}'})
        .set_caption('Table 3.12b: scheme differences, paired month bootstrap (vs the pooled trailing mean)'))
# raw excess returns: forecast r-hat = y-hat * sigma-hat, against the literal per-asset trailing mean of returns and zero
raw = {}
for n in KEY:
    rh = RES[n] * OOS.sig.to_numpy()
    raw[n] = {**{f'class {c}: vs asset mean': r2_oos(OOS.r[OOS.cls == c], rh[OOS.cls == c], OOS.rb_asset[OOS.cls == c]) for c in 'ABCD'},
              **{f'class {c}: vs zero': r2_oos(OOS.r[OOS.cls == c], rh[OOS.cls == c], ZERO[OOS.cls == c]) for c in 'ABCD'}}
display(pd.DataFrame(raw).T.style.format('{:.2%}').set_caption('Table 3.13: R2_OOS in raw excess-return units, by class'))'''

E4 = r'''# ---------- Tuning choices, edges, and which predictors carry the signal ----------
tuned = [n for n in ledger.index if ledger.loc[n, 'model'] in ('ridge', 'lasso', 'pcr', 'ridge-per-class')]
tune_tab = pd.DataFrame({n: {'refits': len(LOGS[n]), 'share of refits at a grid edge': LOGS[n]['at grid edge'].mean(),
                             'median choice': np.median(LOGS[n]['choice'].astype(float)),
                             'min choice': LOGS[n]['choice'].astype(float).min(), 'max choice': LOGS[n]['choice'].astype(float).max()}
                         for n in tuned}).T
display(tune_tab.style.format('{:.3g}').set_caption('Table 3.14: chosen penalty (ridge: alpha/n; lasso: alpha) or number of components (PCR)'))
print('grid edges: ridge alpha/n in [1e-4, 1e3]; lasso alpha in [3e-5, 0.32]; PCR K in [1, 10]. '
      'The top edge of ridge/lasso is the class-means model ("no signal"); the bottom edge is close to OLS.')

coef = pd.DataFrame([r['coef'] for _, r in LOGS['ridge|M2|expanding'].iterrows()],
                    index=[d.year for d in LOGS['ridge|M2|expanding']['refit']])
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 3.8))
for cn, col in zip(CHARS, [C_BLUE, C_ORANGE, C_AQUA, C_YELLOW, C_GREY]):
    ax1.plot(coef.index, coef[cn], marker='o', markersize=4, color=col, label=f'{cn} ({XNAMES[CHARS.index(cn)]})')
ax1.axhline(0, color=C_GREY, linewidth=1)
ax1.set_xlabel('refit (December of year)')
ax1.set_ylabel('coefficient per 1 SD of the rank')
ax1.set_title('Figure 3.5a: ridge M2 (expanding), coefficients by refit')
ax1.legend(fontsize=8, ncol=3)
coef3 = pd.DataFrame([r['coef'] for _, r in LOGS['ridge|M3|expanding'].iterrows()],
                     index=[d.year for d in LOGS['ridge|M3|expanding']['refit']])
blocks = {'characteristics': CHARS, 'global macro x class': GLOB, 'country macro x class': CTRY_INT}
for (bn, cols), col in zip(blocks.items(), [C_BLUE, C_ORANGE, C_AQUA]):
    ax2.plot(coef3.index, np.sqrt((coef3[cols] ** 2).sum(axis=1)), marker='o', markersize=4, color=col, label=bn)
ax2.set_xlabel('refit (December of year)')
ax2.set_ylabel('size of the block: sqrt(sum of squared coefficients)')
ax2.set_title('Figure 3.5b: ridge M3 (expanding), size of each block')
ax2.legend(fontsize=8)
plt.tight_layout()
plt.show()
stab = pd.DataFrame({'mean coefficient': coef.mean(), 'share of refits with the same sign as the mean': (np.sign(coef) == np.sign(coef.mean())).mean()})
display(stab.style.format('{:.3f}').set_caption('Table 3.15: stability of the ridge M2 coefficients across the 14 refits'))
sel = pd.DataFrame([r['coef'] for _, r in LOGS['lasso|M3|expanding'].iterrows()]).ne(0).mean()
print('lasso M3 (expanding): share of refits in which each block has at least one selected variable:',
      {bn: round(float(pd.DataFrame([r['coef'] for _, r in LOGS['lasso|M3|expanding'].iterrows()])[cols].ne(0).any(axis=1).mean()), 2)
       for bn, cols in blocks.items()})
print('lasso M3 (expanding): variables selected in at least half the refits:', list(sel[sel >= 0.5].index))
print('PCR M3: components chosen per refit (expanding):', list(LOGS['pcr|M3|expanding']['choice']))'''

E5 = r'''# ---------- The forest: out-of-sample permutation importance, year by year (L8 p.57) ----------
imp_rows = {}
for key, imp in IMPORTANCE.items():
    mse_b = pd.Series({DATES[b + 1].year: ((OOS.y - OOS.b_pool)[(OOS.m > b) & (OOS.m <= b + 12)] ** 2).mean() for b in REFIT_ENDS})
    pts = imp.div(mse_b, axis=0)                      # increase in MSE when shuffled, in R2_OOS points
    imp_rows[key] = pts
    display(pd.DataFrame({'mean over the 14 test years': pts.mean(), 'years with a positive importance': (pts > 0).sum()})
            .sort_values('mean over the 14 test years', ascending=False).style.format({'mean over the 14 test years': '{:.3%}'})
            .set_caption(f'Table 3.16: {key}: loss of R2_OOS when one input is shuffled in its test year'))'''

E6 = r'''# ---------- Per-class models vs the pooled model ----------
pc = {}
for fs in ['M2', 'M3']:
    for n in [f'ridge|{fs}|expanding', f'ridge-per-class|{fs}|expanding']:
        pc[n] = {**{f'class {c}': r2_oos(OOS.y[OOS.cls == c], RES[n][OOS.cls == c], OOS.b_pool[OOS.cls == c]) for c in 'ABCD'},
                 'all': res_tab.loc[n, 'R2 vs pooled mean']}
    d, se = delta_boot(RES[f'ridge-per-class|{fs}|expanding'], RES[f'ridge|{fs}|expanding'], OOS.b_pool.to_numpy())
    print(f'{fs}: per-class minus pooled R2_OOS = {d:.3%} (SE {se:.3%})')
display(pd.DataFrame(pc).T.style.format('{:.2%}').set_caption('Table 3.17: pooled vs per-class ridge (expanding), R2_OOS vs the pooled mean'))
first_rows = F[F.m <= INIT_END_M].groupby('cls').size()
print('training rows per class at the first refit (2010-12):', first_rows.to_dict(),
      '| slopes per class in M3: A', len(class_cols('M3', 'A')), '| B, C, D', len(class_cols('M3', 'B')))'''

E7 = r'''# ---------- When do the gains accrue, and are the forecasts calibrated? ----------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 4))
odates = DATES[OOS_M]
for n, col in [('ridge|M2|expanding', C_BLUE), ('ridge|M3|expanding', C_ORANGE), ('rf|M2|expanding', C_AQUA), ('ols|M1|expanding', C_GREY)]:
    gain = month_sums((OOS.y - OOS.b_pool) ** 2 - (OOS.y - RES[n]) ** 2)
    ax1.plot(odates, np.cumsum(gain), color=col, label=n)
gain0 = month_sums((OOS.y - OOS.b_pool) ** 2 - (OOS.y - ZERO) ** 2)
ax1.plot(odates, np.cumsum(gain0), color=C_INK, linestyle='--', linewidth=1.2, label='zero forecast')
ax1.axhline(0, color=C_GREY, linewidth=1)
ax1.set_ylabel('cumulative (SSE benchmark - SSE model)')
ax1.set_title('Figure 3.6a: gains over the pooled trailing mean, month by month')
ax1.legend(fontsize=7.5)
yh = RES['ridge|M2|expanding']
dec = pd.qcut(yh, 10, labels=False)
cal = pd.DataFrame({'forecast': yh, 'realised': OOS.y}).groupby(dec).mean()
ax2.plot(cal.forecast, cal.realised, 'o-', color=C_BLUE, label='ridge M2 (expanding): 10 forecast deciles')
lim = [min(cal.min()), max(cal.max())]
ax2.plot(lim, lim, color=C_GREY, linestyle='--', linewidth=1, label='45-degree line')
ax2.set_xlabel('mean forecast of y in the decile')
ax2.set_ylabel('mean realised y')
ax2.set_title('Figure 3.6b: forecast deciles vs outcomes (2011-2024)')
ax2.legend(fontsize=8)
plt.tight_layout()
plt.show()
cal.index = [f'decile {d + 1}' for d in cal.index]
display(cal.T.style.format('{:.3f}').set_caption('Table 3.17b: the numbers behind Figure 3.6b (mean forecast and mean realised y by forecast decile)'))
print(f'sd of the ridge M2 forecasts {yh.std(ddof=1):.3f} vs sd of y {OOS.y.std(ddof=1):.3f}; '
      f'corr(forecast, outcome) = {np.corrcoef(yh, OOS.y)[0, 1]:.3f}')'''

from cells_x import E3B, E3C, E4B, E4C
CELLS_E = [('md', MD_RUN), ('code', E1), ('code', E2), ('code', E3), ('code', E3B), ('code', E3C), ('code', E4), ('code', E4B), ('code', E4C), ('code', E5), ('code', E6), ('code', E7)]
