"""Targeted checks on Gemini's improved Problem 3 (cell 39), run in the notebook's own namespace.
The Gemini notebook is not modified: its code cells are executed here, then extra diagnostics are computed.
Run from the repository root:  python3 review/checks/gemini_p3_checks.py"""
import json, io, contextlib, warnings
import matplotlib
matplotlib.use('Agg')
import numpy as np, pandas as pd
warnings.filterwarnings('ignore')

nb = json.load(open('gemini/Final_Autumn_submission_improved.ipynb'))
ns = {}
for i in (2, 35, 39):                                  # imports, P3 data load, the whole P3 pipeline
    src = ''.join(nb['cells'][i]['source'])
    src = '\n'.join(l for l in src.splitlines() if not l.strip().startswith('panel.head()'))
    with contextlib.redirect_stdout(io.StringIO()):
        exec(compile(src, f'cell{i}', 'exec'), ns)
g = type('NS', (), ns)                                   # attribute access to the notebook's variables
print('Gemini pipeline re-executed. OOS rows:', len(g.oos_results))

def r2(y, f, b): return 100 * (1 - np.sum((y - f) ** 2) / np.sum((y - b) ** 2))
R = g.oos_results

# ---------------------------------------------------------------- C1: is the placebo actually shifted?
print('\n=== C1: placebo inputs vs real inputs (M3 uses char ranks + inter_glob + inter_coun) ===')
for s in g.placebo_shifts:
    p = g.placebo_panels[s]
    same_rows = (p[['date', 'asset_id']].values == g.eval_df[['date', 'asset_id']].values).all()
    g_diff = np.nanmax(np.abs(p[g.glob_features].values - g.eval_df[g.glob_features].values))
    m3_diff = np.nanmax(np.abs(p[g.m3_features].values - g.eval_df[g.m3_features].values))
    print(f'shift {s}m: rows aligned with eval_df: {same_rows} | max |change| in the shifted global columns: {g_diff:.3f}'
          f' | max |change| in the 39 M3 inputs actually used: {m3_diff:.3g}')
print('predictions identical to real-macro M3:',
      {s: bool(np.allclose(R[f"pred_pl_{s}m"], R["pred_m3_ridge"])) for s in g.placebo_shifts})

# ---------------------------------------------------------------- C2: a working placebo, same harness
print('\n=== C2: corrected placebo (rebuild the interactions from shifted global AND country macro) ===')
from sklearn.linear_model import Ridge
def build_inputs(df, shift):
    """Circularly shift every macro input (global and country) by `shift` months, then rebuild the interactions."""
    d = df.copy()
    dates = np.sort(d['date'].unique())
    gm = g.mac_g[['date'] + g.glob_features].set_index('date').loc[dates]
    gm = pd.DataFrame(np.roll(gm.values, shift, axis=0), index=gm.index, columns=gm.columns)
    d = d.drop(columns=g.glob_features).merge(gm, left_on='date', right_index=True, how='left')
    cm = g.country_macro_df.set_index(['country', 'date'])[g.coun_features]
    parts = []
    for c, grp in cm.groupby(level=0):
        grp = grp.droplevel(0).reindex(dates)
        parts.append(pd.DataFrame(np.roll(grp.values, shift, axis=0), index=grp.index, columns=grp.columns).assign(country=c))
    cm_s = pd.concat(parts).reset_index().rename(columns={'index': 'date'})
    d = d.drop(columns=g.coun_features).merge(cm_s, on=['country', 'date'], how='left')
    for cc in g.coun_features:
        d.loc[d['asset_class'] == 'A', cc] = 0.0
    for cd in ['class_A', 'class_B', 'class_C', 'class_D']:
        for gf in g.glob_features:
            d[f'{gf}_{cd}'] = d[gf] * d[cd]
    for cd in ['class_B', 'class_C', 'class_D']:
        for cf in g.coun_features:
            d[f'{cf}_{cd}'] = d[cf] * d[cd]
    return d.sort_values(['date', 'asset_id']).reset_index(drop=True)

def run_m3(df, alpha=200.0, feats=None):
    """Gemini's M3 harness (annual December refits, within-class demeaning, fixed alpha), returns OOS preds."""
    feats = feats or g.m3_features
    df = df.sort_values(['date', 'asset_id']).reset_index(drop=True)
    out = []
    for yr in range(2011, 2025):
        tr = df[df['date'] <= f'{yr-1}-12-31']
        te = df[(df['date'] > f'{yr-1}-12-31') & (df['date'] <= f'{yr}-12-31')]
        cm_y = tr.groupby('asset_class')['y_vol_scaled'].mean()
        Xm = tr.groupby('asset_class')[feats].transform('mean')
        mdl = Ridge(alpha=alpha, fit_intercept=False).fit(tr[feats] - Xm, tr['y_vol_scaled'] - tr['asset_class'].map(cm_y))
        mu = tr.groupby('asset_class')[feats].mean()
        Xte = te[feats].values - mu.loc[te['asset_class']].values
        out.append(pd.Series(te['asset_class'].map(cm_y).values + mdl.predict(Xte), index=te.index))
    return df, pd.concat(out)

base_df, p_real = run_m3(g.eval_df)
oos = base_df.loc[p_real.index]
bench = R.set_index(['date', 'asset_id']).loc[list(zip(oos['date'], oos['asset_id'])), 'bench_pooled'].values
y = oos['y_vol_scaled'].values
print(f'real macro M3 (re-implemented): R2_OOS vs pooled mean = {r2(y, p_real.values, bench):+.3f}% '
      f'(notebook: {r2(R.y_vol_scaled, R.pred_m3_ridge, R.bench_pooled):+.3f}%)')
_, p_m2 = run_m3(g.eval_df, alpha=50.0, feats=g.m2_features)
r2_m2 = r2(y, p_m2.values, bench)
print(f'M2 (re-implemented): {r2_m2:+.3f}%  ->  macro gain M3 - M2 = {r2(y, p_real.values, bench) - r2_m2:+.3f} pp')
rows = []
for s in [36, 48, 60, 72, 84, 96, 108, 120]:
    d_s, p_s = run_m3(build_inputs(g.eval_df, s))
    y_s = d_s.loc[p_s.index, 'y_vol_scaled'].values
    rows.append({'shift (months)': s, 'R2_OOS vs pooled mean (%)': r2(y_s, p_s.values, bench),
                 'gain over M2 (pp)': r2(y_s, p_s.values, bench) - r2_m2})
pl = pd.DataFrame(rows)
print(pl.round(3).to_string(index=False))
real_gain = r2(y, p_real.values, bench) - r2_m2
print(f'rank of real macro among real + {len(pl)} placebos (1 = best): {1 + int((pl["gain over M2 (pp)"] > real_gain).sum())} of {len(pl) + 1}')

# ---------------------------------------------------------------- C3: where does P1's Sharpe come from?
print('\n=== C3: P1 built from class means only (pred_m1) vs Gemini P1 (pred_m2_ridge), same costs ===')
def port(pred_col, df=R, cost=0.0010):
    df = df.sort_values(['date', 'asset_id'])
    prev, rows = None, []
    for d, sub in df.groupby('date'):
        raw = sub[pred_col].values / sub['sigma_36m'].values
        w = raw / np.abs(raw).sum()
        to = 1.0 if prev is None else np.abs(w - prev).sum()
        rows.append({'date': d, 'gross': (w * sub['excess_return'].values).sum(), 'to': to,
                     'wD': np.abs(w[sub['asset_class'].values == 'D']).sum(),
                     'w16': abs(w[sub['asset_id'].values == 'asset_16'][0])})
        rows[-1]['net'] = rows[-1]['gross'] - cost * to
        prev = w
    return pd.DataFrame(rows).set_index('date')
def sharpe(r): return r.mean() / r.std(ddof=1) * np.sqrt(12)
P1, M1 = port('pred_m2_ridge'), port('pred_m1')
print(f'P1 (ridge M2) net Sharpe {sharpe(P1.net):.3f} | P1 on class means only (M1) net Sharpe {sharpe(M1.net):.3f}')
wc = []
for d, sub in R.sort_values(['date', 'asset_id']).groupby('date'):
    a = sub['pred_m2_ridge'].values / sub['sigma_36m'].values; b = sub['pred_m1'].values / sub['sigma_36m'].values
    wc.append(np.corrcoef(a / np.abs(a).sum(), b / np.abs(b).sum())[0, 1])
print(f'average monthly correlation of P1 weights with class-means weights: {np.mean(wc):.4f}')
print(f'P1 share of gross exposure in class D: {P1.wD.mean():.0%}; asset_16 average |w| {P1.w16.mean():.0%}, max {P1.w16.max():.0%}')
import statsmodels.api as sm
X = sm.add_constant(pd.DataFrame({'EW': g.port_df.EW_net, 'RP': g.port_df.RP_net, 'TSMOM': g.port_df.TSMOM_net, 'M1tilt': M1.net}))
fit = sm.OLS(P1.net, X).fit()
print(f'P1 regressed on EW, RP, TSMOM and the class-means tilt: beta(M1tilt) {fit.params.M1tilt:.3f}, R2 {fit.rsquared:.4f}, '
      f'alpha {12 * 100 * fit.params.const:+.2f}%/yr (t = {fit.tvalues.const:.2f})')

# ---------------------------------------------------------------- C4: inference Gemini did not report
print('\n=== C4: month-bootstrap SEs (B = 5,000, forecasts held fixed) ===')
rng = np.random.default_rng(7034)
months = np.sort(R['date'].unique()); T = len(months)
by_m = {m: i for i, m in enumerate(months)}
mi = R['date'].map(by_m).values
def agg(col_f, col_b):
    e_f = np.bincount(mi, (R.y_vol_scaled - R[col_f]) ** 2, T); e_b = np.bincount(mi, (R.y_vol_scaled - R[col_b]) ** 2, T)
    return e_f, e_b
stats_ = {}
for name, f, b in [('M2 ridge vs pooled mean', 'pred_m2_ridge', 'bench_pooled'), ('M1 class means vs pooled mean', 'pred_m1', 'bench_pooled'),
                   ('M3 ridge vs pooled mean', 'pred_m3_ridge', 'bench_pooled'), ('M2 ridge vs M1 class means', 'pred_m2_ridge', 'pred_m1'),
                   ('RF vs pooled mean', 'pred_m2_rf', 'bench_pooled')]:
    e_f, e_b = agg(f, b)
    idx = rng.integers(0, T, size=(5000, T))
    boot = 100 * (1 - e_f[idx].sum(1) / e_b[idx].sum(1))
    print(f'{name:32s}: R2_OOS {100 * (1 - e_f.sum() / e_b.sum()):+.3f}% (SE {boot.std(ddof=1):.3f}%)')
ret = pd.DataFrame({'P1': P1.net, 'RP': g.port_df.RP_net, 'EW': g.port_df.EW_net, 'M1': M1.net})
idx = rng.integers(0, len(ret), size=(5000, len(ret)))
for a_, b_ in [('P1', 'RP'), ('P1', 'EW'), ('P1', 'M1')]:
    A, B = ret[a_].values[idx], ret[b_].values[idx]
    d = (A.mean(1) / A.std(1, ddof=1) - B.mean(1) / B.std(1, ddof=1)) * np.sqrt(12)
    print(f'net Sharpe {a_} - {b_}: {sharpe(ret[a_]) - sharpe(ret[b_]):+.3f} (bootstrap SE {d.std(ddof=1):.3f})')

# ---------------------------------------------------------------- C5: RF comparison fairness
print('\n=== C5: RF inputs ===')
print('RF features:', g.m2_features, '-> no class dummies; ridge M2 gets class means via demeaning')
from sklearn.ensemble import RandomForestRegressor
feats = g.m2_features + ['class_A', 'class_B', 'class_C', 'class_D']
preds = []
for yr in range(2011, 2025):
    tr = g.eval_df[g.eval_df['date'] <= f'{yr-1}-12-31']; te = g.eval_df[(g.eval_df['date'] > f'{yr-1}-12-31') & (g.eval_df['date'] <= f'{yr}-12-31')]
    rf = RandomForestRegressor(n_estimators=100, max_features=0.33, min_samples_leaf=150, random_state=7034, n_jobs=-1).fit(tr[feats], tr['y_vol_scaled'])
    preds.append(pd.DataFrame({'date': te['date'].values, 'asset_id': te['asset_id'].values, 'rf_cls': rf.predict(te[feats])}))
pr = R.merge(pd.concat(preds), on=['date', 'asset_id'])
print(f'RF as in the notebook: {r2(pr.y_vol_scaled, pr.pred_m2_rf, pr.bench_pooled):+.3f}% | RF with class dummies: '
      f'{r2(pr.y_vol_scaled, pr.rf_cls, pr.bench_pooled):+.3f}% | ridge M2: {r2(pr.y_vol_scaled, pr.pred_m2_ridge, pr.bench_pooled):+.3f}% | M1: {r2(pr.y_vol_scaled, pr.pred_m1, pr.bench_pooled):+.3f}%')

# ---------------------------------------------------------------- C6: small facts the write-up asserts
print('\n=== C6: other assertions ===')
print('NaN share in log(x14), log(x15) before fillna:', {c: float(np.log(g.mc_merged[c.replace("log_", "")]).isna().mean()) for c in ['log_x14', 'log_x15']},
      '| min x14, x15:', float(g.mac_c.x14.min()), float(g.mac_c.x15.min()))
print('coefficient paths collected but never printed/plotted:', {k: np.round(v, 3).tolist() for k, v in g.coef_paths.items() if k != 'date'})
bf = []
for a, grp in g.panel.sort_values(['asset_id', 'date']).groupby('asset_id'):
    for v in ['x2', 'x3', 'x5']:
        s = grp[v].values
        run = 1
        while run < len(s) and s[run] == s[0]: run += 1
        if run > 1: bf.append((a, v, run, str(grp['date'].iloc[run - 1].date())))
print('leading repeated runs (asset, var, length, last date):', bf)
