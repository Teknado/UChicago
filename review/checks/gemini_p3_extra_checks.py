"""Two reviewer claims about Gemini's improved Problem 3, checked independently:
(1) the extended-file PCA's '97% redundancy' depends on not standardising the columns;
(2) the M3 macro loss depends on the hard-coded ridge penalty (alpha = 200).
Run from the repository root: python3 review/checks/gemini_p3_extra_checks.py"""
import json, io, contextlib, warnings
import matplotlib; matplotlib.use('Agg')
import numpy as np, pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
warnings.filterwarnings('ignore')
nb = json.load(open('gemini/Final_Autumn_submission_improved.ipynb')); ns = {}
for i in (2, 35, 39):
    src = '\n'.join(l for l in ''.join(nb['cells'][i]['source']).splitlines() if not l.strip().startswith('panel.head()'))
    with contextlib.redirect_stdout(io.StringIO()):
        exec(compile(src, f'cell{i}', 'exec'), ns)
mx, cols = ns['mac_x'], ns['clean_ext']
raw = PCA(5).fit(mx[cols]).explained_variance_ratio_
std = PCA(5).fit(StandardScaler().fit_transform(mx[cols])).explained_variance_ratio_
print(f'(1) extended-file PCA on {len(cols)} columns: top-5 share unstandardised {raw.sum():.1%} (as printed), standardised {std.sum():.1%}')

E, R = ns['eval_df'].sort_values(['date', 'asset_id']).reset_index(drop=True), ns['oos_results']
bench = R.set_index(['date', 'asset_id'])['bench_pooled']
def run(feats, alpha):
    out = []
    for yr in range(2011, 2025):
        tr = E[E.date <= f'{yr-1}-12-31']; te = E[(E.date > f'{yr-1}-12-31') & (E.date <= f'{yr}-12-31')]
        cm = tr.groupby('asset_class').y_vol_scaled.mean(); mu = tr.groupby('asset_class')[feats].mean()
        m = Ridge(alpha=alpha, fit_intercept=False).fit(tr[feats] - mu.loc[tr.asset_class].values, tr.y_vol_scaled - tr.asset_class.map(cm))
        out.append(pd.DataFrame({'date': te.date, 'asset_id': te.asset_id, 'y': te.y_vol_scaled,
                                 'f': te.asset_class.map(cm).values + m.predict(te[feats].values - mu.loc[te.asset_class].values)}))
    o = pd.concat(out); b = bench.loc[list(zip(o.date, o.asset_id))].values
    return 100 * (1 - ((o.y - o.f) ** 2).sum() / ((o.y - b) ** 2).sum())
print('(2) R2_OOS vs pooled mean by ridge alpha (Gemini harness; test-period sweep, diagnostic only):')
for a in [50, 200, 1_000, 10_000, 100_000]:
    m2, m3 = run(ns['m2_features'], a), run(ns['m3_features'], a)
    print(f'    alpha {a:>7,}: M2 {m2:+.3f}%  M3 {m3:+.3f}%  macro gain {m3 - m2:+.3f} pp')
