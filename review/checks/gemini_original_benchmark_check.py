"""Gemini's ORIGINAL Problem 3 scores its ridge against a per-asset trailing mean. Re-score the same forecasts
against the pooled trailing mean and zero, and against a per-asset mean, to show how much the headline depends on the benchmark.
Run from the repository root: python3 review/checks/gemini_original_benchmark_check.py"""
import json, io, contextlib, warnings
import matplotlib; matplotlib.use('Agg')
import numpy as np, pandas as pd
warnings.filterwarnings('ignore')
nb = json.load(open('gemini/Final_Autumn_submission.ipynb')); ns = {}
for i in (2, 35, 39):
    src = '\n'.join(l for l in ''.join(nb['cells'][i]['source']).splitlines() if not l.strip().startswith('panel.head()'))
    with contextlib.redirect_stdout(io.StringIO()):
        exec(compile(src, f'cell{i}', 'exec'), ns)
E, PM = ns['df_eval'].copy(), ns['panel_master']
pooled = []
for d in E['date'].drop_duplicates():
    pooled.append((d, PM.loc[PM['date'] < d, 'y_vol_scaled'].mean()))
E = E.merge(pd.DataFrame(pooled, columns=['date', 'bench_pooled']), on='date')
r2 = lambda y, f, b: 100 * (1 - ((y - f) ** 2).sum() / ((y - b) ** 2).sum())
y = E.actual_y
print('Original Gemini ridge (chars only), OOS 2010-2024:')
print(f'  vs per-asset trailing mean (its benchmark): {r2(y, E.pred_ridge_chars, E.bench_trailing):+.3f}%')
print(f'  vs pooled trailing mean:                    {r2(y, E.pred_ridge_chars, E.bench_pooled):+.3f}%')
print(f'  vs zero:                                    {r2(y, E.pred_ridge_chars, 0.0):+.3f}%')
print(f'  per-asset mean vs pooled mean:              {r2(y, E.bench_trailing, E.bench_pooled):+.3f}%')
print('x10 enters the original macro models with a 1-month lag:', 'x10_lag' in ns['all_macro_features'])
print('rows from 2000-2002 (back-fill period) in the first training window:', int((PM['date'] < '2003-01-01').sum()))
