"""Holdout check of the frozen end-of-training Problem 3 models on months after the training data (see DESIGN.md).

Run from the repository root:
    python3 review/holdout_2025/holdout.py dry     # 2023-24 of the existing panel stands in as the holdout (code test only)
    python3 review/holdout_2025/holdout.py real    # needs review/holdout_2025/data/extended_returns.csv from rebuild.py

The model class (`LinearFE`: ridge with unpenalised class intercepts) is taken verbatim from the notebook's code, and
the notebook's primary result is reproduced first as a correctness gate.
"""
import json, re, sys, os
import numpy as np, pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import lasso_path

MODE = sys.argv[1] if len(sys.argv) > 1 else 'dry'
assert MODE in ('dry', 'real')
HERE = 'review/holdout_2025'
OUT = f'{HERE}/results_{MODE}'
os.makedirs(OUT, exist_ok=True)
rng = np.random.default_rng(7034)

# ---------------------------------------------------------------- the notebook's model class, verbatim
nb = json.load(open('Final_Autumn_2026-1.ipynb'))['cells']
src73 = ''.join(nb[73]['source'])
exec(src73[src73.index('class LinearFE'):], globals())          # defines LinearFE exactly as in the notebook
GRID = 10.0 ** np.linspace(3, -4, 36)                             # the notebook's ridge grid (alpha/n), most penalised first

# ---------------------------------------------------------------- data
panel = pd.read_csv('asset_panel.csv', parse_dates=['date'])
info = pd.read_csv('asset_info.csv').set_index('asset_id')
IDS = [f'asset_{i}' for i in range(1, 51)]
CLS = info.loc[IDS, 'asset_class'].to_numpy()
K = pd.Series(CLS).map({'A': 0, 'B': 1, 'C': 2, 'D': 3}).to_numpy()
R_panel = panel.pivot(index='date', columns='asset_id', values='excess_return')[IDS]

if MODE == 'real':
    ext = pd.read_csv(f'{HERE}/data/extended_returns.csv', parse_dates=['date']).set_index('date')
    ACCEPTED = list(ext.columns)
    new = ext[ext.index > '2024-12-31']
    assert len(new) >= 12, 'fewer than 12 holdout months: the design says do not run'
    R = pd.concat([R_panel, new.reindex(columns=IDS)])           # unaccepted assets are missing after 2024-12
    TRAIN_END = pd.Timestamp('2024-12-31')
else:
    ACCEPTED = IDS
    R = R_panel.copy()
    TRAIN_END = pd.Timestamp('2022-12-31')
DATES = R.index
m_of = {d: i for i, d in enumerate(DATES)}
FIRST_M, B = m_of[pd.Timestamp('2003-01-31')], m_of[TRAIN_END]
H0, H1 = B + 1, len(DATES) - 1                                     # holdout months
C0 = m_of[pd.Timestamp('2011-01-31')]                              # comparison period starts
acc = np.array([a in ACCEPTED for a in IDS])
print(f'mode {MODE}: training 2003-01..{TRAIN_END.date()}, holdout {DATES[H0].date()}..{DATES[H1].date()} '
      f'({H1 - H0 + 1} months), {acc.sum()} assets in the holdout')

# ---------------------------------------------------------------- characteristics, exactly as the notebook defines them
src55 = ''.join(nb[55]['source'])
_lc = src55[src55.index('def leading_copies'):]
exec(_lc[:_lc.index('\n\n')], globals())                         # the notebook's back-fill rule, verbatim

def stored(x):
    W = panel.pivot(index='date', columns='asset_id', values=x)[IDS].copy()
    if x in ('x2', 'x3', 'x5'):
        for j in range(len(IDS)):
            k = leading_copies(W.iloc[:, j].to_numpy())
            if k:
                W.iloc[:k, j] = np.nan
    return W.shift(1) if x in ('x1', 'x3', 'x4') else W
RAW = {x: stored(x).reindex(DATES) for x in ['x1', 'x2', 'x3', 'x4', 'x5']}
# after the panel ends, x2 and x5 come from the (spliced) returns with the same definitions; x1, x3, x4 do not exist
R12 = (1 + R).rolling(12, min_periods=12).apply(np.prod, raw=True).shift(1) - 1
SD36 = R.rolling(36, min_periods=36).std(ddof=1).shift(1)
after = DATES > R_panel.index.max()
RAW['x2'].loc[after] = R12.loc[after]
RAW['x5'].loc[after] = SD36.loc[after]
for x in ['x1', 'x3', 'x4']:
    RAW[x].loc[after] = np.nan

def ranks(W, cols_mask):
    """Rank within class and month, scaled to [-0.5, 0.5], among the assets in cols_mask (the notebook's transform)."""
    out = pd.DataFrame(np.nan, index=W.index, columns=IDS)
    for c in 'ABCD':
        cols = [a for a, k, ok in zip(IDS, CLS, cols_mask) if k == c and ok]
        if len(cols) > 1:
            out[cols] = (W[cols].rank(axis=1, method='average') - 1) / (len(cols) - 1) - 0.5
        elif len(cols) == 1:
            out[cols] = 0.0
    return out
U_train = {f'u{i}': ranks(RAW[f'x{i}'], np.ones(50, bool)) for i in range(1, 6)}      # training: all 50 assets
U_hold = {f'u{i}': ranks(RAW[f'x{i}'], acc) for i in range(1, 6)}                      # holdout: accepted assets only

sig = R.rolling(36, min_periods=36).std(ddof=1).shift(1)
Y = R / sig
rows = []
for t in range(FIRST_M, len(DATES)):
    U = U_train if t <= R_panel.index.get_loc(R_panel.index.max()) else U_hold
    for j, a in enumerate(IDS):
        if np.isnan(Y.iat[t, j]):
            continue
        rows.append((t, a, j, K[j], R.iat[t, j], sig.iat[t, j], Y.iat[t, j], R12.iat[t, j],
                     *[U[f'u{i}'].iat[t, j] for i in range(1, 6)]))
F = pd.DataFrame(rows, columns=['m', 'asset_id', 'j', 'k', 'r', 'sig', 'y', 'r12', 'u1', 'u2', 'u3', 'u4', 'u5'])
mon = F.groupby('m').y.agg(['sum', 'count'])
F['b_pool'] = F.m.map((mon['sum'].cumsum() / mon['count'].cumsum()).shift(1))     # pooled trailing mean through t-1

# ---------------------------------------------------------------- tuning and fitting, as the notebook's run_linear
def fit_tuned(cols, a, b, fr):
    """Tune alpha/n on three annual forward folds inside [a, b], refit on [a, b]; return the fitted model and the choice."""
    X, y, k, m = fr[cols].to_numpy(), fr.y.to_numpy(), fr.k.to_numpy(), fr.m.to_numpy()
    j = 0
    if cols:
        mse = np.zeros(len(GRID))
        for jj in (3, 2, 1):
            f1, v0, v1 = b - 12 * jj, b - 12 * jj + 1, b - 12 * jj + 12
            fit, val = (m >= a) & (m <= f1), (m >= v0) & (m <= v1)
            assert m[fit].max() < m[val].min()
            mdl = LinearFE('ridge', GRID).fit(X[fit], y[fit], k[fit])
            mse += ((y[val][:, None] - mdl.predict(X[val], k[val])) ** 2).mean(0) / 3
        j = int(np.argmin(mse))
    win = (m >= a) & (m <= b)
    return LinearFE('ridge', [GRID[j]]).fit(X[win], y[win], k[win]), GRID[j]

TRAIN_ROWS = F[F.m <= B]

# correctness gate: the notebook's primary result, ridge M2 expanding over 2011-2024, is 0.042% vs the pooled mean
if True:                                                          # always run the gate
    NB_OOS = F[(F.m >= C0) & (F.m <= R_panel.index.get_loc(R_panel.index.max()))]
    pred = pd.Series(np.nan, index=NB_OOS.index)
    for b in range(C0 - 1, R_panel.index.get_loc(R_panel.index.max()), 12):
        mdl, _ = fit_tuned(['u1', 'u2', 'u3', 'u4', 'u5'], FIRST_M, b, F[F.m <= b])
        te = NB_OOS[(NB_OOS.m > b) & (NB_OOS.m <= b + 12)]
        pred.loc[te.index] = mdl.predict(te[['u1', 'u2', 'u3', 'u4', 'u5']].to_numpy(), te.k.to_numpy())[:, 0]
    r2_nb = 1 - ((NB_OOS.y - pred) ** 2).sum() / ((NB_OOS.y - NB_OOS.b_pool) ** 2).sum()
    print(f'correctness gate: ridge M2 expanding 2011-2024 R2_OOS vs pooled mean = {r2_nb:.3%} (notebook: 0.042%)')
    assert abs(r2_nb - 0.00042) < 0.00001, 'the rebuilt pipeline does not reproduce the notebook'

MODELS = {'M1 class means': [], 'M2r ridge on x2, x5': ['u2', 'u5'], 'M2n ridge M2, missing ranks neutral': ['u1', 'u2', 'u3', 'u4', 'u5']}
FROZEN = {name: fit_tuned(cols, FIRST_M, B, TRAIN_ROWS) for name, cols in MODELS.items()}
HOLD = F[(F.m >= H0) & F.asset_id.isin(ACCEPTED)].copy()
HOLD[['u1', 'u3', 'u4']] = HOLD[['u1', 'u3', 'u4']].fillna(0.0)                         # M2n only: unavailable ranks at 0
assert HOLD[['y', 'b_pool', 'sig', 'r12', 'u2', 'u5']].notna().all().all()
for name, cols in MODELS.items():
    mdl, choice = FROZEN[name]
    HOLD[name] = mdl.predict(HOLD[cols].to_numpy(), HOLD.k.to_numpy())[:, 0] if cols else mdl.predict(np.zeros((len(HOLD), 0)), HOLD.k.to_numpy())[:, 0]
choices = {name: FROZEN[name][1] if MODELS[name] else None for name in MODELS}

# comparison period: expanding forecasts 2011..train end on the same accepted assets
COMP = F[(F.m >= C0) & (F.m <= B) & F.asset_id.isin(ACCEPTED)].copy()
for name, cols in MODELS.items():
    COMP[name] = np.nan
    for b in range(C0 - 1, B, 12):
        mdl, _ = fit_tuned(cols, FIRST_M, b, F[F.m <= b])
        te = COMP[(COMP.m > b) & (COMP.m <= b + 12)]
        X = te[cols].to_numpy() if cols else np.zeros((len(te), 0))
        COMP.loc[te.index, name] = mdl.predict(X, te.k.to_numpy())[:, 0]
assert COMP[list(MODELS)].notna().all().all()

# ---------------------------------------------------------------- R2_OOS with a month bootstrap
def r2_table(D):
    months = np.sort(D.m.unique()); pos = D.m.map({m: i for i, m in enumerate(months)}).to_numpy(); T = len(months)
    idx = rng.integers(0, T, size=(10_000, T))
    out = {}
    for name in MODELS:
        for bn, b in [('pooled trailing mean', D.b_pool.to_numpy()), ('zero', np.zeros(len(D)))]:
            e_m = np.bincount(pos, (D.y - D[name]) ** 2, T); e_b = np.bincount(pos, (D.y - b) ** 2, T)
            out[(name, bn)] = (1 - e_m.sum() / e_b.sum(), (1 - e_m[idx].sum(1) / e_b[idx].sum(1)).std(ddof=1))
    tab = pd.DataFrame({'R2_OOS': {k: v[0] for k, v in out.items()}, 'SE': {k: v[1] for k, v in out.items()}})
    return tab
r2_hold, r2_comp = r2_table(HOLD), r2_table(COMP)
by_class = {c: r2_table(HOLD[HOLD.k == i]).xs('pooled trailing mean', level=1)['R2_OOS'] for i, c in enumerate('ABCD') if (HOLD.k == i).any()}

# ---------------------------------------------------------------- portfolios, as in the notebook
def portfolios(D, bp=10):
    months = np.sort(D.m.unique())
    strat = {'EW': lambda s: np.ones(len(s)), 'RP': lambda s: 1 / s.sig.to_numpy(), 'TSMOM': lambda s: np.sign(s.r12.to_numpy()) / s.sig.to_numpy()}
    for name in MODELS:
        strat[f'P1 ({name.split(" ")[0]})'] = (lambda s, n=name: s[n].to_numpy() / s.sig.to_numpy())
    res, W = {}, {}
    for sn, fn in strat.items():
        net, prev, wts = [], None, []
        for t in months:
            s = D[D.m == t]
            w = fn(s); w = w / np.abs(w).sum()
            gross = (w * s.r.to_numpy()).sum()
            to = 0.0
            if prev is not None:
                pw, pr, pa = prev
                drift = pd.Series(pw * (1 + pr) / (1 + (pw * pr).sum()), index=pa).reindex(s.asset_id).fillna(0).to_numpy()
                to = np.abs(w - drift).sum()
            net.append(gross - bp / 1e4 * to)
            prev = (w, s.r.to_numpy(), s.asset_id.to_numpy()); wts.append(pd.Series(w, index=s.asset_id.to_numpy()))
        res[sn] = np.array(net); W[sn] = wts
    return res, W, months
def sharpe(x): return x.mean() / x.std(ddof=1) * np.sqrt(12)
def port_table(D):
    res, W, months = portfolios(D)
    T = len(months); idx = rng.integers(0, T, size=(10_000, T))
    rows = {}
    for sn, x in res.items():
        xb = x[idx]
        rows[sn] = {'net Sharpe': sharpe(x), 'SE': (xb.mean(1) / xb.std(1, ddof=1) * np.sqrt(12)).std(ddof=1),
                    'mean net return (annual)': 12 * x.mean()}
    rp, p1 = res['RP'], res['P1 (M2r)']
    diff = (p1[idx].mean(1) / p1[idx].std(1, ddof=1) - rp[idx].mean(1) / rp[idx].std(1, ddof=1)) * np.sqrt(12)
    extra = {'P1 (M2r) minus RP, net Sharpe (SE)': f'{sharpe(p1) - sharpe(rp):+.2f} ({diff.std(ddof=1):.2f})'}
    corr = np.mean([np.corrcoef(a.values, b.reindex(a.index).values)[0, 1] for a, b in zip(W['P1 (M2r)'], W['P1 (M1)'])])
    dshare = np.mean([w[[i for i in w.index if info.loc[i, 'asset_class'] == 'D']].abs().sum() for w in W['P1 (M2r)']])
    extra['avg monthly weight correlation, P1 (M2r) vs P1 (M1)'] = f'{corr:.4f}'
    extra['P1 (M2r) share of gross in class D'] = f'{dshare:.0%}'
    return pd.DataFrame(rows).T, extra
pt_hold, ex_hold = port_table(HOLD)
pt_comp, ex_comp = port_table(COMP)

# ---------------------------------------------------------------- report
lines = [f'# Holdout check ({MODE} run)\n',
         f'- training 2003-01 to {TRAIN_END.date()} (models frozen there); holdout {DATES[H0].date()} to {DATES[H1].date()}, '
         f'{H1 - H0 + 1} months, {acc.sum()} assets ({ {k: int(v) for k, v in pd.Series(CLS[acc]).value_counts().sort_index().items()} } by class)',
         f'- comparison: 2011-01 to {TRAIN_END.date()} on the same assets, expanding refits each December',
         f'- correctness gate: the rebuilt pipeline reproduces the notebook\'s ridge M2 result, {r2_nb:.3%} (notebook 0.042%)',
         f'- penalties chosen (alpha/n) at {TRAIN_END.date()}: ' + ', '.join(f'{k}: {v:.3g}' for k, v in choices.items() if v is not None),
         '\n## R2_OOS (holdout)\n', r2_hold.to_markdown(floatfmt='.3%'),
         '\n## R2_OOS by class, holdout, vs the pooled trailing mean\n', pd.DataFrame(by_class).to_markdown(floatfmt='.3%'),
         '\n## R2_OOS (comparison period, same assets)\n', r2_comp.to_markdown(floatfmt='.3%'),
         '\n## Portfolios, net of 10 bp (holdout)\n', pt_hold.to_markdown(floatfmt='.3f'), '', *[f'- {k}: {v}' for k, v in ex_hold.items()],
         '\n## Portfolios, net of 10 bp (comparison period, same assets)\n', pt_comp.to_markdown(floatfmt='.3f'), '', *[f'- {k}: {v}' for k, v in ex_comp.items()]]
open(f'{OUT}/RESULTS.md', 'w').write('\n'.join(lines) + '\n')
HOLD.to_csv(f'{OUT}/holdout_forecasts.csv', index=False)
print('\n'.join(lines))
