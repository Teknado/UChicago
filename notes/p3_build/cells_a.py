"""Problem 3, stage 1: design fixed before fitting, integrity, trap audit, EDA."""

SETUP_EDIT = '''import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

_DATA_DIR = '/classes/41210_MiF_fall2026/Data/'
# _DATA_DIR = './'    # if the data files sit next to this notebook
import os
_DATA_DIR = _DATA_DIR if os.path.isdir(_DATA_DIR) else './'   # class server if present, else a local copy next to the notebook

panel = pd.read_csv(_DATA_DIR + 'asset_panel.csv', parse_dates=['date'])
info  = pd.read_csv(_DATA_DIR + 'asset_info.csv', index_col=0)
mac_g = pd.read_csv(_DATA_DIR + 'macro_global.csv', parse_dates=['date'])
mac_c = pd.read_csv(_DATA_DIR + 'macro_country.csv', parse_dates=['date'])
mac_x = pd.read_csv(_DATA_DIR + 'macro_country_extended.csv', parse_dates=['date'])
print(panel.shape, panel.date.min().date(), panel.date.max().date())
panel.head()'''

MD_DESIGN = r"""## 3.0 Research design, fixed before anything is fitted

This cell and the next were written **before any model was fitted and before any out-of-sample number was seen** (rule 2 of the exam's "Two rules that apply throughout"). Section 3.10 lists every later deviation.

**Windows.**
- First target month **2003-01**. The 36-month trailing volatility used to scale the target first exists then, and every back-filled characteristic value (the last one is dated 2002-07) is out of every design matrix.
- Initial training block **2003-01 to 2010-12** (96 months, 4,800 asset-months, including the 2008 crisis).
- Out-of-sample (OOS) window **2011-01 to 2024-12** (168 months, 8,400 asset-months), touched once, with sub-periods 2011–17 and 2018–24.

**Schemes** (the static fit follows the exam's suggested workflow; expanding and rolling windows are L5 p.48–52):
- static: fit once at 2010-12;
- **expanding** (primary): fixed start, refit every December, so 14 refits;
- rolling: the 96 months ending each December.

Hyper-parameters are chosen **inside each training window only**:
- three annual validation folds (fit up to year b−j, validate on year b−j+1, for j = 3, 2, 1);
- the lowest mean validation MSE wins, and a tie goes to the more heavily penalised model (L5 p.25);
- then the model is refit on the whole window.

The ridge penalty is set as α/n, so it means the same thing at every sample size.

**Target.** $y_{i,t} = r_{i,t}/\hat\sigma_{i,t-1}$, where $\hat\sigma$ is the 36-month standard deviation (ddof = 1) of the asset's own past excess returns. Forecasts become positions through $w_{i,t}\propto \hat y_{i,t}/\hat\sigma_{i,t-1}$.

**Benchmarks** (fixed now, never switched; L5 p.57):
- **Forecasts.** The trailing mean of $y$ through $t-1$, updated monthly and **pooled over all assets**, is the primary bar. Zero, the per-class and per-asset trailing means, and (for raw returns) the per-asset trailing mean of raw returns are also reported.
- **Portfolios.** Equal weight, risk parity ($\propto 1/\hat\sigma$) and time-series momentum ($\propto \mathrm{sign}(R_{12})/\hat\sigma$), all at unit gross exposure with monthly rebalancing. These are computed before any model (Section 3.3).

**Primary tests, one per question, with the decision rule fixed now:**

| question | primary evidence | "yes" requires |
|---|---|---|
| Q1 forecastable from lagged characteristics? | $R^2_{OOS}$ of ridge on M2 (characteristic ranks + class intercepts), expanding, vs the pooled trailing mean | $R^2_{OOS} > 0$ and more than 2 month-bootstrap SEs above 0 (L2 p.80, p.85) |
| Q2 does macro add? | $\Delta R^2_{OOS}$ = ridge M3 − ridge M2, expanding, same rows and benchmark | $\Delta R^2 > 2$ SE **and** larger than all 8 placebo gains (macro circularly shifted 36, 48, …, 120 months) |
| Q3 does a forecast portfolio beat the rules after costs? | P1 on ridge M2, expanding, net of 10 bp per unit of one-way turnover, 2011–2024 | higher net Sharpe than EW, RP **and** TSMOM, **and** positive α (t > 2) in a regression on the three net benchmark returns |

**Search size.** 38 candidate specifications, listed in the next cell (29 core + 9 pre-listed extras). Any "best of the ledger" claim is judged against the Bonferroni critical value for 38 tests (L4 p.25–28).

**Expectations, written before computing** (L2 p.78; L4 p.26):
- Monthly $R^2_{OOS}$ between −1% and +1%. The published stock-level range is 0.3–0.5% (L5 p.57).
- The forest may beat ridge (L8 p.54, p.62) unless the signal is weak and linear.
- Macro adds about 0 and falls inside the placebo range, partly because `x1` and `x4` of class B already contain the country rate differential.
- No significant net-Sharpe gain over RP or TSMOM, because `x2` and `x5` carry the same information those rules use."""

A1 = r'''# ---------- Design constants: fixed before any model is fitted (the exam's rule 2) ----------
import time
from scipy import stats
P3_T0 = time.perf_counter()
P3_SEED = 7034                       # Problem 3's own seed (Problem 2's SEED is never rebound)
N_JOBS = min(4, os.cpu_count() or 1)

# same colour-blind-checked style as Problems 1-2, re-declared so this problem also runs on its own
C_BLUE, C_ORANGE, C_AQUA, C_YELLOW = '#2a78d6', '#eb6834', '#1baf7a', '#eda100'
C_INK, C_GREY, C_LIGHT = '#0b0b0b', '#52514e', '#e6e5e1'
CLASS_COL = {'A': C_BLUE, 'B': C_ORANGE, 'C': C_AQUA, 'D': C_YELLOW}
plt.rcParams.update({'axes.spines.top': False, 'axes.spines.right': False, 'axes.grid': True,
                     'grid.color': C_LIGHT, 'grid.linewidth': 0.8, 'axes.edgecolor': C_GREY,
                     'axes.labelcolor': C_INK, 'axes.titlesize': 11, 'xtick.color': C_GREY,
                     'ytick.color': C_GREY, 'lines.linewidth': 1.8, 'legend.frameon': False, 'figure.dpi': 110})

DATES = pd.date_range('2000-01-31', '2024-12-31', freq='ME')   # month index m = 0..299
FIRST_M, INIT_END_M, LAST_M = 36, 131, 299                      # 2003-01, 2010-12, 2024-12
OOS_M = np.arange(INIT_END_M + 1, LAST_M + 1)                    # 2011-01 .. 2024-12
REFIT_ENDS = np.arange(INIT_END_M, LAST_M, 12)                   # Decembers 2010 .. 2023
ROLL_LEN = 96
SUBPERIODS = {'2011-2017': (132, 215), '2018-2024': (216, 299)}
MACRO_LAG = 2                                                    # months; 1 is a robustness check
GRIDS = {'ridge': 10.0 ** np.linspace(3, -4, 36),               # alpha/n, most penalised first
         'lasso': 10.0 ** np.linspace(-0.5, -4.5, 33),          # sklearn's alpha (its loss is already per row)
         'pcr': [1, 2, 3, 4, 5, 6, 8, 10]}                       # number of principal components
RF_PARAMS = dict(n_estimators=300, max_features=1/3, min_samples_leaf=200, random_state=P3_SEED, n_jobs=N_JOBS)
COST_BP, HEADLINE_BP = [0, 5, 10, 25, 50], 10
PLACEBO_SHIFTS = [36, 48, 60, 72, 84, 96, 108, 120]
LEAK_ALARM = 0.02
assert (DATES[[FIRST_M, INIT_END_M, OOS_M[0], LAST_M]] == pd.to_datetime(['2003-01-31', '2010-12-31', '2011-01-31', '2024-12-31'])).all()
assert len(OOS_M) == 168 and len(REFIT_ENDS) == 14 and DATES[REFIT_ENDS[-1]] == pd.Timestamp('2023-12-31')

design = pd.Series({
    'first target month': DATES[FIRST_M].date(), 'initial training block': f'{DATES[FIRST_M].date()} .. {DATES[INIT_END_M].date()} (96 months)',
    'out-of-sample window': f'{DATES[OOS_M[0]].date()} .. {DATES[LAST_M].date()} ({len(OOS_M)} months)',
    'schemes': 'static (fit 2010-12) | expanding (14 December refits, primary) | rolling (96 months)',
    'validation': '3 annual folds inside each training window; lowest mean MSE; ties -> heavier penalty; refit on the window',
    'target': 'y = r / sigma-hat(36-month SD of own past excess returns, ddof=1)',
    'macro lag': f'{MACRO_LAG} months (robustness: 1)', 'primary forecast benchmark': 'pooled trailing mean of y through t-1',
    'headline cost': f'{HEADLINE_BP} bp per unit of one-way turnover (grid {COST_BP})',
    'placebo shifts (months)': PLACEBO_SHIFTS, 'seed': P3_SEED}, name='fixed before fitting')
display(design.to_frame())
import sys, sklearn
print(f'Python {sys.version.split()[0]}, pandas {pd.__version__}, numpy {np.__version__}, scikit-learn {sklearn.__version__}')

# ---------- The specification ledger: every candidate that will be scored out of sample ----------
SCHEMES = ['static', 'expanding', 'rolling']
LEDGER = ([('ols', 'M1', 'expanding', 'core')]
          + [(mdl, fs, sch, 'core') for mdl in ['ols', 'ridge', 'rf'] for fs in ['M2', 'M3'] for sch in SCHEMES]
          + [(mdl, 'M3', sch, 'core') for mdl in ['lasso', 'pcr'] for sch in SCHEMES]
          + [('ridge', 'M3-global', 'expanding', 'core'), ('ridge', 'M3-country', 'expanding', 'core'),
             ('ridge-per-class', 'M2', 'expanding', 'core'), ('ridge-per-class', 'M3', 'expanding', 'core')]
          + [('pcr-K3', 'M3', 'expanding', 'extra'), ('pcr-K5', 'M3', 'expanding', 'extra'),
             ('rf-deep', 'M2', 'expanding', 'extra'), ('rf-deep', 'M3', 'expanding', 'extra'),
             ('ridge', 'M3-lag1', 'expanding', 'extra'), ('ridge', 'M2-zscore', 'expanding', 'extra'),
             ('ridge', 'M2-levels', 'expanding', 'extra'), ('ridge', 'M3-levels', 'expanding', 'extra'),
             ('ridge', 'M3-x10', 'expanding', 'extra')])
ledger = pd.DataFrame(LEDGER, columns=['model', 'features', 'scheme', 'tier'])
ledger.index = [f'{m}|{f}|{s}' for m, f, s, _ in LEDGER]
assert len(ledger) == 38 and ledger.index.is_unique
T_OOS = len(OOS_M)
bonf38 = stats.t.ppf(1 - 0.025 / len(ledger), T_OOS - 1)
print(f'{len(ledger)} candidate specifications ({(ledger.tier == "core").sum()} core, {(ledger.tier == "extra").sum()} extras); '
      f'not candidates: {len(PLACEBO_SHIFTS)} placebo runs and one x10 leak demonstration.')
print(f'Critical |t| for the best of {len(ledger)} tests (Bonferroni, 5%, {T_OOS - 1} df): {bonf38:.2f}   vs a single test: {stats.t.ppf(0.975, T_OOS - 1):.2f}')
display(ledger.groupby(['tier', 'model']).size().rename('specifications').to_frame())'''

A1B = r'''# ---------- Power, before any result: how big an R2_OOS could this design detect? ----------
# Our construction from var(xbar) = sigma^2/n (L2 p.63): with y = s + e, var(e) ~ 1, var(s) = R2 and a perfect
# forecast, the monthly average loss gain has mean R2 and sd ~ 2R/sqrt(N_eff), so t ~ R * sqrt(N_eff * T) / 2.
R_wide = panel.pivot(index='date', columns='asset_id', values='excess_return')
cls_of = info['asset_class']
train_corr = R_wide.loc['2003-01-31':'2010-12-31'].corr()
rho = {}
for c in 'ABCD':
    ids = cls_of.index[cls_of == c]
    blk = train_corr.loc[ids, ids].to_numpy()
    rho[c] = blk[~np.eye(len(ids), dtype=bool)].mean()
n_c = cls_of.value_counts()
n_eff_class = sum(n_c[c] / (1 + (n_c[c] - 1) * rho[c]) for c in 'ABCD')
rho_bar = sum(n_c[c] * rho[c] for c in 'ABCD') / n_c.sum()
R2 = 0.003
power = pd.Series({
    'assumed true R2 (monthly)': R2, 'OOS months T': T_OOS,
    'average within-class correlation 2003-2010 (A/B/C/D)': ' / '.join(f'{rho[c]:.2f}' for c in 'ABCD'),
    't if the 50 assets were independent': np.sqrt(R2) * np.sqrt(50 * T_OOS) / 2,
    't for a within-class (ranking) signal: common class shocks cancel': np.sqrt(R2) * np.sqrt(50 * T_OOS / (1 - rho_bar)) / 2,
    'effective number of assets for a class-level signal (e.g. macro)': n_eff_class,
    't for a class-level signal': np.sqrt(R2) * np.sqrt(n_eff_class * T_OOS) / 2}, name='value')
display(power.to_frame())
print('Reading, fixed before results: a ranking signal of 0.3% is detectable; a class-level (macro) signal of the same size is not,\n'
      'so Q2 is read from magnitudes, bootstrap SEs and the placebo rather than from a single test.')'''

MD_DATA = r"""## 3.1 Know your data

The integrity checks and the audit of data traps come first. Every data fact the write-up quotes is printed in this section or in the section where it is first used. All descriptive statistics of returns alone, and of predictors alone, use the full sample. The one statistic that links predictors to future returns, the tercile sort, uses **only the training block 2003–2010**: running it on 2011–2024 before the design was fixed would let out-of-sample returns shape the design (L4 p.43; L5 p.41)."""

A2 = r'''# ---------- Integrity: shapes, keys, dates, classes (AI guide 6.1 and 6.4) ----------
wide = pd.read_csv(_DATA_DIR + 'asset_returns_wide.csv', parse_dates=['date'])
panel['aid'] = panel.asset_id.str.removeprefix('asset_').astype(int)
AIDS = np.sort(panel.aid.unique())
IDS = [f'asset_{a}' for a in AIDS]                       # natural order asset_1 .. asset_50
CLS = info.loc[IDS, 'asset_class'].to_numpy()             # class of each column, in IDS order
CTRY = info.loc[IDS, 'country'].to_numpy()
assert panel.shape == (15000, 11) and panel.drop(columns='aid').notna().all().all()
assert (pd.DatetimeIndex(np.sort(panel.date.unique())) == DATES).all() and not panel.duplicated(['asset_id', 'date']).any()
assert (panel.groupby('asset_id')[['asset_class', 'country']].nunique() == 1).all().all()
assert (panel.groupby('asset_id')[['asset_class', 'country']].first().loc[IDS].to_numpy() == info.loc[IDS, ['asset_class', 'country']].to_numpy()).all()
assert info.asset_class.value_counts().to_dict() == {'A': 26, 'C': 9, 'B': 8, 'D': 7}
assert (info.loc[info.asset_class == 'A', 'country'] == 'country_7').all()
R = panel.pivot(index='date', columns='asset_id', values='excess_return')[IDS]       # 300 x 50 excess returns
assert np.abs(wide.set_index('date')[IDS].to_numpy() - R.to_numpy()).max() == 0
assert mac_g.shape == (300, 5) and mac_g.notna().all().all() and (pd.DatetimeIndex(mac_g.date) == DATES).all()
for f in (mac_c, mac_x):
    assert len(f) == 3600 and not f.duplicated(['country', 'date']).any() and set(f.date) == set(DATES)
print('all integrity checks passed: 50 assets x 300 month-ends, no missing returns, files agree')
print('dtypes:', {c: str(t) for c, t in panel.dtypes.items() if c != 'aid'})
display(pd.crosstab(info.country, info.asset_class).loc[[f'country_{k}' for k in range(1, 13)]])'''

A3 = r'''# ---------- Trap audit 1: back-filled characteristics, frozen x1, the x5 floor ----------
XNAMES = ['x1', 'x2', 'x3', 'x4', 'x5']
X_RAW = {x: panel.pivot(index='date', columns='asset_id', values=x)[IDS] for x in XNAMES}
X_CLEAN = {x: W.copy() for x, W in X_RAW.items()}
def leading_copies(s):
    """Number of back-filled cells at the start of a series: a constant run from the first month is
    copies of the first genuine value, so all but its last cell are fills (0 if there is no run)."""
    run = 1
    while run < len(s) and s[run] == s[0]:
        run += 1
    return run - 1

fills = []
for x in ['x2', 'x3', 'x5']:                              # the notebook: leading months of x2, x3, x5 were filled
    for j, a in enumerate(IDS):
        k = leading_copies(X_RAW[x][a].to_numpy())
        if k:
            X_CLEAN[x].iloc[:k, j] = np.nan               # treat the copies as missing (a backward fill uses the future)
            fills.append({'variable': x, 'asset': a, 'class': CLS[j], 'months filled': k,
                          'last filled month': DATES[k - 1].date(), 'first genuine month': DATES[k].date()})
fill_tab = pd.DataFrame(fills)
display(fill_tab)
LAST_FILL = pd.Timestamp(max(fill_tab['last filled month']))
print(f'{len(fill_tab)} (variable, asset) pairs, {fill_tab["asset"].nunique()} assets; last filled month {LAST_FILL.date()} '
      f'-> the first target month {DATES[FIRST_M].date()} uses none of them, even through x3\'s one-month lag')

# x1 stops updating at the end of the sample for some class-B assets
frozen = []
for j, a in enumerate(IDS):
    s = X_RAW['x1'][a].to_numpy()
    run = 1
    while run < len(s) and s[-1 - run] == s[-1]:
        run += 1
    if run >= 3:
        frozen.append({'asset': a, 'class': CLS[j], 'constant stored months at the end': run,
                       'stale stored months': f'{DATES[-run + 1].date()} .. {DATES[-1].date()}'})
frozen_tab = pd.DataFrame(frozen)
display(frozen_tab)
# a stale value stored in month s is used for target s+1, and target 2025-01 does not exist
FROZEN_ROWS = [(a, DATES[m]) for a, k in zip(frozen_tab.asset, frozen_tab['constant stored months at the end'])
               for m in range(300 - k + 2, 300)]
print(f'target rows that use a stale x1: {len(FROZEN_ROWS)} (kept as stored: a desk would have seen them; excluded in one evaluation-only cut)')

# x5 is the 36-month volatility, floored at 0.004 for asset_16
SD36 = R.rolling(36, min_periods=36).std(ddof=1).shift(1)
floor = X_RAW['x5']['asset_16'] == 0.004
print(f'asset_16: x5 = 0.004 in {int(floor.sum())} months ({DATES[floor.to_numpy()][0].date()} .. {DATES[floor.to_numpy()][-1].date()}); '
      f'its own 36-month SD falls to {SD36.loc[floor, "asset_16"].min():.5f} there -> our sigma-hat is computed from returns, not from x5')'''

A4 = r'''# ---------- Trap audit 2: x10 is a same-year average (look-ahead); x11 stops at a regime change ----------
cm = mac_c.merge(mac_x[['country', 'date', 'x86', 'x85']], on=['country', 'date'], validate='one_to_one')
cm['year'] = cm.date.dt.year
yr = cm.groupby(['country', 'year']).agg(x10=('x10', 'first'), x10_values=('x10', 'nunique'), same_year=('x86', 'mean'))
yr['prior_year'] = yr.groupby(level='country')['same_year'].shift(1)
ok = yr.dropna()
x10_tab = pd.Series({
    'x10 constant within each calendar year (share of country-years)': (yr.x10_values == 1).mean(),
    'x10 = same-year mean of x86 within 0.001 (share)': (np.abs(yr.x10 - yr.same_year) < 1e-3).mean(),
    'x10 within 0.10 of the same-year mean (share)': (np.abs(yr.x10 - yr.same_year) < 0.10).mean(),
    'corr(x10, same-year mean of x86)': ok.x10.corr(ok.same_year),
    'corr(x10, prior-year mean of x86)': ok.x10.corr(ok.prior_year),
    'x10 closer to the same-year than the prior-year mean (share)': (np.abs(ok.x10 - ok.same_year) < np.abs(ok.x10 - ok.prior_year)).mean()},
    name='value')
display(x10_tab.to_frame())
print('-> x10 in January already "knows" the rest of the year: a 1-month lag leaves up to 11 months of look-ahead.\n'
      '   The monthly short rate x86 (extended file) is used instead, at the same 2-month lag as every macro series.')

# x11: where it stops, and how well x86 - x12 rebuilds it
x11_stop = cm.groupby('country').apply(lambda d: pd.Series({
    'last non-missing': d.loc[d.x11.notna(), 'date'].max().date(), 'missing months': int(d.x11.isna().sum()),
    'interior holes': int(d.x11.isna().sum() - (d.date > d.loc[d.x11.notna(), 'date'].max()).sum())}), include_groups=False)
cm['x11_rebuild'] = cm.x86 - cm.x12
ov = cm.dropna(subset=['x11'])
rb = ov.groupby('country').apply(lambda d: pd.Series({'corr': d.x11.corr(d.x11_rebuild),
                                                      'RMSE': np.sqrt(((d.x11 - d.x11_rebuild) ** 2).mean())}), include_groups=False)
stale = []
for c in x11_stop.index[x11_stop['missing months'] > 0]:
    d = cm[cm.country == c].sort_values('date')
    after = d.x11.isna()
    err = (d.x11.ffill() - d.x11_rebuild)[after].abs()
    stale.append({'country': c, 'stale forward-fill error, mean (pp)': err.mean(), 'max (pp)': err.max()})
x11_tab = x11_stop.join(rb).join(pd.DataFrame(stale).set_index('country'))
display(x11_tab.loc[x11_stop['missing months'].sort_values(ascending=False).index])
print(f'pooled rebuild quality on the overlap: corr {ov.x11.corr(ov.x11_rebuild):.3f}, RMSE {np.sqrt(((ov.x11 - ov.x11_rebuild) ** 2).mean()):.3f}. '
      'x11 is not added as a column: it has gaps (three countries stop early) and is nearly redundant with x86 - x12, '
      'whose two parts are both in the model.')

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 3.8))
d7 = cm[(cm.country == 'country_7') & (cm.date >= '2019-01-01') & (cm.date <= '2023-12-31')]
ax1.plot(d7.date, d7.x86, color=C_BLUE, label='x86: monthly short rate')
ax1.step(d7.date, d7.x10, where='post', color=C_ORANGE, label='x10: "curated" value (constant within the year)')
ax1.set_title('Figure 3.1a: x10 is the same-year average of x86 (country 7)')
ax1.set_ylabel('percent')
ax1.legend(fontsize=8)
for c, col in zip(['country_12', 'country_9', 'country_11'], [C_BLUE, C_ORANGE, C_AQUA]):
    d = cm[(cm.country == c) & (cm.date >= '2018-01-01')].sort_values('date')
    ax2.plot(d.date, d.x11_rebuild, color=col, label=f'{c}: rebuilt x86 - x12')
    ax2.plot(d.date, d.x11.ffill(), color=col, linestyle=':', linewidth=1.5)
ax2.set_title('Figure 3.1b: x11 (dotted = stale forward fill) vs its rebuild')
ax2.set_ylabel('real short rate, percent')
ax2.legend(fontsize=8)
plt.tight_layout()
plt.show()'''

A5 = r'''# ---------- Trap audit 3: the extended file (duplicates, hidden global series, frequencies, gaps) ----------
ext_cols = [c for c in mac_x.columns if c.startswith('x')]
cur_cols = [f'x{k}' for k in range(10, 17)]
mc, mx = mac_c.sort_values(['country', 'date']).reset_index(drop=True), mac_x.sort_values(['country', 'date']).reset_index(drop=True)
assert (mc[['country', 'date']].to_numpy() == mx[['country', 'date']].to_numpy()).all()     # rows align on keys
same = lambda a, b: bool(((a == b) | (a.isna() & b.isna())).all())
dups = [(c, e) for c in cur_cols for e in ext_cols if same(mc[c], mx[e])]
by_date = {e: mx.pivot(index='date', columns='country', values=e) for e in ext_cols}
hidden_global = [e for e in ext_cols if (by_date[e].nunique(axis=1, dropna=False) <= 1).all()]
glob_dups = [(g, e) for g in ['x6', 'x7', 'x8', 'x9'] for e in hidden_global if np.allclose(by_date[e].iloc[:, 0].to_numpy(), mac_g[g].to_numpy())]
def frequency(e):
    ch = by_date[e].diff().iloc[1:]
    months = set(ch.index[(ch.fillna(0) != 0).any(axis=1)].month)
    if not months: return 'constant'
    if months <= {1}: return 'annual (changes in January only)'
    if months <= {1, 4, 7, 10}: return 'quarterly'
    return 'monthly or irregular'
freq = pd.Series({e: frequency(e) for e in ext_cols})
gaps = [e for e in ext_cols if mx[e].isna().any()]
print(f'extended file: {len(ext_cols)} columns x{ext_cols[0][1:]}..{ext_cols[-1]}')
print('exact duplicates of curated columns (curated, extended):', dups)
print(f'{len(hidden_global)} columns identical across all 12 countries; exact copies of global series:', glob_dups)
print('frequency of the extended columns:', freq.value_counts().to_dict())
print('columns with gaps:', gaps)
print('-> the extended file is never concatenated or screened against returns; only x86 (the monthly short rate) is used.')'''

A6 = r'''# ---------- Returns: cumulative by class, per-asset table ----------
CLS_S = pd.Series(CLS, index=IDS)
class_ret = R.T.groupby(CLS_S).mean().T                   # equal-weighted class averages
fig, ax = plt.subplots(figsize=(8.5, 4))
for c in 'ABCD':
    ax.plot(class_ret.index, np.log((1 + class_ret[c]).cumprod()), color=CLASS_COL[c], label=f'class {c} ({(CLS == c).sum()} assets)')
ax.axvline(DATES[FIRST_M], color=C_GREY, linestyle=':', linewidth=1)
ax.axvline(DATES[OOS_M[0]], color=C_GREY, linestyle='--', linewidth=1)
ax.text(DATES[OOS_M[0]], ax.get_ylim()[1] * 0.95, ' out-of-sample from here', color=C_GREY, fontsize=8, va='top')
ax.set_ylabel('log cumulative excess return')
ax.set_title('Figure 3.2: cumulative excess return of each class (equal-weighted)')
ax.legend(fontsize=8, loc='upper left')
plt.show()

vol = R.std(ddof=1)
asset_tab = pd.DataFrame({'class': CLS, 'country': CTRY, 'ann. mean': 12 * R.mean(), 'ann. vol': np.sqrt(12) * vol,
                          'Sharpe (ann.)': np.sqrt(12) * R.mean() / vol, 'worst month': R.min(),
                          'worst month date': R.idxmin().dt.strftime('%Y-%m')}, index=IDS)
cls_summary = asset_tab.groupby('class').agg(assets=('country', 'size'), **{'median ann. vol': ('ann. vol', 'median'),
             'min ann. vol': ('ann. vol', 'min'), 'max ann. vol': ('ann. vol', 'max'), 'median Sharpe': ('Sharpe (ann.)', 'median'),
             'worst month': ('worst month', 'min')})
display(cls_summary.style.format('{:.3f}', subset=cls_summary.columns[1:]).set_caption('Table 3.1a: by class (monthly volatility x sqrt(12))'))
display(asset_tab.sort_values(['class', 'Sharpe (ann.)']).style.format({c: '{:.3f}' for c in ['ann. mean', 'ann. vol', 'Sharpe (ann.)', 'worst month']})
        .set_caption('Table 3.1b: every asset, 2000-2024'))
print(f'monthly volatility ranges from {vol.min():.2%} ({vol.idxmin()}) to {vol.max():.2%} ({vol.idxmax()})')'''

A7 = r'''# ---------- Correlation matrix ordered by class, and its block structure ----------
order = sorted(IDS, key=lambda a: ('ABCD'.index(CLS_S[a]), int(a.split('_')[1])))
corr_full = R[order].corr()
fig, ax = plt.subplots(figsize=(7.5, 6.3))
sns.heatmap(corr_full, vmin=-1, vmax=1, center=0, cmap='RdBu_r', square=True, xticklabels=False, yticklabels=False,
            cbar_kws={'label': 'correlation', 'shrink': 0.8}, ax=ax)
edges = np.cumsum([(CLS == c).sum() for c in 'ABCD'])
for e in edges[:-1]:
    ax.axhline(e, color=C_INK, linewidth=1)
    ax.axvline(e, color=C_INK, linewidth=1)
for c, mid in zip('ABCD', edges - np.array([(CLS == c).sum() for c in 'ABCD']) / 2):
    ax.text(-1.5, mid, c, ha='right', va='center', fontsize=10)
ax.set_xlabel('')
ax.set_ylabel('')
ax.set_title('Figure 3.3: return correlations, assets ordered by class (2000-2024)')
plt.show()

def block_table(C):
    """Mean off-diagonal correlation within and between classes."""
    out = pd.DataFrame(index=list('ABCD'), columns=list('ABCD'), dtype=float)
    for c1 in 'ABCD':
        for c2 in 'ABCD':
            b = C.loc[CLS_S.index[CLS_S == c1], CLS_S.index[CLS_S == c2]].to_numpy()
            out.loc[c1, c2] = b[~np.eye(len(b), dtype=bool)].mean() if c1 == c2 else b.mean()
    return out
display(block_table(corr_full).style.format('{:.2f}').set_caption('Table 3.2: average correlation within (diagonal) and between classes, 2000-2024'))
display(block_table(R.loc['2003-01-31':'2010-12-31'].corr()).style.format('{:.2f}').set_caption('Table 3.2 (training block 2003-2010)'))'''

A8 = r'''# ---------- Rolling 12-month Sharpe by class, and how persistent performance is ----------
roll_sr = np.sqrt(12) * class_ret.rolling(12).mean() / class_ret.rolling(12).std(ddof=1)
fig, ax = plt.subplots(figsize=(8.5, 3.8))
for c in 'ABCD':
    ax.plot(roll_sr.index, roll_sr[c], color=CLASS_COL[c], linewidth=1.3, label=f'class {c}')
ax.axhline(0, color=C_GREY, linewidth=1)
ax.set_ylabel('annualised Sharpe, trailing 12 months')
ax.set_title('Figure 3.4: rolling 12-month Sharpe ratio of each class')
ax.legend(fontsize=8, ncol=4, loc='lower left')
plt.show()
yearly = class_ret.groupby(class_ret.index.year).apply(lambda d: np.sqrt(12) * d.mean() / d.std(ddof=1))
persist = pd.Series({c: yearly[c].corr(yearly[c].shift(-1)) for c in 'ABCD'}, name='corr(Sharpe in year k, year k+1)')
display(persist.to_frame().T.style.format('{:.2f}').set_caption('Table 3.3: persistence of calendar-year Sharpe ratios (24 non-overlapping pairs)'))'''

A9 = r'''# ---------- Characteristics: scale, persistence, identities, and what each one measures ----------
LAGGED = {x: (X_CLEAN[x].shift(1) if x in ('x1', 'x3', 'x4') else X_CLEAN[x]) for x in XNAMES}   # known at the end of t-1
rows = []
for x in XNAMES:
    W = LAGGED[x].iloc[FIRST_M:]
    for c in 'ABCD':
        blk = W.loc[:, CLS == c]
        rows.append({'characteristic': x, 'class': c, 'mean': np.nanmean(blk), 'sd': np.nanstd(blk, ddof=1),
                     'p1': np.nanpercentile(blk, 1), 'p99': np.nanpercentile(blk, 99),
                     'AC(1) median': np.median([blk[a].autocorr(1) for a in blk]),
                     'AC(12) median': np.median([blk[a].autocorr(12) for a in blk]),
                     'cross-sectional sd (median month)': blk.std(axis=1, ddof=1).median()})
char_tab = pd.DataFrame(rows).set_index(['characteristic', 'class'])
display(char_tab.style.format('{:.3g}').set_caption('Table 3.4: forecast-aligned characteristics by class, 2003-2024'))

# identities: x2 is the compounded past 12-month return, x5 the past 36-month volatility (both already lagged)
R12 = np.expm1(np.log1p(R).rolling(12, min_periods=12).sum()).shift(1)
share_x2 = (np.abs(X_RAW['x2'] - R12).iloc[12:] < 1e-5).to_numpy().mean()
share_x5 = (np.abs(X_RAW['x5'] - SD36).iloc[FIRST_M:] < 1e-5).to_numpy().mean()
print(f'x2 = prod(1 + r[t-12..t-1]) - 1 within 1e-5 in {share_x2:.2%} of asset-months from 2001-01; '
      f'x5 = sd(r[t-36..t-1]) within 1e-5 in {share_x5:.2%} from 2003-01')
assert share_x2 > 0.99 and share_x5 > 0.99

# what the others track (predictor vs PAST returns and macro only, never the target)
R60 = np.log1p(R).rolling(60, min_periods=60).sum().shift(1)
def ts_corr(A, B, c):
    return np.nanmean([pd.concat([A[a], B[a]], axis=1).dropna().corr().iloc[0, 1] for a in np.array(IDS)[CLS == c]])
hyp = pd.DataFrame({c: {'x1 vs own 12m past return': ts_corr(LAGGED['x1'], R12, c),
                        'x3 vs own 60m past log return': ts_corr(LAGGED['x3'], R60, c),
                        'x4 vs own 12m past return': ts_corr(LAGGED['x4'], R12, c),
                        'x5 vs own 36m past vol': ts_corr(LAGGED['x5'], SD36, c)} for c in 'ABCD'})
rate_diff = cm.pivot(index='date', columns='country', values='x86')
rd = rate_diff.sub(rate_diff['country_7'], axis=0)
b_ids = np.array(IDS)[CLS == 'B']
x1_b = np.nanmean([X_RAW['x1'][a].corr(rd[info.loc[a, 'country']]) for a in b_ids])
x4_b = np.nanmean([X_RAW['x4'][a].corr(rd[info.loc[a, 'country']].diff(12)) for a in b_ids])
display(hyp.style.format('{:.2f}').set_caption('Table 3.5: average within-asset correlation with past returns, by class'))
print(f'class B: corr(x1, short-rate differential vs country 7) = {x1_b:.2f}; corr(x4, its 12-month change) = {x4_b:.2f}')
print('Hypotheses: x1 carry (rate differential / yield), x2 12-month momentum, x3 value / 5-year reversal,\n'
      '            x4 12-month change in the carry fundamental, x5 36-month volatility (the risk measure).')'''

A10 = r'''# ---------- A first look at predictive content: tercile sorts, TRAINING BLOCK ONLY (2003-2010) ----------
Y = R / SD36                                              # the vol-scaled target, defined from 2003-01
sort_months = range(FIRST_M, INIT_END_M + 1)
assert DATES[max(sort_months)] <= pd.Timestamp('2010-12-31')        # no out-of-sample return is used here
rows, terc_rows = [], {}
for x in XNAMES:
    spreads = {c: [] for c in 'ABCD'}
    levels_x = []
    for m in sort_months:
        for c in 'ABCD':
            cols = np.flatnonzero(CLS == c)
            v = LAGGED[x].iloc[m, cols].to_numpy()
            yy = Y.iloc[m, cols].to_numpy()
            k = len(cols) // 3
            o = np.argsort(v, kind='stable')               # ties broken by asset order
            spreads[c].append(yy[o[-k:]].mean() - yy[o[:k]].mean())
            levels_x.append([yy[o[:k]].mean(), yy[o[k:-k]].mean(), yy[o[-k:]].mean()])
    terc_rows[x] = dict(zip(['bottom', 'middle', 'top'], np.mean(levels_x, axis=0)))
    sp = pd.DataFrame(spreads)
    sp['all classes'] = sp.mean(axis=1)
    for col in sp:
        s = sp[col]
        rows.append({'characteristic': x, 'class': col, 'top minus bottom tercile, mean y': s.mean(),
                     't': s.mean() / (s.std(ddof=1) / np.sqrt(len(s)))})
sorts = pd.DataFrame(rows).pivot(index='characteristic', columns='class', values=['top minus bottom tercile, mean y', 't'])
display(sorts.style.format('{:.2f}').set_caption(f'Table 3.6: tercile spreads of next-month vol-scaled return, 2003-2010 ({len(sort_months)} months)'))
display(pd.DataFrame(terc_rows).T.style.format('{:.3f}')
        .set_caption('Table 3.6b: average next-month y in each within-class tercile, 2003-2010 (mean over classes and months)'))
print(f'critical |t|: {stats.t.ppf(0.975, len(sort_months) - 1):.2f} for one test, '
      f'{stats.t.ppf(1 - 0.025 / 25, len(sort_months) - 1):.2f} for the best of these 25 (Bonferroni; L4 p.27-28). '
      'No characteristic is dropped or re-signed because of this table.')'''

from cells_x import A9M, A10M
CELLS_A = [('md', MD_DESIGN), ('code', A1), ('code', A1B), ('md', MD_DATA), ('code', A2), ('code', A3),
           ('code', A4), ('code', A5), ('code', A6), ('code', A7), ('code', A8), ('code', A9), ('code', A9M), ('code', A10), ('code', A10M)]
