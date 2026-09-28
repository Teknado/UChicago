"""Check the internally verifiable claims of gemini/asset_and_country_identities.md against the data files.
External benchmark numbers (e.g. 'S&P 500 TR - Rf = -16.80%') cannot be checked from the files and are not checked here.
Run from the repository root: python3 review/checks/identity_claims_check.py"""
import numpy as np, pandas as pd
P = pd.read_csv('asset_panel.csv', parse_dates=['date']).sort_values(['asset_id', 'date'])
info = pd.read_csv('asset_info.csv')
G = pd.read_csv('macro_global.csv', parse_dates=['date'])
C = pd.read_csv('macro_country.csv', parse_dates=['date'])
X = pd.read_csv('macro_country_extended.csv', parse_dates=['date'])
W = P.pivot(index='date', columns='asset_id', values='excess_return')

print('=== Section 1: class statistics. Which definition reproduces the document\'s numbers? ===')
doc = {'A': (6.02, 30.97, 0.194, -54.65), 'B': (-0.24, 10.19, -0.023, -16.29), 'C': (4.77, 16.95, 0.282, -25.36), 'D': (2.68, 6.63, 0.405, -12.08)}
for c in 'ABCD':
    ids = P.loc[P.asset_class == c, 'asset_id'].unique()
    port = W[ids].mean(axis=1)
    st = pd.DataFrame({'m': W[ids].mean() * 12, 'v': W[ids].std() * np.sqrt(12)}); st['s'] = st.m / st.v
    print(f"{c}: doc mean/vol/Sharpe/worst = {doc[c]} | EW class portfolio: {100*port.mean()*12:.2f} / {100*port.std()*np.sqrt(12):.2f} / "
          f"{port.mean()/port.std()*np.sqrt(12):.3f} / {100*port.min():.2f} | per-asset mean of vol {100*st.v.mean():.2f}, max vol {100*st.v.max():.2f}, "
          f"mean Sharpe {st.s.mean():.3f}, median Sharpe {st.s.median():.3f}, worst single asset-month {100*W[ids].min().min():.2f}")
print('pooled per-asset-month mean check for Sharpe (mean of all returns / sd of all returns):',
      {c: round(float(P.loc[P.asset_class == c, 'excess_return'].mean() / P.loc[P.asset_class == c, 'excess_return'].std() * np.sqrt(12)), 3) for c in 'ABCD'},
      '| pooled vol:', {c: round(float(100 * P.loc[P.asset_class == c, 'excess_return'].std() * np.sqrt(12)), 2) for c in 'ABCD'})

print('\n=== Section 2: country composition (A, B, C, D counts) ===')
comp = pd.crosstab(info['country'], info['asset_class'])
print(comp.to_string())

print('\n=== Section 3: characteristics and macro traps ===')
P['sd36'] = P.groupby('asset_id')['excess_return'].transform(lambda s: s.shift(1).rolling(36).std(ddof=1))
P['c12'] = P.groupby('asset_id')['excess_return'].transform(lambda s: s.shift(1).rolling(12).apply(lambda r: np.prod(1 + r) - 1, raw=True))
ok = P.dropna(subset=['sd36'])
print(f"corr(x5, 36m sd of returns to t-1) = {ok.x5.corr(ok.sd36):.4f}; share within 1e-6: {(np.abs(ok.x5 - ok.sd36) < 1e-6).mean():.2%}")
ok = P.dropna(subset=['c12'])
print(f"corr(x2, compounded return t-12..t-1) = {ok.x2.corr(ok.c12):.4f}; share within 1e-6: {(np.abs(ok.x2 - ok.c12) < 1e-6).mean():.2%}")
a16 = P[P.asset_id == 'asset_16']
print(f"asset_16: months with x5 == 0.004: {(a16.x5 == 0.004).sum()}, from {a16.loc[a16.x5 == 0.004, 'date'].min().date()} to {a16.loc[a16.x5 == 0.004, 'date'].max().date()}")
M = C.merge(X[['country', 'date', 'x86']], on=['country', 'date'])
M['year'] = M.date.dt.year
print(f"x10 constant within every country-year: {(M.groupby(['country', 'year']).x10.nunique() == 1).all()} ({M.groupby(['country','year']).ngroups} country-years)")
M['x86_same_year'] = M.groupby(['country', 'year']).x86.transform('mean')
print(f"corr(x10, same-year mean of x86) = {M.x10.corr(M.x86_same_year):.4f}")
v = M.dropna(subset=['x11'])
print(f"corr(x11, x86 - x12) = {v.x11.corr(v.x86 - v.x12):.4f}; x11 missing values: {int(C.x11.isna().sum())}; last x11 date by country (stopping early):",
      {c: str(g.dropna(subset=['x11']).date.max().date()) for c, g in C.groupby('country') if g.x11.isna().any()})
print(f"x6: 2008-10 value {G.set_index('date').x6.loc['2008-10'].iloc[0]:.2f}, 2020-03 value {G.set_index('date').x6.loc['2020-03'].iloc[0]:.2f}, skewness {G.x6.skew():.2f}")
print(f"x8: 2009-03 {G.set_index('date').x8.loc['2009-03'].iloc[0]:.2f}, 2020-03 {G.set_index('date').x8.loc['2020-03'].iloc[0]:.2f}; mean {G.x8.mean():.2f}, sd {G.x8.std():.2f}")
print(f"x7 months above 1.20: {[str(d.date()) for d in G.loc[G.x7 > 1.2, 'date']]}")

print('\n=== Section 4: event returns quoted in the document (panel values) ===')
ev = [('asset_24', '2008-10', -16.78), ('asset_12', '2008-10', -25.36), ('asset_50', '2008-10', -8.45), ('asset_8', '2008-10', 7.39),
      ('asset_17', '2008-10', -16.29), ('asset_27', '2008-11', 11.97), ('asset_34', '2015-01', 7.94), ('asset_34', '2015-02', -3.09),
      ('asset_22', '2015-01', -6.77), ('asset_30', '2015-01', -8.59), ('asset_31', '2015-01', -3.66), ('asset_31', '2016-06', -8.18),
      ('asset_8', '2016-06', 7.98), ('asset_22', '2016-06', -0.32), ('asset_25', '2022-09', -12.08), ('asset_25', '2022-10', 4.69),
      ('asset_27', '2022-09', -6.61), ('asset_21', '2022-09', -5.81), ('asset_16', '2022-09', -0.58), ('asset_33', '2020-03', -54.65),
      ('asset_15', '2020-03', -54.21), ('asset_6', '2008-09', 5.17), ('asset_6', '2011-08', 12.23)]
bad = []
for a, d, claim in ev:
    val = 100 * W.loc[d, a].iloc[0]
    flag = '' if abs(val - claim) < 0.01 else '  <-- MISMATCH'
    if flag: bad.append((a, d, claim, round(val, 2)))
    print(f"{a} {d}: document {claim:+.2f}%  panel {val:+.2f}%{flag}  (class {info.set_index('asset_id').asset_class[a]}, {info.set_index('asset_id').country[a]})")
print('mismatches:', bad)
D = [a for a in W if info.set_index('asset_id').asset_class[a] == 'D']
wd = W[D].stack(); print(f"worst class-D asset-month: {wd.idxmin()} {100*wd.min():.2f}%")
print(f"worst class-A asset-month: {W[[a for a in W if info.set_index('asset_id').asset_class[a]=='A']].stack().idxmin()}")
print(f"asset_24 is the only class-C asset of country_7: {info[(info.country=='country_7')&(info.asset_class=='C')].asset_id.tolist()}, "
      f"class D of country_7: {info[(info.country=='country_7')&(info.asset_class=='D')].asset_id.tolist()}")
