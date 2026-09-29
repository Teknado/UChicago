"""Rebuild the panel's monthly excess returns from downloaded data, keep the assets that match, extend them past 2024.

Run from the repository root:  python3 review/holdout_2025/rebuild.py            (uses raw/ from fetch.py)
                               python3 review/holdout_2025/rebuild.py --selftest (synthetic raw data: tests the code)
Implements DESIGN.md §2-§3: fixed candidate constructions per class; the best candidate on 2019-01..2024-12 is kept if
its correlation with the panel is at least 0.95; unmatched assets are dropped, never repaired.
Writes data/match_table.csv and data/extended_returns.csv (panel values through 2024-12, rebuilt values after).
"""
import os, sys, glob, tempfile
import numpy as np, pandas as pd

SELFTEST = '--selftest' in sys.argv
HERE = 'review/holdout_2025'
RAW = tempfile.mkdtemp() if SELFTEST else f'{HERE}/raw'
OUT = tempfile.mkdtemp() if SELFTEST else f'{HERE}/data'
os.makedirs(OUT, exist_ok=True)
ACCEPT, OVERLAP = 0.95, ('2019-01-31', '2024-12-31')           # fixed in DESIGN.md before any holdout data
MAX_RATE_FILL = 3                                                # months a lagging FRED rate may be carried forward

panel = pd.read_csv('asset_panel.csv', parse_dates=['date'])
R_panel = panel.pivot(index='date', columns='asset_id', values='excess_return')
info = pd.read_csv('asset_info.csv').set_index('asset_id')

EQUITY = {'asset_24': [('^GSPC', 'US', 'SPY')], 'asset_12': [('^N225', 'JP', 'EWJ')], 'asset_50': [('^SSMI', 'CH', 'EWL')],
          'asset_38': [('^GDAXI', 'EZ', 'EWG')], 'asset_9': [('^FTSE', 'GB', 'EWU')], 'asset_3': [('^GSPTSE', 'CA', 'EWC')],
          'asset_10': [('^FCHI', 'EZ', 'EWQ')], 'asset_45': [('FTSEMIB.MI', 'EZ', 'EWI')],
          'asset_42': [('^AEX', 'EZ', 'EWN'), ('^IBEX', 'EZ', 'EWP')]}
FX = {'asset_8': ('JPY=X', 'inverse', 'JP'), 'asset_34': ('CHF=X', 'inverse', 'CH'), 'asset_22': ('EURUSD=X', 'direct', 'EZ'),
      'asset_31': ('GBPUSD=X', 'direct', 'GB'), 'asset_30': ('CAD=X', 'inverse', 'CA'), 'asset_46': ('SEK=X', 'inverse', 'SE'),
      'asset_17': ('AUDUSD=X', 'direct', 'AU'), 'asset_44': ('NZDUSD=X', 'direct', 'NZ')}
BONDS = {'asset_27': ('US', 'US'), 'asset_16': ('JP', 'JP'), 'asset_41': ('CH', 'CH'), 'asset_21': ('DE', 'EZ'),
         'asset_25': ('GB', 'GB'), 'asset_35': ('CA', 'CA'), 'asset_49': ('SE', 'SE')}
NAMED_COMMODITIES = {'asset_33': 'CL=F', 'asset_15': 'BZ=F', 'asset_6': 'GC=F'}
COMMODITY_POOL = ['NG=F', 'HO=F', 'RB=F', 'SI=F', 'PL=F', 'PA=F', 'HG=F', 'ALI=F', 'ZC=F', 'ZS=F', 'ZW=F', 'KE=F', 'ZL=F',
                  'ZM=F', 'ZO=F', 'KC=F', 'SB=F', 'CT=F', 'CC=F', 'OJ=F', 'LE=F', 'HE=F', 'GF=F']

def fname(t): return f'yahoo_{t.replace("^", "IDX_").replace("=", "_")}.csv'

# ---------------------------------------------------------------- self-test: synthetic raw files built from the panel
if SELFTEST:
    rs = np.random.default_rng(0)
    idx = pd.date_range('2000-01-31', '2026-08-31', freq='ME')
    extra = lambda s: pd.concat([s, pd.Series(rs.normal(0, s.std(), len(idx) - len(s)), index=idx[len(s):])])
    rate = pd.Series(2.0, index=idx)                             # every short rate 2% a year
    for s in ['TB3MS'] + [f'IR3TIB01{c}M156N' for c in ['US', 'JP', 'CH', 'EZ', 'GB', 'CA', 'SE', 'AU', 'NZ']]:
        pd.DataFrame({'date': idx, 'value': rate.values}).to_csv(f'{RAW}/fred_{s}.csv', index=False)
    for c in ['US', 'JP', 'CH', 'DE', 'GB', 'CA', 'SE']:          # yields as a random walk: bonds should be rejected
        pd.DataFrame({'date': idx, 'value': 3 + np.cumsum(rs.normal(0, 0.2, len(idx)))}).to_csv(f'{RAW}/fred_IRLTLT01{c}M156N.csv', index=False)
    def price(r): return 100 * np.cumprod(1 + r.values)
    for a, cands in EQUITY.items():                              # the index price return = panel excess + 2%/12: variant (i)
        r = extra(R_panel[a]) + 0.02 / 12
        for t, _, etf in cands:
            for tick in (t, etf):
                pd.DataFrame({'date': idx, 'close': price(r), 'adjclose': price(r)}).to_csv(f'{RAW}/{fname(tick)}', index=False)
    for a, (t, how, _) in FX.items():                             # spot only: variant (i)
        p = price(extra(R_panel[a]))
        p = 1 / p if how == 'inverse' else p
        pd.DataFrame({'date': idx, 'close': p, 'adjclose': p}).to_csv(f'{RAW}/{fname(t)}', index=False)
    pd.DataFrame({'date': idx, 'close': price(pd.Series(rs.normal(0, .02, len(idx)))), 'adjclose': 1}).to_csv(f'{RAW}/{fname("ZN=F")}', index=False)
    A_ids = [a for a in info.index if info.loc[a, 'asset_class'] == 'A' and a not in NAMED_COMMODITIES]
    for a, t in NAMED_COMMODITIES.items():
        p = price(extra(R_panel[a])); pd.DataFrame({'date': idx, 'close': p, 'adjclose': p}).to_csv(f'{RAW}/{fname(t)}', index=False)
    for a, t in zip(A_ids[:10], COMMODITY_POOL[::-1]):           # ten unnamed commodities hidden under shuffled tickers
        p = price(extra(R_panel[a])); pd.DataFrame({'date': idx, 'close': p, 'adjclose': p}).to_csv(f'{RAW}/{fname(t)}', index=False)

# ---------------------------------------------------------------- loading
def yahoo(t, col='close'):
    f = f'{RAW}/{fname(t)}'
    if not os.path.exists(f):
        return None
    d = pd.read_csv(f, parse_dates=['date']).set_index('date')[col].astype(float)
    return d[~d.index.duplicated(keep='last')]
def fred(s):
    f = f'{RAW}/fred_{s}.csv'
    if not os.path.exists(f):
        return None
    d = pd.read_csv(f, parse_dates=['date']).set_index('date')['value'].astype(float)
    full = pd.date_range(d.index.min(), max(d.index.max(), pd.Timestamp('2026-12-31')), freq='ME')
    return d.reindex(full).ffill(limit=MAX_RATE_FILL)            # a lagging release may be carried forward 3 months
def ret(p): return p.pct_change() if p is not None else None
def short(c): return fred(f'IR3TIB01{c}M156N')
TB = fred('TB3MS')

# ---------------------------------------------------------------- candidate constructions (DESIGN.md §2)
cands = {}                                                        # asset -> {candidate name: monthly return series}
for a, lst in EQUITY.items():
    cands[a] = {}
    for t, c, etf in lst:
        px = ret(yahoo(t)); s = short(c); etf_r = ret(yahoo(etf, 'adjclose'))
        if px is not None and TB is not None:
            cands[a][f'{t} price - US bill'] = px - TB.shift(1).reindex(px.index) / 1200
        if px is not None and s is not None:
            cands[a][f'{t} price - local short rate'] = px - s.shift(1).reindex(px.index) / 1200
        if etf_r is not None and TB is not None:
            cands[a][f'{etf} total return (USD) - US bill'] = etf_r - TB.shift(1).reindex(etf_r.index) / 1200
for a, (t, how, c) in FX.items():
    p = yahoo(t)
    if p is None:
        continue
    spot = (p.shift(1) / p - 1) if how == 'inverse' else p.pct_change()
    cands[a] = {f'{t} spot': spot}
    sf, su = short(c), short('US')
    if sf is not None and su is not None:
        cands[a][f'{t} spot + rate differential'] = spot + (sf - su).shift(1).reindex(spot.index) / 1200
for a, (yc, sc) in BONDS.items():
    y = fred(f'IRLTLT01{yc}M156N'); s = short(sc)
    cands[a] = {}
    if y is not None and s is not None:
        yd = y / 100
        D = ((1 - (1 + yd / 2) ** -20) / yd.where(yd.abs() > 1e-4)).fillna(10.0)   # modified duration, 10-year par bond
        cands[a][f'{yc} 10y yield, duration approximation'] = (yd.shift(1) / 12 - D.shift(1) * yd.diff() - s.shift(1) / 1200)
    if a == 'asset_27':
        zn = ret(yahoo('ZN=F'))
        if zn is not None:
            cands[a]['ZN=F futures'] = zn
for a, t in NAMED_COMMODITIES.items():
    r = ret(yahoo(t))
    if r is not None:
        cands[a] = {f'{t} front month': r}

# ---------------------------------------------------------------- matching on 2019-2024
def stats_(a, s):
    both = pd.concat([R_panel[a], s], axis=1, join='inner').loc[OVERLAP[0]:OVERLAP[1]].dropna()
    if len(both) < 60:
        return None
    x, y = both.iloc[:, 0], both.iloc[:, 1]
    return {'months': len(both), 'corr': x.corr(y), 'RMSE': np.sqrt(((x - y) ** 2).mean()), 'mean diff (rebuilt - panel)': (y - x).mean()}
rows, chosen = [], {}
for a, cs in cands.items():
    best = None
    for name, s in cs.items():
        st = stats_(a, s)
        if st:
            rows.append({'asset_id': a, 'class': info.loc[a, 'asset_class'], 'candidate': name, **st})
            if best is None or st['corr'] > best[1]['corr']:
                best = (name, st)
    if best and best[1]['corr'] >= ACCEPT:
        chosen[a] = (best[0], cs[best[0]])
# unnamed commodities: greedy one-to-one assignment by correlation
A_left = [a for a in info.index if info.loc[a, 'asset_class'] == 'A' and a not in NAMED_COMMODITIES]
pool = {t: ret(yahoo(t)) for t in COMMODITY_POOL}
pool = {t: s for t, s in pool.items() if s is not None}
pairs = []
for a in A_left:
    for t, s in pool.items():
        st = stats_(a, s)
        if st:
            pairs.append((st['corr'], a, t, st))
used_a, used_t = set(), set()
for c, a, t, st in sorted(pairs, reverse=True):
    if a in used_a or t in used_t:
        continue
    rows.append({'asset_id': a, 'class': 'A', 'candidate': f'{t} front month (matched)', **st})
    if c >= ACCEPT:
        chosen[a] = (f'{t} front month (matched)', pool[t])
    used_a.add(a); used_t.add(t)
M = pd.DataFrame(rows)
M['chosen'] = [chosen.get(a, (None,))[0] == cn for a, cn in zip(M.asset_id, M.candidate)]
M.sort_values(['class', 'asset_id', 'corr'], ascending=[True, True, False]).to_csv(f'{OUT}/match_table.csv', index=False)

# ---------------------------------------------------------------- the extended panel (DESIGN.md §3)
new = pd.DataFrame({a: s for a, (_, s) in chosen.items()}).loc['2025-01-31':]
if len(chosen):
    have = new.notna().mean(axis=1)
    end = have[have >= 0.9].index.max()                          # the last month at which at least 90% have data
    new = new.loc[:end]
    keep = [a for a in new if new[a].notna().all()]               # an asset missing any holdout month is dropped
    dropped = sorted(set(new) - set(keep))
    ext = pd.concat([R_panel[keep], new[keep]]).sort_index()
    ext.index.name = 'date'
    ext.to_csv(f'{OUT}/extended_returns.csv')
    print(f'accepted {len(chosen)} of 50 assets; kept {len(keep)} with data for every month 2025-01..{end.date()}; dropped for gaps: {dropped}')
    print(pd.Series({a: info.loc[a, 'asset_class'] for a in keep}).value_counts().sort_index().to_dict())
print(M[M.chosen].groupby('class').agg(n=('asset_id', 'size'), min_corr=('corr', 'min')).to_string())
if SELFTEST:
    names = M[M.chosen].set_index('asset_id').candidate
    assert set(names.index) >= set(EQUITY) | set(FX) | set(NAMED_COMMODITIES), 'self-test: a constructed match was missed'
    assert not (set(names.index) & set(BONDS)) or all(M[(M.asset_id.isin(BONDS)) & M.chosen]['corr'] >= ACCEPT)
    assert (names[list(EQUITY)].str.contains('US bill')).all() and names[list(FX)].str.endswith('spot').all()
    assert sum(names.index.isin(A_left)) == 10, 'self-test: the ten hidden commodities should all be found'
    print('SELF-TEST PASSED')
