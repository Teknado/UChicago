"""Problem 3, stage 2: feature engineering (one pure function of the raw files) and its checks."""

MD_FEAT = r"""## 3.2 Feature engineering

One function, `build_features`, turns the raw files into every predictor for every (month, asset). A value in the row for target month $t$ uses only information available at the end of month $t-1$. Because the whole construction is a single function of the raw files, three things run through the same code:
- the causality test (rebuild from data truncated at 2010-12; nothing up to 2010-12 may change);
- the placebo (macro circularly shifted in time);
- the `x10` variants.

| block | construction | why |
|---|---|---|
| lags | `x2`, `x5` as stored (already lagged); `x1`, `x3`, `x4` shifted 1 month within asset; every macro series shifted **2 months** (1 month minimum plus release delay; 1 month as robustness) | the exam's Data section; merge on publication date, not period end (L5 p.58 item 2) |
| characteristics `u1..u5` | rank within asset class and month, scaled to [−0.5, 0.5] | Units differ by up to 3 orders of magnitude across classes. Ranks are bounded, so no single characteristic value can have high leverage (leverage depends on the inputs, L3 p.11–12; the target itself is not bounded). They have no parameters fitted over time, so they cannot leak (L5 p.58). Within-class z-score as robustness (our choice). |
| levels `l1..l5` (secondary) | each asset's own lagged characteristic as a trailing 60-month z-score | Ranks can only compare assets within a month. This gives the characteristics a time-series channel, so macro is not the only source of timing. |
| global macro | log `x6`; trailing 60-month z-score (at least 24 months), then lag; **one slope per class** | "trailing information only" (the exam's suggested workflow); interactions (L3 p.53–54) |
| country macro | `x86` (extended; replaces the look-ahead `x10`), `x12`, `x13`, log `x14`, log `x15`, `x16`: each as the **differential vs country 7**, then a **trailing 60-month z-score per country**, then lag. Enters × class for B, C, D. **Class A gets none**, and assets in country 7 have a zero differential by construction. | Differential vs the natural base (the exam's suggested workflow). The trailing z-score removes static country gaps, so a fixed asset premium cannot pose as macro. Class A has no natural country (the exam's Data section). |
| class intercepts | left **unpenalised**: linear models are fit on within-class demeaned data | the ridge objective penalises the slopes it contains (L5 p.12); keeping the class intercepts out of the penalty is our choice; class intercepts (L3 p.50) |"""

B1 = r'''CHARS = ['u1', 'u2', 'u3', 'u4', 'u5']
LEVELS = ['l1', 'l2', 'l3', 'l4', 'l5']
G_SER = ['x6', 'x7', 'x8', 'x9']
C_SER = [('x86', 'extended', False), ('x12', 'curated', False), ('x13', 'curated', False),
         ('x14', 'curated', True), ('x15', 'curated', True), ('x16', 'curated', False)]   # (series, file, take logs)
COUNTRIES = [f'country_{k}' for k in range(1, 13)]
C_IDX = np.array([COUNTRIES.index(c) for c in CTRY])                  # each asset's country column

def trailing_z(W, window=60, min_periods=24):
    """z-score of each value against the trailing window that ends at it (past and present only)."""
    sd = W.rolling(window, min_periods=min_periods).std(ddof=1)
    return (W - W.rolling(window, min_periods=min_periods).mean()) / sd.where(sd > 1e-12)

def build_features(pan, mg, mc, mx, macro_lag=MACRO_LAG, char_std='rank', short_rate='x86', sr_lag=None, shift=0):
    """Every predictor for every (month, asset), known at the end of month t-1, from the raw files only.
    shift > 0 circularly shifts every raw macro series by `shift` months first (the placebo)."""
    dates = pd.DatetimeIndex(np.sort(pan.date.unique()))
    T, N = len(dates), len(IDS)
    Rw = pan.pivot(index='date', columns='asset_id', values='excess_return')[IDS]
    sig = Rw.rolling(36, min_periods=36).std(ddof=1).shift(1)          # sigma-hat_{t-1} from r_{t-36..t-1}
    lagged = {}
    for x in XNAMES:
        W = pan.pivot(index='date', columns='asset_id', values=x)[IDS].copy()
        if x in ('x2', 'x3', 'x5'):                                     # back-filled leading cells -> missing
            for j in range(N):
                k = leading_copies(W.iloc[:, j].to_numpy())
                if k:
                    W.iloc[:k, j] = np.nan
        lagged[x] = W.shift(1) if x in ('x1', 'x3', 'x4') else W       # x2 and x5 are stored already lagged
    feats = {}
    for x, name in zip(XNAMES, CHARS):                                  # within-class, within-month standardisation
        out = pd.DataFrame(np.nan, index=dates, columns=IDS)
        for c in 'ABCD':
            cols = [a for a, k in zip(IDS, CLS) if k == c]
            B = lagged[x][cols]
            if char_std == 'rank':
                out[cols] = (B.rank(axis=1, method='average') - 1) / (len(cols) - 1) - 0.5
            else:
                out[cols] = B.sub(B.mean(axis=1), axis=0).div(B.std(axis=1, ddof=1), axis=0).clip(-3, 3)
        feats[name] = out.to_numpy()
    for x, name in zip(XNAMES, LEVELS):                                 # own-history levels (0 until 24 months exist)
        feats[name] = trailing_z(lagged[x]).fillna(0).clip(-5, 5).to_numpy()

    roll = (lambda A: np.roll(A, shift, axis=0)) if shift else (lambda A: A)
    G = mg.set_index('date')[G_SER].copy()
    G['x6'] = np.log(G['x6'])
    G = pd.DataFrame(roll(G.to_numpy()), index=G.index, columns=G_SER)
    GZ = trailing_z(G).shift(macro_lag)
    CZ = {}
    for ser, src, take_log in C_SER:
        name, lag = ser, macro_lag
        if ser == 'x86' and short_rate == 'x10':                        # curated-file variant (and its leak demo)
            ser, src, lag = 'x10', 'curated', sr_lag
        P = (mx if src == 'extended' else mc).pivot(index='date', columns='country', values=ser)[COUNTRIES]
        P = np.log(P) if take_log else P
        P = pd.DataFrame(roll(P.to_numpy()), index=P.index, columns=COUNTRIES)
        D = P.sub(P['country_7'], axis=0)                               # differential vs country 7
        Z = trailing_z(D)
        Z['country_7'] = 0.0                                            # country 7 has no differential
        CZ[name] = Z.shift(lag)

    F = pd.DataFrame({'date': np.repeat(dates, N), 'm': np.repeat(np.arange(T), N), 'asset_id': np.tile(IDS, T),
                      'cls': np.tile(CLS, T), 'country': np.tile(CTRY, T),
                      'r': Rw.to_numpy().ravel(), 'sig': sig.to_numpy().ravel()})
    F['y'] = F.r / F.sig
    for name, A in feats.items():
        F[name] = A.ravel()
    for c in 'BCD':
        F['d' + c] = (F.cls == c).astype(float)
    for k in G_SER:
        F['g_' + k] = np.repeat(GZ[k].to_numpy(), N)
        for c in 'ABCD':
            F[f'g_{k}_{c}'] = F['g_' + k] * (F.cls == c)
    for k, _, _ in C_SER:
        F['c_' + k] = CZ[k].to_numpy()[:, C_IDX].ravel()
        F.loc[F.cls == 'A', 'c_' + k] = 0.0                             # class A: no country macro
        for c in 'BCD':
            F[f'c_{k}_{c}'] = F['c_' + k] * (F.cls == c)
    F = F[F.m >= FIRST_M].reset_index(drop=True)
    if short_rate == 'x10' and sr_lag and sr_lag > 12:
        # x10 at a 14-month lag first has a 24-month trailing z for target 2003-02; the 50 rows of 2003-01 get 0,
        # the neutral value (a fixed constant, not an estimate), in this sensitivity variant only
        xcols = [c for c in F if c.startswith('c_x86')]
        F[xcols] = F[xcols].fillna(0.0)
    return F, {'lagged': lagged, 'GZ': GZ, 'CZ': CZ, 'G': G}

GLOB = [f'g_{k}_{c}' for k in G_SER for c in 'ABCD']
CTRY_INT = [f'c_{k}_{c}' for k, _, _ in C_SER for c in 'BCD']
FSETS = {'M1': [], 'M2': CHARS, 'M3': CHARS + GLOB + CTRY_INT, 'M3-global': CHARS + GLOB, 'M3-country': CHARS + CTRY_INT,
         'M2-levels': CHARS + LEVELS, 'M3-levels': CHARS + GLOB + CTRY_INT + LEVELS}
RF_SETS = {'M2': CHARS + ['dB', 'dC', 'dD'],
           'M3': CHARS + ['dB', 'dC', 'dD'] + ['g_' + k for k in G_SER] + ['c_' + k for k, _, _ in C_SER]}'''

B2 = r'''t0 = time.perf_counter()
F, AUX = build_features(panel, mac_g, mac_c, mac_x)
print(f'feature frame: {F.shape[0]:,} rows ({F.m.nunique()} target months x {F.asset_id.nunique()} assets), built in {time.perf_counter() - t0:.1f} s')
model_cols = sorted(set(sum(FSETS.values(), []) + sum(RF_SETS.values(), [])))
assert len(F) == 264 * 50 and F[model_cols + ['y', 'sig']].notna().all().all(), 'a predictor is missing somewhere'
assert (F.groupby(['m', 'cls'])[CHARS].sum().abs() < 1e-12).all().all()          # ranks sum to zero in every class-month
assert F.loc[F.cls == 'A', [c for c in F if c.startswith('c_')]].eq(0).all().all()

# Table 3.7: every predictor and the count (the exam's suggested workflow)
pred_tab = pd.DataFrame([
    ['characteristics', 'u1..u5', 'x1 (lag 1), x2 (as stored), x3 (lag 1), x4 (lag 1), x5 (as stored) -> rank within class and month', 'all', 5],
    ['class intercepts', 'A (reference), dB, dC, dD', 'asset_info; unpenalised', 'all', 3],
    ['global macro x class', 'g_x6..g_x9 x {A,B,C,D}', 'log x6, x7, x8, x9 -> trailing 60m z -> lag 2', 'all', len(GLOB)],
    ['country macro x class', '{x86, x12, x13, log x14, log x15, x16} x {B,C,D}', 'minus country 7 -> trailing 60m z per country -> lag 2', 'B, C, D', len(CTRY_INT)],
    ['levels (secondary only)', 'l1..l5', "each asset's own lagged characteristic -> trailing 60m z", 'all', 5],
    ['target', 'y', 'excess return / 36m SD of own past excess returns', 'all', 1]],
    columns=['block', 'columns', 'source -> transform -> lag', 'classes', 'count']).set_index('block')
display(pred_tab.style.set_caption('Table 3.7: predictors'))
display(pd.Series({k: len(v) + 3 for k, v in FSETS.items()}, name='columns incl. the 3 class intercepts').to_frame().T)
print('Not used: x10/x73 (same-year average: look-ahead), stored x11 (gaps; nearly redundant with x86 - x12, whose parts are included),'
      ' every other extended column (duplicates, hidden global copies, annual/quarterly series whose publication timing is unknown, gaps).')'''

B3 = r'''# ---------- Checks that the features only use the past ----------
# (1) Lag spot checks at 20 random (asset, month) pairs, against the raw files
rng_chk = np.random.default_rng(P3_SEED)
G_raw = mac_g.set_index('date')[G_SER].assign(x6=lambda d: np.log(d.x6))
D86 = cm.pivot(index='date', columns='country', values='x86')[COUNTRIES]
D86 = D86.sub(D86['country_7'], axis=0)
for _ in range(20):
    i = rng_chk.integers(len(F))
    m, a = F.m[i], F.asset_id[i]
    j = IDS.index(a)
    assert AUX['lagged']['x1'].iloc[m, j] == X_RAW['x1'].iloc[m - 1, j]          # x1 from month t-1
    assert AUX['lagged']['x2'].iloc[m, j] == X_RAW['x2'].iloc[m, j]              # x2 as stored (pre-lagged)
    assert np.isclose(F.g_x6[i], trailing_z(G_raw)['x6'].iloc[m - 2])            # global macro from month t-2
    if F.cls[i] != 'A' and F.country[i] != 'country_7':
        assert np.isclose(F.c_x86[i], trailing_z(D86)[F.country[i]].iloc[m - 2])  # country macro from month t-2
    assert np.isclose(F.sig[i], R.iloc[m - 36:m, j].std(ddof=1))                 # sigma-hat from r_{t-36..t-1}
print('20 lag spot checks passed')

# (2) Causality test: rebuild from raw data truncated at 2010-12; no feature up to 2010-12 may change
cut = pd.Timestamp('2010-12-31')
F_cut, _ = build_features(panel[panel.date <= cut], mac_g[mac_g.date <= cut], mac_c[mac_c.date <= cut], mac_x[mac_x.date <= cut])
chk = model_cols + ['y', 'sig'] + LEVELS
same_rows = F[F.date <= cut].reset_index(drop=True)
assert len(F_cut) == len(same_rows) and np.allclose(F_cut[chk].to_numpy(), same_rows[chk].to_numpy(), rtol=0, atol=1e-12)
print(f'causality test passed: all {len(chk)} feature columns for the {len(F_cut):,} rows up to {cut.date()} are identical '
      'when every later observation is deleted')'''

CELLS_B = [('md', MD_FEAT), ('code', B1), ('code', B2), ('code', B3)]
