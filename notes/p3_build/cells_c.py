"""Problem 3, stage 3: benchmarks before models."""

MD_BENCH = r"""## 3.3 Benchmarks first

Rule 1 of the exam's "Two rules that apply throughout" says to compute the simple rules before fitting anything. This section does that: it contains no model code, and its numbers are printed before any model exists in the notebook.

**Forecast benchmarks** for the target $y$ in month $t$. Every one uses data through $t-1$ only and is updated monthly (the expanding harness of HW5 Problem 1.1):
- the **pooled trailing mean** over all assets, the primary benchmark;
- the per-class trailing mean and the per-asset trailing mean;
- zero.

For raw returns, the literal benchmark of that rule is the per-asset trailing mean of the excess return.

$$R^2_{OOS}=1-\frac{\sum (y-\hat y)^2}{\sum (y-\bar y^{\,bench})^2}\qquad\text{(L5 p.55–57; never `r2_score`, which uses the test mean: AI guide §4d)}$$

**Portfolio benchmarks**, all at unit gross exposure and rebalanced monthly [EXAM-DEFINED, rule 1]:
- EW: $w_i = 1/50$;
- RP: $w_i\propto 1/\hat\sigma_{i,t-1}$;
- TSMOM: $w_i\propto \mathrm{sign}(R_{12,i,t-1})/\hat\sigma_{i,t-1}$, where $R_{12}$ is the compounded return over $t-12..t-1$.

$\hat\sigma$ is the same 36-month volatility used for the target.

**Performance measures.** The lectures define none of these, so the formulas are stated here:
- Sharpe ratio: $\sqrt{12}\,\text{mean}/\text{sd}$ of monthly excess returns;
- maximum drawdown of cumulative wealth $\prod(1+r)$;
- turnover: $\sum_i |w_{i,t}-w^{\text{drift}}_{i,t}|$, the traded notional (buys plus sells) against the weights after last month's drift. The cost is charged per unit of this sum. Where "one-way turnover" is defined as half of it, our 10 bp equals 20 bp per unit of one-way turnover, so the costs here are conservative;
- net return = gross − c × turnover.

The first OOS month's initial build is excluded from turnover for every strategy."""

C1 = r'''# ---------- Forecast benchmarks (data through t-1 only) ----------
mon = F.groupby('m').y.agg(['sum', 'count'])
b_pool = (mon['sum'].cumsum() / mon['count'].cumsum()).shift(1)                       # pooled trailing mean, one value per month
cls_mon = F.groupby(['cls', 'm']).y.agg(['sum', 'count'])
b_cls = (cls_mon.groupby(level='cls').cumsum().pipe(lambda d: d['sum'] / d['count'])
         .groupby(level='cls').shift(1))                                            # per-class trailing mean
F['b_pool'] = F.m.map(b_pool)
F = F.merge(b_cls.rename('b_class').reset_index(), on=['cls', 'm'], how='left', validate='many_to_one')
F['b_asset'] = F.groupby('asset_id').y.transform(lambda v: v.expanding().mean().shift(1))
R_mean_raw = R.expanding().mean().shift(1)                                          # raw returns: per-asset trailing mean, from 2000-01
F['rb_asset'] = R_mean_raw.to_numpy()[F.m.to_numpy(), [IDS.index(a) for a in F.asset_id]]
# unit test of the primary benchmark against a brute-force mean
assert np.isclose(b_pool[OOS_M[0]], F.loc[(F.m >= FIRST_M) & (F.m < OOS_M[0]), 'y'].mean())
assert np.isclose(F.loc[(F.m == 150) & (F.asset_id == 'asset_9'), 'b_asset'].iloc[0],
                  F.loc[(F.m < 150) & (F.asset_id == 'asset_9'), 'y'].mean())

OOS = F[F.m >= OOS_M[0]].reset_index(drop=True)                                     # the 8,400 out-of-sample rows
assert len(OOS) == 8400 and OOS[['b_pool', 'b_class', 'b_asset', 'rb_asset']].notna().all().all()

def r2_oos(y, yhat, bench):
    """Out-of-sample R2 against a benchmark fixed before each forecast month (L5 p.55-57)."""
    y, yhat, bench = map(np.asarray, (y, yhat, bench))
    assert y.shape == yhat.shape == bench.shape and np.isfinite(yhat).all() and np.isfinite(bench).all()
    return 1 - ((y - yhat) ** 2).sum() / ((y - bench) ** 2).sum()

# How hard is each benchmark to beat? Each forecast scored against the others, before any model exists
cands = {'pooled trailing mean': OOS.b_pool, 'per-class trailing mean': OOS.b_class,
         'per-asset trailing mean': OOS.b_asset, 'zero': np.zeros(len(OOS))}
bench_tab = pd.DataFrame({f'vs {bn}': {fn: r2_oos(OOS.y, f, b) for fn, f in cands.items()}
                          for bn, b in [('pooled trailing mean', OOS.b_pool), ('zero', np.zeros(len(OOS)))]})
for c in 'ABCD':
    k = OOS.cls == c
    bench_tab[f'class {c}: vs pooled'] = [r2_oos(OOS.y[k], np.asarray(f)[k], OOS.b_pool[k]) for f in cands.values()]
display(bench_tab.style.format('{:.2%}').set_caption('Table 3.8: the benchmarks scored against each other, y, 2011-2024 (R2_OOS)'))
print(f'mean of y: training block {F.loc[F.m <= INIT_END_M, "y"].mean():.4f}, OOS {OOS.y.mean():.4f}; sd of y (OOS) {OOS.y.std(ddof=1):.3f}')'''

C2 = r'''# ---------- Portfolio machinery and the three benchmark portfolios (built from excess_return alone) ----------
SIG_W = SD36                                            # sigma-hat_{t-1}, the same as in the target (date x asset)
R12_W = R12                                             # compounded return over t-12..t-1

def unit_gross(W):
    """Scale each month's weights so that the sum of |w| is 1 (a month with no signal holds cash)."""
    g = W.abs().sum(axis=1)
    return W.div(g.where(g > 1e-12), axis=0).fillna(0.0)

def port_ret(W):
    return (W * R.loc[W.index, W.columns]).sum(axis=1)

def turnover(W):
    """Traded notional (buys plus sells): this month's weights against last month's weights after they drifted."""
    Rs = R.loc[W.index, W.columns]
    Rp = (W * Rs).sum(axis=1)
    drift = (W.shift(1) * (1 + Rs.shift(1))).div(1 + Rp.shift(1), axis=0)
    return (W - drift.fillna(0.0)).abs().sum(axis=1)

def perf(ret, to, months, cost_bp=0.0):
    """Performance over a set of months; the first month's build is excluded from turnover and costs."""
    idx = DATES[months]
    tt = to.loc[idx].copy()
    tt.iloc[0] = 0.0
    r = ret.loc[idx] - cost_bp / 1e4 * tt
    wealth = (1 + r).cumprod()
    return {'ann. mean': 12 * r.mean(), 'ann. vol': np.sqrt(12) * r.std(ddof=1), 'Sharpe': np.sqrt(12) * r.mean() / r.std(ddof=1),
            'max drawdown': (1 - wealth / wealth.cummax()).max(), 'worst month': r.min(), 'turnover / month': tt.iloc[1:].mean()}

def weights_from_forecast(YH, rule='P1'):
    """Positions from forecasts of y (date x asset): P1 ~ yhat / sigma; P3 ~ (yhat - class mean) / sigma."""
    if rule == 'P1':
        return unit_gross(YH / SIG_W.loc[YH.index, YH.columns])
    if rule == 'P3':
        dev = YH - YH.T.groupby(CLS_S[YH.columns]).transform('mean').T
        return unit_gross(dev / SIG_W.loc[YH.index, YH.columns])
    raise ValueError(rule)

bench_idx = DATES[FIRST_M:]
W_EW = pd.DataFrame(1.0 / len(IDS), index=bench_idx, columns=IDS)
W_RP = unit_gross(1.0 / SIG_W.loc[bench_idx])
W_TS = unit_gross(np.sign(R12_W.loc[bench_idx]) / SIG_W.loc[bench_idx])
BENCH_W = {'EW': W_EW, 'RP': W_RP, 'TSMOM': W_TS}
for W in BENCH_W.values():
    assert np.allclose(W.abs().sum(axis=1), 1)
# identity: P1 with a constant forecast is exactly risk parity
assert np.allclose(weights_from_forecast(pd.DataFrame(1.0, index=bench_idx, columns=IDS)), W_RP, atol=1e-12)
# causality: weights up to 2010-12 are unchanged when every later return is deleted
Rc = R.loc[:'2010-12-31']
W_RP_cut = unit_gross(1.0 / Rc.rolling(36, min_periods=36).std(ddof=1).shift(1).loc[DATES[FIRST_M]:])
W_TS_cut = unit_gross(np.sign(np.expm1(np.log1p(Rc).rolling(12, min_periods=12).sum()).shift(1)).loc[DATES[FIRST_M]:]
                      / Rc.rolling(36, min_periods=36).std(ddof=1).shift(1).loc[DATES[FIRST_M]:])
assert np.allclose(W_RP_cut, W_RP.loc[:'2010-12-31']) and np.allclose(W_TS_cut, W_TS.loc[:'2010-12-31'])
BENCH_RET = {k: port_ret(W) for k, W in BENCH_W.items()}
BENCH_TO = {k: turnover(W) for k, W in BENCH_W.items()}

rows = {}
for k in BENCH_W:
    rows[(k, 'OOS 2011-2024, gross')] = perf(BENCH_RET[k], BENCH_TO[k], OOS_M)
    rows[(k, f'OOS 2011-2024, net {HEADLINE_BP} bp')] = perf(BENCH_RET[k], BENCH_TO[k], OOS_M, HEADLINE_BP)
    rows[(k, 'context 2003-2010, gross')] = perf(BENCH_RET[k], BENCH_TO[k], np.arange(FIRST_M, INIT_END_M + 1))
bench_perf = pd.DataFrame(rows).T
display(bench_perf.style.format({c: '{:.3f}' for c in bench_perf.columns}).set_caption('Table 3.9: the three benchmark portfolios'))
share = pd.DataFrame({k: W.loc[DATES[OOS_M]].abs().T.groupby(CLS_S).sum().T.mean() for k, W in BENCH_W.items()}).T
share['largest single weight'] = [W.loc[DATES[OOS_M]].abs().max().max() for W in BENCH_W.values()]
share['asset_16 average weight'] = [W.loc[DATES[OOS_M], 'asset_16'].abs().mean() for W in BENCH_W.values()]
display(share.style.format('{:.3f}').set_caption('Table 3.9b: where the benchmark portfolios put their gross exposure, 2011-2024'))'''

CELLS_C = [('md', MD_BENCH), ('code', C1), ('code', C2)]
