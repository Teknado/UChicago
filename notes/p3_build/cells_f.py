"""Problem 3, stage 6: does macro add? placebo, robustness, x10 variants, evaluation-only cuts."""

MD_MACRO = r"""## 3.6 Does macro add? Placebo and robustness

**Macro vs no macro.** Each model is refit without macro, on the same rows, windows, folds and benchmark, so every comparison is paired. This is the "refit without it" test of the exam's suggested workflow.

**Placebo** [EXAM-DEFINED, suggested workflow].
- Every raw macro series is circularly shifted by $s$ months, then the identical pipeline runs: logs, differentials, trailing z-scores, lags, tuning.
- The data are real, but at the wrong time.
- A persistent series stays partly correlated with its shifted copy, so eight shifts are used, $s = 36, 48, \dots, 120$.
- The shift wraps around: the last $s$ months of each series move to the start of the sample. A target month uses raw macro months $t-61..t-2$ (a 60-month trailing z-score, then the 2-month lag), so every target month up to $s+60$ carries some wrapped values from the end of the sample. That covers 61–96 training months at the first fit and, for $s \ge 72$, the start of the out-of-sample period as well (one month at $s = 72$, 49 months at $s = 120$). The counts are printed below and checked by rebuilding the features.
- These wrapped values are later macro placed at the wrong date. They are no better aligned with the returns next to them than any other placebo value, which is the point of a placebo. But the placebo inputs are not strictly past data.

**Macro is credited only if** its gain over the no-macro model is more than 2 bootstrap SEs above zero **and** larger than every placebo gain (our construction, in the spirit of the simulated null distributions of L4 p.26–28)."""

F1 = r'''# ---------- Macro vs no macro: same model, same rows, same benchmark ----------
BP = OOS.b_pool.to_numpy()
pairs = ([(f'{mdl}|M3|{sch}', f'{mdl}|M2|{sch}') for mdl in ['ols', 'ridge', 'rf'] for sch in SCHEMES]
         + [('ridge|M3-global|expanding', 'ridge|M2|expanding'), ('ridge|M3-country|expanding', 'ridge|M2|expanding'),
            ('ridge-per-class|M3|expanding', 'ridge-per-class|M2|expanding'), ('ridge|M3-levels|expanding', 'ridge|M2-levels|expanding'),
            ('rf-deep|M3|expanding', 'rf-deep|M2|expanding')])
mac = pd.DataFrame([dict(zip(['with macro', 'without macro'], p)) | dict(zip(['gain in R2_OOS from macro', 'SE'], delta_boot(RES[p[0]], RES[p[1]], BP)))
                    for p in pairs])
mac['gain / SE'] = mac['gain in R2_OOS from macro'] / mac['SE']
display(mac.style.format({'gain in R2_OOS from macro': '{:.3%}', 'SE': '{:.3%}', 'gain / SE': '{:.2f}'})
        .set_caption('Table 3.18: what macro adds to each model (R2_OOS vs the pooled trailing mean, 2011-2024)'))
q2_gain, q2_se = delta_boot(RES['ridge|M3|expanding'], RES['ridge|M2|expanding'], BP)
print(f'Q2 primary: ridge M3 - ridge M2 (expanding) = {q2_gain:.3%} (SE {q2_se:.3%})')

# global macro is one number per month (a class-level signal); country macro also varies across countries within a month
xs_share = {}
for c in 'BCD':
    sub_c = F[F.cls == c]
    xs_share[f'class {c}'] = {k: (sub_c['c_' + k] - sub_c.groupby('m')['c_' + k].transform('mean')).var() / sub_c['c_' + k].var()
                              for k, _, _ in C_SER}
xs_share = pd.DataFrame(xs_share)
display(xs_share.style.format('{:.0%}').set_caption('Table 3.18b: share of the variance of each country-macro input that lies across countries within a month'))'''

F2 = r'''# ---------- Placebo: every macro series circularly shifted by s months (real data at the wrong time) ----------
t0 = time.perf_counter()
placebo = {}
for s in PLACEBO_SHIFTS:
    yh_s, _ = run_linear('ridge', FSETS['M3'], 'expanding', frame=variant(shift=s))
    placebo[s] = delta_boot(yh_s, RES['ridge|M2|expanding'], BP)
plac = pd.DataFrame(placebo, index=['gain vs ridge M2', 'SE']).T
plac.index.name = 'shift (months)'
display(plac.style.format('{:.3%}').set_caption('Table 3.19: placebo gains of ridge M3 (expanding) over ridge M2'))
print(f'{len(PLACEBO_SHIFTS)} placebo runs in {time.perf_counter() - t0:.0f} s; real macro gain {q2_gain:.3%} ranks '
      f'{1 + (plac["gain vs ridge M2"] > q2_gain).sum()} of {len(PLACEBO_SHIFTS) + 1} (1 = best)')
q2_verdict = (q2_gain > 2 * q2_se) and (q2_gain > plac['gain vs ridge M2'].max())
print(f'pre-registered rule (gain > 2 SE and above every placebo): {"YES" if q2_verdict else "NO"}')

# where the circular shift wraps: a target month m uses raw macro months m-61..m-2, so m <= s + 58 + MACRO_LAG is touched
last_touched = {s: s + 58 + MACRO_LAG for s in PLACEBO_SHIFTS}
wrap = pd.DataFrame({s: {'first-fit training months (2003-2010) with wrapped macro': min(t, INIT_END_M) - FIRST_M + 1,
                         'OOS target months with wrapped macro': max(0, t - INIT_END_M),
                         'last affected target month': DATES[t].strftime('%Y-%m')} for s, t in last_touched.items()}).T
wrap.index.name = 'shift (months)'
display(wrap.style.set_caption('Table 3.19b: where the circular shift puts end-of-sample macro values'))
def touched_months(s):
    """Rebuild the placebo features after perturbing the last s raw macro months: rows that change use wrapped values."""
    cut = DATES[len(DATES) - s]
    bump = lambda d, cols: d.assign(**{c: np.where(d.date >= cut, d[c] * 1.1 + 0.05, d[c]) for c in cols})
    Fa, _ = build_features(panel, mac_g, mac_c, mac_x, shift=s)
    Fb, _ = build_features(panel, bump(mac_g, G_SER), bump(mac_c, [k for k, f, _ in C_SER if f == 'curated']),
                           bump(mac_x, [k for k, f, _ in C_SER if f == 'extended']), shift=s)
    mcols = GLOB + CTRY_INT
    changed = (Fa[mcols] - Fb[mcols]).abs().max(axis=1) > 1e-8      # pandas' rolling sums leave round-off of ~1e-11 behind
    return Fa.m[changed].max()
for s in (PLACEBO_SHIFTS[0], PLACEBO_SHIFTS[-1]):
    assert touched_months(s) == last_touched[s], s
print(f'checked by rebuilding: the last target month affected is m = s + {58 + MACRO_LAG} for s = {PLACEBO_SHIFTS[0]} and s = {PLACEBO_SHIFTS[-1]}')

# how correlated each lag-ready macro input stays with its own shifted copy (why several shifts are needed)
pers = {}
for k in G_SER:
    z = AUX['GZ'][k]
    pers[f'global {k}'] = {s: z.corr(z.shift(s)) for s in PLACEBO_SHIFTS}
for k, _, _ in C_SER:
    Z = AUX['CZ'][k].drop(columns='country_7')
    pers[f'country {k}'] = {s: np.nanmean([Z[c].corr(Z[c].shift(s)) for c in Z]) for s in PLACEBO_SHIFTS}
display(pd.DataFrame(pers).T.style.format('{:.2f}').set_caption('Table 3.20: corr(input, input shifted s months), the persistence a placebo inherits'))

fig, ax = plt.subplots(figsize=(8, 3.4))
ax.scatter(plac.index, plac['gain vs ridge M2'], color=C_GREY, s=40, label='placebo: macro shifted s months', zorder=3)
ax.axhline(q2_gain, color=C_ORANGE, linewidth=2, label=f'real macro ({q2_gain:.2%})')
ax.axhspan(q2_gain - 2 * q2_se, q2_gain + 2 * q2_se, color=C_ORANGE, alpha=0.12, label='real macro +/- 2 SE')
ax.axhline(0, color=C_INK, linewidth=1)
ax.set_xlabel('shift s (months)')
ax.set_ylabel('gain in R2_OOS over ridge M2')
ax.set_title('Figure 3.7: real macro against placebo macro (ridge, expanding)')
ax.legend(fontsize=8, loc='lower left')
plt.show()'''

F3 = r'''# ---------- Robustness and the pre-listed extras, each against its parent ----------
rob_pairs = [('pcr-K3|M3|expanding', 'pcr|M3|expanding', 'PCR with K fixed at 3 vs K tuned'),
             ('pcr-K5|M3|expanding', 'pcr|M3|expanding', 'PCR with K fixed at 5 vs K tuned'),
             ('rf-deep|M2|expanding', 'rf|M2|expanding', 'deep forest (leaf 5) vs leaf 200, M2'),
             ('rf-deep|M3|expanding', 'rf|M3|expanding', 'deep forest (leaf 5) vs leaf 200, M3'),
             ('ridge|M3-lag1|expanding', 'ridge|M3|expanding', 'macro lag 1 month vs 2'),
             ('ridge|M2-zscore|expanding', 'ridge|M2|expanding', 'within-class z-score vs rank'),
             ('ridge|M2-levels|expanding', 'ridge|M2|expanding', 'M2 + own-history levels vs M2'),
             ('ridge|M3-levels|expanding', 'ridge|M3|expanding', 'M3 + own-history levels vs M3'),
             ('ridge|M3-x10|expanding', 'ridge|M3|expanding', 'curated x10 (14-month lag) vs x86')]
rob = pd.DataFrame([{'comparison': lab, 'R2_OOS (variant)': res_tab.loc[a, 'R2 vs pooled mean'],
                     'R2_OOS (parent)': res_tab.loc[b, 'R2 vs pooled mean'],
                     **dict(zip(['difference', 'SE'], delta_boot(RES[a], RES[b], BP))),
                     'mean IS R2 (variant)': res_tab.loc[a, 'mean IS R2']} for a, b, lab in rob_pairs]).set_index('comparison')
display(rob.style.format('{:.3%}').set_caption('Table 3.21: robustness checks (R2_OOS vs the pooled trailing mean)'))

# PCR's pre-registered grid starts at K = 1, while ridge and lasso can reach the class-means model. Added after the ledger
# (not a candidate): allow K = 0, i.e. class means only.
PCR0_YH, PCR0_LOG = run_linear('pcr', FSETS['M3'], 'expanding', grid=[0] + list(GRIDS['pcr']))
pcr0_d, pcr0_se = delta_boot(PCR0_YH, RES['pcr|M3|expanding'], BP)
print(f"PCR M3 (expanding) with K = 0 allowed: R2_OOS {r2_oos(OOS.y, PCR0_YH, BP):.3%} vs {res_tab.loc['pcr|M3|expanding', 'R2 vs pooled mean']:.3%} "
      f"on the pre-registered grid (difference {pcr0_d:.3%}, SE {pcr0_se:.3%}); K = 0 chosen in {(PCR0_LOG['choice'] == 0).sum()} of {len(PCR0_LOG)} refits; "
      f"choices {list(PCR0_LOG['choice'])}")

# the x10 trap, three ways: the headline (x86, lag 2), the curated file at a correct lag, and the naive 1-month lag
x10_tab = pd.DataFrame({
    'x86 at a 2-month lag (headline)': {'R2 vs pooled mean': res_tab.loc['ridge|M3|expanding', 'R2 vs pooled mean'],
                                         'R2 vs zero': res_tab.loc['ridge|M3|expanding', 'R2 vs zero'],
                                         'mean IS R2': res_tab.loc['ridge|M3|expanding', 'mean IS R2']},
    'x10 at a 14-month lag (curated file, no look-ahead)': {'R2 vs pooled mean': res_tab.loc['ridge|M3-x10|expanding', 'R2 vs pooled mean'],
                                                            'R2 vs zero': res_tab.loc['ridge|M3-x10|expanding', 'R2 vs zero'],
                                                            'mean IS R2': res_tab.loc['ridge|M3-x10|expanding', 'mean IS R2']},
    'x10 at a 1-month lag (LEAK DEMONSTRATION, not a result)': {'R2 vs pooled mean': r2_oos(OOS.y, LEAK_YH, BP),
                                                                'R2 vs zero': r2_oos(OOS.y, LEAK_YH, ZERO),
                                                                'mean IS R2': LEAK_LOG['IS R2'].mean()}}).T
display(x10_tab.style.format('{:.3%}').set_caption('Table 3.22: the x10 trap (ridge M3, expanding)'))
top = GRIDS['ridge'][0]                                   # the no-signal end of the ridge grid (class means only)
print('refits (of 14) in which ridge left the no-signal end of its grid: '
      f'x86 {(LOGS["ridge|M3|expanding"]["choice"] < top).sum()}, x10 at 14 months {(LOGS["ridge|M3-x10|expanding"]["choice"] < top).sum()}, '
      f'x10 leaked {(LEAK_LOG["choice"] < top).sum()}')

# evaluation-only cuts: the same forecasts scored without COVID's first months or without the stale-x1 rows
covid = ~OOS.date.between('2020-03-31', '2020-05-31').to_numpy()
stale = ~pd.Series(list(zip(OOS.asset_id, OOS.date))).isin(FROZEN_ROWS).to_numpy()
assert (~stale).sum() == len(FROZEN_ROWS)
cuts = pd.DataFrame({n: {'all 168 months': res_tab.loc[n, 'R2 vs pooled mean'],
                         'without 2020-03..05': r2_boot(RES[n], BP, covid)[0],
                         'without the 10 stale-x1 rows': r2_boot(RES[n], BP, stale)[0]} for n in KEY}).T
display(cuts.style.format('{:.3%}').set_caption('Table 3.23: evaluation-only cuts (no refitting)'))'''

from cells_x import F2B
CELLS_F = [('md', MD_MACRO), ('code', F1), ('code', F2), ('code', F2B), ('code', F3)]
