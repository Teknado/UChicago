"""Problem 3, stage 4: the evaluation harness (the only splitter in Problem 3) and its unit tests."""

MD_HARNESS = r"""## 3.4 Models and the evaluation harness

**One splitter, and it only moves forward in time.**
- `window(scheme, b)` gives the training months for a model refit in December of year $b$.
- `inner_folds` cuts that window into three annual train/validation pairs.
- Test months always come after every training month; this is asserted in every fold (L5 p.48–52).
- No shuffled split, `KFold`, `TimeSeriesSplit` or `*CV` estimator appears anywhere in Problem 3 (AI guide §4a; L5 p.58–59).

**Linear models** (OLS, ridge, lasso, PCR) are fit in the same way:
1. Demean $y$ and $X$ within asset class on the *fit rows*, so the class intercepts are unpenalised. This equals ridge with unpenalised class dummies, checked below.
2. Standardise with a `StandardScaler` fitted on the fit rows only (AI guide §4b; L5 p.23, p.47).
3. Compute the whole coefficient path over the tuning grid:
   - ridge: closed form, L5 p.12;
   - lasso: `lasso_path`, as L5 p.47 recommends;
   - PCR: principal components of the fit rows (PCA, L1 p.45–74), then OLS on the first K. The components are taken after standardising, i.e. from the correlation matrix; L1 only demeans, so standardising is our choice (the columns differ in scale). The lectures teach PCA and OLS; combining them into a regression on the components is our construction.

**Tuning.** The winning grid value has the lowest mean MSE over the three validation folds; a tie goes to the heavier penalty (L5 p.25, p.35). The model is then refit on the whole window. The test months play no part in the choice (L5 p.52). L5 p.27 asks for the chosen value to sit inside the grid. Here the heaviest ridge and lasso penalties give the class-means model itself, so a pick at that edge means "no signal", not a grid that is too short. PCR's grid should have included K = 0, the class-means model; that is added as a sensitivity in Section 3.6.

**Random forest.** 300 trees, a third of the inputs tried at each split (L8 p.43), minimum leaf 200 observations, untuned (L8 p.44–45, p.55). **The leaf of 200 departs from L8 p.44, where the forest's trees are grown deep.** It is our choice for a signal this weak. The deep version (leaf 5) is an extra that tests the choice. The forest's inputs are the five ranks and three class dummies (M2), plus the four global and six country macro series without class interactions (M3): a tree can form interactions itself.

**Boosted trees** (L8 p.48–55) were not run. Their rounds, depth and learning rate would have to be tuned on the same forward folds. That is possible without leakage, but it multiplies the runtime, and L8 p.55 recommends the forest when time is short. This was a runtime choice.

**Uncertainty.** The 168 OOS months are resampled with replacement; each month keeps its whole cross-section, because all assets share a month's shocks. The standard error of $R^2_{OOS}$ is the SD across 10,000 resamples, and results are reported as estimate ± 2 SE (L2 p.80, p.85). The lectures bootstrap single observations and refit the model on each resample (L2 p.80). Here the forecasts are held fixed and only the out-of-sample months are resampled, as whole months; that is our construction. Months are treated as independent; serial dependence was not checked. The same resampled months are used for every model, so differences between models are paired."""

D1 = r'''from sklearn.linear_model import lasso_path
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance

CODE = {'A': 0, 'B': 1, 'C': 2, 'D': 3}
F['k'] = F.cls.map(CODE).astype(int)

def window(scheme, b):
    """Training months [a, b] for a model refit at month b."""
    if scheme == 'static':
        return FIRST_M, INIT_END_M
    if scheme == 'expanding':
        return FIRST_M, b
    if scheme == 'rolling':
        return b - ROLL_LEN + 1, b
    raise ValueError(scheme)

def inner_folds(a, b):
    """Three annual validation folds inside [a, b]: fit on a..b-12j, validate on the next 12 months (j = 3, 2, 1)."""
    return [((a, b - 12 * j), (b - 12 * j + 1, b - 12 * j + 12)) for j in (3, 2, 1)]

class LinearFE:
    """Linear model with unpenalised class intercepts. On the fit rows: demean y and X within class, standardise,
    then compute the coefficient path over `grid` (ridge: alpha/n; lasso: sklearn alpha; pcr: number of components)."""
    def __init__(self, kind, grid):
        self.kind, self.grid = kind, list(grid)

    def fit(self, X, y, k):
        self.xm = np.vstack([X[k == c].mean(0) if (k == c).any() else np.full(X.shape[1], np.nan) for c in range(4)])
        self.ym = np.array([y[k == c].mean() if (k == c).any() else np.nan for c in range(4)])
        yd = y - self.ym[k]
        if X.shape[1] == 0:                                     # M1: class means only
            self.coefs = np.zeros((0, len(self.grid)))
            return self
        Xd = X - self.xm[k]
        self.sc = StandardScaler().fit(Xd)                       # fitted on the fit rows only
        assert self.sc.n_samples_seen_ == len(Xd)
        Xs, n = self.sc.transform(Xd), len(yd)
        if self.kind == 'ols':
            self.coefs = np.linalg.lstsq(Xs, yd, rcond=None)[0][:, None]
        elif self.kind == 'ridge':                               # b = (X'X + alpha I)^-1 X'y via the SVD (L5 p.12)
            U, s, Vt = np.linalg.svd(Xs, full_matrices=False)
            g = np.asarray(self.grid) * n
            self.coefs = Vt.T @ ((s * (U.T @ yd))[:, None] / (s[:, None] ** 2 + g[None, :]))
        elif self.kind == 'lasso':
            _, self.coefs, _, n_iter = lasso_path(Xs, yd, alphas=self.grid, max_iter=10_000, return_n_iter=True)
            assert max(n_iter) < 10_000, 'lasso did not converge'
        elif self.kind == 'pcr':                                  # PCA of the fit rows (L1), OLS on the first K scores
            U, s, Vt = np.linalg.svd(Xs, full_matrices=False)
            keep = s > 1e-10 * s[0]
            gam = np.where(keep, (U.T @ yd) / np.where(keep, s, 1), 0.0)
            self.coefs = np.column_stack([Vt[:K].T @ gam[:K] for K in self.grid])
        else:
            raise ValueError(self.kind)
        return self

    def predict(self, X, k):
        """Forecasts for every grid value: an (n_rows, n_grid) array."""
        base = self.ym[k][:, None]
        if X.shape[1] == 0:
            return np.repeat(base, len(self.grid), axis=1)
        return base + self.sc.transform(X - self.xm[k]) @ self.coefs
'''

D1R = r'''# ---------- The runners: one forward-only loop over refits, for the linear models and the forest ----------
def run_linear(kind, cols, scheme, frame=None, rows=None, fixed=None, grid=None):
    """Out-of-sample forecasts for one linear specification. Returns yhat on the frame's OOS rows and a refit log."""
    fr = F if frame is None else frame
    if rows is not None:
        fr = fr[rows]
    X, y, k, m = fr[cols].to_numpy(), fr.y.to_numpy(), fr.k.to_numpy(), fr.m.to_numpy()
    grid = [fixed] if fixed is not None else ([None] if kind == 'ols' else list(GRIDS[kind] if grid is None else grid))
    ends = [INIT_END_M] if scheme == 'static' else REFIT_ENDS
    pred, log = np.full(len(fr), np.nan), []
    for b in ends:
        a, b = window(scheme, b)
        j = 0
        if len(grid) > 1:
            mse = np.zeros(len(grid))
            for (f0, f1), (v0, v1) in inner_folds(a, b):
                fit, val = (m >= f0) & (m <= f1), (m >= v0) & (m <= v1)
                assert m[fit].max() < m[val].min() and m[fit].min() >= a and m[val].max() <= b
                mdl = LinearFE(kind, grid).fit(X[fit], y[fit], k[fit])
                mse += ((y[val][:, None] - mdl.predict(X[val], k[val])) ** 2).mean(0) / 3
            j = int(np.argmin(mse))                              # most-penalised first, so a tie goes to the heavier penalty
        win = (m >= a) & (m <= b)
        test = (m > b) & (m <= (LAST_M if scheme == 'static' else b + 12))
        assert m[win].max() < m[test].min()
        mdl = LinearFE(kind, [grid[j]]).fit(X[win], y[win], k[win])
        pred[test] = mdl.predict(X[test], k[test])[:, 0]
        fit_in = mdl.predict(X[win], k[win])[:, 0]
        log.append({'refit': DATES[b].date(), 'train months': b - a + 1, 'choice': grid[j], 'grid index': j,
                    'at grid edge': len(grid) > 1 and j in (0, len(grid) - 1),
                    'IS R2': 1 - ((y[win] - fit_in) ** 2).sum() / ((y[win] - y[win].mean()) ** 2).sum(),
                    'coef': dict(zip(cols, mdl.coefs[:, 0])) if len(cols) else {}})
    oos = m >= OOS_M[0]
    assert np.isfinite(pred[oos]).all()
    return pred[oos], pd.DataFrame(log)


def run_rf(cols, scheme, params, importance=False):
    """Random-forest forecasts on the same windows (untuned). Optionally OOS permutation importance per test year."""
    X, y, m = F[cols].to_numpy(), F.y.to_numpy(), F.m.to_numpy()
    ends = [INIT_END_M] if scheme == 'static' else REFIT_ENDS
    pred, log, imp = np.full(len(F), np.nan), [], []
    for b in ends:
        a, b = window(scheme, b)
        win = (m >= a) & (m <= b)
        test = (m > b) & (m <= (LAST_M if scheme == 'static' else b + 12))
        assert m[win].max() < m[test].min()
        rf = RandomForestRegressor(**params).fit(X[win], y[win])
        pred[test] = rf.predict(X[test])
        log.append({'refit': DATES[b].date(), 'train months': b - a + 1,
                    'IS R2': 1 - ((y[win] - rf.predict(X[win])) ** 2).sum() / ((y[win] - y[win].mean()) ** 2).sum(),
                    'mean leaves per tree': np.mean([t.get_n_leaves() for t in rf.estimators_])})
        if importance:                                           # L8 p.57: on held-out rows, repeated shuffles
            pi = permutation_importance(rf, X[test], y[test], scoring='neg_mean_squared_error', n_repeats=10,
                                        random_state=P3_SEED, n_jobs=1)
            imp.append(pd.Series(pi.importances_mean, index=cols, name=DATES[b + 1].year))
    oos = m >= OOS_M[0]
    return pred[oos], pd.DataFrame(log), (pd.DataFrame(imp) if importance else None)
'''

D1B = r'''# ---------- The month bootstrap: the same resampled months for every model (paired comparisons) ----------
BOOT_B = 10_000
BOOT_IDX = np.random.default_rng(P3_SEED).integers(0, T_OOS, size=(BOOT_B, T_OOS))
OOS_MONTH_POS = OOS.m.to_numpy() - OOS_M[0]                     # 0..167 for every OOS row

def month_sums(v):
    return np.bincount(OOS_MONTH_POS, weights=v, minlength=T_OOS)

def r2_boot(yhat, bench, rows=None):
    """R2_OOS against `bench` and its month-bootstrap SE (optionally on a subset of OOS rows)."""
    w = np.ones(len(OOS)) if rows is None else np.asarray(rows, float)
    e_m, e_b = month_sums(w * (OOS.y - yhat) ** 2), month_sums(w * (OOS.y - bench) ** 2)
    r2 = 1 - e_m.sum() / e_b.sum()
    r2_b = 1 - e_m[BOOT_IDX].sum(1) / e_b[BOOT_IDX].sum(1)
    return r2, r2_b.std(ddof=1)

def delta_boot(yhat1, yhat2, bench, rows=None):
    """R2(model 1) - R2(model 2) against the same benchmark, with its paired month-bootstrap SE."""
    w = np.ones(len(OOS)) if rows is None else np.asarray(rows, float)
    e1, e2, eb = (month_sums(w * (OOS.y - v) ** 2) for v in (yhat1, yhat2, bench))
    d = (e2.sum() - e1.sum()) / eb.sum()
    d_b = (e2[BOOT_IDX].sum(1) - e1[BOOT_IDX].sum(1)) / eb[BOOT_IDX].sum(1)
    return d, d_b.std(ddof=1)'''

D2 = r'''# ---------- Unit tests of the harness, on real training rows (no OOS data involved) ----------
tr = F.m <= INIT_END_M
Xt, yt, kt = F.loc[tr, FSETS['M3']].to_numpy(), F.loc[tr, 'y'].to_numpy(), F.loc[tr, 'k'].to_numpy()
# (1) OLS with demeaning == OLS with class dummies (Frisch-Waugh): identical fitted values
fe = LinearFE('ols', [None]).fit(Xt, yt, kt)
D = np.column_stack([np.eye(4)[kt], Xt])
assert np.allclose(fe.predict(Xt, kt)[:, 0], D @ np.linalg.lstsq(D, yt, rcond=None)[0])
# (2) the ridge path == sklearn Ridge(alpha = a * n) on the same demeaned, standardised data
rp = LinearFE('ridge', GRIDS['ridge']).fit(Xt, yt, kt)
Xs = rp.sc.transform(Xt - rp.xm[kt])
for j in (0, 17, 35):
    sk = Ridge(alpha=GRIDS['ridge'][j] * len(yt), fit_intercept=False).fit(Xs, yt - rp.ym[kt])
    assert np.allclose(sk.coef_, rp.coefs[:, j], atol=1e-8)
# (3) the PCR path == sklearn PCA(K) followed by OLS on the scores
pc = LinearFE('pcr', GRIDS['pcr']).fit(Xt, yt, kt)
for j, K in enumerate(GRIDS['pcr']):
    Z = PCA(n_components=K).fit(Xs)
    lr = LinearRegression(fit_intercept=False).fit(Z.transform(Xs), yt - pc.ym[kt])
    assert np.allclose(lr.predict(Z.transform(Xs)) + pc.ym[kt], pc.predict(Xt, kt)[:, j], atol=1e-8)
# (4) the lasso path converges and its most-penalised end is the class-means model
la = LinearFE('lasso', GRIDS['lasso']).fit(Xt, yt, kt)
assert np.allclose(la.coefs[:, 0], 0), 'the top of the lasso grid should select nothing'
# (5) the scorer: a forecast equal to the benchmark scores 0, a perfect forecast scores 1
assert np.isclose(r2_oos(OOS.y, OOS.b_pool, OOS.b_pool), 0) and np.isclose(r2_oos(OOS.y, OOS.y, OOS.b_pool), 1)
# (6) the splitter: every validation fold lies inside the training window and after its own fit rows
#     (that the test months follow the window is asserted at every refit inside run_linear and run_rf)
for sch in SCHEMES:
    for b in ([INIT_END_M] if sch == 'static' else REFIT_ENDS):
        a, bb = window(sch, b)
        f = inner_folds(a, bb)
        assert all(f0 >= a and f1 < v0 and v1 <= bb for (f0, f1), (v0, v1) in f)
print('harness unit tests passed: demeaning = class dummies; ridge and PCR paths match sklearn; lasso converges; '
      'scorer and splitter behave')'''

CELLS_D = [('md', MD_HARNESS), ('code', D1), ('code', D1R), ('code', D1B), ('code', D2)]
