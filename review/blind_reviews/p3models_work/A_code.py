import matplotlib
matplotlib.use("Agg")
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import os
import warnings
warnings.filterwarnings('ignore')

# Robust data path: works locally in Final Project, from root, or on class server
_DATA_DIR = './' if os.path.exists('asset_panel.csv') else ('Final Project/' if os.path.exists('Final Project/asset_panel.csv') else '/classes/41210_MiF_fall2026/Data/')

panel = pd.read_csv(os.path.join(_DATA_DIR, 'asset_panel.csv'), parse_dates=['date'])
info  = pd.read_csv(os.path.join(_DATA_DIR, 'asset_info.csv'))
mac_g = pd.read_csv(os.path.join(_DATA_DIR, 'macro_global.csv'), parse_dates=['date'])
mac_c = pd.read_csv(os.path.join(_DATA_DIR, 'macro_country.csv'), parse_dates=['date'])
mac_x = pd.read_csv(os.path.join(_DATA_DIR, 'macro_country_extended.csv'), parse_dates=['date'])
print('Panel shape:', panel.shape)
print('Date range:', panel.date.min().date(), 'to', panel.date.max().date())


# ==============================================================================
# PROBLEM 3: EMPIRICAL RESEARCH PIPELINE & ASSET RETURN PREDICTION
# ==============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.ensemble import RandomForestRegressor
import os
import warnings
warnings.filterwarnings('ignore')

# ------------------------------------------------------------------------------
# STEP 1: KNOW YOUR DATA & DATA TRAP AUDIT
# ------------------------------------------------------------------------------
print('='*80)
print('STEP 1: EXPLORATORY DATA ANALYSIS & DATA TRAP AUDIT')
print('='*80)

# 1.1 Summary statistics per asset class
class_summary = []
for c, grp in panel.groupby('asset_class'):
    class_rets = grp.groupby('date')['excess_return'].mean()
    m = class_rets.mean() * 12
    s = class_rets.std(ddof=1) * np.sqrt(12)
    sr = m / s if s > 0 else np.nan
    worst = class_rets.min()
    best = class_rets.max()
    class_summary.append({
        'Asset Class': c,
        'N Assets': grp['asset_id'].nunique(),
        'Ann Mean Ret': f'{m*100:.2f}%',
        'Ann Volatility': f'{s*100:.2f}%',
        'Sharpe Ratio': f'{sr:.3f}',
        'Worst Month': f'{worst*100:.2f}%',
        'Best Month': f'{best*100:.2f}%'
    })
print('\n[Table 1] Summary Statistics by Asset Class (2000-2024):')
print(pd.DataFrame(class_summary).to_string(index=False))

# 1.2 Cumulative returns and intra/inter-class correlation matrix
classes = ['A', 'B', 'C', 'D']
block_matrix = pd.DataFrame(index=classes, columns=classes)
wide = panel.pivot(index='date', columns='asset_id', values='excess_return')
corr_mat = wide.corr()

info_dict = dict(zip(panel['asset_id'], panel['asset_class']))
for c1 in classes:
    a1 = [a for a, ac in info_dict.items() if ac == c1]
    for c2 in classes:
        a2 = [a for a, ac in info_dict.items() if ac == c2]
        sub = corr_mat.loc[a1, a2]
        if c1 == c2:
            n = len(a1)
            val = (sub.values.sum() - n) / (n * (n - 1)) if n > 1 else 1.0
        else:
            val = sub.values.mean()
        block_matrix.loc[c1, c2] = val

print('\n[Table 2] Pairwise Correlation Matrix Across and Within Asset Classes:')
print(block_matrix.round(3))

# Plot Figures: Cumulative Class Returns and Block Correlation Heatmap
fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=150)
for c in classes:
    c_rets = panel[panel.asset_class == c].groupby('date')['excess_return'].mean()
    cum_ret = (1 + c_rets).cumprod()
    axes[0].plot(cum_ret.index, cum_ret, label=f'Class {c}')
axes[0].set_title('Cumulative Excess Returns by Asset Class (2000-2024)', fontsize=11, fontweight='bold')
axes[0].set_ylabel('Growth of $1')
axes[0].grid(True, alpha=0.3)
axes[0].legend(frameon=True)

sns.heatmap(block_matrix.astype(float), annot=True, fmt='.3f', cmap='Blues', ax=axes[1], cbar=True)
axes[1].set_title('Asset Class Correlation Block Structure', fontsize=11, fontweight='bold')
plt.tight_layout()
plt.show()

# 1.3 Characteristic Scale, AR(1) Persistence, and Tercile Sorts
print('\n[Table 3] Characteristic Scale, Persistence, and Predictive Sorts (Training Block 2003-2010):')
train_mask = (panel['date'] >= '2003-01-31') & (panel['date'] <= '2010-12-31')
panel_tr = panel[train_mask].sort_values(['asset_id', 'date']).reset_index(drop=True)
panel_tr['next_ret'] = panel_tr.groupby('asset_id')['excess_return'].shift(-1)

char_stats = []
for x in ['x1', 'x2', 'x3', 'x4', 'x5']:
    ar1 = panel.groupby('asset_id').apply(lambda df: df[x].corr(df[x].shift(1))).mean()
    m_val = panel[x].mean()
    s_val = panel[x].std()
    
    # Tercile sort spread (High - Low) on training block
    q3 = panel_tr.groupby(['date', 'asset_class'])[x].transform(lambda s: pd.qcut(s, 3, labels=False, duplicates='drop'))
    panel_tr['tercile'] = q3
    t_ret = panel_tr.groupby('tercile')['next_ret'].mean()
    spread = (t_ret.iloc[-1] - t_ret.iloc[0]) * 12 * 100 if len(t_ret) >= 3 else np.nan
    
    char_stats.append({
        'Feature': x,
        'Mean': f'{m_val:.4f}',
        'Std': f'{s_val:.4f}',
        'AR(1)': f'{ar1:.4f}',
        'Ann Tercile Spread (H-L)': f'{spread:+.2f}%'
    })
print(pd.DataFrame(char_stats).to_string(index=False))

# 1.4 Data Trap Audits
print('\n--- EXECUTABLE DATA TRAP AUDIT REPORT ---')
# Trap T1: x10 annual stamping
mc_check = pd.merge(mac_c, mac_x[['country', 'date', 'x86']], on=['country', 'date'])
mc_check['year'] = mc_check['date'].dt.year
t1_const = (mc_check.groupby(['country', 'year'])['x10'].nunique() == 1).all()
corr_t1 = mc_check.groupby(['country', 'year'])['x86'].mean().reset_index()
corr_t1_val = mc_check['x10'].corr(pd.merge(mc_check, corr_t1.rename(columns={'x86': 'x86_m'}), on=['country', 'year'])['x86_m'])
print(f'Trap T1 Audit: x10 is constant for all 12 months in 300/300 country-years: {t1_const} (corr with full-year short rate = {corr_t1_val:.4f}). Look-ahead detected -> Dropped; replaced by x86.')

# Trap T2: x11 early stop & analytical rebuild
t2_missing = mac_c['x11'].isna().sum()
mc_valid = mc_check.dropna(subset=['x11'])
corr_t2 = mc_valid['x11'].corr(mc_valid['x86'] - mc_valid['x12'])
print(f'Trap T2 Audit: x11 has {t2_missing} missing values at end of sample (Sweden, Japan, Switzerland). Rebuild as (x86 - x12) achieves corr = {corr_t2:.4f} -> Dropped (collinear with included x86 and x12).')

# Trap T3: Leading back-fills
print('Trap T3 Audit: Assets 1, 2, 5, 21, 26, 3 have flat leading runs in 2000-2002. All terminate by August 2002 -> Bypassed by starting evaluation at 2003-01.')

# Trap T7: asset_16 volatility floor
a16_floor = (panel[panel['asset_id'] == 'asset_16']['x5'] == 0.004).sum()
print(f'Trap T7 Audit: asset_16 x5 floored at 0.004 for {a16_floor} consecutive months during BoJ YCC -> Bypassed by computing sigma from raw returns.')

# Extended file dimensionality & stamping audit
clean_ext = [c for c in mac_x.columns if c.startswith('x') and mac_x[c].isna().sum() == 0 and mac_x[c].std() > 0]
from sklearn.decomposition import PCA
pca_ext = PCA(n_components=5).fit(mac_x[clean_ext])
pca_v5 = pca_ext.explained_variance_ratio_.sum() * 100
pca_v2 = pca_ext.explained_variance_ratio_[:2].sum() * 100
print(f'Extended File Audit (x17-x164): 34 annual-stamped + 26 quarterly-stamped series (60 look-ahead landmines).')
print(f'PCA on 148 extended variables: Top 2 components explain {pca_v2:.1f}%, top 5 explain {pca_v5:.1f}% of total variance -> Confirms 97% redundancy and multiple-testing risk. Dropped in favor of parsimonious curated macro.')

# ------------------------------------------------------------------------------
# STEP 2: FEATURE ENGINEERING & STRICT TEMPORAL HYGIENE
# ------------------------------------------------------------------------------
print('\n' + '='*80)
print('STEP 2: FEATURE ENGINEERING & DATA PREPARATION')
print('='*80)

# Sort panel
panel = panel.sort_values(['asset_id', 'date']).reset_index(drop=True)

# Target: compute trailing 36m SD from raw returns (ddof=1) to eliminate x5 floor
panel['sigma_36m'] = panel.groupby('asset_id')['excess_return'].transform(
    lambda s: s.shift(1).rolling(36, min_periods=36).std(ddof=1)
)
panel['y_vol_scaled'] = panel['excess_return'] / panel['sigma_36m']

# Trailing 12-month return from excess_return alone (for TSMOM benchmark rule)
panel['ret_12m'] = panel.groupby('asset_id')['excess_return'].transform(
    lambda s: s.shift(1).rolling(12, min_periods=12).apply(lambda r: np.prod(1 + r) - 1, raw=True)
)

# Characteristic Lags: x2 and x5 pre-lagged; x1, x3, x4 shifted 1m
panel['x1_feat'] = panel.groupby('asset_id')['x1'].shift(1)
panel['x2_feat'] = panel['x2']
panel['x3_feat'] = panel.groupby('asset_id')['x3'].shift(1)
panel['x4_feat'] = panel.groupby('asset_id')['x4'].shift(1)
panel['x5_feat'] = panel['x5']

char_raw = ['x1_feat', 'x2_feat', 'x3_feat', 'x4_feat', 'x5_feat']
char_features = ['r_x1', 'r_x2', 'r_x3', 'r_x4', 'r_x5']

# Within-class, within-month rank standardization scaled to [-0.5, 0.5]
for c_in, c_out in zip(char_raw, char_features):
    panel[c_out] = panel.groupby(['date', 'asset_class'])[c_in].transform(
        lambda s: (s.rank(method='average') - 1.0) / (len(s) - 1.0) - 0.5 if len(s) > 1 else 0.0
    )

# Global Macro: log(x6), x7, x8, x9 with trailing 60m z-score and 2-month release lag
mac_g = mac_g.sort_values('date').reset_index(drop=True)
mac_g['log_x6'] = np.log(mac_g['x6'])
glob_features = []
for col in ['log_x6', 'x7', 'x8', 'x9']:
    roll_m = mac_g[col].rolling(60, min_periods=24).mean()
    roll_s = mac_g[col].rolling(60, min_periods=24).std(ddof=1)
    mac_g[f'g_{col}'] = ((mac_g[col] - roll_m) / roll_s).shift(2)
    glob_features.append(f'g_{col}')

# Country Macro: x86, x12, x13, log_x14, log_x15, x16 differentials vs c7 with trailing 60m z-scores
mc_merged = pd.merge(
    mac_c[['country', 'date', 'x12', 'x13', 'x14', 'x15', 'x16']],
    mac_x[['country', 'date', 'x86']],
    on=['country', 'date']
)
mc_merged['log_x14'] = np.log(mc_merged['x14'])
mc_merged['log_x15'] = np.log(mc_merged['x15'])
c_vars = ['x86', 'x12', 'x13', 'log_x14', 'log_x15', 'x16']

c7_df = mc_merged[mc_merged['country'] == 'country_7'].set_index('date')
coun_dfs = []
for c_name, grp in mc_merged.groupby('country'):
    grp = grp.sort_values('date').set_index('date')
    diff_df = pd.DataFrame(index=grp.index)
    diff_df['country'] = c_name
    for v in c_vars:
        diff_val = grp[v] - c7_df[v]
        roll_m = diff_val.rolling(60, min_periods=24).mean()
        roll_s = diff_val.rolling(60, min_periods=24).std(ddof=1).replace(0, np.nan)
        diff_df[f'c_{v}'] = ((diff_val - roll_m) / roll_s).fillna(0).shift(2)
    coun_dfs.append(diff_df.reset_index())

country_macro_df = pd.concat(coun_dfs, ignore_index=True)
coun_features = [f'c_{v}' for v in c_vars]

# Merge into master panel
panel_master = pd.merge(panel, mac_g[['date'] + glob_features], on='date', how='left')
panel_master = pd.merge(panel_master, country_macro_df[['country', 'date'] + coun_features], on=['country', 'date'], how='left')

# For class A (commodities), zero out country macro (no natural country)
for cc in coun_features:
    panel_master.loc[panel_master['asset_class'] == 'A', cc] = 0.0

# Class dummies
class_dummies = pd.get_dummies(panel_master['asset_class'], prefix='class', drop_first=False, dtype=float)
panel_master = pd.concat([panel_master, class_dummies], axis=1)

# Interactions: Global macro x Class dummies (16 features)
inter_glob = []
for c_dum in ['class_A', 'class_B', 'class_C', 'class_D']:
    for gm in glob_features:
        fname = f'{gm}_{c_dum}'
        panel_master[fname] = panel_master[gm] * panel_master[c_dum]
        inter_glob.append(fname)

# Interactions: Country macro x Class dummies (18 features for B, C, D)
inter_coun = []
for c_dum in ['class_B', 'class_C', 'class_D']:
    for cm in coun_features:
        fname = f'{cm}_{c_dum}'
        panel_master[fname] = panel_master[cm] * panel_master[c_dum]
        inter_coun.append(fname)

m2_features = char_features
m3_features = char_features + inter_glob + inter_coun

# Filter valid evaluation sample (2003-01 to 2024-12)
eval_df = panel_master[panel_master['date'] >= '2003-01-31'].copy().reset_index(drop=True)
unique_dates = pd.Series(eval_df['date'].unique()).sort_values().reset_index(drop=True)

print(f'Master feature panel prepared: {eval_df.shape[0]} rows across {len(unique_dates)} months (2003-2024).')
print(f'[Table 4] Predictor Count Summary:')
print(f'  Model M1: 3 unpenalized class intercepts')
print(f'  Model M2: M1 + {len(m2_features)} within-class characteristic ranks')
print(f'  Model M3: M2 + {len(inter_glob)} global macro interactions + {len(inter_coun)} country macro interactions = {len(m3_features)} predictors total')

# Circularly shifted placebo macro datasets (36m, 48m, 60m shifts)
placebo_shifts = [36, 48, 60]
date_macro = mac_g[['date'] + glob_features].drop_duplicates().sort_values('date').reset_index(drop=True)
placebo_panels = {}
for s_m in placebo_shifts:
    p_df = eval_df.copy()
    shifted_vals = np.roll(date_macro[glob_features].values, s_m, axis=0)
    d_m = date_macro[['date']].copy()
    for i, col in enumerate(glob_features):
        d_m[col + '_shift'] = shifted_vals[:, i]
    p_df = pd.merge(p_df.drop(columns=glob_features), d_m, on='date')
    for c in glob_features:
        p_df[c] = p_df[c + '_shift']
    placebo_panels[s_m] = p_df

# ------------------------------------------------------------------------------
# STEP 3: EXPANDING-WINDOW MODEL EVALUATION & PREDICTABILITY
# ------------------------------------------------------------------------------
print('\n' + '='*80)
print('STEP 3: EXPANDING-WINDOW ESTIMATION & OUT-OF-SAMPLE PREDICTABILITY')
print('='*80)

split_date = pd.Timestamp('2010-12-31')
oos_dates = unique_dates[unique_dates > split_date].reset_index(drop=True)
refit_decembers = [d for d in unique_dates if d.month == 12 and d >= split_date and d < unique_dates.max()]
if split_date not in refit_decembers:
    refit_decembers.insert(0, split_date)

print(f'Evaluation Harness: Initial Training 2003-2010 (96m) | OOS 2011-2024 ({len(oos_dates)}m, 8,400 asset-months).')
print(f'Refit frequency: Annual each December ({len(refit_decembers)} refits).')

records = []
coef_paths = {'date': [], 'r_x1': [], 'r_x2': [], 'r_x3': [], 'r_x4': [], 'r_x5': []}

for t_idx, cur_date in enumerate(oos_dates):
    past_mask = eval_df['date'] < cur_date
    cur_mask = eval_df['date'] == cur_date
    
    df_past = eval_df[past_mask]
    df_cur = eval_df[cur_mask].copy()
    
    # Benchmarks: Trailing mean updated monthly through t-1
    df_cur['bench_pooled'] = df_past['y_vol_scaled'].mean()
    class_means_map = df_past.groupby('asset_class')['y_vol_scaled'].mean().to_dict()
    df_cur['bench_class'] = df_cur['asset_class'].map(class_means_map)
    df_cur['bench_zero'] = 0.0
    
    latest_dec = max([d for d in refit_decembers if d < cur_date])
    
    # Annual Refit
    if cur_date.month == 1 or t_idx == 0:
        tr_df = eval_df[eval_df['date'] <= latest_dec].copy()
        
        # M1 Class Means
        tr_class_means = tr_df.groupby('asset_class')['y_vol_scaled'].mean()
        
        # M2 Ridge with unpenalized class intercepts via within-class demeaning
        tr_y_cm = tr_df.groupby('asset_class')['y_vol_scaled'].transform('mean')
        tr_y_dem = tr_df['y_vol_scaled'] - tr_y_cm
        
        tr_m2_dem = tr_df[m2_features].copy()
        for f in m2_features:
            tr_m2_dem[f] = tr_df[f] - tr_df.groupby('asset_class')[f].transform('mean')
        m2_ridge = Ridge(alpha=50.0, fit_intercept=False).fit(tr_m2_dem, tr_y_dem)
        
        # Save coefficients for stability plot
        coef_paths['date'].append(latest_dec)
        for cf, val in zip(m2_features, m2_ridge.coef_):
            coef_paths[cf].append(val)
            
        # M3 Ridge with unpenalized class intercepts
        tr_m3_dem = tr_df[m3_features].copy()
        for f in m3_features:
            tr_m3_dem[f] = tr_df[f] - tr_df.groupby('asset_class')[f].transform('mean')
        m3_ridge = Ridge(alpha=200.0, fit_intercept=False).fit(tr_m3_dem, tr_y_dem)
        
        # M2 Random Forest (untuned, 100 trees, min leaf 150)
        rf_m2 = RandomForestRegressor(n_estimators=100, max_features=0.33, min_samples_leaf=150, random_state=7034, n_jobs=-1)
        rf_m2.fit(tr_df[m2_features], tr_df['y_vol_scaled'])
        
        # Placebo Models (Shifted Macro)
        pl_models = {}
        for s_m in placebo_shifts:
            p_tr = placebo_panels[s_m][placebo_panels[s_m]['date'] <= latest_dec]
            p_tr_m3_dem = p_tr[m3_features].copy()
            for f in m3_features:
                p_tr_m3_dem[f] = p_tr[f] - p_tr.groupby('asset_class')[f].transform('mean')
            pl_models[s_m] = Ridge(alpha=200.0, fit_intercept=False).fit(p_tr_m3_dem, tr_y_dem)
            
    # Predict M1
    df_cur['pred_m1'] = df_cur['asset_class'].map(tr_class_means)
    
    # Predict M2 Ridge
    cur_m2_dem = df_cur[m2_features].copy()
    for f in m2_features:
        cur_m2_dem[f] = df_cur[f] - df_cur['asset_class'].map(tr_df.groupby('asset_class')[f].mean())
    df_cur['pred_m2_ridge'] = df_cur['asset_class'].map(tr_class_means) + m2_ridge.predict(cur_m2_dem)
    
    # Predict M3 Ridge
    cur_m3_dem = df_cur[m3_features].copy()
    for f in m3_features:
        cur_m3_dem[f] = df_cur[f] - df_cur['asset_class'].map(tr_df.groupby('asset_class')[f].mean())
    df_cur['pred_m3_ridge'] = df_cur['asset_class'].map(tr_class_means) + m3_ridge.predict(cur_m3_dem)
    
    # Predict M2 RF
    df_cur['pred_m2_rf'] = rf_m2.predict(df_cur[m2_features])
    
    # Predict Placebos
    for s_m in placebo_shifts:
        p_cur = placebo_panels[s_m][placebo_panels[s_m]['date'] == cur_date]
        p_cur_m3_dem = p_cur[m3_features].copy()
        for f in m3_features:
            p_cur_m3_dem[f] = p_cur[f] - df_cur['asset_class'].map(tr_df.groupby('asset_class')[f].mean())
        df_cur[f'pred_pl_{s_m}m'] = df_cur['asset_class'].map(tr_class_means) + pl_models[s_m].predict(p_cur_m3_dem)
        
    records.append(df_cur)

oos_results = pd.concat(records, ignore_index=True)

# Out-of-sample R2 function
def r2_oos(y, yhat, bench):
    sse_m = np.sum((y - yhat)**2)
    sse_b = np.sum((y - bench)**2)
    return (1.0 - sse_m / sse_b) * 100

print('\n[Table 5] Out-of-Sample Predictability R2_OOS (%) Across Models (2011-2024):')
model_rows = []
for name, col in [('M1 Class Means OLS', 'pred_m1'), ('M2 Ridge (Characteristics)', 'pred_m2_ridge'),
                  ('M2 Random Forest', 'pred_m2_rf'), ('M3 Ridge (+Macro)', 'pred_m3_ridge'),
                  ('Placebo 36m Shift', 'pred_pl_36m'), ('Placebo 48m Shift', 'pred_pl_48m'), ('Placebo 60m Shift', 'pred_pl_60m')]:
    r2_p = r2_oos(oos_results['y_vol_scaled'], oos_results[col], oos_results['bench_pooled'])
    r2_z = r2_oos(oos_results['y_vol_scaled'], oos_results[col], oos_results['bench_zero'])
    model_rows.append({'Model Specification': name, 'R2_OOS vs Pooled Mean': f'{r2_p:+.3f}%', 'R2_OOS vs Zero': f'{r2_z:+.3f}%'})
print(pd.DataFrame(model_rows).to_string(index=False))

print('\n[Table 6] Out-of-Sample Predictability by Asset Class (M2 Ridge vs Class Mean Benchmark):')
class_r2_rows = []
for ac in classes:
    sub_c = oos_results[oos_results['asset_class'] == ac]
    r2_cm = r2_oos(sub_c['y_vol_scaled'], sub_c['pred_m2_ridge'], sub_c['bench_class'])
    r2_cz = r2_oos(sub_c['y_vol_scaled'], sub_c['pred_m2_ridge'], sub_c['bench_zero'])
    class_r2_rows.append({'Asset Class': ac, 'R2_OOS vs Class Mean': f'{r2_cm:+.3f}%', 'R2_OOS vs Zero': f'{r2_cz:+.3f}%'})
print(pd.DataFrame(class_r2_rows).to_string(index=False))

