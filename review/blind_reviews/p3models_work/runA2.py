import os, numpy as np, pandas as pd
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
os.chdir('/tmp/claude-0/-home-user-UChicago/198decc2-a5bc-5c5b-85f9-9be4242a489b/scratchpad/blind/data')
code=open('../reports/p3models_work/A_code.py').read()
k=code.find('# STEP 3'); k=code.rfind('# ----',0,k)
ns={}; exec(code[:k], ns)
ev=ns['eval_df']; m2f=ns['m2_features']; m3f=ns['m3_features']
o=pd.read_pickle('../reports/p3models_work/A_oos.pkl')
dates=pd.Series(ev.date.unique()).sort_values().reset_index(drop=True)
oos_d=dates[dates>pd.Timestamp('2010-12-31')].reset_index(drop=True)

def run_ridge(df, feats, alpha):
    out=[]
    for yr in range(2011,2025):
        dec=pd.Timestamp(f'{yr-1}-12-31')
        tr=df[df.date<=dec]; te=df[(df.date>dec)&(df.date<=pd.Timestamp(f'{yr}-12-31'))]
        cm=tr.groupby('asset_class')['y_vol_scaled'].mean()
        ydem=tr['y_vol_scaled']-tr['asset_class'].map(cm)
        xm=tr.groupby('asset_class')[feats].mean()
        Xd=tr[feats]-xm.loc[tr.asset_class].values
        if alpha is None:
            p=te['asset_class'].map(cm)
        else:
            mdl=Ridge(alpha=alpha,fit_intercept=False).fit(Xd,ydem)
            p=te['asset_class'].map(cm)+mdl.predict(te[feats]-xm.loc[te.asset_class].values)
        out.append(pd.Series(p.values,index=te.index))
    return pd.concat(out)

key=['date','asset_id']
base=ev.loc[ev.date>pd.Timestamp('2010-12-31'),key+['y_vol_scaled','asset_class']].copy()
m=base.merge(o[key+['pred_m1','pred_m2_ridge','pred_m3_ridge','pred_m2_rf','bench_pooled','bench_class']],on=key)
assert len(m)==8400
def r2(y,f,b): return 1-((y-f)**2).sum()/((y-b)**2).sum()
p2=run_ridge(ev,m2f,50.0); p3=run_ridge(ev,m3f,200.0)
m['re_m2']=p2.loc[ev.index[ev.date>pd.Timestamp('2010-12-31')]].values
m['re_m3']=p3.loc[ev.index[ev.date>pd.Timestamp('2010-12-31')]].values
print('reproduce M2 max diff', (m.re_m2-m.pred_m2_ridge).abs().max(), 'M3', (m.re_m3-m.pred_m3_ridge).abs().max())
y=m.y_vol_scaled.values; B=m.bench_pooled.values
# bootstrap
mi=pd.factorize(m.date)[0]; T=mi.max()+1
rng=np.random.default_rng(1); idx=rng.integers(0,T,size=(5000,T))
def ms(v): return np.bincount(mi,weights=v,minlength=T)
def r2se(f,b=B):
    e=ms((y-f)**2); eb=ms((y-b)**2); return 1-e.sum()/eb.sum(), (1-e[idx].sum(1)/eb[idx].sum(1)).std(ddof=1)
def dse(f1,f2,b=B):
    e1=ms((y-f1)**2); e2=ms((y-f2)**2); eb=ms((y-b)**2)
    return (e2.sum()-e1.sum())/eb.sum(), ((e2[idx].sum(1)-e1[idx].sum(1))/eb[idx].sum(1)).std(ddof=1)
print('\n== month-bootstrap SEs for A (vs pooled mean) ==')
for n in ['pred_m1','pred_m2_ridge','pred_m2_rf','pred_m3_ridge']:
    r,s=r2se(m[n].values); print(f'{n}: R2 {r:.3%} SE {s:.3%}')
for a_,b_ in [('pred_m2_ridge','pred_m1'),('pred_m3_ridge','pred_m2_ridge'),('pred_m2_rf','pred_m2_ridge')]:
    d,s=dse(m[a_].values,m[b_].values); print(f'{a_} minus {b_}: {d:.3%} SE {s:.3%} t={d/s:.2f}')
print('\n== hindsight alpha sweep (OOS R2 vs pooled) ==')
for a in [1,10,50,200,1000,1e4,1e5,1e6]:
    f2=run_ridge(ev,m2f,a).loc[ev.index[ev.date>pd.Timestamp('2010-12-31')]].values
    f3=run_ridge(ev,m3f,a).loc[ev.index[ev.date>pd.Timestamp('2010-12-31')]].values
    print(f'alpha {a:g}: M2 {r2(y,f2,B):.3%}  M3 {r2(y,f3,B):.3%}')
# correct placebo: shift raw global and country macro, rebuild z-scores & interactions
mac_g=pd.read_csv('macro_global.csv',parse_dates=['date']).sort_values('date').reset_index(drop=True)
mac_c=pd.read_csv('macro_country.csv',parse_dates=['date']); mac_x=pd.read_csv('macro_country_extended.csv',parse_dates=['date'])
def build(shift):
    g=mac_g.copy(); g['log_x6']=np.log(g['x6'])
    gf=[]
    for col in ['log_x6','x7','x8','x9']:
        v=pd.Series(np.roll(g[col].values,shift)) if shift else g[col]
        rm=v.rolling(60,min_periods=24).mean(); rs=v.rolling(60,min_periods=24).std(ddof=1)
        g[f'g_{col}']=((v-rm)/rs).shift(2).values; gf.append(f'g_{col}')
    mc=pd.merge(mac_c[['country','date','x12','x13','x14','x15','x16']],mac_x[['country','date','x86']],on=['country','date'])
    mc['log_x14']=np.log(mc['x14']); mc['log_x15']=np.log(mc['x15'])
    cv=['x86','x12','x13','log_x14','log_x15','x16']
    piv={v: mc.pivot(index='date',columns='country',values=v).sort_index() for v in cv}
    rows=[]
    outs={}
    for v in cv:
        P=piv[v]
        if shift: P=pd.DataFrame(np.roll(P.values,shift,axis=0),index=P.index,columns=P.columns)
        D=P.sub(P['country_7'],axis=0)
        rm=D.rolling(60,min_periods=24).mean(); rs=D.rolling(60,min_periods=24).std(ddof=1).replace(0,np.nan)
        outs[v]=((D-rm)/rs).fillna(0).shift(2)
    cdf=pd.concat({f'c_{v}':outs[v].stack(future_stack=True) for v in cv},axis=1).reset_index()
    cdf.columns=['date','country']+[f'c_{v}' for v in cv]
    df=ev.drop(columns=[c for c in ev.columns if c.startswith('g_') or c.startswith('c_')]).merge(g[['date']+gf],on='date',how='left').merge(cdf,on=['country','date'],how='left')
    df.index=ev.index
    for cc in [f'c_{v}' for v in cv]: df.loc[df.asset_class=='A',cc]=0.0
    for cd in ['class_A','class_B','class_C','class_D']:
        for gm in gf: df[f'{gm}_{cd}']=df[gm]*df[cd]
    for cd in ['class_B','class_C','class_D']:
        for cm in [f'c_{v}' for v in cv]: df[f'{cm}_{cd}']=df[cm]*df[cd]
    return df
d0=build(0)
print('\nrebuild(0) matches A M3 features max diff:', np.nanmax(np.abs(d0[m3f].values-ev[m3f].values)))
oi=ev.index[ev.date>pd.Timestamp('2010-12-31')]
g_real=dse(m.pred_m3_ridge.values,m.pred_m2_ridge.values)
print(f'real macro gain M3-M2: {g_real[0]:.3%} SE {g_real[1]:.3%}')
for s in [36,48,60,72,84,96,108,120]:
    ds=build(s); f=run_ridge(ds,m3f,200.0).loc[oi].values
    d,se=dse(f,m.pred_m2_ridge.values)
    print(f'correct placebo s={s}: R2 {r2(y,f,B):.3%}; gain vs M2 {d:.3%} (SE {se:.3%}); max|diff vs real M3| {np.abs(f-m.pred_m3_ridge.values).max():.3f}')
# RF fairness: add class dummies
print('\n== RF with class dummies (A settings) ==')
feats=m2f+['class_B','class_C','class_D']
out=[]
for yr in range(2011,2025):
    dec=pd.Timestamp(f'{yr-1}-12-31'); tr=ev[ev.date<=dec]; te=ev[(ev.date>dec)&(ev.date<=pd.Timestamp(f'{yr}-12-31'))]
    rf=RandomForestRegressor(n_estimators=100,max_features=0.33,min_samples_leaf=150,random_state=7034,n_jobs=-1).fit(tr[feats],tr.y_vol_scaled)
    out.append(pd.Series(rf.predict(te[feats]),index=te.index))
frf=pd.concat(out).loc[oi].values
print(f'RF+dummies R2 {r2(y,frf,B):.3%}; minus ridge M2 {dse(frf,m.pred_m2_ridge.values)}')
