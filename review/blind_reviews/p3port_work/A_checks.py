import pandas as pd, numpy as np, statsmodels.api as sm
oos = pd.read_pickle('A_oos.pkl').sort_values(['date','asset_id'])
port = pd.read_pickle('A_port.pkl')
R = oos.pivot(index='date', columns='asset_id', values='excess_return')
S = oos.pivot(index='date', columns='asset_id', values='sigma_36m')
cls = oos.drop_duplicates('asset_id').set_index('asset_id')['asset_class'].reindex(R.columns)
def W_of(col):
    Y = oos.pivot(index='date', columns='asset_id', values=col)
    W = Y / S
    return W.div(W.abs().sum(1), axis=0)
def ug(W): return W.div(W.abs().sum(1), axis=0)
W = {'EW': ug(pd.DataFrame(1.0, index=R.index, columns=R.columns)),
     'RP': ug(1/S),
     'TSMOM': ug(np.sign(oos.pivot(index='date', columns='asset_id', values='ret_12m'))/S),
     'P1': W_of('pred_m2_ridge'), 'M1': W_of('pred_m1')}
# within-class char-only (P3-like): demean forecast within class
Y = oos.pivot(index='date', columns='asset_id', values='pred_m2_ridge')
dev = Y - Y.T.groupby(cls).transform('mean').T
W['P3_charonly'] = ug(dev/S)
# frozen 2010 class-means tilt
cm2010 = oos[oos.date==oos.date.min()].set_index('asset_id')['pred_m1'].reindex(R.columns)
W['M1_frozen2010'] = ug((1/S).mul(cm2010, axis=1))
def pret(W): return (W*R).sum(1)
def to_target(W): 
    t = (W - W.shift(1)).abs().sum(1); t.iloc[0]=0; return t
def to_drift(W):
    rp = pret(W)
    d = (W.shift(1)*(1+R.shift(1))).div(1+rp.shift(1), axis=0)
    t = (W - d.fillna(0)).abs().sum(1); t.iloc[0]=0; return t
def stats(r):
    w=(1+r).cumprod(); return dict(mean=12*r.mean(), vol=np.sqrt(12)*r.std(), SR=np.sqrt(12)*r.mean()/r.std(), MDD=(1-w/w.cummax()).max())
rows={}; NET={}
for k,w in W.items():
    g=pret(w); tt=to_target(w); td=to_drift(w)
    NET[k]=g-0.001*td
    rows[k]={**{f'g_{a}':b for a,b in stats(g).items()}, 'SR_net_drift10':stats(g-0.001*td)['SR'],'SR_net_target10':stats(g-0.001*tt)['SR'],
             'TO_target':tt.iloc[1:].mean(),'TO_drift':td.iloc[1:].mean()}
tab=pd.DataFrame(rows).T
print(tab.round(3).to_string())
# check reproduction vs A's port_df
print('P1 gross max abs diff vs A:', (pret(W['P1'])-port['P1_gross']).abs().max(), ' EW:', (pret(W['EW'])-port['EW_gross']).abs().max(),
      ' TSMOM:', (pret(W['TSMOM'])-port['TSMOM_gross']).abs().max())
# weight correlations
def avgcorr(a,b): return np.mean([np.corrcoef(a.iloc[i],b.iloc[i])[0,1] for i in range(len(a))])
print('avg corr P1-M1 weights', round(avgcorr(W['P1'],W['M1']),4), ' P1-RP', round(avgcorr(W['P1'],W['RP']),4), ' P1-TSMOM', round(avgcorr(W['P1'],W['TSMOM']),4))
# class shares
sh = pd.DataFrame({k: W[k].abs().T.groupby(cls).sum().T.mean() for k in ['RP','P1','M1']}).T
print('gross share by class\n', sh.round(3))
print('P1 asset_16 avg |w|', round(W['P1']['asset_16'].abs().mean(),3), ' max |w| any asset', round(W['P1'].abs().max().max(),3),
      ' argmax asset', W['P1'].abs().max().idxmax())
print('P1 top-5 avg |w| assets\n', W['P1'].abs().mean().sort_values(ascending=False).head(5).round(3))
print('P1 net exposure avg', round(W['P1'].sum(1).mean(),3), ' months short a class', round(np.mean([(W['P1'].iloc[i].groupby(cls).sum()<0).any() for i in range(len(R))]),3))
print('class B mean-forecast sign by year:'); print(oos.groupby([oos.date.dt.year,'asset_class'])['pred_m1'].mean().unstack().round(3))
# regressions using A's own net series (A's cost convention) and drift-based
dfA = port[['P1_net','EW_net','RP_net','TSMOM_net']].copy()
M1g = pret(W['M1']); M1net_A = M1g - 0.001*to_target(W['M1']).where(lambda s: s.index!=s.index[0], 1.0)
dfA['M1_net']=M1net_A.values
for lab, X in [('P1 ~ EW+RP+TS', ['EW_net','RP_net','TSMOM_net']), ('P1 ~ EW+RP+TS+M1', ['EW_net','RP_net','TSMOM_net','M1_net']), ('M1 ~ EW+RP+TS', None)]:
    y = dfA['M1_net'] if X is None else dfA['P1_net']; X = X or ['EW_net','RP_net','TSMOM_net']
    f = sm.OLS(y, sm.add_constant(dfA[X])).fit()
    print(lab, 'alpha ann %.2f%% t=%.2f R2=%.3f'%(1200*f.params['const'], f.tvalues['const'], f.rsquared), 'betas', f.params.drop('const').round(3).to_dict())
    fh = sm.OLS(y, sm.add_constant(dfA[X])).fit(cov_type='HAC', cov_kwds={'maxlags':6}); print('   HAC t(alpha)=%.2f'%fh.tvalues['const'])
# P1 minus M1 return spread
d = port['P1_net'].values - M1net_A.values
print('P1-M1 net spread ann mean %.3f%%, t=%.2f'%(1200*d.mean(), d.mean()/d.std(ddof=1)*np.sqrt(len(d))))
# sub-period alpha
for a,b in [('2011','2017'),('2018','2024')]:
    s = dfA.loc[a:b]; f=sm.OLS(s['P1_net'], sm.add_constant(s[['EW_net','RP_net','TSMOM_net']])).fit()
    print('subperiod',a,b,'alpha %.2f%% t=%.2f'%(1200*f.params['const'], f.tvalues['const']), ' P1 SR %.2f RP SR %.2f EW SR %.2f'%tuple(np.sqrt(12)*s[c].mean()/s[c].std() for c in ['P1_net','RP_net','EW_net']))
# bootstrap Sharpe differences (A's net series)
rng=np.random.default_rng(0); T=len(dfA); B=10000; idx=rng.integers(0,T,(B,T))
def sr(a): return np.sqrt(12)*a.mean(1)/a.std(1,ddof=1)
P=dfA['P1_net'].to_numpy()
for b in ['EW_net','RP_net','TSMOM_net','M1_net']:
    bb=dfA[b].to_numpy(); diff=sr(P[idx])-sr(bb[idx]); pt=np.sqrt(12)*(P.mean()/P.std(ddof=1)-bb.mean()/bb.std(ddof=1))
    print('SR(P1)-SR(%s) = %.3f, boot SE %.3f, 95%% CI [%.3f, %.3f]'%(b,pt,diff.std(ddof=1),*np.percentile(diff,[2.5,97.5])))
# block bootstrap 12m for RP diff
L=12; nb=int(np.ceil(T/L)); diffs=[]
for _ in range(5000):
    st=rng.integers(0,T-L+1,nb); ix=np.concatenate([np.arange(s,s+L) for s in st])[:T]
    diffs.append(np.sqrt(12)*(P[ix].mean()/P[ix].std(ddof=1)-dfA['RP_net'].to_numpy()[ix].mean()/dfA['RP_net'].to_numpy()[ix].std(ddof=1)))
print('block(12) boot SE SR diff vs RP %.3f'%np.std(diffs,ddof=1))
