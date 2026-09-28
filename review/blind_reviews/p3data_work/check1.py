import pandas as pd, numpy as np
D='data/'
panel=pd.read_csv(D+'asset_panel.csv',parse_dates=['date'])
mg=pd.read_csv(D+'macro_global.csv',parse_dates=['date'])
mc=pd.read_csv(D+'macro_country.csv',parse_dates=['date'])
mx=pd.read_csv(D+'macro_country_extended.csv',parse_dates=['date'])
print(panel.columns.tolist(), mc.columns.tolist(), mx.shape)
# x10 annual
cm=mc.merge(mx[['country','date','x86']],on=['country','date'])
cm['year']=cm.date.dt.year
g=cm.groupby(['country','year'])
print('x10 const share', (g.x10.nunique()==1).mean())
yr=g.agg(x10=('x10','first'),same=('x86','mean'))
yr['prior']=yr.groupby(level=0).same.shift(1)
print('corr same', yr.x10.corr(yr.same), 'prior', yr.dropna().x10.corr(yr.dropna().prior))
print('max |x10-same|', (yr.x10-yr.same).abs().describe())
# x11
for c,d in cm.groupby('country'):
    n=d.x11.isna().sum()
    if n: print(c, n, d.loc[d.x11.notna(),'date'].max().date(), 'interior?', d.x11.isna().sum()-(d.date>d.loc[d.x11.notna(),'date'].max()).sum())
ov=cm.dropna(subset=['x11'])
print('rebuild corr x86-x12', ov.x11.corr(ov.x86-ov.x12), 'rmse', np.sqrt(((ov.x11-(ov.x86-ov.x12))**2).mean()))
# try better rebuild: regress x11 on other complete columns? check x-cols in mx highly corr with x11
mm=mc.merge(mx,on=['country','date'])
cols=[c for c in mx.columns if c.startswith('x') and mx[c].notna().all()]
cc=mm.dropna(subset=['x11'])[cols+['x11']].corr()['x11'].drop('x11').abs().sort_values(ascending=False)
print(cc.head(8))
# stale ffill error at the end
for c in ['country_12','country_9','country_11']:
    d=cm[cm.country==c].sort_values('date')
    miss=d.x11.isna()
    print(c,'stale err mean',(d.x11.ffill()-(d.x86-d.x12))[miss].abs().mean())
# back-fills
X=['x1','x2','x3','x4','x5']
for x in X:
    W=panel.pivot(index='date',columns='asset_id',values=x)
    for a in W:
        s=W[a].to_numpy(); run=1
        while run<len(s) and s[run]==s[0]: run+=1
        if run>1: print('leading run',x,a,run-1,'fills')
    # trailing constant runs
    for a in W:
        s=W[a].to_numpy(); run=1
        while run<len(s) and s[-1-run]==s[-1]: run+=1
        if run>=3: print('trailing run',x,a,run)
W5=panel.pivot(index='date',columns='asset_id',values='x5')
print('asset_16 floor', (W5['asset_16']==0.004).sum())
print((W5==W5.min().min()).sum().sort_values().tail(3))
# x5 floor elsewhere
print('min x5 per asset lowest', W5.min().sort_values().head(3))
