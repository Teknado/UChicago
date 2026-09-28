import pandas as pd, numpy as np
D='data/'
mc=pd.read_csv(D+'macro_country.csv',parse_dates=['date'])
mx=pd.read_csv(D+'macro_country_extended.csv',parse_dates=['date'])
mm=mc.merge(mx,on=['country','date']).sort_values(['country','date'])
cols=[c for c in mx.columns if c.startswith('x') and mx[c].notna().all()]
ov=mm.dropna(subset=['x11'])
best=[]
for a in cols+['x10','x12','x13']:
    for b in ['x12']+[c for c in cols if ov[c].std()>0]:
        if a==b: continue
        r=ov[a]-ov[b]
        e=np.sqrt(((ov.x11-r)**2).mean())
        best.append((e,a,b))
best.sort(); print(best[:8])
# per-country quality for top
for e,a,b in best[:2]:
    for c in ['country_12','country_9','country_11']:
        d=ov[ov.country==c]; print(a,b,c, d.x11.corr(d[a]-d[b]), np.sqrt(((d.x11-(d[a]-d[b]))**2).mean()))
# lagged inflation version
mm['x12l']=mm.groupby('country').x12.shift(1)
ov=mm.dropna(subset=['x11','x12l'])
print('x86 - x12 lag1', np.sqrt(((ov.x11-(ov.x86-ov.x12l))**2).mean()))
# per country 12 : what is x11 - (x86-x12)
d=mm[mm.country=='country_12'].dropna(subset=['x11'])
res=d.x11-(d.x86-d.x12)
print(res.describe())
print(d[['date','x11','x86','x12']].assign(res=res).iloc[::24].to_string())
