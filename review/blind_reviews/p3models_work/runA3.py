import numpy as np, pandas as pd
o=pd.read_pickle('/tmp/claude-0/-home-user-UChicago/198decc2-a5bc-5c5b-85f9-9be4242a489b/scratchpad/blind/reports/p3models_work/A_oos.pkl')
rng=np.random.default_rng(2)
for c in 'ABCD':
    s=o[o.asset_class==c]; mi=pd.factorize(s.date)[0]; T=mi.max()+1; idx=rng.integers(0,T,size=(5000,T))
    ms=lambda v: np.bincount(mi,weights=v,minlength=T)
    y=s.y_vol_scaled.values
    for n,b in [('pred_m2_ridge','bench_class'),('pred_m1','bench_class')]:
        e=ms((y-s[n].values)**2); eb=ms((y-s[b].values)**2)
        print(c,n,'vs',b,f'{1-e.sum()/eb.sum():.3%}', f'SE {(1-e[idx].sum(1)/eb[idx].sum(1)).std(ddof=1):.3%}')
    e1=ms((y-s.pred_m2_ridge.values)**2); e0=ms((y-s.pred_m1.values)**2); eb=ms((y-s.bench_class.values)**2)
    print(c,'M2 minus M1 (vs class mean):', f'{(e0.sum()-e1.sum())/eb.sum():.3%}')
