import os, pickle, numpy as np, pandas as pd
os.chdir('/tmp/claude-0/-home-user-UChicago/198decc2-a5bc-5c5b-85f9-9be4242a489b/scratchpad/blind/data')
ns={}
exec(open('../reports/p3models_work/A_code.py').read(), ns)
o=ns['oos_results']
o.to_pickle('../reports/p3models_work/A_oos.pkl')
for s in [36,48,60]:
    d=(o[f'pred_pl_{s}m']-o['pred_m3_ridge']).abs().max()
    print('placebo',s,'max |diff| vs M3:',d)
# check whether placebo panels differ in the model features
pp=ns['placebo_panels']; ev=ns['eval_df']; m3=ns['m3_features']; gf=ns['glob_features']
for s in [36]:
    p=pp[s]
    print('raw glob cols differ:', (p[gf].values!=ev[gf].values).any())
    print('m3 feature cols differ:', np.nanmax(np.abs(p[m3].values-ev[m3].values)))
    print('glob cols in m3?', [g for g in gf if g in m3])
pickle.dump({'coef_paths':ns['coef_paths']}, open('../reports/p3models_work/A_coef.pkl','wb'))
print(pd.DataFrame(ns['coef_paths']))
