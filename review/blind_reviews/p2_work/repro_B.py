import numpy as np, pandas as pd
from scipy.stats import kurtosis, norm
SEED=2694; N_SCEN=500; NIGHTS=2500; N_DESKS=1000; Z01=norm.ppf(.99)
STREAMS = ['one_desk', 'desks_1000', 'mart_s0_1', 'mart_s0_one', 'drift_50', 'drift_500', 'drift_5000',
           'drift_50_precise', 'mixed_real', 'mixed_synth', 'dj_scen', 'kurt_null', 'dj_boot']
_SS = dict(zip(STREAMS, np.random.SeedSequence(SEED).spawn(len(STREAMS))))
rng_for=lambda n: np.random.default_rng(_SS[n])
rng=rng_for('one_desk'); s=np.empty(NIGHTS); s[0]=1
for k in range(1,NIGHTS): s[k]=rng.normal(0,np.sqrt(s[k-1]),N_SCEN).var(ddof=1)
print('B one desk night2500',s[-1], 'log at nights 100,300,500,1000:',np.log(s[[99,299,499,999]]))
rng=rng_for('desks_1000'); s2=np.ones(N_DESKS)
for k in range(1,NIGHTS): s2=(rng.standard_normal((N_DESKS,N_SCEN))*np.sqrt(s2)[:,None]).var(axis=1,ddof=1)
print('B 1000: mean',s2.mean(),'median',np.median(s2),'p5,p95',np.percentile(s2,[5,95]),'max',s2.max(),'frac<.01',(s2<.01).mean(),'pct one',(s2<s[-1]).mean())
srt=np.sort(s2)[::-1]; print('top10',srt[:10].sum()/srt.sum())
# 2.4b scenario kurtosis
dj=pd.read_csv('../../data/dj30.csv'); r=dj.groupby('date').MrkRet.first()
sd=r.std(ddof=1); print('sd',sd,'kurt',kurtosis(r),'empVaR',-np.percentile(r,1),'pipeVaR',Z01*sd,'n',len(r))
print('B night1 scen kurt',kurtosis(rng_for('dj_scen').normal(0,sd,N_SCEN)))
