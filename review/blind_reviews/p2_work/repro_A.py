import numpy as np
SEED=2694; N_SCEN=500; NIGHTS=2500; N_DESKS=1000
np.random.seed(SEED)
s=np.zeros(NIGHTS+1); s[0]=1; v=1.0
for t in range(1,NIGHTS+1):
    v=np.var(np.random.normal(0,np.sqrt(v),N_SCEN),ddof=1); s[t]=v
print('one desk idx2500 (night 2501):',s[-1],' idx2499 (night 2500):',s[-2])
np.random.seed(SEED)
cv=np.ones(N_DESKS); hist=[]
for t in range(1,NIGHTS+1):
    Z=np.random.normal(0,1,(N_DESKS,N_SCEN)); cv*=np.var(Z,axis=1,ddof=1)
    if t==NIGHTS-1: cv_2499=cv.copy()
print('A 1000 desks (2500 steps): mean',cv.mean(),'median',np.median(cv),'p5',np.percentile(cv,5),'p95',np.percentile(cv,95),'max',cv.max())
print('frac<0.01',np.mean(cv<0.01),'frac<1',np.mean(cv<1),'pct one desk',np.mean(cv<s[-1]))
print('A with correct night count (2499 steps): median',np.median(cv_2499),'mean',cv_2499.mean(),'frac<0.01',np.mean(cv_2499<0.01))
print('story sigma2',(0.24/2.8)**2,'percentile in A run',np.mean(cv<(0.24/2.8)**2))
print('top10 share',np.sort(cv)[-10:].sum()/cv.sum())
np.save('A_cv.npy',cv)
