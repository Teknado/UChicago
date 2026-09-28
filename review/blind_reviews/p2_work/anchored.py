import numpy as np
D,n,NIGHTS=1000,500,2500
def run(real, rng, init_from_real=True):
    s2 = real.var(axis=1,ddof=1) if init_from_real else np.ones(real.shape[0])
    for k in range(1,NIGHTS):
        scen=rng.standard_normal(real.shape)*np.sqrt(s2)[:,None]
        s2=np.concatenate([real,scen],axis=1).var(axis=1,ddof=1)
    return s2
def summ(lbl,s2,real):
    print(f'{lbl:55s} median {np.median(s2):.4f} p5 {np.percentile(s2,5):.4f} p95 {np.percentile(s2,95):.4f} mean {s2.mean():.4f} | corr(s2, real var) {np.corrcoef(s2,real.var(axis=1,ddof=1))[0,1]:.2f}')
rng0=np.random.default_rng(777); Z=rng0.standard_normal((D,n))
Zr=Z/Z.std(axis=1,ddof=1,keepdims=True)
summ('rescaled real days (var exactly 1)',run(Zr,np.random.default_rng(1)),Zr)
summ('raw N(0,1) real days, night-1 fit = var(real)',run(Z,np.random.default_rng(1)),Z)
summ('raw real days, night-1 forced to 1 (A-style)',run(Z,np.random.default_rng(1),False),Z)
print('spread of real-day sample variance across desks: p5,p95',np.percentile(Z.var(axis=1,ddof=1),[5,95]))
# exact A code
np.random.seed(2694)
lib=np.empty((D,1000)); lib[:,:500]=np.random.normal(0,1,(D,500)); cv=np.ones(D)
for t in range(NIGHTS):
    lib[:,500:]=np.random.normal(0,np.sqrt(cv)[:,None],(D,500)); cv=np.var(lib,axis=1,ddof=1)
print('A exact repro: median',np.median(cv),'p5',np.percentile(cv,5),'p95',np.percentile(cv,95),'mean',cv.mean(),
      'corr with real var',np.corrcoef(cv,lib[:,:500].var(axis=1,ddof=1))[0,1])
