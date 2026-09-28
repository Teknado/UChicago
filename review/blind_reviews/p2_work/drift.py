import numpy as np
from scipy.special import digamma, polygamma
SEED=2694
# A's design: one step, 5000 desks -> SE
np.random.seed(SEED)
sv=np.var(np.random.normal(0,1,(100000,500)),axis=1,ddof=1)
print('A verif mean',sv.mean())
for n in [50,500,5000]:
    Z=np.random.normal(0,1,(5000,n)); lv=np.log(np.var(Z,axis=1,ddof=1))
    k=n-1; ex=digamma(k/2)-np.log(k/2)
    print(f'A design n={n}: mean {lv.mean():+.6f}  SE {lv.std(ddof=1)/np.sqrt(5000):.6f}  exact {ex:+.6f}  z vs exact {(lv.mean()-ex)/(lv.std(ddof=1)/np.sqrt(5000)):+.2f}  rel SE {lv.std(ddof=1)/np.sqrt(5000)/abs(ex):.0%}')
# proper design: 300 desks x 300 nights, independent seed
rng=np.random.default_rng(12345)
for n in [50,500,5000]:
    D,T=300,300; s2=np.ones(D); L0=np.zeros(D)
    for t in range(T):
        s2=(rng.standard_normal((D,n))*np.sqrt(s2)[:,None]).var(axis=1,ddof=1)
    pd_=np.log(s2)/T
    k=n-1; ex=digamma(k/2)-np.log(k/2)
    print(f'300x300 n={n}: mean {pd_.mean():+.6f} SE {pd_.std(ddof=1)/np.sqrt(D):.6f} exact {ex:+.6f} n*mean {n*pd_.mean():.3f}')
