import pandas as pd, numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
ads = pd.read_csv('../../data/Social_Network_Ads.csv'); ads['Gender']=(ads.Gender=='Male').astype(int)
Xa=ads[['Gender','Age','EstimatedSalary']]; ya=ads.Purchased
rf=RandomForestClassifier(n_estimators=300, random_state=7034).fit(Xa,ya)
c=np.zeros(3)
for e in rf.estimators_:
    f=e.tree_.feature; f=f[f>=0]; c+=np.bincount(f,minlength=3)
print('split share (G,A,S):', (c/c.sum()).round(3), int(c.sum()))
cv5=StratifiedKFold(5,shuffle=True,random_state=7034)
for mf in [1,2,3]:
    print('max_features',mf, cross_val_score(RandomForestClassifier(n_estimators=300,max_features=mf,random_state=7034),Xa,ya,cv=cv5).mean())
X2=Xa[['Age','EstimatedSalary']]
print('RF without Gender', cross_val_score(RandomForestClassifier(n_estimators=300,random_state=7034),X2,ya,cv=cv5).mean())
for msl in [5,10,20]:
    print('msl',msl, cross_val_score(RandomForestClassifier(n_estimators=300,min_samples_leaf=msl,random_state=7034),Xa,ya,cv=cv5).mean())
