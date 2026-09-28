import pandas as pd, numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold, train_test_split, cross_validate
from sklearn.inspection import permutation_importance
ads = pd.read_csv('../../data/Social_Network_Ads.csv')
ads['Gender'] = (ads.Gender == 'Male').astype(int)
Xa = ads[['Gender','Age','EstimatedSalary']]; ya = ads['Purchased']
cv5 = StratifiedKFold(5, shuffle=True, random_state=7034)
cv = lambda m: cross_val_score(m, Xa, ya, cv=cv5).mean()
print('baseline', (ya==0).mean())
print('nunique', Xa.nunique().to_dict())
print('unpruned', cv(DecisionTreeClassifier(random_state=7034)))
for k in [2,3,4,6,8,12]:
    r = cross_validate(DecisionTreeClassifier(max_leaf_nodes=k, random_state=7034), Xa, ya, cv=cv5, return_train_score=True)
    print(k, r['test_score'].mean(), r['train_score'].mean(), np.rint(r['test_score']*80).astype(int))
t = DecisionTreeClassifier(max_leaf_nodes=3, random_state=7034).fit(Xa, ya)
from sklearn.tree import export_text; print(export_text(t, feature_names=list(Xa.columns)))
t4 = DecisionTreeClassifier(max_leaf_nodes=4, random_state=7034).fit(Xa, ya)
print(export_text(t4, feature_names=list(Xa.columns), show_weights=True))
rf = RandomForestClassifier(n_estimators=300, random_state=7034); gb = GradientBoostingClassifier(n_estimators=100, random_state=7034)
for m in [rf, gb]:
    r = cross_validate(m, Xa, ya, cv=cv5, return_train_score=True)
    print(type(m).__name__, r['test_score'].mean(), r['train_score'].mean(), np.rint(r['test_score']*80).astype(int))
Xtr, Xte, ytr, yte = train_test_split(Xa, ya, test_size=0.3, random_state=7034, stratify=ya)
rf14 = RandomForestClassifier(n_estimators=300, random_state=7034).fit(Xtr, ytr)
print('MDI', dict(zip(Xa.columns, rf14.feature_importances_.round(4))))
for nr in [10, 50]:
    p = permutation_importance(rf14, Xte, yte, n_repeats=nr, random_state=7034)
    print('perm', nr, p.importances_mean.round(4), p.importances_std.round(4))
p = permutation_importance(rf14, Xtr, ytr, n_repeats=50, random_state=7034)
print('perm train', p.importances_mean.round(4))
print('rf14 train/test acc', rf14.score(Xtr,ytr), rf14.score(Xte,yte))
print('nunique train', Xtr.nunique().to_dict())
# Gender signal
print(ads.groupby('Gender').Purchased.mean())
# fraction of splits in RF where Gender only candidate -> 1/3 of nodes pick Gender as sole candidate? check actual Gender split count
splits = [sum(int((e.tree_.feature==j).sum()) for e in rf14.estimators_) for j in range(3)]
print('splits', splits)
# RF with max_features=None
print('rf maxfeat None', cv(RandomForestClassifier(n_estimators=300, random_state=7034, max_features=None)))
print('rf min_samples_leaf 10', cv(RandomForestClassifier(n_estimators=300, random_state=7034, min_samples_leaf=10)))
for n in [10, 20, 50, 100]:
    print('gb', n, cv(GradientBoostingClassifier(n_estimators=n, random_state=7034)))
print('gb depth1', cv(GradientBoostingClassifier(n_estimators=100, max_depth=1, random_state=7034)))
