#Sequential Forward Selection (SFS)
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from sklearn.linear_model import LogisticRegression
sfs = SFS(LogisticRegression(max_iter=1000), k_features=5,
          forward=True, floating=False, scoring='accuracy', cv=5)
sfs.fit(X, y)
X_new = X.iloc[:, sfs.k_feature_idx_]

#Sequential Backward Selection (SBS)
sbs = SFS(LogisticRegression(max_iter=1000), k_features=5,
          forward=False, floating=False, scoring='accuracy', cv=5)
sbs.fit(X, y)
X_new = X.iloc[:, sbs.k_feature_idx_]

#Exhaustive (best subset) – small p only (p ≤ 15-20)
from mlxtend.feature_selection import ExhaustiveFeatureSelector as EFS
efs = EFS(LogisticRegression(max_iter=1000), min_features=3,
          max_features=5, scoring='accuracy', cv=3)
efs.fit(X, y)
X_new = X.iloc[:, efs.best_idx_]

#Recursive Feature Elimination (RFE) – “recursive backward”
from sklearn.feature_selection import RFE
rfe = RFE(LogisticRegression(max_iter=1000), n_features_to_select=5)
X_new = rfe.fit_transform(X, y)
