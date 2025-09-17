#Filter – univariate ANOVA F (regression)
from sklearn.feature_selection import SelectKBest, f_regression
X_new = SelectKBest(f_regression, k=10).fit_transform(X, y)

#Filter – χ² (categorical targets, non-negative X)
from sklearn.feature_selection import chi2, SelectKBest
X_new = SelectKBest(chi2, k=8).fit_transform(abs(X), y)

#Filter – Pearson correlation (keep |r| > 0.3)
corr = X.corrwith(y).abs().sort_values(ascending=False)
X_new = X[corr[corr > 0.3].index]
