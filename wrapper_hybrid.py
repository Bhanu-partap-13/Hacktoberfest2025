#LDA + recursive (filter LDA scores, then RFE)
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
lda = LinearDiscriminantAnalysis()
lda.fit(X, y)
idx = np.argsort(np.abs(lda.coef_[0]))[-10:]   # top-10 LDA weights
X_lda = X.iloc[:, idx]
# now run any wrapper on X_lda
rfe = RFE(LogisticRegression(), 5).fit(X_lda, y)
X_new = X_lda.iloc[:, rfe.support_]

#Shuffle-forward (stability selection) – quick DIY
from sklearn.linear_model import LassoCV
import pandas as pd, numpy as np
counts = np.zeros(X.shape[1])
for run in range(100):
    lasso = LassoCV(cv=3, random_state=run).fit(X.sample(frac=0.7, random_state=run), y)
    counts += (np.abs(lasso.coef_) > 0)
selected = counts >= 80   # kept in ≥80 % of bootstrap draws
X_new = X.loc[:, selected]

"""Filter – ultra-fast, baseline, huge data, but ignores feature interactions.
Wrapper – finds interaction-friendly subsets, costlier, risk of over-fit → always cross-validate.
Exhaustive – gold standard for p < 20; NP-hard beyond that.
RFE / SFS / SBS – good middle ground; rule of thumb: keep ≤ 1 000 features.
Hybrid – filter first (throw away 90 % junk), then wrapper on remainder → speed + quality."""
