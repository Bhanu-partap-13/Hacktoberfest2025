# ------------------------------------------------------------
# 0. Imports
# ------------------------------------------------------------
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (
    KFold, StratifiedKFold, ShuffleSplit, StratifiedShuffleSplit,
    GroupKFold, TimeSeriesSplit, cross_val_score, cross_validate
)
from sklearn.metrics import accuracy_score, f1_score

# ------------------------------------------------------------
# 1. Data
# ------------------------------------------------------------
X, y = load_iris(return_X_y=True)
df = pd.DataFrame(X, columns=[f'f{i}' for i in range(X.shape[1])])
df['target'] = y

# quick model
clf = LogisticRegression(max_iter=1000, random_state=42)

# helper: pretty print CV results
def cv_summary(name, cv, X, y, groups=None):
    scores = cross_validate(clf, X, y, cv=cv, groups=groups,
                            scoring=['accuracy', 'f1_macro'],
                            return_train_score=True)
    print(f"{name:20} | "
          f"acc (test) : {scores['test_accuracy'].mean():.3f} ± {scores['test_accuracy'].std():.3f} | "
          f"f1 (test)  : {scores['test_f1_macro'].mean():.3f} ± {scores['test_f1_macro'].std():.3f}")

# ------------------------------------------------------------
# 2. Classic K-Fold  (k=5)
# ------------------------------------------------------------
print('\n=== K-Fold flavours ===')
cv_kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_summary('KFold (5)', cv_kf, X, y)

# ------------------------------------------------------------
# 3. Stratified K-Fold  (preserves class ratio)
# ------------------------------------------------------------
cv_skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_summary('StratifiedKFold', cv_skf, X, y)

# ------------------------------------------------------------
# 4. Repeated / ShuffleSplit  (Monte-Carlo, no fixed k)
# ------------------------------------------------------------
cv_ss = ShuffleSplit(n_splits=5, test_size=0.2, random_state=42)
cv_summary('ShuffleSplit', cv_ss, X, y)

cv_sstrat = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=42)
cv_summary('StratifiedShuffle', cv_sstrat, X, y)

# ------------------------------------------------------------
# 5. Group-aware  (items from same group stay together)
# ------------------------------------------------------------
groups = np.random.randint(0, 10, size=len(y))  # 10 random groups
cv_gkf = GroupKFold(n_splits=5)
cv_summary('GroupKFold', cv_gkf, X, y, groups=groups)

# ------------------------------------------------------------
# 6. Time-Series Split  (train window always before test)
# ------------------------------------------------------------
cv_ts = TimeSeriesSplit(n_splits=5)
cv_summary('TimeSeriesSplit', cv_ts, X, y)

# ------------------------------------------------------------
# 7. Manual split (loop) – same as KFold but you control everything
# ------------------------------------------------------------
print('\n=== Manual K-Fold loop ===')
kf = KFold(n_splits=5, shuffle=True, random_state=42)
manual_scores = []
for fold, (train_idx, test_idx) in enumerate(kf.split(X, y), 1):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    clf.fit(X_train, y_train)
    acc = accuracy_score(y_test, clf.predict(X_test))
    manual_scores.append(acc)
    print(f'Fold {fold}: accuracy = {acc:.3f}')
print(f'Manual avg : {np.mean(manual_scores):.3f} ± {np.std(manual_scores):.3f}')
