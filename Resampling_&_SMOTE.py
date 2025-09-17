# ------------------------------------------------------------
# 0. Imports
# ------------------------------------------------------------
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from imblearn.over_sampling import SMOTE, RandomOverSampler, ADASYN
from imblearn.under_sampling import RandomUnderSampler, TomekLinks
from imblearn.combine import SMOTETomek

# ------------------------------------------------------------
# 1. Create an imbalanced data set (5 % minority class)
# ------------------------------------------------------------
X, y = make_classification(n_samples=5000, n_features=20,
                           n_informative=10, n_redundant=10,
                           weights=[0.95, 0.05], random_state=42)
print('Class distribution:', np.bincount(y))

# ------------------------------------------------------------
# 2. Baseline (no resampling)
# ------------------------------------------------------------
rf = RandomForestClassifier(n_estimators=300, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y,
                                                    test_size=0.25,
                                                    random_state=42)

def score_model(name, sampler=None):
    """Fit model with/without sampler and return test ROC-AUC."""
    if sampler is None:               # baseline
        X_res, y_res = X_train, y_train
    else:
        X_res, y_res = sampler.fit_resample(X_train, y_train)
    rf.fit(X_res, y_res)
    preds = rf.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, preds)
    print(f"{name:20} | train shape {np.bincount(y_res)} | ROC-AUC: {auc:.3f}")
    return auc

# ------------------------------------------------------------
# 3. Resampling techniques
# ------------------------------------------------------------
results = {'Baseline': score_model('Baseline')}

# ---- oversampling ----
results['RandomOverSampler'] = score_model('RandomOverSampler',
                                           RandomOverSampler(random_state=42))
results['SMOTE'] = score_model('SMOTE',
                               SMOTE(random_state=42))
results['ADASYN'] = score_model('ADASYN',
                                ADASYN(random_state=42))

# ---- undersampling ----
results['RandomUnderSampler'] = score_model('RandomUnderSampler',
                                            RandomUnderSampler(random_state=42))
results['TomekLinks'] = score_model('TomekLinks',
                                    TomekLinks())

# ---- hybrid ----
results['SMOTE + Tomek'] = score_model('SMOTE + Tomek',
                                       SMOTETomek(random_state=42))

# ------------------------------------------------------------
# 4. Summary table
# ------------------------------------------------------------
summary = pd.Series(results).sort_values(ascending=False)
print('\n=== Summary (ROC-AUC) ===')
print(summary.round(3))
