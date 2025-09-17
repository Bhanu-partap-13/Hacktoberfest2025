# ------------------------------------------------------------
# 0. Imports
# ------------------------------------------------------------
from sklearn.datasets import load_iris
from sklearn.ensemble import (BaggingClassifier, RandomForestClassifier,
                              AdaBoostClassifier, GradientBoostingClassifier)
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# ------------------------------------------------------------
# 1. Data
# ------------------------------------------------------------
iris = load_iris()
X, y = iris.data, iris.target
feat_names = iris.feature_names
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y)

# ------------------------------------------------------------
# 2. Instantiate ensembles
# ------------------------------------------------------------
models = {
    'Bagging': BaggingClassifier(
        estimator=DecisionTreeClassifier(), n_estimators=100, random_state=42),
    'RandomForest': RandomForestClassifier(
        n_estimators=100, random_state=42),
    'AdaBoost': AdaBoostClassifier(
        estimator=DecisionTreeClassifier(max_depth=1), n_estimators=100, random_state=42),
    'GradientBoost': GradientBoostingClassifier(
        n_estimators=100, learning_rate=0.1, max_depth=1, random_state=42)
}

# ------------------------------------------------------------
# 3. Fit & evaluate
# ------------------------------------------------------------
scores, importances = {}, {}
for name, model in models.items():
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    scores[name] = accuracy_score(y_test, pred)
    # importances: Bagging & RF use .feature_importances_
    if hasattr(model, 'feature_importances_'):
        importances[name] = model.feature_importances_
    else:  # AdaBoost
        importances[name] = model.estimator_.feature_importances_

print("Hold-out accuracies:")
for k, v in scores.items():
    print(f"{k:15}: {v:.3f}")

# ------------------------------------------------------------
# 4. Feature-importance comparison plot
# ------------------------------------------------------------
imp_df = pd.DataFrame(importances, index=feat_names)
imp_df.plot(kind='bar', figsize=(8, 4))
plt.title('Feature importance across ensembles')
plt.ylabel('Importance')
plt.tight_layout()
plt.show()
