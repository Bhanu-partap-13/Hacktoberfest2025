# ------------------------------------------------------------
# 0. Imports
# ------------------------------------------------------------
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import accuracy_score, classification_report

# ------------------------------------------------------------
# 1. Data
# ------------------------------------------------------------
X, y = load_iris(return_X_y=True)

# ------------------------------------------------------------
# 2. Model pipeline
# ------------------------------------------------------------
rf = RandomForestClassifier(random_state=42)

# ------------------------------------------------------------
# 3. Hyper-parameter space
# ------------------------------------------------------------
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 3, 5, 10],
    'min_samples_split': [2, 4, 6],
    'max_features': ['sqrt', 'log2']
}
# total combos for GridSearch: 3 × 4 × 3 × 2 = 72 fits

param_dist = {  # same keys, but distributions for RandomizedSearchCV
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 3, 5, 10],
    'min_samples_split': [2, 4, 6],
    'max_features': ['sqrt', 'log2']
}

# ------------------------------------------------------------
# 4. GridSearchCV  (exhaustive)
# ------------------------------------------------------------
print('=== GridSearchCV (exhaustive) ===')
grid = GridSearchCV(rf, param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=1)
grid.fit(X, y)
print("Best params :", grid.best_params_)
print("Best CV acc : {:.3f}".format(grid.best_score_))
print("Test acc    : {:.3f}".format(accuracy_score(y, grid.predict(X))))

# ------------------------------------------------------------
# 5. RandomizedSearchCV  (random, budget = 20 draws)
# ------------------------------------------------------------
print('\n=== RandomizedSearchCV (20 draws) ===')
rand = RandomizedSearchCV(rf, param_dist, n_iter=20, cv=5, scoring='accuracy',
                          random_state=42, n_jobs=-1, verbose=1)
rand.fit(X, y)
print("Best params :", rand.best_params_)
print("Best CV acc : {:.3f}".format(rand.best_score_))
print("Test acc    : {:.3f}".format(accuracy_score(y, rand.predict(X))))

# ------------------------------------------------------------
# 6. Quick comparison table
# ------------------------------------------------------------
comparison = pd.DataFrame({
    'Strategy': ['GridSearch', 'RandomSearch'],
    'Fits': [grid.cv_results_['params'].__len__(), rand.cv_results_['params'].__len__()],
    'Best CV Acc': [grid.best_score_, rand.best_score_],
    'Best Params': [str(grid.best_params_), str(rand.best_params_)]
})
print('\n=== Side-by-side ===')
print(comparison.to_string(index=False))
