import numpy as np
from sklearn.linear_model import Lasso
from sklearn.datasets import make_regression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

# 1. Toy data: 100 samples, 10 features, only 3 truly matter
X, y, true_coef = make_regression(
    n_samples=100, n_features=10, n_informative=3,
    noise=0.5, coef=True, random_state=42)

# 2. Pipeline: Standardize → LASSO with mild regularization
model = make_pipeline(StandardScaler(), Lasso(alpha=0.1))
model.fit(X, y)

# 3. Show the learned coefficients (many ≈ 0)
print("True coefficients (non-zero only):", true_coef[:3])
print("LASSO coefficients:", model.named_steps['lasso'].coef_)
