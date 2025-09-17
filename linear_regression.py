import numpy as np
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

# 1. Toy data ----------------------------------------------------------
X, y, true_coef = make_regression(n_samples=100, n_features=10,
                                  n_informative=3, noise=3.0,
                                  coef=True, random_state=42)

# 2. Build models (all standardized) -----------------------------------
models = {
    'Linear': LinearRegression(),
    'Lasso' : Lasso(alpha=1.0),
    'Ridge' : Ridge(alpha=1.0),
    'ElasticNet': ElasticNet(alpha=1.0, l1_ratio=0.5)  # 50 % L1, 50 % L2
}

# 3. Fit & display -----------------------------------------------------
print("True non-zero coef:", [round(c, 2) for c in true_coef if c != 0])
print("-" * 50)
for name, model in models.items():
    pipe = make_pipeline(StandardScaler(), model)
    pipe.fit(X, y)
    coef = pipe.named_steps[name.lower()].coef_
    print(f"{name:12} coef:", [round(c, 2) for c in coef])
