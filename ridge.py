import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

# 1. Tiny data set: 1 feature, 10 samples, y = 3x + noise
X = np.linspace(-2, 2, 10).reshape(-1, 1)
y = 3 * X.ravel() + np.random.randn(10) * 0.2

# 2. Pipeline: Standardize → Ridge (α = 1.0)
model = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
model.fit(X, y)

# 3. Print learned slope and intercept
ridge = model.named_steps['ridge']
print(f"Ridge slope: {ridge.coef_[0]:.3f}, intercept: {ridge.intercept_:.3f}")
