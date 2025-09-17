# ------------------------------------------------------------
# 0. Imports
# ------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error

# ------------------------------------------------------------
# 1. Data
# ------------------------------------------------------------
X, y = fetch_california_housing(return_X_y=True, as_frame=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

# ------------------------------------------------------------
# 2. Train three SVR models
# ------------------------------------------------------------
models = {
    'Linear': SVR(kernel='linear', C=1.0, epsilon=0.1),
    'RBF':    SVR(kernel='rbf',    gamma='scale', C=1.0, epsilon=0.1),
    'Poly-2': SVR(kernel='poly',   degree=2, gamma='scale', C=1.0, epsilon=0.1)
}

for name, model in models.items():
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, pred))
    print(f"{name:8} RMSE: {rmse:.3f}")

# ------------------------------------------------------------
# 3. Scatter plot: predicted vs actual (best model here = RBF)
# ------------------------------------------------------------
best = models['RBF']
pred_best = best.predict(X_test)
plt.figure(figsize=(5, 5))
plt.scatter(y_test, pred_best, alpha=0.3)
plt.plot([y_test.min(), y_test.max()],
         [y_test.min(), y_test.max()], 'r--')
plt.xlabel('Actual house price (100k $)')
plt.ylabel('Predicted house price (100k $)')
plt.title('SVR (RBF) – predicted vs actual')
plt.tight_layout()
plt.show()
