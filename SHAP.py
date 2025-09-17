# ------------------------------------------------------------
# 0. Imports
# ------------------------------------------------------------
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import shap

# ------------------------------------------------------------
# 1. Toy SENSOR data with built-in correlations
# ------------------------------------------------------------
np.random.seed(42)
n = 200
# latent drivers
t = np.linspace(0, 4*np.pi, n)
s1 = np.sin(t) + np.random.normal(0, 0.1, n)          # sensor 1
s2 = s1*0.8 + np.random.normal(0, 0.1, n)             # corr ~0.8
s3 = np.cos(t) + np.random.normal(0, 0.1, n)          # sensor 3
s4 = np.random.normal(0, 1, n)                          # independent
s5 = s3*0.7 + np.random.normal(0, 0.15, n)            # corr ~0.7

sensors = pd.DataFrame({'s1': s1, 's2': s2, 's3': s3, 's4': s4, 's5': s5})
target  = 3*s1 + 2*s3 + s4 + np.random.normal(0, 0.2, n)  # y depends on sensors

print("Correlation matrix")
print(sensors.corr().round(2))

# ------------------------------------------------------------
# 2. Train / test split & model
# ------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    sensors, target, test_size=0.3, random_state=42)

model = GradientBoostingRegressor(n_estimators=300, max_depth=3, random_state=42)
model.fit(X_train, y_train)
print("R² on test:", model.score(X_test, y_test).round(3))

# ------------------------------------------------------------
# 3. SHAP explanation
# ------------------------------------------------------------
explainer = shap.Explainer(model)
shap_values = explainer(X_test)          # shap_values.values shape = (n_test, n_features)

# 3a. Summary bar plot (mean absolute SHAP)
shap.plots.bar(shap_values)

# 3b. Beeswarm (feature value vs SHAP magnitude)
shap.plots.beeswarm(shap_values)

# 3c. Dependency scatter for the most correlated pair (s1 vs s2)
corr_pair = ['s1', 's2']                # highest Pearson r
shap.plots.scatter(shap_values[:, 's1'], color=shap_values)   # color by s1’s own SHAP
# optional: overlay the correlated feature on the color axis
shap.plots.scatter(shap_values[:, 's1'], color=X_test['s2'])
