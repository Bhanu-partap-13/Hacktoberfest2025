# ------------------------------------------------------------
# 0. Imports
# ------------------------------------------------------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

# ------------------------------------------------------------
# 1. Data (wine, but we ignore the labels → unsupervised)
# ------------------------------------------------------------
wine = load_wine()
X, y = wine.data, wine.target          # y only used for colouring plots
feat_names = wine.feature_names

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ------------------------------------------------------------
# 2. Fit Isolation Forest
# ------------------------------------------------------------
iso = IsolationForest(n_estimators=200,
                      max_samples='auto',
                      contamination=0.05,   # expect 5 % outliers
                      random_state=42)
iso.fit(X_scaled)

# anomaly score (lower = more abnormal)
scores = iso.decision_function(X_scaled)
# -1 = outlier, 1 = inlier
pred = iso.predict(X_scaled)

# ------------------------------------------------------------
# 3. Top 5 anomalies
# ------------------------------------------------------------
top_idx = np.argsort(scores)[:5]   # lowest scores
ano_df = pd.DataFrame(X[top_idx], columns=feat_names)
ano_df['score'] = scores[top_idx]
print("Top 5 anomalies (raw features + score):")
print(ano_df.round(2))

# ------------------------------------------------------------
# 4. Visualise in 2-D (first two features)
# ------------------------------------------------------------
plt.figure(figsize=(6, 4))
plt.scatter(X[:, 0], X[:, 1], c=pred, cmap='coolwarm', s=40, edgecolors='k')
plt.scatter(X[top_idx, 0], X[top_idx, 1],
            c='red', s=100, marker='x', label='Top anomalies')
plt.xlabel(feat_names[0])
plt.ylabel(feat_names[1])
plt.title('Isolation Forest – anomalies in red')
plt.legend()
plt.tight_layout()
plt.show()
