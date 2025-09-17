# ------------------------------------------------------------
# 0. Imports
# ------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

# ------------------------------------------------------------
# 1. Create dumbbell + noise: two dense blobs + random junk
# ------------------------------------------------------------
X1, y1 = make_blobs(n_samples=300, centers=[[0, 0]], cluster_std=0.2,
                    random_state=42)
X2, y2 = make_blobs(n_samples=300, centers=[[3, 3]], cluster_std=0.2,
                    random_state=42)
noise = np.random.uniform(-1, 4, size=(50, 2))
X = np.vstack([X1, X2, noise])

# ------------------------------------------------------------
# 2. Standardise (DBSCAN is distance-based)
# ------------------------------------------------------------
X = StandardScaler().fit_transform(X)

# ------------------------------------------------------------
# 3. Fit DBSCAN
# ------------------------------------------------------------
db = DBSCAN(eps=0.3, min_samples=5)
labels = db.fit_predict(X)

# ------------------------------------------------------------
# 4. Results
# ------------------------------------------------------------
n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
n_noise    = list(labels).count(-1)
print(f'DBSCAN found {n_clusters} clusters and {n_noise} noise points')

# ------------------------------------------------------------
# 5. Plot
# ------------------------------------------------------------
unique_labels = set(labels)
colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]
for k, col in zip(unique_labels, colors):
    if k == -1:            # noise
        col = [0, 0, 0, 1]
    class_member_mask = (labels == k)
    xy = X[class_member_mask]
    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
             markeredgecolor='k', markersize=6 if k != -1 else 10,
             marker='x' if k == -1 else 'o')

plt.title('DBSCAN: two clusters + noise (black crosses)')
plt.show()
