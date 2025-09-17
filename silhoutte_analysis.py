# ------------------------------------------------------------
# 0. Imports
# ------------------------------------------------------------
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples

# ------------------------------------------------------------
# 1. Create data: 2 tight blobs + 1 loose blob
# ------------------------------------------------------------
X, y = make_blobs(n_samples=[120, 120, 40], centers=[[0, 0], [3, 3], [1, 5]],
                  cluster_std=[0.3, 0.3, 1.0], random_state=42)

# ------------------------------------------------------------
# 2. Try k = 2 … 6 and pick the one with highest silhouette score
# ------------------------------------------------------------
scores = {}
for k in range(2, 7):
    km = KMeans(n_clusters=k, random_state=42, n_init='auto')
    labels = km.fit_predict(X)
    scores[k] = silhouette_score(X, labels)

best_k = max(scores, key=scores.get)
print(f'Silhouette scores: {scores}')
print(f'Best k = {best_k} (score = {scores[best_k]:.3f})')

# ------------------------------------------------------------
# 3. Silhouette diagram for the best k
# ------------------------------------------------------------
km_best = KMeans(n_clusters=best_k, random_state=42, n_init='auto')
labels = km_best.fit_predict(X)
sil_vals = silhouette_samples(X, labels)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
# left: silhouette plot
y_lower = 0
cmap = plt.cm.get_cmap('tab10', best_k)
for i in range(best_k):
    ith_cluster_sil = sil_vals[labels == i]
    ith_cluster_sil.sort()
    size_cluster_i = ith_cluster_sil.shape[0]
    y_upper = y_lower + size_cluster_i
    color = cmap(i)
    ax1.fill_betweenx(np.arange(y_lower, y_upper),
                      0, ith_cluster_sil,
                      facecolor=color, edgecolor=color, alpha=0.7)
    ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
    y_lower = y_upper
ax1.axvline(x=sil_vals.mean(), color="red", linestyle="--", label='mean silhouette')
ax1.set_xlabel('Silhouette coefficient')
ax1.set_ylabel('Cluster')
ax1.legend()

# right: actual clusters
ax2.scatter(X[:, 0], X[:, 1], c=labels, cmap=cmap, s=30, edgecolor='k')
ax2.scatter(km_best.cluster_centers_[:, 0], km_best.cluster_centers_[:, 1],
            marker='x', c='black', s=100, label='centroids')
ax2.set_title(f'Clusters (k={best_k})')
ax2.legend()
plt.tight_layout()
plt.show()
