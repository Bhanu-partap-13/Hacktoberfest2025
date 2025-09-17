# ------------------------------------------------------------
# 0. Imports
# ------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import AgglomerativeClustering, BisectingKMeans
from sklearn.metrics import adjusted_rand_score
from scipy.cluster.hierarchy import dendrogram, linkage

# ------------------------------------------------------------
# 1. 2-D toy data (3 blobs)
# ------------------------------------------------------------
X, y = make_blobs(n_samples=150, centers=3, cluster_std=0.60,
                  random_state=42, shuffle=True)

# ------------------------------------------------------------
# 2. AGGLOMERATIVE  (bottom-up, ward linkage)
# ------------------------------------------------------------
agg = AgglomerativeClustering(n_clusters=3, linkage='ward')
agg_labels = agg.fit_predict(X)

# dendrogram needs explicit linkage matrix
Z = linkage(X, method='ward')
plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1)
dendrogram(Z, truncate_mode='level', p=3)
plt.title('Agglomerative Dendrogram')

# ------------------------------------------------------------
# 3. DIVISIVE  (top-down, implemented as repeated 2-means splits)
#    sklearn ≥1.2 offers BisectingKMeans – exact same idea.
# ------------------------------------------------------------
div = BisectingKMeans(n_clusters=3, random_state=42)
div_labels = div.fit_predict(X)

# ------------------------------------------------------------
# 4. Visual comparison
# ------------------------------------------------------------
def plot_clusters(ax, labels, title):
    ax.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', s=30, edgecolor='k')
    ax.set_title(title)

plot_clusters(plt.subplot(1, 3, 2), agg_labels, 'Agglomerative (ward)')
plot_clusters(plt.subplot(1, 3, 3), div_labels, 'Divisive (bisecting K-means)')

# ------------------------------------------------------------
# 5. Quality vs ground-truth
# ------------------------------------------------------------
print("Adjusted Rand Index")
print("Agglomerative:", adjusted_rand_score(y, agg_labels))
print("Divisive     :", adjusted_rand_score(y, div_labels))

plt.tight_layout()
plt.show()
