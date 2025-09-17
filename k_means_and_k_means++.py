# ------------------------------------------------------------
# 0. Imports
# ------------------------------------------------------------
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans

# ------------------------------------------------------------
# 1. 2-D toy data (three blobs)
# ------------------------------------------------------------
X, y = make_blobs(n_samples=450, centers=3, cluster_std=0.70,
                  random_state=42, shuffle=True)

# ------------------------------------------------------------
# 2. K-means with random init  VS  K-means++ (smart init)
# ------------------------------------------------------------
init_methods = {'classic (random)': 'random', 'k-means++': 'k-means++'}
results = {}

for name, init in init_methods.items():
    km = KMeans(n_clusters=3, init=init, n_init=10, random_state=42)
    km.fit(X)
    results[name] = {'inertia': km.inertia_,
                     'n_iter': km.n_iter_}
    print(f"{name:15} | inertia: {km.inertia_:.2f} | iters: {km.n_iter_}")

# ------------------------------------------------------------
# 3. Visual comparison
# ------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
for ax, (name, init) in zip(axes, init_methods.items()):
    km = KMeans(n_clusters=3, init=init, n_init=10, random_state=42)
    km.fit(X)
    ax.scatter(X[:, 0], X[:, 1], c=km.labels_, cmap='viridis', s=30, alpha=0.8)
    ax.scatter(km.cluster_centers_[:, 0], km.cluster_centers_[:, 1],
               c='red', marker='x', s=100, label='centroids')
    ax.set_title(f'{name} init')
    ax.legend()
plt.tight_layout()
plt.show()
