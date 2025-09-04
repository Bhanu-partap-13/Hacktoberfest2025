import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# 1. Toy 2-D data (elongated Gaussian blob)
rng = np.random.RandomState(42)
X = rng.multivariate_normal([0, 0], [[3, 2], [2, 2]], size=200)

# 2. Standardize (mean=0, var=1) → PCA
X_std = StandardScaler().fit_transform(X)
pca = PCA(n_components=1)
X_pca = pca.fit_transform(X_std)

# 3. Plot original points + principal component line
plt.scatter(X_std[:, 0], X_std[:, 1], c='steelblue', s=20)
# First PC direction
line = pca.components_[0] * np.linspace(-3, 3, 100)[:, None]
plt.plot(line[:, 0], line[:, 1], 'r', linewidth=3, label='1st PC')
plt.title("Hello-World PCA")
plt.xlabel("x0 (standardized)")
plt.ylabel("x1 (standardized)")
plt.legend()
plt.axis('equal')
plt.show()

# 3. Plot original points + principal component line
plt.scatter(X[:, 0], X[:, 1], c='steelblue', s=20)
# First PC direction
line = pca.components_[0] * np.linspace(-3, 3, 100)[:, None]
plt.plot(line[:, 0], line[:, 1], 'r', linewidth=3, label='1st PC')
plt.title("Hello-World PCA")
plt.xlabel("x0 (standardized)")
plt.ylabel("x1 (standardized)")
plt.legend()
plt.axis('equal')
plt.show()
