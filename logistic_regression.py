import numpy as np, matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification

# 1. 2-D toy data: two blobs, 200 points
X, y = make_classification(n_samples=200, n_features=2,
                           n_redundant=0, n_informative=2,
                           n_clusters_per_class=1, random_state=42)

# 2. Fit logistic regression
clf = LogisticRegression().fit(X, y)

# 3. Plot data + decision boundary
xx = np.linspace(X[:,0].min()-1, X[:,0].max()+1, 200)
yy = -(clf.intercept_ + clf.coef_[0,0]*xx) / clf.coef_[0,1]
plt.scatter(X[:,0], X[:,1], c=y, cmap='bwr', s=30)
plt.plot(xx, yy, 'k--', label='decision boundary')
plt.title("Hello-World Logistic Regression"); plt.legend(); plt.show()
