# ------------------------------------------------------------
# 0. Imports
# ------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.linear_model import Perceptron
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# ------------------------------------------------------------
# 1. Data (2-D for easy plotting)
# ------------------------------------------------------------
iris = load_iris()
X, y = iris.data[:, :2], iris.target          # sepal length & width only

# Perceptron needs 2 classes → use setosa (0) vs versicolor (1)
mask = y < 2
X_bin, y_bin = X[mask], y[mask]

# MLP will use all 3 classes
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,
                                                    random_state=42, stratify=y)
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

# same scaling for binary set
Xb_train, Xb_test, yb_train, yb_test = train_test_split(X_bin, y_bin,
                                                        test_size=0.3,
                                                        random_state=42,
                                                        stratify=y_bin)
Xb_train_s = scaler.fit_transform(Xb_train)
Xb_test_s  = scaler.transform(Xb_test)

# ------------------------------------------------------------
# 2. Perceptron (single layer, no hidden units)
# ------------------------------------------------------------
pc = Perceptron(max_iter=1000, eta0=0.1, random_state=42)
pc.fit(Xb_train_s, yb_train)
pc_pred = pc.predict(Xb_test_s)
print('Perceptron accuracy (2-class):', accuracy_score(yb_test, pc_pred))

# ------------------------------------------------------------
# 3. MLP (Multi-layer perceptron) – 1 hidden layer
# ------------------------------------------------------------
mlp = MLPClassifier(hidden_layer_sizes=(10,), max_iter=1000, random_state=42)
mlp.fit(X_train_s, y_train)
mlp_pred = mlp.predict(X_test_s)
print('MLP accuracy (3-class):', accuracy_score(y_test, mlp_pred))

# ------------------------------------------------------------
# 4. Decision-boundary plot (2-D projection)
# ------------------------------------------------------------
def plot_boundaries(ax, model, X, y, title):
    h = 0.02
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    ax.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.Paired)
    ax.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', cmap=plt.cm.Paired)
    ax.set_title(title)

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
plot_boundaries(axes[0], pc,  Xb_test_s, yb_test, 'Perceptron (2-class)')
plot_boundaries(axes[1], mlp, X_test_s,  y_test,  'MLP (3-class, 10 hidden)')
plt.tight_layout()
plt.show()
