# ------------------------------------------------------------
# 0. Imports
# ------------------------------------------------------------
from sklearn import svm
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import numpy as np

# ------------------------------------------------------------
# 1. Data (use only 2 features so we can plot the boundary)
# ------------------------------------------------------------
iris = load_iris()
X, y = iris.data[:, :2], iris.target          # sepal length & width only
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

# ------------------------------------------------------------
# 2. Train two SVM models
# ------------------------------------------------------------
models = {
    'Linear': svm.SVC(kernel='linear', C=1.0),
    'RBF':    svm.SVC(kernel='rbf', gamma=0.7, C=1.0)
}

for name, model in models.items():
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    print(f"{name} kernel accuracy: {accuracy_score(y_test, pred):.3f}")

# ------------------------------------------------------------
# 3. Plot decision boundaries
# ------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
for ax, (name, model) in zip(axes, models.items()):
    # mesh
    h = 0.02
    x_min, x_max = X_train[:, 0].min() - 1, X_train[:, 0].max() + 1
    y_min, y_max = X_train[:, 1].min() - 1, X_train[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    # contour + points
    ax.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.Paired)
    ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, edgecolors='k', cmap=plt.cm.Paired)
    ax.set_title(f'SVM ({name} kernel)')
plt.tight_layout()
plt.show()
