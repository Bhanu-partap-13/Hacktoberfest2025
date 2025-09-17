# ------------------------------------------------------------
# 0. Imports
# ------------------------------------------------------------
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# ------------------------------------------------------------
# 1. Data
# ------------------------------------------------------------
iris = load_iris()
X, y = iris.data, iris.target
feature_names = iris.feature_names
class_names   = iris.target_names

# ------------------------------------------------------------
# 2. Train / test split
# ------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y)

# ------------------------------------------------------------
# 3. Fit a single Decision-Tree (default gini, max_depth=None)
# ------------------------------------------------------------
tree = DecisionTreeClassifier(random_state=42)
tree.fit(X_train, y_train)

# ------------------------------------------------------------
# 4. Evaluate
# ------------------------------------------------------------
pred = tree.predict(X_test)
print("Accuracy:", accuracy_score(y_test, pred))

# ------------------------------------------------------------
# 5. Text rules
# ------------------------------------------------------------
from sklearn.tree import export_text
rules = export_text(tree, feature_names=feature_names)
print("\nDecision rules:\n", rules[:1200], "..." if len(rules) > 1200 else "")

# ------------------------------------------------------------
# 6. Visual tree
# ------------------------------------------------------------
plt.figure(figsize=(10, 6))
plot_tree(tree, feature_names=feature_names,
          class_names=class_names, filled=True, rounded=True)
plt.title("Hello-World Decision-Tree on Iris")
plt.show()
