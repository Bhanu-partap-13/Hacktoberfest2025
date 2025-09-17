import numpy as np
np.random.seed(42)

# ---------- 1. toy data (XOR gate) ----------
X = np.array([[0,0],[0,1],[1,0],[1,1]])
y = np.array([[0],[1],[1],[0]])

# ---------- 2. hyper-parameters ----------
input_size  = 2
hidden_size = 4
output_size = 1
lr = 0.1
epochs = 5000

# ---------- 3. weights ----------
W1 = np.random.randn(input_size, hidden_size)
b1 = np.zeros((1, hidden_size))
W2 = np.random.randn(hidden_size, output_size)
b2 = np.zeros((1, output_size))

# ---------- 4. helpers ----------
def sigmoid(x): return 1 / (1 + np.exp(-x))
def sigmoid_deriv(x): return x * (1 - x)
def mse(y_true, y_pred): return ((y_true - y_pred)**2).mean()

# ---------- 5. training loop ----------
for i in range(epochs):
    # forward
    z1 = X @ W1 + b1
    a1 = sigmoid(z1)
    z2 = a1 @ W2 + b2
    a2 = sigmoid(z2)

    # backward
    loss = mse(y, a2)
    da2 = 2*(a2 - y) / y.size
    dz2 = da2 * sigmoid_deriv(a2)
    dW2 = a1.T @ dz2
    db2 = dz2.sum(axis=0, keepdims=True)

    da1 = dz2 @ W2.T
    dz1 = da1 * sigmoid_deriv(a1)
    dW1 = X.T @ dz1
    db1 = dz1.sum(axis=0, keepdims=True)

    # update
    W1 -= lr * dW1; b1 -= lr * db1
    W2 -= lr * dW2; b2 -= lr * db2

    if i % 500 == 0:
        print(f"epoch {i:4d} | loss {loss:.6f}")

# ---------- 6. evaluation ----------
pred = (a2 > 0.5).astype(int)
print("\nPredictions:\n", pred)
print("Accuracy :", (pred == y).mean())
