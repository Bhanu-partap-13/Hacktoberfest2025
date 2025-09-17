import numpy as np
import matplotlib.pyplot as plt

# ------------------------------------------------------------
# 1. Define activations and their names
# ------------------------------------------------------------
def sigmoid(x): return 1 / (1 + np.exp(-x))
def tanh(x): return np.tanh(x)
def relu(x): return np.maximum(0, x)
def leaky_relu(x, alpha=0.01): return np.where(x > 0, x, alpha * x)
def elu(x, alpha=1.0): return np.where(x > 0, x, alpha * (np.exp(x) - 1))
def swish(x): return x * sigmoid(x)
def gelu(x): return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))
def softplus(x): return np.log1p(np.exp(x))
def softmax(x): return np.exp(x) / np.sum(np.exp(x), axis=0, keepdims=True)

funcs = {
    'Sigmoid': sigmoid,
    'Tanh': tanh,
    'ReLU': relu,
    'Leaky-ReLU(α=0.1)': lambda x: leaky_relu(x, 0.1),
    'ELU(α=1)': elu,
    'Swish': swish,
    'GELU': gelu,
    'Softplus': softplus
}

# ------------------------------------------------------------
# 2. Plot them
# ------------------------------------------------------------
x = np.linspace(-4, 4, 400)
plt.figure(figsize=(12, 8))
for i, (name, f) in enumerate(funcs.items(), 1):
    plt.subplot(2, 4, i)
    y = f(x)
    plt.plot(x, y, label=name)
    plt.title(f'{name}  Range [{y.min():.2f}, {y.max():.2f}]')
    plt.grid(alpha=0.3)
    plt.axhline(0, color='k', lw=0.5)
    plt.axvline(0, color='k', lw=0.5)
plt.tight_layout()
plt.show()

# ------------------------------------------------------------
# 3. Quick comparison table (range + derivative cheat)
# ------------------------------------------------------------
print("Range & Gradient cheat-sheet")
print("-" * 45)
for name, f in funcs.items():
    y = f(x)
    grad = np.gradient(y, x)
    print(f"{name:15} | range [{y.min():.2f}, {y.max():.2f}] | grad max {np.abs(grad).max():.2f}")
