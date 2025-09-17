import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

# ---------- 1. toy sequence: sine wave ----------
T = 200
t = np.linspace(0, 4*np.pi, T)
x = np.sin(t).reshape(-1, 1)          # column vector

# ---------- 2. build dataset: 10-step window → next value ----------
window = 10
X, y = [], []
for i in range(len(x) - window):
    X.append(x[i:i+window])
    y.append(x[i+window])
X, y = np.array(X), np.array(y)       # X shape (samples, window, 1)

# ---------- 3. train/test split ----------
split = int(0.8*len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# ---------- 4. hyper-parameters ----------
input_size  = 1
hidden_size = 16
output_size = 1
lr          = 0.001
epochs      = 100

# ---------- 5. initialise weights ----------
Wxh = np.random.randn(input_size, hidden_size) * 0.01
Whh = np.random.randn(hidden_size, hidden_size) * 0.01
Why = np.random.randn(hidden_size, output_size) * 0.01
bh  = np.zeros((1, hidden_size))
by  = np.zeros((1, output_size))

# ---------- 6. forward pass for one sample ----------
def forward(x_seq):
    """
    x_seq: (time_steps, input_size)
    returns hidden states h (time_steps+1, hidden_size) and final output
    """
    h = np.zeros((len(x_seq)+1, hidden_size))
    for t in range(len(x_seq)):
        h[t+1] = np.tanh(x_seq[t] @ Wxh + h[t] @ Whh + bh)
    y_pred = h[-1] @ Why + by
    return h, y_pred

# ---------- 7. backward pass (BPTT) ----------
def backward(x_seq, h, y_true, y_pred):
    """
    returns gradients (same shapes as weights)
    """
    dy = y_pred - y_true
    dWhy = h[-1].T @ dy
    dby  = dy
    dh = dy @ Why.T
    dWxh = np.zeros_like(Wxh)
    dWhh = np.zeros_like(Whh)
    dbh  = np.zeros_like(bh)
    # propagate backwards through time
    for t in reversed(range(len(x_seq))):
        dh_raw = (1 - h[t+1]**2) * dh
        dbh   += dh_raw
        dWxh  += x_seq[t].T.reshape(-1,1) @ dh_raw
        dWhh  += h[t].T.reshape(-1,1) @ dh_raw
        dh = dh_raw @ Whh.T
    # clip gradients
    for d in (dWxh, dWhh, dWhy, dbh, dby):
        np.clip(d, -1, 1, out=d)
    return dWxh, dWhh, dWhy, dbh, dby

# ---------- 8. training loop ----------
losses = []
for epoch in range(epochs+1):
    total_loss = 0
    # shuffle training set each epoch
    perm = np.random.permutation(len(X_train))
    for i in perm:
        x_seq = X_train[i]      # (window, 1)
        y_true = y_train[i]     # (1, 1)
        h, y_pred = forward(x_seq)
        loss = 0.5*(y_pred - y_true)**2
        total_loss += loss
        # gradients
        grads = backward(x_seq, h, y_true, y_pred)
        # update weights
        params = [Wxh, Whh, Why, bh, by]
        for param, grad in zip(params, grads):
            param -= lr * grad
    avg_loss = total_loss / len(X_train)
    losses.append(avg_loss)
    if epoch % 10 == 0:
        print(f"epoch {epoch:3d} | loss {avg_loss:.6f}")

# ---------- 9. inference on test set ----------
preds = []
for x_seq in X_test:
    _, y_hat = forward(x_seq)
    preds.append(y_hat)
preds = np.array(preds)

# ---------- 10. visualise ----------
plt.figure(figsize=(10, 3))
plt.plot(y_test, label='True next value')
plt.plot(preds, label='RNN prediction')
plt.legend()
plt.title('Vanilla RNN – next-step prediction on sine wave')
plt.tight_layout()
plt.show()

rmse = np.sqrt(np.mean((y_test - preds)**2))
print(f'Test RMSE: {rmse:.4f}')
