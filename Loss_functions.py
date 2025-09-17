import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import torch
import torch.nn as nn

# ------------------------------------------------------------
# 1. Regression losses
# ------------------------------------------------------------
def mse_loss(y_true, y_pred):
    return np.mean((y_true - y_pred)**2)

def mae_loss(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))

def huber_loss(y_true, y_pred, delta=1.0):
    err = y_true - y_pred
    mask = np.abs(err) <= delta
    loss = np.where(mask, 0.5*err**2, delta*(np.abs(err) - 0.5*delta))
    return np.mean(loss)

# ------------------------------------------------------------
# 2. Classification losses (probabilistic)
# ------------------------------------------------------------
def bce_loss(y_true, y_prob):
    eps = 1e-15
    y_prob = np.clip(y_prob, eps, 1-eps)
    return -np.mean(y_true*np.log(y_prob) + (1-y_true)*np.log(1-y_prob))

def cce_loss(y_true, y_prob):
    # y_true is one-hot, y_prob is softmax
    eps = 1e-15
    y_prob = np.clip(y_prob, eps, 1.0)
    return -np.mean(np.sum(y_true * np.log(y_prob), axis=1))

def hinge_loss(y_true, y_raw):
    # y_true ∈ {-1,1}, y_raw = raw score
    margin = 1 - y_true * y_raw
    margin = np.maximum(0, margin)
    return np.mean(margin)

# ------------------------------------------------------------
# 3. TensorFlow / PyTorch one-liners
# ------------------------------------------------------------
tf_losses = {
    'MSE': tf.keras.losses.MeanSquaredError(),
    'MAE': tf.keras.losses.MeanAbsoluteError(),
    'Huber': tf.keras.losses.Huber(),
    'BCE': tf.keras.losses.BinaryCrossentropy(from_logits=False),
    'CCE': tf.keras.losses.CategoricalCrossentropy(from_logits=False),
    'Hinge': tf.keras.losses.Hinge()
}

torch_losses = {
    'MSE': nn.MSELoss(),
    'MAE': nn.L1Loss(),
    'Huber': nn.HuberLoss(),
    'BCE': nn.BCELoss(),
    'CCE': nn.CrossEntropyLoss(),  # combines LogSoftmax + NLL
    'Hinge': nn.HingeEmbeddingLoss()
}

# ------------------------------------------------------------
# 4. Visualise regression losses
# ------------------------------------------------------------
y_true = 0.0
err = np.linspace(-3, 3, 500)
plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1)
plt.plot(err, mse_loss(y_true, err), label='MSE')
plt.plot(err, mae_loss(y_true, err), label='MAE')
plt.plot(err, huber_loss(y_true, err), label='Huber')
plt.xlabel('error (y_true - y_pred)')
plt.ylabel('loss')
plt.title('Regression Losses')
plt.legend()
plt.grid(alpha=0.3)

# ------------------------------------------------------------
# 5. Visualise classification losses
# ------------------------------------------------------------
# binary case
y_true_bin = 1
p = np.linspace(0.01, 0.99, 100)
plt.subplot(1, 3, 2)
plt.plot(p, bce_loss(y_true_bin, p), label='BCE')
plt.xlabel('predicted probability')
plt.ylabel('loss')
plt.title('Binary Classification')
plt.legend()
plt.grid(alpha=0.3)

# hinge
raw = np.linspace(-2, 2, 100)
plt.subplot(1, 3, 3)
plt.plot(raw, hinge_loss(1, raw), label='Hinge (y=1)')
plt.plot(raw, hinge_loss(-1, raw), label='Hinge (y=-1)')
plt.xlabel('raw score')
plt.ylabel('loss')
plt.title('Hinge Loss')
plt.legend()
plt.grid(alpha=0.3)

plt.tight_layout()
plt.show()

# ------------------------------------------------------------
# 6. Quick lookup table
# ------------------------------------------------------------
print("Loss Cheat-Sheet")
print("-" * 60)
print("Regression")
print("  MSE   – mean squared error (L2)")
print("  MAE   – mean absolute error (L1)")
print("  Huber – quadratic near 0, linear far")
print("Classification")
print("  BCE   – binary cross-entropy (log-loss)")
print("  CCE   – categorical cross-entropy")
print("  Hinge – SVM style margin loss")
print("Framework snippets")
print("  TF : tf.keras.losses.MeanSquaredError()")
print("  PT : nn.MSELoss()")
