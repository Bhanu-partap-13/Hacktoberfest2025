# ------------------------------------------------------------
# 0. Imports
# ------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam

# ------------------------------------------------------------
# 1. Toy data: noisy sine waves (500 samples, 50 timesteps)
# ------------------------------------------------------------
np.random.seed(42)
n_samples, timesteps = 500, 50
x = np.linspace(0, 2 * np.pi, timesteps)
# each row is a noisy sine
X_clean = np.sin(x) + 0.1 * np.random.randn(n_samples, timesteps)
X_noisy = X_clean + 0.2 * np.random.randn(n_samples, timesteps)

# ------------------------------------------------------------
# 2. Build autoencoder
# ------------------------------------------------------------
input_dim = timesteps
encoding_dim = 2  # bottleneck dimension

inputs = Input(shape=(input_dim,))
encoded = Dense(16, activation='relu')(inputs)
encoded = Dense(encoding_dim, activation='relu')(encoded)   # LATENT SPACE

decoded = Dense(16, activation='relu')(encoded)
decoded = Dense(input_dim, activation='linear')(decoded)

autoencoder = Model(inputs, decoded)
encoder = Model(inputs, encoded)

autoencoder.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
autoencoder.summary()

# ------------------------------------------------------------
# 3. Train
# ------------------------------------------------------------
history = autoencoder.fit(X_noisy, X_clean,
                          epochs=100,
                          batch_size=32,
                          verbose=0)

# ------------------------------------------------------------
# 4. Evaluation
# ------------------------------------------------------------
recon = autoencoder.predict(X_noisy)
mse = np.mean((X_clean - recon) ** 2)
print(f'MSE (clean vs reconstruct): {mse:.4f}')

# ------------------------------------------------------------
# 5. Visualise
# ------------------------------------------------------------
plt.figure(figsize=(12, 4))

# 5a. original vs reconstructed (first 3 samples)
plt.subplot(1, 3, 1)
for i in range(3):
    plt.plot(x, X_clean[i], '--', label=f'clean {i}')
    plt.plot(x, recon[i], label=f'recon {i}')
plt.title('Original vs Reconstructed')
plt.legend()

# 5b. latent 2-D manifold (encoder output)
latent = encoder.predict(X_noisy)
plt.subplot(1, 3, 2)
plt.scatter(latent[:, 0], latent[:, 1], c=X_clean[:, 0], cmap='viridis')
plt.colorbar(label='first timestep value')
plt.title('Latent 2-D space')

# 5c. training curve
plt.subplot(1, 3, 3)
plt.plot(history.history['loss'])
plt.title('Training loss')
plt.xlabel('epoch')
plt.tight_layout()
plt.show()
