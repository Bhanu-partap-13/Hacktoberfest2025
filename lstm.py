import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam

# 1️⃣ generate sine wave
np.random.seed(42)
T = 1000
t = np.linspace(0, 100, T)
x = np.sin(t).reshape(-1, 1)

# 2️⃣ make dataset: 20-step window → next value
window = 20
X, y = [], []
for i in range(len(x) - window):
    X.append(x[i:i+window])
    y.append(x[i+window])
X, y = np.array(X), np.array(y)
print('X shape:', X.shape, 'y shape:', y.shape)   # (samples, 20, 1)

# 3️⃣ train / test split
split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# 4️⃣ build LSTM model
model = Sequential([
    LSTM(32, activation='tanh', input_shape=(window, 1)),
    Dense(1)
])
model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
model.summary()

# 5️⃣ train
history = model.fit(X_train, y_train,
                    epochs=50,
                    batch_size=32,
                    validation_split=0.2,
                    verbose=0)

# 6️⃣ predict
pred = model.predict(X_test)

# 7️⃣ visualise
plt.figure(figsize=(8, 3))
plt.plot(y_test, label='True next value')
plt.plot(pred, label='LSTM prediction')
plt.legend()
plt.title('LSTM on sine wave (next-step prediction)')
plt.tight_layout()
plt.show()

# 8️⃣ metrics
rmse = np.sqrt(np.mean((y_test - pred)**2))
print(f'Test RMSE: {rmse:.4f}')
