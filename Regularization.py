import numpy as np
import tensorflow as tf
from tensorflow.keras import regularizers
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization

# 1. data (3 k only → fast demo)
(x, y), (x_test, y_test) = mnist.load_data()
x, x_test = x/255., x_test/255.
x, y = x[:3000], y[:3000]
x = x.reshape(-1, 784)

# 2. baseline (no reg)
def run(name, model):
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    model.fit(x, y, epochs=3, verbose=0)
    _, acc = model.evaluate(x_test, y_test, verbose=0)
    w = model.layers[0].get_weights()[0]  # first dense weights
    print(f"{name:12} | acc: {acc:.3f} | |W|₁: {np.abs(w).sum():.0f}")

baseline = Sequential([Dense(128, activation='relu'), Dense(10)])
run("Baseline", baseline)

# 3. L2 weight penalty
l2_net = Sequential([Dense(128, activation='relu',
                           kernel_regularizer=regularizers.l2(1e-4)),
                     Dense(10)])
run("L2", l2_net)

# 4. L1 (sparsity)
l1_net = Sequential([Dense(128, activation='relu',
                           kernel_regularizer=regularizers.l1(1e-4)),
                     Dense(10)])
run("L1", l1_net)

# 5. Dropout
drop_net = Sequential([Dense(128, activation='relu'), Dropout(0.3), Dense(10)])
run("Dropout", drop_net)

# 6. BatchNorm (implicit reg)
bn_net = Sequential([Dense(128), BatchNormalization(), Dense(10)])
run("BatchNorm", bn_net)

# 7. Elastic-net (L1+L2)
elastic = Sequential([Dense(128, activation='relu',
                            kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4)),
                      Dense(10)])
run("Elastic", elastic)
