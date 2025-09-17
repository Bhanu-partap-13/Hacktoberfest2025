# ------------------------------------------------------------
# 0. Imports
# ------------------------------------------------------------
import time, pandas as pd, numpy as np, matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.utils import to_categorical

# 1. fix GPU memory growth (optional)
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    tf.config.experimental.set_memory_growth(gpus[0], True)

# ------------------------------------------------------------
# 2. Data – tiny subset for speed
# ------------------------------------------------------------
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
y_train, y_test = to_categorical(y_train), to_categorical(y_test)
# use only 5 k samples → fast demo
x_train, y_train = x_train[:5000], y_train[:5000]

# ------------------------------------------------------------
# 3. Model builder (same weights seed each run)
# ------------------------------------------------------------
def build_model():
    model = Sequential([
        Flatten(input_shape=(28, 28)),
        Dense(128, activation='relu'),
        Dense(10, activation='softmax')
    ])
    return model

# ------------------------------------------------------------
# 4. Optimisers to compare
# ------------------------------------------------------------
optimizers = {
    'SGD': tf.keras.optimizers.SGD(learning_rate=0.01),
    'SGD-momentum': tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9),
    'AdaGrad': tf.keras.optimizers.Adagrad(learning_rate=0.01),
    'RMSprop': tf.keras.optimizers.RMSprop(learning_rate=0.001),
    'Adam': tf.keras.optimizers.Adam(learning_rate=0.001),
    'AdamW': tf.keras.optimizers.AdamW(learning_rate=0.001, weight_decay=0.01),
    'Nadam': tf.keras.optimizers.Nadam(learning_rate=0.001),
    'Adamax': tf.keras.optimizers.Adamax(learning_rate=0.001),
}

results = []

# ------------------------------------------------------------
# 5. Train each optimiser (1 epoch only for speed demo)
# ------------------------------------------------------------
EPOCHS = 1
BATCH = 128
for name, opt in optimizers.items():
    print(f'\n>>> {name}')
    model = build_model()
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    start = time.time()
    hist = model.fit(x_train, y_train,
                     epochs=EPOCHS,
                     batch_size=BATCH,
                     validation_split=0.2,
                     verbose=0)
    wall = time.time() - start
    final_train_loss = hist.history['loss'][-1]
    final_val_acc = hist.history['val_accuracy'][-1]
    results.append({'Optimizer': name,
                    'TrainLoss': final_train_loss,
                    'ValAcc': final_val_acc,
                    'Time(s)': wall})

# ------------------------------------------------------------
# 6. Pretty table + bar chart
# ------------------------------------------------------------
df = pd.DataFrame(results).sort_values('ValAcc', ascending=False)
print('\n=== Summary (1 epoch on 5 k MNIST) ===')
print(df.to_string(index=False, float_format='%.4f'))

plt.figure(figsize=(8, 4))
plt.bar(df['Optimizer'], df['ValAcc'])
plt.title('Validation accuracy after 1 epoch')
plt.ylabel('Accuracy')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


"""SGD – plain, needs momentum for speed
AdaGrad – adaptive per-dim learning rates, stalls late
RMSprop – AdaGrad fix (decaying avg)
Adam – RMSprop + momentum, default choice
AdamW – Adam + decoupled weight decay (better generalisation)
Nadam – Adam + Nesterov momentum
Adamax – Adam with L∞ norm (stable on inf gradients)"""
