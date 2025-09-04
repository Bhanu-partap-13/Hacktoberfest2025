import tensorflow as tf
from tensorflow.keras import layers, models

# 1. Parameters
img_size   = (64, 64)
batch_size = 32
epochs     = 5            # toy run; increase for real accuracy

# 2. Load images from folders ---------------------------------
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    "data/train",
    image_size=img_size,
    batch_size=batch_size)

val_ds   = tf.keras.preprocessing.image_dataset_from_directory(
    "data/val",
    image_size=img_size,
    batch_size=batch_size)

# 3. Build a mini CNN -----------------------------------------
model = models.Sequential([
    layers.Rescaling(1./255, input_shape=(*img_size, 3)),
    layers.Conv2D(16, 3, activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(32, 3, activation='relu'),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(1,  activation='sigmoid')   # 2 classes
])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# 4. Train -----------------------------------------------------
model.fit(train_ds, validation_data=val_ds, epochs=epochs)

# 5. Quick test on a new image ---------------------------------
# img = tf.keras.preprocessing.image.load_img("test/cat1.jpg", target_size=img_size)
# x   = tf.keras.preprocessing.image.img_to_array(img)[None]/255.0
# print("Cat?" , model.predict(x)[0,0] < 0.5)
