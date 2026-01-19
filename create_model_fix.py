import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import os

print("Building U-Net model...")

inputs = keras.Input(shape=(256, 256, 3))
x = layers.Conv2D(64, 3, activation='relu', padding='same')(inputs)
x = layers.MaxPooling2D(2)(x)
x = layers.Conv2D(128, 3, activation='relu', padding='same')(x)
x = layers.MaxPooling2D(2)(x)
x = layers.Conv2D(256, 3, activation='relu', padding='same')(x)
x = layers.UpSampling2D(2)(x)
x = layers.Conv2D(128, 3, activation='relu', padding='same')(x)
x = layers.UpSampling2D(2)(x)
x = layers.Conv2D(64, 3, activation='relu', padding='same')(x)
outputs = layers.Conv2D(1, 1, activation='sigmoid')(x)

model = keras.Model(inputs=inputs, outputs=outputs)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

print("Model built successfully!")

os.makedirs("models", exist_ok=True)
model.save("models/water_unet_model.h5")

file_size = os.path.getsize("models/water_unet_model.h5")
print(f"✅ Model saved: {file_size:,} bytes ({file_size / 1024 / 1024:.2f} MB)")

# Test load
test_model = tf.keras.models.load_model("models/water_unet_model.h5")
print("✅ Model can be loaded successfully!")