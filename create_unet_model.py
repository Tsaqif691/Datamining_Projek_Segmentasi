import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import os

def build_unet_model(input_shape=(256, 256, 3)):
    inputs = keras.Input(shape=input_shape)
    
    conv1 = layers.Conv2D(64, 3, activation="relu", padding="same")(inputs)
    conv1 = layers.Conv2D(64, 3, activation="relu", padding="same")(conv1)
    pool1 = layers.MaxPooling2D(pool_size=(2, 2))(conv1)
    
    conv2 = layers.Conv2D(128, 3, activation="relu", padding="same")(pool1)
    conv2 = layers.Conv2D(128, 3, activation="relu", padding="same")(conv2)
    pool2 = layers.MaxPooling2D(pool_size=(2, 2))(conv2)
    
    conv3 = layers.Conv2D(256, 3, activation="relu", padding="same")(pool2)
    conv3 = layers.Conv2D(256, 3, activation="relu", padding="same")(conv3)
    pool3 = layers.MaxPooling2D(pool_size=(2, 2))(conv3)
    
    conv4 = layers.Conv2D(512, 3, activation="relu", padding="same")(pool3)
    conv4 = layers.Conv2D(512, 3, activation="relu", padding="same")(conv4)
    
    up5 = layers.UpSampling2D(size=(2, 2))(conv4)
    up5 = layers.concatenate([up5, conv3])
    conv5 = layers.Conv2D(256, 3, activation="relu", padding="same")(up5)
    conv5 = layers.Conv2D(256, 3, activation="relu", padding="same")(conv5)
    
    up6 = layers.UpSampling2D(size=(2, 2))(conv5)
    up6 = layers.concatenate([up6, conv2])
    conv6 = layers.Conv2D(128, 3, activation="relu", padding="same")(up6)
    conv6 = layers.Conv2D(128, 3, activation="relu", padding="same")(conv6)
    
    up7 = layers.UpSampling2D(size=(2, 2))(conv6)
    up7 = layers.concatenate([up7, conv1])
    conv7 = layers.Conv2D(64, 3, activation="relu", padding="same")(up7)
    conv7 = layers.Conv2D(64, 3, activation="relu", padding="same")(conv7)
    
    outputs = layers.Conv2D(1, 1, activation="sigmoid")(conv7)
    
    model = keras.Model(inputs=inputs, outputs=outputs, name="unet_model")
    return model

def main():
    print("=" * 60)
    print("CREATING U-NET MODEL")
    print("=" * 60)
    
    print("\n📦 Building U-Net architecture...")
    model = build_unet_model()
    
    print("⚙️  Compiling model...")
    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )
    
    print("\n📊 Model Summary:")
    model.summary()
    
    os.makedirs("models", exist_ok=True)
    
    save_path = "models/water_unet_model.h5"
    print(f"\n💾 Saving model to: {save_path}")
    model.save(save_path)
    
    file_size = os.path.getsize(save_path)
    print(f"✅ Model saved successfully!")
    print(f"   File size: {file_size:,} bytes ({file_size / (1024*1024):.2f} MB)")
    
    print("\n🔄 Testing model load...")
    loaded_model = tf.keras.models.load_model(save_path)
    print("✅ Model can be loaded successfully!")
    
    print("\n" + "=" * 60)
    print("✅ DONE! Model is ready to use.")
    print("=" * 60)

if __name__ == "__main__":
    main()
