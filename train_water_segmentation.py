"""
Script untuk training U-Net model untuk segmentasi badan air
Dataset: Satellite Images of Water Bodies (Kaggle)
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# ==================== KONFIGURASI ====================
IMG_SIZE = 256
BATCH_SIZE = 8
EPOCHS = 50
LEARNING_RATE = 0.001

# Path dataset (sesuaikan dengan lokasi extract Anda)
DATASET_PATH = "dataset/Water Bodies Dataset"
IMAGES_PATH = os.path.join(DATASET_PATH, "Images")
MASKS_PATH = os.path.join(DATASET_PATH, "Masks")

# ==================== FUNGSI HELPER ====================

def load_image_and_mask(image_path, mask_path, img_size=IMG_SIZE):
    """Load dan preprocessing image dan mask"""
    image = Image.open(image_path).convert('RGB')
    image = image.resize((img_size, img_size))
    image = np.array(image) / 255.0
    
    mask = Image.open(mask_path).convert('L')
    mask = mask.resize((img_size, img_size))
    mask = np.array(mask) / 255.0
    mask = (mask > 0.5).astype(np.float32)
    mask = np.expand_dims(mask, axis=-1)
    
    return image.astype(np.float32), mask.astype(np.float32)

def create_dataset(image_dir, mask_dir, img_size=IMG_SIZE):
    """Buat dataset dari folder images dan masks"""
    images = []
    masks = []
    
    image_files = sorted([f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.png', '.jpeg'))])
    
    print(f"Loading {len(image_files)} images...")
    
    for img_file in image_files:
        mask_file = img_file
        
        image_path = os.path.join(image_dir, img_file)
        mask_path = os.path.join(mask_dir, mask_file)
        
        if os.path.exists(mask_path):
            try:
                image, mask = load_image_and_mask(image_path, mask_path, img_size)
                images.append(image)
                masks.append(mask)
            except Exception as e:
                print(f"Error loading {img_file}: {e}")
    
    print(f"Successfully loaded {len(images)} image-mask pairs")
    
    return np.array(images), np.array(masks)

def build_unet_model(input_shape=(IMG_SIZE, IMG_SIZE, 3)):
    """Build U-Net architecture"""
    inputs = keras.Input(shape=input_shape)
    
    c1 = layers.Conv2D(64, 3, activation='relu', padding='same')(inputs)
    c1 = layers.Conv2D(64, 3, activation='relu', padding='same')(c1)
    p1 = layers.MaxPooling2D(2)(c1)
    
    c2 = layers.Conv2D(128, 3, activation='relu', padding='same')(p1)
    c2 = layers.Conv2D(128, 3, activation='relu', padding='same')(c2)
    p2 = layers.MaxPooling2D(2)(c2)
    
    c3 = layers.Conv2D(256, 3, activation='relu', padding='same')(p2)
    c3 = layers.Conv2D(256, 3, activation='relu', padding='same')(c3)
    p3 = layers.MaxPooling2D(2)(c3)
    
    c4 = layers.Conv2D(512, 3, activation='relu', padding='same')(p3)
    c4 = layers.Conv2D(512, 3, activation='relu', padding='same')(c4)
    
    u5 = layers.UpSampling2D(2)(c4)
    u5 = layers.concatenate([u5, c3])
    c5 = layers.Conv2D(256, 3, activation='relu', padding='same')(u5)
    c5 = layers.Conv2D(256, 3, activation='relu', padding='same')(c5)
    
    u6 = layers.UpSampling2D(2)(c5)
    u6 = layers.concatenate([u6, c2])
    c6 = layers.Conv2D(128, 3, activation='relu', padding='same')(u6)
    c6 = layers.Conv2D(128, 3, activation='relu', padding='same')(c6)
    
    u7 = layers.UpSampling2D(2)(c6)
    u7 = layers.concatenate([u7, c1])
    c7 = layers.Conv2D(64, 3, activation='relu', padding='same')(u7)
    c7 = layers.Conv2D(64, 3, activation='relu', padding='same')(c7)
    
    outputs = layers.Conv2D(1, 1, activation='sigmoid')(c7)
    
    model = keras.Model(inputs=inputs, outputs=outputs, name='unet_water_segmentation')
    
    return model

def dice_coefficient(y_true, y_pred, smooth=1):
    """Dice coefficient metric"""
    y_true_f = tf.keras.backend.flatten(y_true)
    y_pred_f = tf.keras.backend.flatten(y_pred)
    intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) + smooth)

def dice_loss(y_true, y_pred):
    """Dice loss function"""
    return 1 - dice_coefficient(y_true, y_pred)

def visualize_predictions(model, X_test, y_test, num_samples=3):
    """Visualize model predictions"""
    plt.figure(figsize=(15, 5 * num_samples))
    
    for i in range(num_samples):
        idx = np.random.randint(0, len(X_test))
        image = X_test[idx]
        true_mask = y_test[idx]
        
        pred_mask = model.predict(np.expand_dims(image, axis=0), verbose=0)[0]
        pred_mask_binary = (pred_mask > 0.5).astype(np.float32)
        
        plt.subplot(num_samples, 3, i * 3 + 1)
        plt.imshow(image)
        plt.title("Original Image")
        plt.axis('off')
        
        plt.subplot(num_samples, 3, i * 3 + 2)
        plt.imshow(true_mask.squeeze(), cmap='gray')
        plt.title("True Mask")
        plt.axis('off')
        
        plt.subplot(num_samples, 3, i * 3 + 3)
        plt.imshow(pred_mask_binary.squeeze(), cmap='gray')
        plt.title("Predicted Mask")
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('training_results.png', dpi=150, bbox_inches='tight')
    print("✅ Visualization saved to 'training_results.png'")
    plt.close()

# ==================== MAIN TRAINING ====================

def main():
    print("=" * 60)
    print("WATER SEGMENTATION MODEL TRAINING")
    print("=" * 60)
    
    if not os.path.exists(IMAGES_PATH):
        print(f"❌ Error: Images folder not found at {IMAGES_PATH}")
        print("Please download and extract the Kaggle dataset first!")
        return
    
    print("\n📂 Loading dataset...")
    X, y = create_dataset(IMAGES_PATH, MASKS_PATH)
    
    if len(X) == 0:
        print("❌ No data loaded! Check your dataset paths.")
        return
    
    print(f"✅ Dataset loaded: {X.shape}, Masks: {y.shape}")
    
    print("\n🔀 Splitting data (80% train, 20% validation)...")
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"Training set: {X_train.shape}")
    print(f"Validation set: {X_val.shape}")
    
    print("\n🏗️  Building U-Net model...")
    model = build_unet_model()
    
    print("⚙️  Compiling model...")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss=dice_loss,
        metrics=[dice_coefficient, 'accuracy']
    )
    
    print("\n📊 Model Summary:")
    model.summary()
    
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            'models/water_unet_model_best.h5',
            monitor='val_dice_coefficient',
            mode='max',
            save_best_only=True,
            verbose=1
        ),
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        )
    ]
    
    print(f"\n🚀 Starting training for {EPOCHS} epochs...")
    print("=" * 60)
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=1
    )
    
    print("\n💾 Saving final model...")
    os.makedirs('models', exist_ok=True)
    model.save('models/water_unet_model.h5')
    
    file_size = os.path.getsize('models/water_unet_model.h5')
    print(f"✅ Model saved: models/water_unet_model.h5")
    print(f"   Size: {file_size / (1024*1024):.2f} MB")
    
    print("\n📊 Creating visualizations...")
    visualize_predictions(model, X_val, y_val, num_samples=5)
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE!")
    print("=" * 60)
    
    final_metrics = model.evaluate(X_val, y_val, verbose=0)
    print(f"\nFinal Validation Metrics:")
    print(f"  Loss (Dice Loss):       {final_metrics[0]:.4f}")
    print(f"  Dice Coefficient:       {final_metrics[1]:.4f}")
    print(f"  Accuracy:               {final_metrics[2]:.4f}")
    
    print("\n✅ Model is ready to use in your Streamlit app!")
    print("   Location: models/water_unet_model.h5")

if __name__ == "__main__":
    main()