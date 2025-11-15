"""
Image Segmentation with U-Net
==============================
Category 11: Computer Vision

This example demonstrates:
- U-Net architecture for semantic segmentation
- Encoder-decoder structure with skip connections
- Pixel-wise classification
- Dice coefficient and IoU metrics
- Binary and multi-class segmentation

Use cases:
- Medical image segmentation
- Satellite imagery analysis
- Autonomous driving (lane/object segmentation)
- Image editing and manipulation
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


class UNet(keras.Model):
    """U-Net architecture for image segmentation"""

    def __init__(self, num_classes=1, input_shape=(256, 256, 3)):
        super(UNet, self).__init__()

        # Encoder (Contracting Path)
        self.enc1 = self.conv_block(64, input_shape[2])
        self.pool1 = layers.MaxPooling2D(pool_size=(2, 2))

        self.enc2 = self.conv_block(128, 64)
        self.pool2 = layers.MaxPooling2D(pool_size=(2, 2))

        self.enc3 = self.conv_block(256, 128)
        self.pool3 = layers.MaxPooling2D(pool_size=(2, 2))

        self.enc4 = self.conv_block(512, 256)
        self.pool4 = layers.MaxPooling2D(pool_size=(2, 2))

        # Bottleneck
        self.bottleneck = self.conv_block(1024, 512)

        # Decoder (Expanding Path)
        self.upconv4 = layers.Conv2DTranspose(512, (2, 2), strides=(2, 2), padding='same')
        self.dec4 = self.conv_block(512, 1024)

        self.upconv3 = layers.Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')
        self.dec3 = self.conv_block(256, 512)

        self.upconv2 = layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')
        self.dec2 = self.conv_block(128, 256)

        self.upconv1 = layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')
        self.dec1 = self.conv_block(64, 128)

        # Output
        self.output_layer = layers.Conv2D(num_classes, (1, 1), activation='sigmoid')

    def conv_block(self, filters, input_channels):
        """Convolutional block with two conv layers"""
        return keras.Sequential([
            layers.Conv2D(filters, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.Conv2D(filters, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization()
        ])

    def call(self, inputs, training=False):
        # Encoder
        enc1 = self.enc1(inputs, training=training)
        pool1 = self.pool1(enc1)

        enc2 = self.enc2(pool1, training=training)
        pool2 = self.pool2(enc2)

        enc3 = self.enc3(pool2, training=training)
        pool3 = self.pool3(enc3)

        enc4 = self.enc4(pool3, training=training)
        pool4 = self.pool4(enc4)

        # Bottleneck
        bottleneck = self.bottleneck(pool4, training=training)

        # Decoder with skip connections
        up4 = self.upconv4(bottleneck)
        concat4 = layers.concatenate([up4, enc4])
        dec4 = self.dec4(concat4, training=training)

        up3 = self.upconv3(dec4)
        concat3 = layers.concatenate([up3, enc3])
        dec3 = self.dec3(concat3, training=training)

        up2 = self.upconv2(dec3)
        concat2 = layers.concatenate([up2, enc2])
        dec2 = self.dec2(concat2, training=training)

        up1 = self.upconv1(dec2)
        concat1 = layers.concatenate([up1, enc1])
        dec1 = self.dec1(concat1, training=training)

        # Output
        output = self.output_layer(dec1)

        return output


def generate_synthetic_segmentation_data(num_samples=500, img_size=256):
    """Generate synthetic images with segmentation masks"""
    np.random.seed(42)

    images = []
    masks = []

    for _ in range(num_samples):
        # Create image with random shapes
        image = np.random.rand(img_size, img_size, 3) * 0.3
        mask = np.zeros((img_size, img_size, 1))

        # Add random circles
        num_circles = np.random.randint(2, 5)

        for _ in range(num_circles):
            center_x = np.random.randint(50, img_size - 50)
            center_y = np.random.randint(50, img_size - 50)
            radius = np.random.randint(20, 50)

            # Create circle in image
            y, x = np.ogrid[:img_size, :img_size]
            circle_mask = (x - center_x) ** 2 + (y - center_y) ** 2 <= radius ** 2

            # Random color for circle
            color = np.random.rand(3)
            image[circle_mask] = color

            # Add to segmentation mask
            mask[circle_mask] = 1

        images.append(image.astype(np.float32))
        masks.append(mask.astype(np.float32))

    return np.array(images), np.array(masks)


def dice_coefficient(y_true, y_pred, smooth=1e-6):
    """
    Dice coefficient metric for segmentation

    Dice = 2 * |A ∩ B| / (|A| + |B|)
    """
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])

    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    dice = (2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)

    return dice


def dice_loss(y_true, y_pred):
    """Dice loss = 1 - Dice coefficient"""
    return 1 - dice_coefficient(y_true, y_pred)


def iou_metric(y_true, y_pred, smooth=1e-6):
    """
    Intersection over Union (IoU) metric

    IoU = |A ∩ B| / |A ∪ B|
    """
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])

    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    union = tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) - intersection

    iou = (intersection + smooth) / (union + smooth)

    return iou


def train_unet_segmentation():
    """Train U-Net for image segmentation"""
    print("=" * 60)
    print("U-Net Image Segmentation Training")
    print("=" * 60)

    # Set random seeds
    np.random.seed(42)
    tf.random.set_seed(42)

    # Generate data
    print("\nGenerating synthetic segmentation data...")
    images, masks = generate_synthetic_segmentation_data(num_samples=500, img_size=256)

    print(f"Images shape: {images.shape}")
    print(f"Masks shape: {masks.shape}")

    # Train/val/test split
    X_train, X_temp, y_train, y_temp = train_test_split(
        images, masks, test_size=0.3, random_state=42
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42
    )

    print(f"\nTrain: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

    # Build model
    model = UNet(num_classes=1, input_shape=(256, 256, 3))

    # Compile with custom metrics
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss=dice_loss,
        metrics=[dice_coefficient, iou_metric]
    )

    print("\nModel Architecture:")
    model.build(input_shape=(None, 256, 256, 3))
    model.summary()

    # Callbacks
    early_stopping = keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True
    )

    reduce_lr = keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-6
    )

    # Train
    print("\nTraining U-Net model...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=30,
        batch_size=4,
        callbacks=[early_stopping, reduce_lr],
        verbose=1
    )

    # Evaluate on test set
    print("\nEvaluating on test set...")
    test_results = model.evaluate(X_test, y_test, verbose=0)

    print(f"\nTest Results:")
    print(f"Loss: {test_results[0]:.4f}")
    print(f"Dice Coefficient: {test_results[1]:.4f}")
    print(f"IoU: {test_results[2]:.4f}")

    # Save model
    model.save('/tmp/unet_segmentation_model.keras')
    print("\nModel saved to /tmp/unet_segmentation_model.keras")

    # Visualize results
    visualize_segmentation_results(model, X_test, y_test, history)

    return model, history


def visualize_segmentation_results(model, X_test, y_test, history):
    """Visualize segmentation results"""
    print("\nGenerating visualizations...")

    # Predict on test samples
    predictions = model.predict(X_test[:5])

    # Plot training history
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Loss
    axes[0].plot(history.history['loss'], label='Train Loss')
    axes[0].plot(history.history['val_loss'], label='Val Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Dice Loss')
    axes[0].set_title('Training History - Loss')
    axes[0].legend()
    axes[0].grid(True)

    # Dice Coefficient
    axes[1].plot(history.history['dice_coefficient'], label='Train Dice')
    axes[1].plot(history.history['val_dice_coefficient'], label='Val Dice')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Dice Coefficient')
    axes[1].set_title('Training History - Dice Coefficient')
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    plt.savefig('/tmp/unet_training_history.png')
    print("Training history saved to /tmp/unet_training_history.png")

    # Plot sample predictions
    fig, axes = plt.subplots(3, 3, figsize=(12, 12))

    for i in range(3):
        # Original image
        axes[i, 0].imshow(X_test[i])
        axes[i, 0].set_title('Original Image')
        axes[i, 0].axis('off')

        # Ground truth mask
        axes[i, 1].imshow(y_test[i].squeeze(), cmap='gray')
        axes[i, 1].set_title('Ground Truth')
        axes[i, 1].axis('off')

        # Predicted mask
        pred_mask = predictions[i].squeeze()
        axes[i, 2].imshow(pred_mask, cmap='gray')
        axes[i, 2].set_title(f'Prediction')
        axes[i, 2].axis('off')

    plt.tight_layout()
    plt.savefig('/tmp/unet_predictions.png')
    print("Predictions saved to /tmp/unet_predictions.png")


def main():
    """Main execution function"""
    print("U-Net Image Segmentation with TensorFlow\n")

    # Train U-Net
    model, history = train_unet_segmentation()

    print("\n" + "=" * 60)
    print("U-Net Segmentation Complete!")
    print("=" * 60)
    print("\nKey Takeaways:")
    print("- U-Net uses encoder-decoder architecture")
    print("- Skip connections preserve spatial information")
    print("- Dice coefficient measures overlap quality")
    print("- IoU is standard metric for segmentation")
    print("- Binary cross-entropy or Dice loss work well")
    print("- Works excellently for medical imaging")
    print("- Can be extended to multi-class segmentation")


if __name__ == "__main__":
    main()
