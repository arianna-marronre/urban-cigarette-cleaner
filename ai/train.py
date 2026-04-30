import tensorflow as tf
from model import get_improved_unet
from data import get_dataset
import os

# --- Losses and Metrics ---
def dice_coef(y_true, y_pred):
    smooth = 1.0
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)

def dice_loss(y_true, y_pred):
    return 1.0 - dice_coef(y_true, y_pred)

def bce_dice_loss(y_true, y_pred):
    """
    Combined loss to handle class imbalance and pixel-wise accuracy.
    """
    bce = tf.keras.losses.BinaryCrossentropy()(y_true, y_pred)
    return bce + dice_loss(y_true, y_pred)

# --- Training Pipeline ---
def train():
    IMG_SIZE = (256, 256)
    BATCH_SIZE = 16
    EPOCHS = 50
    LEARNING_RATE = 1e-4

    # 1. Setup Data (Assume standardized paths)
    train_dataset = get_dataset("dataset/train/images", "dataset/train/masks", batch_size=BATCH_SIZE, is_training=True)
    val_dataset = get_dataset("dataset/val/images", "dataset/val/masks", batch_size=BATCH_SIZE, is_training=False)

    # 2. Build Model
    model = get_improved_unet(input_shape=(*IMG_SIZE, 3))
    
    # 3. Compile
    optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
    model.compile(
        optimizer=optimizer,
        loss=bce_dice_loss,
        metrics=[dice_coef, 'accuracy', tf.keras.metrics.MeanIoU(num_classes=2)]
    )

    # 4. Callbacks
    checkpoint_path = "checkpoints/unet_best.keras"
    os.makedirs("checkpoints", exist_ok=True)
    
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(checkpoint_path, monitor='val_dice_coef', mode='max', save_best_only=True, verbose=1),
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6)
    ]

    # 5. Execute Training
    print("Starting training...")
    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=EPOCHS,
        callbacks=callbacks
    )

    # 6. Save final model
    model.save("cigarette_butt_seg_v2.keras")
    print("Training complete. Model saved.")

if __name__ == "__main__":
    train()
