import tensorflow as tf
import os
import glob
import numpy as np

def load_data(img_path, mask_path, img_size=(256, 256)):
    """
    Load and preprocess image/mask pairs.
    """
    # Load Image
    img = tf.io.read_file(img_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, img_size)
    img = tf.cast(img, tf.float32) / 255.0 # Normalization

    # Load Mask
    mask = tf.io.read_file(mask_path)
    mask = tf.image.decode_png(mask, channels=1)
    mask = tf.image.resize(mask, img_size, method='nearest')
    mask = tf.cast(mask, tf.float32) / 255.0
    # Ensure binary mask
    mask = tf.where(mask > 0.5, 1.0, 0.0)

    return img, mask

def augment_data(img, mask):
    """
    Augmentation pipeline to reduce synthetic-to-real gap.
    """
    # 1. Geometric augmentations
    if tf.random.uniform(()) > 0.5:
        img = tf.image.flip_left_right(img)
        mask = tf.image.flip_left_right(mask)
    
    if tf.random.uniform(()) > 0.5:
        img = tf.image.flip_up_down(img)
        mask = tf.image.flip_up_down(mask)

    # 2. Photometric augmentations (Bridging Synthetic Gap)
    # Synthetic images often have perfect lighting. Real images don't.
    img = tf.image.random_brightness(img, max_delta=0.2)
    img = tf.image.random_contrast(img, lower=0.8, upper=1.2)
    img = tf.image.random_saturation(img, lower=0.8, upper=1.2)
    img = tf.image.random_hue(img, max_delta=0.1)

    # 3. Add random noise (Common in real low-quality webcams)
    if tf.random.uniform(()) > 0.7:
        noise = tf.random.normal(shape=tf.shape(img), mean=0.0, stddev=0.05, dtype=tf.float32)
        img = img + noise
        img = tf.clip_by_value(img, 0.0, 1.0)

    return img, mask

def get_dataset(image_dir, mask_dir, batch_size=16, is_training=True):
    """
    Create a tf.data.Dataset for efficient training.
    """
    image_list = sorted(glob.glob(os.path.join(image_dir, "*.jpg")))
    mask_list = sorted(glob.glob(os.path.join(mask_dir, "*.png")))

    dataset = tf.data.Dataset.from_tensor_slices((image_list, mask_list))
    dataset = dataset.map(load_data, num_parallel_calls=tf.data.AUTOTUNE)

    if is_training:
        dataset = dataset.map(augment_data, num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.shuffle(buffer_size=100)
    
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset
