import tensorflow as tf
from tensorflow.keras import layers, models

def attention_block(x, gating, inter_shape):
    shape_x = tf.keras.backend.int_shape(x)
    shape_g = tf.keras.backend.int_shape(gating)

    # Getting the gating signal to the same shape as the feature map
    theta_x = layers.Conv2D(inter_shape, (2, 2), strides=(2, 2), padding='same')(x)  # 16
    shape_theta_x = tf.keras.backend.int_shape(theta_x)

    phi_g = layers.Conv2D(inter_shape, (1, 1), padding='same')(gating)
    upsample_g = layers.Conv2DTranspose(inter_shape, (3, 3),
                                 strides=(shape_theta_x[1] // shape_g[1], shape_theta_x[2] // shape_g[2]),
                                 padding='same')(phi_g)  # 16

    concat_xg = layers.add([upsample_g, theta_x])
    act_xg = layers.Activation('relu')(concat_xg)
    psi = layers.Conv2D(1, (1, 1), padding='same')(act_xg)
    sigmoid_psi = layers.Activation('sigmoid')(psi)
    shape_sigmoid = tf.keras.backend.int_shape(sigmoid_psi)

    upsample_psi = layers.UpSampling2D(size=(shape_x[1] // shape_sigmoid[1], shape_x[2] // shape_sigmoid[2]))(sigmoid_psi)  # 32

    y = layers.multiply([upsample_psi, x])

    result = layers.Conv2D(shape_x[3], (1, 1), padding='same')(y)
    result_bn = layers.BatchNormalization()(result)
    return result_bn

def get_improved_unet(input_shape=(256, 256, 3)):
    """
    Improved U-Net with pre-trained MobileNetV2 encoder and Spatial Attention gates.
    """
    inputs = layers.Input(shape=input_shape)

    # --- Pre-trained Encoder (MobileNetV2) ---
    # We use a lightweight encoder for real-time inference
    base_model = tf.keras.applications.MobileNetV2(input_tensor=inputs, include_top=False, weights='imagenet')
    
    # Layer names for skip connections:
    # block_1_expand_relu  (128, 128, 96)
    # block_3_expand_relu  (64, 64, 144)
    # block_6_expand_relu  (32, 32, 192)
    # block_13_expand_relu (16, 16, 576)
    # out                  (8, 8, 1280)
    
    skip_names = [
        "block_1_expand_relu",   # 128x128
        "block_3_expand_relu",   # 64x64
        "block_6_expand_relu",   # 32x32
        "block_13_expand_relu",  # 16x16
    ]
    skips = [base_model.get_layer(name).output for name in skip_names]
    
    # Bottleneck
    x = base_model.output # 8x8x1280

    # --- Decoder with Attention ---
    decoder_filters = [512, 256, 128, 64]
    
    for i in range(len(decoder_filters)):
        # 1. Upsample
        x = layers.UpSampling2D((2, 2), interpolation='bilinear')(x)
        
        # 2. Attention Gate (Skip connection refinement)
        skip = skips[-(i+1)]
        # inter_shape for attention: half of skip channels
        att = attention_block(skip, x, tf.keras.backend.int_shape(skip)[3] // 2)
        
        # 3. Concatenate
        x = layers.Concatenate()([x, att])
        
        # 4. Conv Block
        x = layers.Conv2D(decoder_filters[i], (3, 3), padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.Conv2D(decoder_filters[i], (3, 3), padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)

    # Last Upsample to reach original size (256x256)
    x = layers.UpSampling2D((2, 2), interpolation='bilinear')(x)
    
    # Final Layer
    outputs = layers.Conv2D(1, (1, 1), activation='sigmoid')(x)

    model = models.Model(inputs, outputs, name="Attention_UNet_MobileNetV2")
    return model

if __name__ == "__main__":
    model = get_improved_unet()
    model.summary()
