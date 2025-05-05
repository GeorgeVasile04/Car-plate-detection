"""
Model architectures for license plate detection.
"""

import tensorflow as tf
from tensorflow.keras import layers, models, applications


def create_enhanced_license_plate_detector(input_shape=(224, 224, 3)):
    """
    Create an enhanced CNN-based license plate detector with advanced architecture.
    Optimized to ~12M parameters for improved feature extraction and detection.
    Uses wider and deeper networks with multi-scale feature extraction.
    
    Args:
        input_shape: Input image shape (height, width, channels)
        
    Returns:
        tf.keras.Model: A compiled license plate detector model
    """
    # Input layer
    inputs = layers.Input(shape=input_shape)
      # Feature extraction with residual connections and wider layers
    # Block 1
    x = layers.Conv2D(32, (7, 7), activation='swish', padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(32, (5, 5), activation='swish', padding='same')(x)
    x = layers.BatchNormalization()(x)
    block_1 = layers.MaxPooling2D((2, 2))(x)
    
    # Block 2
    x = layers.Conv2D(64, (3, 3), activation='swish', padding='same')(block_1)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(64, (3, 3), activation='swish', padding='same')(x)
    x = layers.BatchNormalization()(x)
    # Use larger kernels for enhanced feature extraction
    x2 = layers.Conv2D(64, (5, 5), activation='swish', padding='same')(block_1)
    x = layers.Add()([x, x2, layers.Conv2D(64, (1, 1), padding='same')(block_1)])  # Residual connection    
    block_2 = layers.MaxPooling2D((2, 2))(x)
    
    # Block 3 - Filter count set to 128 (closest to specified 126)
    x = layers.Conv2D(128, (3, 3), activation='swish', padding='same')(block_2)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(128, (3, 3), activation='swish', padding='same')(x)
    x = layers.BatchNormalization()(x)
    # Dilated convolution branch for larger receptive field
    x_dilated = layers.Conv2D(128, (3, 3), dilation_rate=2, activation='swish', padding='same')(block_2)
    x_dilated = layers.BatchNormalization()(x_dilated)
    x = layers.Add()([x, x_dilated, layers.Conv2D(128, (1, 1), padding='same')(block_2)])    
    block_3 = layers.MaxPooling2D((2, 2))(x)
    
    # Block 4 - Set to 256 filters
    x = layers.Conv2D(256, (3, 3), activation='swish', padding='same')(block_3)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(256, (3, 3), activation='swish', padding='same')(x)
    x = layers.BatchNormalization()(x)
    # Attention mechanism - channel attention
    se = layers.GlobalAveragePooling2D()(x)
    se = layers.Dense(256 // 4, activation='swish')(se)
    se = layers.Dense(256, activation='sigmoid')(se)
    se = layers.Reshape((1, 1, 256))(se)
    x_att = layers.multiply([x, se])
    x = layers.Add()([x_att, layers.Conv2D(256, (1, 1), padding='same')(block_3)])  # Residual connection
    block_4 = layers.MaxPooling2D((2, 2))(x)
    
      # Block 5 - Deeper network with more capacity
    x = layers.Conv2D(768, (3, 3), activation='swish', padding='same')(block_4)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(768, (3, 3), activation='swish', padding='same')(x)
    x = layers.BatchNormalization()(x)
    # Spatial attention
    x_spatial = layers.Conv2D(768, (1, 1), activation='swish')(block_4)
    x_spatial = layers.BatchNormalization()(x_spatial)
    x = layers.Add()([x, x_spatial, layers.Conv2D(768, (1, 1), padding='same')(block_4)])
    block_5 = layers.MaxPooling2D((2, 2))(x)
      # Multi-scale feature fusion - combine features from different blocks
    # Upsample and add features from block 5
    up_b5 = layers.Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(block_5)
    
    # Global features
    global_features = layers.GlobalAveragePooling2D()(block_5)
    global_features = layers.Dense(512, activation='swish')(global_features)
    
    # Use more comprehensive spatial feature extraction
    spatial_features = layers.Conv2D(256, (1, 1), activation='swish')(block_5)
    spatial_features = layers.Flatten()(spatial_features)
    
    # Use features from block 4 as well
    block4_features = layers.GlobalAveragePooling2D()(block_4)
    block4_features = layers.Dense(256, activation='swish')(block4_features)
      # Combine global and spatial features
    x = layers.Concatenate()([global_features, spatial_features, block4_features])
    
    # Bounding box regression - Dense layers aligned with specified filter counts
    x = layers.Dense(512, activation='swish')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(256, activation='swish')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(128, activation='swish')(x)
    outputs = layers.Dense(4, activation='sigmoid')(x)  # x, y, w, h (normalized)
    
    # Create model
    model = models.Model(inputs=inputs, outputs=outputs)
    
    return model


def create_mobilenet_license_plate_detector(input_shape=(224, 224, 3), weights='imagenet'):
    """
    Create a license plate detector based on MobileNetV2.
    
    Args:
        input_shape: Input image shape (height, width, channels)
        weights: Pre-trained weights ('imagenet' or None)
        
    Returns:
        tf.keras.Model: A compiled license plate detector model
    """
    # Base model
    base_model = applications.MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights=weights
    )
    
    # Freeze early layers for transfer learning
    for layer in base_model.layers[:100]:
        layer.trainable = False
    
    # Feature extraction
    inputs = layers.Input(shape=input_shape)
    x = base_model(inputs, training=False)
    
    # Bounding box regression
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(128, activation='relu')(x)
    outputs = layers.Dense(4, activation='sigmoid')(x)  # x, y, w, h (normalized)
    
    # Create model
    model = models.Model(inputs=inputs, outputs=outputs)
    
    return model


def create_efficientnet_license_plate_detector(input_shape=(224, 224, 3), version='B0', weights='imagenet'):
    """
    Create a license plate detector based on EfficientNet.
    
    Args:
        input_shape: Input image shape (height, width, channels)
        version: EfficientNet version (B0, B1, B2, etc.)
        weights: Pre-trained weights ('imagenet' or None)
        
    Returns:
        tf.keras.Model: A compiled license plate detector model
    """
    # Select the appropriate EfficientNet model based on version
    if version == 'B0':
        base_model = applications.EfficientNetB0(
            input_shape=input_shape,
            include_top=False,
            weights=weights
        )
    elif version == 'B1':
        base_model = applications.EfficientNetB1(
            input_shape=input_shape,
            include_top=False,
            weights=weights
        )
    elif version == 'B2':
        base_model = applications.EfficientNetB2(
            input_shape=input_shape,
            include_top=False,
            weights=weights
        )
    elif version == 'B3':
        base_model = applications.EfficientNetB3(
            input_shape=input_shape,
            include_top=False,
            weights=weights
        )
    else:  # Default to B0
        base_model = applications.EfficientNetB0(
            input_shape=input_shape,
            include_top=False,
            weights=weights
        )
    
    # Freeze early layers for transfer learning
    freeze_percentage = 0.7  # Freeze 70% of base model layers
    num_layers = len(base_model.layers)
    for layer in base_model.layers[:int(num_layers * freeze_percentage)]:
        layer.trainable = False
    
    # Feature extraction
    inputs = layers.Input(shape=input_shape)
    x = base_model(inputs, training=False)
    
    # Bounding box regression
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(128, activation='relu')(x)
    outputs = layers.Dense(4, activation='sigmoid')(x)  # x, y, w, h (normalized)
    
    # Create model
    model = models.Model(inputs=inputs, outputs=outputs)
    
    return model