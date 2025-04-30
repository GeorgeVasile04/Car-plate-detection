"""
Model architectures for license plate detection.
"""

import tensorflow as tf
from tensorflow.keras import layers, models, applications


def create_license_plate_detector(input_shape=(224, 224, 3)):
    """
    Create a basic CNN-based license plate detector.
    
    Args:
        input_shape: Input image shape (height, width, channels)
        
    Returns:
        tf.keras.Model: A compiled license plate detector model
    """
    # Input layer
    inputs = layers.Input(shape=input_shape)
    
    # Feature extraction
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    
    # Bounding box regression
    x = layers.Flatten()(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(128, activation='relu')(x)
    outputs = layers.Dense(4, activation='sigmoid')(x)  # x, y, w, h (normalized)
    
    # Create model
    model = models.Model(inputs=inputs, outputs=outputs)
    
    return model


def create_enhanced_license_plate_detector(input_shape=(224, 224, 3)):
    """
    Create an enhanced CNN-based license plate detector with residual connections.
    
    Args:
        input_shape: Input image shape (height, width, channels)
        
    Returns:
        tf.keras.Model: A compiled license plate detector model
    """
    # Input layer
    inputs = layers.Input(shape=input_shape)
    
    # Feature extraction with residual connections
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    block_1 = layers.MaxPooling2D((2, 2))(x)
    
    # Block 2
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(block_1)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Add()([x, layers.Conv2D(64, (1, 1), padding='same')(block_1)])  # Residual connection
    block_2 = layers.MaxPooling2D((2, 2))(x)
    
    # Block 3
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(block_2)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Add()([x, layers.Conv2D(128, (1, 1), padding='same')(block_2)])  # Residual connection
    block_3 = layers.MaxPooling2D((2, 2))(x)
    
    # Block 4
    x = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(block_3)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Add()([x, layers.Conv2D(256, (1, 1), padding='same')(block_3)])  # Residual connection
    x = layers.MaxPooling2D((2, 2))(x)
    
    # Global context
    global_avg = layers.GlobalAveragePooling2D()(x)
    
    # Bounding box regression
    x = layers.Flatten()(x)
    x = layers.Concatenate()([x, global_avg])  # Add global context
    x = layers.Dense(512, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(128, activation='relu')(x)
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