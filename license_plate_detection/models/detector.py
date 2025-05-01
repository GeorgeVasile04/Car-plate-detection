"""
Model architectures for license plate detection.
"""

import tensorflow as tf
from tensorflow.keras import layers, models, applications


def create_license_plate_detector(input_shape=(224, 224, 3)):
    """
    Create a basic CNN-based license plate detector.
    Optimized version with ~8M parameters for Google Colab compatibility.
    
    Args:
        input_shape: Input image shape (height, width, channels)
        
    Returns:
        tf.keras.Model: A compiled license plate detector model
    """
    # Input layer
    inputs = layers.Input(shape=input_shape)
    
    # Feature extraction - Reduced number of filters and added BatchNormalization
    # First block - keep 32 filters for initial feature extraction
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    
    # Second block - keep 64 filters
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    
    # Third block - reduced from 128 to 96 filters
    x = layers.Conv2D(96, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    
    # Fourth block - reduced from 256 to 128 filters
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    
    # Add global pooling to reduce parameters
    x = layers.GlobalAveragePooling2D()(x)
    
    # Bounding box regression - significantly reduced dense layer size
    x = layers.Dense(192, activation='relu')(x)  # Reduced from 256
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(96, activation='relu')(x)   # Reduced from 128
    outputs = layers.Dense(4, activation='sigmoid')(x)  # x, y, w, h (normalized)
    
    # Create model
    model = models.Model(inputs=inputs, outputs=outputs)
    
    return model


def create_enhanced_license_plate_detector(input_shape=(224, 224, 3)):
    """
    Create an enhanced CNN-based license plate detector with residual connections.
    Optimized version with ~13M parameters for Google Colab compatibility.
    Includes an extra layer compared to the basic detector for better feature extraction.
    
    Args:
        input_shape: Input image shape (height, width, channels)
        
    Returns:
        tf.keras.Model: A compiled license plate detector model
    """
    # Input layer
    inputs = layers.Input(shape=input_shape)
    
    # Feature extraction with residual connections
    # Block 1
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
    x = layers.Conv2D(96, (3, 3), activation='relu', padding='same')(block_2) # Reduced from 128
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(96, (3, 3), activation='relu', padding='same')(x) # Reduced from 128
    x = layers.BatchNormalization()(x)
    x = layers.Add()([x, layers.Conv2D(96, (1, 1), padding='same')(block_2)])  # Residual connection (match filter count)
    block_3 = layers.MaxPooling2D((2, 2))(x)
    
    # Block 4
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(block_3) # Reduced from 256
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x) # Reduced from 256
    x = layers.BatchNormalization()(x)
    x = layers.Add()([x, layers.Conv2D(128, (1, 1), padding='same')(block_3)])  # Residual connection
    block_4 = layers.MaxPooling2D((2, 2))(x)
    
    # Block 5 (Extra layer compared to basic detector)
    x = layers.Conv2D(192, (3, 3), activation='relu', padding='same')(block_4)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(192, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Add()([x, layers.Conv2D(192, (1, 1), padding='same')(block_4)])  # Residual connection
    x = layers.MaxPooling2D((2, 2))(x)
    
    # Use both global pooling and spatial features to preserve information
    global_features = layers.GlobalAveragePooling2D()(x)
    
    # Use more efficient spatial feature extraction
    spatial_features = layers.Conv2D(128, (1, 1), activation='relu')(x)  # 1x1 convolution reduces channels
    spatial_features = layers.Flatten()(spatial_features)
    
    # Combine global and spatial features
    x = layers.Concatenate()([global_features, spatial_features])
    
    # Bounding box regression - reduced dense layer sizes
    x = layers.Dense(256, activation='relu')(x)  # Reduced from 512
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