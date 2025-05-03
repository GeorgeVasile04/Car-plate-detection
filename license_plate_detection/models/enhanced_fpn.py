"""
Enhanced FPN architecture specifically designed for license plate detection.
Features bidirectional feature fusion, attention mechanisms, and scale-aware components.
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def create_bifpn_license_plate_detector(input_shape=(320, 320, 3)):
    """
    Creates an enhanced license plate detector with Bidirectional Feature Pyramid Network,
    scale-aware detection heads, and advanced attention mechanisms.
    
    Args:
        input_shape: Input image shape (height, width, channels)
        
    Returns:
        tf.keras.Model: A compilable license plate detector model
    """
    # Input layer
    inputs = keras.layers.Input(shape=input_shape)
    
    # Initial feature extraction with larger receptive field
    x = keras.layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same', name='conv1')(inputs)
    x = keras.layers.BatchNormalization(name='bn1')(x)
    x = keras.layers.Activation('swish', name='swish1')(x)
    x = keras.layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same', name='pool1')(x)
    
    # Track feature maps for bidirectional FPN
    feature_maps = []
    
    # Block 1: First Block with Channel Attention
    block1_out = create_attention_block(x, filters=128, block_id=1)
    feature_maps.append(block1_out)  # P2 feature map (1/4 resolution)
    
    # Block 2: Second Block with Dilated Convolutions
    x = keras.layers.MaxPooling2D((2, 2), name='block2_pool')(block1_out)
    block2_out = create_dual_path_block(x, filters=256, block_id=2)
    feature_maps.append(block2_out)  # P3 feature map (1/8 resolution)
    
    # Block 3: Third Block with Spatial Attention
    x = keras.layers.MaxPooling2D((2, 2), name='block3_pool')(block2_out)
    block3_out = create_attention_block(x, filters=512, block_id=3, attention_type='spatial')
    feature_maps.append(block3_out)  # P4 feature map (1/16 resolution)
    
    # Block 4: Deepest feature extraction
    x = keras.layers.MaxPooling2D((2, 2), name='block4_pool')(block3_out)
    block4_out = create_attention_block(x, filters=1024, block_id=4, attention_type='dual')
    feature_maps.append(block4_out)  # P5 feature map (1/32 resolution)
    
    # Bidirectional Feature Pyramid Network
    bifpn_features = create_bifpn(feature_maps, feature_size=256, name='bifpn')
    
    # Detection heads for multi-scale features
    detection_outputs = []
    
    # Create detection heads for each scale level
    for i, feature in enumerate(bifpn_features):
        # Create scale-specific detection head
        scale_level = i + 2  # P2, P3, P4, P5
        detection_head = create_detection_head(
            feature, 
            name=f'detection_head_P{scale_level}',
            scale_level=scale_level
        )
        detection_outputs.append(detection_head)
    
    # Global context features
    global_context = keras.layers.GlobalAveragePooling2D(name='global_context')(bifpn_features[-1])
    global_context = keras.layers.Dense(256, activation='swish', name='global_context_dense')(global_context)
    
    # Fuse detection outputs with global context
    # First, flatten and concatenate all detection outputs
    detection_features = []
    for i, output in enumerate(detection_outputs):
        # Use global average pooling to reduce spatial dimensions
        pooled = keras.layers.GlobalAveragePooling2D(name=f'pool_level{i}')(output)
        detection_features.append(pooled)
    
    # Concatenate all detection features
    if len(detection_features) > 1:
        fused_features = keras.layers.Concatenate(name='concat_detections')(detection_features)
    else:
        fused_features = detection_features[0]
    
    # Combine with global context
    combined_features = keras.layers.Concatenate(name='concat_global')([fused_features, global_context])
    
    # Final prediction layers
    x = keras.layers.Dense(512, activation='swish', name='final_dense1')(combined_features)
    x = keras.layers.BatchNormalization(name='final_bn1')(x)
    x = keras.layers.Dropout(0.3, name='final_dropout1')(x)
    
    x = keras.layers.Dense(256, activation='swish', name='final_dense2')(x)
    x = keras.layers.BatchNormalization(name='final_bn2')(x)
    x = keras.layers.Dropout(0.2, name='final_dropout2')(x)
    
    # Output layer - bounding box regression [x, y, width, height]
    outputs = keras.layers.Dense(4, activation='sigmoid', name='bbox_output')(x)
    
    # Create model
    model = keras.models.Model(inputs=inputs, outputs=outputs, name='bifpn_license_plate_detector')
    
    return model


def create_attention_block(inputs, filters, block_id, attention_type='channel'):
    """
    Creates an attention-enhanced convolutional block.
    
    Args:
        inputs: Input tensor
        filters: Number of filters
        block_id: Block identifier for naming
        attention_type: Type of attention ('channel', 'spatial', or 'dual')
        
    Returns:
        Output tensor
    """
    # Store input for residual connection
    residual = inputs
    
    # Main path
    x = keras.layers.Conv2D(filters, (3, 3), padding='same', name=f'block{block_id}_conv1')(inputs)
    x = keras.layers.BatchNormalization(name=f'block{block_id}_bn1')(x)
    x = keras.layers.Activation('swish', name=f'block{block_id}_swish1')(x)
    
    x = keras.layers.Conv2D(filters, (3, 3), padding='same', name=f'block{block_id}_conv2')(x)
    x = keras.layers.BatchNormalization(name=f'block{block_id}_bn2')(x)
    
    # Apply attention mechanism
    if attention_type == 'channel' or attention_type == 'dual':
        # Channel attention (squeeze and excitation)
        ca = keras.layers.GlobalAveragePooling2D(name=f'block{block_id}_gap')(x)
        ca = keras.layers.Reshape((1, 1, filters), name=f'block{block_id}_reshape')(ca)
        ca = keras.layers.Conv2D(filters // 4, (1, 1), activation='swish', name=f'block{block_id}_ca_conv1')(ca)
        ca = keras.layers.Conv2D(filters, (1, 1), activation='sigmoid', name=f'block{block_id}_ca_conv2')(ca)
        x = keras.layers.Multiply(name=f'block{block_id}_ca_mul')([x, ca])
    
    if attention_type == 'spatial' or attention_type == 'dual':
        # Spatial attention
        sa_avg = keras.layers.Lambda(lambda x: keras.backend.mean(x, axis=-1, keepdims=True),
                                    name=f'block{block_id}_sa_avg')(x)
        sa_max = keras.layers.Lambda(lambda x: keras.backend.max(x, axis=-1, keepdims=True),
                                    name=f'block{block_id}_sa_max')(x)
        sa = keras.layers.Concatenate(name=f'block{block_id}_sa_concat')([sa_avg, sa_max])
        sa = keras.layers.Conv2D(1, (7, 7), padding='same', activation='sigmoid',
                                name=f'block{block_id}_sa_conv')(sa)
        x = keras.layers.Multiply(name=f'block{block_id}_sa_mul')([x, sa])
    
    # Adjust residual connection if needed
    if residual.shape[-1] != x.shape[-1]:
        residual = keras.layers.Conv2D(filters, (1, 1), name=f'block{block_id}_res_conv')(residual)
        residual = keras.layers.BatchNormalization(name=f'block{block_id}_res_bn')(residual)
    
    # Add residual connection
    x = keras.layers.Add(name=f'block{block_id}_add')([x, residual])
    x = keras.layers.Activation('swish', name=f'block{block_id}_output')(x)
    
    return x


def create_dual_path_block(inputs, filters, block_id):
    """
    Creates a dual-path convolutional block with different receptive fields.
    
    Args:
        inputs: Input tensor
        filters: Number of filters
        block_id: Block identifier for naming
        
    Returns:
        Output tensor
    """
    residual = inputs
    
    # Path 1: Standard convolutions
    path1 = keras.layers.Conv2D(filters // 2, (3, 3), padding='same', name=f'block{block_id}_p1_conv1')(inputs)
    path1 = keras.layers.BatchNormalization(name=f'block{block_id}_p1_bn1')(path1)
    path1 = keras.layers.Activation('swish', name=f'block{block_id}_p1_swish1')(path1)
    path1 = keras.layers.Conv2D(filters // 2, (3, 3), padding='same', name=f'block{block_id}_p1_conv2')(path1)
    path1 = keras.layers.BatchNormalization(name=f'block{block_id}_p1_bn2')(path1)
    
    # Path 2: Dilated convolutions for larger receptive field
    path2 = keras.layers.Conv2D(filters // 2, (3, 3), dilation_rate=(2, 2), padding='same', 
                              name=f'block{block_id}_p2_conv1')(inputs)
    path2 = keras.layers.BatchNormalization(name=f'block{block_id}_p2_bn1')(path2)
    path2 = keras.layers.Activation('swish', name=f'block{block_id}_p2_swish1')(path2)
    path2 = keras.layers.Conv2D(filters // 2, (3, 3), dilation_rate=(2, 2), padding='same', 
                              name=f'block{block_id}_p2_conv2')(path2)
    path2 = keras.layers.BatchNormalization(name=f'block{block_id}_p2_bn2')(path2)
    
    # Concatenate paths
    x = keras.layers.Concatenate(name=f'block{block_id}_concat')([path1, path2])
    
    # 1x1 Conv to adjust channel dimensions
    x = keras.layers.Conv2D(filters, (1, 1), padding='same', name=f'block{block_id}_fusion_conv')(x)
    x = keras.layers.BatchNormalization(name=f'block{block_id}_fusion_bn')(x)
    
    # Adjust residual connection if needed
    if residual.shape[-1] != x.shape[-1]:
        residual = keras.layers.Conv2D(filters, (1, 1), name=f'block{block_id}_res_conv')(residual)
        residual = keras.layers.BatchNormalization(name=f'block{block_id}_res_bn')(residual)
    
    # Add residual connection
    x = keras.layers.Add(name=f'block{block_id}_add')([x, residual])
    x = keras.layers.Activation('swish', name=f'block{block_id}_output')(x)
    
    return x


def create_bifpn(features, feature_size=256, name="bifpn"):
    """
    Creates a Bidirectional Feature Pyramid Network for enhanced multi-scale feature fusion.
    
    Args:
        features: List of features at different scales [P2, P3, P4, P5]
        feature_size: Number of channels for all feature maps
        name: Name prefix for layers
        
    Returns:
        List of enhanced feature maps [P2, P3, P4, P5]
    """
    # Ensure all feature maps have the same channel dimensions
    adjusted_features = []
    for level, feature in enumerate(features):
        adjusted_features.append(
            keras.layers.Conv2D(feature_size, (1, 1), padding='same', 
                          name=f'{name}_adjust_P{level+2}')(feature)
        )
    
    # Top-down pathway (from deep to shallow layers)
    top_down_features = [adjusted_features[-1]]  # Start with deepest feature (P5)
    
    # Process from second deepest to shallowest (P4 to P2)
    for level in range(len(adjusted_features) - 2, -1, -1):
        # Upsample deeper feature
        td_feature = keras.layers.UpSampling2D(2, name=f'{name}_up_P{level+3}')(top_down_features[-1])
        
        # Add connection from adjusted feature at current level
        fusion = keras.layers.Add(name=f'{name}_td_add_P{level+2}')([adjusted_features[level], td_feature])
        
        # Apply 3x3 conv for better feature fusion
        fusion = keras.layers.Conv2D(feature_size, (3, 3), padding='same', 
                               name=f'{name}_td_conv_P{level+2}')(fusion)
        fusion = keras.layers.BatchNormalization(name=f'{name}_td_bn_P{level+2}')(fusion)
        fusion = keras.layers.Activation('swish', name=f'{name}_td_swish_P{level+2}')(fusion)
        
        top_down_features.append(fusion)
    
    # Reverse to get P2 to P5 order (currently it's P5, P4, P3, P2)
    top_down_features = top_down_features[::-1]
    
    # Bottom-up pathway (from shallow to deep layers)
    bottom_up_features = [top_down_features[0]]  # Start with P2
    
    # Process from second shallowest to deepest (P3 to P5)
    for level in range(1, len(top_down_features)):
        # Downsample shallower feature
        bu_feature = keras.layers.MaxPooling2D((2, 2), name=f'{name}_down_P{level+1}')(bottom_up_features[-1])
        
        # Add connection from top-down pathway
        fusion = keras.layers.Add(name=f'{name}_bu_add_P{level+2}')([top_down_features[level], bu_feature])
        
        # Apply 3x3 conv for better feature fusion
        fusion = keras.layers.Conv2D(feature_size, (3, 3), padding='same', 
                               name=f'{name}_bu_conv_P{level+2}')(fusion)
        fusion = keras.layers.BatchNormalization(name=f'{name}_bu_bn_P{level+2}')(fusion)
        fusion = keras.layers.Activation('swish', name=f'{name}_bu_swish_P{level+2}')(fusion)
        
        bottom_up_features.append(fusion)
    
    # Apply final 3x3 convolution on each feature map for further refinement
    final_features = []
    for level, feature in enumerate(bottom_up_features):
        refined = keras.layers.Conv2D(feature_size, (3, 3), padding='same', 
                                 name=f'{name}_final_conv_P{level+2}')(feature)
        refined = keras.layers.BatchNormalization(name=f'{name}_final_bn_P{level+2}')(refined)
        refined = keras.layers.Activation('swish', name=f'{name}_final_swish_P{level+2}')(refined)
        final_features.append(refined)
    
    return final_features


def create_detection_head(feature, name, scale_level):
    """
    Creates a scale-aware detection head optimized for license plate detection.
    
    Args:
        feature: Input feature map
        name: Name prefix for layers
        scale_level: Feature map scale level (2-5, corresponds to P2-P5)
        
    Returns:
        Output tensor with detection features
    """
    # Config based on scale level - larger receptive fields for deeper levels
    if scale_level <= 3:  # P2, P3 (better for small objects)
        kernel_size = 3
        num_convs = 4
        dilation_rate = 1
    else:  # P4, P5 (better for larger objects)
        kernel_size = 5  
        num_convs = 3
        dilation_rate = 2
    
    # Sequential convolutional blocks
    x = feature
    for i in range(num_convs):
        x = keras.layers.Conv2D(
            256, 
            kernel_size=kernel_size, 
            padding='same',
            dilation_rate=dilation_rate if i == 1 else 1,  # Apply dilation in middle layer
            name=f'{name}_conv{i+1}'
        )(x)
        x = keras.layers.BatchNormalization(name=f'{name}_bn{i+1}')(x)
        x = keras.layers.Activation('swish', name=f'{name}_swish{i+1}')(x)
    
    return x


def enable_gradient_checkpointing(model):
    """
    Enables gradient checkpointing to reduce memory usage during training.
    
    Args:
        model: Keras model
        
    Returns:
        Model with gradient checkpointing enabled
    """
    if hasattr(keras.backend, 'tf') and hasattr(keras.backend.tf, 'recompute_grad'):
        # Define which layers to apply checkpointing to (typically convolutional layers)
        ckpt_layers = []
        for layer in model.layers:
            if isinstance(layer, keras.layers.Conv2D) and layer.trainable:
                ckpt_layers.append(layer)
        
        # Apply gradient checkpointing to these layers
        for layer in ckpt_layers:
            layer.call = keras.backend.tf.recompute_grad(layer.call)
        
    return model
