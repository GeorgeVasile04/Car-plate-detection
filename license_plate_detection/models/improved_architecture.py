"""
This module implements improved architectures for license plate detection
with enhanced multi-scale feature extraction and attention mechanisms.
"""

import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np


def create_bifpn_block(features, feature_size=256, name="bifpn"):
    """
    Creates a Bidirectional Feature Pyramid Network (BiFPN) block
    which enhances information flow compared to standard FPN.
    
    Args:
        features: List of input feature maps at different scales
        feature_size: Channel dimension for feature maps
        name: Prefix for layer names
        
    Returns:
        List of enhanced feature maps
    """
    # Use weighted feature fusion for better training stability
    # Generate weights for feature fusion
    def _fusion_weight(num_inputs):
        w_init = tf.constant_initializer(1.0 / num_inputs)
        return tf.Variable(w_init(shape=[num_inputs]), name=f"{name}_fusion_w", trainable=True)
    
    # Number of feature maps
    num_levels = len(features)
    output_features = []
    
    # Adjust channel dimensions for all input features
    adjusted_features = []
    for level, feature in enumerate(features):
        adjusted_features.append(
            layers.Conv2D(feature_size, 1, 1, padding='same', 
                          name=f'{name}_adjust_level{level}')(feature)
        )
    
    # Top-down pathway (from deep to shallow layers)
    top_down_features = [adjusted_features[-1]]  # Start with deepest feature
    
    # Process from second deepest to shallowest
    for level in range(num_levels - 2, -1, -1):
        # Upsample deeper feature
        td_feature = layers.UpSampling2D(2, name=f'{name}_up_level{level}')(top_down_features[-1])
        
        # Add connection from adjusted feature at current level
        w = _fusion_weight(2)
        td_feature = layers.Add(name=f'{name}_td_add_level{level}')(
            [td_feature * w[0], adjusted_features[level] * w[1]]
        )
        td_feature = layers.Activation('swish')(td_feature)
        td_feature = layers.Conv2D(feature_size, 3, 1, padding='same',
                                  name=f'{name}_td_conv_level{level}')(td_feature)
        td_feature = layers.BatchNormalization(name=f'{name}_td_bn_level{level}')(td_feature)
        td_feature = layers.Activation('swish', name=f'{name}_td_act_level{level}')(td_feature)
        
        top_down_features.append(td_feature)
    
    # Reverse for bottom-up path
    top_down_features = top_down_features[::-1]
    
    # Bottom-up pathway (from shallow to deep)
    bottom_up_features = [top_down_features[0]]  # Start with shallowest feature
    
    # Process from second shallowest to deepest
    for level in range(1, num_levels):
        # Downsample shallower feature
        bu_feature = layers.MaxPooling2D(2, name=f'{name}_down_level{level}')(bottom_up_features[-1])
        
        # Add connection from top-down feature at current level
        w = _fusion_weight(2)
        bu_feature = layers.Add(name=f'{name}_bu_add_level{level}')(
            [bu_feature * w[0], top_down_features[level] * w[1]]
        )
        bu_feature = layers.Activation('swish')(bu_feature)
        bu_feature = layers.Conv2D(feature_size, 3, 1, padding='same',
                                  name=f'{name}_bu_conv_level{level}')(bu_feature)
        bu_feature = layers.BatchNormalization(name=f'{name}_bu_bn_level{level}')(bu_feature)
        bu_feature = layers.Activation('swish', name=f'{name}_bu_act_level{level}')(bu_feature)
        
        bottom_up_features.append(bu_feature)
    
    return bottom_up_features


def coordconv_layer(inputs, name='coordconv'):
    """
    Adds coordinate channels (x, y) to input feature maps.
    This helps with position-dependent learning.
    
    Args:
        inputs: Input tensor
        name: Prefix for layer names
        
    Returns:
        Feature maps with added coordinate channels
    """
    batch_size = tf.shape(inputs)[0]
    height = tf.shape(inputs)[1]
    width = tf.shape(inputs)[2]
    
    # Generate normalized coordinates
    y_coords = tf.linspace(-1.0, 1.0, height)
    y_coords = tf.expand_dims(y_coords, 1)
    y_coords = tf.tile(y_coords, [1, width])
    y_coords = tf.expand_dims(y_coords, 0)
    y_coords = tf.expand_dims(y_coords, -1)
    y_coords = tf.tile(y_coords, [batch_size, 1, 1, 1])
    
    x_coords = tf.linspace(-1.0, 1.0, width)
    x_coords = tf.expand_dims(x_coords, 0)
    x_coords = tf.tile(x_coords, [height, 1])
    x_coords = tf.expand_dims(x_coords, 0)
    x_coords = tf.expand_dims(x_coords, -1)
    x_coords = tf.tile(x_coords, [batch_size, 1, 1, 1])
    
    # Concatenate coordinates with input
    coords = tf.concat([x_coords, y_coords], axis=-1)
    outputs = tf.concat([inputs, coords], axis=-1)
    
    # Project back to original depth with 1x1 convolution
    outputs = layers.Conv2D(
        inputs.shape[-1], kernel_size=1, padding='same',
        kernel_initializer='he_normal', name=f'{name}_proj')(outputs)
    
    return outputs


def cbam_attention(inputs, reduction_ratio=16, kernel_size=7, name="cbam"):
    """
    Convolutional Block Attention Module (CBAM)
    Combines channel and spatial attention for improved feature refinement.
    
    Args:
        inputs: Input tensor
        reduction_ratio: Reduction ratio for the channel attention
        kernel_size: Size of the kernel for spatial attention
        name: Prefix for layer names
        
    Returns:
        Feature maps with applied attention
    """
    # Channel Attention
    avg_pool = layers.GlobalAveragePooling2D(name=f'{name}_channel_avg_pool')(inputs)
    avg_pool = layers.Reshape((1, 1, inputs.shape[-1]))(avg_pool)
    
    max_pool = layers.GlobalMaxPooling2D(name=f'{name}_channel_max_pool')(inputs)
    max_pool = layers.Reshape((1, 1, inputs.shape[-1]))(max_pool)
    
    # Shared MLP
    shared_mlp_1 = layers.Dense(
        inputs.shape[-1] // reduction_ratio,
        activation='relu',
        kernel_initializer='he_normal',
        name=f'{name}_channel_mlp_1'
    )
    shared_mlp_2 = layers.Dense(
        inputs.shape[-1],
        kernel_initializer='he_normal',
        name=f'{name}_channel_mlp_2'
    )
    
    avg_pool = shared_mlp_2(shared_mlp_1(avg_pool))
    max_pool = shared_mlp_2(shared_mlp_1(max_pool))
    
    channel_attention = layers.Add()([avg_pool, max_pool])
    channel_attention = layers.Activation('sigmoid')(channel_attention)
    
    # Apply channel attention
    refined_features = layers.Multiply()([inputs, channel_attention])
    
    # Spatial Attention
    avg_pool_spatial = layers.Lambda(lambda x: tf.reduce_mean(x, axis=-1, keepdims=True))(refined_features)
    max_pool_spatial = layers.Lambda(lambda x: tf.reduce_max(x, axis=-1, keepdims=True))(refined_features)
    
    spatial_features = layers.Concatenate()([avg_pool_spatial, max_pool_spatial])
    spatial_attention = layers.Conv2D(
        1, kernel_size, padding='same', 
        kernel_initializer='he_normal',
        name=f'{name}_spatial_conv')(spatial_features)
    spatial_attention = layers.Activation('sigmoid')(spatial_attention)
    
    # Apply spatial attention
    output = layers.Multiply()([refined_features, spatial_attention])
    
    return output


def improved_residual_block(x, filters, kernel_size=3, strides=1, dilation_rate=1, dropout_rate=0.0, name=None):
    """
    Improved residual block with scale calibration and normalization.
    
    Args:
        x: Input tensor
        filters: Number of filters in the convolution layers
        kernel_size: Size of the kernel for convolutions
        strides: Stride length for convolutions
        dilation_rate: Dilation rate for convolutions
        dropout_rate: Dropout rate for regularization
        name: Prefix for layer names
        
    Returns:
        Output tensor after applying the residual block
    """
    shortcut = x
    
    # First convolution block with pre-activation pattern
    x = layers.BatchNormalization(name=f'{name}_bn1')(x)
    x = layers.Activation('swish', name=f'{name}_act1')(x)
    x = layers.Conv2D(
        filters, kernel_size, strides=strides, padding='same',
        dilation_rate=dilation_rate, name=f'{name}_conv1')(x)
        
    # Dropout for regularization
    if dropout_rate > 0:
        x = layers.Dropout(dropout_rate, name=f'{name}_dropout')(x)
    
    # Second convolution block
    x = layers.BatchNormalization(name=f'{name}_bn2')(x)
    x = layers.Activation('swish', name=f'{name}_act2')(x)
    x = layers.Conv2D(
        filters, kernel_size, padding='same', name=f'{name}_conv2')(x)
    
    # Adjust shortcut if shapes don't match
    if shortcut.shape[-1] != filters or strides > 1:
        shortcut = layers.Conv2D(filters, 1, strides=strides, padding='same', name=f'{name}_shortcut')(shortcut)
    
    # Add shortcut connection with scale calibration
    scale = tf.Variable(1.0, trainable=True, name=f'{name}_scale')
    x = layers.Add(name=f'{name}_add')([shortcut, x * scale])
    
    return x


def spatial_pyramid_pooling(x, pool_sizes=[1, 2, 3, 6], name='spp'):
    """
    Spatial Pyramid Pooling module to capture multi-scale context.
    
    Args:
        x: Input tensor
        pool_sizes: List of pool bin sizes
        name: Prefix for layer names
        
    Returns:
        Output tensor after spatial pyramid pooling
    """
    # Get dimensions
    shape = x.get_shape().as_list()
    h, w = shape[1], shape[2]
    
    # Apply pooling at different scales
    pooled_outputs = [x]
    for i, pool_size in enumerate(pool_sizes):
        if pool_size == 1:  # Global pooling
            pool = layers.GlobalAveragePooling2D(name=f'{name}_global_pool')(x)
            pool = layers.Reshape((1, 1, shape[-1]))(pool)
            pool = layers.UpSampling2D(size=(h, w), interpolation='bilinear', 
                                     name=f'{name}_up_global')(pool)
        else:
            # Calculate pool dimensions
            stride_h = h // pool_size
            stride_w = w // pool_size
            pool = layers.AveragePooling2D((stride_h, stride_w), strides=(stride_h, stride_w),
                                         name=f'{name}_pool_{pool_size}')(x)
            pool = layers.UpSampling2D(size=(pool_size, pool_size), interpolation='bilinear',
                                     name=f'{name}_up_{pool_size}')(pool)
        
        pooled_outputs.append(pool)
    
    # Concatenate pooled features
    output = layers.Concatenate(name=f'{name}_concat')(pooled_outputs)
    
    # Project back to original depth with 1x1 convolution
    output = layers.Conv2D(shape[-1], 1, 1, padding='same', 
                         name=f'{name}_proj')(output)
    output = layers.BatchNormalization(name=f'{name}_bn')(output)
    output = layers.Activation('swish', name=f'{name}_act')(output)
    
    return output


def create_improved_license_plate_detector(input_shape=(224, 224, 3), high_res_input=False):
    """
    Creates an improved license plate detector with:
    - EfficientDet-like BiFPN architecture
    - CBAM attention mechanism
    - CoordConv for better spatial understanding
    - SPP for multi-scale context
    - Scale-calibrated residual blocks
    - Size-sensitive outputs
    
    Args:
        input_shape: Input image shape (height, width, channels)
        high_res_input: Whether to use high-resolution input processing branch
        
    Returns:
        tf.keras.Model: License plate detector model
    """
    inputs = layers.Input(shape=input_shape)
    
    # Add coordinate channels to input for better spatial context
    x = coordconv_layer(inputs, name='input_coordconv')
    
    # Optional high-resolution branch for small plates
    if high_res_input:
        x_high_res = layers.Conv2D(32, 3, 1, padding='same', name='high_res_conv1')(x)
        x_high_res = layers.BatchNormalization(name='high_res_bn1')(x_high_res)
        x_high_res = layers.Activation('swish', name='high_res_act1')(x_high_res)
        
        # More detail-preserving path
        x_high_res = improved_residual_block(x_high_res, 64, name='high_res_block1')
    
    # Initial stem
    x = layers.Conv2D(64, 7, strides=2, padding='same', name='stem_conv')(x)
    x = layers.BatchNormalization(name='stem_bn')(x)
    x = layers.Activation('swish', name='stem_act')(x)
    
    # First stage - keep resolution relatively high for small license plates
    x = improved_residual_block(x, 64, strides=1, name='stage1_block1')
    p1 = cbam_attention(x, name='stage1_cbam')
    
    # Second stage
    x = layers.MaxPooling2D(3, strides=2, padding='same', name='pool1')(p1)
    x = improved_residual_block(x, 128, name='stage2_block1')
    x = improved_residual_block(x, 128, name='stage2_block2')
    p2 = cbam_attention(x, name='stage2_cbam')
    
    # Third stage with dilated convolutions for larger receptive field
    x = layers.MaxPooling2D(3, strides=2, padding='same', name='pool2')(p2)
    x = improved_residual_block(x, 256, dilation_rate=1, name='stage3_block1')
    x = improved_residual_block(x, 256, dilation_rate=2, name='stage3_block2')
    p3 = cbam_attention(x, name='stage3_cbam')
    
    # Fourth stage with further dilation
    x = layers.MaxPooling2D(3, strides=2, padding='same', name='pool3')(p3)
    x = improved_residual_block(x, 512, dilation_rate=1, name='stage4_block1')
    x = improved_residual_block(x, 512, dilation_rate=4, name='stage4_block2')
    p4 = spatial_pyramid_pooling(x, name='stage4_spp')
    p4 = cbam_attention(p4, name='stage4_cbam')
    
    # Create feature pyramid with BiFPN (bidirectional)
    feature_maps = [p1, p2, p3, p4]
    bifpn_features = create_bifpn_block(feature_maps, feature_size=256, name='bifpn1')
    # Apply a second BiFPN block for deeper feature interaction
    bifpn_features = create_bifpn_block(bifpn_features, feature_size=256, name='bifpn2')
    
    # If using high-resolution branch, merge it with appropriate BiFPN level
    if high_res_input:
        # Match dimensions
        x_high_res = layers.Conv2D(256, 1, 1, padding='same', name='high_res_proj')(x_high_res)
        x_high_res = layers.BatchNormalization(name='high_res_bn_proj')(x_high_res)
        x_high_res = layers.Activation('swish', name='high_res_act_proj')(x_high_res)
        
        # Add high-resolution features to the shallowest BiFPN features
        bifpn_features[0] = layers.Add(name='high_res_merge')(
            [bifpn_features[0], x_high_res]
        )
    
    # Detection head with multi-level fusion
    detection_features = []
    
    # Process each level with dedicated detection heads
    for i, feature in enumerate(bifpn_features):
        # Scale-specific detection head
        detection_head = layers.Conv2D(256, 3, 1, padding='same', 
                                     activation='swish', name=f'detection_head_{i}_conv1')(feature)
        detection_head = layers.BatchNormalization(name=f'detection_head_{i}_bn1')(detection_head)
        detection_head = layers.Conv2D(256, 3, 1, padding='same', 
                                     activation='swish', name=f'detection_head_{i}_conv2')(detection_head)
        detection_head = layers.BatchNormalization(name=f'detection_head_{i}_bn2')(detection_head)
        
        # Global feature extraction for this level
        if i > 0:  # Skip the smallest scale (most detailed)
            global_feat = layers.GlobalAveragePooling2D(name=f'detection_head_{i}_gap')(detection_head)
            detection_features.append(global_feat)
        
        # For the most detailed level, use more feature maps
        if i == 0:
            # Extract more local features for detailed predictions
            pooled_feat = layers.AveragePooling2D(2, name=f'detection_head_{i}_avgpool')(detection_head)
            pooled_feat = layers.Flatten(name=f'detection_head_{i}_flatten')(pooled_feat)
            detection_features.append(pooled_feat)
    
    # Concatenate multi-scale detection features
    merged_features = layers.Concatenate(name='detection_concat')(detection_features)
    
    # Final bounding box regression with specific components for size and position
    shared = layers.Dense(512, activation='swish', name='bbox_shared_1')(merged_features)
    shared = layers.BatchNormalization(name='bbox_shared_bn1')(shared)
    shared = layers.Dropout(0.3, name='bbox_shared_dropout1')(shared)
    shared = layers.Dense(256, activation='swish', name='bbox_shared_2')(shared)
    shared = layers.BatchNormalization(name='bbox_shared_bn2')(shared)
    
    # Split prediction paths for better specialization
    # Position prediction branch
    pos_branch = layers.Dense(128, activation='swish', name='pos_dense1')(shared)
    pos_branch = layers.Dense(64, activation='swish', name='pos_dense2')(pos_branch)
    pos_output = layers.Dense(2, activation='sigmoid', name='pos_output')(pos_branch)  # x, y
    
    # Size prediction branch - with custom activation for better scaling
    size_branch = layers.Dense(128, activation='swish', name='size_dense1')(shared)
    size_branch = layers.Dense(64, activation='swish', name='size_dense2')(size_branch)
    # Use custom activation with constraints for width/height ratio
    size_output = layers.Dense(2, activation='sigmoid', name='size_output')(size_branch)  # w, h
    
    # Combine outputs
    outputs = layers.Concatenate(name='bbox_output')([pos_output, size_output])
    
    # Create model
    model = models.Model(inputs=inputs, outputs=outputs, name='improved_license_plate_detector')
    
    return model


def weighted_giou_loss(y_true, y_pred, size_weight=3.0):
    """
    Weighted GIoU loss that places more emphasis on size accuracy.
    
    Args:
        y_true: Ground truth bounding boxes [batch_size, 4] with normalized coordinates [x, y, w, h]
        y_pred: Predicted bounding boxes [batch_size, 4] with normalized coordinates [x, y, w, h]
        size_weight: Weight factor for size-related terms in the loss
        
    Returns:
        Weighted GIoU loss value
    """
    # Convert [x, y, w, h] to [x1, y1, x2, y2]
    def _bbox_to_corners(bbox):
        x, y, w, h = bbox[..., 0], bbox[..., 1], bbox[..., 2], bbox[..., 3]
        x1 = x - w/2
        y1 = y - h/2
        x2 = x + w/2
        y2 = y + h/2
        return tf.stack([x1, y1, x2, y2], axis=-1)
    
    # Get corners
    true_corners = _bbox_to_corners(y_true)
    pred_corners = _bbox_to_corners(y_pred)
    
    # Intersection coordinates
    xmin_inter = tf.maximum(true_corners[..., 0], pred_corners[..., 0])
    ymin_inter = tf.maximum(true_corners[..., 1], pred_corners[..., 1])
    xmax_inter = tf.minimum(true_corners[..., 2], pred_corners[..., 2])
    ymax_inter = tf.minimum(true_corners[..., 3], pred_corners[..., 3])
    
    # Calculate intersection area
    w_inter = tf.maximum(0., xmax_inter - xmin_inter)
    h_inter = tf.maximum(0., ymax_inter - ymin_inter)
    intersection = w_inter * h_inter
    
    # Calculate areas
    true_area = (true_corners[..., 2] - true_corners[..., 0]) * (true_corners[..., 3] - true_corners[..., 1])
    pred_area = (pred_corners[..., 2] - pred_corners[..., 0]) * (pred_corners[..., 3] - pred_corners[..., 1])
    
    # Calculate union
    union = true_area + pred_area - intersection
    
    # Calculate IoU
    iou = intersection / (union + tf.keras.backend.epsilon())
    
    # Calculate enclosing box coordinates
    xmin_encl = tf.minimum(true_corners[..., 0], pred_corners[..., 0])
    ymin_encl = tf.minimum(true_corners[..., 1], pred_corners[..., 1])
    xmax_encl = tf.maximum(true_corners[..., 2], pred_corners[..., 2])
    ymax_encl = tf.maximum(true_corners[..., 3], pred_corners[..., 3])
    
    # Calculate enclosing box area
    w_encl = xmax_encl - xmin_encl
    h_encl = ymax_encl - ymin_encl
    area_encl = w_encl * h_encl
    
    # Calculate GIoU
    giou = iou - (area_encl - union) / (area_encl + tf.keras.backend.epsilon())
    
    # Weighted size-specific loss component
    # Calculate size difference (normalized)
    size_diff_w = tf.abs(y_true[..., 2] - y_pred[..., 2]) / (y_true[..., 2] + tf.keras.backend.epsilon())
    size_diff_h = tf.abs(y_true[..., 3] - y_pred[..., 3]) / (y_true[..., 3] + tf.keras.backend.epsilon())
    size_penalty = size_weight * (size_diff_w + size_diff_h)
    
    # Final loss: original GIoU loss plus weighted size penalty
    loss = 1 - giou + size_penalty
    
    return tf.reduce_mean(loss)


def ciou_loss(y_true, y_pred):
    """
    Complete IoU Loss
    
    Args:
        y_true: Ground truth bounding boxes [batch_size, 4] with normalized coordinates [x, y, w, h]
        y_pred: Predicted bounding boxes [batch_size, 4] with normalized coordinates [x, y, w, h]
        
    Returns:
        CIoU loss value (1 - CIoU)
    """
    # Convert [x, y, w, h] to [x1, y1, x2, y2]
    def _bbox_to_corners(bbox):
        x, y, w, h = bbox[..., 0], bbox[..., 1], bbox[..., 2], bbox[..., 3]
        x1 = x - w/2
        y1 = y - h/2
        x2 = x + w/2
        y2 = y + h/2
        return tf.stack([x1, y1, x2, y2], axis=-1)
    
    true_boxes = _bbox_to_corners(y_true)
    pred_boxes = _bbox_to_corners(y_pred)
    
    # Calculate intersection area
    intersect_mins = tf.maximum(true_boxes[..., :2], pred_boxes[..., :2])
    intersect_maxes = tf.minimum(true_boxes[..., 2:], pred_boxes[..., 2:])
    intersect_wh = tf.maximum(intersect_maxes - intersect_mins, 0.)
    intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
    
    # Calculate union area
    true_area = (true_boxes[..., 2] - true_boxes[..., 0]) * (true_boxes[..., 3] - true_boxes[..., 1])
    pred_area = (pred_boxes[..., 2] - pred_boxes[..., 0]) * (pred_boxes[..., 3] - pred_boxes[..., 1])
    union_area = true_area + pred_area - intersect_area
    
    # Calculate IoU
    iou = intersect_area / (union_area + tf.keras.backend.epsilon())
    
    # Calculate enclosing box coordinates
    encl_mins = tf.minimum(true_boxes[..., :2], pred_boxes[..., :2])
    encl_maxes = tf.maximum(true_boxes[..., 2:], pred_boxes[..., 2:])
    encl_wh = encl_maxes - encl_mins
    
    # Calculate distance component
    center_dist_squared = tf.reduce_sum(tf.square(y_true[..., :2] - y_pred[..., :2]), axis=-1)
    encl_diag_squared = tf.reduce_sum(tf.square(encl_wh), axis=-1)
    
    # Calculate aspect ratio consistency term
    true_wh = y_true[..., 2:4]
    pred_wh = y_pred[..., 2:4]
    
    # To avoid division by zero
    v = 4 / (np.pi**2) * tf.square(
        tf.atan(true_wh[..., 0] / (true_wh[..., 1] + tf.keras.backend.epsilon())) - 
        tf.atan(pred_wh[..., 0] / (pred_wh[..., 1] + tf.keras.backend.epsilon()))
    )
    
    # CIoU formula components
    with tf.control_dependencies([tf.debugging.assert_greater_equal(encl_diag_squared, 0)]):
        alpha = v / (1 - iou + v + tf.keras.backend.epsilon())
    
    # Final CIoU
    ciou = iou - center_dist_squared / (encl_diag_squared + tf.keras.backend.epsilon()) - alpha * v
    
    # Return loss (1 - CIoU)
    return tf.reduce_mean(1 - ciou)


def size_balanced_focal_loss(y_true, y_pred, gamma=2.0, alpha=0.25, size_balance_factor=1.5):
    """
    Focal loss with specific weighting for small license plates.
    
    Args:
        y_true: Ground truth bounding boxes [batch_size, 4] with normalized coordinates
        y_pred: Predicted bounding boxes [batch_size, 4] with normalized coordinates
        gamma: Focusing parameter for hard examples
        alpha: Balancing parameter
        size_balance_factor: Factor to weight samples based on plate size
        
    Returns:
        Focal loss value
    """
    # Extract size information from ground truth
    true_size = y_true[..., 2] * y_true[..., 3]  # width * height
    
    # Calculate IoU as the base metric
    # This is a simplified IoU calculation for the focal loss weighting
    pred_width = y_pred[..., 2]
    pred_height = y_pred[..., 3]
    pred_size = pred_width * pred_height
    
    size_ratio = tf.minimum(true_size, pred_size) / (tf.maximum(true_size, pred_size) + tf.keras.backend.epsilon())
    
    # Convert to probability-like representation (IoU as "probability of correct detection")
    p = size_ratio  # Higher IoU means higher probability of correct detection
    
    # Calculate size-based weights (smaller objects get higher weights)
    size_weights = tf.pow(1.0 / (true_size + 0.1), size_balance_factor)  # Normalize weights
    size_weights = size_weights / tf.reduce_mean(size_weights)  # Scale to have mean=1
    
    # Focal loss formula with size balancing
    focal_weights = alpha * tf.pow(1.0 - p, gamma) * size_weights
    loss = -focal_weights * tf.math.log(p + tf.keras.backend.epsilon())
    
    return tf.reduce_mean(loss)


def adaptive_license_plate_loss(y_true, y_pred):
    """
    Adaptive loss function specifically designed for license plate detection.
    Combines CIoU loss with size-balancing and size-sensitive weighting.
    
    Args:
        y_true: Ground truth bounding boxes [batch_size, 4] with normalized coordinates
        y_pred: Predicted bounding boxes [batch_size, 4] with normalized coordinates
        
    Returns:
        Combined loss value
    """
    # Get individual loss components
    ciou_component = ciou_loss(y_true, y_pred)
    focal_component = size_balanced_focal_loss(y_true, y_pred)
    weighted_giou_component = weighted_giou_loss(y_true, y_pred)
    
    # Extract size information for adaptive weighting
    true_size = y_true[..., 2] * y_true[..., 3]  # width * height
    
    # Calculate adaptive weights based on plate size
    # Small plates get higher ciou and focal weight
    # Medium and large plates get higher weighted_giou weight
    small_plate_mask = tf.cast(true_size < 0.05, dtype=tf.float32)  # Threshold for "small" plates
    large_plate_mask = tf.cast(true_size >= 0.1, dtype=tf.float32)  # Threshold for "large" plates
    medium_plate_mask = 1.0 - small_plate_mask - large_plate_mask  # Everything in between
    
    # Batch-level adaptive weighting
    small_ratio = tf.reduce_mean(small_plate_mask)
    large_ratio = tf.reduce_mean(large_plate_mask)
    medium_ratio = tf.reduce_mean(medium_plate_mask)
    
    # Calculate loss coefficients (ensure they sum to approximately 1)
    ciou_weight = 0.4 + 0.2 * small_ratio  # Higher for batches with more small plates
    focal_weight = 0.3 + 0.1 * small_ratio  # Higher for batches with more small plates
    giou_weight = 0.3 + 0.1 * (medium_ratio + large_ratio)  # Higher for batches with more medium/large plates
    
    # Combined loss
    combined_loss = (ciou_weight * ciou_component + 
                    focal_weight * focal_component + 
                    giou_weight * weighted_giou_component)
    
    return combined_loss


def calculate_mean_iou(y_true, y_pred):
    """
    Calculate mean IoU between ground truth and predicted boxes.
    
    Args:
        y_true: Ground truth bounding boxes [batch_size, 4] with normalized coordinates [x, y, w, h]
        y_pred: Predicted bounding boxes [batch_size, 4] with normalized coordinates [x, y, w, h]
        
    Returns:
        Mean IoU value
    """
    # Convert [x, y, w, h] to [x1, y1, x2, y2]
    def _bbox_to_corners(bbox):
        x, y, w, h = bbox[..., 0], bbox[..., 1], bbox[..., 2], bbox[..., 3]
        x1 = x - w/2
        y1 = y - h/2
        x2 = x + w/2
        y2 = y + h/2
        return tf.stack([x1, y1, x2, y2], axis=-1)
    
    true_boxes = _bbox_to_corners(y_true)
    pred_boxes = _bbox_to_corners(y_pred)
    
    # Calculate intersection area
    intersect_mins = tf.maximum(true_boxes[..., :2], pred_boxes[..., :2])
    intersect_maxes = tf.minimum(true_boxes[..., 2:], pred_boxes[..., 2:])
    intersect_wh = tf.maximum(intersect_maxes - intersect_mins, 0.)
    intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
    
    # Calculate union area
    true_area = (true_boxes[..., 2] - true_boxes[..., 0]) * (true_boxes[..., 3] - true_boxes[..., 1])
    pred_area = (pred_boxes[..., 2] - pred_boxes[..., 0]) * (pred_boxes[..., 3] - pred_boxes[..., 1])
    union_area = true_area + pred_area - intersect_area
    
    # Calculate IoU
    iou = intersect_area / (union_area + tf.keras.backend.epsilon())
    
    return tf.reduce_mean(iou)
