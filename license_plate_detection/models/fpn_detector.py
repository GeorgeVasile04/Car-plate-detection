"""
Feature Pyramid Network (FPN) implementation for license plate detection.
Focuses on improving size estimation and small plate detection.
"""

import tensorflow as tf
from tensorflow.keras import layers, models


class ChannelAttention(layers.Layer):
    """
    Channel attention module that helps the model focus on the most important channels.
    Especially useful for detecting small license plates.
    """
    def __init__(self, filters, ratio=8):
        super(ChannelAttention, self).__init__()
        self.avg_pool = layers.GlobalAveragePooling2D()
        self.max_pool = layers.GlobalMaxPooling2D()
        
        self.dense1 = layers.Dense(filters // ratio, activation='relu')
        self.dense2 = layers.Dense(filters, activation='sigmoid')
        
        self.reshape = layers.Reshape((1, 1, filters))
        
    def call(self, inputs):
        avg_pool = self.avg_pool(inputs)
        max_pool = self.max_pool(inputs)
        
        avg_pool = self.dense1(avg_pool)
        avg_pool = self.dense2(avg_pool)
        
        max_pool = self.dense1(max_pool)
        max_pool = self.dense2(max_pool)
        
        attention = avg_pool + max_pool
        attention = self.reshape(attention)
        
        return inputs * attention


class SpatialAttention(layers.Layer):
    """
    Spatial attention module that helps the model focus on the most important regions.
    """
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv = layers.Conv2D(1, kernel_size, padding='same', activation='sigmoid')
        
    def call(self, inputs):
        # Average pooling along channel axis
        avg_pool = tf.reduce_mean(inputs, axis=-1, keepdims=True)
        # Max pooling along channel axis
        max_pool = tf.reduce_max(inputs, axis=-1, keepdims=True)
        
        # Concatenate pooled features
        pooled = tf.concat([avg_pool, max_pool], axis=-1)
        
        # Apply convolution
        attention = self.conv(pooled)
        
        return inputs * attention


class CBAM(layers.Layer):
    """
    Convolutional Block Attention Module (CBAM) that combines channel and spatial attention.
    """
    def __init__(self, filters, ratio=8, kernel_size=7):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(filters, ratio)
        self.spatial_attention = SpatialAttention(kernel_size)
        
    def call(self, inputs):
        x = self.channel_attention(inputs)
        x = self.spatial_attention(x)
        return x


def create_fpn_license_plate_detector(input_shape=(416, 416, 3), train_backbone=True):
    """
    Creates an improved license plate detector with Feature Pyramid Network.
    This architecture is specifically designed to address size estimation issues
    and improve detection of small license plates.
    
    Args:
        input_shape: Input image shape (height, width, channels)
        train_backbone: Whether to train the backbone network
        
    Returns:
        tf.keras.Model: A compiled license plate detector model
    """
    # Input layer - Using 416x416 (slightly larger than original 224x224)
    # to better capture small license plates
    inputs = layers.Input(shape=input_shape)
    
    # Initial pre-processing: normalize pixel values to [-1, 1] for better gradient flow
    x = layers.Lambda(lambda x: x / 127.5 - 1.0)(inputs)
    
    # Initial feature extraction with larger receptive field
    x = layers.Conv2D(64, (7, 7), strides=(2, 2), padding='same', name='conv1')(x)
    x = layers.BatchNormalization(name='bn1')(x)
    x = layers.Activation('swish', name='swish1')(x)  # Swish activation for better gradient flow
    x = layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same', name='pool1')(x)
    
    # Track feature maps for FPN
    feature_maps = []
    
    # ===== Block 1: First Residual Block with Channel Attention =====
    residual = x
    x = layers.Conv2D(128, (3, 3), padding='same', name='block1_conv1')(x)
    x = layers.BatchNormalization(name='block1_bn1')(x)
    x = layers.Activation('swish', name='block1_swish1')(x)
    x = layers.Conv2D(128, (3, 3), padding='same', name='block1_conv2')(x)
    x = layers.BatchNormalization(name='block1_bn2')(x)
    
    # Add CBAM attention
    x = CBAM(128)(x)
    
    # Add residual connection
    if residual.shape[-1] != x.shape[-1]:
        residual = layers.Conv2D(128, (1, 1), name='block1_res_conv')(residual)
        residual = layers.BatchNormalization(name='block1_res_bn')(residual)
    x = layers.Add(name='block1_add')([x, residual])
    x = layers.Activation('swish', name='block1_swish_out')(x)
    
    feature_maps.append(x)  # P2 feature map (high resolution)
    
    # ===== Block 2: Second Residual Block with Advanced Feature Extraction =====
    x = layers.MaxPooling2D((2, 2), name='block2_pool')(x)
    residual = x
    
    # Multi-scale feature extraction with dilated convolutions
    # Path 1: Standard 3x3 convolutions
    x1 = layers.Conv2D(256, (3, 3), padding='same', name='block2_conv1')(x)
    x1 = layers.BatchNormalization(name='block2_bn1')(x1)
    x1 = layers.Activation('swish', name='block2_swish1')(x1)
    x1 = layers.Conv2D(256, (3, 3), padding='same', name='block2_conv2')(x1)
    x1 = layers.BatchNormalization(name='block2_bn2')(x1)
    
    # Path 2: Dilated convolutions for larger receptive field
    x2 = layers.Conv2D(256, (3, 3), dilation_rate=(2, 2), padding='same', name='block2_dconv1')(x)
    x2 = layers.BatchNormalization(name='block2_dbn1')(x2)
    x2 = layers.Activation('swish', name='block2_dswish1')(x2)
    
    # Path 3: Even larger dilated convolutions for global context
    x3 = layers.Conv2D(256, (3, 3), dilation_rate=(4, 4), padding='same', name='block2_dconv2')(x)
    x3 = layers.BatchNormalization(name='block2_dbn2')(x3)
    x3 = layers.Activation('swish', name='block2_dswish2')(x3)
    
    # Combine all paths
    x = layers.Concatenate(name='block2_concat')([x1, x2, x3])
    x = layers.Conv2D(256, (1, 1), padding='same', name='block2_fusion_conv')(x)
    x = layers.BatchNormalization(name='block2_fusion_bn')(x)
    x = layers.Activation('swish', name='block2_fusion_swish')(x)
    
    # Add CBAM attention
    x = CBAM(256)(x)
    
    # Add residual connection
    if residual.shape[-1] != x.shape[-1]:
        residual = layers.Conv2D(256, (1, 1), name='block2_res_conv')(residual)
        residual = layers.BatchNormalization(name='block2_res_bn')(residual)
    x = layers.Add(name='block2_add')([x, residual])
    x = layers.Activation('swish', name='block2_out')(x)
    
    feature_maps.append(x)  # P3 feature map (medium resolution)
    
    # ===== Block 3: Third Residual Block with Spatial Attention =====
    x = layers.MaxPooling2D((2, 2), name='block3_pool')(x)
    residual = x
    
    x = layers.Conv2D(512, (3, 3), padding='same', name='block3_conv1')(x)
    x = layers.BatchNormalization(name='block3_bn1')(x)
    x = layers.Activation('swish', name='block3_swish1')(x)
    x = layers.Conv2D(512, (3, 3), padding='same', name='block3_conv2')(x)
    x = layers.BatchNormalization(name='block3_bn2')(x)
    
    # Add CBAM attention
    x = CBAM(512)(x)
    
    # Add residual connection
    if residual.shape[-1] != x.shape[-1]:
        residual = layers.Conv2D(512, (1, 1), name='block3_res_conv')(residual)
        residual = layers.BatchNormalization(name='block3_res_bn')(residual)
    x = layers.Add(name='block3_add')([x, residual])
    x = layers.Activation('swish', name='block3_out')(x)
    
    feature_maps.append(x)  # P4 feature map (low resolution)
    
    # ===== Feature Pyramid Network Implementation =====
    # Add one more level for deeper features
    x = layers.MaxPooling2D((2, 2), name='block4_pool')(x)
    x = layers.Conv2D(1024, (3, 3), padding='same', name='block4_conv1')(x)
    x = layers.BatchNormalization(name='block4_bn1')(x)
    x = layers.Activation('swish', name='block4_swish1')(x)
    feature_maps.append(x)  # P5 feature map (lowest resolution)
    
    # Now build the FPN top-down pathway
    # P5 (lowest resolution)
    p5 = layers.Conv2D(256, (1, 1), padding='same', name='fpn_p5')(feature_maps[3])
    
    # P4
    p5_upsampled = layers.UpSampling2D(size=(2, 2), interpolation='bilinear', name='fpn_p5_upsample')(p5)
    p4_conv = layers.Conv2D(256, (1, 1), padding='same', name='fpn_p4_conv')(feature_maps[2])
    p4 = layers.Add(name='fpn_p4_add')([p5_upsampled, p4_conv])
    p4 = layers.Conv2D(256, (3, 3), padding='same', name='fpn_p4_post_conv')(p4)
    
    # P3
    p4_upsampled = layers.UpSampling2D(size=(2, 2), interpolation='bilinear', name='fpn_p4_upsample')(p4)
    p3_conv = layers.Conv2D(256, (1, 1), padding='same', name='fpn_p3_conv')(feature_maps[1])
    p3 = layers.Add(name='fpn_p3_add')([p4_upsampled, p3_conv])
    p3 = layers.Conv2D(256, (3, 3), padding='same', name='fpn_p3_post_conv')(p3)
    
    # P2 (highest resolution)
    p3_upsampled = layers.UpSampling2D(size=(2, 2), interpolation='bilinear', name='fpn_p3_upsample')(p3)
    p2_conv = layers.Conv2D(256, (1, 1), padding='same', name='fpn_p2_conv')(feature_maps[0])
    p2 = layers.Add(name='fpn_p2_add')([p3_upsampled, p2_conv])
    p2 = layers.Conv2D(256, (3, 3), padding='same', name='fpn_p2_post_conv')(p2)
    
    # ===== Size-Aware Detection Heads for Different Scales =====
    # Using different FPN levels for different object sizes
    
    # P2 (highest resolution) - Best for small plates
    p2_att = CBAM(256)(p2)  # Add attention
    small_head = layers.Conv2D(256, (3, 3), padding='same', activation='swish', name='small_head_conv1')(p2_att)
    small_head = layers.Conv2D(256, (3, 3), padding='same', activation='swish', name='small_head_conv2')(small_head)
    small_feat = layers.GlobalAveragePooling2D(name='small_head_gap')(small_head)
    
    # P3 (medium resolution) - Best for medium plates
    p3_att = CBAM(256)(p3)  # Add attention
    medium_head = layers.Conv2D(256, (3, 3), padding='same', activation='swish', name='medium_head_conv1')(p3_att)
    medium_head = layers.Conv2D(256, (3, 3), padding='same', activation='swish', name='medium_head_conv2')(medium_head)
    medium_feat = layers.GlobalAveragePooling2D(name='medium_head_gap')(medium_head)
    
    # P4 (low resolution) - Best for large plates
    p4_att = CBAM(256)(p4)  # Add attention
    large_head = layers.Conv2D(256, (3, 3), padding='same', activation='swish', name='large_head_conv1')(p4_att)
    large_head = layers.Conv2D(256, (3, 3), padding='same', activation='swish', name='large_head_conv2')(large_head)
    large_feat = layers.GlobalAveragePooling2D(name='large_head_gap')(large_head)
    
    # Combine features from different levels weighted by attention
    combined_feat = layers.Concatenate(name='combined_features')([small_feat, medium_feat, large_feat])
    
    # Final prediction layers with size-aware components
    x = layers.Dense(512, activation='swish', name='final_dense1')(combined_feat)
    x = layers.BatchNormalization(name='final_bn1')(x)
    x = layers.Dropout(0.3, name='final_dropout1')(x)  # Reduced dropout for better learning
    
    x = layers.Dense(256, activation='swish', name='final_dense2')(x)
    x = layers.BatchNormalization(name='final_bn2')(x)
    x = layers.Dropout(0.2, name='final_dropout2')(x)  # Reduced dropout for better learning
    
    # Split prediction into position (x,y) and size (w,h) components
    position_pred = layers.Dense(128, activation='swish', name='position_dense')(x)
    size_pred = layers.Dense(128, activation='swish', name='size_dense')(x)
    
    # Output layers with independent position and size predictions
    position_output = layers.Dense(2, activation='sigmoid', name='position_output')(position_pred)
    size_output = layers.Dense(2, activation='sigmoid', name='size_output')(size_pred)
    
    # Concatenate position and size for final output [x, y, w, h]
    outputs = layers.Concatenate(name='final_output')([position_output, size_output])
    
    # Create model
    model = models.Model(inputs=inputs, outputs=outputs, name='fpn_license_plate_detector')
    
    return model
