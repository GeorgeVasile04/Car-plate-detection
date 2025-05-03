"""
Enhanced loss functions specifically designed to address license plate detection challenges.
These functions place stronger emphasis on size estimation accuracy and small plate detection.
"""

import tensorflow as tf
import numpy as np

def enhanced_ciou_loss(y_true, y_pred, size_weight=5.0):
    """
    Enhanced Complete IoU Loss with additional emphasis on size accuracy.
    
    Args:
        y_true: Ground truth bounding boxes [batch_size, 4] with normalized coordinates [x, y, w, h]
        y_pred: Predicted bounding boxes [batch_size, 4] with normalized coordinates [x, y, w, h]
        size_weight: Weight multiplier for size-related components (higher = more emphasis)
        
    Returns:
        Enhanced CIoU loss value
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
    
    # Calculate enclosing box
    encl_mins = tf.minimum(true_boxes[..., :2], pred_boxes[..., :2])
    encl_maxes = tf.maximum(true_boxes[..., 2:], pred_boxes[..., 2:])
    encl_wh = encl_maxes - encl_mins
    
    # Distance component
    center_dist_squared = tf.reduce_sum(tf.square(y_true[..., :2] - y_pred[..., :2]), axis=-1)
    encl_diag_squared = tf.reduce_sum(tf.square(encl_wh), axis=-1)
    
    # Enhanced size penalty - squared error on width and height
    true_w, true_h = y_true[..., 2], y_true[..., 3]
    pred_w, pred_h = y_pred[..., 2], y_pred[..., 3]
    
    # Size errors with higher emphasis on small plates
    # Apply relative size error (not absolute) to normalize across different scales
    size_error_w = tf.square((true_w - pred_w) / (true_w + tf.keras.backend.epsilon()))
    size_error_h = tf.square((true_h - pred_h) / (true_h + tf.keras.backend.epsilon()))
    
    # Small plate awareness - give higher weight to errors on small plates
    plate_size = true_w * true_h
    small_plate_factor = tf.pow(0.05 / (plate_size + 0.01), 0.5)  # Higher weight for smaller plates
    
    # Combined size penalty
    size_penalty = size_weight * (size_error_w + size_error_h) * small_plate_factor
    
    # Aspect ratio consistency term
    v = 4 / (np.pi**2) * tf.square(
        tf.atan(true_w / (true_h + tf.keras.backend.epsilon())) - 
        tf.atan(pred_w / (pred_h + tf.keras.backend.epsilon()))
    )
    
    # Alpha parameter for balancing
    with tf.control_dependencies([tf.debugging.assert_greater_equal(encl_diag_squared, 0)]):
        alpha = v / (1 - iou + v + tf.keras.backend.epsilon())
    
    # Standard CIoU loss
    std_ciou = 1 - (iou - center_dist_squared / (encl_diag_squared + tf.keras.backend.epsilon()) - alpha * v)
    
    # Enhanced CIoU with additional size penalty
    enhanced_ciou = std_ciou + size_penalty
    
    return tf.reduce_mean(enhanced_ciou)

def enhanced_focal_loss(y_true, y_pred, gamma=2.5, alpha=0.25):
    """
    Enhanced focal loss with stronger emphasis on challenging plates.
    
    Args:
        y_true: Ground truth bounding boxes [batch_size, 4] with normalized coordinates
        y_pred: Predicted bounding boxes [batch_size, 4] with normalized coordinates
        gamma: Focusing parameter (higher = more focus on hard examples)
        alpha: Balancing factor
        
    Returns:
        Focal loss value
    """
    # Extract ground truth information
    true_center = y_true[..., :2]  # x, y
    true_size = y_true[..., 2] * y_true[..., 3]  # width * height
    
    # Calculate positional accuracy
    center_error = tf.reduce_sum(tf.square(true_center - y_pred[..., :2]), axis=-1)
    
    # Calculate size accuracy 
    pred_size = y_pred[..., 2] * y_pred[..., 3]
    size_ratio = tf.minimum(true_size, pred_size) / (tf.maximum(true_size, pred_size) + tf.keras.backend.epsilon())
    
    # Convert to probability-like representation (1 = perfect match, 0 = no match)
    p_center = tf.exp(-10 * center_error)  # Exponential decay for center error
    p_size = size_ratio  # Direct size ratio
    
    # Combine probabilities with emphasis on size
    p = 0.4 * p_center + 0.6 * p_size
    
    # Small plate weighting
    small_plate_weight = tf.pow(0.05 / (true_size + 0.01), 0.75)
    
    # Focal loss formula with size-based weighting
    focal_weights = alpha * tf.pow(1.0 - p, gamma) * small_plate_weight
    loss = -focal_weights * tf.math.log(p + tf.keras.backend.epsilon())
    
    return tf.reduce_mean(loss)

def size_aware_smooth_l1_loss(y_true, y_pred):
    """
    Size-aware smooth L1 loss that places higher emphasis on small plates.
    
    Args:
        y_true: Ground truth bounding boxes [batch_size, 4] with normalized coordinates
        y_pred: Predicted bounding boxes [batch_size, 4] with normalized coordinates
        
    Returns:
        Loss value
    """
    # Extract width and height
    true_width, true_height = y_true[..., 2], y_true[..., 3]
    pred_width, pred_height = y_pred[..., 2], y_pred[..., 3]
    
    # Calculate plate size
    plate_size = true_width * true_height
    
    # Small plate weighting (higher weight for smaller plates)
    size_weights = tf.pow(0.1 / (plate_size + 0.05), 0.5)
    size_weights = size_weights / tf.reduce_mean(size_weights)  # Normalize
    
    # Calculate absolute difference
    abs_diff = tf.abs(y_true - y_pred)
    
    # Smooth L1 loss - less sensitive to outliers than MSE
    smooth_l1 = tf.where(
        abs_diff < 1.0,
        0.5 * tf.square(abs_diff),
        abs_diff - 0.5
    )
    
    # Weight losses differently for position and size
    position_loss = tf.reduce_sum(smooth_l1[..., :2], axis=-1) * 0.3  # Less weight on position
    size_loss = tf.reduce_sum(smooth_l1[..., 2:], axis=-1) * 0.7  # More weight on size
    
    # Combine with size weights
    weighted_loss = (position_loss + size_loss) * size_weights
    
    return tf.reduce_mean(weighted_loss)

def ultra_plate_detection_loss(y_true, y_pred):
    """
    Comprehensive loss function optimized specifically for license plate detection.
    Combines multiple loss components with adaptive weighting based on plate characteristics.
    
    Args:
        y_true: Ground truth bounding boxes [batch_size, 4] with normalized coordinates
        y_pred: Predicted bounding boxes [batch_size, 4] with normalized coordinates
        
    Returns:
        Combined loss value
    """
    # Extract plate size information
    true_size = y_true[..., 2] * y_true[..., 3]  # width * height
    
    # Categorize plates by size for adaptive weighting
    small_plate_mask = tf.cast(true_size < 0.05, dtype=tf.float32)
    large_plate_mask = tf.cast(true_size >= 0.1, dtype=tf.float32)
    medium_plate_mask = 1.0 - small_plate_mask - large_plate_mask
    
    # Calculate loss components
    ciou_component = enhanced_ciou_loss(y_true, y_pred)
    focal_component = enhanced_focal_loss(y_true, y_pred)
    l1_component = size_aware_smooth_l1_loss(y_true, y_pred)
    
    # Calculate batch composition ratios
    small_ratio = tf.reduce_mean(small_plate_mask)
    large_ratio = tf.reduce_mean(large_plate_mask)
    medium_ratio = tf.reduce_mean(medium_plate_mask)
    
    # Adaptive weighting based on batch composition
    # More small plates → higher weight on focal and size components
    ciou_weight = 0.45 + 0.15 * small_ratio
    focal_weight = 0.30 + 0.20 * small_ratio
    l1_weight = 0.25 + 0.15 * small_ratio
    
    # Ensure weights sum to 1
    total_weight = ciou_weight + focal_weight + l1_weight
    ciou_weight = ciou_weight / total_weight
    focal_weight = focal_weight / total_weight
    l1_weight = l1_weight / total_weight
    
    # Final combined loss
    return (
        ciou_weight * ciou_component + 
        focal_weight * focal_component + 
        l1_weight * l1_component
    )

def enhanced_iou_metric(y_true, y_pred):
    """
    Enhanced IoU metric with smoothing for more stable training.
    
    Args:
        y_true: Ground truth bounding boxes [batch_size, 4] with normalized coordinates
        y_pred: Predicted bounding boxes [batch_size, 4] with normalized coordinates
        
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
    
    # Calculate intersection
    intersect_mins = tf.maximum(true_boxes[..., :2], pred_boxes[..., :2])
    intersect_maxes = tf.minimum(true_boxes[..., 2:], pred_boxes[..., 2:])
    intersect_wh = tf.maximum(intersect_maxes - intersect_mins, 0.)
    intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
    
    # Calculate union
    true_area = (true_boxes[..., 2] - true_boxes[..., 0]) * (true_boxes[..., 3] - true_boxes[..., 1])
    pred_area = (pred_boxes[..., 2] - pred_boxes[..., 0]) * (pred_boxes[..., 3] - pred_boxes[..., 1])
    union_area = true_area + pred_area - intersect_area + tf.keras.backend.epsilon()
    
    # Calculate IoU with epsilon for numerical stability
    iou = intersect_area / union_area
    
    return tf.reduce_mean(iou)
