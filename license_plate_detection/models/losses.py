"""
Custom loss functions and metrics for license plate detection.
"""

import tensorflow as tf
import numpy as np


def calculate_iou(boxes1, boxes2):
    """
    Calculate Intersection over Union (IoU) between bounding boxes.
    
    Args:
        boxes1: Tensor of shape (batch_size, 4) containing bounding boxes in format (x, y, w, h)
        boxes2: Tensor of shape (batch_size, 4) containing bounding boxes in format (x, y, w, h)
        
    Returns:
        Tensor of shape (batch_size,) containing IoU values for each box pair
    """
    # Convert boxes from [x, y, w, h] to [x1, y1, x2, y2] format
    boxes1_x1 = boxes1[:, 0] - boxes1[:, 2] / 2
    boxes1_y1 = boxes1[:, 1] - boxes1[:, 3] / 2
    boxes1_x2 = boxes1[:, 0] + boxes1[:, 2] / 2
    boxes1_y2 = boxes1[:, 1] + boxes1[:, 3] / 2
    
    boxes2_x1 = boxes2[:, 0] - boxes2[:, 2] / 2
    boxes2_y1 = boxes2[:, 1] - boxes2[:, 3] / 2
    boxes2_x2 = boxes2[:, 0] + boxes2[:, 2] / 2
    boxes2_y2 = boxes2[:, 1] + boxes2[:, 3] / 2
    
    # Calculate intersection area
    intersect_x1 = tf.maximum(boxes1_x1, boxes2_x1)
    intersect_y1 = tf.maximum(boxes1_y1, boxes2_y1)
    intersect_x2 = tf.minimum(boxes1_x2, boxes2_x2)
    intersect_y2 = tf.minimum(boxes1_y2, boxes2_y2)
    
    intersect_width = tf.maximum(0.0, intersect_x2 - intersect_x1)
    intersect_height = tf.maximum(0.0, intersect_y2 - intersect_y1)
    intersection = intersect_width * intersect_height
    
    # Calculate union area
    boxes1_area = boxes1[:, 2] * boxes1[:, 3]
    boxes2_area = boxes2[:, 2] * boxes2[:, 3]
    union = boxes1_area + boxes2_area - intersection
    
    # Calculate IoU
    iou = tf.divide(intersection, union + 1e-7)  # Add small epsilon to avoid division by zero
    
    return iou


def enhanced_iou_metric(y_true, y_pred):
    """
    Enhanced IoU metric for license plate detection.
    
    Args:
        y_true: Tensor of shape (batch_size, 4) containing true bounding boxes
        y_pred: Tensor of shape (batch_size, 4) containing predicted bounding boxes
        
    Returns:
        Mean IoU across the batch
    """
    return tf.reduce_mean(calculate_iou(y_true, y_pred))


def combined_detection_loss(y_true, y_pred):
    """
    Combined loss function for license plate detection.
    Combines IoU loss with MSE on box coordinates.
    
    Args:
        y_true: Tensor of shape (batch_size, 4) containing true bounding boxes
        y_pred: Tensor of shape (batch_size, 4) containing predicted bounding boxes
        
    Returns:
        Combined loss value
    """    # MSE loss for coordinate regression - using tf.reduce_mean(tf.square()) instead of mean_squared_error
    mse_loss = tf.reduce_mean(tf.square(y_true - y_pred))
    
    # IoU-based loss
    iou = calculate_iou(y_true, y_pred)
    iou_loss = 1.0 - tf.reduce_mean(iou)
    
    # Position loss (more weight on center coordinates)
    position_loss = tf.reduce_mean(tf.square(y_true[:, :2] - y_pred[:, :2])) * 2.0
    
    # Size loss
    size_loss = tf.reduce_mean(tf.square(y_true[:, 2:] - y_pred[:, 2:]))
    
    # Combine losses with weights
    combined_loss = iou_loss * 0.5 + position_loss * 0.3 + size_loss * 0.2
    
    return combined_loss


def giou_loss(y_true, y_pred):
    """
    Generalized IoU loss function for bounding box regression.
    
    Args:
        y_true: Tensor of shape (batch_size, 4) containing true bounding boxes in format (x, y, w, h)
        y_pred: Tensor of shape (batch_size, 4) containing predicted bounding boxes in format (x, y, w, h)
        
    Returns:
        GIoU loss value
    """
    # Convert boxes from [x, y, w, h] to [x1, y1, x2, y2] format
    boxes1_x1 = y_true[:, 0] - y_true[:, 2] / 2
    boxes1_y1 = y_true[:, 1] - y_true[:, 3] / 2
    boxes1_x2 = y_true[:, 0] + y_true[:, 2] / 2
    boxes1_y2 = y_true[:, 1] + y_true[:, 3] / 2
    
    boxes2_x1 = y_pred[:, 0] - y_pred[:, 2] / 2
    boxes2_y1 = y_pred[:, 1] - y_pred[:, 3] / 2
    boxes2_x2 = y_pred[:, 0] + y_pred[:, 2] / 2
    boxes2_y2 = y_pred[:, 1] + y_pred[:, 3] / 2
    
    # Calculate intersection area
    intersect_x1 = tf.maximum(boxes1_x1, boxes2_x1)
    intersect_y1 = tf.maximum(boxes1_y1, boxes2_y1)
    intersect_x2 = tf.minimum(boxes1_x2, boxes2_x2)
    intersect_y2 = tf.minimum(boxes1_y2, boxes2_y2)
    
    intersect_width = tf.maximum(0.0, intersect_x2 - intersect_x1)
    intersect_height = tf.maximum(0.0, intersect_y2 - intersect_y1)
    intersection = intersect_width * intersect_height
    
    # Calculate areas of boxes
    boxes1_area = (boxes1_x2 - boxes1_x1) * (boxes1_y2 - boxes1_y1)
    boxes2_area = (boxes2_x2 - boxes2_x1) * (boxes2_y2 - boxes2_y1)
    
    # Calculate union area
    union = boxes1_area + boxes2_area - intersection
    
    # Calculate IoU
    iou = tf.divide(intersection, union + 1e-7)
    
    # Calculate the smallest enclosing box
    enclosing_x1 = tf.minimum(boxes1_x1, boxes2_x1)
    enclosing_y1 = tf.minimum(boxes1_y1, boxes2_y1)
    enclosing_x2 = tf.maximum(boxes1_x2, boxes2_x2)
    enclosing_y2 = tf.maximum(boxes1_y2, boxes2_y2)
    
    enclosing_width = enclosing_x2 - enclosing_x1
    enclosing_height = enclosing_y2 - enclosing_y1
    enclosing_area = enclosing_width * enclosing_height
    
    # Calculate GIoU
    giou = iou - tf.divide(enclosing_area - union, enclosing_area + 1e-7)
    
    # GIoU loss
    giou_loss = 1.0 - giou
    
    return tf.reduce_mean(giou_loss)


def focal_loss_bbox(y_true, y_pred, gamma=2.0, alpha=0.25):
    """
    Focal loss for bounding box regression that emphasizes hard examples.
    
    Args:
        y_true: Ground truth bounding boxes [x, y, width, height]
        y_pred: Predicted bounding boxes [x, y, width, height]
        gamma: Focusing parameter for hard examples
        alpha: Balancing parameter
        
    Returns:
        Focal loss value
    """
    # IoU between true and predicted boxes
    iou = calculate_iou(y_true, y_pred)
    
    # Calculate focal weight
    focal_weight = alpha * tf.pow(1.0 - iou, gamma)
    
    # Square error with focal weighting
    squared_error = tf.square(y_true - y_pred)
    
    # Apply weights to emphasize size (width and height) components
    component_weights = tf.constant([1.0, 1.0, 2.0, 2.0], dtype=tf.float32)  # More weight on size
    weighted_squared_error = squared_error * component_weights
    
    # Apply focal weighting
    focal_loss = focal_weight * tf.reduce_sum(weighted_squared_error, axis=-1)
    
    return tf.reduce_mean(focal_loss)


def size_sensitive_loss(y_true, y_pred, size_threshold=0.1):
    """
    Loss function that gives higher weights to small objects.
    
    Args:
        y_true: Ground truth bounding boxes [x, y, width, height]
        y_pred: Predicted bounding boxes [x, y, width, height]
        size_threshold: Threshold to define small objects
        
    Returns:
        Size sensitive loss value
    """
    # Calculate box areas
    true_areas = y_true[..., 2] * y_true[..., 3]  # width * height
    
    # Create weights based on size
    small_box_weights = tf.where(
        true_areas < size_threshold,
        2.0,  # Higher weight for small boxes
        1.0   # Normal weight for large boxes
    )
    
    # Calculate squared errors
    squared_error = tf.square(y_true - y_pred)
    
    # Apply size-based weights
    weighted_error = tf.expand_dims(small_box_weights, axis=-1) * squared_error
    
    return tf.reduce_mean(tf.reduce_sum(weighted_error, axis=-1))


def improved_combined_detection_loss(y_true, y_pred):
    """
    Combined loss function with improved weights based on error analysis.
    Emphasizes size accuracy (width/height) over position (x/y).
    
    Args:
        y_true: Ground truth bounding boxes [x, y, width, height]
        y_pred: Predicted bounding boxes [x, y, width, height]
        
    Returns:
        Combined loss value
    """
    # Calculate individual component losses
    # Position loss - x, y coordinates
    position_loss = tf.reduce_mean(tf.square(y_true[..., :2] - y_pred[..., :2]))  
    
    # Size loss with more emphasis on accuracy
    # Apply L1 loss for sizes to be less sensitive to outliers
    size_diff = tf.abs(y_true[..., 2:] - y_pred[..., 2:])
    size_loss = tf.reduce_mean(size_diff + 0.5 * tf.square(size_diff))  # Huber-like loss for size
    
    # IoU loss for overall shape and position
    iou_loss = 1.0 - enhanced_iou_metric(y_true, y_pred)
    
    # GIoU loss to better handle non-overlapping cases
    giou_component = giou_loss(y_true, y_pred)
    
    # Focal loss component to focus on hard examples
    focal_component = focal_loss_bbox(y_true, y_pred, gamma=2.5)
    
    # Size sensitive component with higher weight for small plates
    # Reduce the size threshold to 0.05 to focus more on very small plates
    size_sensitive_component = size_sensitive_loss(y_true, y_pred, size_threshold=0.05)
    
    # Calculate additional aspect ratio loss to maintain plate proportions
    true_aspect = y_true[..., 2] / (y_true[..., 3] + 1e-7)  # width/height
    pred_aspect = y_pred[..., 2] / (y_pred[..., 3] + 1e-7)
    aspect_ratio_loss = tf.reduce_mean(tf.abs(true_aspect - pred_aspect))
    
    # Adjusted weights based on error analysis results:
    # - Size given highest priority (0.4) to address the 1.6132 size error
    # - GIoU and IoU combined given significant weight (0.3) for overall placement
    # - Position weight reduced (0.1) as it's less problematic
    # - Added aspect ratio loss (0.1) to maintain proper plate dimensions
    # - Focal and size-sensitive components to handle challenging cases (0.1)
    return (
        0.1 * position_loss +
        0.4 * size_loss + 
        0.15 * iou_loss + 
        0.15 * giou_component +
        0.1 * aspect_ratio_loss +
        0.05 * focal_component + 
        0.05 * size_sensitive_component
    )


def focal_loss(y_true, y_pred, alpha=0.25, gamma=2.0):
    """
    Focal loss for imbalanced classification problems.
    
    Args:
        y_true: Tensor of ground truth targets
        y_pred: Tensor of predicted targets
        alpha: Weighting factor
        gamma: Focusing parameter
        
    Returns:
        Focal loss value
    """
    epsilon = 1e-7
    y_pred = tf.clip_by_value(y_pred, epsilon, 1 - epsilon)
    
    # For binary classification
    binary_cross_entropy = -(y_true * tf.math.log(y_pred) + 
                           (1 - y_true) * tf.math.log(1 - y_pred))
    
    p_t = (y_true * y_pred) + ((1 - y_true) * (1 - y_pred))
    alpha_t = y_true * alpha + (1 - y_true) * (1 - alpha)
    focal_weight = alpha_t * tf.pow(1 - p_t, gamma)
    
    return tf.reduce_mean(focal_weight * binary_cross_entropy)
