"""
Generate demonstration predictions and comprehensive results for the demo visualization in the notebook.
"""

import numpy as np
import tensorflow as tf
from typing import Dict, List, Tuple, Any, Optional

def generate_demo_predictions(y_val: np.ndarray, 
                             variance: float = 0.1,
                             quality_level: str = "medium") -> np.ndarray:
    """
    Generate simulated predictions based on ground truth with controlled quality.
    
    Args:
        y_val: Ground truth bounding boxes in [x, y, w, h] normalized format
        variance: Standard deviation for noise addition
        quality_level: Quality level - "poor", "medium", or "good"
        
    Returns:
        np.ndarray: Mock predictions with shape identical to y_val
    """
    # Set noise level based on quality
    if quality_level == "poor":
        variance *= 2.0  # Double the noise for poor quality
    elif quality_level == "good":
        variance *= 0.5  # Half the noise for good quality
        
    # Create a copy of ground truth to avoid modifying the original
    predictions = y_val.copy()
    
    # Add controlled noise to each component
    for i in range(len(predictions)):
        # Add noise to x, y (position)
        predictions[i, 0] = np.clip(predictions[i, 0] + np.random.normal(0, variance * 0.5), 0, 1-predictions[i, 2])
        predictions[i, 1] = np.clip(predictions[i, 1] + np.random.normal(0, variance * 0.5), 0, 1-predictions[i, 3])
        
        # Add more noise to w, h (size) since our error analysis shows size is more challenging
        predictions[i, 2] = np.clip(predictions[i, 2] * (1 + np.random.normal(0, variance * 1.5)), 0.01, 1-predictions[i, 0])
        predictions[i, 3] = np.clip(predictions[i, 3] * (1 + np.random.normal(0, variance * 1.5)), 0.01, 1-predictions[i, 1])
    
    return predictions

def create_mock_comprehensive_results(y_val: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
    """
    Create mock comprehensive evaluation results for demo purposes.
    
    Args:
        y_val: Ground truth bounding boxes
        y_pred: Predicted bounding boxes
        
    Returns:
        dict: Mock comprehensive evaluation results
    """
    # Calculate basic IoU for each prediction
    iou_values = []
    
    for i in range(len(y_val)):
        true_box = y_val[i]
        pred_box = y_pred[i]
        
        # Calculate IoU
        # Convert to x1, y1, x2, y2 format
        x1_true, y1_true = true_box[0], true_box[1]
        x2_true, y2_true = x1_true + true_box[2], y1_true + true_box[3]
        
        x1_pred, y1_pred = pred_box[0], pred_box[1]
        x2_pred, y2_pred = x1_pred + pred_box[2], y1_pred + pred_box[3]
        
        # Calculate intersection
        x1_inter = max(x1_true, x1_pred)
        y1_inter = max(y1_true, y1_pred)
        x2_inter = min(x2_true, x2_pred)
        y2_inter = min(y2_true, y2_pred)
        
        # Calculate areas
        w_inter = max(0, x2_inter - x1_inter)
        h_inter = max(0, y2_inter - y1_inter)
        area_inter = w_inter * h_inter
        
        area_true = true_box[2] * true_box[3]
        area_pred = pred_box[2] * pred_box[3]
        area_union = area_true + area_pred - area_inter
        
        # IoU
        iou = area_inter / area_union if area_union > 0 else 0
        iou_values.append(iou)
    
    # Create basic metrics
    results = {
        'mean_iou': np.mean(iou_values),
        'median_iou': np.median(iou_values),
        'min_iou': np.min(iou_values),
        'max_iou': np.max(iou_values),
        'std_iou': np.std(iou_values),
        'iou_values': iou_values
    }
    
    # Add mock size-based performance statistics
    # Define size thresholds
    small_threshold = 0.03
    large_threshold = 0.1
    
    # Categorize by size
    small_ious = []
    medium_ious = []
    large_ious = []
    
    for i in range(len(y_val)):
        true_area = y_val[i, 2] * y_val[i, 3]
        
        if true_area < small_threshold:
            small_ious.append(iou_values[i])
        elif true_area > large_threshold:
            large_ious.append(iou_values[i])
        else:
            medium_ious.append(iou_values[i])
    
    # Add size-based metrics
    if small_ious:
        results['small_mean_iou'] = np.mean(small_ious)
        results['small_count'] = len(small_ious)
        results['small_map50'] = np.mean([1.0 if iou >= 0.5 else 0.0 for iou in small_ious])
    
    if medium_ious:
        results['medium_mean_iou'] = np.mean(medium_ious)
        results['medium_count'] = len(medium_ious)
        results['medium_map50'] = np.mean([1.0 if iou >= 0.5 else 0.0 for iou in medium_ious])
    
    if large_ious:
        results['large_mean_iou'] = np.mean(large_ious)
        results['large_count'] = len(large_ious)
        results['large_map50'] = np.mean([1.0 if iou >= 0.5 else 0.0 for iou in large_ious])
    
    # Add coordinate errors
    results['mean_x_error'] = np.mean([abs(y_val[i, 0] - y_pred[i, 0]) for i in range(len(y_val))])
    results['mean_y_error'] = np.mean([abs(y_val[i, 1] - y_pred[i, 1]) for i in range(len(y_val))])
    results['mean_w_error'] = np.mean([abs(y_val[i, 2] - y_pred[i, 2]) for i in range(len(y_val))])
    results['mean_h_error'] = np.mean([abs(y_val[i, 3] - y_pred[i, 3]) for i in range(len(y_val))])
    
    # Calculate center errors
    center_errors = []
    for i in range(len(y_val)):
        true_center_x = y_val[i, 0] + y_val[i, 2]/2
        true_center_y = y_val[i, 1] + y_val[i, 3]/2
        
        pred_center_x = y_pred[i, 0] + y_pred[i, 2]/2
        pred_center_y = y_pred[i, 1] + y_pred[i, 3]/2
        
        error = np.sqrt((true_center_x - pred_center_x)**2 + (true_center_y - pred_center_y)**2)
        center_errors.append(error)
    
    results['mean_center_error'] = np.mean(center_errors)
    
    # Calculate size errors
    size_errors = []
    for i in range(len(y_val)):
        true_size = y_val[i, 2] * y_val[i, 3]
        pred_size = y_pred[i, 2] * y_pred[i, 3]
        
        error = abs(true_size - pred_size) / true_size if true_size > 0 else 0
        size_errors.append(error)
    
    results['mean_size_error'] = np.mean(size_errors)
    
    # Add mAP metrics
    iou_thresholds = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
    ap_values = []
    
    for thresh in iou_thresholds:
        ap = np.mean([1.0 if iou >= thresh else 0.0 for iou in iou_values])
        ap_values.append(ap)
    
    results['map50'] = ap_values[0]  # mAP@0.5
    results['map'] = np.mean(ap_values)  # mAP@0.5:0.95
    
    return results
