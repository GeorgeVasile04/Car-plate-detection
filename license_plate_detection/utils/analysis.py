"""
Error analysis tools for license plate detection.
"""

import numpy as np
import matplotlib.pyplot as plt
import cv2

def analyze_error_patterns(model, X_val, y_val, y_pred=None, plate_sizes=None):
    """
    Analyze error patterns to understand where the model struggles
    and provide insights for further improvements.
    
    Args:
        model: Trained detector model (can be None if y_pred is provided)
        X_val: Validation images
        y_val: Ground truth bounding boxes
        y_pred: Optional pre-computed predictions (if None, will use model to predict)
        plate_sizes: Optional list of plate sizes (areas)
        
    Returns:
        dict: Dictionary with error metrics and analysis results
    """
    print("Analyzing error patterns...")
    
    # Get predictions if not provided
    if y_pred is None and model is not None:
        y_pred = model.predict(X_val, verbose=1)
    elif y_pred is None and model is None:
        raise ValueError("Either model or y_pred must be provided")
    
    # If plate sizes not provided, calculate them
    if plate_sizes is None:
        plate_sizes = [box[2] * box[3] for box in y_val]
    
    # Calculate errors and IoU
    iou_values = []
    x_errors = []
    y_errors = []
    w_errors = []
    h_errors = []
    center_errors = []
    area_errors = []
    
    for i in range(len(y_val)):
        true_box = y_val[i]
        pred_box = y_pred[i]
        
        # Basic coordinate errors
        x_errors.append(abs(true_box[0] - pred_box[0]))
        y_errors.append(abs(true_box[1] - pred_box[1]))
        w_errors.append(abs(true_box[2] - pred_box[2]))
        h_errors.append(abs(true_box[3] - pred_box[3]))
        
        # Center point error
        true_center_x = true_box[0] + true_box[2]/2
        true_center_y = true_box[1] + true_box[3]/2
        pred_center_x = pred_box[0] + pred_box[2]/2
        pred_center_y = pred_box[1] + pred_box[3]/2
        center_errors.append(np.sqrt((true_center_x - pred_center_x)**2 + 
                                    (true_center_y - pred_center_y)**2))
        
        # Area error
        true_area = true_box[2] * true_box[3]
        pred_area = pred_box[2] * pred_box[3]
        area_errors.append(abs(true_area - pred_area) / true_area)
        
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
        w_intersect = max(0, x2_inter - x1_inter)
        h_intersect = max(0, y2_inter - y1_inter)
        area_intersect = w_intersect * h_intersect
        
        area_true = true_box[2] * true_box[3]
        area_pred = pred_box[2] * pred_box[3]
        area_union = area_true + area_pred - area_intersect
        
        # IoU
        iou = area_intersect / area_union if area_union > 0 else 0
        iou_values.append(iou)
    
    # Characterize plates by size
    small_threshold = 0.03
    large_threshold = 0.1
    
    small_indices = [i for i, size in enumerate(plate_sizes) if size < small_threshold]
    medium_indices = [i for i, size in enumerate(plate_sizes) if small_threshold <= size <= large_threshold]
    large_indices = [i for i, size in enumerate(plate_sizes) if size > large_threshold]
    
    # Calculate error metrics by plate size
    def get_metrics_by_indices(indices, name):
        if not indices:
            print(f"No {name} plates found")
            return None
            
        size_iou = [iou_values[i] for i in indices]
        size_x_err = [x_errors[i] for i in indices]
        size_y_err = [y_errors[i] for i in indices]
        size_w_err = [w_errors[i] for i in indices]
        size_h_err = [h_errors[i] for i in indices]
        size_center_err = [center_errors[i] for i in indices]
        size_area_err = [area_errors[i] for i in indices]
        
        print(f"\n{name} Plates (n={len(indices)}):")
        print(f"  Mean IoU: {np.mean(size_iou):.4f}")
        print(f"  Center Error: {np.mean(size_center_err):.4f}")
        print(f"  Area Error: {np.mean(size_area_err):.4f}")
        print(f"  X Error: {np.mean(size_x_err):.4f}, Y Error: {np.mean(size_y_err):.4f}")
        print(f"  Width Error: {np.mean(size_w_err):.4f}, Height Error: {np.mean(size_h_err):.4f}")
        
        return {
            'mean_iou': np.mean(size_iou),
            'center_error': np.mean(size_center_err),
            'area_error': np.mean(size_area_err),
            'x_error': np.mean(size_x_err),
            'y_error': np.mean(size_y_err),
            'w_error': np.mean(size_w_err),
            'h_error': np.mean(size_h_err)
        }
    
    # Get metrics by plate size
    small_metrics = get_metrics_by_indices(small_indices, "Small")
    medium_metrics = get_metrics_by_indices(medium_indices, "Medium")
    large_metrics = get_metrics_by_indices(large_indices, "Large")
    
    # Analyze correlation between error types and IoU
    plt.figure(figsize=(18, 5))
    
    # 1. Scatter plot of center error vs IoU
    plt.subplot(1, 3, 1)
    plt.scatter(center_errors, iou_values, alpha=0.6)
    plt.title('Center Error vs IoU')
    plt.xlabel('Center Error')
    plt.ylabel('IoU')
    plt.grid(True, linestyle='--', alpha=0.5)
    
    # Calculate and show correlation
    center_corr = np.corrcoef(center_errors, iou_values)[0, 1]
    plt.text(0.05, 0.05, f"Correlation: {center_corr:.3f}", transform=plt.gca().transAxes)
    
    # 2. Scatter plot of area error vs IoU
    plt.subplot(1, 3, 2)
    plt.scatter(area_errors, iou_values, alpha=0.6)
    plt.title('Area Error vs IoU')
    plt.xlabel('Relative Area Error')
    plt.ylabel('IoU')
    plt.grid(True, linestyle='--', alpha=0.5)
    
    # Calculate and show correlation
    area_corr = np.corrcoef(area_errors, iou_values)[0, 1]
    plt.text(0.05, 0.05, f"Correlation: {area_corr:.3f}", transform=plt.gca().transAxes)
    
    # 3. Scatter plot of plate size vs IoU
    plt.subplot(1, 3, 3)
    plt.scatter(plate_sizes, iou_values, alpha=0.6)
    plt.title('Plate Size vs IoU')
    plt.xlabel('Plate Size (Area)')
    plt.ylabel('IoU')
    plt.grid(True, linestyle='--', alpha=0.5)
    
    # Calculate and show correlation
    size_corr = np.corrcoef(plate_sizes, iou_values)[0, 1]
    plt.text(0.05, 0.05, f"Correlation: {size_corr:.3f}", transform=plt.gca().transAxes)
    
    plt.tight_layout()
    plt.show()
    
    # Show error distribution comparison between coordinates
    plt.figure(figsize=(15, 5))
    
    # Get error percentiles for box plot
    error_data = [x_errors, y_errors, w_errors, h_errors]
    
    plt.boxplot(error_data, labels=['X', 'Y', 'Width', 'Height'])
    plt.title('Coordinate Error Distribution')
    plt.ylabel('Error')
    plt.grid(True, linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.show()
    
    # Provide insights based on analysis
    print("\n===== ERROR ANALYSIS INSIGHTS =====")
    
    # Check main source of error
    mean_x_error = np.mean(x_errors)
    mean_y_error = np.mean(y_errors)
    mean_w_error = np.mean(w_errors)
    mean_h_error = np.mean(h_errors)
    mean_center_error = np.mean(center_errors)
    mean_area_error = np.mean(area_errors)
    
    # Find the largest error
    error_types = ['position (x/y)', 'width', 'height', 'center point', 'area/size']
    error_values = [max(mean_x_error, mean_y_error), mean_w_error, mean_h_error, mean_center_error, mean_area_error]
    main_error_type = error_types[np.argmax(error_values)]
    
    print(f"1. Main source of error is in the {main_error_type}")
    
    # Check size-specific issues
    if small_metrics and medium_metrics and large_metrics:
        worst_size = min([
            ('small', small_metrics['mean_iou']), 
            ('medium', medium_metrics['mean_iou']), 
            ('large', large_metrics['mean_iou'])
        ], key=lambda x: x[1])[0]
        
        print(f"2. Model performs worst on {worst_size} license plates")
        
        # Check if width/height ratio is a problem (different for different sizes)
        small_w_h_ratio = small_metrics['w_error'] / small_metrics['h_error']
        large_w_h_ratio = large_metrics['w_error'] / large_metrics['h_error']
        
        if abs(small_w_h_ratio - large_w_h_ratio) > 0.3:
            print("3. Model has inconsistent width/height prediction across different plate sizes")
    
    # Check correlation-based insights
    if center_corr < -0.7:
        print("4. Center point localization is a critical factor in model performance")
    if area_corr < -0.7:
        print("5. Size prediction is a critical factor in model performance")
    if abs(size_corr) > 0.3:
        if size_corr > 0:
            print("6. Model performs better on larger plates")
        else:
            print("6. Model performs better on smaller plates")
            
    # Final recommendations
    print("\n===== IMPROVEMENT RECOMMENDATIONS =====")
    
    # Based on main error type
    if main_error_type == 'position (x/y)':
        print("1. Focus on improving positional accuracy:")
        print("   - Add more positional penalties in the loss function")
        print("   - Consider using anchor points for better localization")
        print("   - Add more spatial context in the model architecture")
        
    elif main_error_type in ['width', 'height', 'area/size']:
        print("1. Focus on improving size estimation:")
        print("   - Increase weight for width/height in the loss function")
        print("   - Consider multi-scale feature fusion techniques")
        print("   - Add size-specific regularization terms")
        
    # General recommendations
    print("\n2. General model improvements:")
    print("   - Increase training time (more epochs)")
    print("   - Add more diverse data augmentation focusing on size variations")
    print("   - Consider two-stage detection (region proposal + regression)")
    print("   - Experiment with anchor-based detection approach")
    
    # Size-specific recommendations
    if 'worst_size' in locals():
        print(f"\n3. {worst_size.capitalize()} plate-specific improvements:")
        if worst_size == 'small':
            print("   - Add more small plate examples through augmentation")
            print("   - Use feature pyramid networks for better small object detection")
            print("   - Consider higher resolution input images")
        elif worst_size == 'large':
            print("   - Add more large plate examples")
            print("   - Use dilated convolutions for larger receptive field")
        else:
            print("   - Balance the dataset with more medium-sized examples")
            print("   - Review medium plate examples for potential issues")
    
    # Return results dictionary
    return {
        'metrics': {
            'mean_iou': np.mean(iou_values),
            'median_iou': np.median(iou_values),
            'min_iou': np.min(iou_values),
            'max_iou': np.max(iou_values),
            'mean_center_error': mean_center_error,
            'mean_area_error': mean_area_error,
            'mean_x_error': mean_x_error,
            'mean_y_error': mean_y_error,
            'mean_w_error': mean_w_error,
            'mean_h_error': mean_h_error
        },
        'correlations': {
            'center_iou_corr': center_corr,
            'area_iou_corr': area_corr,
            'size_iou_corr': size_corr
        },
        'size_metrics': {
            'small': small_metrics,
            'medium': medium_metrics,
            'large': large_metrics
        },
        'main_error_type': main_error_type,
        'worst_size': worst_size if 'worst_size' in locals() else None,
        'raw_data': {
            'iou_values': iou_values,
            'x_errors': x_errors,
            'y_errors': y_errors,
            'w_errors': w_errors,
            'h_errors': h_errors,
            'center_errors': center_errors,
            'area_errors': area_errors,
            'plate_sizes': plate_sizes
        }
    }


def visualize_worst_cases(model, X_val, y_val, y_pred=None, n_samples=5, sort_by='iou'):
    """
    Visualize the worst predictions to better understand error modes
    
    Args:
        model: Trained model (can be None if y_pred is provided)
        X_val: Validation images
        y_val: Ground truth bounding boxes
        y_pred: Optional pre-computed predictions
        n_samples: Number of worst cases to visualize
        sort_by: Metric to sort by ('iou', 'center_error', 'area_error')
    """
    # Get predictions if not provided
    if y_pred is None and model is not None:
        y_pred = model.predict(X_val)
    elif y_pred is None and model is None:
        raise ValueError("Either model or y_pred must be provided")
    
    # Calculate metrics for sorting
    metrics = []
    for i in range(len(y_val)):
        true_box = y_val[i]
        pred_box = y_pred[i]
        
        # Calculate IoU
        x1_true, y1_true = true_box[0], true_box[1]
        x2_true, y2_true = x1_true + true_box[2], y1_true + true_box[3]
        
        x1_pred, y1_pred = pred_box[0], pred_box[1]
        x2_pred, y2_pred = x1_pred + pred_box[2], y1_pred + pred_box[3]
        
        # Intersection coordinates
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
        
        # Center error
        true_center_x = true_box[0] + true_box[2]/2
        true_center_y = true_box[1] + true_box[3]/2
        pred_center_x = pred_box[0] + pred_box[2]/2
        pred_center_y = pred_box[1] + pred_box[3]/2
        center_error = np.sqrt((true_center_x - pred_center_x)**2 + 
                              (true_center_y - pred_center_y)**2)
        
        # Area error
        true_area = true_box[2] * true_box[3]
        pred_area = pred_box[2] * pred_box[3]
        area_error = abs(true_area - pred_area) / true_area
        
        metrics.append({
            'index': i,
            'iou': iou,
            'center_error': center_error,
            'area_error': area_error
        })
    
    # Sort by selected metric
    if sort_by == 'iou':
        sorted_metrics = sorted(metrics, key=lambda x: x['iou'])
    elif sort_by == 'center_error':
        sorted_metrics = sorted(metrics, key=lambda x: -x['center_error'])
    elif sort_by == 'area_error':
        sorted_metrics = sorted(metrics, key=lambda x: -x['area_error'])
    else:
        raise ValueError(f"Invalid sort_by value: {sort_by}")
    
    # Get worst cases
    worst_indices = [m['index'] for m in sorted_metrics[:n_samples]]
    
    # Visualize worst cases
    plt.figure(figsize=(15, 4 * n_samples))
    
    for i, idx in enumerate(worst_indices):
        img = X_val[idx]
        true_box = y_val[idx]
        pred_box = y_pred[idx]
        
        # Get metrics for this sample
        iou = next(m['iou'] for m in metrics if m['index'] == idx)
        center_error = next(m['center_error'] for m in metrics if m['index'] == idx)
        area_error = next(m['area_error'] for m in metrics if m['index'] == idx)
        
        # Display image with both bounding boxes
        img_display = (img * 255).astype(np.uint8).copy()
        h, w = img.shape[:2]
        
        # True bbox (green)
        x, y = int(true_box[0] * w), int(true_box[1] * h)
        box_w, box_h = int(true_box[2] * w), int(true_box[3] * h)
        cv2.rectangle(img_display, (x, y), (x + box_w, y + box_h), (0, 255, 0), 2)
        
        # Pred bbox (red)
        x, y = int(pred_box[0] * w), int(pred_box[1] * h)
        box_w, box_h = int(pred_box[2] * w), int(pred_box[3] * h)
        cv2.rectangle(img_display, (x, y), (x + box_w, y + box_h), (255, 0, 0), 2)
        
        plt.subplot(n_samples, 1, i+1)
        plt.imshow(img_display)
        plt.title(f"Sample {idx} - IoU: {iou:.4f}, Center Error: {center_error:.4f}, Area Error: {area_error:.4f}")
        plt.axis('off')
    
    plt.tight_layout()
    plt.suptitle(f"Worst {n_samples} Cases by {sort_by.replace('_', ' ').title()}", fontsize=16)
    plt.subplots_adjust(top=0.95)
    plt.show()


def compare_model_performance(models, names, X_val, y_val):
    """
    Compare performance of multiple models side by side
    
    Args:
        models: List of trained models
        names: List of model names for display
        X_val: Validation images
        y_val: Ground truth bounding boxes
    """
    if len(models) != len(names):
        raise ValueError("Number of models and names must match")
    
    # Get predictions from each model
    predictions = []
    for model in models:
        predictions.append(model.predict(X_val))
    
    # Calculate metrics for each model
    metrics = []
    for i, y_pred in enumerate(predictions):
        model_metrics = {}
        
        # Calculate IoU
        iou_values = []
        for j in range(len(y_val)):
            true_box = y_val[j]
            pred_box = y_pred[j]
            
            # Convert to x1, y1, x2, y2 format
            x1_true, y1_true = true_box[0], true_box[1]
            x2_true, y2_true = x1_true + true_box[2], y1_true + true_box[3]
            
            x1_pred, y1_pred = pred_box[0], pred_box[1]
            x2_pred, y2_pred = x1_pred + pred_box[2], y1_pred + pred_box[3]
            
            # Intersection
            x1_inter = max(x1_true, x1_pred)
            y1_inter = max(y1_true, y1_pred)
            x2_inter = min(x2_true, x2_pred)
            y2_inter = min(y2_true, y2_pred)
            
            # Areas
            w_inter = max(0, x2_inter - x1_inter)
            h_inter = max(0, y2_inter - y1_inter)
            area_inter = w_inter * h_inter
            
            area_true = true_box[2] * true_box[3]
            area_pred = pred_box[2] * pred_box[3]
            area_union = area_true + area_pred - area_inter
            
            # IoU
            iou = area_inter / area_union if area_union > 0 else 0
            iou_values.append(iou)
        
        # Store metrics
        model_metrics['mean_iou'] = np.mean(iou_values)
        model_metrics['median_iou'] = np.median(iou_values)
        model_metrics['min_iou'] = np.min(iou_values)
        model_metrics['max_iou'] = np.max(iou_values)
        model_metrics['iou_values'] = iou_values
        
        metrics.append(model_metrics)
    
    # Visualize comparison
    plt.figure(figsize=(15, 8))
    
    # Bar chart of mean and median IoU
    plt.subplot(1, 2, 1)
    x = np.arange(len(names))
    width = 0.35
    
    plt.bar(x - width/2, [m['mean_iou'] for m in metrics], width, label='Mean IoU')
    plt.bar(x + width/2, [m['median_iou'] for m in metrics], width, label='Median IoU')
    
    plt.xlabel('Model')
    plt.ylabel('IoU')
    plt.title('Mean and Median IoU by Model')
    plt.xticks(x, names)
    plt.ylim(0, 1.0)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    
    # Boxplot of IoU distributions
    plt.subplot(1, 2, 2)
    plt.boxplot([m['iou_values'] for m in metrics], labels=names)
    plt.title('IoU Distribution by Model')
    plt.ylabel('IoU')
    plt.ylim(0, 1.0)
    plt.grid(True, linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.show()
    
    # Print metrics
    print("Model Comparison:")
    for i, name in enumerate(names):
        print(f"\n{name}:")
        print(f"  Mean IoU: {metrics[i]['mean_iou']:.4f}")
        print(f"  Median IoU: {metrics[i]['median_iou']:.4f}")
        print(f"  Min IoU: {metrics[i]['min_iou']:.4f}")
        print(f"  Max IoU: {metrics[i]['max_iou']:.4f}")
    
    return metrics
