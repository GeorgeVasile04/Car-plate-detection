"""
Error analysis module for license plate detection models.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error
from pathlib import Path
import tensorflow as tf
import cv2
from collections import defaultdict

# Import visualization utilities
from license_plate_detection.utils.visualization import calculate_iou, visualize_prediction


def analyze_predictions(y_true, y_pred, iou_threshold=0.5):
    """
    Analyze predictions and calculate various metrics.
    
    Args:
        y_true: Ground truth bounding boxes [batch, (x, y, width, height)]
        y_pred: Predicted bounding boxes [batch, (x, y, width, height)]
        iou_threshold: Threshold for considering a detection correct
        
    Returns:
        dict: Dictionary of metrics
    """
    # Convert to numpy arrays if needed
    if isinstance(y_true, tf.Tensor):
        y_true = y_true.numpy()
    if isinstance(y_pred, tf.Tensor):
        y_pred = y_pred.numpy()
    
    # Initialize metrics
    metrics = {
        'iou_scores': [],
        'correct_detections': 0,
        'position_errors': [],
        'size_errors': [],
        'mse': {},
        'mae': {}
    }
    
    # Calculate IoU for each prediction
    for i in range(len(y_true)):
        true_box = y_true[i]
        pred_box = y_pred[i]
        
        # Calculate IoU
        iou = calculate_iou(true_box, pred_box)
        metrics['iou_scores'].append(iou)
        
        # Check if detection is correct
        if iou >= iou_threshold:
            metrics['correct_detections'] += 1
        
        # Calculate position error (L2 distance between centers)
        true_center_x = true_box[0] + true_box[2] / 2
        true_center_y = true_box[1] + true_box[3] / 2
        pred_center_x = pred_box[0] + pred_box[2] / 2
        pred_center_y = pred_box[1] + pred_box[3] / 2
        
        position_error = np.sqrt((true_center_x - pred_center_x) ** 2 + 
                                 (true_center_y - pred_center_y) ** 2)
        metrics['position_errors'].append(position_error)
        
        # Calculate size error (absolute difference in area)
        true_area = true_box[2] * true_box[3]
        pred_area = pred_box[2] * pred_box[3]
        size_error = abs(true_area - pred_area)
        metrics['size_errors'].append(size_error)
    
    # Calculate accuracy
    metrics['accuracy'] = metrics['correct_detections'] / len(y_true)
    
    # Calculate mean IoU
    metrics['mean_iou'] = np.mean(metrics['iou_scores'])
    
    # Calculate mean position and size errors
    metrics['mean_position_error'] = np.mean(metrics['position_errors'])
    metrics['mean_size_error'] = np.mean(metrics['size_errors'])
    
    # Calculate MSE and MAE for each component
    for i, component in enumerate(['x', 'y', 'width', 'height']):
        true_component = y_true[:, i]
        pred_component = y_pred[:, i]
        
        metrics['mse'][component] = mean_squared_error(true_component, pred_component)
        metrics['mae'][component] = mean_absolute_error(true_component, pred_component)
    
    # Calculate overall MSE and MAE
    metrics['mse']['overall'] = mean_squared_error(y_true, y_pred)
    metrics['mae']['overall'] = mean_absolute_error(y_true, y_pred)
    
    return metrics


def identify_error_patterns(y_true, y_pred, images=None, iou_threshold=0.5):
    """
    Identify patterns in prediction errors.
    
    Args:
        y_true: Ground truth bounding boxes [batch, (x, y, width, height)]
        y_pred: Predicted bounding boxes [batch, (x, y, width, height)]
        images: Optional batch of images for visualization
        iou_threshold: Threshold for considering a detection correct
        
    Returns:
        dict: Dictionary of error patterns and analysis
    """
    # Convert to numpy arrays if needed
    if isinstance(y_true, tf.Tensor):
        y_true = y_true.numpy()
    if isinstance(y_pred, tf.Tensor):
        y_pred = y_pred.numpy()
    
    # Initialize error patterns
    patterns = {
        'positional_bias': {
            'x_bias': 0,
            'y_bias': 0
        },
        'size_bias': {
            'width_bias': 0,
            'height_bias': 0
        },
        'error_by_true_size': defaultdict(list),
        'error_by_position': defaultdict(list),
        'worst_predictions': [],
        'best_predictions': []
    }
    
    # Calculate errors for each prediction
    errors = []
    for i in range(len(y_true)):
        true_box = y_true[i]
        pred_box = y_pred[i]
        
        # Calculate IoU
        iou = calculate_iou(true_box, pred_box)
        
        # Calculate component-wise errors
        x_error = pred_box[0] - true_box[0]
        y_error = pred_box[1] - true_box[1]
        width_error = pred_box[2] - true_box[2]
        height_error = pred_box[3] - true_box[3]
        
        # Calculate true box properties
        true_area = true_box[2] * true_box[3]
        true_center_x = true_box[0] + true_box[2] / 2
        true_center_y = true_box[1] + true_box[3] / 2
        true_aspect_ratio = true_box[2] / true_box[3] if true_box[3] > 0 else 0
        
        # Accumulate biases
        patterns['positional_bias']['x_bias'] += x_error
        patterns['positional_bias']['y_bias'] += y_error
        patterns['size_bias']['width_bias'] += width_error
        patterns['size_bias']['height_bias'] += height_error
        
        # Group errors by size
        size_category = categorize_size(true_area)
        patterns['error_by_true_size'][size_category].append(iou)
        
        # Group errors by position
        position_category = categorize_position(true_center_x, true_center_y)
        patterns['error_by_position'][position_category].append(iou)
        
        # Add to errors list for sorting
        errors.append({
            'index': i,
            'iou': iou,
            'x_error': x_error,
            'y_error': y_error,
            'width_error': width_error,
            'height_error': height_error,
            'true_box': true_box,
            'pred_box': pred_box,
            'true_area': true_area,
            'true_aspect_ratio': true_aspect_ratio
        })
    
    # Calculate average biases
    n = len(y_true)
    patterns['positional_bias']['x_bias'] /= n
    patterns['positional_bias']['y_bias'] /= n
    patterns['size_bias']['width_bias'] /= n
    patterns['size_bias']['height_bias'] /= n
    
    # Sort errors by IoU
    errors_sorted = sorted(errors, key=lambda x: x['iou'])
    
    # Get worst predictions
    patterns['worst_predictions'] = errors_sorted[:min(10, n)]
    
    # Get best predictions
    patterns['best_predictions'] = errors_sorted[-min(10, n):]
    
    # Calculate average IoU by size category
    for category, ious in patterns['error_by_true_size'].items():
        patterns['error_by_true_size'][category] = np.mean(ious)
    
    # Calculate average IoU by position category
    for category, ious in patterns['error_by_position'].items():
        patterns['error_by_position'][category] = np.mean(ious)
    
    # Provide summary
    patterns['summary'] = {
        'positional_bias': f"X bias: {patterns['positional_bias']['x_bias']:.4f}, Y bias: {patterns['positional_bias']['y_bias']:.4f}",
        'size_bias': f"Width bias: {patterns['size_bias']['width_bias']:.4f}, Height bias: {patterns['size_bias']['height_bias']:.4f}",
        'worst_iou': errors_sorted[0]['iou'] if errors_sorted else 0,
        'best_iou': errors_sorted[-1]['iou'] if errors_sorted else 0
    }
    
    return patterns


def categorize_size(area):
    """
    Categorize a bounding box by its area.
    
    Args:
        area: Area of the bounding box (normalized to [0, 1])
        
    Returns:
        str: Size category
    """
    if area < 0.05:
        return 'very_small'
    elif area < 0.1:
        return 'small'
    elif area < 0.2:
        return 'medium'
    elif area < 0.3:
        return 'large'
    else:
        return 'very_large'


def categorize_position(center_x, center_y):
    """
    Categorize a bounding box by the position of its center.
    
    Args:
        center_x: X-coordinate of center (normalized to [0, 1])
        center_y: Y-coordinate of center (normalized to [0, 1])
        
    Returns:
        str: Position category
    """
    # Determine horizontal position
    if center_x < 0.33:
        h_pos = 'left'
    elif center_x < 0.66:
        h_pos = 'center'
    else:
        h_pos = 'right'
    
    # Determine vertical position
    if center_y < 0.33:
        v_pos = 'top'
    elif center_y < 0.66:
        v_pos = 'middle'
    else:
        v_pos = 'bottom'
    
    return f"{v_pos}_{h_pos}"


def visualize_error_patterns(patterns, images=None, y_true=None, y_pred=None, save_dir=None):
    """
    Visualize error patterns.
    
    Args:
        patterns: Error patterns dictionary from identify_error_patterns
        images: Optional batch of images for visualization
        y_true: Ground truth bounding boxes (for visualization)
        y_pred: Predicted bounding boxes (for visualization)
        save_dir: Optional directory to save visualizations
        
    Returns:
        list: List of figure objects
    """
    figures = []
    
    # Create save directory if needed
    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
    
    # Plot positional and size biases
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Positional bias
    pos_bias = patterns['positional_bias']
    axes[0].bar(['X Bias', 'Y Bias'], [pos_bias['x_bias'], pos_bias['y_bias']])
    axes[0].set_title('Positional Bias')
    axes[0].axhline(y=0, color='r', linestyle='-', alpha=0.3)
    axes[0].grid(True, alpha=0.3)
    
    # Size bias
    size_bias = patterns['size_bias']
    axes[1].bar(['Width Bias', 'Height Bias'], 
               [size_bias['width_bias'], size_bias['height_bias']])
    axes[1].set_title('Size Bias')
    axes[1].axhline(y=0, color='r', linestyle='-', alpha=0.3)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    if save_dir:
        plt.savefig(save_dir / 'bias_analysis.png', bbox_inches='tight', dpi=150)
    figures.append(fig)
    
    # Plot IoU by size category
    fig, ax = plt.subplots(figsize=(10, 6))
    size_categories = ['very_small', 'small', 'medium', 'large', 'very_large']
    size_ious = [patterns['error_by_true_size'].get(cat, 0) for cat in size_categories]
    
    ax.bar(size_categories, size_ious)
    ax.set_title('Mean IoU by Bounding Box Size')
    ax.set_ylim(0, 1)
    ax.set_xlabel('Size Category')
    ax.set_ylabel('Mean IoU')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    if save_dir:
        plt.savefig(save_dir / 'iou_by_size.png', bbox_inches='tight', dpi=150)
    figures.append(fig)
    
    # Plot IoU by position
    fig, ax = plt.subplots(figsize=(15, 8))
    
    # Get position categories and IoUs
    position_categories = sorted(patterns['error_by_position'].keys())
    position_ious = [patterns['error_by_position'][cat] for cat in position_categories]
    
    ax.bar(position_categories, position_ious)
    ax.set_title('Mean IoU by Bounding Box Position')
    ax.set_ylim(0, 1)
    ax.set_xlabel('Position Category')
    ax.set_ylabel('Mean IoU')
    ax.tick_params(axis='x', rotation=45)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    if save_dir:
        plt.savefig(save_dir / 'iou_by_position.png', bbox_inches='tight', dpi=150)
    figures.append(fig)
    
    # Visualize worst predictions if images are provided
    if images is not None and y_true is not None and y_pred is not None:
        worst_pred_dir = None
        if save_dir:
            worst_pred_dir = save_dir / 'worst_predictions'
            worst_pred_dir.mkdir(exist_ok=True)
        
        worst_figs = []
        for i, pred in enumerate(patterns['worst_predictions'][:5]):
            idx = pred['index']
            image = images[idx]
            true_box = pred['true_box']
            pred_box = pred['pred_box']
            
            save_path = None
            if worst_pred_dir:
                save_path = worst_pred_dir / f"worst_{i+1}.png"
            
            fig = visualize_prediction(
                image, true_box, pred_box,
                title=f"Worst Prediction {i+1}: IoU = {pred['iou']:.4f}",
                save_path=save_path
            )
            worst_figs.append(fig)
        
        figures.extend(worst_figs)
    
    return figures


def compare_models(model_names, metrics_list, colors=None, figsize=(15, 10), save_path=None):
    """
    Compare performance metrics for multiple models.
    
    Args:
        model_names: List of model names
        metrics_list: List of metrics dictionaries from analyze_predictions
        colors: Optional list of colors for plotting
        figsize: Figure size
        save_path: Optional path to save the plot
        
    Returns:
        matplotlib.figure.Figure: Figure object
    """
    # Check input lengths
    assert len(model_names) == len(metrics_list), "Number of model names must match number of metrics"
    
    # Set default colors if not provided
    if colors is None:
        colors = plt.cm.tab10(np.linspace(0, 1, len(model_names)))
    
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    
    # Flatten axes for easier indexing
    axes = axes.flatten()
    
    # Metrics to compare
    metrics_to_compare = [
        ('mean_iou', 'Mean IoU'),
        ('accuracy', 'Accuracy'),
        ('mean_position_error', 'Mean Position Error'),
        ('mean_size_error', 'Mean Size Error')
    ]
    
    # Plot each metric
    for i, (metric_key, metric_name) in enumerate(metrics_to_compare):
        ax = axes[i]
        
        # Get metric values for each model
        metric_values = []
        for metrics in metrics_list:
            metric_values.append(metrics.get(metric_key, 0))
        
        # Plot bars
        bars = ax.bar(model_names, metric_values, color=colors)
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.4f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom')
        
        # Add labels and title
        ax.set_title(metric_name)
        ax.set_ylim(0, max(metric_values) * 1.2)
        
        # For error metrics, lower is better
        if 'error' in metric_key:
            ax.invert_yaxis()
        
        # Rotate x-axis labels if there are many models
        if len(model_names) > 3:
            ax.tick_params(axis='x', rotation=45)
        
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save figure if path provided
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
    
    return fig


def confusion_matrix_analysis(y_true, y_pred, iou_thresholds=[0.3, 0.5, 0.7], figsize=(15, 5), save_path=None):
    """
    Analyze detection performance at different IoU thresholds.
    
    Args:
        y_true: Ground truth bounding boxes [batch, (x, y, width, height)]
        y_pred: Predicted bounding boxes [batch, (x, y, width, height)]
        iou_thresholds: List of IoU thresholds
        figsize: Figure size
        save_path: Optional path to save the plot
        
    Returns:
        tuple: (Figure object, DataFrame with results)
    """
    # Convert to numpy arrays if needed
    if isinstance(y_true, tf.Tensor):
        y_true = y_true.numpy()
    if isinstance(y_pred, tf.Tensor):
        y_pred = y_pred.numpy()
    
    # Initialize results
    results = {
        'threshold': [],
        'precision': [],
        'recall': [],
        'f1_score': []
    }
    
    # Calculate IoU for each prediction
    iou_scores = []
    for i in range(len(y_true)):
        true_box = y_true[i]
        pred_box = y_pred[i]
        iou = calculate_iou(true_box, pred_box)
        iou_scores.append(iou)
    
    # Calculate metrics for each threshold
    for threshold in iou_thresholds:
        # Calculate true positives, false positives, false negatives
        tp = sum(iou >= threshold for iou in iou_scores)
        fp = sum(iou < threshold for iou in iou_scores)
        fn = 0  # For single bbox per image, this is 0
        
        # Calculate precision and recall
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 1
        
        # Calculate F1 score
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        # Add to results
        results['threshold'].append(threshold)
        results['precision'].append(precision)
        results['recall'].append(recall)
        results['f1_score'].append(f1)
    
    # Convert to DataFrame
    df_results = pd.DataFrame(results)
    
    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    # Plot metrics vs threshold
    metrics = ['precision', 'recall', 'f1_score']
    titles = ['Precision', 'Recall', 'F1 Score']
    
    for i, (metric, title) in enumerate(zip(metrics, titles)):
        axes[i].plot(df_results['threshold'], df_results[metric], 'o-', linewidth=2)
        axes[i].set_xlabel('IoU Threshold')
        axes[i].set_ylabel(title)
        axes[i].set_title(f'{title} vs IoU Threshold')
        axes[i].grid(True, alpha=0.3)
        axes[i].set_ylim(0, 1)
    
    plt.tight_layout()
    
    # Save figure if path provided
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
    
    return fig, df_results


def advanced_error_visualization(images, y_true, y_pred, indices=None, save_dir=None):
    """
    Create advanced visualizations of prediction errors.
    
    Args:
        images: Batch of images
        y_true: Ground truth bounding boxes [batch, (x, y, width, height)]
        y_pred: Predicted bounding boxes [batch, (x, y, width, height)]
        indices: Specific indices to visualize (if None, selects interesting examples)
        save_dir: Optional directory to save visualizations
        
    Returns:
        list: List of figure objects
    """
    # Convert to numpy arrays if needed
    if isinstance(y_true, tf.Tensor):
        y_true = y_true.numpy()
    if isinstance(y_pred, tf.Tensor):
        y_pred = y_pred.numpy()
    
    # Calculate IoU for all predictions
    iou_scores = []
    for i in range(len(y_true)):
        true_box = y_true[i]
        pred_box = y_pred[i]
        iou = calculate_iou(true_box, pred_box)
        iou_scores.append((i, iou))
    
    # If indices not provided, select interesting examples
    if indices is None:
        # Sort by IoU
        iou_scores.sort(key=lambda x: x[1])
        
        # Get worst examples
        worst_indices = [idx for idx, _ in iou_scores[:3]]
        
        # Get best examples
        best_indices = [idx for idx, _ in iou_scores[-3:]]
        
        # Get medium examples (around median IoU)
        median_idx = len(iou_scores) // 2
        medium_indices = [idx for idx, _ in iou_scores[median_idx-1:median_idx+2]]
        
        # Combine indices
        indices = worst_indices + medium_indices + best_indices
    
    # Create save directory if needed
    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
    
    # Create visualizations
    figures = []
    for i, idx in enumerate(indices):
        # Get image and boxes
        image = images[idx]
        true_box = y_true[idx]
        pred_box = y_pred[idx]
        
        # Calculate IoU
        iou = calculate_iou(true_box, pred_box)
        
        # Calculate error types
        center_error = np.sqrt(
            ((true_box[0] + true_box[2]/2) - (pred_box[0] + pred_box[2]/2))**2 +
            ((true_box[1] + true_box[3]/2) - (pred_box[1] + pred_box[3]/2))**2
        )
        
        size_error = abs(true_box[2] * true_box[3] - pred_box[2] * pred_box[3])
        
        # Determine error type
        if center_error > 0.1 and size_error < 0.05:
            error_type = "Position Error"
        elif center_error < 0.05 and size_error > 0.1:
            error_type = "Size Error"
        elif center_error > 0.1 and size_error > 0.1:
            error_type = "Position & Size Error"
        else:
            error_type = "Good Prediction"
        
        # Set save path if directory provided
        save_path = None
        if save_dir:
            save_path = save_dir / f"error_viz_{i}.png"
        
        # Create visualization
        fig = visualize_prediction(
            image, true_box, pred_box,
            figsize=(10, 8),
            title=f"IoU: {iou:.4f} - {error_type}\nCenter Error: {center_error:.4f}, Size Error: {size_error:.4f}",
            save_path=save_path
        )
        
        figures.append(fig)
    
    return figures


def analyze_error_distribution(y_true, y_pred):
    """
    Analyze the distribution of errors.
    
    Args:
        y_true: Ground truth bounding boxes [batch, (x, y, width, height)]
        y_pred: Predicted bounding boxes [batch, (x, y, width, height)]
        
    Returns:
        tuple: (Figure object, Dictionary with error statistics)
    """
    # Convert to numpy arrays if needed
    if isinstance(y_true, tf.Tensor):
        y_true = y_true.numpy()
    if isinstance(y_pred, tf.Tensor):
        y_pred = y_pred.numpy()
    
    # Calculate errors for each component
    x_errors = y_pred[:, 0] - y_true[:, 0]
    y_errors = y_pred[:, 1] - y_true[:, 1]
    width_errors = y_pred[:, 2] - y_true[:, 2]
    height_errors = y_pred[:, 3] - y_true[:, 3]
    
    # Calculate statistics
    stats = {
        'x': {
            'mean': np.mean(x_errors),
            'std': np.std(x_errors),
            'min': np.min(x_errors),
            'max': np.max(x_errors),
            'median': np.median(x_errors)
        },
        'y': {
            'mean': np.mean(y_errors),
            'std': np.std(y_errors),
            'min': np.min(y_errors),
            'max': np.max(y_errors),
            'median': np.median(y_errors)
        },
        'width': {
            'mean': np.mean(width_errors),
            'std': np.std(width_errors),
            'min': np.min(width_errors),
            'max': np.max(width_errors),
            'median': np.median(width_errors)
        },
        'height': {
            'mean': np.mean(height_errors),
            'std': np.std(height_errors),
            'min': np.min(height_errors),
            'max': np.max(height_errors),
            'median': np.median(height_errors)
        }
    }
    
    # Create figure for error distributions
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot error distributions
    components = ['x', 'y', 'width', 'height']
    errors = [x_errors, y_errors, width_errors, height_errors]
    
    for i, (component, error) in enumerate(zip(components, errors)):
        row, col = i // 2, i % 2
        ax = axes[row, col]
        
        # Plot histogram with KDE
        sns.histplot(error, kde=True, ax=ax)
        
        # Add vertical line at zero
        ax.axvline(0, color='r', linestyle='--', alpha=0.7)
        
        # Add mean and median lines
        ax.axvline(stats[component]['mean'], color='g', linestyle='-', alpha=0.7,
                 label=f"Mean: {stats[component]['mean']:.4f}")
        ax.axvline(stats[component]['median'], color='b', linestyle='-', alpha=0.7,
                 label=f"Median: {stats[component]['median']:.4f}")
        
        # Add title and labels
        ax.set_title(f"{component.capitalize()} Error Distribution")
        ax.set_xlabel(f"{component.capitalize()} Error")
        ax.set_ylabel("Frequency")
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    return fig, stats
