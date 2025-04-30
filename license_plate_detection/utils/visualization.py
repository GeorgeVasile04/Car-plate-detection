"""
Visualization utilities for license plate detection.
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import io
from PIL import Image
import tensorflow as tf


def draw_bounding_box(image, box, color=(0, 255, 0), thickness=2, normalized=True):
    """
    Draw a bounding box on an image.
    
    Args:
        image: Image as numpy array (RGB format)
        box: Bounding box in format [x, y, width, height]
        color: Box color in BGR format
        thickness: Line thickness
        normalized: Whether box coordinates are normalized (0-1) or absolute
        
    Returns:
        Image with drawn bounding box
    """
    # Make a copy of the image to avoid modifying the original
    img_with_box = image.copy()
    
    # Convert to uint8 if needed
    if img_with_box.dtype != np.uint8:
        img_with_box = (img_with_box * 255).astype(np.uint8)
    
    # Get image dimensions
    height, width = img_with_box.shape[:2]
    
    # Get box coordinates
    x, y, w, h = box
    
    # Convert normalized coordinates to absolute if needed
    if normalized:
        x = int(x * width)
        y = int(y * height)
        w = int(w * width)
        h = int(h * height)
    
    # Calculate top-left and bottom-right corners
    x1 = int(x - w / 2)
    y1 = int(y - h / 2)
    x2 = int(x + w / 2)
    y2 = int(y + h / 2)
    
    # Ensure coordinates are within image bounds
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(width, x2)
    y2 = min(height, y2)
    
    # Convert RGB to BGR for OpenCV
    if len(img_with_box.shape) == 3 and img_with_box.shape[2] == 3:
        img_with_box = cv2.cvtColor(img_with_box, cv2.COLOR_RGB2BGR)
    
    # Draw rectangle
    cv2.rectangle(img_with_box, (x1, y1), (x2, y2), color, thickness)
    
    # Convert back to RGB for display
    if len(img_with_box.shape) == 3 and img_with_box.shape[2] == 3:
        img_with_box = cv2.cvtColor(img_with_box, cv2.COLOR_BGR2RGB)
    
    return img_with_box


def visualize_detection_result(image, true_box=None, pred_box=None, normalized=True):
    """
    Visualize license plate detection result.
    
    Args:
        image: Image as numpy array
        true_box: Ground truth bounding box [x, y, width, height]
        pred_box: Predicted bounding box [x, y, width, height]
        normalized: Whether box coordinates are normalized (0-1) or absolute
        
    Returns:
        Image with visualized detection results
    """
    # Make a copy of the image
    img_result = image.copy()
    
    # Convert to uint8 if needed
    if img_result.dtype != np.uint8:
        img_result = (img_result * 255).astype(np.uint8)
    
    # Draw ground truth box in green if available
    if true_box is not None:
        img_result = draw_bounding_box(img_result, true_box, color=(0, 255, 0), thickness=2, normalized=normalized)
    
    # Draw predicted box in blue if available
    if pred_box is not None:
        img_result = draw_bounding_box(img_result, pred_box, color=(255, 0, 0), thickness=2, normalized=normalized)
    
    return img_result


def create_detection_grid(images, true_boxes=None, pred_boxes=None, num_cols=4, figure_size=(15, 15), normalized=True):
    """
    Create a grid of images with detection results.
    
    Args:
        images: List of images as numpy arrays
        true_boxes: List of ground truth bounding boxes
        pred_boxes: List of predicted bounding boxes
        num_cols: Number of columns in the grid
        figure_size: Figure size in inches
        normalized: Whether box coordinates are normalized (0-1) or absolute
        
    Returns:
        Matplotlib figure with visualization grid
    """
    # Calculate grid dimensions
    num_images = len(images)
    num_rows = (num_images + num_cols - 1) // num_cols
    
    # Create figure and axes
    fig, axes = plt.subplots(num_rows, num_cols, figsize=figure_size)
    
    # Convert axes to array for consistent indexing
    if num_rows == 1 and num_cols == 1:
        axes = np.array([axes])
    elif num_rows == 1:
        axes = np.array([axes])
    elif num_cols == 1:
        axes = axes.reshape(-1, 1)
    
    # Flatten axes for easier iteration
    axes_flat = axes.flatten()
    
    # Plot each image
    for i in range(num_images):
        ax = axes_flat[i]
        img = images[i]
        
        # Get corresponding boxes if available
        true_box = true_boxes[i] if true_boxes is not None else None
        pred_box = pred_boxes[i] if pred_boxes is not None else None
        
        # Display image
        ax.imshow(img)
        
        # Add bounding boxes if available
        if true_box is not None or pred_box is not None:
            height, width = img.shape[:2]
            
            # Draw ground truth box
            if true_box is not None:
                x, y, w, h = true_box
                if normalized:
                    x = x * width
                    y = y * height
                    w = w * width
                    h = h * height
                
                x1 = x - w / 2
                y1 = y - h / 2
                rect = patches.Rectangle((x1, y1), w, h, linewidth=2, edgecolor='g', facecolor='none', label='Ground Truth')
                ax.add_patch(rect)
            
            # Draw predicted box
            if pred_box is not None:
                x, y, w, h = pred_box
                if normalized:
                    x = x * width
                    y = y * height
                    w = w * width
                    h = h * height
                
                x1 = x - w / 2
                y1 = y - h / 2
                rect = patches.Rectangle((x1, y1), w, h, linewidth=2, edgecolor='b', facecolor='none', label='Prediction')
                ax.add_patch(rect)
            
            # Add legend to the first plot only
            if i == 0 and true_box is not None and pred_box is not None:
                ax.legend()
        
        # Remove axis ticks
        ax.set_xticks([])
        ax.set_yticks([])
    
    # Hide any unused axes
    for i in range(num_images, len(axes_flat)):
        axes_flat[i].axis('off')
    
    plt.tight_layout()
    return fig


def plot_training_history(history, metrics=None, figsize=(12, 8)):
    """
    Plot training history for a model.
    
    Args:
        history: Training history object or dictionary
        metrics: List of metrics to plot (if None, plots all metrics)
        figsize: Figure size in inches
        
    Returns:
        Matplotlib figure with training history plots
    """
    if isinstance(history, tf.keras.callbacks.History):
        history = history.history
    
    # If metrics not specified, plot all except for validation metrics
    if metrics is None:
        metrics = [m for m in history.keys() if not m.startswith('val_')]
    
    num_metrics = len(metrics)
    fig, axes = plt.subplots(num_metrics, 1, figsize=figsize, sharex=True)
    
    # Convert to array for consistent indexing
    if num_metrics == 1:
        axes = [axes]
    
    # Plot each metric
    for i, metric in enumerate(metrics):
        ax = axes[i]
        
        # Plot training metric
        ax.plot(history[metric], label=f'Training {metric}')
        
        # Plot validation metric if available
        val_metric = f'val_{metric}'
        if val_metric in history:
            ax.plot(history[val_metric], label=f'Validation {metric}')
        
        ax.set_ylabel(metric.replace('_', ' ').title())
        ax.grid(True)
        ax.legend()
    
    # Set x-axis label on the bottom plot
    axes[-1].set_xlabel('Epoch')
    
    plt.tight_layout()
    return fig


def visualize_model_performance(model, X_test, y_test, num_samples=10, normalized=True):
    """
    Visualize model performance on test data.
    
    Args:
        model: Trained model
        X_test: Test images
        y_test: Test labels (bounding boxes)
        num_samples: Number of samples to visualize
        normalized: Whether box coordinates are normalized
        
    Returns:
        Matplotlib figure with performance visualization
    """
    # Get a subset of test data
    num_samples = min(num_samples, len(X_test))
    indices = np.random.choice(len(X_test), num_samples, replace=False)
    
    # Get images and true labels
    images = X_test[indices]
    true_boxes = y_test[indices]
    
    # Make predictions
    pred_boxes = model.predict(images)
    
    # Create visualization grid
    fig = create_detection_grid(images, true_boxes, pred_boxes, normalized=normalized)
    
    return fig


def visualize_attention_maps(model, image, layer_names=None):
    """
    Visualize attention maps for a given image.
    
    Args:
        model: Model with attention layers
        image: Input image (can be a numpy array or tensor)
        layer_names: Names of layers to visualize (if None, tries to find attention layers)
        
    Returns:
        Matplotlib figure with attention maps
    """
    # Ensure image is a tensor with batch dimension
    if isinstance(image, np.ndarray):
        if len(image.shape) == 3:
            image = np.expand_dims(image, axis=0)
        image_tensor = tf.convert_to_tensor(image)
    else:
        image_tensor = image
    
    # If layer names not provided, try to find attention layers
    if layer_names is None:
        layer_names = []
        for layer in model.layers:
            if 'attention' in layer.name.lower():
                layer_names.append(layer.name)
    
    if not layer_names:
        return None
    
    # Create a model that outputs the activations of specified layers
    outputs = [model.get_layer(name).output for name in layer_names]
    activation_model = tf.keras.Model(inputs=model.input, outputs=outputs)
    
    # Get activations
    activations = activation_model(image_tensor)
    if not isinstance(activations, list):
        activations = [activations]
    
    # Create figure
    num_layers = len(layer_names)
    fig, axes = plt.subplots(2, num_layers, figsize=(4 * num_layers, 8))
    
    # Display original image in the first row
    for i in range(num_layers):
        axes[0, i].imshow(image[0])
        axes[0, i].set_title('Original Image')
        axes[0, i].axis('off')
    
    # Display attention maps in the second row
    for i, activation in enumerate(activations):
        # Process activation to create a heatmap
        # This depends on the specific architecture and what the activation represents
        if len(activation.shape) == 4:  # [batch, height, width, channels]
            # Average across channels to get attention map
            attention_map = tf.reduce_mean(activation, axis=-1)[0]
        else:
            # Use as is
            attention_map = activation[0]
        
        # Normalize for visualization
        attention_map = (attention_map - tf.reduce_min(attention_map)) / (
            tf.reduce_max(attention_map) - tf.reduce_min(attention_map) + 1e-7)
        
        # Display the attention map
        axes[1, i].imshow(attention_map, cmap='hot')
        axes[1, i].set_title(f'Attention: {layer_names[i]}')
        axes[1, i].axis('off')
    
    plt.tight_layout()
    return fig
