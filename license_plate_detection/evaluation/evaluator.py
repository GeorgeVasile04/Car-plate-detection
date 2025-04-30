"""
Evaluation utilities for license plate detection.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt


def evaluate_license_plate_detection(model, X_val, y_val, df_val=None, num_samples=5):
    """
    Comprehensive evaluation of the license plate detection model with metrics matching YOLO approach
    for consistent and fair comparison between the two models.

    Args:
        model: Trained CNN model
        X_val: Validation images
        y_val: Ground truth bounding boxes
        df_val: Optional dataframe with original image paths for more advanced analysis
        num_samples: Number of best/worst samples to visualize

    Returns:
        iou_values: List of IoU values for all validation samples
    """
    # Get predictions for all validation samples
    y_pred = model.predict(X_val)

    # Calculate IoU for each sample
    iou_values = []
    pred_boxes_list = []  # For consistent naming with YOLO evaluation
    confidences_list = []  # For consistency with YOLO (using prediction max value as confidence)
    plate_sizes = []  # For plate size analysis

    for i in range(len(y_val)):
        # Extract bounding box coordinates
        true_bbox = y_val[i]
        pred_bbox = y_pred[i]

        # Store normalized plate area for size analysis
        plate_area = true_bbox[2] * true_bbox[3]  # width * height (normalized)
        plate_sizes.append(plate_area)

        # Convert to x1, y1, x2, y2 format for IoU calculation
        x1_true, y1_true = true_bbox[0], true_bbox[1]
        x2_true, y2_true = x1_true + true_bbox[2], y1_true + true_bbox[3]

        x1_pred, y1_pred = pred_bbox[0], pred_bbox[1]
        x2_pred, y2_pred = x1_pred + pred_bbox[2], y1_pred + true_bbox[3]

        # Calculate intersection
        x1_inter = max(x1_true, x1_pred)
        y1_inter = max(y1_true, y1_pred)
        x2_inter = min(x2_true, x2_pred)
        y2_inter = min(y2_true, y2_pred)

        # Calculate areas
        w_intersect = max(0, x2_inter - x1_inter)
        h_intersect = max(0, y2_inter - y1_inter)
        area_intersect = w_intersect * h_intersect

        area_true = true_bbox[2] * true_bbox[3]
        area_pred = pred_bbox[2] * pred_bbox[3]
        area_union = area_true + area_pred - area_intersect

        # IoU calculation
        iou = area_intersect / area_union if area_union > 0 else 0
        iou_values.append(iou)

        # Store prediction in format consistent with YOLO evaluation
        pred_boxes_list.append([x1_pred, y1_pred, x2_pred, y2_pred])  # x1, y1, x2, y2 format

        # Calculate a "confidence" score - use the maximum value in the prediction as proxy
        # This is just to match YOLO's format which includes confidence scores
        confidence = np.max(pred_bbox)
        confidences_list.append(confidence)

    # Find best and worst predictions
    iou_indices = np.argsort(iou_values)
    worst_indices = iou_indices[:num_samples//2]
    best_indices = iou_indices[-num_samples//2:]

    # Visualization of best and worst cases with same format as YOLO
    plt.figure(figsize=(15, 4*num_samples))
    samples_to_show = np.concatenate([worst_indices, best_indices])

    for i, idx in enumerate(samples_to_show):
        img = X_val[idx]
        true_bbox = y_val[idx]
        pred_bbox = y_pred[idx]

        # Display image with both bounding boxes
        img_display = (img * 255).astype(np.uint8).copy()
        h, w = img.shape[:2]

        # True bbox (green) - ground truth
        x, y = int(true_bbox[0] * w), int(true_bbox[1] * h)
        bbox_w, bbox_h = int(true_bbox[2] * w), int(true_bbox[3] * h)
        cv2.rectangle(img_display, (x, y), (x + bbox_w, y + bbox_h), (0, 255, 0), 2)

        # Pred bbox (red) - prediction
        x, y = int(pred_bbox[0] * w), int(pred_bbox[1] * h)
        bbox_w, bbox_h = int(pred_bbox[2] * w), int(pred_bbox[3] * h)
        cv2.rectangle(img_display, (x, y), (x + bbox_w, y + bbox_h), (255, 0, 0), 2)

        # Add confidence text (like YOLO does)
        confidence = confidences_list[idx]
        cv2.putText(img_display, f"{confidence:.2f}", (x, y-10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        plt.subplot(num_samples, 2, i+1)
        plt.imshow(img_display)
        plt.title(f"IoU: {iou_values[idx]:.4f} {'(Worst)' if idx in worst_indices else '(Best)'}", fontsize=10)
        plt.axis('off')

    plt.tight_layout()
    plt.show()

    # Use the same thresholds as in YOLO notebook
    small_threshold = 0.03
    large_threshold = 0.1

    # Categorize plates by size
    size_categories = []
    for area in plate_sizes:
        if area < small_threshold:
            size_categories.append(0)  # Small
        elif area > large_threshold:
            size_categories.append(2)  # Large
        else:
            size_categories.append(1)  # Medium

    # Group by plate size
    small_ious = [iou for iou, cat in zip(iou_values, size_categories) if cat == 0]
    medium_ious = [iou for iou, cat in zip(iou_values, size_categories) if cat == 1]
    large_ious = [iou for iou, cat in zip(iou_values, size_categories) if cat == 2]

    # Print statistics in exactly the same format as YOLO
    print("Overall Performance:")
    print(f"Average IoU: {np.mean(iou_values):.4f}")
    print(f"Median IoU: {np.median(iou_values):.4f}")
    print(f"Min IoU: {np.min(iou_values):.4f}")
    print(f"Max IoU: {np.max(iou_values):.4f}")
    print("\nPerformance by Plate Size:")
    print(f"Small Plates: Avg IoU = {np.mean(small_ious) if small_ious else 0:.4f}, Count = {len(small_ious)}")
    print(f"Medium Plates: Avg IoU = {np.mean(medium_ious) if medium_ious else 0:.4f}, Count = {len(medium_ious)}")
    print(f"Large Plates: Avg IoU = {np.mean(large_ious) if large_ious else 0:.4f}, Count = {len(large_ious)}")

    # Plot IoU distribution (identical format to YOLO notebook)
    plt.figure(figsize=(15, 6))

    # Histogram of IoU values
    plt.subplot(1, 2, 1)
    plt.hist(iou_values, bins=20, alpha=0.7, color='blue')
    plt.axvline(np.mean(iou_values), color='red', linestyle='dashed', linewidth=2, label=f'Mean IoU: {np.mean(iou_values):.4f}')
    plt.axvline(np.median(iou_values), color='green', linestyle='dashed', linewidth=2, label=f'Median IoU: {np.median(iou_values):.4f}')
    plt.title('IoU Distribution')
    plt.xlabel('IoU Value')
    plt.ylabel('Count')
    plt.legend()

    # IoU by plate size - boxplot
    plt.subplot(1, 2, 2)
    boxplot_data = [small_ious, medium_ious, large_ious]
    plt.boxplot(boxplot_data, labels=['Small', 'Medium', 'Large'])
    plt.title('IoU by License Plate Size')
    plt.ylabel('IoU Value')
    plt.xlabel('Plate Size')
    plt.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.show()

    # Calculate mAP-like metrics (similar to YOLO)
    # Using IoU thresholds of 0.5 and 0.5:0.95 (standard COCO metrics)
    map50 = np.mean([1.0 if iou >= 0.5 else 0.0 for iou in iou_values])
    map_range = np.mean([np.mean([1.0 if iou >= thresh else 0.0 for iou in iou_values])
                        for thresh in np.arange(0.5, 1.0, 0.05)])

    print("\nTraditional Object Detection Metrics (like YOLO):")
    print(f"mAP@0.5: {map50:.4f}")
    print(f"mAP@0.5:0.95: {map_range:.4f}")

    # If df_val is provided, we can do more advanced analysis
    if df_val is not None and len(df_val) == len(X_val):
        # Calculate precision/recall curves for different IoU thresholds
        thresholds = [0.5, 0.75, 0.9]  # Standard thresholds

        print("\nPrecision at different IoU thresholds:")
        for threshold in thresholds:
            precision = np.mean([1.0 if iou >= threshold else 0.0 for iou in iou_values])
            print(f"Precision@{threshold}: {precision:.4f}")

    return iou_values


def detect_license_plate(model, image_path, image_size=(224, 224)):
    """
    Detect license plate in a new image and visualize the detection
    
    Args:
        model: Trained license plate detector model
        image_path: Path to the image
        image_size: Size used for model input
        
    Returns:
        tuple: Detected plate region and bounding box
    """
    # Load and preprocess image
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    orig_h, orig_w = img_rgb.shape[:2]

    # Resize and normalize for model input
    img_resized = cv2.resize(img_rgb, image_size)
    img_normalized = img_resized / 255.0

    # Make prediction
    prediction = model.predict(np.expand_dims(img_normalized, axis=0))[0]

    # Convert normalized coordinates back to original image size
    x_pred, y_pred, w_pred, h_pred = prediction
    x = int(x_pred * orig_w)
    y = int(y_pred * orig_h)
    width = int(w_pred * orig_w)
    height = int(h_pred * orig_h)

    # Draw detection on image
    result_img = img_rgb.copy()
    cv2.rectangle(result_img, (x, y), (x + width, y + height), (0, 255, 0), 2)

    # Display result
    plt.figure(figsize=(10, 8))
    plt.imshow(result_img)
    plt.axis('off')
    plt.title("License Plate Detection")
    plt.show()

    # Extract the detected license plate region
    plate_region = img_rgb[y:y + height, x:x + width]

    # Show the extracted plate
    if plate_region.size > 0:
        plt.figure(figsize=(6, 2))
        plt.imshow(plate_region)
        plt.axis('off')
        plt.title("Extracted License Plate")
        plt.show()

    return plate_region, [x, y, width, height]


def detect_plate_from_dataset(model, dataset_index, df, image_size=(224, 224)):
    """
    Detects a license plate in an image from the dataset and displays metrics
    matching the YOLO detection format for easy comparison between models.

    Args:
        model: Trained model
        dataset_index: Index of the image in the dataset DataFrame
        df: DataFrame containing dataset information
        image_size: Size used for model input

    Returns:
        tuple: Detected plate region, IoU with ground truth, and confidence
    """
    try:
        # Get image path and ground truth
        row = df.iloc[dataset_index]
        img_path = row["image_path"]

        # Ground truth box
        gt_x, gt_y = row["x"], row["y"]
        gt_w, gt_h = row["w"], row["h"]

        # Load image
        img = cv2.imread(img_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        orig_h, orig_w = img_rgb.shape[:2]

        # Preprocess for model
        img_resized = cv2.resize(img_rgb, image_size)
        img_normalized = img_resized / 255.0

        # Make prediction
        prediction = model.predict(np.expand_dims(img_normalized, axis=0))[0]

        # Get confidence score (max value of prediction - for consistency with YOLO)
        confidence = np.max(prediction)

        # Convert normalized coordinates back to original image size
        x_pred, y_pred, w_pred, h_pred = prediction
        x = int(x_pred * orig_w)
        y = int(y_pred * orig_h)
        width = int(w_pred * orig_w)
        height = int(h_pred * orig_h)

        # Draw on image
        result_img = img_rgb.copy()

        # Draw ground truth (green)
        cv2.rectangle(result_img, (gt_x, gt_y), (gt_x + gt_w, gt_y + gt_h), (0, 255, 0), 2)

        # Draw prediction (red)
        cv2.rectangle(result_img, (x, y), (x + width, y + height), (255, 0, 0), 2)

        # Add confidence text (like YOLO format)
        cv2.putText(result_img, f"{confidence:.2f}", (x, y-10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        # Calculate IoU (same calculation as in YOLO notebook)
        # Convert to x1, y1, x2, y2 format
        x1_true, y1_true = gt_x, gt_y
        x2_true, y2_true = x1_true + gt_w, y1_true + gt_h

        x1_pred, y1_pred = x, y
        x2_pred, y2_pred = x1_pred + width, y1_pred + height

        # Calculate intersection coordinates
        x1_inter = max(x1_true, x1_pred)
        y1_inter = max(y1_true, y1_pred)
        x2_inter = min(x2_true, x2_pred)
        y2_inter = min(y2_true, y2_pred)

        # Calculate intersection area
        w_inter = max(0, x2_inter - x1_inter)
        h_inter = max(0, y2_inter - y1_inter)
        area_intersect = w_inter * h_inter

        # Calculate union area
        area_true = gt_w * gt_h
        area_pred = width * height
        area_union = area_true + area_pred - area_intersect

        # Calculate IoU
        iou = area_intersect / area_union if area_union > 0 else 0

        # Calculate normalized plate area (for size categorization)
        norm_area = (gt_w * gt_h) / (orig_w * orig_h)

        # Determine plate size category (small, medium, large)
        small_threshold = 0.03
        large_threshold = 0.1
        if norm_area < small_threshold:
            size_category = "Small"
        elif norm_area > large_threshold:
            size_category = "Large"
        else:
            size_category = "Medium"

        # Display result
        plt.figure(figsize=(12, 8))
        plt.imshow(result_img)
        plt.axis('off')
        title = f"License Plate Detection - IoU: {iou:.4f}, Conf: {confidence:.2f}, Size: {size_category}"
        if row["plate_text"] and row["plate_text"] != "Unknown":
            title += f" - Plate: {row['plate_text']}"
        plt.title(title)
        plt.show()

        # Extract and show the detected plate
        if width > 0 and height > 0:
            plate_region = img_rgb[y:y + height, x:x + width]

            plt.figure(figsize=(6, 2))
            plt.imshow(plate_region)
            plt.axis('off')
            plt.title(f"Extracted License Plate (Conf: {confidence:.2f})")
            plt.show()

            return plate_region, iou, confidence
        else:
            print("Invalid detection dimensions")
            return None, iou, confidence

    except Exception as e:
        print(f"Error detecting plate: {e}")
        return None, 0.0, 0.0


def evaluate_model_comprehensive(model, X_val, y_val, df_val=None, threshold=0.5, num_visualize=5):
    """
    Comprehensive model evaluation with detailed metrics and visualizations
    
    Args:
        model: Trained license plate detector model
        X_val: Validation images
        y_val: Ground truth bounding boxes
        df_val: Optional DataFrame with additional metadata
        threshold: IoU threshold for considering a detection as correct
        num_visualize: Number of examples to visualize in each category
        
    Returns:
        dict: Dictionary with evaluation metrics
    """
    print("Starting comprehensive model evaluation...")
    
    # Get predictions
    y_pred = model.predict(X_val, verbose=1)
    
    # Calculate IoU for all predictions
    iou_values = []
    giou_values = []
    coord_errors = {
        'x_error': [], 'y_error': [], 'w_error': [], 'h_error': [], 
        'center_error': [], 'size_error': []
    }
    
    # Plate size information
    plate_sizes = []
    plate_size_categories = []
    small_threshold = 0.03
    large_threshold = 0.1
    
    for i in range(len(y_val)):
        # Get true and predicted boxes
        true_box = y_val[i]
        pred_box = y_pred[i]
        
        # Store plate size
        plate_area = true_box[2] * true_box[3]  # w * h
        plate_sizes.append(plate_area)
        
        # Categorize plate size
        if plate_area < small_threshold:
            plate_size_categories.append("Small")
        elif plate_area > large_threshold:
            plate_size_categories.append("Large") 
        else:
            plate_size_categories.append("Medium")
            
        # Calculate errors in each coordinate
        x_error = abs(true_box[0] - pred_box[0])
        y_error = abs(true_box[1] - pred_box[1])
        w_error = abs(true_box[2] - pred_box[2])
        h_error = abs(true_box[3] - pred_box[3])
        
        # Calculate center point error
        true_center_x = true_box[0] + true_box[2]/2
        true_center_y = true_box[1] + true_box[3]/2
        pred_center_x = pred_box[0] + pred_box[2]/2
        pred_center_y = pred_box[1] + pred_box[3]/2
        center_error = np.sqrt((true_center_x - pred_center_x)**2 + 
                              (true_center_y - pred_center_y)**2)
        
        # Calculate size error (area difference)
        true_size = true_box[2] * true_box[3]
        pred_size = pred_box[2] * pred_box[3]
        size_error = abs(true_size - pred_size) / true_size  # Relative size error
        
        # Store all errors
        coord_errors['x_error'].append(x_error)
        coord_errors['y_error'].append(y_error)
        coord_errors['w_error'].append(w_error)
        coord_errors['h_error'].append(h_error)
        coord_errors['center_error'].append(center_error)
        coord_errors['size_error'].append(size_error)
        
        # Calculate IoU
        # Convert to x1, y1, x2, y2 format
        x1_true, y1_true = true_box[0], true_box[1]
        x2_true, y2_true = x1_true + true_box[2], y1_true + true_box[3]

        x1_pred, y1_pred = pred_box[0], pred_box[1]
        x2_pred, y2_pred = x1_pred + pred_box[2], y1_pred + pred_box[3]
        
        # Calculate intersection coordinates
        x1_inter = max(x1_true, x1_pred)
        y1_inter = max(y1_true, y1_pred)
        x2_inter = min(x2_true, x2_pred)
        y2_inter = min(y2_true, y2_pred)
        
        # Calculate intersection area
        w_inter = max(0, x2_inter - x1_inter)
        h_inter = max(0, y2_inter - y1_inter)
        area_intersect = w_inter * h_inter
        
        area_true = true_box[2] * true_box[3]
        area_pred = pred_box[2] * pred_box[3]
        area_union = area_true + area_pred - area_intersect
        
        # IoU
        iou = area_intersect / area_union if area_union > 0 else 0
        iou_values.append(iou)
        
        # Calculate GIoU
        # Find enclosing box
        x1_enclosing = min(x1_true, x1_pred)
        y1_enclosing = min(y1_true, y1_pred)
        x2_enclosing = max(x2_true, x2_pred)
        y2_enclosing = max(y2_true, y2_pred)
        
        # Calculate area of enclosing box
        area_enclosing = (x2_enclosing - x1_enclosing) * (y2_enclosing - y1_enclosing)
        
        # GIoU
        giou = iou - ((area_enclosing - area_union) / area_enclosing if area_enclosing > 0 else 0)
        giou_values.append(giou)
    
    # Calculate metrics
    metrics = {}
    metrics['mean_iou'] = np.mean(iou_values)
    metrics['median_iou'] = np.median(iou_values)
    metrics['min_iou'] = np.min(iou_values)
    metrics['max_iou'] = np.max(iou_values)
    metrics['std_iou'] = np.std(iou_values)
    
    metrics['mean_giou'] = np.mean(giou_values)
    
    # Calculate average precision (AP) at different IoU thresholds
    iou_thresholds = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
    ap_values = []
    
    for iou_thresh in iou_thresholds:
        # For our regression model, we count a detection as correct if IoU >= threshold
        correct_detections = sum(1 for iou in iou_values if iou >= iou_thresh)
        ap = correct_detections / len(iou_values)
        ap_values.append(ap)
    
    metrics['map50'] = ap_values[0]  # mAP@0.5
    metrics['map'] = np.mean(ap_values)  # mAP@0.5:0.95
    
    # Calculate metrics by plate size
    size_metrics = {}
    for size in ["Small", "Medium", "Large"]:
        # Get indices for this size category
        size_indices = [i for i, s in enumerate(plate_size_categories) if s == size]
        
        if size_indices:
            size_ious = [iou_values[i] for i in size_indices]
            size_metrics[f'{size.lower()}_count'] = len(size_indices)
            size_metrics[f'{size.lower()}_mean_iou'] = np.mean(size_ious)
            size_metrics[f'{size.lower()}_median_iou'] = np.median(size_ious)
            
            # Calculate mAP@0.5 for this size
            correct = sum(1 for iou in size_ious if iou >= 0.5)
            size_metrics[f'{size.lower()}_map50'] = correct / len(size_ious)
    
    # Combine all metrics
    metrics.update(size_metrics)
    
    # Calculate average coordinate errors
    for key, values in coord_errors.items():
        metrics[f'mean_{key}'] = np.mean(values)
        metrics[f'median_{key}'] = np.median(values)
    
    # Print summary of metrics
    print("\n===== EVALUATION RESULTS =====")
    print(f"Overall Performance:")
    print(f"  Mean IoU: {metrics['mean_iou']:.4f}")
    print(f"  Median IoU: {metrics['median_iou']:.4f}")
    print(f"  mAP@0.5: {metrics['map50']:.4f}")
    print(f"  mAP@0.5:0.95: {metrics['map']:.4f}")
    
    print("\nPerformance by Plate Size:")
    for size in ["Small", "Medium", "Large"]:
        size_key = size.lower()
        if f'{size_key}_count' in metrics:
            count = metrics[f'{size_key}_count']
            mean_iou = metrics[f'{size_key}_mean_iou']
            map50 = metrics[f'{size_key}_map50']
            print(f"  {size} Plates: Count={count}, Mean IoU={mean_iou:.4f}, mAP@0.5={map50:.4f}")
    
    print("\nCoordinate Errors (Normalized):")
    print(f"  Center Point Error: {metrics['mean_center_error']:.4f}")
    print(f"  Size Error: {metrics['mean_size_error']:.4f}")
    print(f"  X Error: {metrics['mean_x_error']:.4f}, Y Error: {metrics['mean_y_error']:.4f}")
    print(f"  Width Error: {metrics['mean_w_error']:.4f}, Height Error: {metrics['mean_h_error']:.4f}")
    
    # Create visualizations
    
    # 1. IoU Distribution
    plt.figure(figsize=(18, 6))
    
    plt.subplot(1, 3, 1)
    plt.hist(iou_values, bins=20, alpha=0.7)
    plt.axvline(metrics['mean_iou'], color='r', linestyle='--', label=f"Mean: {metrics['mean_iou']:.3f}")
    plt.axvline(metrics['median_iou'], color='g', linestyle='--', label=f"Median: {metrics['median_iou']:.3f}")
    plt.title('IoU Distribution')
    plt.xlabel('IoU')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    
    # 2. IoU by Plate Size
    sizes = ["Small", "Medium", "Large"]
    size_data = [
        [iou for iou, size in zip(iou_values, plate_size_categories) if size == s]
        for s in sizes
    ]
    
    plt.subplot(1, 3, 2)
    plt.boxplot(size_data, labels=sizes)
    plt.title('IoU by Plate Size')
    plt.ylabel('IoU')
    plt.grid(True, linestyle='--', alpha=0.5)
    
    # 3. Coordinate Errors
    plt.subplot(1, 3, 3)
    error_labels = ['x', 'y', 'width', 'height', 'center', 'size']
    error_values = [
        metrics['mean_x_error'], 
        metrics['mean_y_error'],
        metrics['mean_w_error'],
        metrics['mean_h_error'],
        metrics['mean_center_error'],
        metrics['mean_size_error']
    ]
    plt.bar(error_labels, error_values)
    plt.title('Average Coordinate Errors')
    plt.ylabel('Normalized Error')
    plt.grid(True, linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.show()
    
    # Visualize predictions
    # Sort by IoU for visualization
    sorted_indices = np.argsort(iou_values)
    
    # Get indices for worst, median, and best predictions
    worst_indices = sorted_indices[:num_visualize]
    median_start = len(sorted_indices)//2 - num_visualize//2
    median_indices = sorted_indices[median_start:median_start+num_visualize]
    best_indices = sorted_indices[-num_visualize:]
    
    # Function to plot prediction examples
    def plot_predictions(indices, title):
        plt.figure(figsize=(15, 3*min(len(indices), 5)))
        
        for i, idx in enumerate(indices[:5]):  # Limit to 5 examples
            img = X_val[idx]
            true_box = y_val[idx]
            pred_box = y_pred[idx]
            
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
            
            # Plot
            plt.subplot(min(len(indices), 5), 1, i+1)
            plt.imshow(img_display)
            plt.title(f"IoU: {iou_values[idx]:.4f}, Size: {plate_size_categories[idx]}")
            plt.axis('off')
        
        plt.suptitle(title, fontsize=16)
        plt.tight_layout()
        plt.subplots_adjust(top=0.9)
        plt.show()
    
    # Plot worst, median, and best predictions
    plot_predictions(worst_indices, "Worst Predictions")
    plot_predictions(median_indices, "Median Predictions")
    plot_predictions(best_indices, "Best Predictions")
    
    return metrics
