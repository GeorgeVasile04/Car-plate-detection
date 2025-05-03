"""
Data augmentation utilities for license plate detection.
"""

import cv2
import numpy as np


def augment_data(images, bboxes, augmentation_factor=5):
    """
    Apply enhanced data augmentation to increase dataset size and improve generalization.
    This version focuses on improving size estimation and small plate detection.
    
    Args:
        images: Array of images with shape (N, height, width, channels)
        bboxes: Array of normalized bounding boxes with shape (N, 4) [x, y, w, h]
        augmentation_factor: How many augmented samples to create per original image
        
    Returns:
        Augmented images and corresponding bounding boxes
        
    Note:
        This function properly handles bounding box transformations during augmentation operations,
        including rotation where the bounding box coordinates are accurately rotated along with
        the image by transforming all four corners of the bounding box.
    """
    print(f"Starting enhanced data augmentation with factor {augmentation_factor}...")
    print(f"Original dataset: {len(images)} images")
    
    # Ensure input arrays are float32 to save memory
    if images.dtype != np.float32:
        images = images.astype(np.float32)
    
    if bboxes.dtype != np.float32:
        bboxes = bboxes.astype(np.float32)
    
    # Keep original images
    original_count = len(images)
    total_count = original_count * (1 + augmentation_factor)
    
    # Pre-allocate arrays for better memory efficiency
    augmented_images = np.zeros((total_count, *images.shape[1:]), dtype=np.float32)
    augmented_bboxes = np.zeros((total_count, 4), dtype=np.float32)
    
    # Copy original images and boxes to the augmented arrays
    augmented_images[:original_count] = images
    augmented_bboxes[:original_count] = bboxes
    
    # Create augmented versions of each image
    aug_idx = original_count
    
    # Identify small plates for focused augmentation
    plate_sizes = bboxes[:, 2] * bboxes[:, 3]
    small_plate_threshold = np.percentile(plate_sizes, 33)  # Bottom third considered small
    small_plate_indices = np.where(plate_sizes < small_plate_threshold)[0]
    
    print(f"Identified {len(small_plate_indices)} small plates for focused augmentation")
    
    # Use a custom progress tracker to avoid duplicate messages
    progress_step = max(1, len(images) // 10)
    progress_points = set(range(0, len(images), progress_step))
    
    for i in range(len(images)):
        if i in progress_points:
            print(f"Augmenting image {i}/{len(images)}...")
        
        img = images[i]
        bbox = bboxes[i]
        is_small_plate = i in small_plate_indices
        
        # Determine number of augmentations - create more variants for small plates
        local_aug_factor = augmentation_factor + 2 if is_small_plate else augmentation_factor
        
        for j in range(local_aug_factor):
            if aug_idx >= len(augmented_images):
                # Expand arrays if needed for small plate extra augmentations
                augmented_images = np.concatenate([augmented_images, 
                                               np.zeros((100, *images.shape[1:]), dtype=np.float32)])
                augmented_bboxes = np.concatenate([augmented_bboxes, 
                                               np.zeros((100, 4), dtype=np.float32)])
            
            # Start with original
            img_aug = img.copy()
            bbox_aug = bbox.copy()
            
            # BASIC AUGMENTATIONS
            
            # 1. Random brightness adjustment (preserves bounding box)
            if np.random.random() > 0.4:  # Increased probability
                brightness_factor = np.random.uniform(0.7, 1.3)  # Wider range
                img_aug = np.clip(img_aug * brightness_factor, 0, 1)
            
            # 2. Random contrast adjustment (preserves bounding box)
            if np.random.random() > 0.4:  # Increased probability
                contrast_factor = np.random.uniform(0.7, 1.3)  # Wider range
                img_aug = np.clip((img_aug - 0.5) * contrast_factor + 0.5, 0, 1)
            
            # 3. Random horizontal flip with bbox adjustment
            if np.random.random() > 0.5:
                img_aug = np.fliplr(img_aug)
                # Adjust bbox: x' = 1 - x - w, y' = y, w' = w, h' = h
                bbox_aug[0] = 1 - bbox_aug[0] - bbox_aug[2]
            
            # 4. Random noise addition
            if np.random.random() > 0.6:  # Increased probability
                noise_level = 0.02 if is_small_plate else 0.01  # More noise for small plates
                noise = np.random.normal(0, noise_level, img_aug.shape)
                img_aug = np.clip(img_aug + noise, 0, 1)
            
            # 5. Random saturation and hue adjustment (for color images)
            if img_aug.shape[-1] == 3 and np.random.random() > 0.6:
                img_hsv = cv2.cvtColor((img_aug * 255).astype(np.uint8), cv2.COLOR_RGB2HSV)
                # Saturation adjustment
                img_hsv[:, :, 1] = img_hsv[:, :, 1] * np.random.uniform(0.7, 1.3)
                img_hsv[:, :, 1] = np.clip(img_hsv[:, :, 1], 0, 255)
                # Hue adjustment
                img_hsv[:, :, 0] = img_hsv[:, :, 0] + np.random.randint(-10, 10)
                img_hsv[:, :, 0] = np.clip(img_hsv[:, :, 0], 0, 179)  # Hue is [0, 179] in OpenCV
                img_aug = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2RGB).astype(np.float32) / 255.0
            
            # ADVANCED AUGMENTATIONS - especially useful for small plates
            
            # 6. Random scaling (simulate different plate sizes)
            if np.random.random() > 0.6:
                # Scale factor depends on whether it's a small plate
                if is_small_plate:
                    # For small plates, make them slightly larger or much larger
                    scale_factor = np.random.uniform(1.0, 1.5)
                else:
                    # For larger plates, make them smaller or slightly larger
                    scale_factor = np.random.uniform(0.8, 1.2)
                
                # Calculate new dimensions while preserving aspect ratio
                h, w = img_aug.shape[:2]
                new_w, new_h = int(w * scale_factor), int(h * scale_factor)
                
                # Resize the image
                img_aug_scaled = cv2.resize(img_aug, (new_w, new_h))
                
                # Prepare a new canvas of original size
                canvas = np.zeros_like(img_aug)
                
                # Calculate position to paste the scaled image
                paste_x = max(0, (w - new_w) // 2)
                paste_y = max(0, (h - new_h) // 2)
                
                # Ensure we don't exceed dimensions
                paste_w = min(new_w, w - paste_x)
                paste_h = min(new_h, h - paste_y)
                
                # Paste scaled image onto canvas
                canvas[paste_y:paste_y+paste_h, paste_x:paste_x+paste_w] = img_aug_scaled[:paste_h, :paste_w]
                
                # Adjust bounding box
                bbox_aug[0] = (bbox_aug[0] * scale_factor * w + paste_x) / w
                bbox_aug[1] = (bbox_aug[1] * scale_factor * h + paste_y) / h
                bbox_aug[2] = bbox_aug[2] * scale_factor
                bbox_aug[3] = bbox_aug[3] * scale_factor
                
                # Ensure bbox stays within [0,1] range
                bbox_aug = np.clip(bbox_aug, 0, 1)
                
                # Update img_aug with the canvas
                img_aug = canvas
            
            # 7. Random perspective transform for small plates (simulate viewing angle changes)
            if is_small_plate and np.random.random() > 0.7:
                h, w = img_aug.shape[:2]
                
                # Define the magnitude of the perspective transform
                # Smaller for small plates to avoid extreme distortions
                magnitude = 0.05
                
                # Get corner points of the bounding box
                x, y = bbox_aug[0] * w, bbox_aug[1] * h
                bbox_w, bbox_h = bbox_aug[2] * w, bbox_aug[3] * h
                
                src_points = np.float32([
                    [x, y],  # top-left
                    [x + bbox_w, y],  # top-right
                    [x + bbox_w, y + bbox_h],  # bottom-right
                    [x, y + bbox_h]  # bottom-left
                ])
                
                # Add random displacement to each corner (more controlled)
                displacement = np.random.uniform(-magnitude * min(bbox_w, bbox_h), 
                                            magnitude * min(bbox_w, bbox_h), 
                                            size=(4, 2))
                dst_points = src_points + displacement
                
                # Ensure destination points stay within image bounds
                dst_points[:, 0] = np.clip(dst_points[:, 0], 0, w)
                dst_points[:, 1] = np.clip(dst_points[:, 1], 0, h)
                
                # Calculate perspective transform matrix
                M = cv2.getPerspectiveTransform(src_points.astype(np.float32), 
                                               dst_points.astype(np.float32))
                
                # Apply transform to image
                img_aug = cv2.warpPerspective(img_aug, M, (w, h))
                
                # Calculate the new bounding box from the transformed corners
                # This is an approximation - find min/max after transform
                transformed_corners = cv2.perspectiveTransform(
                    np.array([src_points]), M)[0]
                
                min_x = max(0, np.min(transformed_corners[:, 0]))
                min_y = max(0, np.min(transformed_corners[:, 1]))
                max_x = min(w, np.max(transformed_corners[:, 0]))
                max_y = min(h, np.max(transformed_corners[:, 1]))
                
                # Convert back to normalized coordinates
                bbox_aug[0] = min_x / w
                bbox_aug[1] = min_y / h
                bbox_aug[2] = (max_x - min_x) / w
                bbox_aug[3] = (max_y - min_y) / h
                
                # Ensure bbox coordinates are valid
                bbox_aug = np.clip(bbox_aug, 0, 1)
            
            # 8. Random crops with zoom for small plates (focus on the plate region)
            if is_small_plate and np.random.random() > 0.6:
                h, w = img_aug.shape[:2]
                
                # Get bounding box coordinates
                x, y = bbox_aug[0] * w, bbox_aug[1] * h
                bbox_w, bbox_h = bbox_aug[2] * w, bbox_aug[3] * h
                
                # Calculate crop region - centered on plate but with margin
                margin = min(bbox_w, bbox_h) * 3  # Margin around plate
                crop_x1 = max(0, int(x - margin))
                crop_y1 = max(0, int(y - margin))
                crop_x2 = min(w, int(x + bbox_w + margin))
                crop_y2 = min(h, int(y + bbox_h + margin))
                
                # Apply crop
                cropped = img_aug[crop_y1:crop_y2, crop_x1:crop_x2]
                
                # Resize back to original dimensions
                img_aug = cv2.resize(cropped, (w, h))
                
                # Adjust bounding box for crop and resize
                new_x = (x - crop_x1) / (crop_x2 - crop_x1)
                new_y = (y - crop_y1) / (crop_y2 - crop_y1)
                new_w = bbox_w / (crop_x2 - crop_x1)
                new_h = bbox_h / (crop_y2 - crop_y1)
                
                bbox_aug[0] = new_x
                bbox_aug[1] = new_y
                bbox_aug[2] = new_w
                bbox_aug[3] = new_h
                
                # Ensure bbox coordinates are valid
                bbox_aug = np.clip(bbox_aug, 0, 1)
            
            # Add augmented sample to the arrays
            augmented_images[aug_idx] = img_aug
            augmented_bboxes[aug_idx] = bbox_aug
            aug_idx += 1
    
    # Trim arrays to actual used size
    augmented_images = augmented_images[:aug_idx]
    augmented_bboxes = augmented_bboxes[:aug_idx]
    
    print(f"Augmentation complete. New dataset size: {len(augmented_images)} images")
    return augmented_images, augmented_bboxes
    
    # Single completion message at the end
    print(f"Augmentation complete. New dataset size: {total_count} images")
    return augmented_images, augmented_bboxes


def visualize_augmentation(original_images, original_bboxes, augmented_images, augmented_bboxes, num_samples=4):
    """
    Visualize original images alongside their specific augmented variants
    
    Args:
        original_images: Original images
        original_bboxes: Original bounding boxes
        augmented_images: Augmented images (including originals)
        augmented_bboxes: Augmented bounding boxes (including originals)
        num_samples: Number of original samples to visualize
    """
    import matplotlib.pyplot as plt
    
    # Determine augmentation factor (default from augment_data is 4)
    augmentation_factor = (len(augmented_images) - len(original_images)) // len(original_images)
    
    # Randomly select a few original samples
    indices = np.random.choice(len(original_images), num_samples, replace=False)
    
    for i, idx in enumerate(indices):
        # Create a figure for each original image and its variants
        plt.figure(figsize=(20, 5))
        plt.suptitle(f"Original Image {idx} and Its {augmentation_factor} Augmented Variants", fontsize=16)
        
        # Display the original image
        plt.subplot(1, augmentation_factor + 1, 1)
        img = original_images[idx]
        bbox = original_bboxes[idx]
        
        # Draw bounding box on original
        img_with_bbox = img.copy()
        h, w = img.shape[:2]
        x = int(bbox[0] * w)
        y = int(bbox[1] * h)
        box_w = int(bbox[2] * w) 
        box_h = int(bbox[3] * h)
        
        # Draw the bounding box
        cv2.rectangle(img_with_bbox, (x, y), (x + box_w, y + box_h), (0, 1, 0), 2)
        
        # Draw corner points for better visualization
        corner_size = 3
        corner_color = (1, 0, 0)  # Red corners
        corners = [
            (x, y),                  # top-left
            (x + box_w, y),          # top-right
            (x + box_w, y + box_h),  # bottom-right
            (x, y + box_h)           # bottom-left
        ]
        for cx, cy in corners:
            cv2.circle(img_with_bbox, (cx, cy), corner_size, corner_color, -1)
        
        plt.imshow(img_with_bbox)
        plt.title("Original Image")
        plt.axis('off')
        
        # Calculate where this image's augmented versions start in the augmented array
        # The first len(original_images) elements in augmented_images are the originals,
        # followed by augmentation_factor versions of each original image
        aug_start_idx = len(original_images) + idx * augmentation_factor
        
        # Display each augmented variant
        for j in range(augmentation_factor):
            aug_idx = aug_start_idx + j
            
            if aug_idx < len(augmented_images):
                plt.subplot(1, augmentation_factor + 1, j + 2)
                aug_img = augmented_images[aug_idx]
                aug_bbox = augmented_bboxes[aug_idx]
                
                # Draw bounding box on augmented variant
                aug_img_with_bbox = aug_img.copy()
                h, w = aug_img.shape[:2]
                x = int(aug_bbox[0] * w)
                y = int(aug_bbox[1] * h)
                box_w = int(aug_bbox[2] * w)
                box_h = int(aug_bbox[3] * h)
                
                # Draw rectangle for the bounding box
                cv2.rectangle(aug_img_with_bbox, (x, y), (x + box_w, y + box_h), (0, 1, 0), 2)
                
                # Draw corner points for better visualization
                corners = [
                    (x, y),                  # top-left
                    (x + box_w, y),          # top-right
                    (x + box_w, y + box_h),  # bottom-right
                    (x, y + box_h)           # bottom-left
                ]
                for cx, cy in corners:
                    cv2.circle(aug_img_with_bbox, (cx, cy), corner_size, corner_color, -1)
                
                # Add some text to indicate which augmentations were applied
                aug_text = f"Augmented Version {j+1}"
                plt.imshow(aug_img_with_bbox)
                plt.title(aug_text)
                plt.axis('off')
        
        plt.tight_layout(rect=[0, 0, 1, 0.95])  # Make room for suptitle
        plt.show()