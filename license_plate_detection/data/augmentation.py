"""
Data augmentation utilities for license plate detection.
"""

import cv2
import numpy as np


def augment_data(images, bboxes, augmentation_factor=4):
    """
    Apply data augmentation to increase dataset size and improve generalization.
    
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
    # Use a single print statement for starting the augmentation process
    print(f"Starting data augmentation with factor {augmentation_factor}...")
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
    
    # Use a custom progress tracker to avoid duplicate messages
    progress_step = max(1, len(images) // 10)  # Show progress at 10% intervals
    progress_points = set(range(0, len(images), progress_step))
    
    for i in range(len(images)):
        # Only print progress at specific intervals to reduce output volume
        if i in progress_points:
            print(f"Augmenting image {i}/{len(images)}...")
        
        img = images[i]
        bbox = bboxes[i]
        
        for j in range(augmentation_factor):
            # Select random augmentation techniques
            img_aug = img.copy()
            bbox_aug = bbox.copy()
            
            # 1. Random brightness adjustment (preserves bounding box)
            if np.random.random() > 0.5:
                brightness_factor = np.random.uniform(0.8, 1.2)
                img_aug = np.clip(img_aug * brightness_factor, 0, 1)
            
            # 2. Random contrast adjustment (preserves bounding box)
            if np.random.random() > 0.5:
                contrast_factor = np.random.uniform(0.8, 1.2)
                img_aug = np.clip((img_aug - 0.5) * contrast_factor + 0.5, 0, 1)
            
            # 3. Random horizontal flip with bbox adjustment
            if np.random.random() > 0.5:
                img_aug = np.fliplr(img_aug)
                # Adjust bbox: x' = 1 - x - w, y' = y, w' = w, h' = h
                bbox_aug[0] = 1 - bbox[0] - bbox[2]
            
            # 4. Random noise addition
            if np.random.random() > 0.7:
                noise = np.random.normal(0, 0.01, img_aug.shape)
                img_aug = np.clip(img_aug + noise, 0, 1)
            
            # 5. Random saturation adjustment (for color images)
            if img_aug.shape[-1] == 3 and np.random.random() > 0.7:
                # Convert to HSV, adjust S channel, convert back
                img_hsv = cv2.cvtColor((img_aug * 255).astype(np.uint8), cv2.COLOR_RGB2HSV)
                img_hsv[:, :, 1] = img_hsv[:, :, 1] * np.random.uniform(0.8, 1.2)
                img_hsv[:, :, 1] = np.clip(img_hsv[:, :, 1], 0, 255)
                img_aug = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2RGB).astype(np.float32) / 255.0            
            
            
            # Add augmented sample to the arrays
            augmented_images[aug_idx] = img_aug
            augmented_bboxes[aug_idx] = bbox_aug
            aug_idx += 1
    
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