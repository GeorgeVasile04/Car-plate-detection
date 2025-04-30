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
    """
    print(f"Starting data augmentation with factor {augmentation_factor}...")
    print(f"Original dataset: {len(images)} images")
    
    augmented_images = []
    augmented_bboxes = []
    
    # Keep original images
    augmented_images.extend(images)
    augmented_bboxes.extend(bboxes)
    
    # Create sequential augmentations
    for i in range(len(images)):
        if i % 100 == 0:
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
            
            # 6. Random small rotation (max 15 degrees)
            if np.random.random() > 0.7:
                angle = np.random.uniform(-15, 15)
                h, w = img_aug.shape[:2]
                M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1)
                img_aug = cv2.warpAffine((img_aug * 255).astype(np.uint8), M, (w, h))
                img_aug = img_aug.astype(np.float32) / 255.0
                
                # Handle bbox rotation (simplified - works well for small angles)
                # For small angles, we can keep the bounding box center and adjust slightly
                center_x = bbox_aug[0] + bbox_aug[2]/2
                center_y = bbox_aug[1] + bbox_aug[3]/2
                
                # Small increase in size to account for rotation
                size_factor = 1 + abs(angle) / 90
                new_width = min(bbox_aug[2] * size_factor, 1.0)
                new_height = min(bbox_aug[3] * size_factor, 1.0)
                
                bbox_aug[0] = max(0, min(center_x - new_width/2, 1 - new_width))
                bbox_aug[1] = max(0, min(center_y - new_height/2, 1 - new_height))
                bbox_aug[2] = new_width
                bbox_aug[3] = new_height
            
            # Add augmented sample
            augmented_images.append(img_aug)
            augmented_bboxes.append(bbox_aug)
    
    augmented_images = np.array(augmented_images)
    augmented_bboxes = np.array(augmented_bboxes)
    
    print(f"Augmentation complete. New dataset size: {len(augmented_images)} images")
    return augmented_images, augmented_bboxes


def visualize_augmentation(original_images, original_bboxes, augmented_images, augmented_bboxes, num_samples=4):
    """
    Visualize a few original and augmented samples side by side for comparison
    
    Args:
        original_images: Original images
        original_bboxes: Original bounding boxes
        augmented_images: Augmented images
        augmented_bboxes: Augmented bounding boxes
        num_samples: Number of samples to visualize
    """
    import matplotlib.pyplot as plt
    
    # Randomly select a few original samples
    indices = np.random.choice(len(original_images), num_samples, replace=False)
    
    plt.figure(figsize=(15, 4 * num_samples))
    
    for i, idx in enumerate(indices):
        # Original image
        plt.subplot(num_samples, 2, 2*i + 1)
        img = original_images[idx]
        bbox = original_bboxes[idx]
        
        # Draw bounding box
        img_with_bbox = img.copy()
        h, w = img.shape[:2]
        x, y = int(bbox[0] * w), int(bbox[1] * h)
        box_w, box_h = int(bbox[2] * w), int(bbox[3] * h)
        cv2.rectangle(img_with_bbox, (x, y), (x + box_w, y + box_h), (0, 1, 0), 2)
        
        plt.imshow(img_with_bbox)
        plt.title(f"Original Image {idx}")
        plt.axis('off')
        
        # Find a corresponding augmented sample (original + num_samples + idx)
        aug_idx = len(original_images) + idx
        if aug_idx < len(augmented_images):
            plt.subplot(num_samples, 2, 2*i + 2)
            aug_img = augmented_images[aug_idx]
            aug_bbox = augmented_bboxes[aug_idx]
            
            # Draw bounding box
            aug_img_with_bbox = aug_img.copy()
            h, w = aug_img.shape[:2]
            x, y = int(aug_bbox[0] * w), int(aug_bbox[1] * h)
            box_w, box_h = int(aug_bbox[2] * w), int(aug_bbox[3] * h)
            cv2.rectangle(aug_img_with_bbox, (x, y), (x + box_w, y + box_h), (0, 1, 0), 2)
            
            plt.imshow(aug_img_with_bbox)
            plt.title(f"Augmented Version")
            plt.axis('off')
    
    plt.tight_layout()
    plt.show()