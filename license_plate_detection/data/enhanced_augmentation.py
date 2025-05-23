"""
Enhanced data augmentation techniques focused on improving license plate detection,
with special emphasis on small plate cases which are the most challenging.
"""

import tensorflow as tf
import numpy as np
import albumentations as A
import cv2
import random
import warnings

# Check NumPy version and provide compatible imports
NUMPY_VERSION = tuple(map(int, np.__version__.split('.')[:2]))

# Try to import imgaug, but provide fallbacks if it fails
try:
    # For NumPy 2.0+, we need to monkey patch sctypes before importing imgaug
    if NUMPY_VERSION >= (2, 0):
        # Create a compatibility layer for np.sctypes
        if not hasattr(np, 'sctypes'):
            np.sctypes = {
                "float": [np.float16, np.float32, np.float64],
                "int": [np.int8, np.int16, np.int32, np.int64],
                "uint": [np.uint8, np.uint16, np.uint32, np.uint64],
                "complex": [np.complex64, np.complex128]
            }
        warnings.warn("Using NumPy 2.0+ compatibility layer for imgaug")
    
    from imgaug import augmenters as iaa
    IMGAUG_AVAILABLE = True
    print("Successfully imported imgaug")
except Exception as e:
    IMGAUG_AVAILABLE = False
    warnings.warn(f"Failed to import imgaug: {str(e)}. Using albumentations only.")
    # Create a dummy iaa namespace to prevent errors
    class DummyIaa:
        pass
    iaa = DummyIaa()


class SmallPlateFocusedAugmentation:
    """
    Augmentation pipeline with strong focus on improving small plate detection.
    Uses a combination of spatial transformations, photometric changes, and 
    specialized techniques for small objects.
    """
    
    def __init__(self, small_plate_threshold=0.05, extreme_aug_prob=0.3):
        """
        Initialize the augmentation pipeline.
        
        Args:
            small_plate_threshold: Threshold for defining small plates (area ratio)
            extreme_aug_prob: Probability of applying more extreme augmentations
        """
        self.small_plate_threshold = small_plate_threshold
        self.extreme_aug_prob = extreme_aug_prob
        
        # Create standard augmentation pipeline
        self.standard_transform = A.Compose([
            # Spatial augmentations
            A.HorizontalFlip(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=15, p=0.7),
            
            # Color augmentations
            A.OneOf([
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2),
                A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20),
                A.RandomGamma(),
                A.CLAHE(),
            ], p=0.7),
            
            # Weather/noise augmentations
            A.OneOf([
                A.GaussianBlur(blur_limit=3),
                A.MotionBlur(blur_limit=3),
                A.GaussNoise(var_limit=(10, 50)),
                A.ISONoise(),
            ], p=0.4),
        ], bbox_params=A.BboxParams(format='coco', label_fields=['class_labels']))
        
        # Create small plate focused augmentation pipeline
        self.small_plate_transform = A.Compose([
            # More aggressive spatial augmentations for small plates
            A.HorizontalFlip(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.15, scale_limit=0.2, rotate_limit=20, p=0.85),
            
            # Random crops to force model to learn from partial views
            A.RandomSizedBBoxSafeCrop(height=224, width=224, erosion_rate=0.2, p=0.6),
            
            # More aggressive color augmentations
            A.OneOf([
                A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3),
                A.HueSaturationValue(hue_shift_limit=30, sat_shift_limit=40, val_shift_limit=30),
                A.RandomGamma(gamma_limit=(80, 120)),
                A.CLAHE(clip_limit=4.0, p=0.7),
            ], p=0.85),
            
            # More aggressive weather/noise simulations
            A.OneOf([
                A.GaussianBlur(blur_limit=5),
                A.MotionBlur(blur_limit=7),
                A.GaussNoise(var_limit=(30, 80)),
                A.ISONoise(intensity=(0.2, 0.8)),
                A.MultiplicativeNoise(),
            ], p=0.7),
            
            # Simulate challenging lighting and environments
            A.OneOf([
                A.RandomShadow(shadow_roi=(0, 0, 1, 1), p=0.5),
                A.RandomSunFlare(flare_roi=(0, 0, 1, 1), p=0.2),
                A.RandomFog(fog_coef_lower=0.1, fog_coef_upper=0.3, p=0.2),
                A.RandomRain(drop_length=8, blur_value=3, p=0.2),
            ], p=0.4),
        ], bbox_params=A.BboxParams(format='coco', label_fields=['class_labels']))
        
        # Extreme augmentation pipeline for challenging cases
        self.extreme_transform = A.Compose([
            # Very aggressive transformations for hard negative mining
            A.HorizontalFlip(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.2, scale_limit=0.3, rotate_limit=30, p=0.9),
            A.RandomSizedBBoxSafeCrop(height=224, width=224, erosion_rate=0.3, p=0.7),
            
            # Extreme color and contrast changes
            A.OneOf([
                A.RandomBrightnessContrast(brightness_limit=0.4, contrast_limit=0.4),
                A.HueSaturationValue(hue_shift_limit=40, sat_shift_limit=60, val_shift_limit=40),
                A.RandomGamma(gamma_limit=(60, 140)),
                A.CLAHE(clip_limit=6.0),
                A.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
            ], p=0.9),
            
            # Extreme image quality degradation
            A.OneOf([
                A.GaussianBlur(blur_limit=7),
                A.MotionBlur(blur_limit=9),
                A.GaussNoise(var_limit=(50, 150)),
                A.ImageCompression(quality_lower=50, quality_upper=90),
                A.Downscale(scale_min=0.6, scale_max=0.9, interpolation=cv2.INTER_LINEAR),
            ], p=0.8),
            
            # Extreme environmental effects
            A.OneOf([
                A.RandomShadow(shadow_roi=(0, 0, 1, 1), p=0.6),
                A.RandomSunFlare(flare_roi=(0, 0, 1, 1), p=0.4),
                A.RandomFog(fog_coef_lower=0.2, fog_coef_upper=0.5, p=0.4),
                A.RandomRain(drop_length=10, blur_value=5, p=0.3),
                A.RandomSnow(snow_point_lower=0.1, snow_point_upper=0.3, p=0.2),
            ], p=0.6),        ], bbox_params=A.BboxParams(format='coco', label_fields=['class_labels']))
    def _convert_yolo_to_coco(self, bbox, img_width, img_height):
        """
        Convert normalized YOLO format [x_center, y_center, width, height] to 
        normalized COCO format [x_min, y_min, width, height] for albumentations.
        
        Albumentations expects bounding box coordinates to be normalized to [0,1]
        """
        x_center, y_center, width, height = bbox
        
        # Convert to normalized COCO format (still within 0-1 range)
        x_min = x_center - (width / 2)
        y_min = y_center - (height / 2)
        
        # Ensure values are within valid range
        x_min = max(0, min(x_min, 1.0 - width))
        y_min = max(0, min(y_min, 1.0 - height))
        width = min(width, 1.0 - x_min)
        height = min(height, 1.0 - y_min)
        
        return [x_min, y_min, width, height] 
       
    def _convert_coco_to_yolo(self, bbox, img_width, img_height):
        """
        Convert normalized COCO format [x_min, y_min, width, height] to 
        normalized YOLO format [x_center, y_center, width, height].
        
        Both formats use coordinates normalized to [0,1] range.
        """
        x_min, y_min, width, height = bbox
        
        # Convert to normalized YOLO format
        x_center = x_min + (width / 2)
        y_center = y_min + (height / 2)
        
        # Ensure values are within valid range
        x_center = max(0, min(x_center, 1.0))
        y_center = max(0, min(y_center, 1.0))
        width = max(0.01, min(width, 1.0))
        height = max(0.01, min(height, 1.0))
        
        return [x_center, y_center, width, height]
        
    def _normalize_after_resize(self, bbox, old_shape, new_shape):
        """
        Adjust bounding box coordinates after image resize.
        
        Args:
            bbox: Bounding box in YOLO format [x_center, y_center, width, height]
            old_shape: Original image shape (height, width)
            new_shape: New image shape (height, width)
            
        Returns:
            Adjusted bounding box in YOLO format
        """
        old_h, old_w = old_shape[:2]
        new_h, new_w = new_shape[:2]
        
        # No need to adjust if same size
        if old_h == new_h and old_w == new_w:
            return bbox
            
        # YOLO format already uses normalized coordinates
        # If the normalization is consistently applied, coordinates should remain valid
        return bbox
    def apply(self, image, bbox):
        """
        Apply data augmentation based on plate size.
        
        Args:
            image: Input image (numpy array)
            bbox: Bounding box in YOLO format [x_center, y_center, width, height]
            
        Returns:
            Tuple of (augmented image, augmented bbox in YOLO format)
        """
        h, w = image.shape[:2]
        original_shape = image.shape
        
        # Calculate plate size as percentage of image area
        plate_area_ratio = bbox[2] * bbox[3]  # width * height (normalized)
        
        # Convert bbox from YOLO to COCO format for albumentations
        try:
            coco_bbox = self._convert_yolo_to_coco(bbox, w, h)
            
            # Choose transformation based on plate size
            if plate_area_ratio < self.small_plate_threshold:
                # For small plates, use specialized transformations
                if np.random.random() < self.extreme_aug_prob:
                    # Apply extreme augmentation occasionally
                    transform = self.extreme_transform
                else:
                    # Apply small plate focused augmentation
                    transform = self.small_plate_transform
            else:
                # For normal-sized plates, use standard transformations
                transform = self.standard_transform
            
            # Apply the selected transformation with proper error handling
            try:
                transformed = transform(image=image, bboxes=[coco_bbox], class_labels=['license_plate'])
                
                # Get the augmented image and bbox
                augmented_image = transformed['image']
                
                # Check if bounding box is preserved (not dropped during augmentation)
                if len(transformed['bboxes']) == 0:
                    # If bbox was lost, return original
                    return image, bbox
                
                augmented_coco_bbox = transformed['bboxes'][0]
                
                # Convert back to YOLO format
                h_new, w_new = augmented_image.shape[:2]
                augmented_bbox = self._convert_coco_to_yolo(augmented_coco_bbox, w_new, h_new)                # Ensure the augmented image has the same dimensions as the original
                if augmented_image.shape[:2] != (h, w):
                    # Store current dimensions and bbox before resize
                    h_aug, w_aug = augmented_image.shape[:2]
                    pre_resize_bbox = augmented_bbox
                    
                    # Resize the image
                    augmented_image = cv2.resize(augmented_image, (w, h))
                    
                    # Adjust the bounding box for the resize
                    # YOLO format is already normalized, so the positions should be preserved
                    # However, very extreme transformations might need minor adjustments
                    augmented_bbox = self._normalize_after_resize(
                        pre_resize_bbox,
                        (h_aug, w_aug),
                        (h, w)
                    )
                
                # Ensure we have the same number of channels
                if len(augmented_image.shape) != len(original_shape) or augmented_image.shape[2] != original_shape[2]:
                    # Handle channel mismatches
                    if len(original_shape) == 3 and len(augmented_image.shape) == 2:
                        # Convert grayscale to RGB if needed
                        augmented_image = cv2.cvtColor(augmented_image, cv2.COLOR_GRAY2RGB)
                    elif len(original_shape) == 2 and len(augmented_image.shape) == 3:
                        # Convert RGB to grayscale if needed
                        augmented_image = cv2.cvtColor(augmented_image, cv2.COLOR_RGB2GRAY)
                    elif len(original_shape) == 3 and len(augmented_image.shape) == 3:
                        # Convert between color spaces if needed
                        if original_shape[2] == 1 and augmented_image.shape[2] == 3:
                            augmented_image = cv2.cvtColor(augmented_image, cv2.COLOR_RGB2GRAY)
                            augmented_image = augmented_image[:, :, np.newaxis]
                        elif original_shape[2] == 3 and augmented_image.shape[2] == 1:
                            augmented_image = cv2.cvtColor(augmented_image, cv2.COLOR_GRAY2RGB)
                
                return augmented_image, augmented_bbox
                
            except (ValueError, IndexError) as e:
                # In case of transformation error, return original
                print(f"Warning: Transformation failed: {e}. Using original image.")
                return image, bbox
                
        except Exception as e:
            # If anything goes wrong, return original
            print(f"Warning: Augmentation error: {e}. Using original image.")
            return image, bbox


def augment_data_with_small_plate_focus(images, bboxes, augmentation_factor=3):
    """
    Perform augmentation with special emphasis on small plates.
    
    Args:
        images: Array of images
        bboxes: Array of bounding boxes in YOLO format [x, y, width, height]
        augmentation_factor: How many augmented versions to create for each image
        
    Returns:
        Tuple of (augmented images, augmented bboxes)
    """
    augmenter = SmallPlateFocusedAugmentation(
        small_plate_threshold=0.05,  # Plates < 5% of image are considered small
        extreme_aug_prob=0.3         # 30% chance of extreme augmentation for small plates
    )
    
    # Calculate plate sizes (area ratio) for each image
    plate_sizes = np.array([box[2] * box[3] for box in bboxes])
    
    # Sort indices by plate size (ascending, so smallest plates come first)
    sorted_indices = np.argsort(plate_sizes)
    
    # Get proportion of dataset with small plates (below threshold)
    small_plate_indices = sorted_indices[plate_sizes[sorted_indices] < 0.05]
    num_small_plates = len(small_plate_indices)
    print(f"Found {num_small_plates} small plates out of {len(images)} images ({num_small_plates/len(images)*100:.1f}%)")
    
    # Create empty arrays for augmented data
    X_aug = []
    y_aug = []
    
    # First add original images
    X_aug.extend(images)
    y_aug.extend(bboxes)
    
    # Calculate augmentations per image (more for small plates)
    base_aug_per_image = augmentation_factor - 1  # -1 because we already added originals
    
    # Oversampling factor for small plates
    small_plate_boost = 2  # Small plates get 2x more augmentations
    
    # Weight for small plates - small plates are more important
    weights = np.ones(len(images))
    weights[small_plate_indices] *= small_plate_boost
    
    # Normalize weights to get probability distribution
    weights = weights / np.sum(weights)
    
    # Calculate total number of augmentations to generate
    total_augmentations = base_aug_per_image * len(images)
      # Randomly select images for augmentation, with higher probability for small plates
    aug_indices = np.random.choice(
        np.arange(len(images)), 
        size=total_augmentations, 
        replace=True, 
        p=weights
    )
    
    # Counter for tracking augmentations per image
    aug_counter = np.zeros(len(images))
    
    # Generate augmentations
    for idx in aug_indices:
        # Limit maximum augmentations per image (to prevent overtraining on just a few)
        if aug_counter[idx] >= (small_plate_boost * base_aug_per_image if idx in small_plate_indices else base_aug_per_image):
            continue
        
        try:
            # Apply augmentation
            aug_img, aug_bbox = augmenter.apply(images[idx], bboxes[idx])
            
            # Verify the augmented image is valid and has the right shape/type
            if aug_img is None or aug_img.size == 0:
                print(f"Warning: Augmentation returned empty image for index {idx}. Skipping.")
                continue
                
            # Add to augmented dataset
            X_aug.append(aug_img)
            y_aug.append(aug_bbox)
            
            # Increment counter
            aug_counter[idx] += 1
            
        except Exception as e:
            print(f"Warning: Failed to augment image at index {idx}: {str(e)}. Skipping.")
            continue
      # Check if all images have the same shape
    shapes = [img.shape for img in X_aug]
    unique_shapes = set(tuple(shape) for shape in shapes)
      # If images have different shapes, resize them to match the original image dimensions
    if len(unique_shapes) > 1:
        print(f"Detected {len(unique_shapes)} different image shapes: {unique_shapes}")
        print("Resizing all augmented images to consistent dimensions...")
        
        # Get the target shape from the original images
        target_shape = images[0].shape
        
        # Resize all augmented images to match the target shape
        for i in range(len(X_aug)):
            if X_aug[i].shape != target_shape:
                # Get current dimensions before resize
                curr_h, curr_w = X_aug[i].shape[:2]
                target_h, target_w = target_shape[:2]
                  # Store the bounding box before resizing
                curr_bbox = y_aug[i].copy()
                
                # Resize the image while preserving the channels
                X_aug[i] = cv2.resize(X_aug[i], (target_w, target_h))
                
                # When we resize an image, we need to ensure the bounding box is still properly normalized
                # Since we're using YOLO format (normalized), the values should remain valid
                # However, we'll ensure they're within proper bounds
                y_aug[i] = np.clip(y_aug[i], 0.0, 1.0)
                
                # Ensure the resized image has the correct number of channels
                if len(X_aug[i].shape) == 2 and len(target_shape) == 3:
                    # Grayscale to RGB
                    X_aug[i] = cv2.cvtColor(X_aug[i], cv2.COLOR_GRAY2RGB)
                elif len(X_aug[i].shape) == 3 and X_aug[i].shape[2] != target_shape[2]:
                    # Fix channel mismatch
                    if target_shape[2] == 3 and X_aug[i].shape[2] == 1:
                        X_aug[i] = cv2.cvtColor(X_aug[i], cv2.COLOR_GRAY2RGB)
                    elif target_shape[2] == 1 and X_aug[i].shape[2] == 3:
                        X_aug[i] = cv2.cvtColor(X_aug[i], cv2.COLOR_RGB2GRAY)
                        X_aug[i] = X_aug[i][:, :, np.newaxis]
    
    # Convert lists to numpy arrays
    X_aug = np.array(X_aug)
    y_aug = np.array(y_aug)
    
    print(f"Generated {len(X_aug)} images (original: {len(images)}, augmented: {len(X_aug) - len(images)})")
    
    return X_aug, y_aug


def create_tf_data_augmenter(cutout_prob=0.3, mixup_prob=0.2):
    """
    Create a function for on-the-fly TensorFlow data augmentation.
    
    Args:
        cutout_prob: Probability of applying cutout augmentation
        mixup_prob: Probability of applying mixup augmentation
        
    Returns:
        Function to apply on-the-fly augmentation to batches
    """
    def augment_fn(images, bboxes):
        """Apply on-the-fly augmentation during training."""
        batch_size = tf.shape(images)[0]
        
        # Additional on-the-fly augmentations
        # These work with batched data during training
        
        # 1. Random brightness and contrast
        images = tf.image.random_brightness(images, max_delta=0.1)
        images = tf.image.random_contrast(images, lower=0.8, upper=1.2)
        
        # 2. Random saturation and hue
        images = tf.image.random_saturation(images, lower=0.8, upper=1.2)
        images = tf.image.random_hue(images, max_delta=0.1)
        
        # 3. Cutout (random erasing) - helps with occlusion robustness
        def apply_cutout(image):
            if tf.random.uniform([]) < cutout_prob:
                # Generate random box
                img_height = tf.cast(tf.shape(image)[0], tf.float32)
                img_width = tf.cast(tf.shape(image)[1], tf.float32)
                
                # Size of cutout (between 10-20% of image)
                size_percent = tf.random.uniform([], 0.1, 0.2)
                cutout_height = tf.cast(img_height * size_percent, tf.int32)
                cutout_width = tf.cast(img_width * size_percent, tf.int32)
                
                # Random position
                y1 = tf.random.uniform([], 0, tf.shape(image)[0] - cutout_height, dtype=tf.int32)
                x1 = tf.random.uniform([], 0, tf.shape(image)[1] - cutout_width, dtype=tf.int32)
                
                # Create mask and apply cutout
                padding = [[y1, tf.shape(image)[0] - y1 - cutout_height], 
                          [x1, tf.shape(image)[1] - x1 - cutout_width],
                          [0, 0]]
                mask = tf.pad(tf.zeros([cutout_height, cutout_width, tf.shape(image)[2]]), padding, constant_values=1)
                return image * mask
            return image
        
        # Apply cutout to each image
        images = tf.map_fn(apply_cutout, images)
        
        # 4. MixUp - linear interpolation of images and labels
        def apply_mixup(batch_images, batch_bboxes):
            if tf.random.uniform([]) < mixup_prob:
                # Random shuffle indices
                indices = tf.random.shuffle(tf.range(batch_size))
                
                # Get shuffled images and bboxes
                shuffled_images = tf.gather(batch_images, indices)
                shuffled_bboxes = tf.gather(batch_bboxes, indices)
                
                # Random interpolation weight
                alpha = tf.random.uniform([], 0.2, 0.4)
                
                # Interpolate images
                mixed_images = alpha * batch_images + (1 - alpha) * shuffled_images
                
                # For bboxes, we just append both sets (will be filtered by NMS later)
                # This is a simplified approach - in production you might want more sophisticated bbox mixing
                mixed_bboxes = batch_bboxes  # Keep original bboxes
                
                return mixed_images, mixed_bboxes
            return batch_images, batch_bboxes
        
        # Only apply mixup if batch size > 1
        if batch_size > 1:
            images, bboxes = apply_mixup(images, bboxes)
        
        # Ensure pixel values remain in [0, 1]
        images = tf.clip_by_value(images, 0.0, 1.0)
        
        # Ensure bbox coordinates remain valid
        bboxes = tf.clip_by_value(bboxes, 0.0, 1.0)
        
        return images, bboxes
    
    return augment_fn
