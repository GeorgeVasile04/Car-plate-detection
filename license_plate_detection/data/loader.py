"""
Data loading and preprocessing functions for license plate detection.
"""

import os
import numpy as np
import tensorflow as tf
import cv2
import xml.etree.ElementTree as ET
from pathlib import Path
from sklearn.model_selection import train_test_split
import albumentations as A


def parse_annotation_xml(xml_path):
    """
    Parse XML annotation file for license plate bounding box.
    
    Args:
        xml_path: Path to XML annotation file
        
    Returns:
        dict: Dictionary with license plate bounding box [x, y, width, height]
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    # Get image dimensions
    size = root.find('size')
    width = int(size.find('width').text)
    height = int(size.find('height').text)
    
    # Find the license plate object
    license_plate_box = None
    for obj in root.findall('object'):
        name = obj.find('name').text
        if name.lower() in ['plate', 'license', 'license_plate', 'licence', 'licence_plate']:
            bbox = obj.find('bndbox')
            xmin = float(bbox.find('xmin').text)
            ymin = float(bbox.find('ymin').text)
            xmax = float(bbox.find('xmax').text)
            ymax = float(bbox.find('ymax').text)
            
            # Convert to [x, y, width, height] format
            x = (xmin + xmax) / 2.0
            y = (ymin + ymax) / 2.0
            w = xmax - xmin
            h = ymax - ymin
            
            # Normalize coordinates
            x /= width
            y /= height
            w /= width
            h /= height
            
            license_plate_box = [x, y, w, h]
            break
    
    return {
        'width': width,
        'height': height,
        'box': license_plate_box
    }


def parse_yolo_annotation(txt_path, img_width, img_height):
    """
    Parse YOLO format annotation file for license plate bounding box.
    
    Args:
        txt_path: Path to YOLO annotation file
        img_width: Image width
        img_height: Image height
        
    Returns:
        dict: Dictionary with license plate bounding box [x, y, width, height]
    """
    license_plate_box = None
    
    if os.path.exists(txt_path):
        with open(txt_path, 'r') as f:
            lines = f.readlines()
            
            for line in lines:
                parts = line.strip().split()
                class_id = int(parts[0])
                # YOLO format: [class_id, x_center, y_center, width, height]
                # Already normalized between 0 and 1
                x = float(parts[1])
                y = float(parts[2])
                w = float(parts[3])
                h = float(parts[4])
                
                license_plate_box = [x, y, w, h]
                break
    
    return {
        'width': img_width,
        'height': img_height,
        'box': license_plate_box
    }


def load_image(img_path, target_size=None):
    """
    Load and preprocess an image.
    
    Args:
        img_path: Path to the image
        target_size: Target size (height, width) or None to keep original size
        
    Returns:
        Preprocessed image as numpy array
    """
    # Read image
    img = cv2.imread(str(img_path))
    if img is None:
        raise ValueError(f"Could not read image: {img_path}")
    
    # Convert from BGR to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Resize if target size is specified
    if target_size is not None:
        img = cv2.resize(img, (target_size[1], target_size[0]))
    
    # Normalize pixel values to [0, 1]
    img = img.astype(np.float32) / 255.0
    
    return img


def load_dataset(images_dir, annotations_dir, annotation_format='xml', target_size=(224, 224), test_size=0.2, random_state=42):
    """
    Load and preprocess a dataset of license plate images and annotations.
    
    Args:
        images_dir: Directory containing images
        annotations_dir: Directory containing annotations
        annotation_format: Format of annotations ('xml' or 'yolo')
        target_size: Target size for images (height, width)
        test_size: Proportion of the dataset to include in the test split
        random_state: Random state for reproducibility
        
    Returns:
        tuple: (X_train, y_train, X_test, y_test) arrays
    """
    images_dir = Path(images_dir)
    annotations_dir = Path(annotations_dir)
    
    # Get list of image files
    image_files = []
    valid_extensions = ['.jpg', '.jpeg', '.png']
    for ext in valid_extensions:
        image_files.extend(list(images_dir.glob(f'*{ext}')))
    
    # Prepare data containers
    X = []
    y = []
    skipped = 0
    
    # Process each image
    for img_path in image_files:
        try:
            # Get corresponding annotation file
            if annotation_format == 'xml':
                anno_path = annotations_dir / f"data_{img_path.stem}.xml"
                if not anno_path.exists():
                    anno_path = annotations_dir / f"{img_path.stem}.xml"
            else:  # YOLO format
                anno_path = annotations_dir / f"{img_path.stem}.txt"
            
            if not anno_path.exists():
                print(f"Annotation not found for {img_path.name}, skipping")
                skipped += 1
                continue
            
            # Load image
            img = load_image(img_path, target_size)
            
            # Parse annotation
            if annotation_format == 'xml':
                anno_data = parse_annotation_xml(anno_path)
            else:
                anno_data = parse_yolo_annotation(anno_path, target_size[1], target_size[0])
            
            # Check if license plate was found
            if anno_data['box'] is None:
                print(f"No license plate found in {anno_path.name}, skipping")
                skipped += 1
                continue
            
            # Add to dataset
            X.append(img)
            y.append(anno_data['box'])
            
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
            skipped += 1
    
    print(f"Loaded {len(X)} images, skipped {skipped} images")
    
    # Convert to numpy arrays
    X = np.array(X)
    y = np.array(y)
    
    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    
    return X_train, y_train, X_test, y_test


def create_data_augmentation_pipeline():
    """
    Create a data augmentation pipeline using Albumentations.
    
    Returns:
        Albumentations Compose object with transformations
    """
    return A.Compose([
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        A.GaussianBlur(blur_limit=3, p=0.3),
        A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
        A.Rotate(limit=15, p=0.5),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.5),
        A.HorizontalFlip(p=0.5),
        A.OneOf([
            A.RandomRain(p=0.5),
            A.RandomFog(p=0.5),
            A.RandomShadow(p=0.5)
        ], p=0.3)
    ], bbox_params=A.BboxParams(format='yolo', label_fields=[]))


def apply_augmentation(images, boxes):
    """
    Apply data augmentation to a batch of images and bounding boxes.
    
    Args:
        images: Batch of images (numpy array)
        boxes: Batch of bounding boxes in [x, y, width, height] format
        
    Returns:
        tuple: (augmented_images, augmented_boxes)
    """
    transform = create_data_augmentation_pipeline()
    
    augmented_images = []
    augmented_boxes = []
    
    for img, box in zip(images, boxes):
        # Apply transformation
        transformed = transform(
            image=img,
            bboxes=[box],  # Albumentations expects a list of boxes
        )
        
        augmented_images.append(transformed['image'])
        # Get the first (and only) box
        augmented_boxes.append(transformed['bboxes'][0])
    
    return np.array(augmented_images), np.array(augmented_boxes)


def create_tf_dataset(X, y, batch_size=16, augment=False, shuffle=True, repeat=False):
    """
    Create a TensorFlow dataset for efficient training.
    
    Args:
        X: Image data
        y: Bounding box labels
        batch_size: Batch size
        augment: Whether to apply data augmentation
        shuffle: Whether to shuffle the dataset
        repeat: Whether to repeat the dataset
        
    Returns:
        tf.data.Dataset: TensorFlow dataset
    """
    # Create dataset
    dataset = tf.data.Dataset.from_tensor_slices((X, y))
    
    # Apply operations
    if shuffle:
        dataset = dataset.shuffle(buffer_size=len(X))
    
    if augment:
        # Apply custom augmentation function
        @tf.function
        def augment_map_fn(x, y):
            aug_x, aug_y = tf.numpy_function(
                apply_augmentation, [tf.expand_dims(x, 0), tf.expand_dims(y, 0)], 
                [tf.float32, tf.float32]
            )
            return tf.squeeze(aug_x, 0), tf.squeeze(aug_y, 0)
        
        dataset = dataset.map(augment_map_fn, num_parallel_calls=tf.data.AUTOTUNE)
    
    # Batch and prefetch
    dataset = dataset.batch(batch_size)
    
    if repeat:
        dataset = dataset.repeat()
    
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    
    return dataset
