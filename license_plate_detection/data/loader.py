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
import pandas as pd  # Adding pandas import for DataFrame operations

# Remove circular import to prevent code execution twice
# Only import specific functions, not the whole module
from license_plate_detection.data.augmentation import augment_data, visualize_augmentation

# Debug print flag - set to False to suppress redundant messages
DEBUG_PRINTS = False

def get_data_path():
    """
    Get the path to the dataset directory.
    
    Returns:
        Path: Path object pointing to the dataset directory
    """
    # Try to detect if running in Colab
    import importlib.util
    IN_COLAB = importlib.util.find_spec("google.colab") is not None
    
    current_dir = os.getcwd()
    
    if IN_COLAB:
        # In Colab, the dataset should be in the cloned repository
        project_root = Path(current_dir) / "Car-plate-detection"
        data_path = project_root / "Dataset"
    else:
        # If not in Colab, assume we're in the project directory
        project_root = Path(os.getcwd()).parent
        data_path = project_root / "Dataset"
    
    # Verify that the dataset exists
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found at {data_path}")
    
    return data_path


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

def load_license_plate_dataset(annotations_dir, images_dir):
    """
    Load a dataset of license plate images and annotations into a DataFrame.
    This function has a single responsibility: loading data into a DataFrame.
    
    Args:
        annotations_dir: Directory containing annotations in XML format
        images_dir: Directory containing images
        
    Returns:
        pandas.DataFrame: DataFrame with columns [image_path, x, y, w, h, plate_text]
    """
    # Convert to Path objects for consistency
    annotations_dir = Path(annotations_dir)
    images_dir = Path(images_dir)
    
    # Prepare a list to collect the dataset records
    dataset = []
    skipped = 0
    
    # Loop through all files in the annotations folder
    for file in annotations_dir.iterdir():
        if file.suffix == ".xml":
            try:
                tree = ET.parse(file)
                root = tree.getroot()
                
                # Get filename from root
                try:
                    img_name = root.find('filename').text
                    img_path = images_dir / img_name
                    
                    if not img_path.exists():
                        # Try alternative ways to find the image
                        alternative_path = images_dir / f"{file.stem.replace('data_', '')}.jpg"
                        if alternative_path.exists():
                            img_path = alternative_path
                        else:
                            if DEBUG_PRINTS:
                                print(f"Image not found for annotation: {img_name}")
                            skipped += 1
                            continue
                except AttributeError:
                    if DEBUG_PRINTS:
                        print(f"Could not find filename in {file}, skipping")
                    skipped += 1
                    continue
                
                for member in root.findall('object'):
                    try:
                        # Extract bounding box coordinates
                        bbox = member.find('bndbox')
                        xmin = int(bbox.find('xmin').text)
                        ymin = int(bbox.find('ymin').text)
                        xmax = int(bbox.find('xmax').text)
                        ymax = int(bbox.find('ymax').text)
                        
                        # Calculate width and height
                        w = xmax - xmin
                        h = ymax - ymin
                        
                        # Try to find plate_text in different elements
                        plate_text = "Unknown"  # Default if no text found
                        
                        # Check multiple possible elements for plate text
                        for text_elem_name in ['n', 'name', 'license_text', 'plate_text']:
                            text_elem = member.find(text_elem_name)
                            if text_elem is not None and text_elem.text:
                                plate_text = text_elem.text
                                break
                        
                        dataset.append({
                            "image_path": str(img_path),
                            "x": xmin,
                            "y": ymin,
                            "w": w,
                            "h": h,
                            "plate_text": plate_text
                        })
                    except Exception as e:
                        if DEBUG_PRINTS:
                            print(f"Error processing object in {file}: {e}")
            except Exception as e:
                if DEBUG_PRINTS:
                    print(f"Error processing {file}: {e}")
                skipped += 1
    
    if DEBUG_PRINTS:
        print(f"Loaded {len(dataset)} annotations, skipped {skipped} files")
    
    # Convert to DataFrame
    df = pd.DataFrame(dataset)
    return df


def preprocess_license_plate_dataset(df, image_size=(224, 224)):
    """
    Preprocess a dataset of license plate images and annotations.
    This function has a single responsibility: converting DataFrame data to model-ready tensors.
    
    Args:
        df: DataFrame with columns ['image_path', 'x', 'y', 'w', 'h']
        image_size: Target size for images (width, height)
        
    Returns:
        tuple: (X, y) where X is normalized images and y is normalized bounding boxes
    """
    # Prepare data containers
    X = []
    y = []
    
    # Process each row in the DataFrame
    for _, row in df.iterrows():
        try:
            # Load image
            img_path = row['image_path']
            img = cv2.imread(img_path)
            
            if img is None:
                if DEBUG_PRINTS:
                    print(f"Could not read image: {img_path}")
                continue
                
            # Get original dimensions
            orig_h, orig_w = img.shape[:2]
            
            # Convert from BGR to RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Resize image
            img = cv2.resize(img, image_size)
            
            # Normalize pixel values to [0, 1]
            img = img.astype(np.float32) / 255.0
            
            # Get bounding box
            x, y_coord, w, h = row['x'], row['y'], row['w'], row['h']
            
            # Normalize bounding box coordinates
            x_norm = x / orig_w
            y_norm = y_coord / orig_h
            w_norm = w / orig_w
            h_norm = h / orig_h
            
            # Add to dataset
            X.append(img)
            y.append([x_norm, y_norm, w_norm, h_norm])
            
        except Exception as e:
            if DEBUG_PRINTS:
                print(f"Error processing {row['image_path']}: {e}")
    
    if DEBUG_PRINTS:
        print(f"Processed {len(X)} images")
    
    # Convert to numpy arrays
    X = np.array(X)
    y = np.array(y)
    
    return X, y


def split_dataset(X, y, test_size=0.2, random_state=42):
    """
    Split a dataset into training and validation sets.
    
    Args:
        X: Image data
        y: Bounding box labels
        test_size: Proportion of the dataset to include in the validation split
        random_state: Random state for reproducibility
        
    Returns:
        tuple: (X_train, X_val, y_train, y_val)
    """
    # Use sklearn's train_test_split to split the data
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    print(f"Training samples: {len(X_train)}, Validation samples: {len(X_val)}")
    
    return X_train, X_val, y_train, y_val


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
