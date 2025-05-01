"""
Helper functions for license plate detection.
Previously housed in main.py, now separated for modularity.
"""

import os
import numpy as np
import cv2
import tensorflow as tf
from pathlib import Path
import matplotlib.pyplot as plt

# Import necessary modules - DIRECTLY IMPORT SPECIFIC FUNCTIONS
# to avoid circular imports
from license_plate_detection.models.losses import enhanced_iou_metric, combined_detection_loss, giou_loss
from license_plate_detection.models.detector import (
    create_license_plate_detector, 
    create_enhanced_license_plate_detector,
    create_mobilenet_license_plate_detector
)

# DO NOT import from loader.py here to prevent circular imports

def detect_license_plate(image_path, model=None, model_path=None, image_size=224, confidence_threshold=None):
    """
    Detect license plate in an image.
    
    Args:
        image_path: Path to the image file
        model: Pre-loaded model (if None, model_path must be provided)
        model_path: Path to the model file (if model is None)
        image_size: Image size for prediction
        confidence_threshold: Threshold for detection confidence (not used in current implementation)
    
    Returns:
        tuple: (plate_region, bbox) where plate_region is the cropped image of the license plate
               and bbox is [x, y, width, height] in original image coordinates
    """
    if model is None and model_path is None:
        raise ValueError("Either model or model_path must be provided")
    
    # Load model if not provided
    if model is None:
        try:
            model = tf.keras.models.load_model(
                model_path,
                custom_objects={
                    'enhanced_iou_metric': enhanced_iou_metric,
                    'combined_detection_loss': combined_detection_loss,
                    'giou_loss': giou_loss
                }
            )
            print(f"Model loaded from {model_path}")
        except Exception as e:
            raise RuntimeError(f"Error loading model: {e}")
    
    try:
        # Load image
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not read image: {image_path}")
            
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Save original dimensions
        orig_h, orig_w = img.shape[:2]
        
        # Resize for model input
        img_resized = cv2.resize(img, (image_size, image_size))
        
        # Normalize
        img_normalized = img_resized.astype(np.float32) / 255.0
        
        # Add batch dimension
        img_batch = np.expand_dims(img_normalized, axis=0)
        
        # Make prediction
        pred_box = model.predict(img_batch)[0]
        
        # Calculate bounding box in original image
        x, y, w, h = pred_box
        
        # Convert normalized coordinates to image coordinates
        x_center = int(x * orig_w)
        y_center = int(y * orig_h)
        width = int(w * orig_w)
        height = int(h * orig_h)
        
        # Calculate top-left corner
        x1 = max(0, x_center - width // 2)
        y1 = max(0, y_center - height // 2)
        
        # Ensure box doesn't go beyond image boundaries
        width = min(width, orig_w - x1)
        height = min(height, orig_h - y1)
        
        # Extract plate region
        plate_region = img[y1:y1+height, x1:x1+width]
        
        # Return plate region and bounding box in [x, y, width, height] format
        # where x,y is the top-left corner in the original image coordinates
        return plate_region, [x1, y1, width, height]
        
    except Exception as e:
        raise RuntimeError(f"Error detecting license plate: {e}")


def load_and_prepare_model(model_path=None, model_type='enhanced', input_shape=(224, 224, 3)):
    """
    Load and prepare a license plate detection model.
    
    Args:
        model_path: Path to the saved model file (if None, creates a new model)
        model_type: Type of model to create if model_path is None
        input_shape: Input shape for the model
    
    Returns:
        model: Loaded or created model
    """
    if model_path is not None:
        try:
            model = tf.keras.models.load_model(
                model_path,
                custom_objects={
                    'enhanced_iou_metric': enhanced_iou_metric,
                    'combined_detection_loss': combined_detection_loss,
                    'giou_loss': giou_loss
                }
            )
            print(f"Model loaded from {model_path}")
            return model
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Creating a new model instead.")
    
    # Create a new model based on the specified type
    if model_type == 'basic':
        model = create_license_plate_detector(input_shape=input_shape)
    elif model_type == 'enhanced':
        model = create_enhanced_license_plate_detector(input_shape=input_shape)
    elif model_type == 'mobilenet':
        model = create_mobilenet_license_plate_detector(input_shape=input_shape)
    else:
        print(f"Unknown model name: {model_type}. Using enhanced model instead.")
        model = create_enhanced_license_plate_detector(input_shape=input_shape)
    
    # Compile model with default settings
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss=combined_detection_loss,
        metrics=[enhanced_iou_metric]
    )
    
    print(f"Created new {model_type} model")
    return model


def load_and_prepare_data(df, image_size=(224, 224), from_df=True):
    """
    Prepare data for model training or evaluation from a DataFrame.
    
    Args:
        df: DataFrame with image paths and bounding box coordinates
        image_size: Target image size
        from_df: Flag indicating the data is coming from a DataFrame
        
    Returns:
        tuple: (X, y) processed images and labels
    """
    # Only import preprocess_license_plate_dataset when function is called
    # to avoid circular import problems
    if from_df:
        from license_plate_detection.data.loader import preprocess_license_plate_dataset
        X, y = preprocess_license_plate_dataset(df, image_size=image_size)
        return X, y
    else:
        raise ValueError("Only DataFrame input is supported in this version")
