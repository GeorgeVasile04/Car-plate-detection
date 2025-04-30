# License Plate Detection Module Documentation

## Overview

This module implements a modular architecture for license plate detection in images and videos. It provides a complete end-to-end pipeline from data loading and preprocessing to model training, evaluation, and prediction.

## Module Structure

```
license_plate_detection/
├── __init__.py           # Package initialization
├── cli.py                # Command-line interface
├── main.py               # Main entry point and high-level functions
├── data/                 # Data handling modules
│   ├── loader.py         # Data loading and preprocessing
│   └── augmentation.py   # Data augmentation techniques
├── models/               # Model architecture modules
│   ├── detector.py       # License plate detector models
│   └── losses.py         # Custom loss functions and metrics
├── train/                # Training modules
│   ├── trainer.py        # Model training functions
│   └── scheduler.py      # Learning rate scheduling
├── evaluation/           # Evaluation modules
│   └── error_analysis.py # Error analysis and metrics
└── utils/                # Utility modules
    └── visualization.py  # Visualization tools
```

## Core Functionality

### Data Handling

- **loader.py**: Functions for loading and preprocessing license plate datasets
  - `get_data_path()`: Get the default data path
  - `load_dataset()`: Load annotations and images
  - `preprocess_dataset()`: Preprocess images and bounding boxes
  - `split_dataset()`: Split data into training and validation sets

### Models

- **detector.py**: License plate detector model architectures
  - `create_license_plate_detector()`: Simple CNN-based detector
  - `create_enhanced_license_plate_detector()`: Enhanced CNN with residual connections
  - `create_mobilenet_license_plate_detector()`: MobileNet-based detector
  - `create_efficientnet_license_plate_detector()`: EfficientNet-based detector

- **losses.py**: Custom loss functions and metrics
  - `enhanced_iou_metric()`: IoU (Intersection over Union) metric
  - `combined_detection_loss()`: Combined loss function for bounding box regression
  - `giou_loss()`: Generalized IoU loss

### Training

- **trainer.py**: Model training functions
  - `train_model()`: Train a license plate detection model
  - `create_training_callbacks()`: Create callbacks for training
  - `save_model()`: Save a trained model

- **scheduler.py**: Learning rate scheduling
  - `cosine_decay_scheduler()`: Cosine decay scheduler
  - `step_decay_scheduler()`: Step decay scheduler

### Evaluation

- **error_analysis.py**: Error analysis and metrics
  - `analyze_predictions()`: Analyze model predictions
  - `calculate_iou()`: Calculate IoU between boxes
  - `identify_error_patterns()`: Identify common error patterns
  - `analyze_error_distribution()`: Analyze error distribution
  - `visualize_error_patterns()`: Visualize error patterns

### Utilities

- **visualization.py**: Visualization tools
  - `visualize_prediction()`: Visualize a single prediction
  - `visualize_batch_predictions()`: Visualize multiple predictions
  - `plot_training_history()`: Plot training history
  - `create_prediction_video()`: Create a video with predictions

## Command-Line Interface

The package provides a command-line interface for common operations:

### Training

```bash
python -m license_plate_detection train \
  --annotations_path path/to/annotations \
  --images_path path/to/images \
  --model enhanced \
  --epochs 50 \
  --batch_size 16 \
  --output_dir output
```

### Evaluation

```bash
python -m license_plate_detection evaluate \
  --model_path path/to/model.h5 \
  --annotations_path path/to/annotations \
  --images_path path/to/images \
  --output_dir evaluation_output
```

### Prediction

```bash
# On an image
python -m license_plate_detection predict \
  --model_path path/to/model.h5 \
  --image_path path/to/image.jpg \
  --output_path path/to/output.png

# On a video
python -m license_plate_detection predict \
  --model_path path/to/model.h5 \
  --video_path path/to/video.mp4 \
  --output_path path/to/output.mp4
```

## Usage Examples

### Python API

```python
import numpy as np
from license_plate_detection import (
    load_and_prepare_data, 
    create_license_plate_detector,
    train_model,
    enhanced_iou_metric,
    combined_detection_loss
)

# Load data
X, y, _ = load_and_prepare_data()

# Split data
from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Create model
model = create_license_plate_detector(input_shape=(224, 224, 3))
model.compile(
    optimizer='adam',
    loss=combined_detection_loss,
    metrics=[enhanced_iou_metric]
)

# Train model
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=10,
    batch_size=16
)

# Make predictions
y_pred = model.predict(X_val)

# Visualize results
from license_plate_detection import visualize_prediction
import matplotlib.pyplot as plt

fig = visualize_prediction(X_val[0], y_val[0], y_pred[0])
plt.show()
```

## Dependencies

- TensorFlow 2.x
- NumPy
- Matplotlib
- OpenCV (for image processing)
- scikit-learn (for train/test split)
