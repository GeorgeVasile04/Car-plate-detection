"""
License plate detection package.

This package provides tools for detecting license plates in images and videos.
"""

__version__ = '1.0.0'

# Import main components for easier access
from license_plate_detection.models.detector import (
    create_license_plate_detector,
    create_enhanced_license_plate_detector,
    create_mobilenet_license_plate_detector
)

from license_plate_detection.data.loader import (
    get_data_path,
    load_dataset,
    preprocess_dataset,
    split_dataset
)

from license_plate_detection.models.losses import (
    enhanced_iou_metric,
    combined_detection_loss,
    giou_loss
)

from license_plate_detection.utils.visualization import (
    visualize_prediction,
    visualize_batch_predictions,
    plot_training_history
)

# Import entry points
from license_plate_detection.main import (
    load_and_prepare_data,
    create_model_by_name,
    train,
    evaluate,
    predict
)
