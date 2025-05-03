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

from license_plate_detection.models.losses import (
    enhanced_iou_metric,
    combined_detection_loss,
    giou_loss,
    focal_loss_bbox,
    size_sensitive_loss,
    improved_combined_detection_loss
)

from license_plate_detection.data.loader import (
    get_data_path,
    load_license_plate_dataset,
    preprocess_license_plate_dataset,
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

# Import helper functions (previously in main.py)
from license_plate_detection.utils.helpers import (
    detect_license_plate,
    load_and_prepare_model,
    load_and_prepare_data
)

# Import evaluation and demo functions
from license_plate_detection.evaluation.demo import (
    generate_demo_predictions,
    create_mock_comprehensive_results
)

from license_plate_detection.evaluation.evaluator import (
    evaluate_model_comprehensive
)
