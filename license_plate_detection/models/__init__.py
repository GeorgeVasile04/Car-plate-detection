"""
Model-related components for license plate detection.
"""

from license_plate_detection.models.detector import (
    create_license_plate_detector,
    create_enhanced_license_plate_detector
)

from license_plate_detection.models.losses import (
    calculate_iou,
    enhanced_iou_metric,
    combined_detection_loss,
    giou_loss,
    focal_loss_bbox,
    size_sensitive_loss,
    improved_combined_detection_loss
)