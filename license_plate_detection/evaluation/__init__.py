"""
Evaluation utilities for license plate detection.
"""

from license_plate_detection.evaluation.evaluator import (
    evaluate_license_plate_detection,
    evaluate_model_comprehensive,
    detect_license_plate,
    detect_plate_from_dataset
)

from license_plate_detection.evaluation.error_analysis import (
    analyze_predictions,
    visualize_error_patterns,
    identify_error_patterns,
    compare_models
)

from license_plate_detection.evaluation.demo import (
    generate_demo_predictions,
    create_mock_comprehensive_results
)

__all__ = [
    'evaluate_license_plate_detection',
    'evaluate_model_comprehensive',
    'detect_license_plate',
    'detect_plate_from_dataset',
    'analyze_predictions',
    'visualize_error_patterns',
    'identify_error_patterns',
    'visualize_batch_predictions',
    'compare_models',
    'generate_demo_predictions',
    'create_mock_comprehensive_results'
]