"""
Utility functions for license plate detection.
"""

from license_plate_detection.utils.helpers import detect_license_plate, load_and_prepare_model
from license_plate_detection.utils.visualization import visualize_prediction, plot_training_history
from license_plate_detection.utils.analysis import analyze_error_patterns
from license_plate_detection.utils.memory_optimizations import (
    optimize_memory_usage, 
    setup_gpu_memory_growth,
    limit_gpu_memory,
    enable_mixed_precision,
    enable_gradient_checkpointing,
    clean_memory
)