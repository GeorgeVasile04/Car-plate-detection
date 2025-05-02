"""
Training utilities for license plate detection models.
"""

from license_plate_detection.train.trainer import (
    train_model, 
    save_model, 
    create_training_callbacks,
    create_efficient_data_pipeline,
    train_model_with_datasets
)

from license_plate_detection.train.scheduler import (
    create_lr_scheduler,
    cosine_decay_scheduler,
    step_decay_scheduler,
)