"""
Training functions for license plate detection models.
"""

import os
import numpy as np
import tensorflow as tf
from pathlib import Path
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    ReduceLROnPlateau,
    TensorBoard
)

from license_plate_detection.train.scheduler import create_lr_scheduler


def create_training_callbacks(checkpoint_path=None, early_stopping=True, patience=10, 
                             reduce_lr=True, tensorboard_log_dir=None):
    """
    Create callbacks for model training.
    
    Args:
        checkpoint_path: Path to save model checkpoints
        early_stopping: Whether to use early stopping
        patience: Patience for early stopping and reduce LR
        reduce_lr: Whether to use reduce learning rate on plateau
        tensorboard_log_dir: Directory for TensorBoard logs
        
    Returns:
        list: List of callbacks
    """
    callbacks = []
    
    # Model checkpoint
    if checkpoint_path is not None:
        checkpoint_dir = os.path.dirname(checkpoint_path)
        if checkpoint_dir:
            os.makedirs(checkpoint_dir, exist_ok=True)
            
        callbacks.append(ModelCheckpoint(
            filepath=checkpoint_path,
            monitor='val_enhanced_iou_metric',
            mode='max',
            save_best_only=True,
            save_weights_only=False,
            verbose=1
        ))
    
    # Early stopping
    if early_stopping:
        callbacks.append(EarlyStopping(
            monitor='val_loss',
            patience=patience,
            restore_best_weights=True,
            verbose=1
        ))
    
    # Reduce learning rate on plateau
    if reduce_lr:
        callbacks.append(ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=patience // 2,
            min_lr=1e-6,
            verbose=1
        ))
    
    # TensorBoard
    if tensorboard_log_dir is not None:
        os.makedirs(tensorboard_log_dir, exist_ok=True)
        callbacks.append(TensorBoard(
            log_dir=tensorboard_log_dir,
            histogram_freq=1,
            update_freq='epoch',
            profile_batch=0
        ))
    
    return callbacks


def train_model(model, X_train, y_train, X_val=None, y_val=None, epochs=50, batch_size=16,
               callbacks=None, custom_scheduler=None, verbose=1):
    """
    Train a license plate detection model.
    
    Args:
        model: Keras model
        X_train: Training data (images)
        y_train: Training labels (bounding boxes)
        X_val: Validation data
        y_val: Validation labels
        epochs: Number of epochs
        batch_size: Batch size
        callbacks: List of callbacks
        custom_scheduler: Custom learning rate scheduler
        verbose: Verbosity level
        
    Returns:
        tuple: (training history, trained model)
    """
    # Prepare validation data
    validation_data = None
    if X_val is not None and y_val is not None:
        validation_data = (X_val, y_val)
    
    # Add custom scheduler if provided
    all_callbacks = callbacks or []
    if custom_scheduler is not None:
        all_callbacks.append(custom_scheduler)
    
    # Train the model
    history = model.fit(
        X_train, y_train,
        validation_data=validation_data,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=all_callbacks,
        verbose=verbose
    )
    
    return history, model


def train_with_augmentation(model, X_train, y_train, X_val=None, y_val=None, 
                           augment_fn=None, epochs=50, batch_size=16, callbacks=None):
    """
    Train a license plate detection model with on-the-fly data augmentation.
    
    Args:
        model: Keras model
        X_train: Training data (images)
        y_train: Training labels (bounding boxes)
        X_val: Validation data
        y_val: Validation labels
        augment_fn: Augmentation function that takes (X, y) and returns augmented (X, y)
        epochs: Number of epochs
        batch_size: Batch size
        callbacks: List of callbacks
        
    Returns:
        tuple: (training history, trained model)
    """
    if augment_fn is None:
        # Use standard training if no augmentation function provided
        return train_model(model, X_train, y_train, X_val, y_val, epochs, batch_size, callbacks)
    
    # Create a custom data generator with augmentation
    def data_generator(X, y, batch_size):
        num_samples = len(X)
        indices = np.arange(num_samples)
        
        while True:
            np.random.shuffle(indices)
            
            for start_idx in range(0, num_samples, batch_size):
                batch_indices = indices[start_idx:start_idx + batch_size]
                X_batch = X[batch_indices]
                y_batch = y[batch_indices]
                
                # Apply augmentation
                X_aug, y_aug = augment_fn(X_batch, y_batch)
                
                yield X_aug, y_aug
    
    # Prepare validation data
    validation_data = None
    if X_val is not None and y_val is not None:
        validation_data = (X_val, y_val)
    
    # Train the model with the generator
    history = model.fit(
        data_generator(X_train, y_train, batch_size),
        validation_data=validation_data,
        steps_per_epoch=len(X_train) // batch_size,
        epochs=epochs,
        callbacks=callbacks,
        verbose=1
    )
    
    return history, model


def save_model(model, filepath, save_optimizer=False):
    """
    Save a trained model.
    
    Args:
        model: Keras model
        filepath: Path to save the model
        save_optimizer: Whether to save optimizer state
        
    Returns:
        str: Path to the saved model
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    if save_optimizer:
        model.save(filepath)
    else:
        # Save only weights and architecture, not optimizer state
        model_weights_path = filepath.with_suffix('.weights.h5')
        model.save_weights(model_weights_path)
        
        # Save model architecture as JSON
        model_json = model.to_json()
        model_json_path = filepath.with_suffix('.json')
        with open(model_json_path, 'w') as f:
            f.write(model_json)
        
        # Also save complete model for convenience
        model.save(filepath)
    
    return str(filepath)
