"""
Training functions for license plate detection models.
"""

import os
import numpy as np
import tensorflow as tf
from pathlib import Path
import matplotlib.pyplot as plt
try:
    from tensorflow.keras.callbacks import (
        ModelCheckpoint,
        EarlyStopping,
        ReduceLROnPlateau,
        TensorBoard
    )
except ImportError:
    from keras.callbacks import (
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


def train_model(model, X_train, y_train, X_val=None, y_val=None, epochs=50, batch_size=32,
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
                           augment_fn=None, epochs=50, batch_size=32, callbacks=None):
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


def create_efficient_data_pipeline(X_train, y_train, X_val, y_val, batch_size=16):
    """
    Create memory-efficient TensorFlow data pipelines for training.
    
    Args:
        X_train: Training data
        y_train: Training labels
        X_val: Validation data
        y_val: Validation labels
        batch_size: Batch size for training
        
    Returns:
        tuple: (train_dataset, val_dataset)
    """
    # Create TF datasets
    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val))
    
    # Optimize pipeline for performance and memory usage
    AUTOTUNE = tf.data.AUTOTUNE
    
    # Smaller shuffle buffer to reduce memory usage
    shuffle_buffer = min(1000, len(X_train) // 10)  # 10% of training data or max 1000
    
    train_dataset = (train_dataset
                    .cache()  # Cache the dataset in memory to avoid recomputing
                    .shuffle(buffer_size=shuffle_buffer)
                    .batch(batch_size)
                    .prefetch(AUTOTUNE))
    
    val_dataset = (val_dataset
                  .batch(batch_size)
                  .prefetch(AUTOTUNE))
    
    # Force the dataset to load only when needed
    options = tf.data.Options()
    options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
    options.deterministic = False  # Improve performance
    
    # Enable parallelism if available in your TF version
    try:
        options.experimental_optimization.map_parallelization = True
        options.experimental_optimization.parallel_batch = True
    except:
        pass  # Older version of TF may not have these options
    
    train_dataset = train_dataset.with_options(options)
    val_dataset = val_dataset.with_options(options)
    
    print(f"Created memory-optimized TF data pipelines with batch size {batch_size}")
    return train_dataset, val_dataset


def train_model_with_datasets(model, train_dataset, val_dataset, epochs=50, callbacks=None, verbose=1):
    """
    Train a model using the TensorFlow Dataset API which is more memory-efficient.
    
    Args:
        model: The model to train
        train_dataset: TensorFlow dataset for training
        val_dataset: TensorFlow dataset for validation
        epochs: Number of training epochs
        callbacks: List of Keras callbacks
        verbose: Verbosity level
        
    Returns:
        tuple: (history, trained_model)
    """
    if callbacks is None:
        callbacks = []
    
    # Train using the dataset API
    history = model.fit(
        train_dataset,
        epochs=epochs,
        validation_data=val_dataset,
        callbacks=callbacks,
        verbose=verbose
    )
    return history, model


def train_with_gradient_accumulation(model, X_train, y_train, X_val=None, y_val=None, 
                                    epochs=50, physical_batch_size=8, virtual_batch_size=32, 
                                    callbacks=None, optimizer=None, loss_fn=None):
    """
    Train using gradient accumulation to simulate larger batch sizes with smaller memory footprint.
    
    Args:
        model: The model to train
        X_train: Training data
        y_train: Training labels
        X_val: Validation data (optional)
        y_val: Validation labels (optional)
        epochs: Number of training epochs
        physical_batch_size: Actual batch size to use (must fit in memory)
        virtual_batch_size: Effective batch size to simulate
        callbacks: List of callbacks (note: some Keras callbacks may not work with custom training loop)
        optimizer: Optimizer to use (defaults to Adam)
        loss_fn: Loss function to use (defaults to the model's loss)
        
    Returns:
        tuple: (history, trained_model)
    """
    # Set up accumulation steps
    accumulation_steps = virtual_batch_size // physical_batch_size
    
    # Default optimizer and loss function
    if optimizer is None:
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        
    if loss_fn is None:
        loss_fn = model.loss
    
    # Set up metrics
    train_loss_metric = tf.keras.metrics.Mean()
    val_metric = tf.keras.metrics.Mean(name='val_metric')
    
    # Generate batches
    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    train_dataset = train_dataset.shuffle(buffer_size=min(1024, len(X_train))).batch(physical_batch_size)
    
    if X_val is not None and y_val is not None:
        val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val)).batch(physical_batch_size)
    else:
        val_dataset = None
    
    @tf.function
    def train_step(x_batch, y_batch, first_batch):
        with tf.GradientTape() as tape:
            predictions = model(x_batch, training=True)
            loss_value = loss_fn(y_batch, predictions)
            scaled_loss = loss_value / accumulation_steps
        
        # Get gradients and scale them
        gradients = tape.gradient(scaled_loss, model.trainable_variables)
        
        # If first batch, apply directly, else accumulate
        if first_batch:
            # Reset accumulated gradients
            for i in range(len(model.trainable_variables)):
                model.accumulated_gradients[i].assign(tf.zeros_like(model.trainable_variables[i]))
            
            # Add current gradients to accumulator
            for i, gradient in enumerate(gradients):
                if gradient is not None:
                    model.accumulated_gradients[i].assign_add(gradient)
        else:
            # Add current gradients to accumulator
            for i, gradient in enumerate(gradients):
                if gradient is not None:
                    model.accumulated_gradients[i].assign_add(gradient)
        
        return loss_value
    
    print(f"Training with gradient accumulation: physical batch={physical_batch_size}, virtual batch={virtual_batch_size}")
    
    # Set up history dictionary
    history = {'loss': [], 'val_loss': []}
    if hasattr(model, 'metrics_names'):
        for metric in model.metrics_names:
            if metric != 'loss':
                history[metric] = []
                history[f'val_{metric}'] = []
    
    # Create accumulated gradients variables
    model.accumulated_gradients = [tf.Variable(tf.zeros_like(var)) for var in model.trainable_variables]
    
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        train_loss_metric.reset_states()
        if val_metric:
            val_metric.reset_states()
        
        # Training loop
        step = 0
        for x_batch, y_batch in train_dataset:
            # Every accumulation_steps, we apply the accumulated gradients
            first_batch = (step % accumulation_steps == 0)
            loss = train_step(x_batch, y_batch, first_batch)
            train_loss_metric.update_state(loss)
            
            step += 1
            if step % accumulation_steps == 0:
                # Apply accumulated gradients
                optimizer.apply_gradients(zip(model.accumulated_gradients, model.trainable_variables))
                
                if step % (accumulation_steps * 10) == 0:
                    print(f"Step {step//accumulation_steps}, Loss: {train_loss_metric.result().numpy():.4f}")
        
        # Validation
        if val_dataset:
            for x_val, y_val in val_dataset:
                val_preds = model(x_val, training=False)
                
                # Assuming the model has metrics
                if len(model.metrics) > 0:
                    for metric in model.metrics:
                        metric_result = metric(y_val, val_preds)
                        val_metric.update_state(metric_result)
        
        # Record metrics
        history['loss'].append(train_loss_metric.result().numpy())
        
        # Print epoch summary
        epoch_summary = f"Epoch {epoch+1}/{epochs} - Loss: {train_loss_metric.result():.4f}"
        if val_dataset:
            for i, metric in enumerate(model.metrics):
                metric_name = model.metrics_names[i+1] if i+1 < len(model.metrics_names) else f"metric_{i}"
                epoch_summary += f" - Val {metric_name}: {val_metric.result():.4f}"
                history[f'val_{metric_name}'].append(val_metric.result().numpy())
        
        print(epoch_summary)
        
        # Handle callbacks (simplified - some callbacks may need special handling)
        if callbacks is not None:
            for callback in callbacks:
                if hasattr(callback, 'on_epoch_end'):
                    callback.on_epoch_end(epoch, logs=history)
    
    return history, model
