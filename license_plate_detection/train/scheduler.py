"""
Learning rate scheduling for license plate detection model training.
"""

import tensorflow as tf
import numpy as np


def cosine_decay_scheduler(initial_learning_rate, epochs, warmup_epochs=0):
    """
    Creates a cosine decay learning rate scheduler with optional warmup.
    
    Args:
        initial_learning_rate: Initial learning rate
        epochs: Total number of epochs
        warmup_epochs: Number of warmup epochs (linear increase in learning rate)
        
    Returns:
        Learning rate scheduler function
    """
    def scheduler(epoch, lr):
        if epoch < warmup_epochs:
            # Linear warmup
            return initial_learning_rate * (epoch + 1) / warmup_epochs
        else:
            # Cosine decay
            progress = (epoch - warmup_epochs) / (epochs - warmup_epochs)
            cosine_decay = 0.5 * (1 + np.cos(np.pi * progress))
            return initial_learning_rate * cosine_decay
    
    return scheduler


def step_decay_scheduler(initial_learning_rate, drop_factor=0.5, epochs_drop=10):
    """
    Creates a step decay learning rate scheduler.
    
    Args:
        initial_learning_rate: Initial learning rate
        drop_factor: Factor by which to drop learning rate
        epochs_drop: Number of epochs after which to drop learning rate
        
    Returns:
        Learning rate scheduler function
    """
    def scheduler(epoch, lr):
        return initial_learning_rate * (drop_factor ** (epoch // epochs_drop))
    
    return scheduler


def create_lr_scheduler(scheduler_type, initial_learning_rate, **kwargs):
    """
    Creates a learning rate scheduler based on the specified type.
    
    Args:
        scheduler_type: Type of scheduler ('cosine', 'step', or 'exponential')
        initial_learning_rate: Initial learning rate
        **kwargs: Additional arguments for the specific scheduler
        
    Returns:
        tf.keras.callbacks.LearningRateScheduler or None if type is not recognized
    """
    if scheduler_type == 'cosine':
        epochs = kwargs.get('epochs', 100)
        warmup_epochs = kwargs.get('warmup_epochs', 0)
        scheduler_fn = cosine_decay_scheduler(initial_learning_rate, epochs, warmup_epochs)
        return tf.keras.callbacks.LearningRateScheduler(scheduler_fn)
    
    elif scheduler_type == 'step':
        drop_factor = kwargs.get('drop_factor', 0.5)
        epochs_drop = kwargs.get('epochs_drop', 10)
        scheduler_fn = step_decay_scheduler(initial_learning_rate, drop_factor, epochs_drop)
        return tf.keras.callbacks.LearningRateScheduler(scheduler_fn)
    
    elif scheduler_type == 'exponential':
        decay_rate = kwargs.get('decay_rate', 0.9)
        return tf.keras.callbacks.ExponentialDecay(
            initial_learning_rate=initial_learning_rate,
            decay_steps=kwargs.get('decay_steps', 1000),
            decay_rate=decay_rate,
            staircase=kwargs.get('staircase', False)
        )
    
    else:
        return None
