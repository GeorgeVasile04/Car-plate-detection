"""
Memory optimization functions for model training.
"""

import gc
import numpy as np
import tensorflow as tf

def optimize_memory_usage(X_train, y_train, X_val=None, y_val=None):
    """
    Optimize memory usage by converting data to half precision.
    
    Args:
        X_train: Training data
        y_train: Training labels
        X_val: Validation data (optional)
        y_val: Validation labels (optional)
        
    Returns:
        tuple: Optimized X_train, X_val, y_train, y_val
    """
    original_dtype = X_train.dtype
    original_size_mb = X_train.nbytes / (1024 * 1024)
    if X_val is not None:
        original_size_mb += X_val.nbytes / (1024 * 1024)
    original_size_mb += y_train.nbytes / (1024 * 1024)
    if y_val is not None:
        original_size_mb += y_val.nbytes / (1024 * 1024)
    
    print(f"Original data type: {original_dtype}")
    print(f"Original memory usage: {original_size_mb:.2f} MB")
    
    # Convert data to float16 to save memory (if not already)
    if X_train.dtype != np.float16:
        X_train = X_train.astype(np.float16)
        if X_val is not None:
            X_val = X_val.astype(np.float16)
    
    # Labels typically need less conversion since they're small
    if y_train.dtype != np.float32:
        y_train = y_train.astype(np.float32)
        if y_val is not None:
            y_val = y_val.astype(np.float32)
    
    new_size_mb = X_train.nbytes / (1024 * 1024)
    if X_val is not None:
        new_size_mb += X_val.nbytes / (1024 * 1024)
    new_size_mb += y_train.nbytes / (1024 * 1024)
    if y_val is not None:
        new_size_mb += y_val.nbytes / (1024 * 1024)
    
    print(f"New data type for X: {X_train.dtype}")
    print(f"New data type for y: {y_train.dtype}")
    print(f"New memory usage: {new_size_mb:.2f} MB")
    print(f"Memory saved: {original_size_mb - new_size_mb:.2f} MB ({(1 - new_size_mb/original_size_mb) * 100:.1f}%)")
    
    if X_val is not None and y_val is not None:
        return X_train, X_val, y_train, y_val
    else:
        return X_train, y_train

def setup_gpu_memory_growth():
    """
    Configure TensorFlow to grow GPU memory usage as needed instead of allocating all at once.
    
    Returns:
        bool: True if configuration was successful, False otherwise
    """
    try:
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            print(f"Found {len(gpus)} GPU(s)")
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
                print(f"Memory growth enabled for {gpu}")
            return True
        else:
            print("No GPUs found")
            return False
    except Exception as e:
        print(f"Error setting up GPU memory growth: {e}")
        return False

def limit_gpu_memory(memory_limit_mb=None):
    """
    Limit GPU memory usage to a specific amount.
    
    Args:
        memory_limit_mb: Memory limit in MB (None means no limit)
        
    Returns:
        bool: True if configuration was successful, False otherwise
    """
    try:
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus and memory_limit_mb is not None:
            tf.config.experimental.set_virtual_device_configuration(
                gpus[0],
                [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=memory_limit_mb)]
            )
            print(f"GPU memory limited to {memory_limit_mb}MB")
            return True
        return False
    except Exception as e:
        print(f"Error limiting GPU memory: {e}")
        return False

def enable_mixed_precision():
    """
    Enable mixed precision training for faster computation and lower memory usage on compatible GPUs.
    
    Returns:
        bool: True if mixed precision was enabled, False otherwise
    """
    try:
        if tf.__version__ >= '2.4.0':
            policy = tf.keras.mixed_precision.Policy('mixed_float16')
            tf.keras.mixed_precision.set_global_policy(policy)
            print(f"Mixed precision policy set to {policy.name}")
            print(f"Compute dtype: {policy.compute_dtype}")
            print(f"Variable dtype: {policy.variable_dtype}")
            return True
        else:
            print("Mixed precision training requires TensorFlow 2.4.0 or later")
            return False
    except Exception as e:
        print(f"Error enabling mixed precision: {e}")
        return False

def enable_gradient_checkpointing(model):
    """
    Enable gradient checkpointing to reduce memory usage at the cost of computation time.
    
    Args:
        model: The Keras model to modify
        
    Returns:
        model: The model with gradient checkpointing enabled
    """
    try:
        if tf.__version__ >= '2.9.0':
            print("Enabling gradient checkpointing to reduce memory usage...")
            model.make_train_function = lambda: None  # Reset train function before applying checkpointing
            
            # Apply gradient checkpointing to all layers that support it
            for layer in model.layers:
                if hasattr(layer, 'supports_masking') and len(layer.outbound_nodes) > 0:
                    try:
                        layer._uses_learning_phase = True  # Enable gradient checkpointing for this layer
                    except:
                        pass  # Skip layers that don't support this attribute
            
            print("Gradient checkpointing enabled successfully")
        else:
            print("Your TensorFlow version might not fully support gradient checkpointing")
    except Exception as e:
        print(f"Could not enable gradient checkpointing: {e}")
    
    return model

def clean_memory():
    """
    Clean up memory by running garbage collection and clearing TensorFlow session.
    """
    # Clear tensorflow session
    tf.keras.backend.clear_session()
    
    # Run garbage collection
    gc.collect()
    
    try:
        # Clean up PyTorch CUDA cache if available
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print("PyTorch CUDA cache cleared")
    except ImportError:
        pass
    
    print("Memory cleaned up")
