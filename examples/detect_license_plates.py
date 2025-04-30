"""
Example script demonstrating the complete license plate detection pipeline.

This script:
1. Loads and preprocesses license plate data
2. Creates and trains a model
3. Makes predictions and visualizes results
4. Performs error analysis

Usage: python examples/detect_license_plates.py
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Add the project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import modules from the package
from license_plate_detection.data.loader import get_data_path, load_dataset, preprocess_dataset, split_dataset
from license_plate_detection.models.detector import create_enhanced_license_plate_detector
from license_plate_detection.models.losses import combined_detection_loss, enhanced_iou_metric
from license_plate_detection.train.trainer import train_model, create_training_callbacks, save_model
from license_plate_detection.utils.visualization import visualize_batch_predictions, plot_training_history
from license_plate_detection.evaluation.error_analysis import analyze_predictions, analyze_error_distribution

# Set random seeds for reproducibility
np.random.seed(42)
import tensorflow as tf
tf.random.set_seed(42)


def main():
    """Main function for the license plate detection demo."""
    print("License Plate Detection Demo")
    print("============================")
    
    # Step 1: Load Dataset
    print("\nStep 1: Loading Dataset")
    try:
        data_path = get_data_path()
        print(f"Dataset found at: {data_path}")
        
        # Load annotations and images
        images_path = data_path / "images"
        annotations_path = data_path / "annotations"
        
        df = load_dataset(annotations_path, images_path)
        print(f"Loaded {len(df)} annotations with corresponding images")
        
        # Select a subset for quick demo
        subset_size = 100  # Limit to speed up the demo
        if len(df) > subset_size:
            df = df.sample(subset_size, random_state=42)
            print(f"Using {len(df)} random samples for this demo")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("Using synthetic data for demonstration...")
        
        # Create synthetic data
        num_samples = 50
        synthetic_X = np.random.rand(num_samples, 224, 224, 3).astype(np.float32)
        synthetic_y = np.random.rand(num_samples, 4).astype(np.float32) * 0.5 + 0.25
        
        print(f"Created synthetic dataset with {num_samples} samples")
        
        # Skip to Step 3 with synthetic data
        X, y = synthetic_X, synthetic_y
        proceed_to_step_3 = True
    else:
        proceed_to_step_3 = False
    
    # Step 2: Preprocess Data (skip if using synthetic data)
    if not proceed_to_step_3:
        print("\nStep 2: Preprocessing Data")
        # Preprocess data
        X, y = preprocess_dataset(df, image_size=(224, 224))
        print(f"Preprocessed {len(X)} images with corresponding bounding boxes")
        
        # Display sample information
        print("Sample images shape:", X.shape)
        print("Sample bounding boxes shape:", y.shape)
    
    # Step 3: Split data into training and validation sets
    print("\nStep 3: Splitting Data")
    X_train, X_val, y_train, y_val = split_dataset(X, y, test_size=0.2, random_state=42)
    print(f"Training set: {len(X_train)} samples")
    print(f"Validation set: {len(X_val)} samples")
    
    # Step 4: Create Model
    print("\nStep 4: Creating Model")
    model = create_enhanced_license_plate_detector(input_shape=(224, 224, 3))
    model.summary()
    
    # Step 5: Configure Training
    print("\nStep 5: Configuring Training")
    # Create callbacks
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    
    checkpoint_path = output_dir / "license_plate_detector_weights.h5"
    tensorboard_log_dir = output_dir / "logs"
    
    callbacks = create_training_callbacks(
        checkpoint_path=checkpoint_path,
        early_stopping=True,
        patience=5,
        reduce_lr=True,
        tensorboard_log_dir=tensorboard_log_dir
    )
    
    # Compile model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss=combined_detection_loss,
        metrics=[enhanced_iou_metric]
    )
    print("Model compiled with Adam optimizer and combined detection loss")
    
    # Step 6: Train Model
    print("\nStep 6: Training Model")
    history, trained_model = train_model(
        model=model,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        epochs=10,  # Low number of epochs for demo
        batch_size=16,
        callbacks=callbacks
    )
    
    # Plot training history
    history_fig = plot_training_history(history)
    history_fig.savefig(output_dir / "training_history.png")
    print(f"Training history saved to {output_dir / 'training_history.png'}")
    
    # Step 7: Evaluate Model
    print("\nStep 7: Evaluating Model")
    # Make predictions on validation set
    y_pred = model.predict(X_val)
    
    # Analyze predictions
    metrics = analyze_predictions(y_val, y_pred)
    
    print(f"Mean IoU: {metrics['mean_iou']:.4f}")
    print(f"Detection Accuracy (IoU >= 0.5): {metrics['accuracy']:.4f}")
    print(f"Mean Position Error: {metrics['mean_position_error']:.4f}")
    print(f"Mean Size Error: {metrics['mean_size_error']:.4f}")
    
    # Analyze error distribution
    error_fig, error_stats = analyze_error_distribution(y_val, y_pred)
    error_fig.savefig(output_dir / "error_distribution.png")
    print(f"Error distribution saved to {output_dir / 'error_distribution.png'}")
    
    # Step 8: Visualize Results
    print("\nStep 8: Visualizing Results")
    # Visualize batch predictions
    vis_figs = visualize_batch_predictions(
        X_val, y_val, y_pred, 
        indices=range(5),  # Show first 5 samples
        max_images=5,
        save_dir=output_dir / "visualizations"
    )
    print(f"Visualizations saved to {output_dir / 'visualizations'}")
    
    # Step 9: Save Model
    print("\nStep 9: Saving Model")
    model_path = save_model(model, output_dir / "license_plate_detector.h5")
    print(f"Model saved to {model_path}")
    
    print("\nDemo completed successfully!")
    

if __name__ == "__main__":
    main()
