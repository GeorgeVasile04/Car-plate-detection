"""
Main module for the license plate detection package.
This serves as the entry point for the package.
"""

import os
import sys
import numpy as np
import argparse
from pathlib import Path
import tensorflow as tf
import matplotlib.pyplot as plt

# Import modules from the package
from license_plate_detection.data.loader import get_data_path, load_dataset, preprocess_dataset, split_dataset
from license_plate_detection.models.detector import (
    create_license_plate_detector, 
    create_enhanced_license_plate_detector,
    create_mobilenet_license_plate_detector, 
    create_efficientnet_license_plate_detector
)
from license_plate_detection.models.losses import enhanced_iou_metric, combined_detection_loss, giou_loss
from license_plate_detection.train.trainer import train_model, create_training_callbacks, save_model
from license_plate_detection.utils.visualization import (
    visualize_prediction, 
    visualize_batch_predictions, 
    plot_training_history,
    create_prediction_video
)
from license_plate_detection.evaluation.error_analysis import (
    analyze_predictions, 
    identify_error_patterns,
    visualize_error_patterns
)


def load_and_prepare_data(annotations_path=None, images_path=None, subset_size=None, image_size=(224, 224)):
    """
    Load and prepare data for training or evaluation.
    
    Args:
        annotations_path: Path to annotations directory
        images_path: Path to images directory
        subset_size: Size of random subset to use (if None, uses all data)
        image_size: Target image size for preprocessing
    
    Returns:
        tuple: (X, y, images_paths) or None if data loading fails
    """
    try:
        # If paths not provided, get from default location
        if annotations_path is None or images_path is None:
            data_path = get_data_path()
            annotations_path = data_path / "annotations"
            images_path = data_path / "images"
        
        # Load dataset
        df = load_dataset(annotations_path, images_path)
        print(f"Loaded {len(df)} annotations with corresponding images")
        
        # Take a subset if requested
        if subset_size is not None and subset_size < len(df):
            df = df.sample(subset_size, random_state=42)
            print(f"Using {len(df)} random samples")
        
        # Save image paths for later reference
        image_paths = df['image_path'].tolist()
        
        # Preprocess data
        X, y = preprocess_dataset(df, image_size=image_size)
        print(f"Preprocessed {len(X)} images with corresponding bounding boxes")
        
        return X, y, image_paths
    
    except Exception as e:
        print(f"Error loading data: {e}")
        return None


def create_model_by_name(model_name, input_shape=(224, 224, 3)):
    """
    Create a model based on the given name.
    
    Args:
        model_name: Name of the model to create
        input_shape: Input shape for the model
    
    Returns:
        model: Created model or None if model name is not recognized
    """
    if model_name == 'basic':
        return create_license_plate_detector(input_shape=input_shape)
    elif model_name == 'enhanced':
        return create_enhanced_license_plate_detector(input_shape=input_shape)
    elif model_name == 'mobilenet':
        return create_mobilenet_license_plate_detector(input_shape=input_shape)
    elif model_name.startswith('efficientnet'):
        # Extract version from name, e.g., 'efficientnet_b0' -> 'B0'
        try:
            version = model_name.split('_')[1].upper()
            return create_efficientnet_license_plate_detector(input_shape=input_shape, version=version)
        except (IndexError, ValueError):
            print(f"Invalid EfficientNet version in {model_name}, using B0")
            return create_efficientnet_license_plate_detector(input_shape=input_shape)
    else:
        print(f"Unknown model name: {model_name}. Using enhanced model instead.")
        return create_enhanced_license_plate_detector(input_shape=input_shape)


def train(args):
    """
    Train a license plate detection model.
    
    Args:
        args: Command line arguments
    """
    print("Training license plate detection model")
    
    # Load and prepare data
    data = load_and_prepare_data(
        annotations_path=args.annotations_path,
        images_path=args.images_path,
        subset_size=args.subset_size,
        image_size=(args.image_size, args.image_size)
    )
    
    if data is None:
        print("Failed to load data. Exiting.")
        return
    
    X, y, _ = data
    
    # Split data
    X_train, X_val, y_train, y_val = split_dataset(
        X, y, test_size=args.validation_split, random_state=42
    )
    print(f"Training set: {len(X_train)} samples")
    print(f"Validation set: {len(X_val)} samples")
    
    # Create model
    model = create_model_by_name(
        args.model, input_shape=(args.image_size, args.image_size, 3)
    )
    model.summary()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Create callbacks
    checkpoint_path = output_dir / f"{args.model}_weights.h5"
    tensorboard_log_dir = output_dir / "logs"
    
    callbacks = create_training_callbacks(
        checkpoint_path=checkpoint_path,
        early_stopping=True,
        patience=args.patience,
        reduce_lr=True,
        tensorboard_log_dir=tensorboard_log_dir if args.use_tensorboard else None
    )
    
    # Select loss function
    if args.loss == 'combined':
        loss_function = combined_detection_loss
    elif args.loss == 'giou':
        loss_function = giou_loss
    else:
        loss_function = 'mse'  # Default TensorFlow loss
    
    # Compile model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=args.learning_rate),
        loss=loss_function,
        metrics=[enhanced_iou_metric]
    )
    
    # Train model
    history, trained_model = train_model(
        model=model,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=callbacks
    )
    
    # Plot training history
    history_fig = plot_training_history(history)
    history_fig.savefig(output_dir / "training_history.png")
    
    # Evaluate on validation set
    y_pred = model.predict(X_val)
    metrics = analyze_predictions(y_val, y_pred)
    
    print(f"Validation Mean IoU: {metrics['mean_iou']:.4f}")
    print(f"Validation Accuracy (IoU >= 0.5): {metrics['accuracy']:.4f}")
    
    # Save model
    model_path = save_model(model, output_dir / f"{args.model}.h5")
    print(f"Model saved to {model_path}")
    
    # Visualize some predictions
    vis_dir = output_dir / "visualizations"
    vis_dir.mkdir(exist_ok=True)
    
    visualize_batch_predictions(
        X_val, y_val, y_pred,
        indices=np.random.choice(len(X_val), min(5, len(X_val)), replace=False),
        save_dir=vis_dir
    )
    
    print(f"Training completed. Results saved to {output_dir}")


def evaluate(args):
    """
    Evaluate a license plate detection model.
    
    Args:
        args: Command line arguments
    """
    print("Evaluating license plate detection model")
    
    # Load model
    try:
        model = tf.keras.models.load_model(
            args.model_path,
            custom_objects={
                'enhanced_iou_metric': enhanced_iou_metric,
                'combined_detection_loss': combined_detection_loss,
                'giou_loss': giou_loss
            }
        )
        print(f"Model loaded from {args.model_path}")
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Load data for evaluation
    data = load_and_prepare_data(
        annotations_path=args.annotations_path,
        images_path=args.images_path,
        subset_size=args.subset_size,
        image_size=(args.image_size, args.image_size)
    )
    
    if data is None:
        print("Failed to load data. Exiting.")
        return
    
    X, y, image_paths = data
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Make predictions
    y_pred = model.predict(X)
    
    # Analyze predictions
    metrics = analyze_predictions(y, y_pred)
    
    # Print metrics
    print("\nEvaluation Metrics:")
    print(f"Mean IoU: {metrics['mean_iou']:.4f}")
    print(f"Accuracy (IoU >= 0.5): {metrics['accuracy']:.4f}")
    print(f"Mean Position Error: {metrics['mean_position_error']:.4f}")
    print(f"Mean Size Error: {metrics['mean_size_error']:.4f}")
    
    # Save metrics to file
    with open(output_dir / "evaluation_metrics.txt", 'w') as f:
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                f.write(f"{key}: {value}\n")
            elif isinstance(value, dict):
                f.write(f"{key}:\n")
                for subkey, subvalue in value.items():
                    f.write(f"  {subkey}: {subvalue}\n")
    
    # Identify error patterns
    patterns = identify_error_patterns(y, y_pred, images=X)
    
    # Visualize error patterns
    vis_dir = output_dir / "error_analysis"
    vis_dir.mkdir(exist_ok=True)
    
    error_figs = visualize_error_patterns(patterns, save_dir=vis_dir)
    
    # Visualize best and worst predictions
    best_worst_dir = vis_dir / "best_worst"
    best_worst_dir.mkdir(exist_ok=True)
    
    # Worst predictions
    worst_indices = [pred['index'] for pred in patterns['worst_predictions'][:5]]
    worst_figs = visualize_batch_predictions(
        X, y, y_pred,
        indices=worst_indices,
        save_dir=best_worst_dir / "worst"
    )
    
    # Best predictions
    best_indices = [pred['index'] for pred in patterns['best_predictions'][-5:]]
    best_figs = visualize_batch_predictions(
        X, y, y_pred,
        indices=best_indices,
        save_dir=best_worst_dir / "best"
    )
    
    print(f"Evaluation completed. Results saved to {output_dir}")


def predict(args):
    """
    Make predictions with a license plate detection model.
    
    Args:
        args: Command line arguments
    """
    print("Making predictions with license plate detection model")
    
    # Load model
    try:
        model = tf.keras.models.load_model(
            args.model_path,
            custom_objects={
                'enhanced_iou_metric': enhanced_iou_metric,
                'combined_detection_loss': combined_detection_loss,
                'giou_loss': giou_loss
            }
        )
        print(f"Model loaded from {args.model_path}")
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Check if input is a video
    if args.video_path:
        # Process video
        print(f"Processing video: {args.video_path}")
        output_video = create_prediction_video(
            args.video_path,
            model,
            output_path=args.output_path,
            resize_shape=(args.image_size, args.image_size),
            show_preview=True
        )
        print(f"Processed video saved to: {output_video}")
        return
    
    # Load and process a single image
    if args.image_path:
        import cv2
        
        try:
            # Load image
            img = cv2.imread(args.image_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Resize for model input
            orig_h, orig_w = img.shape[:2]
            img_resized = cv2.resize(img, (args.image_size, args.image_size))
            
            # Normalize
            img_normalized = img_resized / 255.0
            
            # Add batch dimension
            img_batch = np.expand_dims(img_normalized, axis=0)
            
            # Make prediction
            pred_box = model.predict(img_batch)[0]
            
            # Visualize prediction
            fig = visualize_prediction(img_resized, pred_box=pred_box, denormalize=True)
            
            # Save or show result
            if args.output_path:
                plt.savefig(args.output_path, bbox_inches='tight', dpi=150)
                print(f"Prediction saved to: {args.output_path}")
            else:
                plt.show()
            
            # Print predicted coordinates (denormalized to original image)
            x, y, w, h = pred_box
            x_orig = int(x * orig_w)
            y_orig = int(y * orig_h)
            w_orig = int(w * orig_w)
            h_orig = int(h * orig_h)
            
            print(f"Predicted license plate coordinates (in original image):")
            print(f"  Top-left: ({x_orig}, {y_orig})")
            print(f"  Width x Height: {w_orig} x {h_orig}")
            
        except Exception as e:
            print(f"Error processing image: {e}")
            return
    
    else:
        print("No input specified. Please provide --image_path or --video_path.")


def main():
    """Main function for the license plate detection package."""
    parser = argparse.ArgumentParser(description='License Plate Detection')
    subparsers = parser.add_subparsers(dest='command', help='Command to execute')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train license plate detection model')
    train_parser.add_argument('--annotations_path', type=str, help='Path to annotations directory')
    train_parser.add_argument('--images_path', type=str, help='Path to images directory')
    train_parser.add_argument('--subset_size', type=int, default=None, help='Size of subset to use')
    train_parser.add_argument('--image_size', type=int, default=224, help='Image size for training')
    train_parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training')
    train_parser.add_argument('--epochs', type=int, default=50, help='Number of epochs for training')
    train_parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    train_parser.add_argument('--validation_split', type=float, default=0.2, help='Validation split ratio')
    train_parser.add_argument('--model', type=str, default='enhanced', 
                             choices=['basic', 'enhanced', 'mobilenet', 'efficientnet_b0'], 
                             help='Model architecture to use')
    train_parser.add_argument('--loss', type=str, default='combined', 
                             choices=['mse', 'combined', 'giou'], 
                             help='Loss function to use')
    train_parser.add_argument('--patience', type=int, default=10, help='Patience for early stopping')
    train_parser.add_argument('--output_dir', type=str, default='output', help='Output directory')
    train_parser.add_argument('--use_tensorboard', action='store_true', help='Use TensorBoard for logging')
    
    # Evaluate command
    eval_parser = subparsers.add_parser('evaluate', help='Evaluate license plate detection model')
    eval_parser.add_argument('--model_path', type=str, required=True, help='Path to model file')
    eval_parser.add_argument('--annotations_path', type=str, help='Path to annotations directory')
    eval_parser.add_argument('--images_path', type=str, help='Path to images directory')
    eval_parser.add_argument('--subset_size', type=int, default=None, help='Size of subset to use')
    eval_parser.add_argument('--image_size', type=int, default=224, help='Image size for evaluation')
    eval_parser.add_argument('--output_dir', type=str, default='evaluation_output', help='Output directory')
    
    # Predict command
    predict_parser = subparsers.add_parser('predict', help='Make predictions with license plate detection model')
    predict_parser.add_argument('--model_path', type=str, required=True, help='Path to model file')
    predict_parser.add_argument('--image_path', type=str, help='Path to input image')
    predict_parser.add_argument('--video_path', type=str, help='Path to input video')
    predict_parser.add_argument('--output_path', type=str, help='Path to save output')
    predict_parser.add_argument('--image_size', type=int, default=224, help='Image size for prediction')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Execute command
    if args.command == 'train':
        train(args)
    elif args.command == 'evaluate':
        evaluate(args)
    elif args.command == 'predict':
        predict(args)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
