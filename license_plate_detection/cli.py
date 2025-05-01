"""
Command-line interface for the license plate detection package.
"""

import argparse
import sys
from pathlib import Path
import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Replace imports from main.py with our helpers
from license_plate_detection.utils.helpers import (
    detect_license_plate, 
    load_and_prepare_model,
    load_and_prepare_data
)

from license_plate_detection.data.loader import get_data_path, load_license_plate_dataset
from license_plate_detection.models.losses import enhanced_iou_metric, combined_detection_loss, giou_loss
from license_plate_detection.train.trainer import train_model, create_training_callbacks, save_model
from license_plate_detection.utils.visualization import visualize_prediction, visualize_batch_predictions, plot_training_history
from license_plate_detection.evaluation.error_analysis import analyze_predictions, identify_error_patterns, visualize_error_patterns
from license_plate_detection.models.detector import (
    create_license_plate_detector, 
    create_enhanced_license_plate_detector,
    create_mobilenet_license_plate_detector, 
    create_efficientnet_license_plate_detector
)


def create_model_by_name(model_name, input_shape=(224, 224, 3)):
    """Create a model based on the given name."""
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
    """Train a license plate detection model."""
    print("Training license plate detection model")
    
    # Get paths to dataset
    data_path = get_data_path()
    annotations_path = args.annotations_path or data_path / "annotations"
    images_path = args.images_path or data_path / "images"
    
    # Load dataset
    df = load_license_plate_dataset(annotations_path, images_path)
    
    if args.subset_size and args.subset_size < len(df):
        df = df.sample(args.subset_size, random_state=42)
    
    print(f"Using {len(df)} samples for training")
    
    # Preprocess dataset
    from license_plate_detection.data.loader import preprocess_license_plate_dataset, split_dataset
    X, y = preprocess_license_plate_dataset(df, image_size=(args.image_size, args.image_size))
    
    # Split data
    X_train, X_val, y_train, y_val = split_dataset(
        X, y, test_size=args.validation_split, random_state=42
    )
    
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
    """Evaluate a license plate detection model."""
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
    
    # Get paths to dataset
    data_path = get_data_path()
    annotations_path = args.annotations_path or data_path / "annotations"
    images_path = args.images_path or data_path / "images"
    
    # Load dataset
    df = load_license_plate_dataset(annotations_path, images_path)
    
    if args.subset_size and args.subset_size < len(df):
        df = df.sample(args.subset_size, random_state=42)
    
    # Preprocess dataset
    from license_plate_detection.data.loader import preprocess_license_plate_dataset
    X, y = preprocess_license_plate_dataset(df, image_size=(args.image_size, args.image_size))
    
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
    """Make predictions with a license plate detection model."""
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
        # Video processing functionality is disabled
        print("Video processing is not implemented in this version.")
        print("Please use --image_path instead of --video_path.")
        return
    
    # Load and process a single image
    if args.image_path:
        try:
            # Use detect_license_plate function
            plate_region, bbox = detect_license_plate(
                args.image_path, 
                model=model,
                image_size=args.image_size
            )
            
            # Save output if requested
            if args.output_path:
                plt.imsave(args.output_path, plate_region)
                print(f"Prediction saved to: {args.output_path}")
            
            # Print predicted coordinates
            x, y, w, h = bbox
            print(f"Predicted license plate coordinates:")
            print(f"  Top-left: ({x}, {y})")
            print(f"  Width x Height: {w} x {h}")
            
        except Exception as e:
            print(f"Error processing image: {e}")
    else:
        print("No input specified. Please provide --image_path.")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(description='License Plate Detection CLI')
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
        return 1
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
