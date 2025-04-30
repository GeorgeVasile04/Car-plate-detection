"""
Command-line interface for the license plate detection package.
"""

import argparse
import sys
from pathlib import Path

from license_plate_detection.main import train, evaluate, predict


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
