"""
Unit tests for the license_plate_detection package.
Run with pytest: python -m pytest tests/
"""

import os
import sys
import unittest
import numpy as np
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from license_plate_detection.data.loader import get_data_path, preprocess_dataset
from license_plate_detection.models.detector import create_license_plate_detector
from license_plate_detection.models.losses import enhanced_iou_metric, combined_detection_loss
from license_plate_detection.evaluation.error_analysis import calculate_iou, analyze_predictions
from license_plate_detection.utils.visualization import visualize_prediction


class TestDataProcessing(unittest.TestCase):
    """Test data processing functions"""
    
    def test_preprocess_dataset(self):
        """Test the preprocessing function"""
        # Create dummy dataset
        dummy_df = {
            'image_path': ['dummy.jpg'],
            'x': [100],
            'y': [100],
            'w': [200],
            'h': [100],
            'plate_text': ['TEST1234']
        }
        import pandas as pd
        df = pd.DataFrame(dummy_df)
        
        # Create a dummy image for preprocessing
        dummy_img = np.ones((400, 600, 3), dtype=np.uint8) * 255
        
        # Mock the image reading function
        import cv2
        original_imread = cv2.imread
        
        def mock_imread(*args, **kwargs):
            return dummy_img
        
        cv2.imread = mock_imread
        
        try:
            # Process the dataset with mocked image reading
            X, y = preprocess_dataset(df, image_size=(224, 224))
            
            # Check results
            self.assertEqual(len(X), 1, "Should have 1 processed image")
            self.assertEqual(X[0].shape, (224, 224, 3), "Processed image should have the correct shape")
            self.assertEqual(len(y), 1, "Should have 1 processed bounding box")
            self.assertEqual(len(y[0]), 4, "Bounding box should have 4 values")
            
            # Check normalized bounding box
            x, y_, w, h = y[0]
            self.assertGreaterEqual(x, 0, "x should be >= 0")
            self.assertLessEqual(x + w, 1, "x + w should be <= 1")
            self.assertGreaterEqual(y_, 0, "y should be >= 0")
            self.assertLessEqual(y_ + h, 1, "y + h should be <= 1")
            
        finally:
            # Restore original imread function
            cv2.imread = original_imread


class TestModelCreation(unittest.TestCase):
    """Test model creation functions"""
    
    def test_create_license_plate_detector(self):
        """Test model creation"""
        # Create model with small input size for faster testing
        model = create_license_plate_detector(input_shape=(64, 64, 3))
        
        # Check the model structure
        self.assertIsNotNone(model, "Model should not be None")
        self.assertEqual(model.inputs[0].shape[1:], (64, 64, 3), "Model input shape should match")
        self.assertEqual(model.outputs[0].shape[1:], (4,), "Model output shape should be (4,)")
        
        # Test prediction with random input
        random_input = np.random.rand(1, 64, 64, 3).astype(np.float32)
        prediction = model.predict(random_input)
        
        # Check prediction shape and values
        self.assertEqual(prediction.shape, (1, 4), "Prediction shape should be (1, 4)")
        self.assertTrue(np.all(prediction >= 0) and np.all(prediction <= 1), 
                      "All prediction values should be in range [0, 1]")
    
    def test_create_enhanced_detector(self):
        """Test enhanced detector model creation"""
        # Import the enhanced model function
        from license_plate_detection.models.detector import create_enhanced_license_plate_detector
        
        # Create the enhanced model with small input size
        model = create_enhanced_license_plate_detector(input_shape=(64, 64, 3))
        
        # Check model structure
        self.assertIsNotNone(model, "Enhanced model should not be None")
        self.assertEqual(model.inputs[0].shape[1:], (64, 64, 3), "Model input shape should match")
        self.assertEqual(model.outputs[0].shape[1:], (4,), "Model output shape should be (4,)")
        
        # Test prediction with random input
        random_input = np.random.rand(1, 64, 64, 3).astype(np.float32)
        prediction = model.predict(random_input)
        
        # Check prediction shape and values
        self.assertEqual(prediction.shape, (1, 4), "Prediction shape should be (1, 4)")
        self.assertTrue(np.all(prediction >= 0) and np.all(prediction <= 1), 
                      "All prediction values should be in range [0, 1]")


class TestLossFunctions(unittest.TestCase):
    """Test loss functions and metrics"""
    
    def test_enhanced_iou_metric(self):
        """Test the IoU metric function"""
        import tensorflow as tf
        
        # Create ground truth and predicted boxes
        y_true = tf.constant([[0.1, 0.2, 0.3, 0.4]], dtype=tf.float32)
        y_pred = tf.constant([[0.15, 0.25, 0.3, 0.4]], dtype=tf.float32)
        
        # Calculate IoU
        iou = enhanced_iou_metric(y_true, y_pred)
        
        # Check result
        self.assertIsInstance(iou.numpy(), float, "IoU should be a float")
        self.assertGreaterEqual(iou.numpy(), 0, "IoU should be >= 0")
        self.assertLessEqual(iou.numpy(), 1, "IoU should be <= 1")
    
    def test_combined_detection_loss(self):
        """Test the combined detection loss function"""
        import tensorflow as tf
        
        # Create ground truth and predicted boxes
        y_true = tf.constant([[0.1, 0.2, 0.3, 0.4]], dtype=tf.float32)
        y_pred = tf.constant([[0.15, 0.25, 0.3, 0.4]], dtype=tf.float32)
        
        # Calculate loss
        loss = combined_detection_loss(y_true, y_pred)
        
        # Check result
        self.assertIsInstance(loss.numpy(), float, "Loss should be a float")
        self.assertGreaterEqual(loss.numpy(), 0, "Loss should be >= 0")


class TestErrorAnalysis(unittest.TestCase):
    """Test error analysis functions"""
    
    def test_iou_calculation(self):
        """Test IoU calculation function"""
        # Create ground truth and predicted boxes
        box1 = [0.1, 0.2, 0.3, 0.4]  # [x, y, width, height]
        box2 = [0.15, 0.25, 0.3, 0.4]
        
        # Calculate IoU
        iou = calculate_iou(box1, box2)
        
        # Check result
        self.assertIsInstance(iou, float, "IoU should be a float")
        self.assertGreaterEqual(iou, 0, "IoU should be >= 0")
        self.assertLessEqual(iou, 1, "IoU should be <= 1")
    
    def test_analyze_predictions(self):
        """Test prediction analysis function"""
        # Create ground truth and predicted boxes
        y_true = np.array([
            [0.1, 0.2, 0.3, 0.4],
            [0.5, 0.5, 0.2, 0.3]
        ])
        y_pred = np.array([
            [0.15, 0.25, 0.3, 0.4],
            [0.55, 0.45, 0.25, 0.3]
        ])
        
        # Analyze predictions
        metrics = analyze_predictions(y_true, y_pred)
        
        # Check results
        self.assertIsInstance(metrics, dict, "Result should be a dictionary")
        self.assertIn('mean_iou', metrics, "Result should include mean IoU")
        self.assertGreaterEqual(metrics['mean_iou'], 0, "Mean IoU should be >= 0")
        self.assertLessEqual(metrics['mean_iou'], 1, "Mean IoU should be <= 1")
        self.assertIn('accuracy', metrics, "Result should include accuracy")


class TestVisualization(unittest.TestCase):
    """Test visualization functions"""
    
    def test_visualize_prediction(self):
        """Test prediction visualization function"""
        import matplotlib.pyplot as plt
        
        # Create test image and boxes
        image = np.ones((224, 224, 3))
        true_box = [0.1, 0.2, 0.3, 0.4]
        pred_box = [0.15, 0.25, 0.3, 0.4]
        
        # Visualize prediction
        fig = visualize_prediction(image, true_box, pred_box)
        
        # Check result
        self.assertIsInstance(fig, plt.Figure, "Result should be a matplotlib Figure")
        
        # Close figure to avoid warnings
        plt.close(fig)
        
        # Create some test data
        y_true = tf.constant([
            [0.1, 0.1, 0.2, 0.2],  # x, y, w, h
            [0.5, 0.5, 0.3, 0.3]
        ])
        
        # Perfect prediction
        y_pred_perfect = tf.constant([
            [0.1, 0.1, 0.2, 0.2],
            [0.5, 0.5, 0.3, 0.3]
        ])
        
        # Imperfect prediction
        y_pred_imperfect = tf.constant([
            [0.15, 0.15, 0.2, 0.2],  # Shifted
            [0.5, 0.5, 0.2, 0.2]    # Smaller
        ])
        
        # Calculate IoU
        iou_perfect = enhanced_iou_metric(y_true, y_pred_perfect)
        iou_imperfect = enhanced_iou_metric(y_true, y_pred_imperfect)
        
        # Check results
        self.assertAlmostEqual(float(iou_perfect), 1.0, places=5, 
                             msg="Perfect prediction should give IoU close to 1.0")
        self.assertLess(float(iou_imperfect), 1.0,
                      msg="Imperfect prediction should give IoU less than 1.0")
        self.assertGreater(float(iou_imperfect), 0.0,
                         msg="Imperfect prediction should give IoU greater than 0.0")
    
    def test_combined_detection_loss(self):
        """Test the combined detection loss function"""
        import tensorflow as tf
        
        # Create some test data
        y_true = tf.constant([
            [0.1, 0.1, 0.2, 0.2],  # x, y, w, h
            [0.5, 0.5, 0.3, 0.3]
        ])
        
        # Perfect prediction
        y_pred_perfect = tf.constant([
            [0.1, 0.1, 0.2, 0.2],
            [0.5, 0.5, 0.3, 0.3]
        ])
        
        # Imperfect prediction
        y_pred_imperfect = tf.constant([
            [0.15, 0.15, 0.2, 0.2],  # Shifted
            [0.5, 0.5, 0.2, 0.2]    # Smaller
        ])
        
        # Calculate losses
        loss_perfect = combined_detection_loss(y_true, y_pred_perfect)
        loss_imperfect = combined_detection_loss(y_true, y_pred_imperfect)
        
        # Check results
        self.assertGreaterEqual(float(loss_perfect), 0.0,
                              msg="Loss should be non-negative")
        self.assertGreaterEqual(float(loss_imperfect), float(loss_perfect),
                              msg="Loss for imperfect prediction should be greater than perfect prediction")


if __name__ == '__main__':
    unittest.main()
