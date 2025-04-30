"""
Integration tests for the license plate detection pipeline.
Run with pytest: python -m pytest tests/test_integration.py -v
"""

import os
import sys
import unittest
import numpy as np
import pandas as pd
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from license_plate_detection.data.loader import get_data_path, load_dataset, preprocess_dataset, split_dataset
from license_plate_detection.models.detector import create_license_plate_detector
from license_plate_detection.models.losses import enhanced_iou_metric, combined_detection_loss
from license_plate_detection.utils.visualization import visualize_prediction
from license_plate_detection.evaluation.error_analysis import analyze_predictions


class TestIntegration(unittest.TestCase):
    """Test the complete license plate detection pipeline"""

    def setUp(self):
        """Set up the test by creating a mock dataset"""
        # Create a temporary directory for test data
        self.temp_dir = Path('temp_test_data')
        self.images_dir = self.temp_dir / 'images'
        self.annotations_dir = self.temp_dir / 'annotations'
        
        # Create directories
        self.temp_dir.mkdir(exist_ok=True)
        self.images_dir.mkdir(exist_ok=True)
        self.annotations_dir.mkdir(exist_ok=True)
        
        # Create dummy images and annotations
        self.create_mock_dataset(num_samples=3)
        
    def tearDown(self):
        """Clean up temporary test data"""
        # Remove all files in temp directories
        for path in self.images_dir.glob('*'):
            path.unlink()
        for path in self.annotations_dir.glob('*'):
            path.unlink()
            
        # Remove directories
        self.images_dir.rmdir()
        self.annotations_dir.rmdir()
        self.temp_dir.rmdir()
        self.annotations_dir.rmdir()
        self.temp_dir.rmdir()
        
    def create_mock_dataset(self, num_samples=3):
        """Create a small mock dataset for testing"""
        import cv2
        
        for i in range(num_samples):
            # Create a random image
            img = np.random.randint(0, 255, (300, 400, 3), dtype=np.uint8)
            img_path = self.images_dir / f'test_img_{i}.jpg'
            cv2.imwrite(str(img_path), img)
            
            # Create a corresponding XML annotation
            x, y = np.random.randint(50, 150), np.random.randint(50, 150)
            w, h = np.random.randint(50, 100), np.random.randint(20, 50)
            
            xml_content = f"""
            <annotation>
                <filename>test_img_{i}.jpg</filename>
                <size>
                    <width>400</width>
                    <height>300</height>
                    <depth>3</depth>
                </size>
                <object>
                    <name>license_plate</name>
                    <bndbox>
                        <xmin>{x}</xmin>
                        <ymin>{y}</ymin>
                        <xmax>{x + w}</xmax>
                        <ymax>{y + h}</ymax>
                    </bndbox>
                    <license_text>TEST{i+100}</license_text>
                </object>
            </annotation>
            """
            
            xml_path = self.annotations_dir / f'test_img_{i}.xml'
            with open(xml_path, 'w') as f:
                f.write(xml_content)
    
    @unittest.skipIf(not os.path.exists(os.path.join(project_root, 'Dataset')), 
                    "Skipping test because actual dataset not found")
    def test_dataset_loading_real(self):
        """Test loading the real dataset (if available)"""
        try:
            data_path = get_data_path()            images_path = data_path / "images"
            annotations_path = data_path / "annotations"
            
            df = load_dataset(annotations_path, images_path)
            
            self.assertGreater(len(df), 0, "Dataset should have at least one record")
            
            # Check required columns
            required_columns = ['image_path', 'x', 'y', 'w', 'h']
            for col in required_columns:
                self.assertIn(col, df.columns, f"Dataset should have '{col}' column")
            
            # Preprocess a small subset
            subset = df.head(2)
            X, y = preprocess_dataset(subset, image_size=(128, 128))
            
            self.assertEqual(len(X), len(subset), "Processed data should have same length as input")
            self.assertEqual(X[0].shape, (128, 128, 3), "Processed images should have correct shape")
            
        except Exception as e:
            self.fail(f"Dataset loading failed with error: {str(e)}")
    
    def test_pipeline_with_mock_data(self):
        """Test the complete pipeline with mock data"""
        # Load the mock dataset
        df = load_dataset(self.annotations_dir, self.images_dir)
        
        self.assertGreater(len(df), 0, "Mock dataset should have at least one record")
        
        # Preprocess the data
        X, y = preprocess_dataset(df, image_size=(128, 128))
        
        # Duplicate data to simulate augmentation
        X_aug = np.concatenate([X, X], axis=0)
        y_aug = np.concatenate([y, y], axis=0)
        
        self.assertGreater(len(X_aug), len(X), "Augmented data should be larger than original")
        
        # Split the data
        X_train, X_val, y_train, y_val = split_dataset(X_aug, y_aug, test_size=0.5)
        
        # Create a model
        model = create_license_plate_detector(input_shape=(128, 128, 3))
        
        # Compile the model
        model.compile(
            optimizer='adam',
            loss=combined_detection_loss,
            metrics=[enhanced_iou_metric]
        )
        
        # Make predictions
        y_pred = model.predict(X_val)
        
        self.assertEqual(y_pred.shape, y_val.shape, "Predictions should have the same shape as targets")
        
        # Check prediction values are in valid range
        self.assertTrue(np.all(y_pred >= 0) and np.all(y_pred <= 1), 
                      "All prediction values should be in range [0, 1]")
    
    def test_minimal_training(self):
        """Test a very minimal training run"""
        # Load and preprocess the mock dataset
        df = load_dataset(self.annotations_dir, self.images_dir)
        X, y = preprocess_dataset(df, image_size=(64, 64))
        
        # Create a very small model for quick testing
        model = create_license_plate_detector(input_shape=(64, 64, 3))
        model.compile(
            optimizer='adam',
            loss=combined_detection_loss,
            metrics=[enhanced_iou_metric]
        )
        
        # Fit for just 1 epoch
        history = model.fit(
            X, y,
            epochs=1,
            batch_size=1,
            verbose=0
        )
        
        # Check that training completed and history was recorded
        self.assertIn('loss', history.history, "Training history should include loss")
        if 'enhanced_iou_metric' in history.history:
            self.assertGreaterEqual(history.history['enhanced_iou_metric'][0], 0, 
                                  "IoU metric should be non-negative")


if __name__ == '__main__':
    unittest.main()
