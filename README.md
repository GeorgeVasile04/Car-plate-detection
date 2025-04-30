# License Plate Detection

A comprehensive framework for detecting license plates in images using Convolutional Neural Networks.

## Project Structure

```
license_plate_detection/
│
├── data/
│   ├── loader.py        # Dataset loading and preprocessing
│   └── augmentation.py  # Data augmentation techniques
│
├── models/
│   ├── detector.py      # License plate detector models
│   └── losses.py        # Custom loss functions and metrics
│
├── train/
│   ├── trainer.py       # Training loop and callbacks
│   └── scheduler.py     # Learning rate scheduling
│
├── evaluation/
│   └── evaluator.py     # Model evaluation and metrics
│
├── utils/
│   ├── visualization.py # Visualization utilities
│   └── analysis.py      # Error analysis tools
│
├── main.py              # Main functions
└── cli.py               # Command-line interface
```

## Installation

### Option 1: Install locally

```bash
# Clone the repository
git clone https://github.com/yourusername/license-plate-detection.git
cd license-plate-detection

# Install the package in development mode
pip install -e .
```

### Option 2: Install from GitHub

```bash
pip install git+https://github.com/yourusername/license-plate-detection.git
```

## Usage

### From Python

```python
from license_plate_detection.main import load_and_prepare_model, detect_license_plate

# Load model
model = load_and_prepare_model('path/to/model.h5')

# Detect license plate in image
plate_region, bbox = detect_license_plate(model, 'path/to/image.jpg')
```

### From Command Line

```bash
# Train a new model
license-plate-detect train --data_path path/to/dataset --output_model_path model.h5 --epochs 50

# Detect license plate in an image
license-plate-detect detect --model_path model.h5 --image_path path/to/image.jpg

# Evaluate model
license-plate-detect evaluate --model_path model.h5 --data_path path/to/dataset
```

## Dataset Format

The dataset should be organized as follows:

```
dataset/
├── annotations/
│   ├── file1.xml
│   ├── file2.xml
│   └── ...
└── images/
    ├── file1.jpg
    ├── file2.jpg
    └── ...
```

The XML annotation files should follow the standard format with `<object>`, `<bndbox>`, and coordinates tags.

## Model Architecture

The license plate detector uses a CNN-based architecture with the following components:

- Backbone: Customized CNN with residual connections
- Loss functions: IoU loss and bounding box regression
- Metrics: IoU (Intersection over Union)

## Citation

If you use this code in your research, please cite:

```
@software{license_plate_detection,
  author = {Your Name},
  title = {License Plate Detection},
  year = {2023},
  url = {https://github.com/yourusername/license-plate-detection}
}
```

## License

MIT License
