# Dataset_XML: Creation and Specifications

## Overview
This document details the creation process and specifications of the Dataset_XML dataset, which combines license plate datasets from three separate sources (DataV2, DataV3, and Data) into a unified dataset with consistent annotation format.

## Creation Process

### Source Datasets
The Dataset_XML was created by combining:

1. **DataV2**: 
   - Structure: Separate `images` and `annotations` folders
   - Contains 433 license plate images with corresponding XML annotations
   - Format: PNG images with XML annotation files
   - It comes from: https://www.kaggle.com/datasets/andrewmvd/car-plate-detection

2. **DataV3**:
   - Structure: Single `images` folder containing both images and annotations
   - Contains 207 license plate images with corresponding XML annotations
   - Format: JPEG images with XML annotation files
   - It comes from : https://www.kaggle.com/datasets/alihassanml/car-number-plate

3. **Data**:
   - Structure: Single `Total` folder containing both images and TXT annotations
   - Contains 444 license plate images with corresponding TXT annotations
   - Format: JPG images with TXT annotation files
   - Original format used (x, y, w, h) coordinates rather than XML format
   - It comes from : https://github.com/openalpr/benchmarks/tree/master/endtoend

### Combination Methodology
To create a unified dataset while avoiding filename conflicts, we implemented the following process:

1. **Directory Setup**:
   - Created a new `Dataset_XML` directory with two subdirectories:
     - `images`: For all combined images
     - `annotations`: For all combined XML annotation files

2. **File Prefixing**:
   - Added prefixes to distinguish sources:
     - `v2_` prefix for all files from DataV2
     - `v3_` prefix for all files from DataV3
     - `data_` prefix for all files from Data

3. **XML Annotation Updates and Conversions**:
   - Modified each XML file to ensure the `filename` element points to the correct prefixed image file
   - Updated any `path` elements to reflect the new file locations
   - Preserved all bounding box coordinates and other annotation details
   - Converted TXT format annotations from Data folder to XML format:
     - Transformed (x, y, w, h) coordinates to (xmin, ymin, xmax, ymax)
     - Created standardized XML structure to match other datasets
     - Preserved license plate text information where available in TXT files

4. **Implementation**:
   - Created a Python script (`combine_datasets.py`) that was executed through a Jupyter notebook (`Combined_Datasets.ipynb`) for DataV2 and DataV3
   - Created a second notebook (`Convert_TXT_Anotation_To_XML.ipynb`) to convert the TXT annotations from Data folder to XML format
   - Used the ElementTree library to parse and modify XML files
   - Implemented error handling for missing or malformed files

## Dataset Specifications

### Size and Composition
- **Total Images**: 1084
  - 433 images from DataV2 (40%)
  - 207 images from DataV3 (19%)
  - 444 images from Data (41%)
- **Total Annotations**: 1084 (one XML annotation file per image)

### Directory Structure
```
Dataset_XML/
├── Dataset_XML_Summary.md
├── annotations/
│   ├── v2_*.xml    (433 files)
│   ├── v3_*.xml    (207 files)
│   └── data_*.xml  (444 files)
└── images/
    ├── v2_*.png    (433 files)
    ├── v3_*.jpeg   (207 files)
    └── data_*.jpg  (444 files)
```

### Annotation Format
The dataset uses the XML annotation format with the following structure:

```xml
<annotation>
    <folder>images</folder>
    <filename>v2_Cars0.png</filename>
    <size>
        <width>500</width>
        <height>268</height>
        <depth>3</depth>
    </size>
    <segmented>0</segmented>
    <object>
        <n>licence</n>
        <pose>Unspecified</pose>
        <truncated>0</truncated>
        <occluded>0</occluded>
        <difficult>0</difficult>
        <bndbox>
            <xmin>226</xmin>
            <ymin>125</ymin>
            <xmax>419</xmax>
            <ymax>173</ymax>
        </bndbox>
    </object>
</annotation>
```

### Bounding Box Coordinate System
- Uses the `(xmin, ymin, xmax, ymax)` coordinate system
- `xmin, ymin`: Top-left corner of the bounding box
- `xmax, ymax`: Bottom-right corner of the bounding box
- Origin (0,0) is at the top-left corner of the image
- X-axis increases to the right, Y-axis increases downward
- All coordinates are in pixels

### Visualizing Bounding Boxes
When visualizing the bounding boxes using matplotlib:
```python
# Draw rectangle using matplotlib
rect = plt.Rectangle((xmin, ymin), width=xmax-xmin, height=ymax-ymin,
                     fill=False, edgecolor='lime', linewidth=2)
```

## Differences from Other Dataset Formats
This section describes the original format differences before conversion. Now that all annotations have been converted to XML format, these differences are only relevant for understanding the conversion process.

1. **XML Format (Original in DataV2 and DataV3, now used for all in Dataset_XML)**:
   - Uses `(xmin, ymin, xmax, ymax)` defining the corners
   - Directly specifies the full box coordinates

2. **TXT Format (Original in Data folder, now converted to XML)**:
   - Used `(x, y, w, h)` format
   - Where (x,y) was the top-left corner and w,h were dimensions
   - Often included the license plate text (e.g., "AYO9034")
   - Example: `AYO9034.jpg 528 412 162 52 AYO9034`
   
The conversion from TXT to XML was performed using a coordinate transformation:
```python
# Convert from (x, y, w, h) to (xmin, ymin, xmax, ymax)
xmin = x
ymin = y
xmax = x + w
ymax = y + h
```

This conversion ensures that all annotations in the Dataset_XML have a consistent format and coordinate system.

## Usage Guidelines
This combined dataset is designed to be used with the CNN license plate detection scripts. When using this dataset:

1. Load the images from the `images` directory
2. Parse the corresponding XML annotations from the `annotations` directory
3. Extract the bounding box coordinates from the XML
4. Use the coordinates for training object detection models

The consistent naming convention (prefixed filenames) ensures that each image can be easily matched to its annotation file despite coming from different source datasets with different original formats.

### Additional Features of the Combined Dataset
- **License Plate Text**: Some annotations (particularly those converted from the Data folder) include the actual license plate text, which can be found in the `license_text` element in the XML file. This can be useful for OCR (Optical Character Recognition) tasks.
- **Diverse Image Sources**: With three different source datasets, the combined dataset offers greater diversity in terms of image quality, lighting conditions, angles, and license plate types.
- **Standardized Format**: All annotations now follow the same XML structure and coordinate system, simplifying dataset loading and processing in machine learning pipelines.

## Next Steps
This dataset can now be used for:
- Training CNN-based license plate detectors
- Evaluating model performance on a diverse set of license plate images
- Further augmentation or preprocessing for improved model training
- Training license plate recognition models using the license plate text information where available
- Comparing performance between images from different source datasets
