# Dataset_XML: Creation and Specifications

## Overview
This document details the creation process and specifications of the Dataset_XML dataset, which combines license plate datasets from two separate sources (DataV2 and DataV3) into a unified dataset with consistent annotation format.

## Creation Process

### Source Datasets
The Dataset_XML was created by combining:

1. **DataV2**: 
   - Structure: Separate `images` and `annotations` folders
   - Contains 433 license plate images with corresponding XML annotations
   - Format: PNG images with XML annotation files

2. **DataV3**:
   - Structure: Single `images` folder containing both images and annotations
   - Contains 207 license plate images with corresponding XML annotations
   - Format: JPEG images with XML annotation files

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

3. **XML Annotation Updates**:
   - Modified each XML file to ensure the `filename` element points to the correct prefixed image file
   - Updated any `path` elements to reflect the new file locations
   - Preserved all bounding box coordinates and other annotation details

4. **Implementation**:
   - Created a Python script (`combine_datasets.py`) that was executed through a Jupyter notebook (`Combined_Datasets.ipynb`)
   - Used the ElementTree library to parse and modify XML files
   - Implemented error handling for missing or malformed files

## Dataset Specifications

### Size and Composition
- **Total Images**: 640
  - 433 images from DataV2 (68%)
  - 207 images from DataV3 (32%)
- **Total Annotations**: 640 (one XML annotation file per image)

### Directory Structure
```
Dataset_XML/
├── Data_XML_Specifications.txt
├── annotations/
│   ├── v2_*.xml  (433 files)
│   └── v3_*.xml  (207 files)
└── images/
    ├── v2_*.png  (433 files)
    └── v3_*.jpeg (207 files)
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
It's important to note that the Dataset_XML uses a different coordinate system than the TXT format found in the `Data` folder:

1. **XML Format (Dataset_XML)**:
   - Uses `(xmin, ymin, xmax, ymax)` defining the corners
   - Directly specifies the full box coordinates

2. **TXT Format (Data folder)**:
   - Uses `(x, y, w, h)` format
   - Where (x,y) is the top-left corner and w,h are dimensions

## Usage Guidelines
This combined dataset is designed to be used with the CNN license plate detection scripts. When using this dataset:

1. Load the images from the `images` directory
2. Parse the corresponding XML annotations from the `annotations` directory
3. Extract the bounding box coordinates from the XML
4. Use the coordinates for training object detection models

The consistent naming convention (prefixed filenames) ensures that each image can be easily matched to its annotation file despite coming from different source datasets.

## Next Steps
This dataset can now be used for:
- Training CNN-based license plate detectors
- Evaluating model performance on a diverse set of license plate images
- Further augmentation or preprocessing for improved model training
