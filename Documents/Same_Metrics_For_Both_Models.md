# CNN License Plate Detection with Synchronized Metrics

This notebook has been updated to use the same evaluation metrics and presentation format as the YOLO detection approach. This ensures that both models can be directly and fairly compared using consistent metrics.

## Key metrics now tracked for both models:

1. **IoU Statistics**:
   - Average, Median, Min, and Max IoU
   - IoU distribution visualization
   
2. **Object Detection Metrics**:
   - mAP@0.5
   - mAP@0.5:0.95 (COCO standard)
   - Precision at various thresholds
   
3. **Plate Size Analysis**:
   - Performance on small, medium, and large plates
   - Using identical size thresholds for both models
   
4. **Confidence Scores**:
   - CNN model now also includes confidence score proxies
   - Same visualization format with scores displayed
   
5. **Data Source Comparisons**:
   - Performance breakdown by dataset origin
   - Visual comparisons across data sources
   
By synchronizing these metrics between both approaches, we can now directly compare the CNN-based detector and YOLO-based detector and identify the strengths and weaknesses of each approach.