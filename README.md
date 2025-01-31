
# Defect Detection in Canned Food - Computer Vision Pipeline

![Example Output](assets/sample_prediction.png)

An end-to-end computer vision pipeline for detecting surface defects in canned food products, developed as part of Oneture Technologies' Data Science assignment.

## Features

- **Binary Classification**: Detects defective vs non-defective cans
- **Advanced Architectures**: Implements ResNet50 & EfficientNet-B0
- **Production Ready**: Includes training pipeline & inference scripts
- **Comprehensive Monitoring**: Tracks loss curves & performance metrics
- **COCO Compatibility**: Processes RoboFlow dataset format

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/defect-detection.git
   cd defect-detection
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Dataset Preparation

1. Download the dataset from RoboFlow:
   [Canned Food Surface Defect Dataset](https://universe.roboflow.com/canned-food-surface-defect-classification/canned-food-surface-defect/dataset/6)

2. Organize the dataset structure as follows:
   ```bash
   canned-food-dataset/
   ├── train/
   │   ├── images/
   │   └── _annotations.coco.json
   ├── valid/
   │   ├── images/
   │   └── _annotations.coco.json
   └── test/
       ├── images/
       └── _annotations.coco.json
   ```

## Training Pipeline

Run the complete training process:
```bash
python Defect_Detection_Pipeline.py --dataset_path path/to/canned-food-dataset
```

### Outputs:
- **Trained models**: `best_ResNet.pth`, `best_EfficientNet.pth`
- **Processed data**: Stored in `processed_data/` directory
- **Final models**: `defect_detection_resnet50.pth`, `defect_detection_efficientnet.pth`

## Inference

### Using ResNet50:
```python
# resnet50.py
from inference import predict_defect

image_path = "path/to/test_image.jpg"  # Path to the test image

prediction, confidence = predict_defect(
    image_path=image_path,
    model_path="defect_detection_resnet50.pth",
    model_arch="resnet50"
)
print(f"ResNet50 Prediction: {prediction} ({confidence:.2%})")
```

### Using EfficientNet:
```python
# efficientnet.py
from inference import predict_defect

image_path = "path/to/test_image.jpg"  # Path to the test image

prediction, confidence = predict_defect(
    image_path=image_path,
    model_path="defect_detection_efficientnet.pth",
    model_arch="efficientnet"
)
print(f"EfficientNet Prediction: {prediction} ({confidence:.2%})")
```

## Repository Structure

```bash
├── Defect_Detection_Pipeline.py     # Main training script
├── processed_data/                  # Preprocessed CSV labels
├── defect-detection.ipynb           # Complete workflow notebook
├── requirements.txt                 # Dependency list
├── resnet50.py                      # ResNet inference example
├── efficientnet.py                  # EfficientNet inference example
└── best_*.pth                       # Best model checkpoints
```

## Example Output
Check the `defect-detection.ipynb` for a complete workflow, including:
- Data distribution analysis
- Sample images with annotations
- Training progress visualization
- Model performance comparison

## License
CC BY 4.0 - Match dataset license requirements

## References
[Canned Food Surface Defect Dataset](https://universe.roboflow.com/canned-food-surface-defect-classification/canned-food-surface-defect/dataset/6)
```

This markdown format should display properly on GitHub or any other markdown-compatible platform. Let me know if you'd like to add or modify anything!
