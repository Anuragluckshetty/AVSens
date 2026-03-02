# YOLOv8 Pedestrian and Car Detection - Internship Assignment

## 📋 Project Overview

This project implements a complete YOLOv8-based object detection system for pedestrian and car detection, suitable for autonomous vehicle applications. The implementation covers the entire pipeline from dataset preparation through model training, evaluation, performance analysis, and comprehensive reporting.

## 🎯 Assignment Objectives

- ✅ Train YOLOv8 model for pedestrian and car detection
- ✅ Evaluate using standard metrics (Precision, Recall, mAP@0.5, mAP@0.5:0.95)
- ✅ Analyze model performance and identify failure cases
- ✅ Propose practical improvements
- ✅ Explain CNN and max pooling concepts
- ✅ Generate comprehensive internship report

## 📊 Dataset Information

**Source**: Roboflow (YOLOv8 format)

**Classes**:
- Class 0: `car` - Vehicles (cars, SUVs)
- Class 1: `pedestrains-and-cars` - Combined scenes
- Class 2: `person` - Pedestrians

**Dataset Split**:
- Training Set: ~125 images
- Validation Set: Variable
- Test Set: 19 images

**Annotation Format**: YOLO polygon/segmentation format with normalized coordinates

## 🚀 Quick Start Guide

### Step 1: Install Dependencies

```bash
python setup_environment.py
```

This installs all required packages:
- Ultralytics (YOLOv8)
- PyTorch
- OpenCV
- Matplotlib, Seaborn
- And other dependencies

### Step 2: Train the Model

```bash
python train_yolov8.py
```

**What this does**:
- Updates data.yaml with absolute paths
- Loads YOLOv8 nano pretrained weights
- Trains for 50 epochs (adjustable)
- Uses train and validation sets only
- Saves best model to `runs/detect/pedestrian_car_detection/weights/best.pt`

**Training time**: 1-3 hours depending on hardware

**Key parameters** (edit in train_yolov8.py):
```python
model_size = 'n'  # Options: 'n', 's', 'm', 'l', 'x'
epochs = 50
img_size = 640
batch_size = 16
```

### Step 3: Evaluate on Test Set

```bash
python evaluate_model.py
```

**What this does**:
- Loads trained model
- Runs evaluation on separate test dataset
- Calculates Precision, Recall, mAP@0.5, mAP@0.5:0.95
- Runs inference on sample test images
- Saves detection visualizations to `test_results/`
- Exports detection results to JSON

### Step 4: Generate Visualizations

```bash
python visualize_results.py
```

**What this does**:
- Creates training metrics plots (loss curves, mAP, precision/recall)
- Generates detection results grid
- Produces detection statistics charts
- Saves all visualizations to `visualizations/`

**Output files**:
- `training_metrics.png` - Complete training history
- `detection_grid.png` - Grid of detected images
- `detection_statistics.png` - Class distribution and confidence analysis

### Step 5: Analyze Performance

```bash
python analyze_performance.py
```

**What this does**:
- Detailed interpretation of all metrics
- Identifies failure cases (false positives, false negatives)
- Analyzes difficult scenarios (occlusions, small objects, crowded scenes)
- Proposes specific improvements
- Explains CNN and max pooling concepts

### Step 6: Generate Final Report

```bash
python generate_report.py
```

**What this does**:
- Compiles all results into comprehensive markdown report
- Includes all sections required for assignment submission
- Generates `INTERNSHIP_ASSIGNMENT_REPORT.md`

**Report includes**:
- Introduction and objectives
- Dataset description and annotation format
- Training methodology
- Evaluation metrics and interpretation
- Detailed performance analysis
- Failure case discussion
- Improvement recommendations
- CNN concepts explanation
- Conclusion and future work
- Technical appendix

## 📁 Project Structure

```
AVs/
├── data.yaml                           # Original dataset config
├── data_updated.yaml                   # Auto-generated with absolute paths
│
├── train/                              # Training data
│   ├── images/                         # ~125 images
│   └── labels/                         # YOLO format annotations
│
├── valid/                              # Validation data
│   ├── images/
│   └── labels/
│
├── test/                               # Test data (held out)
│   ├── images/                         # 19 test images
│   └── labels/
│
├── setup_environment.py                # Dependency installer
├── train_yolov8.py                     # Training script
├── evaluate_model.py                   # Evaluation script
├── visualize_results.py                # Visualization generator
├── analyze_performance.py              # Performance analyzer
├── generate_report.py                  # Report generator
│
├── runs/                               # Training outputs
│   └── detect/
│       └── pedestrian_car_detection/
│           ├── weights/
│           │   ├── best.pt             # Best model
│           │   └── last.pt             # Last epoch
│           ├── results.csv             # Training metrics
│           └── *.png                   # Training plots
│
├── test_results/                       # Test predictions
│   ├── detected_*.jpg                  # Visualized detections
│   └── detection_results.json          # Detection data
│
├── visualizations/                     # Generated charts
│   ├── training_metrics.png
│   ├── detection_grid.png
│   └── detection_statistics.png
│
└── INTERNSHIP_ASSIGNMENT_REPORT.md     # Final report
```

## 📝 Complete Workflow

```
1. setup_environment.py
   └─> Installs all dependencies
       └─> Verifies PyTorch, CUDA, etc.

2. train_yolov8.py
   └─> Loads dataset
       └─> Trains YOLOv8 model
           └─> Saves best.pt

3. evaluate_model.py
   └─> Loads best.pt
       └─> Evaluates on test set
           └─> Saves metrics & detections

4. visualize_results.py
   └─> Reads results.csv
       └─> Generates visualizations
           └─> Saves plots

5. analyze_performance.py
   └─> Analyzes metrics
       └─> Identifies failure cases
           └─> Proposes improvements

6. generate_report.py
   └─> Compiles all outputs
       └─> Generates final report
           └─> INTERNSHIP_ASSIGNMENT_REPORT.md
```

## 🔧 Customization Options

### Change Model Size

In [train_yolov8.py](train_yolov8.py#L95):
```python
model = train_yolov8(
    data_yaml_path=data_yaml_path,
    model_size='s',  # 'n', 's', 'm', 'l', 'x'
    epochs=100,
    img_size=640,
    batch_size=16
)
```

### Adjust Training Parameters

In [train_yolov8.py](train_yolov8.py#L50):
```python
results = model.train(
    epochs=100,          # More epochs
    imgsz=1280,          # Higher resolution
    batch=32,            # Larger batch (if GPU allows)
    patience=15,         # Early stopping patience
    lr0=0.01,           # Learning rate
    # ... other parameters
)
```

### Change Confidence Threshold

In [evaluate_model.py](evaluate_model.py#L32):
```python
metrics = model.val(
    conf=0.35,  # Adjust threshold (0.1-0.5)
    iou=0.6,
    # ...
)
```

## 📊 Expected Outputs

### Training Metrics
- **Precision**: ~0.70-0.80
- **Recall**: ~0.65-0.75
- **mAP@0.5**: ~0.70-0.75
- **mAP@0.5:0.95**: ~0.40-0.50

### Training Time
- **CPU**: 3-6 hours
- **GPU (GTX 1060+)**: 1-2 hours
- **GPU (RTX 3080+)**: 30-60 minutes

### File Sizes
- **best.pt**: ~6 MB (nano model)
- **Training plots**: ~1-2 MB total
- **Detected images**: ~5-10 MB total
- **Report**: ~50-100 KB

## 🐛 Troubleshooting

### Issue: CUDA out of memory
**Solution**: Reduce batch size
```python
batch_size=8  # or 4
```

### Issue: Training too slow on CPU
**Solution**: Use smaller model or fewer epochs
```python
model_size='n'  # Nano is fastest
epochs=30      # Reduce epochs
```

### Issue: Import errors
**Solution**: Reinstall dependencies
```bash
python setup_environment.py
```

### Issue: No GPU detected
**Solution**: Check CUDA installation
```python
import torch
print(torch.cuda.is_available())
print(torch.version.cuda)
```

## 📈 Performance Metrics Explained

### Precision
**What it means**: Of all objects the model detected, how many were correct?

**Formula**: `TP / (TP + FP)`

**High Precision**: Few false alarms
**Low Precision**: Many incorrect detections

### Recall
**What it means**: Of all actual objects, how many did the model find?

**Formula**: `TP / (TP + FN)`

**High Recall**: Few missed objects
**Low Recall**: Many objects not detected

### mAP@0.5
**What it means**: Average detection quality at 50% overlap threshold

**Range**: 0.0 (worst) to 1.0 (perfect)

**Good Score**: > 0.7
**Moderate**: 0.5 - 0.7
**Needs Work**: < 0.5

### mAP@0.5:0.95
**What it means**: Average quality across stricter overlap thresholds

**More strict** than mAP@0.5 - requires better localization

## 🎓 Learning Resources

### Understanding YOLO
- [Ultralytics YOLOv8 Docs](https://docs.ultralytics.com/)
- [YOLO Paper (original)](https://arxiv.org/abs/1506.02640)

### Object Detection Fundamentals
- [Stanford CS231n](http://cs231n.stanford.edu/)
- [Dive into Deep Learning](https://d2l.ai/)

### Computer Vision Metrics
- [mAP Explanation](https://jonathan-hui.medium.com/map-mean-average-precision-for-object-detection-45c121a31173)

## ✅ Assignment Checklist

- [x] Dataset prepared in YOLOv8 format
- [x] Model trained on train+val sets
- [x] Test set kept separate for evaluation
- [x] Evaluation metrics calculated (P, R, mAP@0.5, mAP@0.5:0.95)
- [x] Detection examples generated
- [x] Performance analysis completed
- [x] Failure cases identified and discussed
- [x] Improvement recommendations provided
- [x] CNN and max pooling explained
- [x] Comprehensive report generated

## 🎯 Next Steps for Production

1. **Model Improvement**
   - Upgrade to YOLOv8-small or medium
   - Train with higher resolution (1280×1280)
   - Expand dataset with more diverse examples

2. **Deployment Optimization**
   - Convert to ONNX for deployment
   - Quantize to INT8 for faster inference
   - Optimize with TensorRT

3. **Integration**
   - Add object tracking
   - Implement trajectory prediction
   - Integrate with vehicle control system

## 📞 Support

If you encounter issues:

1. Check [Ultralytics GitHub Issues](https://github.com/ultralytics/ultralytics/issues)
2. Review error messages carefully
3. Verify all dependencies are installed
4. Check GPU/CUDA compatibility

## 📄 License

This project uses:
- **Ultralytics YOLOv8**: AGPL-3.0 license
- **Dataset**: CC BY 4.0 (as per Roboflow)
- **Project Code**: Educational use

## 🙏 Acknowledgments

- Ultralytics team for YOLOv8 implementation
- Roboflow for dataset hosting
- FudanPed and PennPed dataset creators
- Open-source computer vision community

---

**Ready to start?** Run the scripts in order:
```bash
python setup_environment.py
python train_yolov8.py
python evaluate_model.py
python visualize_results.py
python analyze_performance.py
python generate_report.py
```

Good luck with your internship assignment! 🚀
