# YOLOv8 Pedestrian and Car Detection

## Overview

This project implements an object detection system using YOLOv8 to detect pedestrians and cars in real-world images.  
The system is trained using a custom annotated dataset prepared in YOLO format.

The goal of the project is to demonstrate how modern deep learning models can be used for real-time object detection in applications such as autonomous driving, traffic monitoring, and surveillance systems.

The project was developed as part of the **AI/ML Systems Internship selection task for AVSens**.

---

## Dataset

The dataset contains images of road scenes with pedestrians and vehicles.

### Dataset Split

| Dataset | Images |
|-------|-------|
| Train | 131 |
| Validation | 37 |
| Test | 19 |
| Total | 187 |

### Classes

The model detects the following classes:

- Car
- Person (Pedestrian)

All images are annotated using the YOLO bounding box format.

Example annotation format:

---

## Model

The project uses **YOLOv8 Nano** from the Ultralytics framework.

### Model Configuration

| Parameter | Value |
|----------|------|
| Model | YOLOv8n |
| Image Size | 640 |
| Epochs | 30 |
| Early Stopping | Epoch 24 |
| Framework | PyTorch |
| Library | Ultralytics YOLOv8 |

---

## Training

Training was performed using the Ultralytics YOLO framework.

### Training Command

During training, **early stopping occurred at epoch 24**, indicating that the model had reached stable validation performance.

---

## Model Performance

Evaluation was performed on the **test dataset**.

### Overall Performance

| Metric | Value |
|------|------|
| Precision | 60.74% |
| Recall | 52.45% |
| mAP@0.5 | 54.35% |
| mAP@0.5:0.95 | 36.70% |

### Per-Class Performance

| Class | mAP@0.5 |
|------|------|
| Car | 45.27% |
| Person | 63.43% |

---

## Detection Examples

The trained YOLOv8 model successfully detects pedestrians and vehicles in different real-world environments such as:

- urban roads
- pedestrian crossings
- crowded scenes
- highway traffic

Detected objects are shown with bounding boxes and confidence scores.

---

## Failure Cases

Some challenges observed during testing include:

- small pedestrians far from the camera
- partially occluded objects
- crowded scenes
- background objects incorrectly detected as vehicles

These limitations are common when training models with relatively small datasets.

---

## Project Structure
AVs/
├── data.yaml
├── train/
│ ├── images/
│ └── labels/
├── valid/
│ ├── images/
│ └── labels/
├── test/
│ ├── images/
│ └── labels/
├── train_yolov8.py
├── evaluate_model.py
├── visualize_results.py
├── runs/
│ └── detect/
│ └── pedestrian_car_detection/
│ ├── weights/
│ │ ├── best.pt
│ │ └── last.pt
│ └── results.csv
└── visualizations/

---


---

## Future Improvements

Possible improvements for the project include:

- increasing dataset size
- using larger YOLO models (YOLOv8s or YOLOv8m)
- applying stronger data augmentation
- training for more epochs
- increasing input image resolution

---

## Author

Name: Anurag Luckshetty 
USN: NNM23AM008

