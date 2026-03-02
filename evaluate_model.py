"""
YOLOv8 Evaluation and Inference Script
Evaluates trained model on test dataset and performs inference
Generates detection results with bounding boxes
"""

from ultralytics import YOLO
import cv2
import os
from pathlib import Path
import json
import numpy as np

def evaluate_model(model_path, data_yaml_path):
    """
    Evaluate the trained model on test dataset
    
    Args:
        model_path: Path to trained model weights
        data_yaml_path: Path to data.yaml file
    """
    
    print("\n" + "=" * 60)
    print("Evaluating YOLOv8 Model on Test Dataset")
    print("=" * 60)
    
    # Load trained model
    model = YOLO(model_path)
    print(f"\n✓ Loaded model from: {model_path}")
    
    # Evaluate on test set
    print("\n📊 Running evaluation on test dataset...")
    print("   (This may take several minutes depending on dataset size)")
    
    metrics = model.val(
        data=data_yaml_path,
        split='test',  # Evaluate on test set
        imgsz=640,
        batch=16,
        conf=0.25,  # Confidence threshold
        iou=0.6,  # IoU threshold for NMS
        verbose=True,
        plots=True,
        save_json=True,
        name='test_evaluation'
    )
    
    print("\n" + "=" * 60)
    print("Evaluation Results")
    print("=" * 60)
    
    # Extract and display metrics
    print(f"\n📈 Performance Metrics:")
    print(f"   Precision: {metrics.box.mp:.4f}")
    print(f"   Recall: {metrics.box.mr:.4f}")
    print(f"   mAP@0.5: {metrics.box.map50:.4f}")
    print(f"   mAP@0.5:0.95: {metrics.box.map:.4f}")
    
    # Per-class metrics
    if hasattr(metrics.box, 'maps'):
        print(f"\n📊 Per-Class mAP@0.5:")
        class_names = ['car', 'pedestrains-and-cars', 'person']
        for i, (name, ap) in enumerate(zip(class_names, metrics.box.ap50)):
            print(f"   {name}: {ap:.4f}")
    
    print("\n✓ Evaluation complete!")
    print(f"   Results saved to: runs/detect/test_evaluation/")
    
    return metrics

def run_inference_on_test_images(model_path, test_images_dir, output_dir, num_samples=10):
    """
    Run inference on test images and save results
    
    Args:
        model_path: Path to trained model
        test_images_dir: Directory containing test images
        output_dir: Directory to save detection results
        num_samples: Number of sample images to process
    """
    
    print("\n" + "=" * 60)
    print("Running Inference on Test Images")
    print("=" * 60)
    
    # Load model
    model = YOLO(model_path)
    print(f"\n✓ Loaded model from: {model_path}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Get test images
    test_images = list(Path(test_images_dir).glob('*.jpg'))
    if not test_images:
        test_images = list(Path(test_images_dir).glob('*.png'))
    
    # Select random sample
    import random
    if len(test_images) > num_samples:
        test_images = random.sample(test_images, num_samples)
    
    print(f"\n🖼️  Processing {len(test_images)} test images...")
    
    # Run inference
    results_data = []
    
    for img_path in test_images:
        print(f"\n   Processing: {img_path.name}")
        
        # Run inference
        results = model(str(img_path), conf=0.25, iou=0.6)
        
        # Save annotated image
        annotated_img = results[0].plot()
        output_path = Path(output_dir) / f"detected_{img_path.name}"
        cv2.imwrite(str(output_path), annotated_img)
        
        # Extract detection information
        boxes = results[0].boxes
        detections = {
            'image': img_path.name,
            'detections': []
        }
        
        for box in boxes:
            det = {
                'class': int(box.cls[0]),
                'class_name': model.names[int(box.cls[0])],
                'confidence': float(box.conf[0]),
                'bbox': box.xyxy[0].tolist()
            }
            detections['detections'].append(det)
        
        results_data.append(detections)
        
        num_dets = len(boxes)
        print(f"   ✓ Detected {num_dets} objects")
        if num_dets > 0:
            for det in detections['detections']:
                print(f"      - {det['class_name']}: {det['confidence']:.3f}")
    
    # Save detection results to JSON
    json_path = Path(output_dir) / 'detection_results.json'
    with open(json_path, 'w') as f:
        json.dump(results_data, f, indent=2)
    
    print("\n" + "=" * 60)
    print("Inference Complete!")
    print("=" * 60)
    print(f"\n✓ Annotated images saved to: {output_dir}")
    print(f"✓ Detection results saved to: {json_path}")
    
    return results_data

def predict_on_single_image(model_path, image_path, output_path=None, conf=0.25):
    """
    Run prediction on a single image
    
    Args:
        model_path: Path to trained model
        image_path: Path to input image
        output_path: Path to save output (optional)
        conf: Confidence threshold
    """
    
    model = YOLO(model_path)
    results = model(image_path, conf=conf)
    
    # Display results
    annotated_img = results[0].plot()
    
    # Save if output path provided
    if output_path:
        cv2.imwrite(output_path, annotated_img)
        print(f"✓ Saved result to: {output_path}")
    
    # Print detections
    boxes = results[0].boxes
    print(f"\nDetected {len(boxes)} objects:")
    for box in boxes:
        cls = int(box.cls[0])
        conf_score = float(box.conf[0])
        print(f"  - {model.names[cls]}: {conf_score:.3f}")
    
    return results

def main():
    """Main evaluation and inference function"""
    
    print("\n" + "=" * 80)
    print(" " * 20 + "YOLOv8 EVALUATION PIPELINE")
    print(" " * 15 + "Testing on Separate Test Dataset")
    print("=" * 80)
    
    # Paths
    current_dir = Path(__file__).parent.absolute()
    model_path = current_dir / 'runs' / 'detect' / 'pedestrian_car_detection' / 'weights' / 'best.pt'
    data_yaml_path = current_dir / 'data_updated.yaml'
    test_images_dir = current_dir / 'test' / 'images'
    output_dir = current_dir / 'test_results'
    
    # Check if model exists
    if not model_path.exists():
        print(f"\n❌ Error: Model not found at {model_path}")
        print("   Please run train_yolov8.py first to train the model.")
        return
    
    # Step 1: Evaluate on test dataset
    print("\nStep 1: Evaluating model on test dataset...")
    metrics = evaluate_model(str(model_path), str(data_yaml_path))
    
    # Step 2: Run inference on sample test images
    print("\n\nStep 2: Running inference on sample test images...")
    results = run_inference_on_test_images(
        str(model_path),
        str(test_images_dir),
        str(output_dir),
        num_samples=10
    )
    
    print("\n" + "=" * 80)
    print("Evaluation pipeline completed successfully!")
    print("=" * 80)
    print("\nGenerated outputs:")
    print(f"1. Evaluation metrics: runs/detect/test_evaluation/")
    print(f"2. Detection visualizations: {output_dir}")
    print(f"3. Detection results JSON: {output_dir}/detection_results.json")
    print("=" * 80 + "\n")

if __name__ == "__main__":
    main()
