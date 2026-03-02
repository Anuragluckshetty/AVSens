"""
YOLOv8 Training Script for Pedestrian and Car Detection
This script trains a YOLOv8 model on your custom dataset with train/val splits
Test dataset is kept separate for final evaluation
"""

from ultralytics import YOLO
import yaml
import os
from pathlib import Path

def update_data_yaml():
    """Update data.yaml with absolute paths"""
    
    # Get current directory
    current_dir = Path(__file__).parent.absolute()
    
    # Read existing data.yaml
    data_yaml_path = current_dir / 'data.yaml'
    with open(data_yaml_path, 'r') as f:
        data = yaml.safe_load(f)
    
    # Update paths to absolute
    data['train'] = str(current_dir / 'train' / 'images')
    data['val'] = str(current_dir / 'valid' / 'images')
    data['test'] = str(current_dir / 'test' / 'images')
    
    # Save updated data.yaml to a new file
    updated_yaml_path = current_dir / 'data_updated.yaml'
    with open(updated_yaml_path, 'w') as f:
        yaml.dump(data, f, default_flow_style=False)
    
    print(f"✓ Updated data.yaml saved to: {updated_yaml_path}")
    print(f"  Train path: {data['train']}")
    print(f"  Val path: {data['val']}")
    print(f"  Test path: {data['test']}")
    print(f"  Number of classes: {data['nc']}")
    print(f"  Class names: {data['names']}")
    
    return str(updated_yaml_path)

def train_yolov8(data_yaml_path, model_size='n', epochs=50, img_size=640, batch_size=16):
    """
    Train YOLOv8 model
    
    Args:
        data_yaml_path: Path to the data.yaml file
        model_size: Model size ('n', 's', 'm', 'l', 'x' - nano, small, medium, large, xlarge)
        epochs: Number of training epochs
        img_size: Input image size
        batch_size: Batch size for training
    """
    
    print("\n" + "=" * 60)
    print(f"Training YOLOv8{model_size} for Pedestrian and Car Detection")
    print("=" * 60)
    
    # Initialize YOLOv8 model
    model_name = f'yolov8{model_size}.pt'
    print(f"\nLoading YOLOv8 model: {model_name}")
    model = YOLO(model_name)
    
    # Training configuration
    print("\nTraining Configuration:")
    print(f"  Epochs: {epochs}")
    print(f"  Image size: {img_size}")
    print(f"  Batch size: {batch_size}")
    print(f"  Model: YOLOv8{model_size}")
    print(f"  Dataset: {data_yaml_path}")
    
    # Train the model
    print("\nStarting training...")
    print("=" * 60)
    
    results = model.train(
        data=data_yaml_path,
        epochs=epochs,
        imgsz=img_size,
        batch=batch_size,
        name='pedestrian_car_detection',
        project='runs/detect',
        patience=10,  # Early stopping patience
        save=True,
        plots=True,  # Generate training plots
        device='cpu',  # Use CPU (no GPU available)
        verbose=True,
        val=True,  # Validate during training
        # Optimization parameters
        lr0=0.01,  # Initial learning rate
        lrf=0.01,  # Final learning rate factor
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3.0,
        warmup_momentum=0.8,
        warmup_bias_lr=0.1,
        # Augmentation
        hsv_h=0.015,  # HSV-Hue augmentation
        hsv_s=0.7,  # HSV-Saturation
        hsv_v=0.4,  # HSV-Value
        degrees=0.0,  # Rotation
        translate=0.1,  # Translation
        scale=0.5,  # Scale
        shear=0.0,  # Shear
        perspective=0.0,  # Perspective
        flipud=0.0,  # Flip up-down
        fliplr=0.5,  # Flip left-right
        mosaic=1.0,  # Mosaic augmentation
        mixup=0.0,  # Mixup augmentation
    )
    
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print(f"\nTrained model saved to: runs/detect/pedestrian_car_detection/weights/best.pt")
    print(f"Training results saved to: runs/detect/pedestrian_car_detection/")
    
    return model

def main():
    """Main training function"""
    
    print("\n" + "=" * 80)
    print(" " * 20 + "YOLOv8 TRAINING PIPELINE")
    print(" " * 15 + "Pedestrian and Car Detection")
    print("=" * 80)
    
    # Step 1: Update data.yaml with absolute paths
    print("\nStep 1: Preparing dataset configuration...")
    data_yaml_path = update_data_yaml()
    
    # Step 2: Train the model
    print("\nStep 2: Training YOLOv8 model...")
    print("\nNOTE: Training uses only train and validation datasets.")
    print("      Test dataset is kept separate for final evaluation.\n")
    
    # You can change these parameters:
    # - model_size: 'n' (nano, fastest), 's' (small), 'm' (medium), 'l' (large), 'x' (xlarge, most accurate)
    # - epochs: number of training epochs (50-100 recommended)
    # - img_size: input image size (640 is default)
    # - batch_size: adjust based on your GPU memory (16, 32, etc.)
    
    model = train_yolov8(
        data_yaml_path=data_yaml_path,
        model_size='n',  # Start with nano for faster training
        epochs=30,  # Reduced for CPU training
        img_size=640,
        batch_size=4  # Reduced for CPU training
    )
    
    print("\n" + "=" * 80)
    print("Training pipeline completed successfully!")
    print("=" * 80)
    print("\nNext steps:")
    print("1. Check training results in: runs/detect/pedestrian_car_detection/")
    print("2. Run evaluation script to test on the test dataset")
    print("3. Run inference script for predictions on new images")
    print("=" * 80 + "\n")

if __name__ == "__main__":
    main()
