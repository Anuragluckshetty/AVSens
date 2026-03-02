"""
Environment Setup Script for YOLOv8 Pedestrian and Car Detection
This script installs all required dependencies for the YOLOv8 training pipeline
"""

import subprocess
import sys

def install_packages():
    """Install required Python packages"""
    
    packages = [
        'ultralytics',  # YOLOv8 implementation
        'opencv-python',  # Image processing
        'matplotlib',  # Visualization
        'seaborn',  # Statistical visualizations
        'pandas',  # Data analysis
        'numpy',  # Numerical operations
        'pillow',  # Image handling
        'pyyaml',  # YAML file handling
        'torch',  # PyTorch deep learning framework
        'torchvision',  # PyTorch vision utilities
    ]
    
    print("=" * 60)
    print("Installing Required Packages for YOLOv8 Training")
    print("=" * 60)
    
    for package in packages:
        print(f"\nInstalling {package}...")
        try:
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', package])
            print(f"✓ {package} installed successfully")
        except subprocess.CalledProcessError as e:
            print(f"✗ Error installing {package}: {e}")
            
    print("\n" + "=" * 60)
    print("Environment Setup Complete!")
    print("=" * 60)
    
    # Verify installations
    print("\nVerifying installations...")
    try:
        import ultralytics
        print(f"✓ Ultralytics version: {ultralytics.__version__}")
        import cv2
        print(f"✓ OpenCV version: {cv2.__version__}")
        import torch
        print(f"✓ PyTorch version: {torch.__version__}")
        print(f"✓ CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"✓ CUDA version: {torch.version.cuda}")
            print(f"✓ GPU: {torch.cuda.get_device_name(0)}")
    except Exception as e:
        print(f"Warning: {e}")
    
if __name__ == "__main__":
    install_packages()
