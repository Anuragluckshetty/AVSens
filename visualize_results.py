"""
Visualization Script for YOLOv8 Detection Results
Creates comprehensive visualizations of training results and detection outputs
"""

import matplotlib.pyplot as plt
import seaborn as sns
import cv2
import numpy as np
from pathlib import Path
import json
import pandas as pd

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

def visualize_training_results(results_dir):
    """
    Visualize training metrics from YOLOv8 results
    
    Args:
        results_dir: Directory containing training results
    """
    
    print("\n" + "=" * 60)
    print("Generating Training Visualizations")
    print("=" * 60)
    
    results_path = Path(results_dir)
    
    # Create output directory for visualizations
    viz_dir = results_path.parent.parent.parent / 'visualizations'
    viz_dir.mkdir(exist_ok=True)
    
    # Check if results CSV exists
    results_csv = results_path / 'results.csv'
    if not results_csv.exists():
        print(f"⚠️  Results CSV not found at {results_csv}")
        return
    
    # Load training results
    df = pd.read_csv(results_csv)
    df.columns = df.columns.str.strip()  # Remove whitespace
    
    print(f"\n✓ Loaded training results: {len(df)} epochs")
    
    # Create comprehensive training visualization
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('YOLOv8 Training Metrics - Pedestrian and Car Detection', 
                 fontsize=16, fontweight='bold')
    
    # Plot 1: Loss curves
    ax = axes[0, 0]
    if 'train/box_loss' in df.columns:
        ax.plot(df.index, df['train/box_loss'], label='Box Loss', linewidth=2)
    if 'train/cls_loss' in df.columns:
        ax.plot(df.index, df['train/cls_loss'], label='Class Loss', linewidth=2)
    if 'train/dfl_loss' in df.columns:
        ax.plot(df.index, df['train/dfl_loss'], label='DFL Loss', linewidth=2)
    ax.set_xlabel('Epoch', fontsize=11)
    ax.set_ylabel('Loss', fontsize=11)
    ax.set_title('Training Losses', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Precision and Recall
    ax = axes[0, 1]
    if 'metrics/precision(B)' in df.columns:
        ax.plot(df.index, df['metrics/precision(B)'], 
                label='Precision', linewidth=2, color='blue')
    if 'metrics/recall(B)' in df.columns:
        ax.plot(df.index, df['metrics/recall(B)'], 
                label='Recall', linewidth=2, color='green')
    ax.set_xlabel('Epoch', fontsize=11)
    ax.set_ylabel('Score', fontsize=11)
    ax.set_title('Precision & Recall', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1])
    
    # Plot 3: mAP scores
    ax = axes[0, 2]
    if 'metrics/mAP50(B)' in df.columns:
        ax.plot(df.index, df['metrics/mAP50(B)'], 
                label='mAP@0.5', linewidth=2, color='purple')
    if 'metrics/mAP50-95(B)' in df.columns:
        ax.plot(df.index, df['metrics/mAP50-95(B)'], 
                label='mAP@0.5:0.95', linewidth=2, color='orange')
    ax.set_xlabel('Epoch', fontsize=11)
    ax.set_ylabel('mAP', fontsize=11)
    ax.set_title('Mean Average Precision', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1])
    
    # Plot 4: Validation losses
    ax = axes[1, 0]
    if 'val/box_loss' in df.columns:
        ax.plot(df.index, df['val/box_loss'], label='Val Box Loss', linewidth=2)
    if 'val/cls_loss' in df.columns:
        ax.plot(df.index, df['val/cls_loss'], label='Val Class Loss', linewidth=2)
    if 'val/dfl_loss' in df.columns:
        ax.plot(df.index, df['val/dfl_loss'], label='Val DFL Loss', linewidth=2)
    ax.set_xlabel('Epoch', fontsize=11)
    ax.set_ylabel('Loss', fontsize=11)
    ax.set_title('Validation Losses', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 5: Learning rate
    ax = axes[1, 1]
    if 'lr/pg0' in df.columns:
        ax.plot(df.index, df['lr/pg0'], linewidth=2, color='red')
    ax.set_xlabel('Epoch', fontsize=11)
    ax.set_ylabel('Learning Rate', fontsize=11)
    ax.set_title('Learning Rate Schedule', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Plot 6: Final metrics summary
    ax = axes[1, 2]
    if len(df) > 0:
        final_metrics = {
            'Precision': df['metrics/precision(B)'].iloc[-1] if 'metrics/precision(B)' in df.columns else 0,
            'Recall': df['metrics/recall(B)'].iloc[-1] if 'metrics/recall(B)' in df.columns else 0,
            'mAP@0.5': df['metrics/mAP50(B)'].iloc[-1] if 'metrics/mAP50(B)' in df.columns else 0,
            'mAP@0.5:0.95': df['metrics/mAP50-95(B)'].iloc[-1] if 'metrics/mAP50-95(B)' in df.columns else 0
        }
        bars = ax.bar(final_metrics.keys(), final_metrics.values(), 
                      color=['blue', 'green', 'purple', 'orange'])
        ax.set_ylabel('Score', fontsize=11)
        ax.set_title('Final Performance Metrics', fontsize=12, fontweight='bold')
        ax.set_ylim([0, 1])
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    
    # Save figure
    output_path = viz_dir / 'training_metrics.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Training visualization saved to: {output_path}")
    plt.close()
    
    return df

def create_detection_grid(test_results_dir, output_path, max_images=9):
    """
    Create a grid visualization of detection results
    
    Args:
        test_results_dir: Directory containing detection results
        output_path: Path to save grid visualization
        max_images: Maximum number of images to display
    """
    
    print("\n" + "=" * 60)
    print("Creating Detection Results Grid")
    print("=" * 60)
    
    test_path = Path(test_results_dir)
    
    # Get detected images
    detected_images = list(test_path.glob('detected_*.jpg'))
    if not detected_images:
        detected_images = list(test_path.glob('detected_*.png'))
    
    if len(detected_images) == 0:
        print("⚠️  No detection images found")
        return
    
    # Limit number of images
    detected_images = detected_images[:max_images]
    num_images = len(detected_images)
    
    # Create grid
    cols = 3
    rows = (num_images + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(18, 6*rows))
    fig.suptitle('YOLOv8 Detection Results - Test Dataset', 
                 fontsize=16, fontweight='bold')
    
    if rows == 1:
        axes = axes.reshape(1, -1)
    
    for idx, img_path in enumerate(detected_images):
        row = idx // cols
        col = idx % cols
        ax = axes[row, col]
        
        # Read and display image
        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        ax.imshow(img)
        ax.set_title(img_path.stem.replace('detected_', ''), fontsize=10)
        ax.axis('off')
    
    # Hide unused subplots
    for idx in range(num_images, rows * cols):
        row = idx // cols
        col = idx % cols
        axes[row, col].axis('off')
    
    plt.tight_layout()
    
    # Save
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Detection grid saved to: {output_path}")
    plt.close()

def analyze_detection_results(json_path, output_dir):
    """
    Analyze and visualize detection statistics
    
    Args:
        json_path: Path to detection results JSON
        output_dir: Directory to save visualizations
    """
    
    print("\n" + "=" * 60)
    print("Analyzing Detection Statistics")
    print("=" * 60)
    
    # Load detection results
    with open(json_path, 'r') as f:
        results = json.load(f)
    
    # Extract statistics
    class_counts = {'car': 0, 'pedestrains-and-cars': 0, 'person': 0}
    confidence_scores = []
    detections_per_image = []
    
    for result in results:
        num_dets = len(result['detections'])
        detections_per_image.append(num_dets)
        
        for det in result['detections']:
            class_name = det['class_name']
            if class_name in class_counts:
                class_counts[class_name] += 1
            confidence_scores.append(det['confidence'])
    
    # Create visualizations
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('Detection Statistics Analysis', fontsize=16, fontweight='bold')
    
    # Plot 1: Class distribution
    ax = axes[0]
    bars = ax.bar(class_counts.keys(), class_counts.values(), 
                  color=['skyblue', 'lightcoral', 'lightgreen'])
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title('Detections by Class', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{int(height)}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # Plot 2: Confidence distribution
    ax = axes[1]
    ax.hist(confidence_scores, bins=20, color='purple', alpha=0.7, edgecolor='black')
    ax.set_xlabel('Confidence Score', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title('Confidence Score Distribution', fontsize=13, fontweight='bold')
    ax.axvline(np.mean(confidence_scores), color='red', linestyle='--', 
               linewidth=2, label=f'Mean: {np.mean(confidence_scores):.3f}')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # Plot 3: Detections per image
    ax = axes[2]
    ax.hist(detections_per_image, bins=range(max(detections_per_image)+2), 
            color='orange', alpha=0.7, edgecolor='black')
    ax.set_xlabel('Number of Detections', fontsize=12)
    ax.set_ylabel('Number of Images', fontsize=12)
    ax.set_title('Detections per Image', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    # Save
    output_path = Path(output_dir) / 'detection_statistics.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Detection statistics saved to: {output_path}")
    plt.close()
    
    # Print summary
    print("\n📊 Detection Summary:")
    print(f"   Total detections: {sum(class_counts.values())}")
    print(f"   Average confidence: {np.mean(confidence_scores):.3f}")
    print(f"   Average detections per image: {np.mean(detections_per_image):.2f}")
    print(f"\n   Class breakdown:")
    for cls, count in class_counts.items():
        print(f"     - {cls}: {count}")

def main():
    """Main visualization function"""
    
    print("\n" + "=" * 80)
    print(" " * 25 + "VISUALIZATION PIPELINE")
    print("=" * 80)
    
    current_dir = Path(__file__).parent.absolute()
    
    # Paths
    training_results = current_dir / 'runs' / 'detect' / 'pedestrian_car_detection'
    test_results = current_dir / 'test_results'
    viz_dir = current_dir / 'visualizations'
    viz_dir.mkdir(exist_ok=True)
    
    # 1. Visualize training results
    if training_results.exists():
        print("\n[1/3] Visualizing training metrics...")
        visualize_training_results(training_results)
    else:
        print(f"\n⚠️  Training results not found at {training_results}")
    
    # 2. Create detection grid
    if test_results.exists():
        print("\n[2/3] Creating detection results grid...")
        grid_output = viz_dir / 'detection_grid.png'
        create_detection_grid(test_results, grid_output)
    else:
        print(f"\n⚠️  Test results not found at {test_results}")
    
    # 3. Analyze detection statistics
    json_path = test_results / 'detection_results.json'
    if json_path.exists():
        print("\n[3/3] Analyzing detection statistics...")
        analyze_detection_results(json_path, viz_dir)
    else:
        print(f"\n⚠️  Detection results JSON not found at {json_path}")
    
    print("\n" + "=" * 80)
    print("Visualization pipeline completed!")
    print("=" * 80)
    print(f"\nAll visualizations saved to: {viz_dir}")
    print("=" * 80 + "\n")

if __name__ == "__main__":
    main()
