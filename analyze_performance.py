"""
Performance Analysis Script for YOLOv8 Pedestrian and Car Detection
Provides detailed analysis of model performance, failure cases, and improvement strategies
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

class PerformanceAnalyzer:
    """Analyze YOLOv8 model performance and generate detailed reports"""
    
    def __init__(self, results_dir, test_results_dir):
        """
        Initialize analyzer
        
        Args:
            results_dir: Directory with training results
            test_results_dir: Directory with test results
        """
        self.results_dir = Path(results_dir)
        self.test_results_dir = Path(test_results_dir)
        self.training_metrics = None
        self.test_detections = None
        
    def load_data(self):
        """Load training and test data"""
        
        print("\nLoading performance data...")
        
        # Load training results
        results_csv = self.results_dir / 'results.csv'
        if results_csv.exists():
            self.training_metrics = pd.read_csv(results_csv)
            self.training_metrics.columns = self.training_metrics.columns.str.strip()
            print(f"✓ Loaded training metrics: {len(self.training_metrics)} epochs")
        else:
            print(f"⚠️  Training results not found")
        
        # Load test detections
        json_path = self.test_results_dir / 'detection_results.json'
        if json_path.exists():
            with open(json_path, 'r') as f:
                self.test_detections = json.load(f)
            print(f"✓ Loaded test detections: {len(self.test_detections)} images")
        else:
            print(f"⚠️  Test detections not found")
    
    def analyze_metrics(self):
        """Analyze and interpret performance metrics"""
        
        print("\n" + "=" * 80)
        print(" " * 25 + "PERFORMANCE METRICS ANALYSIS")
        print("=" * 80)
        
        if self.training_metrics is None or len(self.training_metrics) == 0:
            print("No training metrics available")
            return {}
        
        # Get final epoch metrics
        final_metrics = {}
        metric_cols = {
            'precision': 'metrics/precision(B)',
            'recall': 'metrics/recall(B)',
            'mAP50': 'metrics/mAP50(B)',
            'mAP50-95': 'metrics/mAP50-95(B)'
        }
        
        for name, col in metric_cols.items():
            if col in self.training_metrics.columns:
                final_metrics[name] = self.training_metrics[col].iloc[-1]
        
        # Print metrics interpretation
        print("\n📊 FINAL PERFORMANCE METRICS:")
        print("-" * 80)
        
        for metric, value in final_metrics.items():
            print(f"\n{metric.upper()}: {value:.4f}")
            self._interpret_metric(metric, value)
        
        return final_metrics
    
    def _interpret_metric(self, metric_name, value):
        """Provide interpretation of metric values"""
        
        if metric_name == 'precision':
            print(f"   → Precision measures how many predicted detections are correct.")
            if value >= 0.8:
                print(f"   → {value:.1%} of predictions are true positives (EXCELLENT)")
                print(f"   → Low false positive rate - model rarely misidentifies objects")
            elif value >= 0.6:
                print(f"   → {value:.1%} of predictions are true positives (GOOD)")
                print(f"   → Moderate false positives - some incorrect detections")
            else:
                print(f"   → {value:.1%} of predictions are true positives (NEEDS IMPROVEMENT)")
                print(f"   → High false positive rate - many incorrect detections")
        
        elif metric_name == 'recall':
            print(f"   → Recall measures how many actual objects are detected.")
            if value >= 0.8:
                print(f"   → {value:.1%} of ground truth objects detected (EXCELLENT)")
                print(f"   → Low miss rate - model rarely misses objects")
            elif value >= 0.6:
                print(f"   → {value:.1%} of ground truth objects detected (GOOD)")
                print(f"   → Moderate miss rate - some objects not detected")
            else:
                print(f"   → {value:.1%} of ground truth objects detected (NEEDS IMPROVEMENT)")
                print(f"   → High miss rate - many objects not detected")
        
        elif metric_name == 'mAP50':
            print(f"   → mAP@0.5 measures average precision at 50% IoU threshold.")
            if value >= 0.7:
                print(f"   → Score of {value:.1%} indicates STRONG overall performance")
                print(f"   → Model effectively localizes and classifies objects")
            elif value >= 0.5:
                print(f"   → Score of {value:.1%} indicates MODERATE performance")
                print(f"   → Room for improvement in localization or classification")
            else:
                print(f"   → Score of {value:.1%} indicates WEAK performance")
                print(f"   → Significant improvement needed in detection accuracy")
        
        elif metric_name == 'mAP50-95':
            print(f"   → mAP@0.5:0.95 measures precision across multiple IoU thresholds.")
            if value >= 0.5:
                print(f"   → Score of {value:.1%} indicates EXCELLENT localization")
                print(f"   → Bounding boxes are very accurate and well-aligned")
            elif value >= 0.3:
                print(f"   → Score of {value:.1%} indicates GOOD localization")
                print(f"   → Bounding boxes are reasonably accurate")
            else:
                print(f"   → Score of {value:.1%} indicates POOR localization")
                print(f"   → Bounding boxes need better alignment with ground truth")
    
    def identify_failure_cases(self):
        """Identify and analyze potential failure cases"""
        
        print("\n" + "=" * 80)
        print(" " * 25 + "FAILURE CASE ANALYSIS")
        print("=" * 80)
        
        if self.test_detections is None:
            print("No test detection data available")
            return
        
        failure_analysis = {
            'low_confidence': [],
            'no_detections': [],
            'potential_false_positives': [],
            'crowded_scenes': []
        }
        
        for result in self.test_detections:
            image_name = result['image']
            detections = result['detections']
            
            # No detections (potential false negatives)
            if len(detections) == 0:
                failure_analysis['no_detections'].append(image_name)
            
            # Low confidence detections
            low_conf = [d for d in detections if d['confidence'] < 0.4]
            if low_conf:
                failure_analysis['low_confidence'].append({
                    'image': image_name,
                    'count': len(low_conf),
                    'avg_conf': np.mean([d['confidence'] for d in low_conf])
                })
            
            # Potential false positives (unusual confidence patterns)
            if len(detections) > 0:
                confidences = [d['confidence'] for d in detections]
                if max(confidences) < 0.5:
                    failure_analysis['potential_false_positives'].append(image_name)
            
            # Crowded scenes (many detections)
            if len(detections) > 5:
                failure_analysis['crowded_scenes'].append({
                    'image': image_name,
                    'num_detections': len(detections)
                })
        
        # Print failure analysis
        print("\n🔍 IDENTIFIED FAILURE PATTERNS:")
        print("-" * 80)
        
        # Missing detections
        print(f"\n1. MISSING DETECTIONS (False Negatives):")
        print(f"   Images with no detections: {len(failure_analysis['no_detections'])}")
        if failure_analysis['no_detections']:
            print(f"   Examples: {', '.join(failure_analysis['no_detections'][:3])}")
            print("   Possible reasons:")
            print("     - Objects too small or occluded")
            print("     - Poor lighting or image quality")
            print("     - Objects outside training distribution")
            print("     - Confidence threshold too high")
        
        # Low confidence
        print(f"\n2. LOW CONFIDENCE DETECTIONS:")
        print(f"   Images with low-confidence detections: {len(failure_analysis['low_confidence'])}")
        if failure_analysis['low_confidence']:
            avg_low_conf = np.mean([x['avg_conf'] for x in failure_analysis['low_confidence']])
            print(f"   Average confidence: {avg_low_conf:.3f}")
            print("   Possible reasons:")
            print("     - Ambiguous or partially occluded objects")
            print("     - Unusual viewpoints or perspectives")
            print("     - Objects at image boundaries")
            print("     - Need more training data for these scenarios")
        
        # Potential false positives
        print(f"\n3. POTENTIAL FALSE POSITIVES:")
        print(f"   Images with uncertain detections: {len(failure_analysis['potential_false_positives'])}")
        if failure_analysis['potential_false_positives']:
            print(f"   Examples: {', '.join(failure_analysis['potential_false_positives'][:3])}")
            print("   Possible reasons:")
            print("     - Background objects confused as targets")
            print("     - Image artifacts or reflections")
            print("     - Similar-looking non-target objects")
            print("     - Need better negative examples in training")
        
        # Crowded scenes
        print(f"\n4. CROWDED SCENE CHALLENGES:")
        print(f"   Images with many objects: {len(failure_analysis['crowded_scenes'])}")
        if failure_analysis['crowded_scenes']:
            max_dets = max([x['num_detections'] for x in failure_analysis['crowded_scenes']])
            print(f"   Maximum detections in single image: {max_dets}")
            print("   Challenges:")
            print("     - Object overlap and occlusion")
            print("     - Non-maximum suppression may merge close boxes")
            print("     - Small objects in dense crowds")
            print("     - Potential duplicate detections")
        
        return failure_analysis
    
    def propose_improvements(self, final_metrics):
        """Propose practical improvements based on performance analysis"""
        
        print("\n" + "=" * 80)
        print(" " * 22 + "IMPROVEMENT RECOMMENDATIONS")
        print("=" * 80)
        
        improvements = []
        
        # Based on precision
        if 'precision' in final_metrics:
            if final_metrics['precision'] < 0.7:
                improvements.append({
                    'issue': 'Low Precision (High False Positives)',
                    'recommendations': [
                        'Increase confidence threshold (e.g., 0.4 → 0.5)',
                        'Add more negative examples to training data',
                        'Review and improve annotation quality',
                        'Use more aggressive NMS (Non-Maximum Suppression)',
                        'Add hard negative mining during training'
                    ]
                })
        
        # Based on recall
        if 'recall' in final_metrics:
            if final_metrics['recall'] < 0.7:
                improvements.append({
                    'issue': 'Low Recall (High False Negatives)',
                    'recommendations': [
                        'Lower confidence threshold to catch more objects',
                        'Increase training data diversity',
                        'Add more examples of difficult cases (small objects, occlusions)',
                        'Use multi-scale training and testing',
                        'Try data augmentation (mosaic, mixup, rotation)',
                        'Increase image resolution (640 → 1280)'
                    ]
                })
        
        # Based on mAP
        if 'mAP50-95' in final_metrics:
            if final_metrics['mAP50-95'] < 0.4:
                improvements.append({
                    'issue': 'Poor Localization (Low mAP@0.5:0.95)',
                    'recommendations': [
                        'Improve annotation accuracy - use better tools',
                        'Use polygon annotations instead of bounding boxes',
                        'Increase anchor box diversity',
                        'Train for more epochs with early stopping',
                        'Use a larger model (nano → small or medium)',
                        'Apply aspect ratio-preserving augmentation'
                    ]
                })
        
        # General improvements
        improvements.append({
            'issue': 'General Model Enhancement',
            'recommendations': [
                'Collect more diverse training data (various weather, lighting)',
                'Balance class distribution in dataset',
                'Use transfer learning with pretrained COCO weights',
                'Experiment with different YOLO model sizes',
                'Fine-tune hyperparameters (learning rate, batch size)',
                'Implement class-weighted loss for imbalanced classes',
                'Use TTA (Test Time Augmentation) for inference',
                'Ensemble multiple models for better performance'
            ]
        })
        
        # Print improvements
        for idx, improvement in enumerate(improvements, 1):
            print(f"\n{idx}. {improvement['issue'].upper()}")
            print("-" * 80)
            for rec in improvement['recommendations']:
                print(f"   ✓ {rec}")
        
        return improvements
    
    def explain_cnn_concepts(self):
        """Explain CNN and max pooling concepts"""
        
        print("\n" + "=" * 80)
        print(" " * 15 + "CONVOLUTIONAL NEURAL NETWORKS IN YOLO")
        print("=" * 80)
        
        print("\n📚 CONVOLUTIONAL LAYERS:")
        print("-" * 80)
        print("""
   Convolutional layers are the foundation of YOLO's feature extraction:
   
   1. LOCAL FEATURE DETECTION:
      - Small filters (e.g., 3x3 kernels) slide across the image
      - Each filter learns to detect specific patterns (edges, textures, shapes)
      - Early layers detect simple features (edges, corners)
      - Deeper layers detect complex features (wheels, faces, body parts)
   
   2. HIERARCHICAL LEARNING:
      - Layer 1: Edges and gradients
      - Layer 2: Simple shapes and textures
      - Layer 3: Object parts (windows, wheels, heads)
      - Layer 4+: Complete objects (cars, pedestrians)
   
   3. TRANSLATION INVARIANCE:
      - Same filter applied everywhere in image
      - Detects features regardless of position
      - Crucial for detecting objects anywhere in the frame
   
   4. PARAMETER EFFICIENCY:
      - Shared weights across spatial locations
      - Much fewer parameters than fully connected layers
      - Enables deep networks without overfitting
        """)
        
        print("\n📚 MAX POOLING:")
        print("-" * 80)
        print("""
   Max pooling downsamples feature maps for efficiency and robustness:
   
   1. SPATIAL DIMENSION REDUCTION:
      - Reduces feature map size (e.g., 2x2 pooling halves dimensions)
      - Makes computation faster and more memory-efficient
      - Example: 640x640 → 320x320 → 160x160 → ...
   
   2. INCREASED RECEPTIVE FIELD:
      - Each neuron "sees" a larger area of the original image
      - Helps detect objects at different scales
      - Higher layers have global view of the image
   
   3. TRANSLATION INVARIANCE:
      - Small shifts in object position don't change max value
      - Model becomes robust to minor position variations
      - Improves generalization to unseen data
   
   4. FEATURE SELECTION:
      - Keeps strongest activations (max values)
      - Filters out weak or noisy responses
      - Emphasizes most important features
   
   5. OVERFITTING PREVENTION:
      - Reduces number of parameters in subsequent layers
      - Acts as implicit regularization
      - Improves model generalization
        """)
        
        print("\n📚 HOW THEY WORK TOGETHER IN YOLO:")
        print("-" * 80)
        print("""
   YOLOv8 Architecture Pipeline:
   
   1. INPUT: 640x640 RGB image
   
   2. BACKBONE (Feature Extraction):
      - Conv → ReLU → Max Pool (detect low-level features)
      - Conv → ReLU → Max Pool (detect mid-level features)
      - Multiple Conv blocks (detect high-level features)
      - Creates multi-scale feature pyramid
   
   3. NECK (Feature Fusion):
      - Combines features from different scales
      - Path Aggregation Network (PANet)
      - Helps detect both small and large objects
   
   4. HEAD (Detection):
      - Predicts bounding boxes, class probabilities, confidence
      - Outputs: [x, y, w, h, confidence, class_scores]
      - Multiple detection heads for different scales
   
   5. POST-PROCESSING:
      - Non-Maximum Suppression (NMS) removes duplicates
      - Confidence filtering keeps high-quality detections
      - Final output: Bounding boxes with classes
        """)
    
    def generate_summary(self, final_metrics, failure_analysis):
        """Generate comprehensive performance summary"""
        
        print("\n" + "=" * 80)
        print(" " * 25 + "PERFORMANCE SUMMARY")
        print("=" * 80)
        
        # Overall assessment
        if 'mAP50' in final_metrics:
            mAP = final_metrics['mAP50']
            
            print("\n🎯 OVERALL MODEL ASSESSMENT:")
            print("-" * 80)
            
            if mAP >= 0.7:
                rating = "EXCELLENT"
                message = "Model performs very well and is ready for deployment"
            elif mAP >= 0.5:
                rating = "GOOD"
                message = "Model shows solid performance with room for improvement"
            elif mAP >= 0.3:
                rating = "FAIR"
                message = "Model needs optimization before production use"
            else:
                rating = "NEEDS IMPROVEMENT"
                message = "Model requires significant improvements"
            
            print(f"\n   Rating: {rating}")
            print(f"   Assessment: {message}")
            print(f"   mAP@0.5: {mAP:.1%}")
        
        # Strengths
        print("\n✅ MODEL STRENGTHS:")
        print("-" * 80)
        strengths = []
        
        if 'precision' in final_metrics and final_metrics['precision'] >= 0.7:
            strengths.append("High precision - few false positives")
        if 'recall' in final_metrics and final_metrics['recall'] >= 0.7:
            strengths.append("High recall - detects most objects")
        if 'mAP50-95' in final_metrics and final_metrics['mAP50-95'] >= 0.4:
            strengths.append("Good bounding box localization")
        
        if strengths:
            for strength in strengths:
                print(f"   ✓ {strength}")
        else:
            print("   Model shows basic detection capability but needs improvement")
        
        # Weaknesses
        print("\n⚠️  MODEL WEAKNESSES:")
        print("-" * 80)
        weaknesses = []
        
        if 'precision' in final_metrics and final_metrics['precision'] < 0.6:
            weaknesses.append("Low precision - many false positive detections")
        if 'recall' in final_metrics and final_metrics['recall'] < 0.6:
            weaknesses.append("Low recall - misses many objects")
        if 'mAP50-95' in final_metrics and final_metrics['mAP50-95'] < 0.3:
            weaknesses.append("Poor bounding box alignment")
        
        if failure_analysis:
            if len(failure_analysis.get('no_detections', [])) > 3:
                weaknesses.append("Struggles with certain object types or conditions")
            if len(failure_analysis.get('crowded_scenes', [])) > 0:
                weaknesses.append("Difficulty handling crowded scenes with many objects")
        
        if weaknesses:
            for weakness in weaknesses:
                print(f"   ✗ {weakness}")
        else:
            print("   No major weaknesses identified")

def main():
    """Main performance analysis function"""
    
    print("\n" + "=" * 80)
    print(" " * 20 + "YOLOV8 PERFORMANCE ANALYSIS")
    print(" " * 15 + "Detailed Model Evaluation and Insights")
    print("=" * 80)
    
    current_dir = Path(__file__).parent.absolute()
    
    # Initialize analyzer
    results_dir = current_dir / 'runs' / 'detect' / 'pedestrian_car_detection'
    test_results_dir = current_dir / 'test_results'
    
    analyzer = PerformanceAnalyzer(results_dir, test_results_dir)
    
    # Load data
    analyzer.load_data()
    
    # Perform analysis
    final_metrics = analyzer.analyze_metrics()
    failure_analysis = analyzer.identify_failure_cases()
    improvements = analyzer.propose_improvements(final_metrics)
    analyzer.explain_cnn_concepts()
    analyzer.generate_summary(final_metrics, failure_analysis)
    
    print("\n" + "=" * 80)
    print("Performance analysis completed!")
    print("=" * 80 + "\n")

if __name__ == "__main__":
    main()
