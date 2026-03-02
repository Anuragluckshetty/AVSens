"""
Master Execution Script - Run Complete YOLOv8 Pipeline
This script executes all steps of the YOLOv8 training and evaluation pipeline
"""

import subprocess
import sys
from pathlib import Path
import time

class PipelineRunner:
    """Execute complete YOLOv8 pipeline"""
    
    def __init__(self):
        self.current_dir = Path(__file__).parent.absolute()
        self.steps_completed = []
        
    def print_header(self, message):
        """Print formatted header"""
        print("\n" + "=" * 80)
        print(f" {message}")
        print("=" * 80 + "\n")
    
    def run_script(self, script_name, description):
        """Run a Python script and track execution"""
        
        self.print_header(f"STEP {len(self.steps_completed) + 1}: {description}")
        
        script_path = self.current_dir / script_name
        
        if not script_path.exists():
            print(f"❌ Error: Script not found: {script_path}")
            return False
        
        print(f"Executing: {script_name}")
        print(f"Started at: {time.strftime('%H:%M:%S')}")
        print("-" * 80)
        
        start_time = time.time()
        
        try:
            # Run the script
            result = subprocess.run(
                [sys.executable, str(script_path)],
                capture_output=False,
                text=True,
                check=True
            )
            
            elapsed_time = time.time() - start_time
            
            print("-" * 80)
            print(f"✅ Completed successfully in {elapsed_time:.1f} seconds")
            
            self.steps_completed.append({
                'script': script_name,
                'description': description,
                'time': elapsed_time
            })
            
            return True
            
        except subprocess.CalledProcessError as e:
            elapsed_time = time.time() - start_time
            print("-" * 80)
            print(f"❌ Failed after {elapsed_time:.1f} seconds")
            print(f"Error: {e}")
            return False
        except KeyboardInterrupt:
            print("\n\n⚠️  Pipeline interrupted by user")
            return False
    
    def run_pipeline(self, skip_setup=False, skip_training=False):
        """Run complete pipeline"""
        
        print("\n" + "=" * 80)
        print(" " * 15 + "YOLOV8 COMPLETE PIPELINE EXECUTION")
        print(" " * 10 + "Pedestrian and Car Detection - Internship Assignment")
        print("=" * 80)
        
        print("\nThis will execute the following steps:")
        print("  1. Environment Setup (install dependencies)")
        print("  2. Model Training (1-3 hours depending on hardware)")
        print("  3. Model Evaluation (test dataset)")
        print("  4. Result Visualization (generate plots)")
        print("  5. Performance Analysis (detailed metrics)")
        print("  6. Report Generation (final report)")
        
        print("\n" + "-" * 80)
        input("\nPress ENTER to start the pipeline (or Ctrl+C to cancel)...")
        
        # Step 1: Environment Setup
        if not skip_setup:
            success = self.run_script(
                'setup_environment.py',
                'Environment Setup - Installing Dependencies'
            )
            if not success:
                print("\n⚠️  Environment setup failed. Please resolve errors before continuing.")
                return False
        else:
            print("\n⚠️  Skipping environment setup as requested")
        
        # Step 2: Model Training
        if not skip_training:
            print("\n" + "⚠️ " * 20)
            print("WARNING: Training may take 1-3 hours depending on your hardware!")
            print("You can interrupt training with Ctrl+C and resume later.")
            print("⚠️ " * 20)
            
            response = input("\nProceed with training? (yes/no): ").strip().lower()
            
            if response in ['yes', 'y']:
                success = self.run_script(
                    'train_yolov8.py',
                    'Model Training - Training YOLOv8 on Dataset'
                )
                if not success:
                    print("\n⚠️  Training failed or was interrupted.")
                    print("You can resume by running: python train_yolov8.py")
                    return False
            else:
                print("\n⚠️  Skipping training. Using existing model if available.")
        else:
            print("\n⚠️  Skipping training as requested")
        
        # Step 3: Model Evaluation
        success = self.run_script(
            'evaluate_model.py',
            'Model Evaluation - Testing on Separate Test Dataset'
        )
        if not success:
            print("\n⚠️  Evaluation failed. Make sure training completed successfully.")
            response = input("Continue anyway? (yes/no): ").strip().lower()
            if response not in ['yes', 'y']:
                return False
        
        # Step 4: Visualization
        success = self.run_script(
            'visualize_results.py',
            'Result Visualization - Generating Plots and Charts'
        )
        if not success:
            print("\n⚠️  Visualization failed. Plots may be incomplete.")
            response = input("Continue anyway? (yes/no): ").strip().lower()
            if response not in ['yes', 'y']:
                return False
        
        # Step 5: Performance Analysis
        success = self.run_script(
            'analyze_performance.py',
            'Performance Analysis - Detailed Metrics and Failure Cases'
        )
        if not success:
            print("\n⚠️  Analysis failed. Analysis may be incomplete.")
            response = input("Continue anyway? (yes/no): ").strip().lower()
            if response not in ['yes', 'y']:
                return False
        
        # Step 6: Report Generation
        success = self.run_script(
            'generate_report.py',
            'Report Generation - Creating Final Submission Report'
        )
        if not success:
            print("\n⚠️  Report generation failed.")
            return False
        
        return True
    
    def print_summary(self):
        """Print pipeline execution summary"""
        
        self.print_header("PIPELINE EXECUTION SUMMARY")
        
        total_time = sum(step['time'] for step in self.steps_completed)
        
        print(f"Total steps completed: {len(self.steps_completed)}/6")
        print(f"Total execution time: {total_time/60:.1f} minutes ({total_time:.0f} seconds)")
        print("\nStep breakdown:")
        
        for i, step in enumerate(self.steps_completed, 1):
            print(f"  {i}. {step['description']}")
            print(f"     Time: {step['time']:.1f}s")
        
        print("\n" + "=" * 80)
        print(" " * 25 + "PIPELINE COMPLETED!")
        print("=" * 80)
        
        print("\n📂 Generated Outputs:")
        print("   ✓ Trained model: runs/detect/pedestrian_car_detection/weights/best.pt")
        print("   ✓ Test results: test_results/")
        print("   ✓ Visualizations: visualizations/")
        print("   ✓ Final report: INTERNSHIP_ASSIGNMENT_REPORT.md")
        
        print("\n📝 Next Steps:")
        print("   1. Review the final report: INTERNSHIP_ASSIGNMENT_REPORT.md")
        print("   2. Check visualizations in: visualizations/")
        print("   3. Examine test results in: test_results/")
        print("   4. Submit the report for your internship assignment")
        
        print("\n" + "=" * 80)

def main():
    """Main execution function"""
    
    # Parse command line arguments
    skip_setup = '--skip-setup' in sys.argv
    skip_training = '--skip-training' in sys.argv
    
    # Create and run pipeline
    runner = PipelineRunner()
    
    try:
        success = runner.run_pipeline(
            skip_setup=skip_setup,
            skip_training=skip_training
        )
        
        if success:
            runner.print_summary()
        else:
            print("\n" + "=" * 80)
            print(" " * 25 + "PIPELINE INCOMPLETE")
            print("=" * 80)
            print("\nSome steps failed or were skipped.")
            print("You can run individual scripts manually:")
            print("  - python setup_environment.py")
            print("  - python train_yolov8.py")
            print("  - python evaluate_model.py")
            print("  - python visualize_results.py")
            print("  - python analyze_performance.py")
            print("  - python generate_report.py")
            print("=" * 80 + "\n")
    
    except KeyboardInterrupt:
        print("\n\n" + "=" * 80)
        print(" " * 25 + "PIPELINE INTERRUPTED")
        print("=" * 80)
        print("\nPipeline was interrupted by user.")
        print(f"Completed {len(runner.steps_completed)} steps before interruption.")
        print("\nYou can resume by running individual scripts.")
        print("=" * 80 + "\n")

if __name__ == "__main__":
    print("""
╔════════════════════════════════════════════════════════════════════════════╗
║                                                                            ║
║              YOLOv8 Pedestrian and Car Detection Pipeline                 ║
║                    Complete Internship Assignment                         ║
║                                                                            ║
╚════════════════════════════════════════════════════════════════════════════╝

Usage:
  python run_pipeline.py                  # Run complete pipeline
  python run_pipeline.py --skip-setup     # Skip environment setup
  python run_pipeline.py --skip-training  # Skip training (use existing model)

Note: Training step may take 1-3 hours depending on your hardware.
      You can skip training if model is already trained.
    """)
    
    main()
