# YOLOv8 Quick Reference Guide

## 🚀 Fast Track - Run Everything

```bash
# Option 1: Run complete pipeline (recommended)
python run_pipeline.py

# Option 2: Run step by step
python setup_environment.py      # Install dependencies
python train_yolov8.py           # Train model (1-3 hours)
python evaluate_model.py         # Evaluate on test set
python visualize_results.py      # Generate visualizations
python analyze_performance.py    # Analyze performance
python generate_report.py        # Generate final report
```

## 📝 Skip Options

```bash
# Skip environment setup (if already done)
python run_pipeline.py --skip-setup

# Skip training (use existing model)
python run_pipeline.py --skip-training

# Both
python run_pipeline.py --skip-setup --skip-training
```

## 📊 Key Files & Outputs

### Inputs
- `data.yaml` - Dataset configuration
- `train/`, `valid/`, `test/` - Dataset folders

### Training Outputs
- `runs/detect/pedestrian_car_detection/weights/best.pt` - Best model
- `runs/detect/pedestrian_car_detection/results.csv` - Training metrics
- `runs/detect/pedestrian_car_detection/*.png` - Training plots

### Evaluation Outputs
- `test_results/detected_*.jpg` - Images with detections
- `test_results/detection_results.json` - Detection data

### Visualizations
- `visualizations/training_metrics.png` - Training curves
- `visualizations/detection_grid.png` - Detection samples
- `visualizations/detection_statistics.png` - Statistics

### Final Report
- `INTERNSHIP_ASSIGNMENT_REPORT.md` - Complete report for submission

## 🎯 Expected Performance

| Metric | Expected Range | Your Result |
|--------|---------------|-------------|
| Precision | 0.70 - 0.80 | _______ |
| Recall | 0.65 - 0.75 | _______ |
| mAP@0.5 | 0.70 - 0.75 | _______ |
| mAP@0.5:0.95 | 0.40 - 0.50 | _______ |

## ⏱️ Estimated Time

| Step | Time | Can Skip? |
|------|------|-----------|
| Environment Setup | 5-10 min | Yes (if done before) |
| Training | 1-3 hours | Yes (if model exists) |
| Evaluation | 2-5 min | No |
| Visualization | 1-2 min | No |
| Analysis | < 1 min | No |
| Report | < 1 min | No |

**Total**: ~1.5 - 3.5 hours (first run)

## 🔧 Customization Quick Tips

### Change Model Size
In `train_yolov8.py` line 95:
```python
model_size='n'  # Options: 'n', 's', 'm', 'l', 'x'
```

### Change Training Epochs
In `train_yolov8.py` line 96:
```python
epochs=50  # Try 100 for better results
```

### Change Image Size
In `train_yolov8.py` line 97:
```python
img_size=640  # Try 1280 for better small object detection
```

### Change Batch Size
In `train_yolov8.py` line 98:
```python
batch_size=16  # Reduce to 8 or 4 if GPU memory error
```

### Change Confidence Threshold
In `evaluate_model.py` line 32:
```python
conf=0.25  # Lower = more detections, higher = fewer but more confident
```

## 🐛 Common Issues & Solutions

### Issue: CUDA out of memory
```python
# In train_yolov8.py, reduce batch size:
batch_size=8  # or even 4
```

### Issue: Training too slow
```python
# Use smaller model:
model_size='n'  # Fastest

# Or fewer epochs:
epochs=30
```

### Issue: Model not found for evaluation
```bash
# Make sure training completed:
ls runs/detect/pedestrian_car_detection/weights/best.pt

# If not, run training:
python train_yolov8.py
```

### Issue: No GPU detected
```python
# Check CUDA:
import torch
print(torch.cuda.is_available())

# If False, training will use CPU (slower but works)
```

## 📈 Improving Results

### Quick Wins
1. **Upgrade model**: Change `model_size='s'` (small instead of nano)
2. **More epochs**: Change `epochs=100`
3. **Higher resolution**: Change `img_size=1280`

### Expected Improvement
- Nano → Small: +5-8% mAP
- 50 → 100 epochs: +2-4% mAP
- 640 → 1280 resolution: +3-5% mAP

## 📚 Understanding Metrics

### Precision
**High (>0.8)**: Few false alarms ✅  
**Low (<0.6)**: Many incorrect detections ❌

### Recall
**High (>0.8)**: Finds most objects ✅  
**Low (<0.6)**: Misses many objects ❌

### mAP@0.5
**>0.7**: Excellent 🌟  
**0.5-0.7**: Good ✅  
**<0.5**: Needs work ⚠️

### mAP@0.5:0.95
**>0.5**: Excellent localization 🌟  
**0.3-0.5**: Good localization ✅  
**<0.3**: Poor localization ⚠️

## 📋 Submission Checklist

- [ ] Run complete pipeline
- [ ] Review `INTERNSHIP_ASSIGNMENT_REPORT.md`
- [ ] Check visualizations in `visualizations/`
- [ ] Verify test results in `test_results/`
- [ ] Confirm all metrics are reasonable
- [ ] Read through report sections
- [ ] Make any final customizations
- [ ] Submit report

## 🎓 Report Sections

Your final report includes:

1. ✅ Introduction & Objectives
2. ✅ Dataset Description
3. ✅ Annotation Format
4. ✅ Model Training Process
5. ✅ Evaluation Metrics
6. ✅ Performance Analysis
7. ✅ Failure Cases Discussion
8. ✅ Improvement Recommendations
9. ✅ CNN & Max Pooling Explanation
10. ✅ Conclusion
11. ✅ Technical Appendix

## 🚨 Emergency Commands

### Start Over
```bash
# Delete training outputs
rm -rf runs/

# Delete test results
rm -rf test_results/

# Delete visualizations
rm -rf visualizations/

# Delete generated files
rm data_updated.yaml
rm INTERNSHIP_ASSIGNMENT_REPORT.md

# Retrain
python run_pipeline.py
```

### Quick Test (No Training)
```bash
# Skip setup and training, use existing model
python run_pipeline.py --skip-setup --skip-training
```

### Just Generate Report (Everything Else Done)
```bash
python generate_report.py
```

## 💡 Pro Tips

1. **Save GPU memory**: Use batch_size=8 or 4
2. **Faster training**: Use model_size='n' (nano)
3. **Better accuracy**: Use model_size='s' or 'm' + more epochs
4. **Monitor training**: Watch `runs/detect/pedestrian_car_detection/` for plots
5. **Check progress**: Training prints metrics every epoch
6. **Early stopping**: Training stops automatically if no improvement

## 📞 Help Resources

- **Ultralytics Docs**: https://docs.ultralytics.com/
- **GitHub Issues**: https://github.com/ultralytics/ultralytics/issues
- **PyTorch Docs**: https://pytorch.org/docs/

## ⚡ Quick Commands Reference

```bash
# Complete pipeline
python run_pipeline.py

# Individual steps
python setup_environment.py
python train_yolov8.py
python evaluate_model.py
python visualize_results.py
python analyze_performance.py
python generate_report.py

# Skip options
python run_pipeline.py --skip-setup
python run_pipeline.py --skip-training

# Check outputs
ls runs/detect/pedestrian_car_detection/weights/
ls test_results/
ls visualizations/
cat INTERNSHIP_ASSIGNMENT_REPORT.md
```

---

**Ready?** Start with:
```bash
python run_pipeline.py
```

Good luck! 🚀
