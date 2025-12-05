# Quick Start Guide

## Installation

1. **Navigate to the AI training directory**:
   ```bash
   cd ai-training
   ```

2. **Create a virtual environment** (recommended):
   ```bash
   python -m venv venv
   
   # On Windows:
   venv\Scripts\activate
   
   # On Linux/Mac:
   source venv/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Dataset Setup

1. **Verify dataset path**: 
   - Default path is set to `D:\IXI-T1` in `config/config.py`
   - Update `DATASET_PATH` if your dataset is in a different location

2. **Dataset structure**: 
   - The script will automatically find all `.nii` and `.nii.gz` files
   - Supports both compressed and uncompressed files
   - If files are in zip archives, they will be automatically extracted to a temp folder

## Running Training

### Basic Training

Run the default training configuration:
```bash
python main_train_healthy.py
```

### Custom Training

Override default parameters:
```bash
# Train for more epochs
python main_train_healthy.py --epochs 100

# Use smaller batch size (if running out of memory)
python main_train_healthy.py --batch-size 2

# Specify GPU device
python main_train_healthy.py --gpu 0

# Resume from checkpoint
python main_train_healthy.py --resume checkpoints/best_model.pth
```

## Training Process

The training pipeline will:

1. **Scan for NIfTI files** in the dataset directory
2. **Split dataset** into 70% training (407 scans) and 30% testing (175 scans)
3. **Preprocess volumes** (normalize, resize to 128×128×128)
4. **Train model** with automatic checkpointing every 5 epochs
5. **Evaluate** on test set and save results

## Output Files

After training, you'll find:

- `checkpoints/` - Model checkpoints
  - `best_model.pth` - Best model based on validation loss
  - `checkpoint_epoch_N.pth` - Checkpoints every 5 epochs
- `logs/` - Training logs and TensorBoard files
- `outputs/` - Configuration and test results
  - `config.json` - Training configuration used
  - `test_results.json` - Final evaluation metrics

## Monitoring Training

### TensorBoard

View training progress with TensorBoard:
```bash
tensorboard --logdir logs
```

Then open http://localhost:6006 in your browser.

### Log Files

Check the log files in `logs/` directory for detailed training information.

## Configuration

All training parameters can be modified in `config/config.py`:

- `DATASET_PATH`: Path to your dataset
- `BATCH_SIZE`: Batch size (default: 4)
- `NUM_EPOCHS`: Number of training epochs (default: 50)
- `LEARNING_RATE`: Learning rate (default: 1e-4)
- `TARGET_SHAPE`: Volume size after preprocessing (default: 128×128×128)

## Troubleshooting

### Out of Memory (OOM) Errors

1. Reduce `BATCH_SIZE` in config or via command line
2. Reduce `TARGET_SHAPE` (e.g., to (96, 96, 96))
3. Reduce `NUM_WORKERS` for data loading

### Slow Training

1. Ensure GPU is being used (check console output)
2. Increase `NUM_WORKERS` for faster data loading
3. Reduce `TARGET_SHAPE` for smaller volumes

### Dataset Not Found

1. Verify the path in `config/config.py`
2. Check that `.nii` or `.nii.gz` files exist in the directory
3. Ensure proper file permissions

## Next Steps

After training:

1. Use the trained model for inference on new scans
2. Fine-tune on specific tasks (anomaly detection, segmentation)
3. Export model for deployment

