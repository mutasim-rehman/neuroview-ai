# AI Training - Healthy Brain Scan Analysis

This directory contains the AI/ML training pipeline for analyzing healthy brain scans (T1-weighted MRI).

## Project Structure

```
ai-training/
├── main_train_healthy.py      # Main entry point - runs entire training pipeline
├── requirements.txt            # Python dependencies
├── config/
│   └── config.py              # Training configuration and hyperparameters
├── data/
│   ├── data_loader.py         # Dataset loading and splitting (70/30)
│   └── preprocessing.py       # NIfTI preprocessing and augmentation
├── models/
│   └── brain_model.py         # 3D CNN model architecture
├── training/
│   ├── trainer.py             # Training loop and optimization
│   └── evaluator.py           # Validation and testing evaluation
└── utils/
    └── helpers.py             # Utility functions (logging, metrics, etc.)
```

## Dataset

- **Location**: `D:\IXI-T1`
- **Format**: Zipped `.nii` files (582 total scans)
- **Type**: T1-weighted healthy brain MRI scans
- **Split**: 70% training (407 scans), 30% testing (175 scans)

## Setup

1. **Install dependencies**:
   ```bash
   cd ai-training
   pip install -r requirements.txt
   ```

2. **Verify dataset location**:
   - Update `config/config.py` if your dataset path differs

3. **Run training**:
   ```bash
   python main_train_healthy.py
   ```

## Training Process

1. **Data Loading**: Loads and extracts all `.nii.gz` files from dataset directory
2. **Preprocessing**: Normalizes, resizes, and augments each scan
3. **Splitting**: Randomly splits 70/30 train/test
4. **Training**: Trains 3D CNN on each scan's layers
5. **Validation**: Evaluates on test set and reports metrics
6. **Model Saving**: Saves best model checkpoints

## Model Architecture

- 3D Convolutional Neural Network
- Processes full 3D volumes layer by layer
- Self-supervised learning on healthy brain patterns
- Outputs: reconstruction/representation for anomaly detection

## Features

- ✅ Efficient batch processing for large volumes
- ✅ Memory-efficient data loading
- ✅ Automatic train/test split (70/30)
- ✅ Progress tracking and logging
- ✅ Model checkpointing
- ✅ GPU support (CUDA)

