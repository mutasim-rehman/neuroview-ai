# Brain Tumor Classification Training

This guide explains how to train a model for brain tumor classification using the `main_train_diseases.py` script.

## Dataset Structure

The script expects the following directory structure:

```
D:\Medical Imagining (CT scan, MRI, X-ray, and Microscopic Imagery) Data\
  Medical Imagining (CT scan, MRI, X-ray, and Microscopic Imagery) Data\
    Medical Imagining\
      Brain Tumor Classification\
        Testing\
          glioma\          (300 images)
          meningioma\      (306 images)
          notumor\         (405 images)
          pituitary\       (300 images)
```

## Classes

The model trains on 4 classes:
- **glioma** (300 images)
- **meningioma** (306 images)
- **notumor** (405 images)
- **pituitary** (300 images)

Total: **1,311 images**

## Usage

### Basic Training

Train with default settings (custom CNN model):

```bash
cd ai-training
python main_train_diseases.py
```

### Training Options

```bash
# Specify number of epochs
python main_train_diseases.py --epochs 100

# Specify batch size
python main_train_diseases.py --batch-size 16

# Use ResNet model (with pretrained weights)
python main_train_diseases.py --model-type resnet --pretrained

# Use ResNet model (without pretrained weights)
python main_train_diseases.py --model-type resnet

# Resume from checkpoint
python main_train_diseases.py --resume checkpoints/checkpoint_epoch_20.pth

# Specify GPU device
python main_train_diseases.py --gpu 0
```

### All Options

- `--epochs`: Number of training epochs (default: 50)
- `--batch-size`: Batch size for training (default: 32)
- `--model-type`: Model architecture - 'custom' or 'resnet' (default: 'custom')
- `--pretrained`: Use pretrained ResNet weights (only for ResNet model)
- `--resume`: Path to checkpoint to resume training from
- `--gpu`: GPU device ID to use

## Model Types

### Custom CNN Model
- Lightweight custom architecture
- Good for limited resources
- Trains from scratch
- Default option

### ResNet Model
- Uses ResNet-50 backbone
- Can use pretrained ImageNet weights (recommended)
- Better performance with pretrained weights
- Requires more memory

## Training Output

The training process creates:

1. **Checkpoints**: Saved in `./checkpoints/`
   - `best_model.pth` - Best model based on validation loss
   - `checkpoint_epoch_N.pth` - Periodic checkpoints

2. **Logs**: Saved in `./logs/`
   - Training logs with timestamps
   - TensorBoard logs for visualization

3. **Results**: Saved in `./outputs/`
   - `test_results_classification.json` - Test metrics
   - `classification_report.txt` - Detailed classification report

## Evaluation Metrics

The model reports:
- **Accuracy**: Overall classification accuracy
- **Precision**: Per-class and average precision
- **Recall**: Per-class and average recall
- **F1-Score**: Per-class and average F1-score
- **Confusion Matrix**: Detailed class-wise performance

## Data Split

By default, the dataset is split as:
- **Training**: 70% (stratified by class)
- **Validation**: 15% (stratified by class)
- **Testing**: 15% (stratified by class)

This ensures each split maintains the class distribution.

## Requirements

All dependencies are listed in `requirements.txt`. Key packages:
- PyTorch >= 2.0.0
- torchvision >= 0.15.0
- Pillow >= 10.0.0 (for image loading)
- scikit-learn >= 1.3.0 (for metrics)
- tqdm, tensorboard, numpy, etc.

## Notes

- The script automatically detects and handles image formats (jpg, png, bmp, tiff)
- Images are resized to 224x224 pixels
- Data augmentation is applied during training (rotation, flipping, color jitter)
- Training progress is displayed with progress bars
- Best model is automatically saved based on validation loss

