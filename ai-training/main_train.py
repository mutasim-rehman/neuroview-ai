"""
Main training script for brain tumor classification.
Trains a model using training data split into 85% training and 15% validation.
"""

import os
import sys
import argparse
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

import torch
import numpy as np
import json

from config.config import config
from data.image_classification_loader import create_classification_dataloaders
from models.classification_model import create_classification_model
from training.classification_trainer import ClassificationTrainer
from training.classification_evaluator import ClassificationEvaluator
from utils.helpers import (
    setup_logger,
    save_config,
    get_device,
    count_parameters,
    load_checkpoint
)


def main():
    """Main training pipeline."""
    
    # Parse arguments
    parser = argparse.ArgumentParser(description='Train brain tumor classification model')
    parser.add_argument('--resume', type=str, default=None, help='Resume from checkpoint')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--gpu', type=int, default=None, help='GPU device ID')
    parser.add_argument('--model-type', type=str, default='custom', choices=['custom', 'resnet'],
                        help='Model type: custom or resnet')
    parser.add_argument('--pretrained', action='store_true', help='Use pretrained ResNet')
    args = parser.parse_args()
    
    # Dataset configuration - Training locations
    TRAINING_BASE_DIR = r"D:\Medical Imagining (CT scan, MRI, X-ray, and Microscopic Imagery) Data\Medical Imagining (CT scan, MRI, X-ray, and Microscopic Imagery) Data\Medical Imagining\Brain Tumor Classification\Training"
    CLASS_FOLDERS = ['glioma', 'meningioma', 'notumor', 'pituitary']
    
    # Override config with command line arguments
    num_epochs = args.epochs
    batch_size = args.batch_size
    
    # Setup logging
    logger = setup_logger("brain_tumor_training", config.LOG_DIR)
    logger.info("="*60)
    logger.info("Starting Brain Tumor Classification Training")
    logger.info("="*60)
    logger.info(f"Training data directory: {TRAINING_BASE_DIR}")
    logger.info(f"Classes: {CLASS_FOLDERS}")
    logger.info(f"Train/Validation split: 85% / 15%")
    logger.info(f"Model type: {args.model_type}")
    logger.info(f"Using pretrained: {args.pretrained if args.model_type == 'resnet' else False}")
    logger.info(f"Epochs: {num_epochs}")
    logger.info(f"Batch size: {batch_size}")
    
    # Save configuration
    save_config(config, config.OUTPUT_DIR)
    logger.info(f"Configuration saved to {config.OUTPUT_DIR}")
    
    # Set random seeds for reproducibility
    torch.manual_seed(config.RANDOM_SEED)
    np.random.seed(config.RANDOM_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config.RANDOM_SEED)
        torch.cuda.manual_seed_all(config.RANDOM_SEED)
    
    # Get device
    device = get_device(config.USE_GPU)
    if args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        device = torch.device(f'cuda:{args.gpu}')
        logger.info(f"Using GPU {args.gpu}")
    
    # Verify dataset directory exists
    if not os.path.exists(TRAINING_BASE_DIR):
        logger.error(f"Training directory does not exist: {TRAINING_BASE_DIR}")
        logger.error("Please update TRAINING_BASE_DIR in main_train.py")
        raise FileNotFoundError(f"Training directory not found: {TRAINING_BASE_DIR}")
    
    # Create data loaders with 85/15 split
    logger.info("Creating data loaders...")
    logger.info(f"Training data path: {TRAINING_BASE_DIR}")
    
    try:
        train_loader, val_loader, _, class_names = create_classification_dataloaders(
            base_dir=TRAINING_BASE_DIR,
            class_folders=CLASS_FOLDERS,
            train_split=0.85,  # 85% for training
            val_split=0.15,    # 15% for validation
            test_split=0.0,    # No test split - we'll use separate test data
            batch_size=batch_size,
            num_workers=config.NUM_WORKERS,
            pin_memory=config.PIN_MEMORY,
            target_size=(224, 224),
            normalize=True,
            use_augmentation=config.USE_AUGMENTATION,
            random_seed=config.RANDOM_SEED
        )
        logger.info(f"Data loaders created successfully")
        logger.info(f"  Training batches: {len(train_loader)}")
        logger.info(f"  Validation batches: {len(val_loader)}")
        logger.info(f"  Class names: {class_names}")
    except Exception as e:
        logger.error(f"Failed to create data loaders: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise
    
    # Create model
    logger.info("Creating model...")
    num_classes = len(CLASS_FOLDERS)
    
    model = create_classification_model(
        in_channels=3,  # RGB images
        num_classes=num_classes,
        base_channels=64,
        use_pretrained=args.pretrained if args.model_type == 'resnet' else False,
        model_type=args.model_type
    )
    
    num_params = count_parameters(model)
    logger.info(f"Model created: {num_params:,} parameters")
    logger.info(f"Number of classes: {num_classes}")
    
    # Resume from checkpoint if specified
    start_epoch = 0
    if args.resume:
        logger.info(f"Resuming from checkpoint: {args.resume}")
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.LEARNING_RATE)
        start_epoch = load_checkpoint(model, optimizer, args.resume, device)
        logger.info(f"Resumed from epoch {start_epoch}")
    
    # Create trainer
    logger.info("Initializing trainer...")
    trainer = ClassificationTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        num_classes=num_classes,
        learning_rate=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY,
        log_dir=config.LOG_DIR,
        checkpoint_dir=config.CHECKPOINT_DIR,
        log_every_n_batches=config.LOG_EVERY_N_BATCHES
    )
    
    # Train
    logger.info(f"Starting training for {num_epochs} epochs...")
    trainer.train(
        num_epochs=num_epochs,
        save_every_n_epochs=config.SAVE_EVERY_N_EPOCHS
    )
    
    # Load best model for evaluation on validation set
    logger.info("Loading best model for final evaluation on validation set...")
    best_model_path = os.path.join(config.CHECKPOINT_DIR, "best_model.pth")
    
    if os.path.exists(best_model_path):
        checkpoint = torch.load(best_model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        logger.info(f"Best model loaded from epoch {checkpoint['epoch'] + 1}")
        logger.info(f"Best validation loss: {checkpoint.get('loss', 'N/A')}")
    else:
        logger.warning("Best model checkpoint not found, using current model")
    
    # Evaluate on validation set (the 15% split)
    logger.info("Running final evaluation on validation set (15% split)...")
    
    evaluator = ClassificationEvaluator(
        model=model,
        test_loader=val_loader,
        device=device,
        class_names=class_names
    )
    
    val_metrics = evaluator.evaluate()
    evaluator.print_results(val_metrics)
    
    # Generate classification report
    classification_report_str = evaluator.generate_classification_report(val_metrics)
    logger.info("\nDetailed Classification Report (Validation Set - 15%):")
    logger.info(classification_report_str)
    
    # Save evaluation results
    results_path = os.path.join(config.OUTPUT_DIR, "training_validation_results.json")
    
    # Prepare metrics for JSON serialization
    json_metrics = {
        k: v for k, v in val_metrics.items()
        if k not in ['predictions', 'labels', 'probabilities']
    }
    json_metrics['confusion_matrix'] = val_metrics['confusion_matrix']
    
    with open(results_path, 'w') as f:
        json.dump(json_metrics, f, indent=2)
    logger.info(f"Validation results saved to {results_path}")
    
    # Save classification report
    report_path = os.path.join(config.OUTPUT_DIR, "training_validation_report.txt")
    with open(report_path, 'w') as f:
        f.write("Brain Tumor Classification Report - Training Validation Set (15%)\n")
        f.write("="*60 + "\n\n")
        f.write(f"Classes: {', '.join(class_names)}\n\n")
        f.write(classification_report_str)
    logger.info(f"Classification report saved to {report_path}")
    
    logger.info("="*60)
    logger.info("Training pipeline completed successfully!")
    logger.info("="*60)
    logger.info(f"Best model saved to: {best_model_path}")
    logger.info(f"Logs saved to: {config.LOG_DIR}")
    logger.info(f"Outputs saved to: {config.OUTPUT_DIR}")
    logger.info(f"\nValidation Set Performance (15% split):")
    logger.info(f"  Accuracy: {val_metrics['accuracy']:.4f}")
    logger.info(f"  Precision: {val_metrics['precision']:.4f}")
    logger.info(f"  Recall: {val_metrics['recall']:.4f}")
    logger.info(f"  F1-Score: {val_metrics['f1']:.4f}")
    
    return val_metrics, best_model_path


if __name__ == "__main__":
    main()

