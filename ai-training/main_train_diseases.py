"""
Main training script for brain tumor classification.
Trains a model to classify brain tumors into 4 classes: glioma, meningioma, notumor, pituitary.
Runs the entire training pipeline: data loading, training, and evaluation.
"""

import os
import sys
import argparse
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

import torch
import numpy as np

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
    parser.add_argument('--epochs', type=int, default=None, help='Number of epochs (overrides config)')
    parser.add_argument('--batch-size', type=int, default=None, help='Batch size (overrides config)')
    parser.add_argument('--gpu', type=int, default=None, help='GPU device ID')
    parser.add_argument('--model-type', type=str, default='custom', choices=['custom', 'resnet'],
                        help='Model type: custom or resnet')
    parser.add_argument('--pretrained', action='store_true', help='Use pretrained ResNet (only for resnet model)')
    args = parser.parse_args()
    
    # Dataset configuration
    BASE_DATASET_DIR = r"D:\Medical Imagining (CT scan, MRI, X-ray, and Microscopic Imagery) Data\Medical Imagining (CT scan, MRI, X-ray, and Microscopic Imagery) Data\Medical Imagining\Brain Tumor Classification\Testing"
    CLASS_FOLDERS = ['glioma', 'meningioma', 'notumor', 'pituitary']
    
    # Override config with command line arguments
    num_epochs = args.epochs if args.epochs else 50
    batch_size = args.batch_size if args.batch_size else 32
    
    # Setup logging
    logger = setup_logger("brain_tumor_classification", config.LOG_DIR)
    logger.info("="*60)
    logger.info("Starting Brain Tumor Classification Training Pipeline")
    logger.info("="*60)
    logger.info(f"Classes: {CLASS_FOLDERS}")
    logger.info(f"Model type: {args.model_type}")
    logger.info(f"Using pretrained: {args.pretrained if args.model_type == 'resnet' else False}")
    
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
    if not os.path.exists(BASE_DATASET_DIR):
        logger.error(f"Dataset directory does not exist: {BASE_DATASET_DIR}")
        logger.error("Please update BASE_DATASET_DIR in main_train_diseases.py")
        raise FileNotFoundError(f"Dataset directory not found: {BASE_DATASET_DIR}")
    
    # Create data loaders
    logger.info("Creating data loaders...")
    logger.info(f"Dataset path: {BASE_DATASET_DIR}")
    
    try:
        train_loader, val_loader, test_loader, class_names = create_classification_dataloaders(
            base_dir=BASE_DATASET_DIR,
            class_folders=CLASS_FOLDERS,
            train_split=0.7,
            val_split=0.15,
            test_split=0.15,
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
        logger.info(f"  Test batches: {len(test_loader)}")
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
    
    # Load best model for evaluation
    logger.info("Loading best model for final evaluation...")
    best_model_path = os.path.join(config.CHECKPOINT_DIR, "best_model.pth")
    
    if os.path.exists(best_model_path):
        checkpoint = torch.load(best_model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        logger.info(f"Best model loaded from epoch {checkpoint['epoch'] + 1}")
        logger.info(f"Best validation loss: {checkpoint.get('loss', 'N/A')}")
    else:
        logger.warning("Best model checkpoint not found, using current model")
    
    # Evaluate on test set
    logger.info("Running final evaluation on test set...")
    
    evaluator = ClassificationEvaluator(
        model=model,
        test_loader=test_loader,
        device=device,
        class_names=class_names
    )
    
    test_metrics = evaluator.evaluate()
    evaluator.print_results(test_metrics)
    
    # Generate classification report
    classification_report_str = evaluator.generate_classification_report(test_metrics)
    logger.info("\nDetailed Classification Report:")
    logger.info(classification_report_str)
    
    # Save evaluation results
    import json
    results_path = os.path.join(config.OUTPUT_DIR, "test_results_classification.json")
    
    # Prepare metrics for JSON serialization
    json_metrics = {
        k: v for k, v in test_metrics.items()
        if k not in ['predictions', 'labels', 'probabilities', 'confusion_matrix']
    }
    json_metrics['confusion_matrix'] = test_metrics['confusion_matrix']
    
    with open(results_path, 'w') as f:
        json.dump(json_metrics, f, indent=2)
    logger.info(f"Test results saved to {results_path}")
    
    # Save classification report
    report_path = os.path.join(config.OUTPUT_DIR, "classification_report.txt")
    with open(report_path, 'w') as f:
        f.write("Brain Tumor Classification Report\n")
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
    logger.info(f"\nTest Set Performance:")
    logger.info(f"  Accuracy: {test_metrics['accuracy']:.4f}")
    logger.info(f"  Precision: {test_metrics['precision']:.4f}")
    logger.info(f"  Recall: {test_metrics['recall']:.4f}")
    logger.info(f"  F1-Score: {test_metrics['f1']:.4f}")


if __name__ == "__main__":
    main()

