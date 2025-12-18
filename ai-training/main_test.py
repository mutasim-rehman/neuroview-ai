"""
Main testing script for brain tumor classification.
Tests a trained model on the separate testing dataset.
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
from training.classification_evaluator import ClassificationEvaluator
from utils.helpers import (
    setup_logger,
    get_device,
    count_parameters
)


def main(model_path: str = None):
    """Main testing pipeline."""
    
    # Parse arguments
    parser = argparse.ArgumentParser(description='Test brain tumor classification model')
    parser.add_argument('--model', type=str, default=None, help='Path to model checkpoint')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--gpu', type=int, default=None, help='GPU device ID')
    parser.add_argument('--model-type', type=str, default='custom', choices=['custom', 'resnet'],
                        help='Model type: custom or resnet')
    args = parser.parse_args()
    
    # Use provided model_path or command line argument
    if model_path is None:
        model_path = args.model
    
    # Default to best model if not specified
    if model_path is None:
        model_path = os.path.join(config.CHECKPOINT_DIR, "best_model.pth")
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model checkpoint not found at {model_path}. "
                "Please specify a model path with --model or train a model first."
            )
    
    # Dataset configuration - Testing locations
    TESTING_BASE_DIR = r"D:\Medical Imagining (CT scan, MRI, X-ray, and Microscopic Imagery) Data\Medical Imagining (CT scan, MRI, X-ray, and Microscopic Imagery) Data\Medical Imagining\Brain Tumor Classification\Testing"
    CLASS_FOLDERS = ['glioma', 'meningioma', 'notumor', 'pituitary']
    
    batch_size = args.batch_size
    
    # Setup logging
    logger = setup_logger("brain_tumor_testing", config.LOG_DIR)
    logger.info("="*60)
    logger.info("Starting Brain Tumor Classification Testing")
    logger.info("="*60)
    logger.info(f"Testing data directory: {TESTING_BASE_DIR}")
    logger.info(f"Classes: {CLASS_FOLDERS}")
    logger.info(f"Model path: {model_path}")
    logger.info(f"Model type: {args.model_type}")
    logger.info(f"Batch size: {batch_size}")
    
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
    if not os.path.exists(TESTING_BASE_DIR):
        logger.error(f"Testing directory does not exist: {TESTING_BASE_DIR}")
        logger.error("Please update TESTING_BASE_DIR in main_test.py")
        raise FileNotFoundError(f"Testing directory not found: {TESTING_BASE_DIR}")
    
    # Create test data loader (no split needed - use all testing data)
    logger.info("Creating test data loader...")
    logger.info(f"Testing data path: {TESTING_BASE_DIR}")
    
    try:
        # For testing, we use all data as test set (100% test, 0% train/val)
        _, _, test_loader, class_names = create_classification_dataloaders(
            base_dir=TESTING_BASE_DIR,
            class_folders=CLASS_FOLDERS,
            train_split=0.0,
            val_split=0.0,
            test_split=1.0,  # Use all as test data
            batch_size=batch_size,
            num_workers=config.NUM_WORKERS,
            pin_memory=config.PIN_MEMORY,
            target_size=(224, 224),
            normalize=True,
            use_augmentation=False,  # No augmentation for testing
            random_seed=config.RANDOM_SEED
        )
        logger.info(f"Test data loader created successfully")
        logger.info(f"  Test batches: {len(test_loader)}")
        logger.info(f"  Class names: {class_names}")
    except Exception as e:
        logger.error(f"Failed to create test data loader: {e}")
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
        use_pretrained=False,
        model_type=args.model_type
    )
    
    num_params = count_parameters(model)
    logger.info(f"Model created: {num_params:,} parameters")
    logger.info(f"Number of classes: {num_classes}")
    
    # Load trained model
    logger.info(f"Loading trained model from: {model_path}")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")
    
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    logger.info(f"Model loaded successfully")
    logger.info(f"  Trained for {checkpoint.get('epoch', 'N/A')} epochs")
    logger.info(f"  Validation loss: {checkpoint.get('loss', 'N/A')}")
    
    # Evaluate on test set
    logger.info("Running evaluation on test set...")
    
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
    logger.info("\nDetailed Classification Report (Testing Set):")
    logger.info(classification_report_str)
    
    # Save evaluation results
    results_path = os.path.join(config.OUTPUT_DIR, "testing_results.json")
    
    # Prepare metrics for JSON serialization
    json_metrics = {
        k: v for k, v in test_metrics.items()
        if k not in ['predictions', 'labels', 'probabilities']
    }
    json_metrics['confusion_matrix'] = test_metrics['confusion_matrix']
    
    with open(results_path, 'w') as f:
        json.dump(json_metrics, f, indent=2)
    logger.info(f"Test results saved to {results_path}")
    
    # Save classification report
    report_path = os.path.join(config.OUTPUT_DIR, "testing_report.txt")
    with open(report_path, 'w') as f:
        f.write("Brain Tumor Classification Report - Testing Set\n")
        f.write("="*60 + "\n\n")
        f.write(f"Classes: {', '.join(class_names)}\n\n")
        f.write(classification_report_str)
    logger.info(f"Classification report saved to {report_path}")
    
    logger.info("="*60)
    logger.info("Testing pipeline completed successfully!")
    logger.info("="*60)
    logger.info(f"\nTest Set Performance:")
    logger.info(f"  Accuracy: {test_metrics['accuracy']:.4f}")
    logger.info(f"  Precision: {test_metrics['precision']:.4f}")
    logger.info(f"  Recall: {test_metrics['recall']:.4f}")
    logger.info(f"  F1-Score: {test_metrics['f1']:.4f}")
    
    return test_metrics


if __name__ == "__main__":
    main()

