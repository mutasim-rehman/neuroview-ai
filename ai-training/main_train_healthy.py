"""
Main training script for healthy brain scan analysis.
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
from data.data_loader import create_dataloaders
from models.brain_model import create_model
from training.trainer import Trainer
from training.evaluator import Evaluator
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
    parser = argparse.ArgumentParser(description='Train brain scan analysis model')
    parser.add_argument('--resume', type=str, default=None, help='Resume from checkpoint')
    parser.add_argument('--epochs', type=int, default=None, help='Number of epochs (overrides config)')
    parser.add_argument('--batch-size', type=int, default=None, help='Batch size (overrides config)')
    parser.add_argument('--gpu', type=int, default=None, help='GPU device ID')
    args = parser.parse_args()
    
    # Override config with command line arguments
    if args.epochs:
        config.NUM_EPOCHS = args.epochs
    if args.batch_size:
        config.BATCH_SIZE = args.batch_size
    
    # Setup logging
    logger = setup_logger("brain_training", config.LOG_DIR)
    logger.info("="*60)
    logger.info("Starting Brain Scan Training Pipeline")
    logger.info("="*60)
    
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
    
    # Create data loaders
    logger.info("Creating data loaders...")
    logger.info(f"Dataset path: {config.DATASET_PATH}")
    
    try:
        train_loader, test_loader = create_dataloaders(
            dataset_path=config.DATASET_PATH,
            train_split=config.TRAIN_SPLIT,
            batch_size=config.BATCH_SIZE,
            num_workers=config.NUM_WORKERS,
            pin_memory=config.PIN_MEMORY,
            target_shape=config.TARGET_SHAPE,
            normalize=config.NORMALIZE,
            use_augmentation=config.USE_AUGMENTATION,
            random_seed=config.RANDOM_SEED
        )
        logger.info(f"Data loaders created successfully")
        logger.info(f"  Training batches: {len(train_loader)}")
        logger.info(f"  Test batches: {len(test_loader)}")
    except Exception as e:
        logger.error(f"Failed to create data loaders: {e}")
        raise
    
    # Create model
    logger.info("Creating model...")
    model = create_model(
        in_channels=config.IN_CHANNELS,
        feature_dim=config.FEATURE_DIM,
        base_channels=config.BASE_CHANNELS
    )
    
    num_params = count_parameters(model)
    logger.info(f"Model created: {num_params:,} parameters")
    
    # Resume from checkpoint if specified
    start_epoch = 0
    if args.resume:
        logger.info(f"Resuming from checkpoint: {args.resume}")
        # Note: We'll create a dummy optimizer here for loading
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.LEARNING_RATE)
        start_epoch = load_checkpoint(model, optimizer, args.resume, device)
    
    # Create trainer
    logger.info("Initializing trainer...")
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=test_loader,  # Using test set as validation for now
        device=device,
        learning_rate=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY,
        log_dir=config.LOG_DIR,
        checkpoint_dir=config.CHECKPOINT_DIR,
        log_every_n_batches=config.LOG_EVERY_N_BATCHES
    )
    
    # Train
    logger.info(f"Starting training for {config.NUM_EPOCHS} epochs...")
    trainer.train(
        num_epochs=config.NUM_EPOCHS,
        save_every_n_epochs=config.SAVE_EVERY_N_EPOCHS
    )
    
    # Load best model for evaluation
    logger.info("Loading best model for final evaluation...")
    best_model_path = os.path.join(config.CHECKPOINT_DIR, "best_model.pth")
    
    if os.path.exists(best_model_path):
        checkpoint = torch.load(best_model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        logger.info(f"Best model loaded from epoch {checkpoint['epoch'] + 1}")
    else:
        logger.warning("Best model checkpoint not found, using current model")
    
    # Evaluate on test set
    logger.info("Running final evaluation on test set...")
    
    # Create test loader (separate from validation)
    # For now, we'll use the same test_loader
    evaluator = Evaluator(
        model=model,
        test_loader=test_loader,
        device=device
    )
    
    test_metrics = evaluator.evaluate()
    evaluator.print_results(test_metrics)
    
    # Save evaluation results
    import json
    results_path = os.path.join(config.OUTPUT_DIR, "test_results.json")
    with open(results_path, 'w') as f:
        json.dump({k: float(v) for k, v in test_metrics.items()}, f, indent=2)
    logger.info(f"Test results saved to {results_path}")
    
    logger.info("="*60)
    logger.info("Training pipeline completed successfully!")
    logger.info("="*60)
    logger.info(f"Best model saved to: {best_model_path}")
    logger.info(f"Logs saved to: {config.LOG_DIR}")
    logger.info(f"Outputs saved to: {config.OUTPUT_DIR}")


if __name__ == "__main__":
    main()

