"""
Configuration file for brain scan training pipeline.
Contains all hyperparameters, paths, and training settings.
"""

import os
from pathlib import Path
from dataclasses import dataclass
from typing import Tuple

@dataclass
class Config:
    """Main configuration class for training."""
    
    # Dataset paths
    DATASET_PATH: str = r"D:\IXI-T1"
    OUTPUT_DIR: str = "./outputs"
    CHECKPOINT_DIR: str = "./checkpoints"
    LOG_DIR: str = "./logs"
    
    # Data parameters
    TRAIN_SPLIT: float = 0.7  # 70% for training
    TEST_SPLIT: float = 0.3   # 30% for testing
    RANDOM_SEED: int = 42
    
    # Image preprocessing
    TARGET_SHAPE: Tuple[int, int, int] = (128, 128, 128)  # Resize to this shape for efficiency
    NORMALIZE: bool = True
    CLIP_PERCENTILE: Tuple[float, float] = (0.5, 99.5)  # Clip outliers
    
    # Training parameters
    # For CPU-only / limited-RAM systems, start with batch size 1 for stability.
    BATCH_SIZE: int = 1
    NUM_EPOCHS: int = 50
    LEARNING_RATE: float = 1e-4
    WEIGHT_DECAY: float = 1e-5
    
    # Model parameters
    MODEL_TYPE: str = "3dcnn"  # 3D CNN architecture
    IN_CHANNELS: int = 1  # Single channel (T1)
    # Smaller feature and channel sizes to reduce memory usage.
    FEATURE_DIM: int = 128  # Feature dimension for encoding
    BASE_CHANNELS: int = 16  # Base number of channels in first conv layer
    
    # Training settings
    USE_GPU: bool = True
    # On memory-constrained systems (especially CPU-only on Windows),
    # using multiple workers can exhaust RAM / pagefile. Start with 0
    # (no multiprocessing) for stability and increase cautiously.
    NUM_WORKERS: int = 0  # Data loader workers
    # Pinned memory mainly helps when using a GPU; keep it False on CPU
    # to reduce memory pressure.
    PIN_MEMORY: bool = False
    
    # Checkpointing
    SAVE_EVERY_N_EPOCHS: int = 5
    SAVE_BEST_MODEL: bool = True
    EARLY_STOPPING_PATIENCE: int = 10
    
    # Validation
    VAL_EVERY_N_EPOCHS: int = 1
    VAL_BATCH_SIZE: int = 2
    
    # Augmentation (for training)
    USE_AUGMENTATION: bool = True
    AUGMENTATION_PROB: float = 0.5
    
    # Logging
    LOG_EVERY_N_BATCHES: int = 10
    USE_TENSORBOARD: bool = True
    
    def __post_init__(self):
        """Create output directories if they don't exist."""
        os.makedirs(self.OUTPUT_DIR, exist_ok=True)
        os.makedirs(self.CHECKPOINT_DIR, exist_ok=True)
        os.makedirs(self.LOG_DIR, exist_ok=True)
        
        # Verify dataset path exists
        if not os.path.exists(self.DATASET_PATH):
            print(f"Warning: Dataset path does not exist: {self.DATASET_PATH}")
            print("Please update DATASET_PATH in config/config.py")

# Global config instance
config = Config()

