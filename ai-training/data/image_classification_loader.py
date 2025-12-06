"""
Dataset loading for 2D brain tumor image classification.
Handles loading images from class folders and creating PyTorch DataLoaders.
"""

import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import List, Tuple, Optional
from PIL import Image
import torchvision.transforms as transforms


class BrainTumorImageDataset(Dataset):
    """PyTorch Dataset for brain tumor image classification."""
    
    def __init__(
        self,
        image_paths: List[str],
        labels: List[int],
        target_size: Tuple[int, int] = (224, 224),
        normalize: bool = True,
        augment: bool = False,
        is_training: bool = True
    ):
        """
        Initialize dataset.
        
        Args:
            image_paths: List of paths to image files
            labels: List of class labels (integers)
            target_size: Target size for resizing images (H, W)
            normalize: Whether to normalize pixel values
            augment: Whether to apply augmentation
            is_training: Whether this is training set (affects augmentation)
        """
        self.image_paths = image_paths
        self.labels = labels
        self.target_size = target_size
        self.normalize = normalize
        self.augment = augment and is_training
        self.is_training = is_training
        
        # Build transforms
        self.transform = self._build_transform()
        
        print(f"Dataset initialized with {len(image_paths)} images")
        print(f"Target size: {target_size}")
        print(f"Augmentation: {self.augment}")
        print(f"Number of classes: {len(set(labels))}")
    
    def _build_transform(self):
        """Build image transformation pipeline."""
        transforms_list = []
        
        # Resize
        transforms_list.append(transforms.Resize(self.target_size))
        
        # Augmentation for training
        if self.augment:
            transforms_list.extend([
                transforms.RandomRotation(15),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
                transforms.RandomAffine(degrees=0, translate=(0.1, 0.1))
            ])
        
        # Convert to tensor
        transforms_list.append(transforms.ToTensor())
        
        # Normalize if requested
        if self.normalize:
            # ImageNet normalization (common for medical images)
            transforms_list.append(transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ))
        
        return transforms.Compose(transforms_list)
    
    def __len__(self) -> int:
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Load and preprocess a single image.
        
        Args:
            idx: Index of image to load
            
        Returns:
            Tuple of (preprocessed image tensor, label)
        """
        image_path = self.image_paths[idx]
        label = self.labels[idx]
        
        try:
            # Load image
            image = Image.open(image_path).convert('RGB')
            
            # Apply transforms
            image_tensor = self.transform(image)
            
            return image_tensor, label
            
        except Exception as e:
            print(f"Error loading {image_path}: {e}")
            # Return zero image as fallback
            channels = 3
            image_tensor = torch.zeros((channels, self.target_size[0], self.target_size[1]), dtype=torch.float32)
            return image_tensor, label


def find_image_files(base_dir: str, class_folders: List[str]) -> Tuple[List[str], List[int], List[str]]:
    """
    Find all image files in class folders.
    
    Args:
        base_dir: Base directory containing class folders
        class_folders: List of class folder names
        
    Returns:
        Tuple of (image_paths, labels, class_names)
    """
    image_paths = []
    labels = []
    
    # Supported image extensions
    extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff', '*.tif']
    
    for class_idx, class_folder in enumerate(class_folders):
        class_path = Path(base_dir) / class_folder
        
        if not class_path.exists():
            print(f"Warning: Class folder does not exist: {class_path}")
            continue
        
        # Find all images in this class folder
        class_images = []
        for ext in extensions:
            class_images.extend(glob.glob(str(class_path / ext), recursive=False))
            class_images.extend(glob.glob(str(class_path / ext.upper()), recursive=False))
        
        if len(class_images) == 0:
            print(f"Warning: No images found in {class_path}")
            continue
        
        image_paths.extend(class_images)
        labels.extend([class_idx] * len(class_images))
        
        print(f"Found {len(class_images)} images in class '{class_folder}' (label: {class_idx})")
    
    # Sort by path for reproducibility
    sorted_pairs = sorted(zip(image_paths, labels))
    image_paths, labels = zip(*sorted_pairs)
    
    return list(image_paths), list(labels), class_folders


def split_dataset(
    image_paths: List[str],
    labels: List[int],
    train_split: float = 0.7,
    val_split: float = 0.15,
    test_split: float = 0.15,
    random_seed: int = 42
) -> Tuple[List[str], List[int], List[str], List[int], List[str], List[int]]:
    """
    Split dataset into train, validation, and test sets.
    Maintains class distribution in each split.
    
    Args:
        image_paths: List of all image paths
        labels: List of all labels
        train_split: Fraction for training
        val_split: Fraction for validation
        test_split: Fraction for testing
        random_seed: Random seed for reproducibility
        
    Returns:
        Tuple of (train_paths, train_labels, val_paths, val_labels, test_paths, test_labels)
    """
    assert abs(train_split + val_split + test_split - 1.0) < 1e-6, \
        "Train + val + test split must equal 1.0"
    
    np.random.seed(random_seed)
    
    # Get unique classes
    unique_classes = sorted(set(labels))
    
    train_paths, train_labels = [], []
    val_paths, val_labels = [], []
    test_paths, test_labels = [], []
    
    # Split each class separately to maintain distribution
    for class_label in unique_classes:
        # Get all indices for this class
        class_indices = [i for i, label in enumerate(labels) if label == class_label]
        np.random.shuffle(class_indices)
        
        n_total = len(class_indices)
        n_train = int(n_total * train_split)
        n_val = int(n_total * val_split)
        
        train_indices = class_indices[:n_train]
        val_indices = class_indices[n_train:n_train + n_val]
        test_indices = class_indices[n_train + n_val:]
        
        train_paths.extend([image_paths[i] for i in train_indices])
        train_labels.extend([labels[i] for i in train_indices])
        
        val_paths.extend([image_paths[i] for i in val_indices])
        val_labels.extend([labels[i] for i in val_indices])
        
        test_paths.extend([image_paths[i] for i in test_indices])
        test_labels.extend([labels[i] for i in test_indices])
    
    print(f"\nDataset split:")
    print(f"  Training: {len(train_paths)} images ({len(train_paths)/len(image_paths)*100:.1f}%)")
    print(f"  Validation: {len(val_paths)} images ({len(val_paths)/len(image_paths)*100:.1f}%)")
    print(f"  Testing: {len(test_paths)} images ({len(test_paths)/len(image_paths)*100:.1f}%)")
    
    return train_paths, train_labels, val_paths, val_labels, test_paths, test_labels


def create_classification_dataloaders(
    base_dir: str,
    class_folders: List[str],
    train_split: float = 0.7,
    val_split: float = 0.15,
    test_split: float = 0.15,
    batch_size: int = 32,
    num_workers: int = 0,
    pin_memory: bool = False,
    target_size: Tuple[int, int] = (224, 224),
    normalize: bool = True,
    use_augmentation: bool = True,
    random_seed: int = 42
) -> Tuple[DataLoader, DataLoader, DataLoader, List[str]]:
    """
    Create train, validation, and test DataLoaders for classification.
    
    Args:
        base_dir: Base directory containing class folders
        class_folders: List of class folder names
        train_split: Training split ratio
        val_split: Validation split ratio
        test_split: Test split ratio
        batch_size: Batch size for training
        num_workers: Number of data loader workers
        pin_memory: Whether to pin memory for faster GPU transfer
        target_size: Target image size (H, W)
        normalize: Whether to normalize
        use_augmentation: Whether to use augmentation for training
        random_seed: Random seed
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader, class_names)
    """
    # Find all images
    print("Scanning for image files...")
    image_paths, labels, class_names = find_image_files(base_dir, class_folders)
    
    if len(image_paths) == 0:
        raise ValueError(f"No images found in {base_dir}")
    
    print(f"\nTotal images found: {len(image_paths)}")
    print(f"Classes: {class_names}")
    
    # Split dataset
    train_paths, train_labels, val_paths, val_labels, test_paths, test_labels = split_dataset(
        image_paths,
        labels,
        train_split=train_split,
        val_split=val_split,
        test_split=test_split,
        random_seed=random_seed
    )
    
    # Create datasets
    train_dataset = BrainTumorImageDataset(
        train_paths,
        train_labels,
        target_size=target_size,
        normalize=normalize,
        augment=use_augmentation,
        is_training=True
    )
    
    val_dataset = BrainTumorImageDataset(
        val_paths,
        val_labels,
        target_size=target_size,
        normalize=normalize,
        augment=False,
        is_training=False
    )
    
    test_dataset = BrainTumorImageDataset(
        test_paths,
        test_labels,
        target_size=target_size,
        normalize=normalize,
        augment=False,
        is_training=False
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False
    )
    
    return train_loader, val_loader, test_loader, class_names

