"""
Dataset loading and splitting for brain scan training.
Handles loading .nii.gz files, train/test split (70/30), and PyTorch DataLoader.
"""

import os
import zipfile
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import List, Tuple, Optional
import glob

from data.preprocessing import load_nifti, preprocess_volume
from config.config import config


class BrainScanDataset(Dataset):
    """PyTorch Dataset for brain scan volumes."""
    
    def __init__(
        self,
        file_paths: List[str],
        target_shape: Tuple[int, int, int] = (128, 128, 128),
        normalize: bool = True,
        augment: bool = False,
        is_training: bool = True
    ):
        """
        Initialize dataset.
        
        Args:
            file_paths: List of paths to .nii or .nii.gz files
            target_shape: Target shape for resizing volumes
            normalize: Whether to normalize intensities
            augment: Whether to apply augmentation
            is_training: Whether this is training set (affects augmentation)
        """
        self.file_paths = file_paths
        self.target_shape = target_shape
        self.normalize = normalize
        self.augment = augment and is_training
        self.is_training = is_training
        
        print(f"Dataset initialized with {len(file_paths)} scans")
        print(f"Target shape: {target_shape}")
        print(f"Augmentation: {self.augment}")
    
    def __len__(self) -> int:
        return len(self.file_paths)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, dict]:
        """
        Load and preprocess a single scan.
        
        Args:
            idx: Index of scan to load
            
        Returns:
            Tuple of (preprocessed volume tensor, metadata)
        """
        filepath = self.file_paths[idx]
        
        try:
            # Load NIfTI file
            volume, header = load_nifti(filepath)
            
            # Preprocess
            processed_volume = preprocess_volume(
                volume,
                target_shape=self.target_shape,
                normalize=self.normalize,
                augment=self.augment
            )
            
            # Convert to tensor (add channel dimension: 1, D, H, W)
            volume_tensor = torch.from_numpy(processed_volume).float().unsqueeze(0)
            
            # Metadata
            metadata = {
                'filepath': filepath,
                'original_shape': volume.shape,
                'processed_shape': processed_volume.shape,
                'pixdims': header.get('pixdims', [1.0, 1.0, 1.0])
            }
            
            return volume_tensor, metadata
            
        except Exception as e:
            print(f"Error loading {filepath}: {e}")
            # Return zero volume as fallback
            volume_tensor = torch.zeros((1,) + self.target_shape, dtype=torch.float32)
            metadata = {'filepath': filepath, 'error': str(e)}
            return volume_tensor, metadata


def find_nifti_files(dataset_path: str) -> List[str]:
    """
    Find all .nii and .nii.gz files in dataset directory.
    Handles both zipped and unzipped files.
    
    Args:
        dataset_path: Path to dataset directory
        
    Returns:
        List of file paths
    """
    nifti_files = []
    dataset_path = Path(dataset_path)
    temp_extract_dir = dataset_path / "temp_extracted"
    
    # Find .nii.gz files (compressed) - exclude temp directory
    nifti_files.extend([
        f for f in glob.glob(str(dataset_path / "**/*.nii.gz"), recursive=True)
        if "temp_extracted" not in f
    ])
    
    # Find .nii files (uncompressed) - exclude temp directory
    nifti_files.extend([
        f for f in glob.glob(str(dataset_path / "**/*.nii"), recursive=True)
        if "temp_extracted" not in f and not f.endswith('.nii.gz')
    ])
    
    # Handle zip files
    zip_files = glob.glob(str(dataset_path / "**/*.zip"), recursive=True)
    
    if zip_files:
        print(f"Found {len(zip_files)} zip files, extracting...")
        temp_extract_dir.mkdir(parents=True, exist_ok=True)
    
    for zip_path in zip_files:
        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                file_list = zip_ref.namelist()
                extracted_count = 0
                
                for file_name in file_list:
                    # Only extract .nii files
                    if file_name.endswith('.nii') and not file_name.endswith('.nii.gz'):
                        # Preserve directory structure in temp folder
                        extract_path = temp_extract_dir / Path(file_name).name
                        extract_path.parent.mkdir(parents=True, exist_ok=True)
                        
                        # Only extract if not already extracted
                        if not extract_path.exists():
                            with zip_ref.open(file_name) as source:
                                with open(extract_path, 'wb') as target:
                                    target.write(source.read())
                            extracted_count += 1
                        
                        nifti_files.append(str(extract_path))
                    elif file_name.endswith('.nii.gz'):
                        # Handle .nii.gz files in zip
                        extract_path = temp_extract_dir / Path(file_name).name
                        extract_path.parent.mkdir(parents=True, exist_ok=True)
                        
                        if not extract_path.exists():
                            with zip_ref.open(file_name) as source:
                                with open(extract_path, 'wb') as target:
                                    target.write(source.read())
                            extracted_count += 1
                        
                        nifti_files.append(str(extract_path))
                
                if extracted_count > 0:
                    print(f"  Extracted {extracted_count} files from {Path(zip_path).name}")
                    
        except Exception as e:
            print(f"Warning: Could not process zip file {zip_path}: {e}")
    
    # Remove duplicates and sort
    nifti_files = sorted(list(set(nifti_files)))
    
    print(f"Found {len(nifti_files)} NIfTI files")
    return nifti_files


def split_dataset(
    file_paths: List[str],
    train_split: float = 0.7,
    test_split: float = 0.3,
    random_seed: int = 42
) -> Tuple[List[str], List[str]]:
    """
    Split dataset into train and test sets.
    
    Args:
        file_paths: List of all file paths
        train_split: Fraction for training (default 0.7 = 70%)
        test_split: Fraction for testing (default 0.3 = 30%)
        random_seed: Random seed for reproducibility
        
    Returns:
        Tuple of (train_paths, test_paths)
    """
    assert abs(train_split + test_split - 1.0) < 1e-6, "Train + test split must equal 1.0"
    
    np.random.seed(random_seed)
    shuffled = np.random.permutation(file_paths)
    
    n_total = len(shuffled)
    n_train = int(n_total * train_split)
    
    train_paths = shuffled[:n_train].tolist()
    test_paths = shuffled[n_train:].tolist()
    
    print(f"Dataset split:")
    print(f"  Training: {len(train_paths)} scans ({len(train_paths)/n_total*100:.1f}%)")
    print(f"  Testing: {len(test_paths)} scans ({len(test_paths)/n_total*100:.1f}%)")
    
    return train_paths, test_paths


def create_dataloaders(
    dataset_path: str,
    train_split: float = 0.7,
    batch_size: int = 4,
    num_workers: int = 4,
    pin_memory: bool = True,
    target_shape: Tuple[int, int, int] = (128, 128, 128),
    normalize: bool = True,
    use_augmentation: bool = True,
    random_seed: int = 42
) -> Tuple[DataLoader, DataLoader]:
    """
    Create train and test DataLoaders.
    
    Args:
        dataset_path: Path to dataset directory
        train_split: Training split ratio
        batch_size: Batch size for training
        num_workers: Number of data loader workers
        pin_memory: Whether to pin memory for faster GPU transfer
        target_shape: Target volume shape
        normalize: Whether to normalize
        use_augmentation: Whether to use augmentation for training
        random_seed: Random seed
        
    Returns:
        Tuple of (train_loader, test_loader)
    """
    # Find all NIfTI files
    print("Scanning for NIfTI files...")
    all_files = find_nifti_files(dataset_path)
    
    if len(all_files) == 0:
        raise ValueError(f"No NIfTI files found in {dataset_path}")
    
    # Split dataset
    train_paths, test_paths = split_dataset(
        all_files,
        train_split=train_split,
        test_split=1.0 - train_split,
        random_seed=random_seed
    )
    
    # Create datasets
    train_dataset = BrainScanDataset(
        train_paths,
        target_shape=target_shape,
        normalize=normalize,
        augment=use_augmentation,
        is_training=True
    )
    
    test_dataset = BrainScanDataset(
        test_paths,
        target_shape=target_shape,
        normalize=normalize,
        augment=False,  # No augmentation for testing
        is_training=False
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True  # Drop last incomplete batch
    )
    
    # Ensure test batch size is at least 1, even when training batch_size is 1.
    test_batch_size = max(1, batch_size // 2)

    test_loader = DataLoader(
        test_dataset,
        batch_size=test_batch_size,  # Smaller batch for testing (but >= 1)
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False
    )
    
    return train_loader, test_loader

