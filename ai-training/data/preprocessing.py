"""
NIfTI preprocessing and augmentation functions.
"""

import numpy as np

# ---------------------------------------------------------------------------
# Compatibility shim for NumPy >= 2.0
# ---------------------------------------------------------------------------
# Some versions of nibabel (and possibly other dependencies) still rely on
# deprecated NumPy APIs (`np.sctypes`, `np.maximum_sctype`) that were removed
# in NumPy 2.0. To avoid forcing a downgrade of NumPy, we recreate minimal
# stand-ins when they are missing.

if not hasattr(np, "sctypes"):
    # Minimal set of scalar dtypes that matches what nibabel typically needs.
    float_types = [np.float16, np.float32, np.float64]

    int_types = [np.int8, np.int16, np.int32, np.int64]
    uint_types = [np.uint8, np.uint16, np.uint32, np.uint64]

    complex_types = [np.complex64, np.complex128]

    np.sctypes = {  # type: ignore[attr-defined]
        "int": int_types,
        "uint": uint_types,
        "float": float_types,
        "complex": complex_types,
        # Included for completeness; nibabel usually only needs "float".
        "others": [np.bool_, np.object_, np.bytes_],
    }


if not hasattr(np, "maximum_sctype"):
    # Simple implementation that picks the widest dtype of the given "kind".
    def _maximum_sctype(t):  # type: ignore[override]
        dt = np.dtype(t)
        kind_map = {
            "i": "int",
            "u": "uint",
            "f": "float",
            "c": "complex",
            "b": "others",
        }
        key = kind_map.get(dt.kind, "float")
        candidates = np.sctypes.get(key, [dt])  # type: ignore[attr-defined]
        # Choose dtype with largest itemsize
        return max(candidates, key=lambda d: np.dtype(d).itemsize)

    np.maximum_sctype = _maximum_sctype  # type: ignore[attr-defined]

import nibabel as nib
from scipy import ndimage
from typing import Tuple, Optional
import torch


def load_nifti(filepath: str) -> Tuple[np.ndarray, dict]:
    """
    Load a NIfTI file and return data and metadata.
    
    Args:
        filepath: Path to .nii or .nii.gz file
        
    Returns:
        Tuple of (data array, header metadata)
    """
    nifti_img = nib.load(filepath)
    data = nifti_img.get_fdata()
    header = {
        'affine': nifti_img.affine,
        'header': nifti_img.header,
        'shape': data.shape,
        'pixdims': nifti_img.header.get_zooms()[:3]
    }
    
    return data.astype(np.float32), header


def normalize_intensity(
    volume: np.ndarray,
    clip_percentile: Tuple[float, float] = (0.5, 99.5),
    method: str = 'percentile'
) -> np.ndarray:
    """
    Normalize intensity values of a volume.
    
    Args:
        volume: 3D volume array
        clip_percentile: Percentiles for clipping outliers
        method: Normalization method ('percentile' or 'zscore')
        
    Returns:
        Normalized volume
    """
    if method == 'percentile':
        # Clip outliers
        lower, upper = np.percentile(volume, clip_percentile)
        volume_clipped = np.clip(volume, lower, upper)
        
        # Normalize to [0, 1]
        if (upper - lower) > 0:
            volume_normalized = (volume_clipped - lower) / (upper - lower)
        else:
            volume_normalized = np.zeros_like(volume_clipped)
    
    elif method == 'zscore':
        # Z-score normalization
        mean = np.mean(volume)
        std = np.std(volume)
        if std > 0:
            volume_normalized = (volume - mean) / std
            # Shift to [0, 1] range
            volume_normalized = (volume_normalized - volume_normalized.min()) / (
                volume_normalized.max() - volume_normalized.min() + 1e-8
            )
        else:
            volume_normalized = np.zeros_like(volume)
    else:
        raise ValueError(f"Unknown normalization method: {method}")
    
    return volume_normalized.astype(np.float32)


def resize_volume(
    volume: np.ndarray,
    target_shape: Tuple[int, int, int],
    order: int = 1
) -> np.ndarray:
    """
    Resize a 3D volume to target shape using interpolation.
    
    Args:
        volume: Input 3D volume
        target_shape: Target (depth, height, width) shape
        order: Interpolation order (0=nearest, 1=linear, 2=quadratic, 3=cubic)
        
    Returns:
        Resized volume
    """
    if volume.shape == target_shape:
        return volume
    
    # Calculate zoom factors
    zoom_factors = [
        target_shape[i] / volume.shape[i] for i in range(3)
    ]
    
    # Resize using scipy
    resized = ndimage.zoom(volume, zoom_factors, order=order, mode='constant')
    
    return resized.astype(np.float32)


def random_flip(volume: np.ndarray, axis: Optional[int] = None) -> np.ndarray:
    """
    Randomly flip volume along specified axis.
    
    Args:
        volume: Input 3D volume
        axis: Axis to flip along (0, 1, 2, or None for random)
        
    Returns:
        Flipped volume
    """
    if axis is None:
        axis = np.random.randint(0, 3)
    
    return np.flip(volume, axis=axis).copy()


def random_rotation(
    volume: np.ndarray,
    max_angle: float = 15.0,
    axes: Tuple[int, int] = (0, 1)
) -> np.ndarray:
    """
    Randomly rotate volume around specified axes.
    
    Args:
        volume: Input 3D volume
        max_angle: Maximum rotation angle in degrees
        axes: Axes to rotate around
        
    Returns:
        Rotated volume
    """
    angle = np.random.uniform(-max_angle, max_angle)
    rotated = ndimage.rotate(volume, angle, axes=axes, reshape=False, mode='constant')
    return rotated.astype(np.float32)


def random_shift(
    volume: np.ndarray,
    max_shift: float = 0.1
) -> np.ndarray:
    """
    Randomly shift volume along each axis.
    
    Args:
        volume: Input 3D volume
        max_shift: Maximum shift as fraction of volume size
        
    Returns:
        Shifted volume
    """
    shifts = [
        np.random.uniform(-max_shift, max_shift) * volume.shape[i]
        for i in range(3)
    ]
    shifted = ndimage.shift(volume, shifts, mode='constant', order=1)
    return shifted.astype(np.float32)


def random_noise(volume: np.ndarray, std: float = 0.02) -> np.ndarray:
    """
    Add random Gaussian noise to volume.
    
    Args:
        volume: Input 3D volume
        std: Standard deviation of noise
        
    Returns:
        Volume with added noise
    """
    noise = np.random.normal(0, std, volume.shape).astype(np.float32)
    noisy_volume = volume + noise
    return np.clip(noisy_volume, 0, 1).astype(np.float32)


def augment_volume(
    volume: np.ndarray,
    prob_flip: float = 0.5,
    prob_rotate: float = 0.3,
    prob_shift: float = 0.3,
    prob_noise: float = 0.2
) -> np.ndarray:
    """
    Apply random augmentations to volume.
    
    Args:
        volume: Input 3D volume
        prob_flip: Probability of flipping
        prob_rotate: Probability of rotation
        prob_shift: Probability of shifting
        prob_noise: Probability of adding noise
        
    Returns:
        Augmented volume
    """
    augmented = volume.copy()
    
    # Random flip
    if np.random.rand() < prob_flip:
        augmented = random_flip(augmented)
    
    # Random rotation
    if np.random.rand() < prob_rotate:
        augmented = random_rotation(augmented, max_angle=10.0)
    
    # Random shift
    if np.random.rand() < prob_shift:
        augmented = random_shift(augmented, max_shift=0.05)
    
    # Random noise
    if np.random.rand() < prob_noise:
        augmented = random_noise(augmented, std=0.01)
    
    return augmented


def preprocess_volume(
    volume: np.ndarray,
    target_shape: Tuple[int, int, int] = (128, 128, 128),
    normalize: bool = True,
    clip_percentile: Tuple[float, float] = (0.5, 99.5),
    augment: bool = False
) -> np.ndarray:
    """
    Complete preprocessing pipeline for a volume.
    
    Args:
        volume: Raw 3D volume
        target_shape: Target shape for resizing
        normalize: Whether to normalize intensities
        clip_percentile: Percentiles for clipping
        augment: Whether to apply augmentation (for training)
        
    Returns:
        Preprocessed volume
    """
    # Resize
    processed = resize_volume(volume, target_shape)
    
    # Normalize
    if normalize:
        processed = normalize_intensity(processed, clip_percentile)
    
    # Augment (only for training)
    if augment:
        processed = augment_volume(processed)
    
    return processed


def volume_to_tensor(volume: np.ndarray, add_channel_dim: bool = True) -> torch.Tensor:
    """
    Convert numpy volume to PyTorch tensor.
    
    Args:
        volume: 3D numpy array
        add_channel_dim: Whether to add channel dimension (for CNN)
        
    Returns:
        PyTorch tensor (1, D, H, W) or (D, H, W)
    """
    tensor = torch.from_numpy(volume).float()
    
    if add_channel_dim:
        tensor = tensor.unsqueeze(0)  # Add channel dimension
    
    return tensor

