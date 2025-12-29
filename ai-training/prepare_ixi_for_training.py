"""
Prepare IXI healthy brain dataset for training.
Extracts 2D slices from IXI NIfTI files and saves them as images
to be used in the notumor class.
"""

import os
import sys
import numpy as np
import nibabel as nib
from PIL import Image
from pathlib import Path
from tqdm import tqdm

def extract_slices_from_nifti(nifti_path, output_dir, num_slices=5, prefix=""):
    """
    Extract multiple 2D slices from a NIfTI file and save as images.
    
    Args:
        nifti_path: Path to .nii or .nii.gz file
        output_dir: Directory to save extracted slices
        num_slices: Number of slices to extract per volume (centered around middle)
        prefix: Prefix for output filenames
    """
    try:
        # Load NIfTI
        nifti_img = nib.load(nifti_path)
        data = nifti_img.get_fdata()
        
        # Get volume shape
        if len(data.shape) < 3:
            print(f"Skipping {nifti_path}: not a 3D volume")
            return 0
        
        # Choose the axis with the most slices (usually axial view)
        z_dim = data.shape[2]
        
        # Calculate slice indices (centered around middle)
        middle = z_dim // 2
        half_range = num_slices // 2
        start_idx = max(0, middle - half_range)
        end_idx = min(z_dim, middle + half_range + 1)
        
        saved_count = 0
        for i, slice_idx in enumerate(range(start_idx, end_idx)):
            # Extract slice
            slice_2d = data[:, :, slice_idx]
            
            # Skip empty or near-empty slices
            if slice_2d.max() - slice_2d.min() < 1:
                continue
            
            # Normalize to 0-255
            slice_min, slice_max = slice_2d.min(), slice_2d.max()
            if slice_max > slice_min:
                slice_normalized = ((slice_2d - slice_min) / (slice_max - slice_min) * 255).astype(np.uint8)
            else:
                continue
            
            # Save as image
            img = Image.fromarray(slice_normalized, mode='L')
            
            # Resize to match training size (224x224)
            img = img.resize((224, 224), Image.Resampling.LANCZOS)
            
            # Convert to RGB (to match training data format)
            img = img.convert('RGB')
            
            # Save
            output_filename = f"{prefix}_slice{slice_idx:03d}.jpg"
            output_path = os.path.join(output_dir, output_filename)
            img.save(output_path, quality=95)
            saved_count += 1
        
        return saved_count
        
    except Exception as e:
        print(f"Error processing {nifti_path}: {e}")
        return 0


def prepare_ixi_dataset(ixi_dir, output_dir, max_subjects=None, slices_per_subject=5):
    """
    Process all NIfTI files in IXI directory and extract slices for training.
    
    Args:
        ixi_dir: Path to IXI-T1 directory containing .nii.gz files
        output_dir: Output directory for extracted slices (should be 'notumor' folder)
        max_subjects: Maximum number of subjects to process (None for all)
        slices_per_subject: Number of slices to extract per subject
    """
    print(f"IXI Directory: {ixi_dir}")
    print(f"Output Directory: {output_dir}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all NIfTI files
    nifti_files = []
    for ext in ['*.nii', '*.nii.gz']:
        nifti_files.extend(Path(ixi_dir).glob(ext))
    
    if not nifti_files:
        print(f"No NIfTI files found in {ixi_dir}")
        return
    
    print(f"Found {len(nifti_files)} NIfTI files")
    
    if max_subjects:
        nifti_files = nifti_files[:max_subjects]
        print(f"Processing first {max_subjects} subjects")
    
    total_slices = 0
    for nifti_path in tqdm(nifti_files, desc="Processing NIfTI files"):
        # Create prefix from filename (remove extension)
        stem = nifti_path.stem
        if stem.endswith('.nii'):
            stem = stem[:-4]  # Remove .nii from .nii.gz files
        
        prefix = f"ixi_{stem}"
        
        count = extract_slices_from_nifti(
            str(nifti_path),
            output_dir,
            num_slices=slices_per_subject,
            prefix=prefix
        )
        total_slices += count
    
    print(f"\nDone! Extracted {total_slices} slices to {output_dir}")
    print(f"These can be added to your 'notumor' training folder")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Prepare IXI dataset for training')
    parser.add_argument('--ixi-dir', type=str, default=r'D:\IXI-T1',
                        help='Path to IXI-T1 directory')
    parser.add_argument('--output-dir', type=str, required=True,
                        help='Output directory for extracted slices (should be notumor folder)')
    parser.add_argument('--max-subjects', type=int, default=None,
                        help='Maximum number of subjects to process')
    parser.add_argument('--slices-per-subject', type=int, default=5,
                        help='Number of slices to extract per subject')
    
    args = parser.parse_args()
    
    prepare_ixi_dataset(
        ixi_dir=args.ixi_dir,
        output_dir=args.output_dir,
        max_subjects=args.max_subjects,
        slices_per_subject=args.slices_per_subject
    )

