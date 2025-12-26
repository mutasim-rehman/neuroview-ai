"""
Create a smaller inference-only checkpoint by removing optimizer state.
This significantly reduces memory usage on deployment servers.
"""

import torch
import os
from pathlib import Path

def create_inference_checkpoint(input_path: str, output_path: str = None):
    """
    Create an inference-only checkpoint by keeping only model weights.
    
    Args:
        input_path: Path to full training checkpoint
        output_path: Path for inference checkpoint (default: adds '_inference' suffix)
    """
    if output_path is None:
        base = Path(input_path)
        output_path = str(base.parent / f"{base.stem}_inference{base.suffix}")
    
    print(f"Loading checkpoint: {input_path}")
    checkpoint = torch.load(input_path, map_location='cpu')
    
    print(f"Original checkpoint keys: {list(checkpoint.keys())}")
    
    # Get file sizes
    original_size = os.path.getsize(input_path) / (1024 * 1024)
    print(f"Original size: {original_size:.2f} MB")
    
    # Create inference-only checkpoint (only model weights)
    inference_checkpoint = {
        'model_state_dict': checkpoint['model_state_dict']
    }
    
    # Count parameters
    total_params = sum(p.numel() for p in checkpoint['model_state_dict'].values())
    print(f"Model parameters: {total_params:,}")
    
    # Save inference checkpoint
    print(f"Saving inference checkpoint: {output_path}")
    torch.save(inference_checkpoint, output_path)
    
    new_size = os.path.getsize(output_path) / (1024 * 1024)
    print(f"New size: {new_size:.2f} MB")
    print(f"Size reduction: {(1 - new_size/original_size) * 100:.1f}%")
    
    return output_path


if __name__ == "__main__":
    checkpoint_dir = Path(__file__).parent / "checkpoints"
    input_file = checkpoint_dir / "best_model.pth"
    output_file = checkpoint_dir / "best_model_inference.pth"
    
    if not input_file.exists():
        print(f"Error: {input_file} not found")
        exit(1)
    
    create_inference_checkpoint(str(input_file), str(output_file))
    print("\nDone! Use 'best_model_inference.pth' for deployment.")

