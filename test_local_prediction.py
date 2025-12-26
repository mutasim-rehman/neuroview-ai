"""
Test the prediction pipeline locally to verify it works before deployment.
This helps isolate issues between local vs Render deployment.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'ai-training'))

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
import gc
import traceback

# Import from api_server (the actual code that runs on Render)
from config.config import config
from models.classification_model import create_classification_model
import torchvision.transforms as transforms

# Constants (same as api_server.py)
CLASS_NAMES = ['glioma', 'meningioma', 'notumor', 'pituitary']
NUM_CLASSES = len(CLASS_NAMES)


def get_memory_mb():
    """Get current process memory in MB."""
    try:
        import psutil
        return psutil.Process().memory_info().rss / 1024 / 1024
    except:
        return None


def load_model():
    """Load the model (same as api_server.py)."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model = create_classification_model(
        in_channels=3,
        num_classes=NUM_CLASSES,
        base_channels=64,
        use_pretrained=False,
        model_type='custom'
    )
    
    # Try inference checkpoint first
    # Use absolute path based on script location
    ai_training_dir = os.path.join(os.path.dirname(__file__), 'ai-training')
    checkpoints_dir = os.path.join(ai_training_dir, 'checkpoints')
    
    inference_checkpoint_path = os.path.join(checkpoints_dir, "best_model_inference.pth")
    checkpoint_path = os.path.join(checkpoints_dir, "best_model.pth")
    
    if os.path.exists(inference_checkpoint_path):
        checkpoint_path = inference_checkpoint_path
        print(f"Using inference-optimized checkpoint")
    
    print(f"Loading checkpoint ({os.path.getsize(checkpoint_path) / 1024 / 1024:.1f} MB)...")
    
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    gc.collect()
    
    print(f"Model loaded successfully!")
    mem = get_memory_mb()
    if mem:
        print(f"Memory after loading model: {mem:.1f} MB")
    
    return model, device


def preprocess_image(image: Image.Image) -> torch.Tensor:
    """Preprocess image (same as api_server.py)."""
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    tensor = transform(image)
    return tensor.unsqueeze(0)


def predict(model, device, image_tensor):
    """Run prediction (same as api_server.py)."""
    image_tensor = image_tensor.to(device)
    
    with torch.no_grad():
        logits = model(image_tensor)
        probabilities = F.softmax(logits, dim=1)
        predicted_class_idx = torch.argmax(probabilities, dim=1).item()
    
    probs = probabilities[0].cpu().numpy()
    
    return {
        'prediction': CLASS_NAMES[predicted_class_idx],
        'confidence': float(probs[predicted_class_idx]),
        'class_probabilities': {
            CLASS_NAMES[i]: float(probs[i]) 
            for i in range(len(CLASS_NAMES))
        }
    }


def test_synthetic_prediction():
    """Test prediction with synthetic data."""
    print("\n" + "=" * 60)
    print("Testing Synthetic Prediction Pipeline")
    print("=" * 60)
    
    try:
        # Step 1: Load model
        print("\n[Step 1] Loading model...")
        mem_start = get_memory_mb()
        if mem_start:
            print(f"  Memory before: {mem_start:.1f} MB")
        
        model, device = load_model()
        
        mem_after_load = get_memory_mb()
        if mem_after_load:
            print(f"  Memory after model load: {mem_after_load:.1f} MB")
        
        # Step 2: Create synthetic 2D slice
        print("\n[Step 2] Creating synthetic 2D slice (64x64)...")
        slice_2d = np.random.rand(64, 64).astype(np.float32) * 255
        slice_normalized = slice_2d.astype(np.uint8)
        
        # Step 3: Convert to PIL Image
        print("[Step 3] Converting to PIL Image...")
        image = Image.fromarray(slice_normalized, mode='L').convert('RGB')
        print(f"  Image size: {image.size}, mode: {image.mode}")
        
        # Step 4: Preprocess
        print("[Step 4] Preprocessing image...")
        image_tensor = preprocess_image(image)
        print(f"  Tensor shape: {image_tensor.shape}")
        
        mem_after_preprocess = get_memory_mb()
        if mem_after_preprocess:
            print(f"  Memory after preprocess: {mem_after_preprocess:.1f} MB")
        
        # Step 5: Run prediction
        print("[Step 5] Running prediction...")
        result = predict(model, device, image_tensor)
        
        mem_after_predict = get_memory_mb()
        if mem_after_predict:
            print(f"  Memory after prediction: {mem_after_predict:.1f} MB")
        
        print("\n[SUCCESS] Prediction completed!")
        print(f"  Prediction: {result['prediction']}")
        print(f"  Confidence: {result['confidence']:.4f}")
        print(f"  All probabilities: {result['class_probabilities']}")
        
        # Cleanup
        del image_tensor
        del image
        del slice_2d
        gc.collect()
        
        mem_after_cleanup = get_memory_mb()
        if mem_after_cleanup:
            print(f"\n  Memory after cleanup: {mem_after_cleanup:.1f} MB")
        
        return True
        
    except Exception as e:
        print(f"\n[ERROR] Prediction failed: {e}")
        traceback.print_exc()
        return False


def test_base64_prediction():
    """Test prediction from base64-encoded volume (similar to /predict_base64 endpoint)."""
    print("\n" + "=" * 60)
    print("Testing Base64 Prediction Pipeline")
    print("=" * 60)
    
    import base64
    
    try:
        # Step 1: Load model
        print("\n[Step 1] Loading model...")
        model, device = load_model()
        
        mem_after_load = get_memory_mb()
        if mem_after_load:
            print(f"  Memory after model load: {mem_after_load:.1f} MB")
        
        # Step 2: Create synthetic volume and encode as base64
        print("\n[Step 2] Creating synthetic volume (64x64x64)...")
        volume = np.random.rand(64, 64, 64).astype(np.float32)
        print(f"  Volume shape: {volume.shape}, dtype: {volume.dtype}")
        print(f"  Volume memory: {volume.nbytes / 1024 / 1024:.2f} MB")
        
        volume_bytes = volume.tobytes()
        volume_b64 = base64.b64encode(volume_bytes).decode('utf-8')
        print(f"  Base64 size: {len(volume_b64) / 1024:.1f} KB")
        
        mem_after_encode = get_memory_mb()
        if mem_after_encode:
            print(f"  Memory after encoding: {mem_after_encode:.1f} MB")
        
        # Step 3: Decode (simulating what the endpoint does)
        print("\n[Step 3] Decoding base64...")
        decoded = base64.b64decode(volume_b64)
        volume_decoded = np.frombuffer(decoded, dtype=np.float32).reshape((64, 64, 64))
        
        # Step 4: Extract 2D slice
        print("[Step 4] Extracting 2D slice...")
        dims = volume_decoded.shape
        max_dim_idx = np.argmax(dims)
        slice_idx = dims[max_dim_idx] // 2
        
        if max_dim_idx == 0:
            slice_2d = volume_decoded[slice_idx, :, :]
        elif max_dim_idx == 1:
            slice_2d = volume_decoded[:, slice_idx, :]
        else:
            slice_2d = volume_decoded[:, :, slice_idx]
        
        print(f"  Slice shape: {slice_2d.shape}")
        
        # Step 5: Normalize and convert to image
        print("[Step 5] Converting to PIL Image...")
        slice_min, slice_max = slice_2d.min(), slice_2d.max()
        if slice_max > slice_min:
            slice_normalized = ((slice_2d - slice_min) / (slice_max - slice_min) * 255).astype(np.uint8)
        else:
            slice_normalized = np.zeros_like(slice_2d, dtype=np.uint8)
        
        image = Image.fromarray(slice_normalized, mode='L').convert('RGB')
        
        # Free volume memory
        del volume, volume_decoded
        gc.collect()
        
        mem_after_convert = get_memory_mb()
        if mem_after_convert:
            print(f"  Memory after conversion: {mem_after_convert:.1f} MB")
        
        # Step 6: Preprocess
        print("[Step 6] Preprocessing image...")
        image_tensor = preprocess_image(image)
        
        # Step 7: Predict
        print("[Step 7] Running prediction...")
        result = predict(model, device, image_tensor)
        
        mem_after_predict = get_memory_mb()
        if mem_after_predict:
            print(f"  Memory after prediction: {mem_after_predict:.1f} MB")
        
        print("\n[SUCCESS] Base64 prediction completed!")
        print(f"  Prediction: {result['prediction']}")
        print(f"  Confidence: {result['confidence']:.4f}")
        
        # Cleanup
        del image_tensor, image
        gc.collect()
        
        return True
        
    except Exception as e:
        print(f"\n[ERROR] Base64 prediction failed: {e}")
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("=" * 60)
    print("Local Prediction Test")
    print("=" * 60)
    
    initial_mem = get_memory_mb()
    if initial_mem:
        print(f"Initial memory: {initial_mem:.1f} MB")
    
    # Test 1: Synthetic prediction
    success1 = test_synthetic_prediction()
    
    # Test 2: Base64 prediction
    success2 = test_base64_prediction()
    
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    print(f"Synthetic prediction: {'PASS' if success1 else 'FAIL'}")
    print(f"Base64 prediction: {'PASS' if success2 else 'FAIL'}")
    
    final_mem = get_memory_mb()
    if final_mem:
        print(f"\nFinal memory: {final_mem:.1f} MB")

