"""
Flask API server for brain scan inference.
Provides endpoint to classify brain tumors into specific disease types.
"""

import os
import sys
import io
import base64
import numpy as np
import torch
import torch.nn.functional as F
from flask import Flask, request, jsonify
from flask_cors import CORS
from pathlib import Path
import traceback
from PIL import Image
import torchvision.transforms as transforms

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from config.config import config
from models.classification_model import create_classification_model
from data.preprocessing import load_nifti
import nibabel as nib

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend

# Global variables for model
model = None
device = None

# Disease class names (must match training order)
CLASS_NAMES = ['glioma', 'meningioma', 'notumor', 'pituitary']
NUM_CLASSES = len(CLASS_NAMES)


@app.route('/', methods=['GET'])
def index():
    """
    Root endpoint.
    
    This API is not meant to serve a web page, only JSON endpoints for the
    NeuroView AI frontend and other clients.
    """
    return jsonify({
        "message": "NeuroView AI Brain Tumor Classification API",
        "endpoints": {
            "health": "/health",
            "predict": "/predict",
            "predict_from_array": "/predict_from_array"
        },
        "classes": CLASS_NAMES,
        "status": "ok"
    }), 200


def load_model():
    """Load the trained classification model from checkpoint."""
    global model, device
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create classification model with same architecture as training
    model = create_classification_model(
        in_channels=3,  # RGB images
        num_classes=NUM_CLASSES,
        base_channels=64,
        use_pretrained=False,
        model_type='custom'
    )
    
    # Load checkpoint
    checkpoint_path = os.path.join(config.CHECKPOINT_DIR, "best_model.pth")
    
    if not os.path.exists(checkpoint_path):
        print(f"Warning: Model checkpoint not found at {checkpoint_path}")
        print("Please train the model first using main_train_diseases.py")
        return False
    
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.eval()  # Set to evaluation mode
        print(f"Classification model loaded successfully from {checkpoint_path}")
        print(f"Classes: {CLASS_NAMES}")
        return True
    except Exception as e:
        print(f"Error loading model: {e}")
        traceback.print_exc()
        return False


# Load model when app starts (works with both direct run and gunicorn)
print("Initializing API server...")
print("Loading model...")
if load_model():
    print("Model loaded successfully!")
else:
    print("Warning: Model failed to load. Predictions will fail.")
    print("Please ensure best_model.pth exists in ./checkpoints/ directory")


def volume_to_2d_image(volume: np.ndarray) -> Image.Image:
    """
    Convert 3D volume to 2D image slice for classification.
    
    Args:
        volume: 3D numpy array (D, H, W)
        
    Returns:
        PIL Image (RGB, 224x224)
    """
    # Handle different dimensions
    if len(volume.shape) == 3:
        # Extract middle slice along the first dimension (usually axial)
        slice_idx = volume.shape[0] // 2
        slice_2d = volume[slice_idx, :, :]
    elif len(volume.shape) == 2:
        slice_2d = volume
    else:
        # For 4D, take middle slice of first volume
        slice_idx = volume.shape[0] // 2
        slice_2d = volume[slice_idx, :, :, 0] if len(volume.shape) == 4 else volume[slice_idx, :, :]
    
    # Normalize to 0-255 range
    slice_min, slice_max = slice_2d.min(), slice_2d.max()
    if slice_max > slice_min:
        slice_normalized = ((slice_2d - slice_min) / (slice_max - slice_min) * 255).astype(np.uint8)
    else:
        slice_normalized = np.zeros_like(slice_2d, dtype=np.uint8)
    
    # Convert to PIL Image (grayscale) and then to RGB
    image = Image.fromarray(slice_normalized, mode='L').convert('RGB')
    return image


def preprocess_image_for_classification(image: Image.Image) -> torch.Tensor:
    """
    Preprocess image for classification model.
    
    Args:
        image: PIL Image (RGB)
        
    Returns:
        Preprocessed tensor (1, 3, 224, 224)
    """
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    tensor = transform(image)
    return tensor.unsqueeze(0)  # Add batch dimension


def predict_disease(image_tensor: torch.Tensor):
    """
    Predict brain tumor disease type using classification model.
    
    Args:
        image_tensor: Preprocessed image tensor (1, 3, 224, 224)
        
    Returns:
        Dictionary with prediction results
    """
    global model, device
    
    if model is None:
        return {
            'error': 'Model not loaded. Please train the model first.'
        }
    
    try:
        # Move tensor to device
        image_tensor = image_tensor.to(device)
        
        # Run inference
        with torch.no_grad():
            logits = model(image_tensor)
            probabilities = F.softmax(logits, dim=1)
            predicted_class_idx = torch.argmax(probabilities, dim=1).item()
        
        # Get probabilities for all classes
        probs = probabilities[0].cpu().numpy()
        
        # Create class probabilities dictionary
        class_probs = {
            CLASS_NAMES[i]: float(probs[i]) 
            for i in range(len(CLASS_NAMES))
        }
        
        # Get predicted class
        predicted_class = CLASS_NAMES[predicted_class_idx]
        confidence = float(probs[predicted_class_idx])
        
        return {
            'prediction': predicted_class,
            'confidence': confidence,
            'class_probabilities': class_probs,
            'all_classes': CLASS_NAMES
        }
        
    except Exception as e:
        print(f"Error during prediction: {e}")
        traceback.print_exc()
        return {
            'error': f'Prediction failed: {str(e)}'
        }


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        'status': 'ok',
        'model_loaded': model is not None,
        'device': str(device) if device else None
    })


@app.route('/predict', methods=['POST'])
def predict():
    """
    Predict endpoint. Accepts NIfTI file or volume data.
    Classifies brain tumors into: glioma, meningioma, notumor, pituitary.
    
    Expected input:
    - File upload (multipart/form-data) with key 'file'
    OR
    - JSON with base64 encoded volume data
    
    Returns:
    - JSON with disease classification results
    """
    try:
        # Check if file is uploaded
        if 'file' in request.files:
            file = request.files['file']
            if file.filename == '':
                return jsonify({'error': 'No file provided'}), 400
            
            # Save to temporary file
            temp_path = f"/tmp/{file.filename}"
            file.save(temp_path)
            
            # Load NIfTI file
            volume, header = load_nifti(temp_path)
            
            # Clean up temp file
            os.remove(temp_path)
            
        elif 'volume_data' in request.json:
            # Accept base64 encoded volume data
            volume_data_b64 = request.json['volume_data']
            volume = np.frombuffer(
                base64.b64decode(volume_data_b64),
                dtype=np.float32
            )
            
            # Reshape if shape provided
            if 'shape' in request.json:
                volume = volume.reshape(tuple(request.json['shape']))
            else:
                # Assume it's already in the right shape
                # This is a fallback - should provide shape
                volume = volume.reshape((128, 128, 128))
        else:
            return jsonify({'error': 'No file or volume_data provided'}), 400
        
        # Convert 3D volume to 2D image slice
        image = volume_to_2d_image(volume)
        
        # Preprocess image for classification
        image_tensor = preprocess_image_for_classification(image)
        
        # Run prediction
        result = predict_disease(image_tensor)
        
        return jsonify(result)
        
    except Exception as e:
        print(f"Error in /predict endpoint: {e}")
        traceback.print_exc()
        return jsonify({
            'error': f'Prediction failed: {str(e)}'
        }), 500


@app.route('/predict_from_array', methods=['POST'])
def predict_from_array():
    """
    Predict from numpy array (for direct volume data).
    
    Expected JSON:
    {
        "volume": [[[...]]],  # 3D numpy array as nested lists
        "shape": [128, 128, 128]
    }
    """
    try:
        data = request.json
        
        if 'volume' not in data:
            return jsonify({'error': 'No volume data provided'}), 400
        
        # Convert nested lists to numpy array
        volume = np.array(data['volume'], dtype=np.float32)
        
        # Convert 3D volume to 2D image slice
        image = volume_to_2d_image(volume)
        
        # Preprocess image for classification
        image_tensor = preprocess_image_for_classification(image)
        
        # Run prediction
        result = predict_disease(image_tensor)
        
        return jsonify(result)
        
    except Exception as e:
        print(f"Error in /predict_from_array endpoint: {e}")
        traceback.print_exc()
        return jsonify({
            'error': f'Prediction failed: {str(e)}'
        }), 500


if __name__ == '__main__':
    import os
    port = int(os.environ.get('PORT', 5000))
    
    print("Loading model...")
    if load_model():
        print("Model loaded successfully!")
        print(f"Starting Flask server on port {port}")
        app.run(host='0.0.0.0', port=port, debug=False)
    else:
        print("Failed to load model. Server will start but predictions will fail.")
        print("Please train the model first or check checkpoint path.")
        app.run(host='0.0.0.0', port=port, debug=False)
