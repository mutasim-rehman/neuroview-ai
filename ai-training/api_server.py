"""
Flask API server for brain scan inference.
Provides endpoint to predict if a brain scan is healthy or has defects.
"""

import os
import sys
import io
import base64
import numpy as np
import torch
from flask import Flask, request, jsonify
from flask_cors import CORS
from pathlib import Path
import traceback

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from config.config import config
from models.brain_model import create_model
from data.preprocessing import (
    load_nifti, 
    preprocess_volume, 
    normalize_intensity,
    resize_volume,
    volume_to_tensor
)
import nibabel as nib

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend

# Global variables for model
model = None
device = None


@app.route('/', methods=['GET'])
def index():
    """
    Root endpoint.
    
    This API is not meant to serve a web page, only JSON endpoints for the
    NeuroView AI frontend and other clients.
    """
    return jsonify({
        "message": "NeuroView AI Brain Health Prediction API",
        "endpoints": {
            "health": "/health",
            "predict": "/predict",
            "predict_from_array": "/predict_from_array"
        },
        "status": "ok"
    }), 200


def load_model():
    """Load the trained model from checkpoint."""
    global model, device
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create model with same architecture as training
    model = create_model(
        in_channels=config.IN_CHANNELS,
        feature_dim=config.FEATURE_DIM,
        base_channels=config.BASE_CHANNELS
    )
    
    # Load checkpoint
    checkpoint_path = os.path.join(config.CHECKPOINT_DIR, "best_model.pth")
    
    if not os.path.exists(checkpoint_path):
        print(f"Warning: Model checkpoint not found at {checkpoint_path}")
        print("Please train the model first using main_train_healthy.py")
        return False
    
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.eval()  # Set to evaluation mode
        print(f"Model loaded successfully from {checkpoint_path}")
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


def calculate_reconstruction_error(reconstructed, original):
    """
    Calculate reconstruction error to detect anomalies.
    Healthy scans should have low reconstruction error,
    while scans with defects should have higher error.
    """
    # Mean Squared Error
    mse = torch.mean((reconstructed - original) ** 2)
    
    # Mean Absolute Error
    mae = torch.mean(torch.abs(reconstructed - original))
    
    # Calculate per-voxel error
    error_map = torch.abs(reconstructed - original)
    max_error = torch.max(error_map)
    
    return {
        'mse': float(mse.item()),
        'mae': float(mae.item()),
        'max_error': float(max_error.item())
    }


def predict_health(volume_tensor, threshold_percentile=95.0):
    """
    Predict if brain scan is healthy or has defects.
    
    Args:
        volume_tensor: Preprocessed volume tensor (1, 1, D, H, W)
        threshold_percentile: Percentile threshold for anomaly detection
        
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
        volume_tensor = volume_tensor.to(device)
        
        # Run inference
        with torch.no_grad():
            reconstructed, encoded_features, projected_features = model(volume_tensor)
        
        # Calculate reconstruction error
        error_metrics = calculate_reconstruction_error(reconstructed, volume_tensor)
        
        # Use reconstruction error as anomaly score
        # Higher error = more likely to have defects
        anomaly_score = error_metrics['mse']
        
        # You can adjust this threshold based on validation results
        # For now, use a percentile-based approach
        # In production, this should be calibrated on validation data
        
        # Simple threshold: if MSE > 0.01, likely anomaly
        # This is a heuristic - should be calibrated from validation set
        is_healthy = anomaly_score < 0.01
        
        # Calculate confidence (inverse of error, normalized)
        confidence = min(1.0, max(0.0, 1.0 - (anomaly_score * 10)))
        
        return {
            'prediction': 'healthy' if is_healthy else 'defect',
            'confidence': float(confidence),
            'anomaly_score': float(anomaly_score),
            'error_metrics': error_metrics,
            'feature_vector': encoded_features.cpu().numpy().tolist()[0] if encoded_features is not None else None
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
    
    Expected input:
    - File upload (multipart/form-data) with key 'file'
    OR
    - JSON with base64 encoded volume data
    
    Returns:
    - JSON with prediction results
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
        
        # Preprocess volume (same as training)
        preprocessed = preprocess_volume(
            volume,
            target_shape=config.TARGET_SHAPE,
            normalize=config.NORMALIZE,
            clip_percentile=config.CLIP_PERCENTILE,
            augment=False  # No augmentation for inference
        )
        
        # Convert to tensor
        volume_tensor = volume_to_tensor(preprocessed, add_channel_dim=True)
        volume_tensor = volume_tensor.unsqueeze(0)  # Add batch dimension (B, C, D, H, W)
        
        # Run prediction
        result = predict_health(volume_tensor)
        
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
        
        # Preprocess volume
        preprocessed = preprocess_volume(
            volume,
            target_shape=config.TARGET_SHAPE,
            normalize=config.NORMALIZE,
            clip_percentile=config.CLIP_PERCENTILE,
            augment=False
        )
        
        # Convert to tensor
        volume_tensor = volume_to_tensor(preprocessed, add_channel_dim=True)
        volume_tensor = volume_tensor.unsqueeze(0)
        
        # Run prediction
        result = predict_health(volume_tensor)
        
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
