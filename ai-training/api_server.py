"""
Flask API server for brain scan inference.
Provides endpoint to classify brain tumors into specific disease types.
"""

import os
import sys
import io
import base64
import gc
import numpy as np
import torch
import torch.nn.functional as F
from flask import Flask, request, jsonify
from flask_cors import CORS
from pathlib import Path
import traceback
from PIL import Image
import torchvision.transforms as transforms
import tempfile
import logging

# Memory optimization: Set PyTorch to use less memory
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'max_split_size_mb:128')
torch.set_num_threads(1)  # Limit CPU threads to reduce memory

# Configure logging to ensure output is flushed
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from config.config import config
from models.classification_model import create_classification_model
from data.preprocessing import load_nifti, load_nifti_slice
import nibabel as nib

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend

# Increase max content length for large file uploads (50MB)
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024

# Increase request timeout for slow connections (Render free tier)
# This helps with large file uploads on slow connections
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0


@app.before_request
def log_request_info():
    """Log request information for debugging."""
    logger.info(f"Request: {request.method} {request.path}")
    logger.info(f"Content-Type: {request.content_type}")
    if request.method == 'POST':
        logger.info(f"Content-Length: {request.content_length}")
    sys.stdout.flush()


@app.errorhandler(413)
def request_entity_too_large(error):
    """Handle file too large error."""
    logger.error("File upload too large")
    sys.stdout.flush()
    return jsonify({'error': 'File too large. Maximum size is 50MB.'}), 413


@app.errorhandler(405)
def method_not_allowed(error):
    """Handle method not allowed errors."""
    logger.warning(f"Method not allowed: {request.method} {request.path}")
    sys.stdout.flush()
    return jsonify({
        'error': 'Method not allowed',
        'message': f'The {request.method} method is not supported for this endpoint.',
        'allowed_methods': ['POST'] if '/predict' in request.path else []
    }), 405


@app.errorhandler(500)
def internal_error(error):
    """Handle internal server errors."""
    logger.error(f"Internal server error: {error}")
    logger.error(traceback.format_exc())
    sys.stdout.flush()
    sys.stderr.flush()
    return jsonify({
        'error': 'Internal server error',
        'message': str(error) if os.environ.get('DEBUG', 'False') == 'True' else 'An error occurred processing your request'
    }), 500


@app.errorhandler(Exception)
def handle_exception(e):
    """Global exception handler to prevent 502 errors."""
    logger.error(f"Unhandled exception: {e}")
    logger.error(traceback.format_exc())
    sys.stdout.flush()
    sys.stderr.flush()
    return jsonify({
        'error': 'An unexpected error occurred',
        'message': str(e) if os.environ.get('DEBUG', 'False') == 'True' else 'Please try again later'
    }), 500

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
    try:
        return jsonify({
            "message": "NeuroView AI Brain Tumor Classification API",
            "endpoints": {
                "health": "/health",
                "predict": "/predict",
                "predict_from_array": "/predict_from_array",
                "predict_base64": "/predict_base64",
                "debug_predict": "/debug/predict",
                "debug_upload": "/debug/upload"
            },
            "classes": CLASS_NAMES,
            "status": "ok"
        }), 200
    except Exception as e:
        logger.error(f"Error in index endpoint: {e}")
        logger.error(traceback.format_exc())
        sys.stdout.flush()
        return jsonify({
            "error": "An error occurred",
            "status": "error"
        }), 500


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
    
    # Try inference checkpoint first (smaller, faster loading)
    inference_checkpoint_path = os.path.join(config.CHECKPOINT_DIR, "best_model_inference.pth")
    checkpoint_path = os.path.join(config.CHECKPOINT_DIR, "best_model.pth")
    
    # Prefer inference checkpoint if it exists
    if os.path.exists(inference_checkpoint_path):
        checkpoint_path = inference_checkpoint_path
        print(f"Using inference-optimized checkpoint: {checkpoint_path}")
    elif not os.path.exists(checkpoint_path):
        print(f"Warning: Model checkpoint not found at {checkpoint_path}")
        print("Please train the model first using main_train_diseases.py")
        return False
    
    try:
        print(f"Loading checkpoint ({os.path.getsize(checkpoint_path) / 1024 / 1024:.1f} MB)...")
        sys.stdout.flush()
        
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.eval()  # Set to evaluation mode
        
        # Force garbage collection after loading
        gc.collect()
        
        print(f"Classification model loaded successfully from {checkpoint_path}")
        print(f"Classes: {CLASS_NAMES}")
        
        # Log memory usage if psutil available
        try:
            import psutil
            mem = psutil.Process().memory_info().rss / 1024 / 1024
            print(f"Memory after loading model: {mem:.1f} MB")
        except ImportError:
            pass
        
        sys.stdout.flush()
        return True
    except Exception as e:
        print(f"Error loading model: {e}")
        traceback.print_exc()
        return False


# Load model when app starts (works with both direct run and gunicorn)
# Wrap in try-except to ensure app starts even if model loading fails
try:
    print("Initializing API server...")
    print("Loading model...")
    sys.stdout.flush()
    if load_model():
        print("Model loaded successfully!")
        sys.stdout.flush()
    else:
        print("Warning: Model failed to load. Predictions will fail.")
        print("Please ensure best_model.pth exists in ./checkpoints/ directory")
        sys.stdout.flush()
except Exception as e:
    print(f"Error during model initialization: {e}")
    print("Server will start but predictions will fail.")
    traceback.print_exc()
    sys.stdout.flush()
    model = None
    device = None


def volume_to_2d_image(volume: np.ndarray) -> Image.Image:
    """
    Convert 3D volume to 2D image slice for classification.
    
    Args:
        volume: 3D numpy array (D, H, W) or (H, W, D) or other shapes
        
    Returns:
        PIL Image (RGB)
    """
    logger.info(f"Volume shape: {volume.shape}, dtype: {volume.dtype}")
    sys.stdout.flush()
    
    # Handle different dimensions
    if len(volume.shape) == 3:
        # Extract middle slice - try different dimensions
        # NIfTI files can have different axis orders
        # Try to find the largest dimension and slice along it
        dims = volume.shape
        max_dim_idx = np.argmax(dims)
        slice_idx = dims[max_dim_idx] // 2
        
        if max_dim_idx == 0:
            slice_2d = volume[slice_idx, :, :]
        elif max_dim_idx == 1:
            slice_2d = volume[:, slice_idx, :]
        else:
            slice_2d = volume[:, :, slice_idx]
            
    elif len(volume.shape) == 2:
        slice_2d = volume
    elif len(volume.shape) == 4:
        # For 4D, take middle slice of first volume
        slice_idx = volume.shape[0] // 2
        slice_2d = volume[slice_idx, :, :, 0]
    else:
        raise ValueError(f"Unsupported volume shape: {volume.shape}")
    
    logger.info(f"Extracted 2D slice shape: {slice_2d.shape}")
    sys.stdout.flush()
    
    # Normalize to 0-255 range
    slice_min, slice_max = slice_2d.min(), slice_2d.max()
    if slice_max > slice_min:
        slice_normalized = ((slice_2d - slice_min) / (slice_max - slice_min) * 255).astype(np.uint8)
    else:
        slice_normalized = np.zeros_like(slice_2d, dtype=np.uint8)
    
    # Convert to PIL Image (grayscale) and then to RGB
    image = Image.fromarray(slice_normalized, mode='L').convert('RGB')
    logger.info(f"Final image size: {image.size}, mode: {image.mode}")
    sys.stdout.flush()
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
    try:
        import psutil
        process = psutil.Process()
        memory_info = process.memory_info()
        
        return jsonify({
            'status': 'ok',
            'model_loaded': model is not None,
            'device': str(device) if device else None,
            'memory_mb': round(memory_info.rss / 1024 / 1024, 2),
            'memory_percent': round(process.memory_percent(), 2)
        }), 200
    except ImportError:
        # psutil not available
        return jsonify({
            'status': 'ok',
            'model_loaded': model is not None,
            'device': str(device) if device else None
        }), 200
    except Exception as e:
        logger.error(f"Error in health check: {e}")
        logger.error(traceback.format_exc())
        sys.stdout.flush()
        return jsonify({
            'status': 'error',
            'error': str(e)
        }), 500


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
    # Check HTTP method
    if request.method != 'POST':
        return jsonify({
            'error': 'Method not allowed',
            'message': 'This endpoint only accepts POST requests. Please use POST with a file upload or JSON data.',
            'allowed_methods': ['POST'],
            'usage': {
                'file_upload': 'POST /predict with multipart/form-data containing a "file" field',
                'json_data': 'POST /predict with JSON body containing "volume_data" field'
            }
        }), 405
    
    temp_path = None
    skip_volume_conversion = False
    image = None
    volume = None
    image_tensor = None
    
    try:
        # Check if model is loaded
        if model is None:
            logger.error("Model not loaded - cannot process prediction")
            sys.stdout.flush()
            return jsonify({
                'error': 'Model not loaded. Server may still be initializing.'
            }), 503  # Service Unavailable
        
        logger.info("Received prediction request")
        logger.info(f"Request method: {request.method}, Content-Type: {request.content_type}")
        logger.info(f"Content-Length: {request.content_length}")
        sys.stdout.flush()
        
        # Ensure request data is available
        if request.content_length and request.content_length > 50 * 1024 * 1024:
            return jsonify({'error': 'File too large. Maximum size is 50MB.'}), 413
        
        # Log that we're starting to parse files (this can be slow for large uploads)
        logger.info("Starting to parse uploaded file (this may take time for large files)...")
        sys.stdout.flush()
        
        # Safely check for file upload
        has_file = False
        has_json_data = False
        
        try:
            # Check if request has files (multipart/form-data)
            if request.content_type and 'multipart/form-data' in request.content_type:
                has_file = 'file' in request.files
            elif request.files:
                has_file = 'file' in request.files
        except Exception as e:
            logger.warning(f"Error checking request.files: {e}")
            has_file = False
        
        try:
            # Check if request has JSON data
            if request.is_json and request.json:
                has_json_data = 'volume_data' in request.json
        except Exception as e:
            logger.warning(f"Error checking request.json: {e}")
            has_json_data = False
        
        # Check if file is uploaded
        if has_file:
            file = request.files['file']
            if file.filename == '':
                return jsonify({'error': 'No file provided'}), 400
            
            # Save to temporary file (use tempfile for cross-platform compatibility)
            temp_dir = tempfile.gettempdir()
            temp_path = os.path.join(temp_dir, f"nifti_{os.urandom(8).hex()}.nii")
            
            logger.info(f"Saving file to: {temp_path}")
            sys.stdout.flush()
            
            try:
                # Get the file object
                file = request.files['file']
                if not file or file.filename == '':
                    return jsonify({'error': 'No file provided or empty filename'}), 400
                
                file.save(temp_path)
                file_size = os.path.getsize(temp_path)
                logger.info(f"File saved, size: {file_size} bytes")
                sys.stdout.flush()
                
                # Load NIfTI file - use memory-efficient slice loading
                logger.info("Loading NIfTI file (memory-efficient slice loading)...")
                sys.stdout.flush()
                try:
                    # Load only a single slice instead of entire volume to save memory
                    # This is critical for Render's 512MB RAM limit
                    slice_2d, header = load_nifti_slice(temp_path, slice_axis=2, slice_idx=None)
                    logger.info(f"NIfTI slice loaded, slice shape: {slice_2d.shape}, dtype: {slice_2d.dtype}, memory: {slice_2d.nbytes / 1024 / 1024:.2f} MB")
                    logger.info(f"Original volume shape: {header.get('shape', 'unknown')}")
                    sys.stdout.flush()
                    
                    # Clean up temp file immediately after loading
                    try:
                        os.remove(temp_path)
                        logger.info("Temp file cleaned up after loading")
                        temp_path = None  # Mark as cleaned
                    except Exception as e:
                        logger.warning(f"Could not remove temp file: {e}")
                    sys.stdout.flush()
                    
                    # Convert slice directly to image (skip volume_to_2d_image since we already have a 2D slice)
                    logger.info(f"Converting 2D slice (shape: {slice_2d.shape}) to image...")
                    sys.stdout.flush()
                    
                    # Normalize to 0-255 range
                    slice_min, slice_max = slice_2d.min(), slice_2d.max()
                    if slice_max > slice_min:
                        slice_normalized = ((slice_2d - slice_min) / (slice_max - slice_min) * 255).astype(np.uint8)
                    else:
                        slice_normalized = np.zeros_like(slice_2d, dtype=np.uint8)
                    
                    # Convert to PIL Image (grayscale) and then to RGB
                    image = Image.fromarray(slice_normalized, mode='L').convert('RGB')
                    logger.info(f"Image created, size: {image.size}, mode: {image.mode}")
                    sys.stdout.flush()
                    
                    # Skip volume_to_2d_image step - we already have the image
                    skip_volume_conversion = True
                    
                except Exception as e:
                    logger.error(f"Failed to load NIfTI file: {e}")
                    logger.error(traceback.format_exc())
                    sys.stdout.flush()
                    raise
                
            except Exception as e:
                logger.error(f"Error loading NIfTI file: {e}")
                logger.error(traceback.format_exc())
                sys.stdout.flush()
                raise
            
        elif has_json_data:
            try:
                # Accept base64 encoded volume data
                json_data = request.json
                if not json_data or 'volume_data' not in json_data:
                    return jsonify({'error': 'Invalid JSON: volume_data field is required'}), 400
                
                volume_data_b64 = json_data['volume_data']
                decoded_data = base64.b64decode(volume_data_b64)
                volume = np.frombuffer(decoded_data, dtype=np.float32)
                
                # Reshape if shape provided
                if 'shape' in json_data:
                    volume = volume.reshape(tuple(json_data['shape']))
                else:
                    # Assume it's already in the right shape
                    # This is a fallback - should provide shape
                    volume = volume.reshape((128, 128, 128))
            except Exception as e:
                logger.error(f"Error processing JSON volume data: {e}")
                logger.error(traceback.format_exc())
                sys.stdout.flush()
                return jsonify({
                    'error': 'Failed to process volume_data',
                    'message': str(e)
                }), 400
        else:
            return jsonify({
                'error': 'No file or volume_data provided',
                'message': 'Please provide either a file upload (multipart/form-data with "file" field) or JSON data with "volume_data" field',
                'content_type': request.content_type
            }), 400
        
        # Convert 3D volume to 2D image slice (only if we didn't already load a slice)
        if not skip_volume_conversion:
            logger.info(f"Converting 3D volume (shape: {volume.shape}) to 2D image...")
            sys.stdout.flush()
            image = volume_to_2d_image(volume)
            logger.info(f"Image converted, size: {image.size}")
            sys.stdout.flush()
        
        # Preprocess image for classification
        logger.info("Preprocessing image for classification...")
        sys.stdout.flush()
        image_tensor = preprocess_image_for_classification(image)
        logger.info(f"Image tensor shape: {image_tensor.shape}")
        sys.stdout.flush()
        
        # Run prediction
        logger.info("Running prediction...")
        sys.stdout.flush()
        result = predict_disease(image_tensor)
        logger.info(f"Prediction completed: {result.get('prediction', 'unknown')}")
        sys.stdout.flush()
        
        # Explicit memory cleanup to prevent OOM on Render's 512MB limit
        if image_tensor is not None:
            del image_tensor
        if image is not None:
            del image
        if volume is not None:
            del volume
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Ensure we return a valid JSON response
        if not isinstance(result, dict):
            result = {'error': 'Invalid prediction result'}
        
        response = jsonify(result)
        sys.stdout.flush()
        return response
        
    except Exception as e:
        error_msg = f'Prediction failed: {str(e)}'
        logger.error(f"Error in /predict endpoint: {error_msg}")
        logger.error(traceback.format_exc())
        sys.stdout.flush()
        sys.stderr.flush()
        
        # Clean up any remaining resources
        try:
            if image_tensor is not None:
                del image_tensor
            if image is not None:
                del image
            if volume is not None:
                del volume
            gc.collect()
        except:
            pass
        
        return jsonify({
            'error': error_msg,
            'traceback': traceback.format_exc() if os.environ.get('DEBUG', 'False') == 'True' else None
        }), 500
    finally:
        # Ensure temp file is cleaned up even if there was an error
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
                logger.info("Temp file cleaned up in finally block")
            except Exception as e:
                logger.warning(f"Could not remove temp file in finally: {e}")
            sys.stdout.flush()


@app.route('/debug/predict', methods=['GET'])
def debug_predict():
    """
    Debug endpoint to test the full prediction pipeline with minimal synthetic data.
    This isolates the prediction logic from file upload/transfer issues.
    """
    step = "init"
    mem_before = None
    mem_after_preprocess = None
    mem_after_predict = None
    
    try:
        step = "log_start"
        logger.info("=== DEBUG PREDICT START ===")
        sys.stdout.flush()
        
        step = "check_model"
        if model is None:
            return jsonify({'error': 'Model not loaded', 'step': step}), 503
        
        step = "memory_before"
        # Check memory before
        try:
            import psutil
            mem_before = psutil.Process().memory_info().rss / 1024 / 1024
            logger.info(f"Memory before: {mem_before:.2f} MB")
        except Exception as mem_e:
            logger.warning(f"Could not get memory: {mem_e}")
            mem_before = None
        sys.stdout.flush()
        
        step = "create_slice"
        # Create tiny synthetic 2D slice (much smaller than real data)
        logger.info("Creating synthetic 2D slice (64x64)...")
        sys.stdout.flush()
        
        slice_2d = np.random.rand(64, 64).astype(np.float32) * 255
        slice_normalized = slice_2d.astype(np.uint8)
        
        step = "create_image"
        logger.info("Converting to PIL Image...")
        sys.stdout.flush()
        
        image = Image.fromarray(slice_normalized, mode='L').convert('RGB')
        
        step = "preprocess"
        logger.info("Preprocessing image...")
        sys.stdout.flush()
        
        image_tensor = preprocess_image_for_classification(image)
        
        logger.info(f"Image tensor shape: {image_tensor.shape}")
        sys.stdout.flush()
        
        step = "memory_after_preprocess"
        # Check memory after preprocessing
        try:
            import psutil
            mem_after_preprocess = psutil.Process().memory_info().rss / 1024 / 1024
            logger.info(f"Memory after preprocess: {mem_after_preprocess:.2f} MB")
        except Exception as mem_e:
            logger.warning(f"Could not get memory: {mem_e}")
            mem_after_preprocess = None
        sys.stdout.flush()
        
        step = "predict"
        logger.info("Running prediction...")
        sys.stdout.flush()
        
        # Run actual prediction
        result = predict_disease(image_tensor)
        
        step = "log_result"
        logger.info(f"Prediction result: {result.get('prediction', 'unknown')}")
        sys.stdout.flush()
        
        step = "memory_after_predict"
        # Check memory after prediction
        try:
            import psutil
            mem_after_predict = psutil.Process().memory_info().rss / 1024 / 1024
            logger.info(f"Memory after predict: {mem_after_predict:.2f} MB")
        except Exception as mem_e:
            logger.warning(f"Could not get memory: {mem_e}")
            mem_after_predict = None
        sys.stdout.flush()
        
        step = "cleanup"
        # Cleanup
        del image_tensor
        del image
        del slice_2d
        gc.collect()
        
        step = "return"
        logger.info("=== DEBUG PREDICT SUCCESS ===")
        sys.stdout.flush()
        
        return jsonify({
            'status': 'success',
            'prediction': result,
            'memory_mb': {
                'before': round(mem_before, 2) if mem_before else None,
                'after_preprocess': round(mem_after_preprocess, 2) if mem_after_preprocess else None,
                'after_predict': round(mem_after_predict, 2) if mem_after_predict else None
            }
        }), 200
        
    except Exception as e:
        logger.error(f"Debug predict error at step '{step}': {e}")
        logger.error(traceback.format_exc())
        sys.stdout.flush()
        return jsonify({
            'status': 'error',
            'step': step,
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500


@app.route('/debug/upload', methods=['POST'])
def debug_upload():
    """
    Debug endpoint to test file upload without full prediction.
    This helps diagnose if the issue is upload or processing.
    """
    temp_path = None
    try:
        logger.info("=== DEBUG UPLOAD START ===")
        logger.info(f"Content-Type: {request.content_type}")
        logger.info(f"Content-Length: {request.content_length}")
        sys.stdout.flush()
        
        # Check memory before processing
        try:
            import psutil
            process = psutil.Process()
            mem_before = process.memory_info().rss / 1024 / 1024
            logger.info(f"Memory before: {mem_before:.2f} MB")
        except ImportError:
            mem_before = None
        sys.stdout.flush()
        
        if 'file' not in request.files:
            return jsonify({'error': 'No file in request', 'step': 'file_check'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'Empty filename', 'step': 'filename_check'}), 400
        
        logger.info(f"File received: {file.filename}")
        sys.stdout.flush()
        
        # Save to temp file
        temp_dir = tempfile.gettempdir()
        temp_path = os.path.join(temp_dir, f"debug_{os.urandom(4).hex()}.nii")
        
        logger.info(f"Saving to: {temp_path}")
        sys.stdout.flush()
        
        file.save(temp_path)
        file_size = os.path.getsize(temp_path)
        logger.info(f"File saved, size: {file_size} bytes ({file_size/1024/1024:.2f} MB)")
        sys.stdout.flush()
        
        # Try to load just the header (not the data)
        logger.info("Loading NIfTI header only...")
        sys.stdout.flush()
        
        nifti_img = nib.load(temp_path)
        shape = nifti_img.shape
        logger.info(f"NIfTI shape: {shape}")
        sys.stdout.flush()
        
        # Check memory after loading header
        try:
            mem_after_header = psutil.Process().memory_info().rss / 1024 / 1024
            logger.info(f"Memory after header: {mem_after_header:.2f} MB")
        except:
            mem_after_header = None
        sys.stdout.flush()
        
        # Load a single slice
        logger.info("Loading single slice...")
        sys.stdout.flush()
        
        slice_idx = shape[2] // 2
        slice_img = nifti_img.slicer[:, :, slice_idx:slice_idx+1]
        slice_data = slice_img.get_fdata()
        slice_array = np.squeeze(slice_data).astype(np.float32)
        
        logger.info(f"Slice loaded, shape: {slice_array.shape}")
        sys.stdout.flush()
        
        # Check memory after loading slice
        try:
            mem_after_slice = psutil.Process().memory_info().rss / 1024 / 1024
            logger.info(f"Memory after slice: {mem_after_slice:.2f} MB")
        except:
            mem_after_slice = None
        sys.stdout.flush()
        
        # Cleanup
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)
            temp_path = None
        
        del slice_data
        del slice_array
        gc.collect()
        
        logger.info("=== DEBUG UPLOAD SUCCESS ===")
        sys.stdout.flush()
        
        return jsonify({
            'status': 'success',
            'file_size_bytes': file_size,
            'file_size_mb': round(file_size / 1024 / 1024, 2),
            'nifti_shape': shape,
            'memory_mb': {
                'before': round(mem_before, 2) if mem_before else None,
                'after_header': round(mem_after_header, 2) if mem_after_header else None,
                'after_slice': round(mem_after_slice, 2) if mem_after_slice else None
            }
        }), 200
        
    except Exception as e:
        logger.error(f"Debug upload error: {e}")
        logger.error(traceback.format_exc())
        sys.stdout.flush()
        return jsonify({
            'status': 'error',
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500
    finally:
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except:
                pass


@app.route('/predict_from_array', methods=['POST'])
def predict_from_array():
    """
    Predict from numpy array (for direct volume data).
    
    Expected JSON:
    {
        "volume": [[[...]]],  # 3D numpy array as nested lists (SMALL volumes only!)
        "shape": [128, 128, 128]
    }
    
    WARNING: This endpoint is memory-intensive for large volumes.
    For production use with large data, use /predict with file upload
    or /predict_base64 with base64-encoded numpy array.
    """
    image = None
    image_tensor = None
    volume = None
    
    try:
        # Check if model is loaded
        if model is None:
            logger.error("Model not loaded - cannot process prediction")
            sys.stdout.flush()
            return jsonify({
                'error': 'Model not loaded. Server may still be initializing.'
            }), 503  # Service Unavailable
        
        # Check content length - limit to 10MB for JSON arrays
        if request.content_length and request.content_length > 10 * 1024 * 1024:
            return jsonify({
                'error': 'Request too large. Use /predict with file upload for large data.',
                'max_size_mb': 10
            }), 413
        
        if not request.is_json:
            return jsonify({'error': 'Request must be JSON'}), 400
        
        logger.info("Parsing JSON request...")
        sys.stdout.flush()
        
        data = request.json
        if data is None:
            return jsonify({'error': 'Invalid JSON data'}), 400
        
        if 'volume' not in data:
            return jsonify({'error': 'No volume data provided'}), 400
        
        logger.info("Converting nested lists to numpy array...")
        sys.stdout.flush()
        
        # Convert nested lists to numpy array
        volume = np.array(data['volume'], dtype=np.float32)
        
        logger.info(f"Volume shape: {volume.shape}, converting to 2D image...")
        sys.stdout.flush()
        
        # Convert 3D volume to 2D image slice
        image = volume_to_2d_image(volume)
        
        # Free volume memory immediately
        del volume
        volume = None
        gc.collect()
        
        logger.info("Preprocessing image...")
        sys.stdout.flush()
        
        # Preprocess image for classification
        image_tensor = preprocess_image_for_classification(image)
        
        # Free image memory
        del image
        image = None
        gc.collect()
        
        logger.info("Running prediction...")
        sys.stdout.flush()
        
        # Run prediction
        result = predict_disease(image_tensor)
        
        # Clean up
        del image_tensor
        image_tensor = None
        gc.collect()
        
        logger.info(f"Prediction complete: {result.get('prediction', 'unknown')}")
        sys.stdout.flush()
        
        if not isinstance(result, dict):
            result = {'error': 'Invalid prediction result'}
        
        return jsonify(result), 200
        
    except Exception as e:
        logger.error(f"Error in /predict_from_array endpoint: {e}")
        logger.error(traceback.format_exc())
        sys.stdout.flush()
        sys.stderr.flush()
        
        # Clean up
        try:
            if image_tensor is not None:
                del image_tensor
            if image is not None:
                del image
            if volume is not None:
                del volume
            gc.collect()
        except:
            pass
        
        return jsonify({
            'error': f'Prediction failed: {str(e)}',
            'traceback': traceback.format_exc() if os.environ.get('DEBUG', 'False') == 'True' else None
        }), 500


@app.route('/predict_base64', methods=['POST'])
def predict_base64():
    """
    Memory-efficient prediction from base64-encoded numpy array.
    
    Expected JSON:
    {
        "data": "<base64-encoded-bytes>",
        "shape": [128, 128, 128],
        "dtype": "float32"  # optional, defaults to float32
    }
    
    This is more memory-efficient than /predict_from_array because
    base64 encoding is much more compact than JSON nested lists.
    """
    image = None
    image_tensor = None
    volume = None
    
    try:
        if model is None:
            return jsonify({'error': 'Model not loaded'}), 503
        
        if not request.is_json:
            return jsonify({'error': 'Request must be JSON'}), 400
        
        data = request.json
        if data is None or 'data' not in data or 'shape' not in data:
            return jsonify({
                'error': 'Missing required fields',
                'required': ['data', 'shape']
            }), 400
        
        logger.info("Decoding base64 data...")
        sys.stdout.flush()
        
        # Decode base64
        decoded = base64.b64decode(data['data'])
        
        # Get dtype
        dtype_str = data.get('dtype', 'float32')
        dtype = getattr(np, dtype_str, np.float32)
        
        # Convert to numpy array
        volume = np.frombuffer(decoded, dtype=dtype).reshape(tuple(data['shape']))
        
        logger.info(f"Volume shape: {volume.shape}")
        sys.stdout.flush()
        
        # Convert to 2D image
        image = volume_to_2d_image(volume)
        del volume
        volume = None
        gc.collect()
        
        # Preprocess
        image_tensor = preprocess_image_for_classification(image)
        del image
        image = None
        gc.collect()
        
        # Predict
        result = predict_disease(image_tensor)
        del image_tensor
        image_tensor = None
        gc.collect()
        
        return jsonify(result), 200
        
    except Exception as e:
        logger.error(f"Error in /predict_base64: {e}")
        logger.error(traceback.format_exc())
        sys.stdout.flush()
        
        try:
            if volume is not None: del volume
            if image is not None: del image
            if image_tensor is not None: del image_tensor
            gc.collect()
        except:
            pass
        
        return jsonify({'error': str(e)}), 500


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
