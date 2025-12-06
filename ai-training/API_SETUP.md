# Brain Health Prediction API Setup Guide

This guide explains how to set up and run the Python Flask API server for brain health predictions.

## Prerequisites

1. **Python 3.8+** installed
2. **Trained model checkpoint** (see training instructions)
3. **All dependencies** from `requirements.txt`

## Installation

1. **Navigate to the ai-training directory:**
   ```bash
   cd ai-training
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

   This will install:
   - PyTorch
   - Flask & Flask-CORS
   - Nibabel
   - NumPy, SciPy
   - And other required packages

## Training the Model (If Not Done Yet)

Before running the API server, you need a trained model:

```bash
python main_train_healthy.py
```

This will create a checkpoint file at `./checkpoints/best_model.pth`.

## Running the API Server

1. **Start the Flask server:**
   ```bash
   python api_server.py
   ```

2. **The server will:**
   - Load the trained model from `./checkpoints/best_model.pth`
   - Start listening on `http://localhost:5000`
   - Display model loading status in the console

3. **Expected output:**
   ```
   Loading model...
   Using device: cuda  (or cpu)
   Model loaded successfully from ./checkpoints/best_model.pth
   Starting Flask server on http://localhost:5000
   * Running on http://0.0.0.0:5000
   ```

## API Endpoints

### 1. Health Check
- **URL:** `GET /health`
- **Description:** Check if the API server is running and model is loaded
- **Response:**
  ```json
  {
    "status": "ok",
    "model_loaded": true,
    "device": "cuda"
  }
  ```

### 2. Predict from File
- **URL:** `POST /predict`
- **Description:** Predict brain health from uploaded NIfTI file
- **Request:** `multipart/form-data` with `file` field
- **Response:**
  ```json
  {
    "prediction": "healthy",
    "confidence": 0.85,
    "anomaly_score": 0.008,
    "error_metrics": {
      "mse": 0.008,
      "mae": 0.045,
      "max_error": 0.123
    },
    "feature_vector": [...]
  }
  ```

### 3. Predict from Array
- **URL:** `POST /predict_from_array`
- **Description:** Predict from volume data array
- **Request Body:**
  ```json
  {
    "volume": [[[...]]],  // 3D nested array
    "shape": [128, 128, 128]
  }
  ```
- **Response:** Same as `/predict`

## Frontend Integration

The frontend automatically connects to `http://localhost:5000` by default.

To change the API URL, set the environment variable:
```env
VITE_API_URL=http://your-api-url:5000
```

## Troubleshooting

### Model Not Loading

1. **Check checkpoint path:**
   - Verify `./checkpoints/best_model.pth` exists
   - Or update the path in `api_server.py`

2. **Check model architecture:**
   - Ensure model parameters in `config.py` match training config
   - Verify `FEATURE_DIM`, `BASE_CHANNELS`, etc. are correct

### CORS Errors

If you see CORS errors, ensure `flask-cors` is installed:
```bash
pip install flask-cors
```

### Out of Memory

If you run out of memory:
- Use CPU instead of GPU (model will be slower)
- Reduce `BATCH_SIZE` in config (though inference uses batch size 1)
- Process smaller volumes

### Port Already in Use

If port 5000 is already in use, change it in `api_server.py`:
```python
app.run(host='0.0.0.0', port=5001, debug=True)  # Change port number
```

## Production Deployment

For production:

1. **Use a production WSGI server:**
   ```bash
   pip install gunicorn
   gunicorn -w 4 -b 0.0.0.0:5000 api_server:app
   ```

2. **Disable debug mode:**
   ```python
   app.run(host='0.0.0.0', port=5000, debug=False)
   ```

3. **Add authentication/rate limiting** as needed

4. **Set up proper error handling and logging**

## Testing the API

You can test the API using curl:

```bash
# Health check
curl http://localhost:5000/health

# Predict from file
curl -X POST -F "file=@your_brain_scan.nii.gz" http://localhost:5000/predict
```

## Notes

- The model expects volumes to be preprocessed (resized to 128x128x128, normalized)
- Prediction threshold (0.01 MSE) is a heuristic - should be calibrated on validation data
- For production use, add proper validation and error handling
- Consider adding caching for frequently requested volumes
