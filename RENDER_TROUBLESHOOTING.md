# Render API Troubleshooting Guide

## Current Issue
The prediction endpoints (`/predict`, `/predict_base64`, `/predict_from_array`) are returning 502 errors. The health endpoint works fine, showing the model is loaded.

## Diagnosis Summary

### What We Know
1. **Health endpoint works** - Model is loaded, using ~405 MB memory
2. **Root endpoint works** - Returns API info
3. **Prediction endpoints fail** - All return 502 or timeout
4. **Even tiny requests fail** - A 2x2x2 volume (32 bytes) causes timeout
5. **Local tests pass** - The same code works perfectly locally

### What We've Tried
- Optimized model checkpoint (93 MB → 31 MB)
- Added memory-efficient `/predict_base64` endpoint
- Added memory monitoring to `/health`
- Added various debug endpoints
- Configured gunicorn with proper timeout settings

### Root Cause Hypothesis
The worker process is hanging or crashing during POST request processing. This could be due to:
1. Render's proxy/load balancer configuration
2. Memory spike during request parsing
3. Some library incompatibility in the Render environment
4. Auto-deploy might be disabled

## Immediate Actions Needed

### 1. Check if Auto-Deploy is Enabled
1. Go to your Render dashboard: https://dashboard.render.com
2. Click on your `neuroview-ai-api` service
3. Go to Settings → Build & Deploy
4. Ensure "Auto-Deploy" is set to "Yes"
5. If disabled, enable it and manually trigger a deploy

### 2. Check Build/Deploy Logs
1. In the Render dashboard, go to your service
2. Click "Deploys" tab
3. Look for any failed deployments or build errors
4. Check if the latest commit is deployed

### 3. Check Runtime Logs
1. In the Render dashboard, go to "Logs" tab
2. Look for any error messages when making prediction requests
3. Run this command to trigger an error and watch the logs:
```bash
curl -X POST https://neuroview-ai.onrender.com/predict_base64 \
  -H "Content-Type: application/json" \
  -d '{"data":"AAAA","shape":[1,1,1]}'
```

### 4. Manual Deploy (if auto-deploy is broken)
1. In Render dashboard, click "Manual Deploy"
2. Select "Clear build cache & deploy"
3. Wait for deploy to complete
4. Test the API again

## Testing Commands

### Test Health (should work)
```bash
curl https://neuroview-ai.onrender.com/health
```

### Test Debug Endpoints (once deployed)
```bash
# Simple GET ping
curl https://neuroview-ai.onrender.com/debug/ping

# POST echo (doesn't parse body)
curl -X POST https://neuroview-ai.onrender.com/debug/echo \
  -H "Content-Type: application/json" \
  -d '{"test": "data"}'

# POST JSON parsing test
curl -X POST https://neuroview-ai.onrender.com/debug/json \
  -H "Content-Type: application/json" \
  -d '{"test": "data"}'

# Debug predict (synthetic data, no upload needed)
curl https://neuroview-ai.onrender.com/debug/predict
```

### Test Prediction
```python
import requests
import numpy as np
import base64

# Tiny volume test
volume = np.random.rand(2, 2, 2).astype(np.float32)
payload = {
    'data': base64.b64encode(volume.tobytes()).decode('utf-8'),
    'shape': [2, 2, 2]
}
r = requests.post('https://neuroview-ai.onrender.com/predict_base64', json=payload, timeout=120)
print(r.status_code, r.text)
```

## Alternative Solutions

### Option 1: Use Render Paid Plan
Render's Starter plan has 512 MB RAM. The API uses ~405 MB at rest, leaving little headroom.
- Upgrade to a plan with more RAM
- This would give more memory for request processing

### Option 2: Lazy Model Loading
Instead of loading the model at startup, load it on first request:
- Pro: Faster startup, less idle memory
- Con: First request will be slow (cold start)

### Option 3: Use a Different Platform
Try deploying to:
- **Railway** - Similar to Render, generous free tier
- **Fly.io** - Good for containerized apps
- **Heroku** - Classic PaaS (no longer free)
- **Google Cloud Run** - Pay-per-use, can scale to zero

### Option 4: Optimize Further
- Use `torch.jit.script` for compiled inference
- Reduce model size further (quantization)
- Use smaller base_channels in the model

## Memory Analysis

Current memory usage breakdown:
- PyTorch: ~192 MB
- Torchvision: ~70 MB
- nibabel/scipy: ~32 MB
- Model weights: ~32 MB
- Flask/Gunicorn: ~80 MB
- **Total**: ~405 MB

Render Starter limit: 512 MB
Available headroom: ~107 MB

## Files Changed
- `api_server.py` - Added debug endpoints, memory monitoring
- `Procfile` - Optimized gunicorn settings
- `render.yaml` - Added DEBUG=True env var
- `requirements.txt` - Added psutil

## Commits to Deploy
All recent commits should be deployed. Check `git log --oneline -10` for the list.

