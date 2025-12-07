# Fix Vercel Memory Issues - Free Solutions

## Problem
Vercel's free tier has limited build memory (2GB), and PyTorch installation exceeds this limit.

## Solution 1: Use Minimal Requirements (Recommended)

Use `requirements-api.txt` instead of `requirements.txt` for deployment:

1. **Update Vercel Install Command** to:
   ```
   pip install -r requirements-api.txt
   ```

2. **Why this works:**
   - Removes training-only dependencies (torchvision, scikit-learn, tensorboard, matplotlib, pillow, tqdm, numba)
   - Reduces installation size by ~500MB-1GB
   - Only installs what's needed for API inference

## Solution 2: Install PyTorch CPU-Only (Smaller)

PyTorch CPU-only is much smaller than GPU version. Update `requirements-api.txt`:

```txt
# Use CPU-only PyTorch (much smaller)
--index-url https://download.pytorch.org/whl/cpu
torch>=2.0.0
numpy>=1.24.0,<2.0.0
nibabel>=5.0.0
scipy>=1.10.0
flask>=2.3.0
flask-cors>=4.0.0
```

## Solution 3: Use Pre-built Wheels

Force pip to use pre-built wheels (faster, less memory during install):

Update Install Command to:
```bash
pip install --only-binary :all: -r requirements-api.txt
```

## Solution 4: Split Installation (Advanced)

Install in stages to reduce peak memory:

Update Install Command to:
```bash
pip install numpy nibabel scipy flask flask-cors && pip install torch --index-url https://download.pytorch.org/whl/cpu
```

## Solution 5: Alternative Free Platforms

If Vercel continues to have memory issues, consider these free alternatives:

### Railway (Recommended)
- **Free tier**: $5 credit/month
- **Better for**: Python ML apps
- **Setup**: 
  1. Go to railway.app
  2. Connect GitHub repo
  3. Set root directory to `ai-training`
  4. Railway auto-detects Python and installs dependencies

### Render
- **Free tier**: 750 hours/month
- **Better for**: Long-running services
- **Setup**:
  1. Go to render.com
  2. Create new Web Service
  3. Connect GitHub repo
  4. Set build command: `pip install -r requirements-api.txt`
  5. Set start command: `python api_server.py`

### Fly.io
- **Free tier**: 3 shared VMs
- **Better for**: Containerized apps
- **Setup**: Requires Dockerfile

## Recommended Action Plan

1. **Try Solution 1 first** (minimal requirements)
   - Update Install Command to use `requirements-api.txt`
   - Redeploy

2. **If still fails, try Solution 2** (CPU-only PyTorch)
   - Update `requirements-api.txt` with CPU-only PyTorch
   - Redeploy

3. **If still fails, use Railway or Render** (better for ML apps)

## Update Vercel Settings

1. Go to Vercel Dashboard → Your Project → Settings
2. **Install Command**: Change to:
   ```
   pip install -r requirements-api.txt
   ```
3. **Build Command**: Leave empty
4. Save and redeploy

## Why This Works

- **Training dependencies removed**: torchvision, scikit-learn, tensorboard, matplotlib, pillow, tqdm, numba are NOT needed for API inference
- **CPU-only PyTorch**: ~200MB smaller than full PyTorch
- **Pre-built wheels**: Faster installation, less memory during build

## Verification

After deploying, check build logs. You should see:
```
Collecting torch...
Downloading torch-2.x.x-cp311-cp311-linux_x86_64.whl (XXX MB)
...
Successfully installed torch-2.x.x numpy-1.x.x ...
```

The build should complete without memory errors.

