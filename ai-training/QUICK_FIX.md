# Quick Fix for Vercel Memory Error

## The Problem
Vercel free tier runs out of memory installing PyTorch and all dependencies.

## The Solution (2 Steps)

### Step 1: Update Vercel Install Command

1. Go to Vercel Dashboard → Your Project → Settings → General
2. Find **"Install Command"**
3. Change it to:
   ```
   pip install --extra-index-url https://download.pytorch.org/whl/cpu --only-binary :all: -r requirements-api.txt
   ```
4. **Save**

### Step 2: Redeploy

1. Go to **Deployments** tab
2. Click **"..."** on latest deployment
3. Click **"Redeploy"**

## What Changed

- ✅ Created `requirements-api.txt` with **only** API dependencies (no training tools)
- ✅ Uses **CPU-only PyTorch** (200MB+ smaller)
- ✅ Uses `--only-binary` flag (faster, less memory during install)
- ✅ Removed: torchvision, scikit-learn, tensorboard, matplotlib, pillow, tqdm, numba

## Expected Result

Build should complete successfully with ~1GB less memory usage.

## If Still Fails

Try Railway or Render instead (better for ML apps, free tiers available):
- **Railway**: railway.app (auto-detects Python)
- **Render**: render.com (750 free hours/month)

