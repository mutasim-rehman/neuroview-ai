# Fix: Train Correct Model for API

## Problem
You trained using `main_train.py`, which creates a **classification model** (BrainTumorClassificationModel). 
But the API needs the **autoencoder model** (BrainScanModel) from `main_train_healthy.py`.

## Solution: Retrain with Correct Script

### Step 1: Train the Correct Model

From the `ai-training` directory, run:

```bash
cd ai-training
py main_train_healthy.py
```

Or if you want to train for fewer epochs (faster):

```bash
py main_train_healthy.py --epochs 10
```

This will create `checkpoints/best_model.pth` with the **correct architecture** (BrainScanModel with encoder/decoder).

### Step 2: Verify the Model File

After training completes, check that the file exists:

```bash
# Check file exists and size
dir checkpoints\best_model.pth
```

### Step 3: Commit and Push to Git

```bash
# From ai-training directory
git add checkpoints/best_model.pth
git commit -m "Update best_model.pth with correct BrainScanModel architecture"
git push
```

### Step 4: Wait for Render to Deploy

Render will automatically rebuild your service. Wait 2-5 minutes, then test:

```bash
# From project root
python test_api.py E:\neuroview-ai\demo.nii
```

The health check should now show `"model_loaded": true` and predictions should work!

## Key Difference

- ❌ `main_train.py` → Creates `BrainTumorClassificationModel` (has `features.*`, `classifier.*`)
- ✅ `main_train_healthy.py` → Creates `BrainScanModel` (has `encoder.*`, `decoder.*`, `feature_head.*`)

The API expects the BrainScanModel architecture.

