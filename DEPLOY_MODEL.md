# Deploying the Model to Render

## Problem
The API is deployed on Render, but the model file (`best_model.pth`) is not being loaded. This happens because:

1. **The model file is not committed to git** - Model files are often large and may not be tracked in the repository
2. **The model needs to be available when Render builds your application**

## Solution: Commit the Model File to Git

### Step 1: Check Model File Size
First, check if the model file is small enough to commit to git (GitHub has a 100MB file size limit):

```bash
# Check file size
ls -lh ai-training/checkpoints/best_model.pth
```

If the file is **under 100MB**, you can commit it directly.

### Step 2: Ensure Model File is Tracked

```bash
# Check if file is ignored
git check-ignore ai-training/checkpoints/best_model.pth

# If not ignored, add it to git
git add ai-training/checkpoints/best_model.pth

# Commit
git commit -m "Add trained model checkpoint for deployment"

# Push to repository
git push
```

### Step 3: Verify Deployment

After pushing, Render will automatically rebuild your service. Wait a few minutes, then test:

```bash
python test_api.py
```

The health check should now show `"model_loaded": true`.

## Alternative Solutions (If Model is Too Large)

If your model file is **over 100MB**, you have these options:

### Option A: Use Git LFS (Git Large File Storage)

1. **Install Git LFS:**
   ```bash
   git lfs install
   ```

2. **Track .pth files with LFS:**
   ```bash
   git lfs track "*.pth"
   git add .gitattributes
   git add ai-training/checkpoints/best_model.pth
   git commit -m "Add model checkpoint using Git LFS"
   git push
   ```

3. **Note:** Git LFS requires a GitHub account with LFS quota (1GB free)

### Option B: Download Model During Build

Create a build script that downloads the model from cloud storage:

1. **Upload model to cloud storage** (Google Drive, Dropbox, AWS S3, etc.)
2. **Create a build script** (`ai-training/download_model.sh`):
   ```bash
   #!/bin/bash
   # Download model from cloud storage if not present
   if [ ! -f "checkpoints/best_model.pth" ]; then
       echo "Downloading model..."
       # Add your download command here
       # Example: wget https://your-storage.com/best_model.pth -O checkpoints/best_model.pth
   fi
   ```

3. **Update render.yaml** to run the script:
   ```yaml
   buildCommand: cd ai-training && bash download_model.sh && pip install -r requirements.txt
   ```

### Option C: Use Render Disk Storage

Render provides persistent disk storage. You can:

1. SSH into your Render service
2. Upload the model file manually
3. The file will persist across deployments

However, this requires manual setup and isn't ideal for automated deployments.

## Recommended Approach

**For most cases:** Commit the model file directly to git if it's under 100MB. This is the simplest and most reliable solution.

**For large models:** Use Git LFS if you have GitHub, or set up automated download from cloud storage during the build process.

## Verification

After deploying, verify the model is loaded:

```bash
# Test API health
curl https://neuroview-ai.onrender.com/health

# Should return:
# {
#   "status": "ok",
#   "model_loaded": true,
#   "device": "cpu"
# }
```

## Troubleshooting

### Model Still Not Loading?

1. **Check Render logs:**
   - Go to your Render dashboard
   - Click on your service
   - View the "Logs" tab
   - Look for model loading messages

2. **Verify file path:**
   - The model should be at: `ai-training/checkpoints/best_model.pth`
   - Check that the path is correct in `api_server.py`

3. **Check file permissions:**
   - Ensure the file is readable
   - Render should have read access to the file

4. **Verify model file exists in repository:**
   ```bash
   git ls-files ai-training/checkpoints/best_model.pth
   ```
   If this returns nothing, the file is not tracked.

## Current Status

After fixing `api_server.py`, the model will now load automatically when the app starts (works with both direct run and gunicorn). You just need to ensure the model file is in your git repository.

