# Vercel Deployment Guide

## Issue Fixed

The error you encountered was because the **Install Command** in Vercel was set to `python api_server.py` instead of installing dependencies.

## Vercel Project Settings

When deploying on Vercel, make sure to configure these settings correctly:

### Build and Output Settings

1. **Install Command:** 
   ```
   pip install -r requirements.txt
   ```
   ⚠️ **NOT** `python api_server.py` - that runs the server, not installs dependencies!

2. **Build Command:**
   ```
   (leave empty or remove)
   ```
   Or you can set it to an empty string.

3. **Output Directory:**
   ```
   (leave empty or set to '.')
   ```

### Root Directory
- Set to: `ai-training`

### Framework Preset
- Set to: **Other** (or Python)

## Important Notes

### 1. Model Checkpoint
The model checkpoint file (`checkpoints/best_model.pth`) needs to be:
- Committed to your repository, OR
- Loaded from external storage (S3, etc.) at runtime

If the checkpoint is large (>50MB), consider:
- Using Git LFS
- Storing in cloud storage and downloading at runtime
- Including it in the deployment

### 2. Vercel Limitations

⚠️ **Important:** Vercel serverless functions have limitations:
- **Execution timeout:** 10 seconds (Hobby plan) or 60 seconds (Pro plan)
- **Memory:** Limited (may not be enough for large PyTorch models)
- **Package size:** Large dependencies like PyTorch may cause issues

### 3. Alternative Solutions

If Vercel doesn't work due to these limitations, consider:
- **Railway** - Better for Python apps with ML models
- **Render** - Supports long-running Python services
- **Fly.io** - Good for containerized Python apps
- **AWS Lambda** - With Lambda Layers for PyTorch
- **Google Cloud Run** - Container-based, flexible resources

## Deployment Steps

1. **Update Vercel Settings:**
   - Go to your project settings in Vercel dashboard
   - Navigate to "Settings" → "General" → "Build & Development Settings"
   - Set **Install Command** to: `pip install -r requirements.txt`
   - Set **Build Command** to: (empty)
   - Set **Output Directory** to: (empty or '.')

2. **Ensure vercel.json is committed:**
   - The `vercel.json` file in `ai-training/` directory should be committed to your repo

3. **Deploy:**
   - Push your changes to GitHub
   - Vercel will automatically redeploy with the correct settings

## Testing After Deployment

Once deployed, test the endpoints:

```bash
# Health check
curl https://your-project.vercel.app/health

# Predict endpoint
curl -X POST https://your-project.vercel.app/predict \
  -F "file=@your_brain_scan.nii.gz"
```

## Troubleshooting

### Still getting "ModuleNotFoundError"
- Check that `requirements.txt` is in the `ai-training` directory
- Verify the root directory is set to `ai-training` in Vercel
- Check build logs to see if pip install actually ran

### Model loading errors
- Ensure `checkpoints/best_model.pth` is included in deployment
- Check file paths are correct (they should be relative to `ai-training/`)

### Timeout errors
- PyTorch inference might take longer than Vercel's timeout
- Consider using a different platform for production

### Memory errors
- Vercel serverless functions have limited memory
- Large models may not fit - consider model optimization or different platform

