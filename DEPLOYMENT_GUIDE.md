# Deployment Guide: Render vs Railway

This guide provides step-by-step instructions for deploying the NeuroView AI API to both Render and Railway platforms.

## Table of Contents
- [Prerequisites](#prerequisites)
- [Render Deployment](#render-deployment)
- [Railway Deployment](#railway-deployment)
- [Post-Deployment](#post-deployment)

---

## Prerequisites

Before deploying, ensure you have:

1. ✅ **Git repository** with all code committed
2. ✅ **Model files** committed to git (checkpoints, outputs, logs)
3. ✅ **GitHub account** (for connecting to deployment platforms)
4. ✅ **Trained model** (`ai-training/checkpoints/best_model.pth`)

---

## Render Deployment

### Step 1: Create Render Account
1. Go to [render.com](https://render.com)
2. Sign up with your GitHub account
3. Verify your email address

### Step 2: Create New Web Service
1. Click **"New +"** → **"Web Service"**
2. Connect your GitHub repository (`neuroview-ai`)
3. Select the repository

### Step 3: Configure Service Settings

**Basic Settings:**
- **Name:** `neuroview-ai-api` (or your preferred name)
- **Region:** Choose closest to your users (e.g., `Oregon (US West)`)
- **Branch:** `master` or `main` (your default branch)
- **Root Directory:** Leave empty (or set to `ai-training` if deploying from subdirectory)

**Build & Deploy:**
- **Runtime:** `Python 3`
- **Build Command:**
  ```bash
  cd ai-training && pip install -r requirements.txt
  ```
- **Start Command:**
  ```bash
  cd ai-training && gunicorn --bind 0.0.0.0:$PORT --workers 2 --timeout 120 --worker-class sync api_server:app
  ```

**Environment Variables:**
Click **"Add Environment Variable"** and add:
- `PORT` = `10000` (Render will override this, but set as default)
- `PYTHON_VERSION` = `3.10` (or `3.11`)

**Advanced Settings:**
- **Health Check Path:** `/health`
- **Auto-Deploy:** `Yes` (deploys on every push to main branch)

### Step 4: Deploy
1. Click **"Create Web Service"**
2. Render will:
   - Clone your repository
   - Install dependencies
   - Build the service
   - Deploy it
3. Wait for deployment to complete (5-10 minutes for first deploy)

### Step 5: Verify Deployment
1. Once deployed, you'll get a URL like: `https://neuroview-ai-api.onrender.com`
2. Test the health endpoint:
   ```bash
   curl https://your-service-name.onrender.com/health
   ```
3. Expected response:
   ```json
   {
     "status": "ok",
     "model_loaded": true,
     "device": "cpu"
   }
   ```

### Step 6: Update Frontend (If Applicable)
Update your frontend API URL to point to the Render service:
```env
VITE_API_URL=https://your-service-name.onrender.com
```

### Render Configuration File (Optional)
You can also use the existing `render.yaml` file:
1. Ensure `render.yaml` is in the root directory
2. In Render dashboard, select **"Apply Render Blueprint"**
3. Render will read the YAML and create services automatically

**Note:** The existing `render.yaml` is already configured correctly.

---

## Railway Deployment

### Step 1: Create Railway Account
1. Go to [railway.app](https://railway.app)
2. Sign up with your GitHub account
3. Complete the onboarding process

### Step 2: Create New Project
1. Click **"New Project"**
2. Select **"Deploy from GitHub repo"**
3. Choose your `neuroview-ai` repository
4. Railway will create a new project

### Step 3: Configure Service
1. Railway will auto-detect it's a Python project
2. Click on the service to configure it

**Settings Tab:**
- **Name:** `neuroview-ai-api`
- **Source:** Your GitHub repo (auto-connected)

**Deploy Tab:**
- **Root Directory:** Leave empty (or set to `ai-training`)
- **Build Command:**
  ```bash
  cd ai-training && pip install -r requirements.txt
  ```
- **Start Command:**
  ```bash
  cd ai-training && gunicorn --bind 0.0.0.0:$PORT --workers 2 --timeout 120 --worker-class sync api_server:app
  ```

### Step 4: Set Environment Variables
1. Go to **"Variables"** tab
2. Add the following:
   - `PORT` = `${{PORT}}` (Railway automatically provides this)
   - `PYTHON_VERSION` = `3.10` (optional, Railway auto-detects)

**Note:** Railway automatically sets `$PORT` environment variable, so you don't need to set it manually.

### Step 5: Create Procfile (Optional but Recommended)
Create a `Procfile` in the `ai-training/` directory:

```bash
web: cd ai-training && gunicorn --bind 0.0.0.0:$PORT --workers 2 --timeout 120 --worker-class sync api_server:app
```

Or create `Procfile` in root with:
```bash
web: cd ai-training && gunicorn --bind 0.0.0.0:$PORT --workers 2 --timeout 120 --worker-class sync api_server:app
```

### Step 6: Deploy
1. Railway will automatically deploy on push to main branch
2. Or click **"Deploy"** button to trigger manual deployment
3. Wait for build to complete (5-10 minutes for first deploy)

### Step 7: Generate Public URL
1. Go to **"Settings"** tab
2. Under **"Networking"**, click **"Generate Domain"**
3. Railway will create a public URL like: `https://neuroview-ai-api-production.up.railway.app`
4. Copy the URL

### Step 8: Verify Deployment
1. Test the health endpoint:
   ```bash
   curl https://your-service-name.up.railway.app/health
   ```
2. Expected response:
   ```json
   {
     "status": "ok",
     "model_loaded": true,
     "device": "cpu"
   }
   ```

### Step 9: Update Frontend (If Applicable)
Update your frontend API URL:
```env
VITE_API_URL=https://your-service-name.up.railway.app
```

---

## Post-Deployment

### Testing Your Deployed API

1. **Health Check:**
   ```bash
   curl https://your-api-url/health
   ```

2. **Test Prediction (using curl):**
   ```bash
   curl -X POST \
     -F "file=@path/to/your/brain_scan.nii.gz" \
     https://your-api-url/predict
   ```

3. **Test from Frontend:**
   - Update frontend API URL
   - Upload a brain scan
   - Verify predictions work

### Monitoring

**Render:**
- Go to your service dashboard
- View logs in real-time
- Monitor metrics (CPU, memory, requests)

**Railway:**
- Go to your project dashboard
- View deployment logs
- Monitor resource usage
- Set up alerts if needed

### Troubleshooting

#### Model Not Loading
- **Check:** Model file exists in `ai-training/checkpoints/best_model.pth`
- **Solution:** Ensure model files are committed to git
- **Verify:** Check deployment logs for model loading errors

#### Out of Memory
- **Render:** Upgrade to a paid plan for more memory
- **Railway:** Upgrade plan or optimize model loading
- **Solution:** Consider using CPU-only inference or model quantization

#### Build Failures
- **Check:** All dependencies in `requirements.txt`
- **Solution:** Review build logs for missing packages
- **Common issue:** NumPy version conflicts - ensure `numpy<2.0.0`

#### Timeout Errors
- **Solution:** Increase timeout in gunicorn command:
  ```bash
  --timeout 300  # 5 minutes
  ```

#### CORS Issues
- **Check:** `flask-cors` is installed
- **Solution:** Verify CORS is enabled in `api_server.py`:
  ```python
  CORS(app)  # Should be present
  ```

---

## Comparison: Render vs Railway

| Feature | Render | Railway |
|---------|--------|---------|
| **Free Tier** | ✅ Yes (with limitations) | ✅ Yes (with $5 credit) |
| **Auto-Deploy** | ✅ Yes | ✅ Yes |
| **Custom Domain** | ✅ Yes (free tier) | ✅ Yes |
| **SSL/HTTPS** | ✅ Automatic | ✅ Automatic |
| **Build Time** | ~5-10 min | ~5-10 min |
| **Cold Start** | ~30-60 sec | ~10-30 sec |
| **Logs** | ✅ Real-time | ✅ Real-time |
| **Scaling** | Manual | Automatic (paid) |
| **Ease of Setup** | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |

### Recommendation

**Choose Render if:**
- You want a simple, straightforward deployment
- You need a free tier with custom domain
- You prefer YAML-based configuration

**Choose Railway if:**
- You want faster cold starts
- You prefer a more modern interface
- You need better resource monitoring
- You want automatic scaling (paid plans)

---

## Next Steps

1. ✅ Deploy to your chosen platform
2. ✅ Test all API endpoints
3. ✅ Update frontend to use new API URL
4. ✅ Set up monitoring and alerts
5. ✅ Configure custom domain (optional)
6. ✅ Set up CI/CD for automatic deployments

---

## Additional Resources

- [Render Documentation](https://render.com/docs)
- [Railway Documentation](https://docs.railway.app)
- [Gunicorn Configuration](https://docs.gunicorn.org/en/stable/settings.html)
- [Flask Production Best Practices](https://flask.palletsprojects.com/en/2.3.x/deploying/)

---

## Support

If you encounter issues:
1. Check deployment logs
2. Verify all files are committed
3. Test API locally first
4. Review platform-specific documentation
5. Check GitHub issues for similar problems

