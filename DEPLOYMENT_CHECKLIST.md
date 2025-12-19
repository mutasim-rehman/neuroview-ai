# Deployment Checklist

Use this checklist to ensure everything is ready for deployment.

## Pre-Deployment Checklist

### Backend (Render)

- [ ] **Model Checkpoint Exists**
  - [ ] `ai-training/checkpoints/best_model.pth` exists
  - [ ] If file is large (>100MB), consider using Git LFS or external storage
  - [ ] Verify model was trained successfully

- [ ] **Dependencies**
  - [ ] `requirements.txt` includes all dependencies
  - [ ] `gunicorn` is in requirements.txt ✅
  - [ ] All Python packages are specified with versions

- [ ] **Configuration**
  - [ ] `api_server.py` uses `PORT` environment variable ✅
  - [ ] CORS is enabled ✅
  - [ ] Error handling is in place ✅

- [ ] **Files Ready**
  - [ ] `render.yaml` exists (optional, can configure via dashboard) ✅
  - [ ] `start.sh` exists (optional) ✅
  - [ ] All code is committed to Git

### Frontend (Vercel)

- [ ] **Build Configuration**
  - [ ] `vercel.json` exists ✅
  - [ ] `package.json` has build script ✅
  - [ ] All dependencies are in `package.json`

- [ ] **Environment Variables**
  - [ ] Know your Render backend URL (get after backend deployment)
  - [ ] Have Gemini API key ready (optional)

- [ ] **Files Ready**
  - [ ] All code is committed to Git
  - [ ] No build errors locally (`npm run build` works)

## Deployment Order

1. **Deploy Backend First** (Render)
   - Get the API URL
   - Test `/health` endpoint
   - Verify model loads

2. **Deploy Frontend Second** (Vercel)
   - Use backend URL in `VITE_API_URL`
   - Test frontend connects to backend

## Post-Deployment Checklist

### Backend

- [ ] Health endpoint works: `curl https://your-api.onrender.com/health`
- [ ] Model loads successfully (check Render logs)
- [ ] Predict endpoint responds (test with a small file)
- [ ] CORS headers are present in responses
- [ ] Service stays online (or keep-alive is configured)

### Frontend

- [ ] Site loads at Vercel URL
- [ ] No console errors
- [ ] Environment variables are set correctly
- [ ] API calls go to correct backend URL
- [ ] File upload works
- [ ] Brain health prediction works end-to-end

## Common Issues & Solutions

### Model Checkpoint Not Found

**Problem:** Render can't find `best_model.pth`

**Solutions:**
1. **Commit the checkpoint file:**
   ```bash
   git add -f ai-training/checkpoints/best_model.pth
   git commit -m "Add model checkpoint"
   git push
   ```

2. **Or use Git LFS for large files:**
   ```bash
   git lfs install
   git lfs track "*.pth"
   git add .gitattributes
   git add ai-training/checkpoints/best_model.pth
   git commit -m "Add model checkpoint with LFS"
   git push
   ```

3. **Or store externally:**
   - Upload to cloud storage (S3, Google Cloud Storage)
   - Download during Render build process
   - Update `api_server.py` to download from URL

### Build Fails on Render

**Check:**
- Python version is 3.10
- All dependencies in `requirements.txt`
- Root directory is set to `ai-training`
- Build command is correct

### Frontend Can't Connect to Backend

**Check:**
- `VITE_API_URL` is set correctly (no trailing slash)
- Backend is running (not spun down)
- CORS is enabled on backend
- Check browser console for specific errors

### Service Spins Down (Render Free Tier)

**Solutions:**
1. Upgrade to Starter plan ($7/month)
2. Use external keep-alive service (cron job)
3. Accept cold starts (30 second delay on first request)

## Quick Test Commands

```bash
# Test backend health
curl https://your-api.onrender.com/health

# Test backend prediction (with file)
curl -X POST https://your-api.onrender.com/predict \
  -F "file=@test_scan.nii"

# Check frontend
# Open in browser and check console
```

## Next Steps After Deployment

1. ✅ Monitor logs for errors
2. ✅ Set up error tracking (Sentry, etc.)
3. ✅ Configure custom domains (optional)
4. ✅ Set up monitoring/alerts
5. ✅ Document API endpoints for users

---

**Ready to deploy?** Follow `DEPLOYMENT_STEPS.md` for detailed instructions!

