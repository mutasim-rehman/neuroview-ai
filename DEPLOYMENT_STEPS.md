# Deployment Steps for NeuroView AI

This guide provides step-by-step instructions for deploying:
- **Frontend** → Vercel
- **Backend API** → Render

---

## Prerequisites

1. **GitHub Account** - Your code should be in a GitHub repository
2. **Vercel Account** - Sign up at [vercel.com](https://vercel.com)
3. **Render Account** - Sign up at [render.com](https://render.com)
4. **Trained Model** - Ensure `ai-training/checkpoints/best_model.pth` exists (or train it first)

---

## Part 1: Deploy Backend API to Render

### Step 1: Prepare Your Repository

1. **Verify model checkpoint exists:**
   - Check that `ai-training/checkpoints/best_model.pth` exists
   - If not, train the model first (see `ai-training/README.md`)

2. **⚠️ IMPORTANT: Commit model checkpoint**
   - The checkpoint file is in `.gitignore` by default
   - You need to commit it for deployment:
   ```bash
   git add -f ai-training/checkpoints/best_model.pth
   git commit -m "Add model checkpoint for deployment"
   ```
   - **Note:** If the file is very large (>100MB), consider using Git LFS:
   ```bash
   git lfs install
   git lfs track "*.pth"
   git add .gitattributes
   git add ai-training/checkpoints/best_model.pth
   git commit -m "Add model checkpoint with Git LFS"
   ```

3. **Commit all other changes:**
   ```bash
   git add .
   git commit -m "Prepare for deployment"
   git push origin main
   ```

### Step 2: Create Render Web Service

1. **Go to [Render Dashboard](https://dashboard.render.com/)**

2. **Click "New +" → "Web Service"**

3. **Connect your GitHub repository:**
   - Select your repository
   - Click "Connect"

4. **Configure the service:**
   - **Name:** `neuroview-ai-api` (or your preferred name)
   - **Region:** Choose closest to your users
   - **Branch:** `main` (or your default branch)
   - **Root Directory:** `ai-training` ⚠️ **Important!**
   - **Runtime:** `Python 3`
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `gunicorn --bind 0.0.0.0:$PORT --workers 2 --timeout 120 --worker-class sync api_server:app`

5. **Set Environment Variables:**
   - Click "Advanced" → "Add Environment Variable"
   - Add: `PYTHON_VERSION` = `3.10`
   - The `PORT` variable is automatically set by Render

6. **Choose Plan:**
   - **Starter Plan** ($7/month) - Recommended for testing
   - **Standard Plan** - For production with more resources

7. **Click "Create Web Service"**

### Step 3: Wait for Deployment

- Render will:
  1. Install dependencies
  2. Build your service
  3. Start the server
  4. Run health checks

- **Expected build time:** 5-10 minutes (PyTorch is large)

### Step 4: Get Your API URL

1. Once deployed, you'll see a URL like: `https://neuroview-ai-api.onrender.com`
2. **Test the health endpoint:**
   ```bash
   curl https://your-api-url.onrender.com/health
   ```
   Should return:
   ```json
   {
     "status": "ok",
     "model_loaded": true,
     "device": "cpu"
   }
   ```

3. **Copy this URL** - You'll need it for the frontend!

### Step 5: Handle Render's Free Tier Limitations

⚠️ **Important:** Render's free tier spins down after 15 minutes of inactivity.

**Options:**
1. **Upgrade to Starter Plan** ($7/month) - Keeps service always on
2. **Use a keep-alive service** (external cron job to ping `/health` every 10 minutes)
3. **Accept cold starts** - First request after spin-down takes ~30 seconds

---

## Part 2: Deploy Frontend to Vercel

### Step 1: Prepare Frontend

1. **Ensure all files are committed:**
   ```bash
   git status
   git add .
   git commit -m "Add deployment config"
   git push origin main
   ```

### Step 2: Deploy via Vercel Dashboard (Recommended)

1. **Go to [Vercel Dashboard](https://vercel.com/dashboard)**

2. **Click "Add New Project"**

3. **Import your GitHub repository:**
   - Select your repository
   - Click "Import"

4. **Configure Project:**
   - **Framework Preset:** Vite (auto-detected)
   - **Root Directory:** `./` (root)
   - **Build Command:** `npm run build` (auto-detected)
   - **Output Directory:** `dist` (auto-detected)
   - **Install Command:** `npm install` (auto-detected)

5. **Add Environment Variables:**
   - Click "Environment Variables"
   - Add the following:
     
     **Variable 1:**
     - **Name:** `VITE_API_URL`
     - **Value:** `https://your-api-url.onrender.com` (use your Render URL from Part 1, Step 4)
     - **Environments:** Production, Preview, Development
     
     **Variable 2 (Optional - for Gemini AI features):**
     - **Name:** `GEMINI_API_KEY`
     - **Value:** Your Google Gemini API key
     - **Environments:** Production, Preview, Development

6. **Click "Deploy"**

### Step 3: Wait for Deployment

- Vercel will:
  1. Install dependencies
  2. Build the frontend
  3. Deploy to CDN
  4. Provide you with a URL

- **Expected build time:** 1-2 minutes

### Step 4: Get Your Frontend URL

- Once deployed, you'll get a URL like: `https://neuroview-ai.vercel.app`
- **Test it:** Open the URL in your browser

---

## Part 3: Connect Frontend to Backend

### Step 1: Verify Backend is Running

1. **Test backend health:**
   ```bash
   curl https://your-api-url.onrender.com/health
   ```

2. **If it's the first request after spin-down, wait ~30 seconds**

### Step 2: Update Frontend Environment Variable

1. **Go to Vercel Dashboard** → Your Project → Settings → Environment Variables

2. **Edit `VITE_API_URL`:**
   - Update with your Render API URL
   - Ensure it's set for all environments

3. **Redeploy:**
   - Go to Deployments tab
   - Click "..." on latest deployment → "Redeploy"
   - Or push a new commit to trigger auto-deploy

### Step 3: Test Integration

1. **Open your Vercel frontend URL**
2. **Upload a NIfTI file** (or use the brain health prediction feature)
3. **Check browser console** for any errors
4. **Verify API calls** are going to your Render backend

---

## Part 4: Post-Deployment Checklist

### ✅ Backend (Render)

- [ ] Health endpoint responds: `/health`
- [ ] Model loads successfully (check logs)
- [ ] Predict endpoint works: `/predict`
- [ ] CORS is enabled (should work automatically)
- [ ] Service stays online (or you've set up keep-alive)

### ✅ Frontend (Vercel)

- [ ] Site loads correctly
- [ ] Environment variables are set
- [ ] API calls go to correct backend URL
- [ ] No CORS errors in console
- [ ] File upload works
- [ ] Brain health prediction works

### ✅ Integration

- [ ] Frontend can reach backend
- [ ] Predictions return successfully
- [ ] Error handling works
- [ ] Loading states display correctly

---

## Troubleshooting

### Backend Issues

**Problem: Model not loading**
- **Solution:** Check Render logs, verify `checkpoints/best_model.pth` is committed to repo
- **Note:** Large model files (>100MB) may need Git LFS

**Problem: Out of memory**
- **Solution:** Upgrade Render plan or reduce model size

**Problem: Service spins down (free tier)**
- **Solution:** Upgrade to Starter plan or use keep-alive service

**Problem: Build fails**
- **Solution:** Check Python version (should be 3.10), verify all dependencies in `requirements.txt`

### Frontend Issues

**Problem: API calls fail**
- **Solution:** 
  1. Check `VITE_API_URL` is set correctly
  2. Verify backend is running
  3. Check CORS settings
  4. Check browser console for errors

**Problem: Build fails**
- **Solution:** 
  1. Check Node.js version (Vercel uses 18.x)
  2. Verify all dependencies in `package.json`
  3. Check build logs in Vercel dashboard

**Problem: Environment variables not working**
- **Solution:** 
  1. Ensure variables start with `VITE_` prefix
  2. Redeploy after adding variables
  3. Check variable is set for correct environment

### Integration Issues

**Problem: CORS errors**
- **Solution:** Backend already has CORS enabled, but verify `flask-cors` is installed

**Problem: 404 on API calls**
- **Solution:** Check API URL doesn't have trailing slash, verify endpoint paths

**Problem: Timeout errors**
- **Solution:** Increase timeout in Render settings or reduce file size

---

## Quick Reference

### Backend URLs
- **Health Check:** `https://your-api.onrender.com/health`
- **Predict:** `https://your-api.onrender.com/predict`
- **Predict from Array:** `https://your-api.onrender.com/predict_from_array`

### Frontend URLs
- **Production:** `https://your-app.vercel.app`
- **Preview:** `https://your-app-git-branch.vercel.app`

### Environment Variables

**Frontend (Vercel):**
- `VITE_API_URL` - Backend API URL
- `GEMINI_API_KEY` - Google Gemini API key (optional)

**Backend (Render):**
- `PORT` - Automatically set by Render
- `PYTHON_VERSION` - Set to `3.10`

---

## Updating Deployments

### Update Backend

1. **Make changes to code**
2. **Commit and push:**
   ```bash
   git add .
   git commit -m "Update backend"
   git push origin main
   ```
3. **Render auto-deploys** on push to main branch

### Update Frontend

1. **Make changes to code**
2. **Commit and push:**
   ```bash
   git add .
   git commit -m "Update frontend"
   git push origin main
   ```
3. **Vercel auto-deploys** on push to main branch

### Update Environment Variables

**Vercel:**
1. Dashboard → Project → Settings → Environment Variables
2. Edit or add variables
3. Redeploy (or wait for next deployment)

**Render:**
1. Dashboard → Service → Environment
2. Edit or add variables
3. Service automatically restarts

---

## Cost Estimation

### Render (Backend)
- **Free Tier:** $0/month (spins down after inactivity)
- **Starter Plan:** $7/month (always on, recommended)
- **Standard Plan:** $25/month (more resources)

### Vercel (Frontend)
- **Hobby Plan:** Free (sufficient for most projects)
- **Pro Plan:** $20/month (for teams)

**Total Estimated Cost:** $7-25/month (depending on Render plan)

---

## Support

- **Render Docs:** https://render.com/docs
- **Vercel Docs:** https://vercel.com/docs
- **Project Issues:** Check GitHub issues or create new one

---

## Next Steps

1. ✅ Deploy backend to Render
2. ✅ Deploy frontend to Vercel
3. ✅ Connect them together
4. ✅ Test everything
5. 🎉 Share your deployed app!

Good luck with your deployment! 🚀

