# Fixing "Predict Health Status" Button (Greyed Out)

## Quick Fix Steps

### Step 1: Deploy Backend API First

You need to deploy the backend API to Railway or Render before the button will work.

**Option A: Railway (Easiest)**
1. Go to https://railway.app
2. Sign up/login
3. Click "New Project" → "Deploy from GitHub repo"
4. Select your `neuroview-ai` repository
5. Set **Root Directory** to: `ai-training`
6. Railway will auto-detect Python and deploy
7. Wait for deployment, then copy your Railway URL (e.g., `https://your-app.railway.app`)

**Option B: Render**
1. Go to https://render.com
2. Sign up/login
3. New → Web Service
4. Connect your GitHub repo
5. Set:
   - **Root Directory**: `ai-training`
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `python api_server.py`
6. Copy your Render URL

### Step 2: Set Environment Variable in Vercel

1. Go to **Vercel Dashboard** → Your Project → **Settings** → **Environment Variables**

2. Click **Add New**

3. Add this variable:
   - **Key**: `VITE_API_URL`
   - **Value**: Your backend URL (e.g., `https://your-app.railway.app`)
   - **Environment**: Select **Production**, **Preview**, and **Development**

4. Click **Save**

5. **Redeploy** your Vercel project:
   - Go to **Deployments** tab
   - Click the 3 dots on latest deployment → **Redeploy**

### Step 3: Verify

After redeploy:
1. Check browser console (F12) for API health status
2. Button should show status indicator
3. If still greyed out, check console for error messages

## Troubleshooting

**Button still greyed out?**
- Check browser console for errors
- Verify `VITE_API_URL` is set correctly in Vercel
- Test backend API: `curl https://your-backend-url.com/health`
- Make sure backend is actually running

**API Health shows "Model Not Loaded"?**
- Backend is running but model checkpoint is missing
- Train the model first: `cd ai-training && python main_train_healthy.py`
- Or upload `best_model.pth` to your backend deployment

## Quick Commands

```bash
# After setting VITE_API_URL, redeploy
vercel --prod
```

---

**Note**: The button will now work even if API check fails (it will try to connect), but you'll see a warning message.
